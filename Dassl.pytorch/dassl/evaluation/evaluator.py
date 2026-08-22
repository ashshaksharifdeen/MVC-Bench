import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict

import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless save
import matplotlib.pyplot as plt

from .build import EVALUATOR_REGISTRY


def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    bin_accuracy   = [0.0] * num_bins
    bin_confidence = [0.0] * num_bins
    bin_count      = [0]   * num_bins

    # assign each sample to its confidence bin
    for i, conf in enumerate(confidences):
        for j, (low, up) in enumerate(zip(bin_lowers, bin_uppers)):
            if low.item() < conf <= up.item():
                bin_count[j]      += 1
                bin_accuracy[j]   += correct[i]
                bin_confidence[j] += conf
                break

    # average out per-bin accuracy and confidence
    for j in range(num_bins):
        if bin_count[j] > 0:
            bin_accuracy[j]   /= bin_count[j]
            bin_confidence[j] /= bin_count[j]

    # weighted absolute differences
    total = len(predictions)
    ece = 0.0
    for j in range(num_bins):
        ece += abs(bin_accuracy[j] - bin_confidence[j]) * (bin_count[j] / total)

    return ece


def MCE(conf, pred, gt, conf_bin_num=10):
    """
    Maximal Calibration Error
    """
    df = pd.DataFrame({'true': gt, 'pred': pred, 'conf': conf})
    df['correct'] = (df.pred == df.true).astype(int)

    # digitize into bins
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))

    # compute per-bin accuracy, confidence, counts
    group_acc   = df.groupby('conf_bin')['correct'].mean()
    group_conf  = df.groupby('conf_bin')['conf'].mean()
    counts      = df.groupby('conf_bin')['conf'].count()

    # maximal weighted deviation
    mce = (abs(group_acc - group_conf) * (counts / len(df))).max()
    return mce


def AdaptiveECE(conf, pred, gt, conf_bin_num=10):
    """
    Adaptive (quantile) Expected Calibration Error
    """
    df = pd.DataFrame({'true': gt, 'pred': pred, 'conf': conf})
    df['correct'] = (df.pred == df.true).astype(int)

    # quantile-based binning
    df['conf_bin'] = KBinsDiscretizer(
        n_bins=conf_bin_num,
        encode='ordinal',
        strategy='quantile'
    ).fit_transform(conf[:, None]).astype(int)

    group_acc  = df.groupby('conf_bin')['correct'].mean()
    group_conf = df.groupby('conf_bin')['conf'].mean()
    counts     = df.groupby('conf_bin')['conf'].count()

    ace = (abs(group_acc - group_conf) * (counts / len(df))).sum()
    return ace

def logits_to_probs(model_output):
    """
    Convert model output to probabilities.

    In your CoOp/CLIP pipeline, model_output is normally raw logits.
    This function is also safe if model_output is already probability-like.
    """
    with torch.no_grad():
        output = model_output.detach()

        if output.ndim != 2:
            raise ValueError(
                f"Expected model output shape [B, C], but got {tuple(output.shape)}"
            )

        row_sums = output.sum(dim=1)
        is_non_negative = torch.all(output >= -1e-7).item()
        sums_to_one = torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4
        )

        if is_non_negative and sums_to_one:
            probs = output.clamp(min=0.0, max=1.0)
        else:
            probs = torch.softmax(output, dim=1)

        return probs


def binary_calibration_error(scores, targets, num_bins=20):
    """
    Binary calibration error.

    scores:
        Probability/confidence values, shape [N]

    targets:
        Binary labels, shape [N]
        1 = positive/correct
        0 = negative/incorrect

    Returns:
        Calibration error in [0, 1]
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if len(scores) == 0:
        return np.nan

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(scores, bin_edges[1:-1], right=True)

    ece = 0.0
    n = len(scores)

    for b in range(num_bins):
        mask = bin_ids == b

        if not np.any(mask):
            continue

        bin_acc = targets[mask].mean()
        bin_conf = scores[mask].mean()
        bin_weight = mask.sum() / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def multiclass_brier_score(probs, y_true):
    """
    Multi-class Brier score.

    Brier = mean_i sum_c (p_ic - y_ic)^2

    Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    n, num_classes = probs.shape

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(n), y_true] = 1.0

    per_sample_brier = np.sum((probs - one_hot) ** 2, axis=1)
    brier = float(np.mean(per_sample_brier))

    return brier, per_sample_brier


def classwise_toplabel_ece(confs, y_pred, y_true, num_classes, num_bins=20):
    """
    True-class-conditioned top-label class-wise ECE.

    For each ground-truth class c:
        - take samples where y_true == c
        - use max softmax confidence as confidence
        - correctness = 1 if y_pred == y_true else 0

    This is useful for imbalance-aware calibration.
    """
    confs = np.asarray(confs, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)

    rows = []

    for c in range(num_classes):
        mask = y_true == c
        n_c = int(mask.sum())
        freq_c = n_c / max(len(y_true), 1)

        if n_c == 0:
            rows.append({
                "class_id": c,
                "n": 0,
                "freq": 0.0,
                "class_acc": np.nan,
                "toplabel_ece": np.nan,
            })
            continue

        scores_c = confs[mask]
        targets_c = (y_pred[mask] == y_true[mask]).astype(np.float64)

        ece_c = binary_calibration_error(
            scores=scores_c,
            targets=targets_c,
            num_bins=num_bins
        )

        rows.append({
            "class_id": c,
            "n": n_c,
            "freq": freq_c,
            "class_acc": float(targets_c.mean()),
            "toplabel_ece": ece_c,
        })

    return pd.DataFrame(rows)


def classwise_ovr_ece(probs, y_true, num_bins=20):
    """
    One-vs-rest class-wise ECE.

    For each class c:
        - score = probability assigned to class c
        - target = 1 if true class is c else 0

    This checks whether each class probability is calibrated.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    num_classes = probs.shape[1]
    rows = []

    for c in range(num_classes):
        scores_c = probs[:, c]
        targets_c = (y_true == c).astype(np.float64)

        ece_c = binary_calibration_error(
            scores=scores_c,
            targets=targets_c,
            num_bins=num_bins
        )

        n_c = int((y_true == c).sum())
        freq_c = n_c / max(len(y_true), 1)

        rows.append({
            "class_id": c,
            "n": n_c,
            "freq": freq_c,
            "ovr_ece": ece_c,
        })

    return pd.DataFrame(rows)


def summarize_classwise_metric(df, metric_name):
    """
    Return macro, weighted, and max class-wise metric.
    """
    valid = df[(df["n"] > 0) & np.isfinite(df[metric_name])]

    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    macro_value = float(valid[metric_name].mean())
    weighted_value = float(np.average(valid[metric_name], weights=valid["n"]))
    max_value = float(valid[metric_name].max())

    return macro_value, weighted_value, max_value


def add_class_names(df, lab2cname):
    """
    Add class names to class-wise calibration table.
    """
    class_names = []

    for c in df["class_id"].tolist():
        if lab2cname is None:
            class_names.append(f"class_{c}")
        else:
            class_names.append(lab2cname.get(c, f"class_{c}"))

    df.insert(1, "class_name", class_names)
    return df



def PIECE(conf, knndist, pred, gt,
          dist_bin_num=10, conf_bin_num=10,
          knn_strategy='quantile'):
    """
    Proximity-Informed Expected Calibration Error
    """
    df = pd.DataFrame({
        'true':    gt,
        'pred':    pred,
        'conf':    conf,
        'knndist': knndist
    })
    df['correct'] = (df.pred == df.true).astype(int)

    # bin by knn distance
    df['knn_bin'] = KBinsDiscretizer(
        n_bins=dist_bin_num,
        encode='ordinal',
        strategy=knn_strategy
    ).fit_transform(df[['knndist']]).astype(int)

    # uniform bins for confidence
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))

    # compute per-(knn,conf) stats
    grp_acc   = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    grp_conf  = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    counts    = df.groupby(['knn_bin', 'conf_bin'])['conf'].count()

    piece = (abs(grp_acc - grp_conf) * (counts / len(df))).sum()
    return piece


def save_reliability_diagram_refstyle(confs, y_true, y_pred, out_path, n_bins=15, title=None):
    # ---- binning ----
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confs, bins) - 1  # [0..n_bins-1]

    bin_acc, bin_conf = [], []
    for i in range(n_bins):
        m = (bin_indices == i)
        if m.any():
            bin_acc.append((y_pred[m] == y_true[m]).mean())
            bin_conf.append(confs[m].mean())
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)

    bin_acc  = np.asarray(bin_acc, dtype=float)
    bin_conf = np.asarray(bin_conf, dtype=float)

    # ---- ECE (uses mean confidence per bin, like your ref code) ----
    weights = np.histogram(confs, bins)[0] / max(len(confs), 1)
    ece = float(np.sum(weights * np.abs(bin_conf - bin_acc)))

    # ---- plot (identical style to your reference) ----
    delta = 1.0 / n_bins
    x     = np.arange(0.0, 1.0, delta)                  # bar left edges
    mid   = np.linspace(delta/2, 1.0 - delta/2, n_bins) # bin centers
    error = np.abs(mid - bin_acc)                       # "Gap" vs bin center

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(6, 6))
    plt.xlim(0, 1); plt.ylim(0, 1)

    # dotted grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)

    # blue accuracy bars
    plt.bar(x, bin_acc, color='b', width=delta, align='edge',
            edgecolor='k', label='Outputs', zorder=5)

    # pale red hatched "gap" (stacked around the lower of acc vs bin center)
    plt.bar(x, error, bottom=np.minimum(bin_acc, mid),
            color='mistyrose', alpha=0.5, width=delta, align='edge',
            edgecolor='r', hatch='/', label='Gap', zorder=10)

    # y=x line
    plt.plot([0, 1], [0, 1], linestyle='--', color='tab:grey', zorder=15)

    # labels/legend + ECE badge
    plt.ylabel('Accuracy', fontsize=13); plt.xlabel('Confidence', fontsize=13)
    plt.legend(loc='upper left', framealpha=1.0, fontsize='medium')
    plt.text(0.025, 0.85, f'ECE: {ece*100:.2f}%',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round, pad=0.5', facecolor='wheat', edgecolor='orange'))

    if title is not None:
        plt.title(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_incorrect_fraction_histogram(
    incorrect_confidences,
    save_path,
    title="Fraction of Incorrect Samples by Confidence"
):
    """
    Plot and save a histogram showing the fraction of incorrect samples
    as a function of confidence.
    """
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(incorrect_confidences, bins=edges)
    frac = hist / len(incorrect_confidences) if len(incorrect_confidences) > 0 else np.zeros_like(hist)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.bar(
        edges[:-1],
        frac,
        width=np.diff(edges),
        align='edge',
        color='skyblue',
        edgecolor='black',
        zorder=5
    )

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Confidence", fontsize=18)
    ax.set_ylabel("Fraction of Incorrect Samples", fontsize=18)
    # start x-axis at 0.2 instead of 0
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.0, 0.35)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(alpha=0.3, linestyle='--', zorder=0)

    plt.tight_layout()
    fig.savefig(save_path, format="png", dpi=300)
    print(f"[✓] Saved incorrect-fraction histogram to {save_path}")
    plt.close(fig)

def plot_learning_curve(
    train_losses,
    val_losses,
    save_path,
    title="Learning Curve (Loss)",
):
    """
    Save an aesthetic learning curve plot (train loss + validation loss).
    train_losses, val_losses: list[float] of per-epoch mean losses.
    save_path: full output path to .png
    """

    import numpy as np

    # Convert to arrays (val may contain NaN if val loader not available)
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses   = np.asarray(val_losses,   dtype=float)

    epochs = np.arange(1, len(train_losses) + 1)

    plt.rcParams["font.family"] = "serif"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Main curves
    ax.plot(epochs, train_losses, linewidth=2.5, marker="o", markersize=4, label="Train Loss")
    ax.plot(epochs, val_losses,   linewidth=2.5, marker="s", markersize=4, label="Validation Loss")

    # Optional: highlight best validation epoch (ignore NaNs)
    if np.isfinite(val_losses).any():
        best_idx = np.nanargmin(val_losses)
        best_ep  = epochs[best_idx]
        best_v   = val_losses[best_idx]
        ax.scatter([best_ep], [best_v], s=70, zorder=10)
        ax.annotate(
            f"Best val @ epoch {best_ep}\n{best_v:.4f}",
            xy=(best_ep, best_v),
            xytext=(best_ep, best_v),
            textcoords="offset points",
            xycoords="data",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    # Cosmetics (paper-friendly)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=12)

    # Tight layout and save
    plt.tight_layout()
    fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] Saved learning curve to {save_path}")    


class EvaluatorBase:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname     = lab2cname
        self._correct       = 0
        self._total         = 0
        self._y_true        = []
        self._y_pred        = []
        self._confidences   = []
        self._probabilities = []  # full probability vectors for Brier and class-wise ECE
        self._knn_dists     = []  # for PIECE

        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None, "lab2cname is required for per-class results"
            self._per_class_res = defaultdict(list)
        else:
            self._per_class_res = None

    def reset(self):
        self._correct       = 0
        self._total         = 0
        self._y_true        = []
        self._y_pred        = []
        self._confidences   = []
        self._probabilities = []
        self._knn_dists     = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, model_output, ground_truth, knn_dist=None):
        # predictions and confidences
        # Convert logits to probabilities
        probs = logits_to_probs(model_output)

        # Predictions and top-label confidences
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1)[0]
        matches = preds.eq(ground_truth).float()

        # Update overall counters
        self._correct += int(matches.sum().item())
        self._total += ground_truth.size(0)

        # Store for final metrics
        self._y_true.extend(ground_truth.cpu().tolist())
        self._y_pred.extend(preds.cpu().tolist())
        self._confidences.extend(confs.cpu().tolist())

        # Store full probability vectors
        self._probabilities.extend(probs.cpu().numpy().tolist())

        if knn_dist is not None:
            self._knn_dists.extend(knn_dist.tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(ground_truth):
                self._per_class_res[label.item()].append(int(matches[i].item()))

    def evaluate(self):
        results = OrderedDict()

        # convert to numpy arrays
        #y_true = np.array(self._y_true)
        #y_pred = np.array(self._y_pred)
        #confs  = np.array(self._confidences)
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        confs  = np.array(self._confidences)
        probs  = np.array(self._probabilities, dtype=np.float64)

        num_classes = probs.shape[1]
        num_bins = getattr(self.cfg.TEST, "CALIBRATION_BINS", 20)


        # overall accuracy & error
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc

        # macro-F1
        macro_f1 = 100.0 * f1_score(
            y_true, y_pred,
            average="macro",
            labels=np.unique(y_true)
        )

        # calibration metrics
        ece_value       = ECE_Loss(
            num_bins= num_bins,  #20,
            predictions=y_pred,
            confidences=confs,
            correct=(y_pred == y_true).astype(int)
        ) * 100.0

        mce_value       = MCE(confs, y_pred, y_true) * 100.0
        adaptive_ece    = AdaptiveECE(confs, y_pred, y_true) * 100.0

        # ---------------------------------------------------------
        # New metric 1: Multi-class Brier score
        # ---------------------------------------------------------
        brier_value, per_sample_brier = multiclass_brier_score(probs, y_true)
        brier_norm = brier_value / num_classes

        # ---------------------------------------------------------
        # New metric 2: Class-wise ECE
        # ---------------------------------------------------------

        # A) True-class-conditioned top-label ECE
        toplabel_df = classwise_toplabel_ece(
            confs=confs,
            y_pred=y_pred,
            y_true=y_true,
            num_classes=num_classes,
            num_bins=num_bins
        )

        # B) One-vs-rest class-wise ECE
        ovr_df = classwise_ovr_ece(
            probs=probs,
            y_true=y_true,
            num_bins=num_bins
        )

        # Merge both class-wise tables
        classwise_df = toplabel_df.merge(
            ovr_df[["class_id", "ovr_ece"]],
            on="class_id",
            how="left"
        )

        # Add class-wise Brier score
        class_brier = []
        class_brier_norm = []

        for c in range(num_classes):
            mask = y_true == c

            if not np.any(mask):
                class_brier.append(np.nan)
                class_brier_norm.append(np.nan)
            else:
                value = float(np.mean(per_sample_brier[mask]))
                class_brier.append(value)
                class_brier_norm.append(value / num_classes)

        classwise_df["brier_true_class"] = class_brier
        classwise_df["brier_true_class_norm"] = class_brier_norm

        # Add class names
        classwise_df = add_class_names(classwise_df, self._lab2cname)

        # Summary values
        toplabel_ece_macro, toplabel_ece_weighted, toplabel_ece_max = summarize_classwise_metric(
            classwise_df,
            "toplabel_ece"
        )

        ovr_ece_macro, ovr_ece_weighted, ovr_ece_max = summarize_classwise_metric(
            classwise_df,
            "ovr_ece"
        )

        classwise_brier_macro = float(np.nanmean(classwise_df["brier_true_class"].values))
        classwise_brier_norm_macro = float(np.nanmean(classwise_df["brier_true_class_norm"].values))

        # PIECE only if we have knn distances
        if len(self._knn_dists) == len(confs):
            knn_arr    = np.array(self._knn_dists)
            piece_value = PIECE(confs, knn_arr, y_pred, y_true) * 100.0
        else:
            piece_value = None

        # build result dict
        """results["accuracy"]       = acc
        results["error_rate"]     = err
        results["macro_f1"]       = macro_f1
        results["ece"]            = ece_value
        results["mce"]            = mce_value
        results["adaptive_ece"]   = adaptive_ece"""

        results["accuracy"]       = acc
        results["error_rate"]     = err
        results["macro_f1"]       = macro_f1

        # Existing calibration metrics
        results["ece"]            = ece_value
        results["mce"]            = mce_value
        results["adaptive_ece"]   = adaptive_ece

        # New Brier metrics
        results["brier"]          = brier_value * 100.0
        results["brier_norm"]     = brier_norm  * 100.0

        # New class-wise ECE metrics
        results["toplabel_ece_macro"]    = toplabel_ece_macro * 100.0
        results["toplabel_ece_weighted"] = toplabel_ece_weighted * 100.0
        results["toplabel_ece_max"]      = toplabel_ece_max * 100.0

        results["ovr_ece_macro"]         = ovr_ece_macro * 100.0
        results["ovr_ece_weighted"]      = ovr_ece_weighted * 100.0
        results["ovr_ece_max"]           = ovr_ece_max * 100.0

        results["classwise_brier_macro"]      = classwise_brier_macro * 100.0
        results["classwise_brier_norm_macro"] = classwise_brier_norm_macro * 100.0
        if piece_value is not None:
            results["piece"]      = piece_value

        # print summary
        print(f"=> Total samples: {self._total:,}")
        print(f"=> Accuracy: {acc:.2f}%  Error rate: {err:.2f}%")
        print(f"=> Macro-F1: {macro_f1:.2f}%")
        #print(f"=> ECE: {ece_value:.2f}%  MCE: {mce_value:.2f}%  Adaptive ECE: {adaptive_ece:.2f}%")
        print(
            f"=> ECE@{num_bins}: {ece_value:.2f}%  "
            f"MCE: {mce_value:.2f}%  "
            f"Adaptive ECE: {adaptive_ece:.2f}%"
        )

        print(
            f"=> Brier score: {brier_value:.6f}  "
            f"Normalized Brier: {brier_norm:.6f}"
        )

        print(
            f"=> Top-label Class-wise ECE: "
            f"Macro={toplabel_ece_macro * 100.0:.2f}%  "
            f"Weighted={toplabel_ece_weighted * 100.0:.2f}%  "
            f"Max={toplabel_ece_max * 100.0:.2f}%"
        )

        print(
            f"=> One-vs-rest Class-wise ECE: "
            f"Macro={ovr_ece_macro * 100.0:.2f}%  "
            f"Weighted={ovr_ece_weighted * 100.0:.2f}%  "
            f"Max={ovr_ece_max * 100.0:.2f}%"
        )

        print(
            f"=> Class-wise Brier: "
            f"Macro={classwise_brier_macro:.6f}  "
            f"Normalized Macro={classwise_brier_norm_macro:.6f}"
        )
        if piece_value is not None:
            print(f"=> PIECE: {piece_value:.2f}%")

        # per-class results
        if self._per_class_res is not None:
            accs = []
            print("=> Per-class accuracies:")
            for lbl in sorted(self._per_class_res.keys()):
                corrects = self._per_class_res[lbl]
                cls_acc  = 100.0 * sum(corrects) / len(corrects)
                cname    = self._lab2cname[lbl]
                accs.append(cls_acc)
                print(f"* Class {lbl} ({cname}): {cls_acc:.2f}% [{len(corrects)} samples]")
            mean_pc = float(np.mean(accs))
            results["perclass_accuracy"] = mean_pc
            print(f"=> Average per-class accuracy: {mean_pc:.2f}%")

        # optionally save confusion matrix
        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(y_true, y_pred, normalize="true")
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        # --- save reliability diagram per seed ---
        fname = f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_overall_ece_seed{self.cfg.SEED}_ours.png"
        plot_path = osp.join(
            self.cfg.OUTPUT_DIR,
            f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_overall_ece_seed{self.cfg.SEED}.png"
        )
        save_reliability_diagram_refstyle(confs, y_true, y_pred, plot_path, n_bins=num_bins, title=None)
        print(f"Reliability diagram saved to {plot_path}")

        # --- save incorrect-fraction histogram (incorrect samples only) ---
        incorrect_mask  = (y_pred != y_true)
        incorrect_confs = confs[incorrect_mask]
        hist_path = osp.join(
            self.cfg.OUTPUT_DIR,
            f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_incorrect_fraction_seed{self.cfg.SEED}.png"
        )
        plot_incorrect_fraction_histogram(
            incorrect_confs,
            hist_path,
            title="Fraction of Incorrect Samples by Confidence"
        )

        if getattr(self.cfg.TEST, "SAVE_CLASSWISE_CALIBRATION", True):
            classwise_csv_path = osp.join(
                self.cfg.OUTPUT_DIR,
                f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_classwise_calibration_seed{self.cfg.SEED}.csv"
            )

            classwise_df.to_csv(classwise_csv_path, index=False)
            print(f"Class-wise calibration table saved to {classwise_csv_path}")

        return results
