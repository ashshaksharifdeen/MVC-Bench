import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import Classification

#from tools.metrics import ECE, MCE, AdaptiveECE, PIECE, ECE_KDE
from tools.metrics import (
    ECE,
    MCE,
    AdaptiveECE,
    PIECE,
    ECE_KDE,
    ensure_probability_matrix,
    multiclass_brier_score,
    top_label_classwise_ece,
    one_vs_rest_classwise_ece,
    summarize_classwise_metric,
)
from tools.plot import plot_reliability_diagram

@EVALUATOR_REGISTRY.register()
class VLClassification(Classification):
    """Evaluator for Vision-Language models."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        self._text_features = []
        self._image_features = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)



    def process(self, mo, gt, image_features, text_features):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # pred = mo.max(1)[1]
        # matches = pred.eq(gt).float()
        # self._correct += int(matches.sum().item())
        # self._total += gt.shape[0]
        self._y_score.extend(mo.data.cpu().numpy().tolist())
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._text_features.extend(text_features.data.cpu().numpy().tolist()) # record text feature and image features in CLIP
        self._image_features.extend(image_features.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)

    def evaluate(self, probs, labels, text_proximity):
        results = OrderedDict()

        ece_bin = self.cfg.CALIBRATION.METRICS.ECE_BINS

        # ---------------------------------------------------------
        # Safety conversion
        # ---------------------------------------------------------
        probs = ensure_probability_matrix(probs)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        total = len(labels)
        num_classes = probs.shape[1]

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------
        preds = np.argmax(probs, axis=1)
        confs = probs[np.arange(total), preds]

        correct = int(np.sum(preds == labels))
        accuracy = 100.0 * correct / total
        error = 100.0 - accuracy

        macro_f1 = 100.0 * f1_score(
            labels,
            preds,
            average="macro",
            labels=np.unique(labels)
        )

        # Store confidence as percentage for logging consistency
        avg_conf = 100.0 * float(np.mean(confs))

        # ---------------------------------------------------------
        # Existing calibration metrics
        # ---------------------------------------------------------
        ece = 100.0 * ECE(confs, preds, labels, ece_bin)
        mce = 100.0 * MCE(confs, preds, labels, ece_bin)
        ace = 100.0 * AdaptiveECE(confs, preds, labels, ece_bin)
        ece_kde = 100.0 * ECE_KDE(confs, preds, labels, p=1)

        # Optional PIECE
        piece = None
        # If you later use text_proximity, uncomment and define piece bins.
        # if text_proximity is not None:
        #     piece = 100.0 * PIECE(confs, text_proximity, preds, labels, 10, ece_bin)

        # ---------------------------------------------------------
        # New metric 1: Multi-class Brier score
        # ---------------------------------------------------------
        brier, per_sample_brier = multiclass_brier_score(probs, labels)

        # Normalized Brier is useful when comparing datasets
        # with different number of classes.
        brier_norm = brier / num_classes

        # ---------------------------------------------------------
        # New metric 2: Class-wise ECE
        # ---------------------------------------------------------

        # A) True-class-conditioned top-label class-wise ECE
        toplabel_df = top_label_classwise_ece(
            probs,
            labels,
            conf_bin_num=ece_bin
        )

        # B) One-vs-rest class-wise ECE
        ovr_df = one_vs_rest_classwise_ece(
            probs,
            labels,
            conf_bin_num=ece_bin
        )

        # Merge both class-wise tables
        classwise_df = toplabel_df.merge(
            ovr_df[["class_id", "ovr_ece"]],
            on="class_id",
            how="left"
        )

        # Add class-wise Brier score based on true class
        class_brier = []
        class_brier_norm = []

        for c in range(num_classes):
            mask = labels == c

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
        class_names = []
        for c in classwise_df["class_id"].tolist():
            if self._lab2cname is not None:
                class_names.append(self._lab2cname.get(c, f"class_{c}"))
            else:
                class_names.append(f"class_{c}")

        classwise_df.insert(1, "class_name", class_names)

        # Summarise class-wise metrics
        toplabel_ece_macro, toplabel_ece_weighted, toplabel_ece_max = summarize_classwise_metric(
            classwise_df,
            "toplabel_ece"
        )

        ovr_ece_macro, ovr_ece_weighted, ovr_ece_max = summarize_classwise_metric(
            classwise_df,
            "ovr_ece"
        )

        classwise_brier_macro = float(
            np.nanmean(classwise_df["brier_true_class"].values)
        )

        classwise_brier_norm_macro = float(
            np.nanmean(classwise_df["brier_true_class_norm"].values)
        )

        # ---------------------------------------------------------
        # Save class-wise CSV
        # ---------------------------------------------------------
        classwise_csv_path = osp.join(
            self.cfg.OUTPUT_DIR,
            f"{self.cfg.DATASET.NAME}_{self.cfg.TRAINER.NAME}_classwise_calibration.csv"
        )

        classwise_df.to_csv(classwise_csv_path, index=False)

        # ---------------------------------------------------------
        # Results dictionary
        # ---------------------------------------------------------
        results["accuracy"] = accuracy
        results["error_rate"] = error
        results["macro_f1"] = macro_f1
        results["confidence"] = avg_conf

        # Existing calibration metrics
        results["ece"] = ece
        results["mce"] = mce
        results["ace"] = ace
        results["ece_kde"] = ece_kde

        if piece is not None:
            results["piece"] = piece

        # New Brier metrics
        results["brier"] = brier * 100.0
        results["brier_norm"] = brier_norm * 100.0

        # New class-wise ECE metrics
        results["toplabel_ece_macro"] = toplabel_ece_macro * 100.0
        results["toplabel_ece_weighted"] = toplabel_ece_weighted * 100.0
        results["toplabel_ece_max"] = toplabel_ece_max * 100.0

        results["ovr_ece_macro"] = ovr_ece_macro * 100.0
        results["ovr_ece_weighted"] = ovr_ece_weighted * 100.0
        results["ovr_ece_max"] = ovr_ece_max * 100.0

        # New class-wise Brier metrics
        results["classwise_brier_macro"] = classwise_brier_macro * 100.0
        results["classwise_brier_norm_macro"] = classwise_brier_norm_macro * 100.0

        # ---------------------------------------------------------
        # Print class-wise table
        # ---------------------------------------------------------
        print("\n=> Class-wise calibration table")
        print(
            f"{'Class':<25} {'N':<8} {'Freq':<10} "
            f"{'Acc':<10} {'Top-ECE':<10} {'OVR-ECE':<10} {'Brier':<10}"
        )
        print("-" * 90)

        for _, row in classwise_df.iterrows():
            print(
                f"{row['class_name']:<25} "
                f"{int(row['n']):<8d} "
                f"{row['freq']:<10.4f} "
                f"{row['class_acc'] * 100.0:>7.2f}%  "
                f"{row['toplabel_ece'] * 100.0:>7.2f}%  "
                f"{row['ovr_ece'] * 100.0:>7.2f}%  "
                f"{row['brier_true_class']:>9.6f}"
            )

        print(f"\nClass-wise calibration CSV saved to: {classwise_csv_path}")

        # ---------------------------------------------------------
        # Final log block
        # Keep this block parse-friendly.
        # ---------------------------------------------------------
        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {accuracy:.2f}%\n"
            f"* error: {error:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* confidence: {avg_conf:.2f}%\n"
            f"* ece: {ece:.2f}%\n"
            f"* mce: {mce:.2f}%\n"
            f"* ace: {ace:.2f}%\n"
            f"* ece_kde: {ece_kde:.2f}%\n"
            f"* brier: {brier:.6f}%\n"
            f"* brier_norm: {brier_norm:.6f}%\n"
            f"* toplabel_ece_macro: {toplabel_ece_macro * 100.0:.2f}%\n"
            f"* toplabel_ece_weighted: {toplabel_ece_weighted * 100.0:.2f}%\n"
            f"* toplabel_ece_max: {toplabel_ece_max * 100.0:.2f}%\n"
            f"* ovr_ece_macro: {ovr_ece_macro * 100.0:.2f}%\n"
            f"* ovr_ece_weighted: {ovr_ece_weighted * 100.0:.2f}%\n"
            f"* ovr_ece_max: {ovr_ece_max * 100.0:.2f}%\n"
            f"* classwise_brier_macro: {classwise_brier_macro:.6f}%\n"
            f"* classwise_brier_norm_macro: {classwise_brier_norm_macro:.6f}%\n"
        )

        # ---------------------------------------------------------
        # Reliability diagram: overall
        # ---------------------------------------------------------
        base_dir = self.cfg.OUTPUT_DIR
        base_name = self.cfg.DATASET.NAME + "_" + self.cfg.TRAINER.NAME
        overall_plot_name = base_name + "_overall_ece.png"
        overall_plot_path = osp.join(base_dir, overall_plot_name)

        plot_reliability_diagram(
            preds,
            confs,
            labels,
            ece_bin,
            None,
            overall_plot_path
        )

        # ---------------------------------------------------------
        # Reliability diagrams: true-class-conditioned
        # ---------------------------------------------------------
        for c in range(num_classes):
            mask = labels == c

            if not np.any(mask):
                continue

            class_name = self._lab2cname.get(c, f"class_{c}") if self._lab2cname else f"class_{c}"
            safe_name = str(class_name).replace("/", "_").replace(" ", "_")

            plot_name = base_name + f"_true_class_{safe_name}_ece.png"
            plot_path = osp.join(base_dir, plot_name)

            plot_reliability_diagram(
                preds[mask],
                confs[mask],
                labels[mask],
                ece_bin,
                None,
                plot_path
            )

        return results