'''
    https://github.com/MiaoXiong2320/ProximityBias-Calibration/blob/main/utils/metrics.py
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    
    This file contains the code for evaluation metrics:
    - ECE 
    - MCE
    - Dist-aware ECE
    - Adaptive ECE
    ...
'''

import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score, roc_auc_score, auc
import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin



   
# def ECE(conf, pred, gt, conf_bin_num = 10):

#     """
#     Expected Calibration Error
    
#     Args:
#         conf (numpy.ndarray): list of confidences
#         pred (numpy.ndarray): list of predictions
#         true (numpy.ndarray): list of true labels
#         bin_size: (float): size of one bin (0,1)  
        
#     Returns:
#         ece: expected calibration error
#     """
#     df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
#     df['correct'] = (df.pred == df.ys).astype('int')
    

#     bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
#     df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
#     # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
#     # groupy by knn + conf
#     group_acc = df.groupby(['conf_bin'])['correct'].mean()
#     group_confs = df.groupby(['conf_bin'])['conf'].mean()
#     counts = df.groupby(['conf_bin'])['conf'].count()
#     ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
#     return ece

def ECE(conf, pred, gt, conf_bin_num = 10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    bins = np.linspace(0, 1, conf_bin_num+1)
    bin_indices = np.digitize(conf, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(conf_bin_num):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(gt[in_bin] == pred[in_bin])
            mean_confidence = np.mean(conf[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)


    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)


    weights = np.histogram(conf, bins)[0] / len(conf)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))
        
    return ece
     
def PIECE(conf, knndist, pred, gt, dist_bin_num =10, conf_bin_num = 10, knn_strategy='quantile'):

    """
    Proximity-Informed Expected Calibration Error 
    
    Args:
        conf (numpy.ndarray): list of confidences
        knndist (numpy.ndarray): list of distances of which a sample to its K nearest neighbors
        pred (numpy.ndarray): list of predictions
        gt (numpy.ndarray): list of true labels
        dist_bin_num: (float): the number of bins for knndist
        conf_bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    
    
    df = pd.DataFrame({'ys':gt, 'knndist':knndist, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['knn_bin'] = KBinsDiscretizer(n_bins=dist_bin_num, encode='ordinal',strategy=knn_strategy).fit_transform(knndist[:, np.newaxis])
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    group_confs = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    counts = df.groupby(['knn_bin', 'conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    # group by only knn
    # group_acc = df.groupby(['knn_bin'])['correct'].mean()
    # group_confs = df.groupby(['knn_bin'])['conf'].mean()
    # counts = df.groupby(['knn_bin'])['conf'].count()
    # ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    
    # n = len(conf)
    # ece = 0  # Starting error
    # upper_bounds = np.arange(conf_bin_size, 1+conf_bin_size, conf_bin_size)  # Get bounds of bins
    # for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
    #     acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-conf_bin_size, conf_thresh, conf, pred, gt)        
    #     ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def MCE(conf, pred, gt, conf_bin_num = 10):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce


def AdaptiveECE(conf, pred, gt, conf_bin_num=10):
    """
    Adaptive Expected Calibration Error (ACE)
    - Robust to NaNs/Infs in inputs
    - Quantile-based binning
    """
    # --- to numpy 1D ---
    conf = np.asarray(conf, dtype=float).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    gt   = np.asarray(gt).reshape(-1)

    # --- keep only finite rows across all three arrays ---
    finite_mask = np.isfinite(conf) & np.isfinite(pred) & np.isfinite(gt)
    if not finite_mask.any():
        return 0.0
    conf = conf[finite_mask]
    pred = pred[finite_mask]
    gt   = gt[finite_mask]

    # --- clip confidences to [0,1] just in case ---
    conf = np.clip(conf, 0.0, 1.0)

    # --- adapt bin count to data ---
    # unique confs may be < requested bins (e.g., identical conf everywhere)
    unique_confs = np.unique(conf)
    # At least 2 bins for KBinsDiscretizer to work sensibly; otherwise fall back to 1 bin manually
    max_bins = min(len(conf), unique_confs.size)
    if max_bins <= 1:
        # single bin case: everything in one bucket
        df = pd.DataFrame({'ys': gt, 'conf': conf, 'pred': pred})
        df['correct'] = (df['pred'] == df['ys']).astype(int)
        acc = df['correct'].mean()
        cmean = df['conf'].mean()
        return float(abs(acc - cmean))  # weight = 1 since only one bin

    n_bins = int(max(2, min(conf_bin_num, max_bins)))

    # --- binning (quantiles) ---
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    conf_bins = binner.fit_transform(conf[:, None]).astype(int).ravel()

    # --- group & compute ACE ---
    df = pd.DataFrame({'ys': gt, 'conf': conf, 'pred': pred, 'conf_bin': conf_bins})
    df['correct'] = (df['pred'] == df['ys']).astype(int)

    group_acc   = df.groupby('conf_bin')['correct'].mean()
    group_confs = df.groupby('conf_bin')['conf'].mean()
    counts      = df.groupby('conf_bin')['conf'].count()

    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    return float(ace)

#def AdaptiveECE(conf, pred, gt, conf_bin_num=10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ace: expected calibration error
    """
    """df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='quantile').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ace"""

def ECE_KDE(conf, pred, gt, p=1, bandwidth=None):
    """
    Expected Calibration Error using Kernel Density Estimation (ECE-KDE)
    
    This implements a simplified version of the ECE-KDE metric from the paper:
    "A Consistent and Differentiable Lp Canonical Calibration Error Estimator" (NeurIPS 2022)
    
    The implementation uses a Gaussian kernel for simplicity and computational efficiency,
    rather than the Dirichlet/Beta kernels in the original paper. This simplification is
    appropriate for medical imaging datasets with few-shot learning for several reasons:
    
    1. Computational Efficiency: The Gaussian kernel is faster to compute and has lower 
       memory requirements than the Dirichlet kernel, important for iterative training.
    
    2. Top-Label Focus: For medical diagnosis tasks, top-label (confidence) calibration 
       is often the primary concern, making the full canonical calibration unnecessary.
    
    3. Few-Shot Robustness: In few-shot learning (e.g., 8-shot), simpler models with fewer
       parameters tend to be more robust, and the same applies to calibration metrics.
    
    4. Adaptive Bandwidth: The implementation uses an adaptive bandwidth based on dataset 
       size, which is particularly important for few-shot learning where test sets can vary.
    
    The key difference from traditional binning-based ECE is that KDE provides a smooth, 
    continuous estimate of the relationship between confidence and accuracy, avoiding
    artifacts from arbitrary bin boundaries.
    
    Args:
        conf (numpy.ndarray): list of confidences (max probability values)
        pred (numpy.ndarray): list of predictions (class indices)
        gt (numpy.ndarray): list of true labels (class indices)
        p (int): order of the norm (1 for L1 norm, 2 for L2 norm)
        bandwidth (float): optional manual bandwidth parameter, auto-selected if None
        
    Returns:
        ece_kde: expected calibration error using KDE
    """
    
    # Convert inputs to torch tensors if needed
    if not isinstance(conf, torch.Tensor):
        conf = torch.tensor(conf, dtype=torch.float32)
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.long)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt, dtype=torch.long)
    
    # Calculate accuracy (1 if correct, 0 if wrong)
    acc = (pred == gt).float()
    
    # Get number of samples
    n = len(conf)
    
    # Set bandwidth based on dataset size (determined from test set)
    # For few-shot learning scenarios, larger bandwidths prevent overfitting
    if bandwidth is None:
        if n < 100:  # Very small datasets like test sets for few-shot learning
            bandwidth = 0.3
        elif n < 500:  # Small test sets
            bandwidth = 0.2
        elif n < 2000:  # Medium test sets
            bandwidth = 0.1
        else:  # Large test sets
            bandwidth = 0.05
    
    # Calculate kernel matrix using Gaussian kernel
    # This is simpler than the Beta/Dirichlet kernels in the paper but still effective
    conf_expanded = conf.unsqueeze(1)
    diff = conf_expanded - conf_expanded.T
    kernel = torch.exp(-(diff**2) / (2 * bandwidth**2))
    kernel.fill_diagonal_(0)  # exclude self-comparisons for leave-one-out estimation
    
    # Normalize kernel to ensure proper weighting
    kernel_sum = kernel.sum(dim=1, keepdim=True)
    kernel_sum = torch.clamp(kernel_sum, min=1e-10)  # avoid division by zero
    kernel_norm = kernel / kernel_sum
    
    # Estimate accuracy for each confidence value using KDE
    estimated_acc = torch.matmul(kernel_norm, acc)
    
    # Calculate ECE-KDE using Lp norm (default p=1 for L1 norm)
    ece_kde = torch.mean(torch.abs(conf - estimated_acc)**p)
    
    return ece_kde.item()

def ensure_probability_matrix(scores):
    """
    Ensure input is a valid probability matrix [N, C].

    If the input already looks like probabilities, return it after clipping.
    If it looks like logits, apply softmax.
    """
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 2:
        raise ValueError(f"Expected shape [N, C], but got {scores.shape}")

    row_sums = scores.sum(axis=1)
    is_non_negative = np.all(scores >= -1e-8)
    sums_to_one = np.allclose(row_sums, 1.0, atol=1e-4, rtol=1e-4)

    if is_non_negative and sums_to_one:
        probs = np.clip(scores, 0.0, 1.0)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        return probs

    # Stable softmax for logits
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)

    return probs


def multiclass_brier_score(probs, labels):
    """
    Multi-class Brier score.

    Brier = mean_i sum_c (p_ic - y_ic)^2

    Lower is better.
    """
    probs = ensure_probability_matrix(probs)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    n, num_classes = probs.shape

    if len(labels) != n:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match probabilities ({n})"
        )

    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Labels must be in [0, {num_classes - 1}], "
            f"but got min={labels.min()}, max={labels.max()}"
        )

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0

    per_sample_brier = np.sum((probs - one_hot) ** 2, axis=1)
    brier = float(np.mean(per_sample_brier))

    return brier, per_sample_brier


def binary_calibration_error(scores, targets, conf_bin_num=10):
    """
    Generic binary calibration error.

    scores:
        probability/confidence values, shape [N]

    targets:
        binary targets, shape [N]
        1 = positive/correct
        0 = negative/incorrect

    Returns value in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)

    finite_mask = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[finite_mask]
    targets = targets[finite_mask]

    if len(scores) == 0:
        return np.nan

    scores = np.clip(scores, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, conf_bin_num + 1)
    bin_ids = np.digitize(scores, bins[1:-1], right=True)

    ece = 0.0
    n = len(scores)

    for b in range(conf_bin_num):
        mask = bin_ids == b

        if not np.any(mask):
            continue

        bin_acc = targets[mask].mean()
        bin_conf = scores[mask].mean()
        bin_weight = mask.sum() / n

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def top_label_classwise_ece(probs, labels, conf_bin_num=10):
    """
    True-class-conditioned top-label class-wise ECE.

    For each ground-truth class c:
        - select samples where label == c
        - confidence = max softmax probability
        - correctness = 1 if prediction is correct else 0

    This is the most direct class-wise version of your current ECE.
    """
    probs = ensure_probability_matrix(probs)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    preds = np.argmax(probs, axis=1)
    confs = probs[np.arange(len(labels)), preds]

    num_classes = probs.shape[1]
    rows = []

    for c in range(num_classes):
        mask = labels == c
        n_c = int(mask.sum())
        freq_c = n_c / max(len(labels), 1)

        if n_c == 0:
            rows.append({
                "class_id": c,
                "n": 0,
                "freq": 0.0,
                "class_acc": np.nan,
                "toplabel_ece": np.nan,
            })
            continue

        class_confs = confs[mask]
        class_correct = (preds[mask] == labels[mask]).astype(np.float64)

        class_ece = binary_calibration_error(
            class_confs,
            class_correct,
            conf_bin_num=conf_bin_num
        )

        class_acc = float(class_correct.mean())

        rows.append({
            "class_id": c,
            "n": n_c,
            "freq": freq_c,
            "class_acc": class_acc,
            "toplabel_ece": class_ece,
        })

    return pd.DataFrame(rows)


def one_vs_rest_classwise_ece(probs, labels, conf_bin_num=10):
    """
    One-vs-rest class-wise ECE.

    For each class c:
        - score = predicted probability assigned to class c
        - target = 1 if true label is c else 0

    This checks whether each class probability is calibrated.
    """
    probs = ensure_probability_matrix(probs)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    num_classes = probs.shape[1]
    rows = []

    for c in range(num_classes):
        scores_c = probs[:, c]
        targets_c = (labels == c).astype(np.float64)

        class_ece = binary_calibration_error(
            scores_c,
            targets_c,
            conf_bin_num=conf_bin_num
        )

        n_c = int((labels == c).sum())
        freq_c = n_c / max(len(labels), 1)

        rows.append({
            "class_id": c,
            "n": n_c,
            "freq": freq_c,
            "ovr_ece": class_ece,
        })

    return pd.DataFrame(rows)


def summarize_classwise_metric(df, metric_name):
    """
    Return macro, weighted, and max class-wise metric.
    Values are returned in raw scale [0, 1].
    """
    valid = df[(df["n"] > 0) & np.isfinite(df[metric_name])]

    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    macro_value = float(valid[metric_name].mean())
    weighted_value = float(np.average(valid[metric_name], weights=valid["n"]))
    max_value = float(valid[metric_name].max())

    return macro_value, weighted_value, max_value