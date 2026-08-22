# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from torch.autograd import Variable

from scipy.special import lambertw
import numpy as np
import math


class LossRegistry:
    _losses = {}
    _weights = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._losses[name] = func
            return func
        return decorator
    
    @classmethod
    def get_loss(cls, name: str):
        return cls._losses.get(name)
    
    @classmethod
    def init_weights(cls, cfg):
        """Initialize weights from config"""
        for loss_name in cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES:
            cls._weights[loss_name] = getattr(cfg.TRAINER.COOP.LOSS[loss_name], 'WEIGHT')
        return cls._weights
    
    @classmethod
    def get_weight(cls, name: str):
        return cls._weights.get(name, 1.0)  # Default weight 1.0

@LossRegistry.register("CE")
def cross_entropy_loss(logits, label, **kwargs):
    """Standard cross-entropy loss"""
    return F.cross_entropy(logits, label)

@LossRegistry.register("DCA")
def dca_loss(logits, label, **kwargs):
    """Distance-Confidence Adjustment Loss"""
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    mean_conf = confidences.float().mean()
    acc = accuracies.float().mean()
    return torch.abs(mean_conf - acc)

@LossRegistry.register("MDCA")
def mdca_loss(logits, label, **kwargs):
    """MDCA (Multi-class Difference in Confidence and Accuracy) Loss"""
    output = torch.softmax(logits, dim=1)
    loss = torch.tensor(0.0).cuda()
    batch, classes = output.shape
    for c in range(classes):
        avg_count = (label == c).float().mean()
        avg_conf = torch.mean(output[:,c])
        loss += torch.abs(avg_conf - avg_count)
    total_classes = classes  
    loss /= total_classes   
    return loss


@LossRegistry.register("LS")
def label_smoothing_loss(logits, label, **kwargs):
    """Label Smoothing Loss
    
    Args:
        logits: Model predictions
        label: Ground truth labels
        alpha: Smoothing parameter (default: 0.05)
    """
    cfg = kwargs.get('cfg', None)
    alpha = cfg.TRAINER.COOP.LOSS.LS.ALPHA if cfg is not None else 0.1
    pred = logits.log_softmax(dim=-1)
    num_classes = pred.shape[-1]
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(alpha / (num_classes - 1))
        true_dist.scatter_(1, label.data.unsqueeze(1), 1.0 - alpha)
    return torch.mean(torch.sum(-true_dist * pred, dim=-1))


@LossRegistry.register("FL")
def focal_loss(logits, label, **kwargs):
    """Focal Loss implementation
    
    Args:
        logits: Model predictions
        label: Ground truth labels
        kwargs: Additional configuration, expects cfg.TRAINER.COOP.LOSS.FL.GAMMA
    """
    cfg = kwargs.get('cfg', None)
    gamma = cfg.TRAINER.COOP.LOSS.FL.GAMMA if cfg is not None else 3.0
    target = label.view(-1, 1)
    logpt = F.log_softmax(logits, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.exp()
    loss = -1 * (1-pt)**gamma * logpt
    return loss.mean()


class MMCE_weighted(nn.Module):
    """
    Computes MMCE_w loss.
    """
    def __init__(self):
        super(MMCE_weighted, self).__init__()
        self.device = torch.device("cuda")

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1*tensor2)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(torch.eq(predicted_labels, target),
                                    torch.ones(predicted_labels.shape).to(self.device),
                                    torch.zeros(predicted_labels.shape).to(self.device))

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
        k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (correct_mask.shape[0] - 2))


        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)  

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)
        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)
        return torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))

@LossRegistry.register("MMCE")
def mmce_loss(logits, label, **kwargs):
    """Maximum Mean Calibration Error Loss combined with Cross Entropy
    Combines standard cross entropy with MMCE calibration loss.
    
    Args:
        logits: Raw model output logits [batch_size, num_classes]
        label: Ground truth labels [batch_size]
    """
    # MMCE calibration loss
    mmce = MMCE_weighted()
    loss = mmce(logits, label)
    
    return loss

@LossRegistry.register("AS")
def angular_seperation_loss(features):
   """
   Angular Seperation Loss
   Encourages feature vectors to be more dissimilar/orthogonal to each other
   by minimizing average cosine similarity between feature pairs.
   
   Args:
       features: Feature embeddings tensor of shape (N, feature_dim)
                where N is number of classes
   Returns:
       loss: Scalar loss value representing average cosine similarity
   """
   # Get number of feature vectors
   N = features.shape[0]
   
   # Compute cosine similarity matrix between all pairs of features
   # shape: (N, N) where entry (i,j) is similarity between features i and j
   cos_sim_matrix = torch.matmul(features, features.T)
   
   # Create boolean mask for diagonal elements (self-similarities)
   mask = torch.eye(N, device=cos_sim_matrix.device).bool()
   
   # Zero out diagonal elements to ignore self-similarities 
   cos_sim_matrix_no_self = cos_sim_matrix.masked_fill(mask, 0)
   
   # For each feature, compute mean similarity with all other features
   # Divide by (N-1) to exclude self-similarity from mean
   neighbor_sims_mean = cos_sim_matrix_no_self.sum(dim=1) / (N - 1)
   
   # Average across all features to get final loss
   cosine_loss = neighbor_sims_mean.mean()
   
   return cosine_loss

@LossRegistry.register("O-TPT")
def O_tpt_loss(features):
            #------------------------------------------------- Householder Transform--------
            text_feature = features
            #print("text feature shape model:",text_feature.shape)
            #computing orthogonal constrained  SVD
            Wwt  =  torch.matmul(text_feature,text_feature.T)
            wwt_norm_col_HT = torch.linalg.norm(Wwt,dim=-1)
            Wwt_val_HT = wwt_norm_col_HT.mean()
            #wtW  =  torch.matmul(text_feature.T,text_feature)
            e = torch.eye(Wwt.shape[1], device=text_feature.device)
            M_norm = torch.linalg.norm(Wwt, dim=0,keepdim=True)
            scaled_e = e * M_norm
            # Subtract the scaled identity matrix from Wwt
            u = Wwt - scaled_e
            u_norm = torch.linalg.norm(u, dim=-1,keepdim=True)
            #u_norm = u_norm ** 2
            # We need to expand u_norm to shape (47, 47, 1) for broadcasting
            #u_norm_exp = u_norm.unsqueeze(2)  # Shape: (1, 47, 1)
            
            #Transposing the u for batch element column and coresponding column transpose matrix multiplication
            #u = u.T 
            v = u/u_norm
            normalized_matrix_exp = v.unsqueeze(2)  # Shape: (47, 47, 1)
            normalized_matrix_T_exp = v.unsqueeze(1)  # Shape: (47, 1, 47)
            
            # This will create a batch of 3 matrices, each of shape (47, 47)
            outer_products = normalized_matrix_exp @ normalized_matrix_T_exp  # Shape: (47, 47, 47)
            
            # Perform element-wise division of each outer product by the corresponding u_norm value
            divided_matrix = outer_products #/ u_norm_exp  # Shape: (47, 47, 47)

            # Multiply the result by 2
            scaled_matrix = 2 * divided_matrix  # Shape: (47, 47, 47)
            # Subtract the scaled result from the corresponding identity matrix for each batch
            identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)  # Shape: (47, 47, 47)
            # Subtract from identity matrix
            transformed_matrix = identity_matrix_dim - scaled_matrix  # Shape: (47, 47, 47)
            # Reshape M so that its columns are aligned for batch multiplication
            Wwt_exp = Wwt.unsqueeze(2)  # Shape: (47, 47, 1)

            # Perform batched matrix multiplication between transformed matrix and M_exp
            Hx = torch.bmm(transformed_matrix, Wwt_exp)  # Shape: (47, 47, 1)
            
            # Reshape back the result to (3, 3) by removing the last singleton dimension
            Hx = Hx.squeeze(2)  # Shape: (47, 47)
            #print("shape of Hx:",Hx.shape)
            #print("Hx:",Hx)
            #normalizing Column wise
            #Hx_norm =torch.linalg.norm(Hx, dim=0,keepdim=True)
            #Hx = Hx/Hx_norm
            Ht_ortho = Hx - e  
            Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
            Ht_ortho_norm_val = Ht_ortho_norm.mean()

            return Ht_ortho_norm_val


@LossRegistry.register("SMAC")
def smoothed_accuracy_and_confidence_loss(logits, label, **kwargs):
    """
    Smoothed Accuracy and Confidence Matching Loss
    Applies label smoothing to the target frequencies in MDCA loss calculation.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        label: Ground truth labels (batch_size,)
        alpha: Label smoothing parameter (default: 0.1)
            alpha=0 means no smoothing (hard labels)
            alpha>0 applies smoothing to target frequencies
    Returns:
        loss: Scalar loss value comparing model confidence with smoothed frequencies
    """
    cfg = kwargs.get('cfg', None)
    alpha = cfg.TRAINER.COOP.LOSS.SMAC.ALPHA if cfg is not None else 0.05

    # Input validation
    assert torch.is_tensor(logits), "Logits must be a tensor"
    assert torch.is_tensor(label), "Label must be a tensor"
    assert logits.dim() == 2, f"Logits must be 2D but got shape {logits.shape}"
    assert label.dim() == 1, f"Label must be 1D but got shape {label.shape}"
    
    # Convert logits to probabilities
    output = torch.softmax(logits, dim=1)
    batch, classes = output.shape
    
    # print(f"Label shape: {label.shape}, Label values: {label[:5]}")
    # print(f"Output shape: {output.shape}")
    # print(f"Sample output probabilities: {output[0]}")
    
    # Initialize loss
    loss = torch.tensor(0.0).cuda()
    
    for c in range(classes):
        # Get original hard label frequency
        avg_count = (label == c).float().mean()
        
        # Apply label smoothing to frequency:
        # - True class gets (1 - alpha) weight
        # - Other classes share alpha weight evenly
        smooth_freq = avg_count * (1 - alpha) + (1 - avg_count) * (alpha / (classes - 1))
        
        # Get model's average confidence for this class
        avg_conf = torch.mean(output[:,c])
        
        # print(f"Class {c}:")
        # print(f"  Original freq: {avg_count:.4f}")
        # print(f"  Smoothed freq: {smooth_freq:.4f}")
        # print(f"  Avg confidence: {avg_conf:.4f}")
        
        # Add absolute difference to loss
        class_loss = torch.abs(avg_conf - smooth_freq)
        loss += class_loss
        # print(f"  Class loss: {class_loss:.4f}")
    
    loss /= classes
    # print(f"Final loss: {loss:.4f}")
    
    return loss



@LossRegistry.register("ECCV_PENALTY")
def eccv_penalty_loss(logits, label, **kwargs):
    """ECCV Penalty Loss
    
    Constrains the output logits to remain within the bounds of the zero-shot prediction logits,
    penalizing values that exceed the max or go below the min of zero-shot predictions.
    """
    zs_pred = kwargs.get('zero_shot_logits', None)
    if zs_pred is None:
        return torch.tensor(0.0).cuda()  # Return zero if no zero-shot logits provided
        
    b, c = zs_pred.shape
    min_zs, max_zs = torch.min(zs_pred, 1)[0].unsqueeze(1), torch.max(zs_pred, 1)[0].unsqueeze(1)
            
    constr1 = F.relu(logits - max_zs.repeat(1, c)).mean()
    constr2 = F.relu(min_zs.repeat(1, c) - logits).mean()                             
    return (constr1 + constr2)


@LossRegistry.register("ECCV_ZS")
def eccv_zs_loss(logits, label, **kwargs):
    """ECCV Zero-Shot Loss with numerical stability safeguards
    
    Normalizes the output logits to match the range of the zero-shot logits,
    and then applies cross-entropy loss on these normalized values.
    """
    zs_pred = kwargs.get('zero_shot_logits', None)
    if zs_pred is None:
        return F.cross_entropy(logits, label)
    
    # Get min/max values
    b, c = zs_pred.shape
    min_op, max_op = torch.min(logits, 1)[0].unsqueeze(1), torch.max(logits, 1)[0].unsqueeze(1) 
    min_zs, max_zs = torch.min(zs_pred, 1)[0].unsqueeze(1), torch.max(zs_pred, 1)[0].unsqueeze(1)
    
    # Add safety epsilon and check for degenerate range
    eps = 1e-6
    range_op = max_op - min_op
    # If range is too small, use a minimum safe range
    range_op = torch.clamp(range_op, min=eps)
    
    # Normalize logits to [0,1] range with safety clamp
    op_norm = (logits - min_op) / range_op
    op_norm = torch.clamp(op_norm, min=0.0, max=1.0)
    
    # Scale to zero-shot range
    range_zs = torch.clamp(max_zs - min_zs, min=eps)  # Ensure ZS range is non-zero
    op_norm = op_norm * range_zs + min_zs
    
    # Apply cross-entropy loss with safety check
    return F.cross_entropy(op_norm, label)


@LossRegistry.register("MBLS")
def margin_based_logit_suppression_loss(logits, label, **kwargs):
    """Margin-based Logit Suppression Loss (MbLS)
    
    A complete loss function that combines CE with a marginal penalty:
    loss = CE + alpha * max(0, max(l^n) - l^n - margin)
    
    This is a standalone loss (no need to use CE separately).
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        label: Ground truth labels (batch_size,)
        cfg: Configuration object containing MBLS parameters
        
    Returns:
        loss: Combined loss value
    """
    cfg = kwargs.get('cfg', None)
    margin = cfg.TRAINER.COOP.LOSS.MBLS.MARGIN if cfg is not None else 10.0
    alpha = cfg.TRAINER.COOP.LOSS.MBLS.ALPHA if cfg is not None else 0.1
    
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(logits, label)
    
    # Get logit distances (difference between max logit and all logits)
    max_values = logits.max(dim=1, keepdim=True)[0]
    max_values = max_values.expand_as(logits)  # Expand to same shape as logits
    diff = max_values - logits
    
    # Apply linear penalty where logit distances exceed the margin
    margin_loss = F.relu(diff - margin).mean()
    
    # Combine losses
    total_loss = ce_loss + alpha * margin_loss
    
    return total_loss

@LossRegistry.register("LOGITNORM")
def logit_norm_loss(logits, label, **kwargs):
    """LogitNorm Loss
    
    Normalizes logits to unit length before applying cross-entropy.
    This helps with calibration by constraining the range of logit values.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        label: Ground truth labels (batch_size,)
        cfg: Configuration object containing LogitNorm parameters
        
    Returns:
        loss: Cross-entropy on normalized logits
    """
    cfg = kwargs.get('cfg', None)
    t = cfg.TRAINER.COOP.LOSS.LOGITNORM.TEMPERATURE if cfg is not None else 1.0
    
    # L2 normalize logits
    norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = torch.div(logits, norms) / t
    
    # Standard cross-entropy
    return F.cross_entropy(logit_norm, label)

@LossRegistry.register("MARGIN_MEAN_VAR")
def margin_mean_var_loss(logits, label, **kwargs):
    """
    Margin Mean Variance Loss
    
    R_margin = -alpha * mean(m_i) + beta * Var(m_i)
    where m_i = true_score - runner_up_score
    """
    alpha = 0.1  # Fixed value from your original code
    beta = 0.01  # Fixed value from your original code
    
    B, C = logits.shape
    # true-class scores
    true_scores = logits[torch.arange(B, device=logits.device), label]  # (B,)
    # runner-up scores
    tmp = logits.clone()
    tmp[torch.arange(B, device=logits.device), label] = -float("inf")
    runner_up = tmp.max(dim=1).values  # (B,)
    
    # margins
    margins = true_scores - runner_up  # (B,)
    mean_margin = margins.mean()
    var_margin = margins.var(unbiased=False)
    
    return -alpha * mean_margin + beta * var_margin


@LossRegistry.register("TEXT_MOMENT_MATCHING")
def text_moment_matching_loss(logits, label, **kwargs):
    """
    Text Moment Matching Loss
    
    Match mean & covariance of tuned vs. frozen text features.
    """
    text_features = kwargs.get('text_features', None)
    zero_shot_text_features = kwargs.get('zero_shot_text_features', None)
    
    if text_features is None or zero_shot_text_features is None:
        return torch.tensor(0.0, device=logits.device)
        
    tuned = text_features
    frozen = zero_shot_text_features
    
    B, D = tuned.shape
    
    # 1) compute per-batch means
    mu_t = tuned.mean(dim=0, keepdim=True)    # [1, D]
    mu_f = frozen.mean(dim=0, keepdim=True)   # [1, D]
    
    # 2) mean-matching term (squared L2)
    mean_cost = F.mse_loss(mu_t, mu_f, reduction="sum")
    
    # 3) center both sets
    Ct = tuned - mu_t  # [B, D]
    Cf = frozen - mu_f  # [B, D]
    
    # 4) compute covariances: (D×D) = (D×B) @ (B×D) / B
    cov_t = Ct.t() @ Ct / B
    cov_f = Cf.t() @ Cf / B
    
    # 5) covariance-matching term (Frobenius norm squared)
    cov_cost = (cov_t - cov_f).pow(2).sum()
    
    return mean_cost + cov_cost

@LossRegistry.register("MARGIN_MEAN_VAR_ALLCLASS_EXPLICIT")
def margin_mean_var_allclass_loss_explicit(logits, label, **kwargs):
    """
    Explicit pairwise implementation:
      m_{i,k} = z[i,y_i] - z[i,k] for k != y_i
      mbar_i  = mean_k m_{i,k}
      mu      = mean_i mbar_i
      var: 'per_sample' or 'all_pairs'
    """
    alpha = float(kwargs.get("alpha", 0.1))
    beta  = float(kwargs.get("beta", 0.01))
    variance_mode = kwargs.get("variance_mode", "per_sample")

    B, C = logits.shape
    if C < 2:
        raise ValueError("Need at least 2 classes.")

    device = logits.device
    idx = torch.arange(B, device=device)

    # (B,1)
    z_true = logits[idx, label].unsqueeze(1)

    # Collect other-class logits into (B, C-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[idx, label] = False
    others = logits[mask].view(B, C - 1)  # (B, C-1)

    # Pairwise margins and per-sample average
    margins = z_true - others             # (B, C-1)
    mbar = margins.mean(dim=1)            # (B,)
    mu = mbar.mean()

    # Variance term
    if variance_mode == "per_sample":
        var_term = mbar.var(unbiased=False)
    elif variance_mode == "all_pairs":
        Em2 = margins.pow(2).mean()  # average over all (i,k!=y)
        var_term = Em2 - mu**2
    else:
        raise ValueError("variance_mode must be 'per_sample' or 'all_pairs'.")

    return -alpha * mu + beta * var_term


@LossRegistry.register("EMBEDING_LOSS")
def embedding_loss(text_features):
        
    tuned = text_features.T
    embeding_mean = tuned.mean(0)
    embeding_distance = tuned - embeding_mean
    embeding_distance_norm = torch.linalg.norm(embeding_distance,dim=-1)
    embeding_distance_mean = embeding_distance_norm.mean()

    return embeding_distance_mean

@LossRegistry.register("MARGIN_MEAN_VAR_ALLCLASS_CB")
def margin_mean_var_allclass_cb(logits, labels, **kwargs):
    """
    Class-Balanced Margin Mean–Variance Loss (true vs ALL other classes).

    R_cb = -alpha * mu_cb + beta * var_cb
      where mu_cb and var_cb are computed with class-balanced weights.

    kwargs:
      alpha (float): weight for mean term (default 0.1)
      beta  (float): weight for variance term (default 0.01)
      variance_mode (str): 'all_pairs' (default) or 'per_sample'
      clamp_var_min (float): clamp lower bound for variance (default 0.0)
    """
    alpha = float(kwargs.get("alpha", 0.1))
    beta  = float(kwargs.get("beta", 0.01))
    variance_mode = kwargs.get("variance_mode", "all_pairs")
    clamp_var_min = float(kwargs.get("clamp_var_min", 0.0))

    B, C = logits.shape
    if C < 2:
        raise ValueError("Need at least 2 classes for pairwise margins.")
    device = logits.device
    idx = torch.arange(B, device=device)

    # True-class score and aggregates
    z_true = logits[idx, labels]                   # (B,)
    sum_all = logits.sum(dim=1)                    # (B,)
    sum_sq_all = (logits**2).sum(dim=1)            # (B,)

    others_sum = sum_all - z_true                  # (B,)
    others_sum_sq = sum_sq_all - z_true**2         # (B,)
    denom = (C - 1)

    # Per-sample average margin: mbar_i
    mbar = z_true - others_sum / denom             # (B,)

    # Per-sample all-pairs second moment: E_k[m_{i,k}^2]
    Em2_per_sample = ((denom * z_true**2)
                      - 2 * z_true * others_sum
                      + others_sum_sq) / denom     # (B,)

    # ----- Class-balanced weights -----
    # Count per class in the batch (ensure minlength=C)
    n_per_class = torch.bincount(labels, minlength=C).clamp_min(1)  # (C,)
    # Raw inverse-frequency per-sample weights
    w_raw = (1.0 / n_per_class[labels]).to(logits.dtype)            # (B,)
    # Normalize so sum_i w_i = 1
    w = w_raw / w_raw.sum()

    # Weighted mean
    mu_cb = (w * mbar).sum()

    # Weighted variance
    if variance_mode == "all_pairs":
        Em2_cb = (w * Em2_per_sample).sum()
        var_cb = Em2_cb - mu_cb**2
    elif variance_mode == "per_sample":
        var_cb = (w * (mbar - mu_cb)**2).sum()
    else:
        raise ValueError("variance_mode must be 'all_pairs' or 'per_sample'")

    var_cb = torch.clamp(var_cb, min=clamp_var_min)  # numeric safety if desired

    # Final regularizer
    return -alpha * mu_cb + beta * var_cb

@LossRegistry.register("MARGIN_MEAN_VAR_PER_SAMPLE")
def margin_mean_var_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.01,
    clamp_var_min: float = 0.0,
    reduction: str = "mean",  # "mean" | "sum" | "none"
):
    """
    Per-sample margin mean–variance loss (no batch statistics, no hinge).
      L_i = -alpha * mbar_i + beta * var_i
    where mbar_i and var_i are computed across the C-1 negatives of sample i.

    Args:
      logits: (B, C)
      labels: (B,)
      alpha, beta: weights for mean and variance terms
      clamp_var_min: numeric floor for per-sample variance (e.g., 0.0)
      reduction: "mean" (default), "sum", or "none" (return per-sample vector)

    Returns:
      loss (scalar if reduced, else (B,)), and a dict of diagnostics.
    """
    B, C = logits.shape
    if C < 2:
        raise ValueError("Need at least 2 classes.")

    device = logits.device
    idx = torch.arange(B, device=device)

    # True-class logit
    z_true = logits[idx, labels]              # (B,)

    # Aggregates over all classes (avoid building big masks)
    sum_all    = logits.sum(dim=1)            # (B,)
    sum_sq_all = (logits**2).sum(dim=1)       # (B,)
    others_sum    = sum_all - z_true          # (B,)
    others_sum_sq = sum_sq_all - z_true**2    # (B,)
    denom = (C - 1)

    # Per-sample mean margin across negatives: mbar_i
    mbar = z_true - others_sum / denom        # (B,)

    # Per-sample second moment E_k[m_{i,k}^2] (closed form)
    Em2 = ((denom * z_true**2)
           - 2 * z_true * others_sum
           + others_sum_sq) / denom           # (B,)

    # Per-sample variance across negatives
    var_i = torch.clamp(Em2 - mbar**2, min=clamp_var_min)   # (B,)

    # Per-sample objective
    loss_i = -alpha * mbar + beta * var_i                   # (B,)

    if reduction == "mean":
        loss = loss_i.mean()
    elif reduction == "sum":
        loss = loss_i.sum()
    elif reduction == "none":
        loss = loss_i
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    logs = {
        "mean_mbar": mbar.mean().detach(),
        "mean_var": var_i.mean().detach(),
        "mean_Em2": Em2.mean().detach(),
    }
    return loss, logs