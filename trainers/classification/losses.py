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

