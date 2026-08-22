import torch
from dassl.utils import Registry
from torch.nn import functional as F
import torch.nn as nn
# a mini‐registry just for regularizers
import math
REGULARIZER_REGISTRY = Registry('regularizers')

@REGULARIZER_REGISTRY.register()
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


