# path: src/models/losses/edl.py
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EDLConfig:
    lambda_kl: float = 0.1  # Training Details


def kl_dirichlet(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL(Dir(alpha) || Dir(beta)).

    KL = log Γ(sum α) - sum log Γ(α_i) - (log Γ(sum β) - sum log Γ(β_i))
         + sum (α_i - β_i) (ψ(α_i) - ψ(sum α))

    Used in Eq. (7): beta = ones (Dir(1)).
    """
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_beta = torch.lgamma(sum_beta) - torch.lgamma(beta).sum(dim=1, keepdim=True)

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    t = ((alpha - beta) * (digamma_alpha - digamma_sum_alpha)).sum(dim=1, keepdim=True)
    kl = lnB_alpha - lnB_beta + t
    return kl


def edl_pixel_loss(alpha: torch.Tensor, y_onehot: torch.Tensor, cfg: EDLConfig) -> torch.Tensor:
    """
    Eq. (7): LEDL(α,y) = sum_c y_c ( ψ(S) - ψ(α_c) ) + lambda_KL * KL(Dir(α)||Dir(1))
    alpha: (B,K,H,W)
    y_onehot: (B,K,H,W)
    """
    s = alpha.sum(dim=1, keepdim=True)
    term = (y_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1, keepdim=True)

    beta = torch.ones_like(alpha)
    kl = kl_dirichlet(alpha, beta)

    return term + cfg.lambda_kl * kl


def edl_loss(alpha: torch.Tensor, y_onehot: torch.Tensor, cfg: EDLConfig, reduction: str = "mean") -> torch.Tensor:
    l = edl_pixel_loss(alpha, y_onehot, cfg)  # (B,1,H,W)
    if reduction == "mean":
        return l.mean()
    if reduction == "none":
        return l
    raise ValueError(reduction)
