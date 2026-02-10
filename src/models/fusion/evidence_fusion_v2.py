# path: src/models/fusion/evidence_fusion_v2.py
"""
Evidence Fusion with Stop-Gradient Gating (Updated for Stage 3).

Changes from original evidence_fusion.py:
  B) Apply stop-gradient on π_H, π_L (gating weights) to prevent self-reinforcing loop.
  B) Fuse evidence (not probabilities): e* = π_H * e_H + π_L * e_L

The key insight: if gradients flow through the gating weights, the model might
learn to always favor one branch regardless of confidence, creating a feedback loop.
By detaching the weights, each branch must earn its contribution based on evidence quality.

References:
  - Goal B in task spec: evidence fusion with stop-gradient
  - DUET Paper Eq. (10)(11)(12)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EvidenceFusionV2Config:
    """Configuration for evidence fusion module.
    
    Attributes:
        eps: Small constant for numerical stability in division.
        detach_weights: Apply stop-gradient to gating weights (recommended: True).
        detach_concentration: Also detach concentration S before weight computation.
        fusion_type: 'evidence_weighted' (paper) or 'avg_prob' (ablation).
        log_weights: Log mean fusion weights for diagnostics.
    """
    eps: float = 1e-6
    detach_weights: bool = True  # CRITICAL: prevents self-reinforcing loop
    detach_concentration: bool = False  # Optional additional stop-grad
    fusion_type: str = "evidence_weighted"
    log_weights: bool = False


def dirichlet_mean_binary(alpha: torch.Tensor) -> torch.Tensor:
    """Compute expected probability for binary classification under Dirichlet.
    
    Eq. (6): p_hat = alpha_polyp / sum_c alpha_c
    
    ASSUMPTION: channel 1 = polyp, channel 0 = background.
    """
    s = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p_polyp = alpha[:, 1:2, :, :] / s
    return p_polyp


def dirichlet_variance_binary(alpha: torch.Tensor) -> torch.Tensor:
    """Compute variance of Bernoulli under Beta (binary Dirichlet).
    
    Var(p) = a*b / ((a+b)^2 * (a+b+1))
    where a = alpha_polyp, b = alpha_bg
    """
    a = alpha[:, 1:2, :, :]  # polyp
    b = alpha[:, 0:1, :, :]  # background
    s = (a + b).clamp_min(1e-12)
    return (a * b) / (s * s * (s + 1.0))


def compute_dirichlet_uncertainty(alpha: torch.Tensor, method: str = 'variance') -> torch.Tensor:
    """Compute uncertainty from Dirichlet concentration.
    
    Args:
        alpha: (B, K, H, W) Dirichlet concentration parameters
        method: 'variance' (Beta variance) or 'entropy' (Dirichlet entropy proxy)
        
    Returns:
        uncertainty: (B, 1, H, W) per-pixel uncertainty
    """
    if method == 'variance':
        return dirichlet_variance_binary(alpha)
    elif method == 'entropy':
        # Approximate entropy: higher concentration = lower uncertainty
        s = alpha.sum(dim=1, keepdim=True)
        return 1.0 / (s + 1.0)  # Simple proxy
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


class EvidenceFusionV2(nn.Module):
    """Evidence-level fusion with stop-gradient gating.
    
    Key equations:
    - Eq. (10): π_H = S_H / (S_H + S_L + ε), π_L = 1 - π_H
    - Eq. (11): e* = π_H * e_H + π_L * e_L  [FUSE EVIDENCE, not probs]
    - Eq. (12): α* = e* + 1, p* = α*_polyp / Σ α*
    
    CRITICAL: π_H and π_L are detached (no gradient flow) to prevent
    the gating mechanism from self-reinforcing.
    """
    
    def __init__(self, cfg: EvidenceFusionV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        
        if cfg.detach_weights:
            logger.info("EvidenceFusionV2: stop-gradient enabled for fusion weights")
    
    def forward(
        self,
        alpha_h: torch.Tensor,
        e_h: torch.Tensor,
        alpha_l: torch.Tensor,
        e_l: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Fuse evidence from high-freq and low-freq branches.
        
        Args:
            alpha_h: (B, K, H, W) Dirichlet concentration from high-freq branch
            e_h: (B, K, H, W) evidence from high-freq branch (alpha - 1)
            alpha_l: (B, K, H, W) Dirichlet concentration from low-freq branch
            e_l: (B, K, H, W) evidence from low-freq branch
            return_weights: Return fusion weights for visualization
            
        Returns:
            alpha_f: (B, K, H, W) fused Dirichlet concentration
            p_f: (B, 1, H, W) fused polyp probability
            u_f: (B, 1, H, W) fused uncertainty
            (pi_h, pi_l): Optional fusion weights if return_weights=True
        """
        if self.cfg.fusion_type == "avg_prob":
            # Ablation: simple probability averaging (NOT paper method)
            p_h = dirichlet_mean_binary(alpha_h)
            p_l = dirichlet_mean_binary(alpha_l)
            p_f = 0.5 * (p_h + p_l)
            
            # Pseudo alpha for uncertainty computation (NOT rigorous)
            alpha_f = torch.cat([1.0 - p_f, p_f], dim=1) * 10.0
            u_f = dirichlet_variance_binary(alpha_f)
            
            if return_weights:
                pi_h = torch.full_like(p_f, 0.5)
                return alpha_f, p_f, u_f, (pi_h, 1.0 - pi_h)
            return alpha_f, p_f, u_f, None
        
        # ======== Evidence-weighted fusion (paper method) ========
        
        # Compute per-pixel concentration: S = Σ_c α_c
        s_h = alpha_h.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        s_l = alpha_l.sum(dim=1, keepdim=True)
        
        # Optionally detach concentration before weight computation
        if self.cfg.detach_concentration:
            s_h = s_h.detach()
            s_l = s_l.detach()
        
        # Compute fusion weights: π_H = S_H / (S_H + S_L + ε)
        denom = s_h + s_l + self.cfg.eps
        pi_h = s_h / denom
        pi_l = s_l / denom  # Equivalent to 1 - pi_h if eps=0
        
        # CRITICAL: Apply stop-gradient to prevent self-reinforcing loop
        if self.cfg.detach_weights:
            pi_h = pi_h.detach()
            pi_l = pi_l.detach()
        
        # Fuse EVIDENCE (not probabilities): e* = π_H * e_H + π_L * e_L
        e_f = pi_h * e_h + pi_l * e_l
        
        # Compute fused concentration: α* = e* + 1
        alpha_f = e_f + 1.0
        
        # Compute fused probability: p* = α*_polyp / Σ α*
        p_f = dirichlet_mean_binary(alpha_f)
        
        # Compute fused uncertainty
        u_f = dirichlet_variance_binary(alpha_f)
        
        # Log mean weights if enabled
        if self.cfg.log_weights:
            logger.debug(
                f"Fusion weights: mean(π_H)={pi_h.mean().item():.4f}, "
                f"mean(π_L)={pi_l.mean().item():.4f}"
            )
        
        if return_weights:
            return alpha_f, p_f, u_f, (pi_h, pi_l)
        return alpha_f, p_f, u_f, None
    
    def compute_branch_correctness(
        self,
        p_h: torch.Tensor,
        p_l: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-pixel correctness masks for each branch.
        
        Useful for gating correctness correlation analysis.
        
        Args:
            p_h: (B, 1, H, W) high-freq branch probability
            p_l: (B, 1, H, W) low-freq branch probability
            target: (B, 1, H, W) ground truth mask
            threshold: Classification threshold
            
        Returns:
            correct_h: (B, 1, H, W) binary mask where high-freq is correct
            correct_l: (B, 1, H, W) binary mask where low-freq is correct
        """
        pred_h = (p_h > threshold).float()
        pred_l = (p_l > threshold).float()
        target_binary = (target > threshold).float()
        
        correct_h = (pred_h == target_binary).float()
        correct_l = (pred_l == target_binary).float()
        
        return correct_h, correct_l


class EvidenceFusionV2Module(nn.Module):
    """Drop-in replacement for original EvidenceFusion with backward compatibility."""
    
    def __init__(self, cfg: EvidenceFusionV2Config) -> None:
        super().__init__()
        self.fusion = EvidenceFusionV2(cfg)
        self.cfg = cfg
    
    def forward(
        self,
        alpha_h: torch.Tensor,
        e_h: torch.Tensor,
        alpha_l: torch.Tensor,
        e_l: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maintains original interface (3 return values)."""
        alpha_f, p_f, u_f, _ = self.fusion(alpha_h, e_h, alpha_l, e_l, return_weights=False)
        return alpha_f, p_f, u_f


def verify_stop_gradient(fusion: EvidenceFusionV2) -> bool:
    """Unit test helper: verify that gradients don't flow through fusion weights.
    
    Usage in tests:
        assert verify_stop_gradient(fusion_module)
    """
    # Create dummy inputs with gradients
    alpha_h = torch.randn(2, 2, 8, 8, requires_grad=True)
    e_h = alpha_h - 1
    alpha_l = torch.randn(2, 2, 8, 8, requires_grad=True)
    e_l = alpha_l - 1
    
    # Forward pass
    alpha_f, p_f, u_f, weights = fusion(alpha_h, e_h, alpha_l, e_l, return_weights=True)
    pi_h, pi_l = weights
    
    if fusion.cfg.detach_weights:
        # Verify weights have no gradient function
        assert pi_h.grad_fn is None, "π_H should have no grad_fn when detach_weights=True"
        assert pi_l.grad_fn is None, "π_L should have no grad_fn when detach_weights=True"
        return True
    
    return False  # Not testing when detach is disabled
