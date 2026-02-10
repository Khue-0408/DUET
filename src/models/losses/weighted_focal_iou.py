# path: src/models/losses/weighted_focal_iou.py
"""
Weighted Focal Loss and Weighted IoU Loss with boundary-aware pixel weights.

Implements equations from task spec Goal C:
  - Boundary-aware weights: β_ij = |mean_{N_ij} g_mn - g_ij|
  - Weighted Focal Loss: L_w_focal = -Σ w_ij * q_ij^γ * log(q_ij) / Σ w_ij
  - Weighted IoU Loss: L_w_iou = 1 - (Σ w_ij * p_ij * g_ij) / (Σ w_ij * (p_ij + g_ij - p_ij * g_ij))

References:
  - Goal C in task spec: boundary-aware focal + IoU loss
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class WeightedLossConfig:
    """Configuration for weighted focal and IoU losses.
    
    Attributes:
        kernel_size: Local window size for boundary weight computation.
        lambda_boundary: Scaling factor for boundary weights (1 + λ * β).
        gamma_focal: Focal loss exponent (controls hard example mining).
        eps: Small constant for numerical stability.
        normalize_weights: Whether to normalize weights to sum to 1.
    """
    kernel_size: int = 5
    lambda_boundary: float = 5.0
    gamma_focal: float = 2.0
    eps: float = 1e-7
    normalize_weights: bool = True


def compute_boundary_weights(
    gt_mask: torch.Tensor,
    kernel_size: int = 5,
    lambda_boundary: float = 5.0
) -> torch.Tensor:
    """Compute boundary-aware pixel weights.
    
    Implements: β_ij = |mean_{(m,n)∈N_ij} g_mn - g_ij|
    Then: w_ij = 1 + λ * β_ij
    
    This efficiently uses average pooling to compute local mean.
    
    Args:
        gt_mask: (B, 1, H, W) ground truth binary mask in [0, 1]
        kernel_size: Size of local neighborhood window
        lambda_boundary: Scaling factor for boundary emphasis
        
    Returns:
        weights: (B, 1, H, W) pixel-wise weights
    """
    # Ensure mask is in [0, 1]
    gt = gt_mask.float().clamp(0, 1)
    
    # Compute local mean using average pooling
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(gt, kernel_size=kernel_size, stride=1, padding=padding)
    
    # Handle edge cases where pooling doesn't preserve size perfectly
    if local_mean.shape != gt.shape:
        local_mean = F.interpolate(local_mean, size=gt.shape[-2:], mode='bilinear', align_corners=False)
    
    # β_ij = |local_mean - g_ij|
    # This is high at boundaries where local mean differs from pixel value
    beta = (local_mean - gt).abs()
    
    # w_ij = 1 + λ * β_ij
    weights = 1.0 + lambda_boundary * beta
    
    return weights


def weighted_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-7,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Weighted Focal Loss.
    
    Implements:
        q_ij = p_ij if g_ij=1 else (1 - p_ij)
        L_w_focal = -Σ w_ij * q_ij^γ * log(q_ij) / Σ w_ij
    
    This is a modification of standard focal loss with boundary-aware weights.
    
    Args:
        pred: (B, 1, H, W) predicted probabilities in [0, 1]
        target: (B, 1, H, W) ground truth binary mask in {0, 1}
        weights: (B, 1, H, W) pixel-wise weights
        gamma: Focal exponent (higher = more focus on hard examples)
        eps: Numerical stability constant
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Weighted focal loss value
    """
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.float()
    
    # q_ij = p_ij if g_ij=1, else (1 - p_ij)
    # Equivalently: q = p * g + (1 - p) * (1 - g)
    q = pred * target + (1.0 - pred) * (1.0 - target)
    
    # Focal loss: -q^γ * log(q)
    focal_term = -(q.pow(gamma)) * torch.log(q + eps)
    
    # Apply weights
    weighted_loss = weights * focal_term
    
    if reduction == 'none':
        return weighted_loss
    
    # Normalize by sum of weights
    weight_sum = weights.sum().clamp_min(eps)
    
    if reduction == 'sum':
        return weighted_loss.sum() / weight_sum * weighted_loss.numel()
    else:  # 'mean'
        return weighted_loss.sum() / weight_sum


def weighted_iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-7,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Weighted IoU Loss (soft IoU with boundary-aware weights).
    
    Implements:
        L_w_iou = 1 - (Σ w_ij * p_ij * g_ij) / (Σ w_ij * (p_ij + g_ij - p_ij * g_ij))
    
    Args:
        pred: (B, 1, H, W) predicted probabilities in [0, 1]
        target: (B, 1, H, W) ground truth binary mask in {0, 1}
        weights: (B, 1, H, W) pixel-wise weights
        eps: Numerical stability constant
        reduction: 'mean' or 'none' (per-batch)
        
    Returns:
        Weighted IoU loss value (1 - IoU)
    """
    pred = pred.clamp(0, 1)
    target = target.float()
    
    # Weighted intersection: Σ w * p * g
    intersection = (weights * pred * target).sum(dim=(-2, -1))
    
    # Weighted union: Σ w * (p + g - p * g)
    union = (weights * (pred + target - pred * target)).sum(dim=(-2, -1))
    
    # IoU per sample
    iou = intersection / (union + eps)
    
    # Loss = 1 - IoU
    loss = 1.0 - iou
    
    if reduction == 'none':
        return loss  # (B, 1)
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'mean'
        return loss.mean()


class WeightedFocalIOULoss(nn.Module):
    """Combined weighted focal and IoU loss with boundary-aware weights.
    
    L_total_seg = 0.5 * (L_w_focal + L_w_iou)
    """
    
    def __init__(self, cfg: WeightedLossConfig) -> None:
        super().__init__()
        self.cfg = cfg
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            pred: (B, 1, H, W) predicted probabilities
            target: (B, 1, H, W) ground truth mask
            weights: (B, 1, H, W) pre-computed weights (optional)
            reduction: Loss reduction mode
            
        Returns:
            Tuple of (total_loss, focal_loss, iou_loss)
        """
        # Compute boundary weights if not provided
        if weights is None:
            weights = compute_boundary_weights(
                target,
                kernel_size=self.cfg.kernel_size,
                lambda_boundary=self.cfg.lambda_boundary
            )
        
        # Focal loss
        l_focal = weighted_focal_loss(
            pred, target, weights,
            gamma=self.cfg.gamma_focal,
            eps=self.cfg.eps,
            reduction=reduction
        )
        
        # IoU loss
        l_iou = weighted_iou_loss(
            pred, target, weights,
            eps=self.cfg.eps,
            reduction=reduction
        )
        
        # Combined: L_total_seg = 0.5 * (L_w_focal + L_w_iou)
        l_total = 0.5 * (l_focal + l_iou)
        
        return l_total, l_focal, l_iou
    
    def compute_boundary_weights(
        self,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary weights for target mask."""
        return compute_boundary_weights(
            target,
            kernel_size=self.cfg.kernel_size,
            lambda_boundary=self.cfg.lambda_boundary
        )


class CombinedSegmentationLoss(nn.Module):
    """Complete segmentation loss combining EDL, weighted focal/IoU, and Dice.
    
    L_stage3 = λ_edl * (R-EDL_H + R-EDL_L) + λ_seg * L_total_seg + λ_dice * L_dice
    
    This provides a unified interface for all loss terms in Stage 3.
    """
    
    def __init__(
        self,
        lambda_edl: float = 1.0,
        lambda_seg: float = 1.0,
        lambda_dice: float = 0.5,
        weighted_loss_cfg: Optional[WeightedLossConfig] = None
    ) -> None:
        super().__init__()
        self.lambda_edl = lambda_edl
        self.lambda_seg = lambda_seg
        self.lambda_dice = lambda_dice
        
        cfg = weighted_loss_cfg or WeightedLossConfig()
        self.weighted_loss = WeightedFocalIOULoss(cfg)
        
        # Log weights for transparency
        logger.info(
            f"CombinedSegmentationLoss initialized: "
            f"λ_edl={lambda_edl}, λ_seg={lambda_seg}, λ_dice={lambda_dice}"
        )
    
    def forward(
        self,
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        edl_loss_h: Optional[torch.Tensor] = None,
        edl_loss_l: Optional[torch.Tensor] = None,
        dice_loss_val: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            pred_prob: (B, 1, H, W) fused predicted probability
            target: (B, 1, H, W) ground truth mask
            edl_loss_h: Pre-computed EDL loss for high-freq branch
            edl_loss_l: Pre-computed EDL loss for low-freq branch
            dice_loss_val: Pre-computed Dice loss (optional)
            
        Returns:
            Tuple of (total_loss, loss_dict with individual terms)
        """
        # Weighted focal + IoU
        l_seg, l_focal, l_iou = self.weighted_loss(pred_prob, target)
        
        # Combine EDL losses
        l_edl = torch.tensor(0.0, device=pred_prob.device)
        if edl_loss_h is not None:
            l_edl = l_edl + edl_loss_h
        if edl_loss_l is not None:
            l_edl = l_edl + edl_loss_l
        
        # Dice loss
        l_dice = dice_loss_val if dice_loss_val is not None else torch.tensor(0.0, device=pred_prob.device)
        
        # Total loss
        total = (
            self.lambda_edl * l_edl +
            self.lambda_seg * l_seg +
            self.lambda_dice * l_dice
        )
        
        loss_dict = {
            'loss_total': total.item(),
            'loss_edl': l_edl.item() if torch.is_tensor(l_edl) else l_edl,
            'loss_seg': l_seg.item(),
            'loss_focal': l_focal.item(),
            'loss_iou': l_iou.item(),
            'loss_dice': l_dice.item() if torch.is_tensor(l_dice) else l_dice,
        }
        
        return total, loss_dict
