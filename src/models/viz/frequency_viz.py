# path: src/models/viz/frequency_viz.py
"""
Visualization utilities for FFT frequency split and diagnostics.

Provides functions to:
- Visualize I_low, I_high, and frequency masks
- Plot energy curves E(r)/E_tot vs radius
- Show gradient magnitude maps
- Export diagnostic plots

References:
  - Goal A3 in task spec: visualization utilities
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid issues in headless environments
_plt = None
def get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy for plotting."""
    if t.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C)
        t = t.permute(0, 2, 3, 1)
    return t.detach().cpu().numpy()


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] for display."""
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 1e-6:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img


def compute_gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude using Sobel operator.
    
    Args:
        img: (B, C, H, W) input tensor
        
    Returns:
        grad_mag: (B, 1, H, W) gradient magnitude
    """
    import torch.nn.functional as F
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    
    # Convert to grayscale
    if img.size(1) > 1:
        img_gray = img.mean(dim=1, keepdim=True)
    else:
        img_gray = img
    
    grad_x = F.conv2d(img_gray, sobel_x, padding=1)
    grad_y = F.conv2d(img_gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    return grad_mag


def plot_fft_split_single(
    original: np.ndarray,
    i_low: np.ndarray,
    i_high: np.ndarray,
    mask_low: Optional[np.ndarray] = None,
    d0_radius: Optional[float] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot FFT split visualization for a single image.
    
    Args:
        original: (H, W, C) original image
        i_low: (H, W, C) low-frequency component
        i_high: (H, W, C) high-frequency component
        mask_low: (H, W) frequency domain low-pass mask (optional)
        d0_radius: Selected cutoff radius (optional)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure
    """
    plt = get_plt()
    
    n_cols = 3 if mask_low is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Original
    axes[0].imshow(normalize_for_display(original))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # I_low
    axes[1].imshow(normalize_for_display(i_low))
    axes[1].set_title("I_low (Low-Freq)")
    axes[1].axis('off')
    
    # I_high
    axes[2].imshow(normalize_for_display(i_high))
    axes[2].set_title("I_high (High-Freq)")
    axes[2].axis('off')
    
    # Frequency mask
    if mask_low is not None and n_cols > 3:
        axes[3].imshow(mask_low, cmap='gray')
        if d0_radius is not None:
            axes[3].set_title(f"Mask (D0={d0_radius:.3f})")
        else:
            axes[3].set_title("Freq Mask")
        axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved FFT split plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fft_split_batch(
    originals: torch.Tensor,
    i_lows: torch.Tensor,
    i_highs: torch.Tensor,
    d0_radii: Optional[torch.Tensor] = None,
    max_images: int = 4,
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot FFT split visualization for a batch of images.
    
    Args:
        originals: (B, C, H, W) original images
        i_lows: (B, C, H, W) low-frequency components
        i_highs: (B, C, H, W) high-frequency components
        d0_radii: (B,) cutoff radii per image
        max_images: Maximum number of images to plot
        save_dir: Directory to save figures
        show: Whether to display figures
    """
    b = min(originals.size(0), max_images)
    
    originals_np = tensor_to_numpy(originals[:b])
    i_lows_np = tensor_to_numpy(i_lows[:b])
    i_highs_np = tensor_to_numpy(i_highs[:b])
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(b):
        d0 = d0_radii[i].item() if d0_radii is not None else None
        save_path = f"{save_dir}/fft_split_{i}.png" if save_dir else None
        
        plot_fft_split_single(
            originals_np[i],
            i_lows_np[i],
            i_highs_np[i],
            d0_radius=d0,
            title=f"Image {i}",
            save_path=save_path,
            show=show
        )


def plot_energy_curve(
    radii: np.ndarray,
    cum_energy: np.ndarray,
    d0: float,
    p_threshold: float = 0.95,
    title: str = "Cumulative Energy vs Radius",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot cumulative energy curve E(r)/E_tot vs radius.
    
    Args:
        radii: (N,) radial bin centers
        cum_energy: (N,) cumulative energy ratio at each radius
        d0: Selected cutoff radius
        p_threshold: Energy percentile threshold
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    """
    plt = get_plt()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(radii, cum_energy, 'b-', linewidth=2, label='E(r)/E_tot')
    ax.axhline(y=p_threshold, color='r', linestyle='--', label=f'p = {p_threshold}')
    ax.axvline(x=d0, color='g', linestyle='--', label=f'D0 = {d0:.3f}')
    
    ax.set_xlabel('Normalized Radius', fontsize=12)
    ax.set_ylabel('Cumulative Energy Ratio', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, radii.max()])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved energy curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gradient_comparison(
    original: torch.Tensor,
    i_high: torch.Tensor,
    title: str = "Gradient Magnitude Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Compare gradient magnitudes of original and I_high.
    
    Args:
        original: (B, C, H, W) original images
        i_high: (B, C, H, W) high-frequency components
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    """
    plt = get_plt()
    
    grad_orig = compute_gradient_magnitude(original)
    grad_high = compute_gradient_magnitude(i_high)
    
    b = original.size(0)
    fig, axes = plt.subplots(b, 4, figsize=(16, 4 * b))
    
    if b == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(b):
        # Original
        axes[i, 0].imshow(normalize_for_display(tensor_to_numpy(original[i:i+1])[0]))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # I_high
        axes[i, 1].imshow(normalize_for_display(tensor_to_numpy(i_high[i:i+1])[0]))
        axes[i, 1].set_title("I_high")
        axes[i, 1].axis('off')
        
        # Gradient of original
        axes[i, 2].imshow(tensor_to_numpy(grad_orig[i:i+1])[0, :, :, 0], cmap='hot')
        axes[i, 2].set_title("∇Original")
        axes[i, 2].axis('off')
        
        # Gradient of I_high
        axes[i, 3].imshow(tensor_to_numpy(grad_high[i:i+1])[0, :, :, 0], cmap='hot')
        axes[i, 3].set_title("∇I_high")
        axes[i, 3].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved gradient comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fusion_weights(
    pi_h: torch.Tensor,
    pi_l: torch.Tensor,
    original: Optional[torch.Tensor] = None,
    prediction: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    max_images: int = 4,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Visualize fusion weights (π_H, π_L) as heatmaps.
    
    Args:
        pi_h: (B, 1, H, W) high-freq branch weights
        pi_l: (B, 1, H, W) low-freq branch weights
        original: (B, C, H, W) original images (optional)
        prediction: (B, 1, H, W) fused prediction (optional)
        target: (B, 1, H, W) ground truth (optional)
        max_images: Maximum images to plot
        save_path: Path to save figure
        show: Whether to display
    """
    plt = get_plt()
    
    b = min(pi_h.size(0), max_images)
    n_cols = 2  # pi_h, pi_l
    if original is not None:
        n_cols += 1
    if prediction is not None:
        n_cols += 1
    if target is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(b, n_cols, figsize=(3 * n_cols, 3 * b))
    
    if b == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(b):
        col = 0
        
        if original is not None:
            axes[i, col].imshow(normalize_for_display(tensor_to_numpy(original[i:i+1])[0]))
            axes[i, col].set_title("Original")
            axes[i, col].axis('off')
            col += 1
        
        # π_H heatmap
        im_h = axes[i, col].imshow(
            tensor_to_numpy(pi_h[i:i+1])[0, :, :, 0],
            cmap='RdYlBu_r', vmin=0, vmax=1
        )
        axes[i, col].set_title(f"π_H (mean={pi_h[i].mean():.3f})")
        axes[i, col].axis('off')
        plt.colorbar(im_h, ax=axes[i, col], fraction=0.046)
        col += 1
        
        # π_L heatmap
        im_l = axes[i, col].imshow(
            tensor_to_numpy(pi_l[i:i+1])[0, :, :, 0],
            cmap='RdYlBu_r', vmin=0, vmax=1
        )
        axes[i, col].set_title(f"π_L (mean={pi_l[i].mean():.3f})")
        axes[i, col].axis('off')
        plt.colorbar(im_l, ax=axes[i, col], fraction=0.046)
        col += 1
        
        if prediction is not None:
            axes[i, col].imshow(
                tensor_to_numpy(prediction[i:i+1])[0, :, :, 0],
                cmap='Blues', vmin=0, vmax=1
            )
            axes[i, col].set_title("Prediction")
            axes[i, col].axis('off')
            col += 1
        
        if target is not None:
            axes[i, col].imshow(
                tensor_to_numpy(target[i:i+1])[0, :, :, 0],
                cmap='Greens', vmin=0, vmax=1
            )
            axes[i, col].set_title("Ground Truth")
            axes[i, col].axis('off')
    
    plt.suptitle("Evidence Fusion Weights", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved fusion weights plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_uncertainty_heatmap(
    uncertainty: torch.Tensor,
    original: Optional[torch.Tensor] = None,
    prediction: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    error_mask: Optional[torch.Tensor] = None,
    max_images: int = 4,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """Visualize uncertainty heatmaps with optional error mask overlay.
    
    Args:
        uncertainty: (B, 1, H, W) per-pixel uncertainty
        original: (B, C, H, W) original images
        prediction: (B, 1, H, W) predictions
        target: (B, 1, H, W) ground truth
        error_mask: (B, 1, H, W) binary error mask (optional)
        max_images: Maximum images to plot
        save_path: Path to save figure
        show: Whether to display
    """
    plt = get_plt()
    
    b = min(uncertainty.size(0), max_images)
    n_cols = 2
    if original is not None:
        n_cols += 1
    if error_mask is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(b, n_cols, figsize=(4 * n_cols, 4 * b))
    
    if b == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(b):
        col = 0
        
        if original is not None:
            axes[i, col].imshow(normalize_for_display(tensor_to_numpy(original[i:i+1])[0]))
            axes[i, col].set_title("Original")
            axes[i, col].axis('off')
            col += 1
        
        # Uncertainty heatmap
        u_np = tensor_to_numpy(uncertainty[i:i+1])[0, :, :, 0]
        im = axes[i, col].imshow(u_np, cmap='plasma')
        axes[i, col].set_title(f"Uncertainty (max={u_np.max():.3f})")
        axes[i, col].axis('off')
        plt.colorbar(im, ax=axes[i, col], fraction=0.046)
        col += 1
        
        # Prediction vs GT overlay
        if prediction is not None and target is not None:
            pred_np = tensor_to_numpy(prediction[i:i+1])[0, :, :, 0]
            tgt_np = tensor_to_numpy(target[i:i+1])[0, :, :, 0]
            
            # Create RGB overlay: green=correct, red=error
            overlay = np.zeros((*pred_np.shape, 3))
            pred_bin = pred_np > 0.5
            tgt_bin = tgt_np > 0.5
            overlay[..., 1] = (pred_bin == tgt_bin).astype(float) * 0.5  # Green = correct
            overlay[..., 0] = (pred_bin != tgt_bin).astype(float)  # Red = error
            overlay[..., 2] = tgt_bin.astype(float) * 0.3  # Blue = GT region
            
            axes[i, col].imshow(overlay)
            axes[i, col].set_title("Pred vs GT")
            axes[i, col].axis('off')
            col += 1
        
        if error_mask is not None:
            axes[i, col].imshow(
                tensor_to_numpy(error_mask[i:i+1])[0, :, :, 0],
                cmap='Reds', vmin=0, vmax=1
            )
            axes[i, col].set_title("Error Mask")
            axes[i, col].axis('off')
    
    plt.suptitle("Uncertainty Visualization", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
