# path: src/models/freq/fft_split_adaptive.py
"""
Stage 2 Frequency Split with Adaptive Energy Percentile Cutoff.

Changes from fixed-radius FFT split:
  A1) Adaptive cutoff D0: smallest radius such that E(D0)/E_tot >= p (default p=0.95)
  A2) Optional edge-alignment auto-check/calibration for I_high

References:
  - DUET Paper Section 3.2: frequency decomposition
  - Goal A in task spec: per-image adaptive cutoff based on energy percentile
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class FFTSplitAdaptiveConfig:
    """Configuration for adaptive FFT frequency split.
    
    Attributes:
        p_default: Default energy percentile threshold (E(D0)/E_tot >= p).
        p_min: Minimum percentile for auto-adjustment.
        p_step: Step size for auto-adjustment (p_default -> p_default - p_step -> ...).
        auto_edge_adjust: Enable edge-alignment auto-calibration for I_high.
        edge_metric_threshold: Gradient-energy ratio threshold to trigger adjustment.
        auto_adjust_max_steps: Maximum number of adjustment steps.
        per_channel: Compute FFT per-channel (True) or on luminance (False).
        log_cutoff: Log chosen D0 per image (for diagnostics).
        high_boost_factor: Mild high-boost applied to I_high if auto-adjust triggered.
    """
    p_default: float = 0.95
    p_min: float = 0.85
    p_step: float = 0.02
    auto_edge_adjust: bool = False
    edge_metric_threshold: float = 0.3  # Gradient-energy ratio below this triggers adjust
    auto_adjust_max_steps: int = 5
    per_channel: bool = True
    log_cutoff: bool = False
    high_boost_factor: float = 1.2  # Mild boost for I_high if edge emphasis weak


@dataclass
class FFTSplitOutput:
    """Output from adaptive FFT split.
    
    Attributes:
        i_high: High-frequency component (B, C, H, W).
        i_low: Low-frequency component (B, C, H, W).
        d0_radii: Selected cutoff radii per image (B,).
        mask_low: Low-pass frequency mask (B, 1, H, W).
        mask_high: High-pass frequency mask (B, 1, H, W).
        edge_metrics: Edge alignment metrics per image (B,) if computed.
    """
    i_high: torch.Tensor
    i_low: torch.Tensor
    d0_radii: torch.Tensor
    mask_low: torch.Tensor
    mask_high: torch.Tensor
    edge_metrics: Optional[torch.Tensor] = None


class FFTSplitAdaptive(nn.Module):
    """
    FFT-based frequency decomposition with adaptive energy-percentile cutoff.
    
    For each image:
    1. Compute FFT power spectrum P(u,v) = |F(u,v)|^2
    2. Compute cumulative energy inside radius r: E(r) = sum_{D(u,v)<=r} P(u,v)
    3. Choose smallest D0 such that E(D0)/E_tot >= p
    4. Construct masks: M_low = 1[D <= D0], M_high = 1 - M_low
    5. Get I_low = IFFT(F * M_low), I_high = IFFT(F * M_high)
    
    Optional: edge-alignment auto-check and calibration.
    """

    def __init__(self, cfg: FFTSplitAdaptiveConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Register buffers for radial distance grids (will be built on first forward)
        self._cached_h: int = 0
        self._cached_w: int = 0
        
    def _build_radial_grid(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Build normalized radial distance grid centered at (h//2, w//2).
        
        Returns:
            rr: (H, W) tensor with normalized distances in [0, sqrt(2)]
        """
        # Create coordinate grids centered at origin
        cy, cx = h // 2, w // 2
        yy = torch.arange(h, device=device, dtype=torch.float32) - cy
        xx = torch.arange(w, device=device, dtype=torch.float32) - cx
        
        # Normalize by max possible distance
        max_dist = ((h // 2) ** 2 + (w // 2) ** 2) ** 0.5
        yy = yy / (max_dist + 1e-6)
        xx = xx / (max_dist + 1e-6)
        
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        rr = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        return rr
    
    def _compute_cumulative_energy(
        self, 
        power_spectrum: torch.Tensor, 
        radial_dist: torch.Tensor,
        n_bins: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cumulative energy as function of radius.
        
        Efficient binning approach for fast D0 selection.
        
        Args:
            power_spectrum: (H, W) power spectrum |F(u,v)|^2
            radial_dist: (H, W) normalized radial distance grid
            n_bins: Number of radial bins for approximation
            
        Returns:
            radii: (n_bins,) bin centers
            cum_energy_ratio: (n_bins,) cumulative energy ratio E(r)/E_tot
        """
        device = power_spectrum.device
        r_max = radial_dist.max().item()
        
        # Create radial bins
        bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
        radii = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Digitize distances into bins
        flat_r = radial_dist.flatten()
        flat_p = power_spectrum.flatten()
        
        # Compute energy per bin (vectorized)
        bin_energies = torch.zeros(n_bins, device=device)
        for i in range(n_bins):
            mask = (flat_r >= bin_edges[i]) & (flat_r < bin_edges[i + 1])
            bin_energies[i] = flat_p[mask].sum()
        
        # Cumulative sum
        total_energy = flat_p.sum().clamp_min(1e-12)
        cum_energy = torch.cumsum(bin_energies, dim=0)
        cum_energy_ratio = cum_energy / total_energy
        
        return radii, cum_energy_ratio
    
    def _select_d0_for_image(
        self,
        power_spectrum: torch.Tensor,
        radial_dist: torch.Tensor,
        p: float,
        n_bins: int = 100
    ) -> float:
        """Select D0 for a single image based on energy percentile.
        
        Args:
            power_spectrum: (H, W) power spectrum
            radial_dist: (H, W) radial distance grid
            p: Energy percentile threshold (0 < p < 1)
            
        Returns:
            D0: Selected cutoff radius (normalized)
        """
        radii, cum_energy = self._compute_cumulative_energy(power_spectrum, radial_dist, n_bins)
        
        # Find smallest radius where cumulative energy >= p
        mask = cum_energy >= p
        if mask.any():
            idx = mask.float().argmax()  # First True index
            return radii[idx].item()
        else:
            # Fallback: return max radius
            return radii[-1].item()
    
    def _create_frequency_mask(
        self,
        h: int, w: int,
        d0: float,
        device: torch.device,
        radial_dist: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create low-pass frequency mask for given D0.
        
        Args:
            h, w: Image dimensions
            d0: Cutoff radius (normalized)
            device: Target device
            radial_dist: Pre-computed radial distance grid (optional)
            
        Returns:
            mask_low: (1, 1, H, W) low-pass mask (1 inside D0, 0 outside)
        """
        if radial_dist is None:
            radial_dist = self._build_radial_grid(h, w, device)
        
        mask_low = (radial_dist <= d0).float()
        return mask_low.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def _compute_gradient_energy_ratio(
        self,
        i_high: torch.Tensor,
        i_orig: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient-energy ratio: ||∇I_high||_1 / ||∇I||_1.
        
        Used as edge-alignment metric.
        
        Args:
            i_high: (B, C, H, W) high-frequency component
            i_orig: (B, C, H, W) original image
            
        Returns:
            ratio: (B,) gradient energy ratio per image
        """
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=i_high.dtype, device=i_high.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=i_high.dtype, device=i_high.device).view(1, 1, 3, 3)
        
        b, c, h, w = i_high.shape
        
        # Convert to grayscale if needed
        if c > 1:
            i_high_gray = i_high.mean(dim=1, keepdim=True)
            i_orig_gray = i_orig.mean(dim=1, keepdim=True)
        else:
            i_high_gray = i_high
            i_orig_gray = i_orig
        
        # Compute gradients
        grad_high_x = F.conv2d(i_high_gray, sobel_x, padding=1)
        grad_high_y = F.conv2d(i_high_gray, sobel_y, padding=1)
        grad_high_mag = (grad_high_x.abs() + grad_high_y.abs())
        
        grad_orig_x = F.conv2d(i_orig_gray, sobel_x, padding=1)
        grad_orig_y = F.conv2d(i_orig_gray, sobel_y, padding=1)
        grad_orig_mag = (grad_orig_x.abs() + grad_orig_y.abs())
        
        # Compute L1 norms per image
        grad_high_norm = grad_high_mag.view(b, -1).sum(dim=1)
        grad_orig_norm = grad_orig_mag.view(b, -1).sum(dim=1).clamp_min(1e-12)
        
        return grad_high_norm / grad_orig_norm
    
    def _apply_high_boost(self, i_high: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply mild high-boost filter to enhance edges in I_high.
        
        Args:
            i_high: (B, C, H, W) high-frequency component
            factor: Boost factor (> 1 increases contrast)
            
        Returns:
            Boosted high-frequency component
        """
        return i_high * factor
    
    def forward(
        self,
        x: torch.Tensor,
        return_masks: bool = True
    ) -> FFTSplitOutput:
        """Perform adaptive FFT frequency split.
        
        Args:
            x: (B, C, H, W) input image batch
            return_masks: Whether to return frequency masks
            
        Returns:
            FFTSplitOutput with i_high, i_low, d0_radii, masks, and optional metrics
        """
        b, c, h, w = x.shape
        device = x.device
        
        # Build radial distance grid
        radial_dist = self._build_radial_grid(h, w, device)
        
        # FFT (per-image processing for adaptive cutoff)
        X = torch.fft.fft2(x, dim=(-2, -1))
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        
        # Compute power spectrum (average across channels if per_channel=False)
        if self.cfg.per_channel:
            # Use mean power across channels for D0 selection
            power = (Xs.abs() ** 2).mean(dim=1)  # (B, H, W)
        else:
            # Convert to luminance first
            if c == 3:
                lum_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
                x_lum = (x * lum_weights).sum(dim=1, keepdim=True)
            else:
                x_lum = x.mean(dim=1, keepdim=True)
            X_lum = torch.fft.fft2(x_lum, dim=(-2, -1))
            Xs_lum = torch.fft.fftshift(X_lum, dim=(-2, -1))
            power = (Xs_lum.abs() ** 2).squeeze(1)  # (B, H, W)
        
        # Select D0 per image with potential auto-adjustment
        d0_list = []
        p_used_list = []
        
        for i in range(b):
            p_current = self.cfg.p_default
            
            if self.cfg.auto_edge_adjust:
                # Iterative adjustment loop
                for step in range(self.cfg.auto_adjust_max_steps):
                    d0 = self._select_d0_for_image(power[i], radial_dist, p_current)
                    
                    # Build masks and compute I_high for this image
                    mask_low_i = self._create_frequency_mask(h, w, d0, device, radial_dist)
                    mask_high_i = 1.0 - mask_low_i
                    
                    X_high_i = Xs[i:i+1] * mask_high_i
                    i_high_i = torch.fft.ifft2(
                        torch.fft.ifftshift(X_high_i, dim=(-2, -1)), 
                        dim=(-2, -1)
                    ).real
                    
                    # Compute edge metric
                    ratio = self._compute_gradient_energy_ratio(i_high_i, x[i:i+1])
                    
                    if ratio.item() >= self.cfg.edge_metric_threshold:
                        # Edge emphasis sufficient
                        break
                    
                    # Adjust p downward
                    p_current = max(p_current - self.cfg.p_step, self.cfg.p_min)
                    
                    if p_current <= self.cfg.p_min:
                        break
                
                d0_list.append(d0)
                p_used_list.append(p_current)
            else:
                d0 = self._select_d0_for_image(power[i], radial_dist, self.cfg.p_default)
                d0_list.append(d0)
                p_used_list.append(self.cfg.p_default)
        
        # Log cutoff values if enabled
        if self.cfg.log_cutoff:
            for i, (d0, p) in enumerate(zip(d0_list, p_used_list)):
                logger.debug(f"Image {i}: D0={d0:.4f}, p_used={p:.3f}")
        
        # Create masks for each image and stack
        masks_low = []
        masks_high = []
        for d0 in d0_list:
            ml = self._create_frequency_mask(h, w, d0, device, radial_dist)
            masks_low.append(ml)
            masks_high.append(1.0 - ml)
        
        mask_low = torch.cat(masks_low, dim=0)  # (B, 1, H, W)
        mask_high = torch.cat(masks_high, dim=0)  # (B, 1, H, W)
        
        # Apply masks and inverse FFT
        X_low = Xs * mask_low
        X_high = Xs * mask_high
        
        i_low = torch.fft.ifft2(
            torch.fft.ifftshift(X_low, dim=(-2, -1)), 
            dim=(-2, -1)
        ).real
        i_high = torch.fft.ifft2(
            torch.fft.ifftshift(X_high, dim=(-2, -1)), 
            dim=(-2, -1)
        ).real
        
        # Apply high-boost if auto-adjust was triggered and edge emphasis still weak
        edge_metrics = None
        if self.cfg.auto_edge_adjust:
            edge_metrics = self._compute_gradient_energy_ratio(i_high, x)
            
            # Apply boost where needed
            needs_boost = edge_metrics < self.cfg.edge_metric_threshold
            if needs_boost.any():
                boost_mask = needs_boost.float().view(-1, 1, 1, 1)
                i_high = i_high * (1 - boost_mask) + \
                         self._apply_high_boost(i_high, self.cfg.high_boost_factor) * boost_mask
        
        # Normalize to preserve original scale
        # Option 1: Normalize to [0, 1] per-image
        def normalize_01(t: torch.Tensor) -> torch.Tensor:
            t_min = t.amin(dim=(-2, -1), keepdim=True)
            t_max = t.amax(dim=(-2, -1), keepdim=True)
            return (t - t_min) / (t_max - t_min + 1e-6)
        
        i_high_norm = normalize_01(i_high)
        i_low_norm = normalize_01(i_low)
        
        d0_radii = torch.tensor(d0_list, device=device, dtype=torch.float32)
        
        return FFTSplitOutput(
            i_high=i_high_norm,
            i_low=i_low_norm,
            d0_radii=d0_radii,
            mask_low=mask_low if return_masks else None,
            mask_high=mask_high if return_masks else None,
            edge_metrics=edge_metrics,
        )
    
    def get_diagnostic_info(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get detailed diagnostic info for visualization.
        
        Args:
            x: (B, C, H, W) input image batch
            
        Returns:
            Dictionary with power spectra, cumulative energy curves, etc.
        """
        b, c, h, w = x.shape
        device = x.device
        
        radial_dist = self._build_radial_grid(h, w, device)
        
        X = torch.fft.fft2(x, dim=(-2, -1))
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        power = (Xs.abs() ** 2).mean(dim=1)  # (B, H, W)
        
        # Compute cumulative energy curves for all images
        all_radii = []
        all_cum_energy = []
        all_d0 = []
        
        for i in range(b):
            radii, cum_energy = self._compute_cumulative_energy(power[i], radial_dist)
            all_radii.append(radii)
            all_cum_energy.append(cum_energy)
            d0 = self._select_d0_for_image(power[i], radial_dist, self.cfg.p_default)
            all_d0.append(d0)
        
        return {
            'power_spectrum': power,
            'radii': torch.stack(all_radii),
            'cumulative_energy': torch.stack(all_cum_energy),
            'd0_radii': torch.tensor(all_d0, device=device),
            'radial_dist': radial_dist,
        }


# Backward compatibility: wrapper that matches old interface
class FFTSplitAdaptiveWrapper(nn.Module):
    """Wrapper to match old FFTSplit interface (returns tuple instead of dataclass)."""
    
    def __init__(self, cfg: FFTSplitAdaptiveConfig) -> None:
        super().__init__()
        self.split = FFTSplitAdaptive(cfg)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.split(x, return_masks=False)
        return out.i_high, out.i_low
