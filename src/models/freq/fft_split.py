# path: src/models/freq/fft_split.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class FFTSplitConfig:
    method: str = "radius_ratio"     # {radius_ratio, energy_percentile}
    radius_ratio: float = 0.10       # ASSUMPTION (Sec.3.2 cutoff unspecified)
    energy_percentile: float = 0.90  # alternative


class FFTSplit(nn.Module):
    """
    FFT-based frequency decomposition (Sec.3.2).
    Paper: create Ihigh and Ilow using frequency masks (p.11).
    Cutoff not specified -> provide options + default assumption.

    Implementation notes:
    - Uses torch.fft on each channel.
    - Builds low-pass mask (circle radius) and high-pass as complement.
    """

    def __init__(self, cfg: FFTSplitConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _radius_mask(self, h: int, w: int, radius: float, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        rr = torch.sqrt(xx * xx + yy * yy)
        return (rr <= radius).float()  # (H,W)

    def _energy_cutoff_radius(self, mag: torch.Tensor, percentile: float) -> float:
        # mag: (H,W) magnitude spectrum (already shifted)
        flat = mag.flatten()
        total = flat.sum().clamp_min(1e-12)
        sorted_vals, _ = torch.sort(flat, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=0) / total
        k = torch.searchsorted(cumsum, torch.tensor(percentile, device=mag.device))
        thresh = sorted_vals[k.clamp(max=sorted_vals.numel() - 1)]
        # Convert threshold into a radius by taking average radius of points above thresh
        h, w = mag.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=mag.device),
            torch.linspace(-1.0, 1.0, w, device=mag.device),
            indexing="ij",
        )
        rr = torch.sqrt(xx * xx + yy * yy)
        mask = mag >= thresh
        if mask.sum() < 10:
            return 0.10
        return rr[mask].mean().item()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        device = x.device

        # FFT
        X = torch.fft.fft2(x, dim=(-2, -1))
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        mag = torch.abs(Xs).mean(dim=1)  # (B,H,W)

        if self.cfg.method == "radius_ratio":
            radius = float(self.cfg.radius_ratio)
            low_mask = self._radius_mask(h, w, radius, device)[None, None, :, :]  # (1,1,H,W)
        elif self.cfg.method == "energy_percentile":
            # compute radius per-batch then average (simple)
            radii = []
            for i in range(b):
                radii.append(self._energy_cutoff_radius(mag[i], self.cfg.energy_percentile))
            radius = float(sum(radii) / max(len(radii), 1))
            low_mask = self._radius_mask(h, w, radius, device)[None, None, :, :]
        else:
            raise ValueError(f"Unknown FFT split method: {self.cfg.method}")

        high_mask = 1.0 - low_mask

        X_low = Xs * low_mask
        X_high = Xs * high_mask

        il = torch.fft.ifft2(torch.fft.ifftshift(X_low, dim=(-2, -1)), dim=(-2, -1)).real
        ih = torch.fft.ifft2(torch.fft.ifftshift(X_high, dim=(-2, -1)), dim=(-2, -1)).real

        # Normalize to [0,1] per-image to keep dynamic range stable (ASSUMPTION safe).
        def norm01(t: torch.Tensor) -> torch.Tensor:
            tmin = t.amin(dim=(-2, -1), keepdim=True)
            tmax = t.amax(dim=(-2, -1), keepdim=True)
            return (t - tmin) / (tmax - tmin + 1e-6)

        return norm01(ih), norm01(il)
