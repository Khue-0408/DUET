# path: src/models/duet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.freq.fft_split import FFTSplit
from src.models.encoders.efficientnet_b4 import EfficientNetB4Backbone
from src.models.encoders.vit_mae import ViTMAEBackbone
from src.models.decoders.fpn_decoder import FPNDecoder
from src.models.evidential.evidence import logits_to_dirichlet, dirichlet_mean_binary, beta_variance_binary
from src.models.fusion.evidence_fusion import EvidenceFusion


@dataclass
class DUETOutputs:
    # per-stream
    alpha_h: torch.Tensor  # (B,2,H,W)
    alpha_l: torch.Tensor  # (B,2,H,W)
    p_h: torch.Tensor      # (B,1,H,W) polyp prob
    p_l: torch.Tensor      # (B,1,H,W)
    # fused
    alpha_f: torch.Tensor  # (B,2,H,W)
    p_f: torch.Tensor      # (B,1,H,W)
    u_f: torch.Tensor      # (B,1,H,W)


class DUETModel(nn.Module):
    """
    DUET Stage2/3 forward:
    - FFT split: Ihigh, Ilow (Sec.3.2)
    - Dual encoders EH (EfficientNet-B4) and EL (ViT-Base/16, MAE init) (Sec.3.2)
    - Dual decoders (lightweight FPN) output logits oH, oL (Sec.3.3)
    - Evidential mapping + fusion Eq.(4)-(12) + uncertainty (end Sec.3.3)
    """

    def __init__(
        self,
        fft: FFTSplit,
        eh: EfficientNetB4Backbone,
        el: ViTMAEBackbone,
        dh: FPNDecoder,
        dl: FPNDecoder,
        fusion: EvidenceFusion,
    ) -> None:
        super().__init__()
        self.fft = fft
        self.eh = eh
        self.el = el
        self.dh = dh
        self.dl = dl
        self.fusion = fusion

    def forward(self, x: torch.Tensor) -> DUETOutputs:
        ih, il = self.fft(x)

        feats_h = self.eh(ih)  # list multi-scale
        feats_l = self.el(il)  # list multi-scale

        logits_h = self.dh(feats_h)  # (B,2,H,W)
        logits_l = self.dl(feats_l)

        alpha_h, e_h = logits_to_dirichlet(logits_h)  # Eq.(4)(5)
        alpha_l, e_l = logits_to_dirichlet(logits_l)

        p_h = dirichlet_mean_binary(alpha_h)          # Eq.(6)
        p_l = dirichlet_mean_binary(alpha_l)

        alpha_f, p_f, u_f = self.fusion(alpha_h, e_h, alpha_l, e_l)

        return DUETOutputs(
            alpha_h=alpha_h,
            alpha_l=alpha_l,
            p_h=p_h,
            p_l=p_l,
            alpha_f=alpha_f,
            p_f=p_f,
            u_f=u_f,
        )
