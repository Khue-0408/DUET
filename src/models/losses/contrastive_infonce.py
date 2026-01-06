# path: src/models/losses/contrastive_infonce.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class InfoNCEConfig:
    tau: float = 0.2
    pairs_per_image: int = 64


def sampled_pixel_infonce(
    z: torch.Tensor,
    region_ids: torch.Tensor,
    cfg: InfoNCEConfig,
) -> torch.Tensor:
    """
    Pixel-wise contrastive InfoNCE (Eq. (1), Sec.3.1).

    Paper says: "sampled 64 positive/negative pairs per image per iteration" (Training Details).
    Implementation (ASSUMPTION consistent with that statement):
    - For each image: sample `pairs_per_image` anchors from valid pixels.
    - For each anchor: sample 1 positive pixel from same region and 1 negative from different region.
    - InfoNCE with 1 pos / 1 neg: -log exp(sim(a,p)/tau) / (exp(sim(a,p)/tau)+exp(sim(a,n)/tau)).

    Inputs:
    - z: (B,C,h,w) normalized embeddings
    - region_ids: (B,1,h,w) values in {0:bg,1:core,2:boundary} with -1 for invalid
    """
    b, c, h, w = z.shape
    tau = float(cfg.tau)
    losses = []

    z_flat = z.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B,HW,C)
    rid_flat = region_ids.reshape(b, h * w)              # (B,HW)

    for i in range(b):
        rid = rid_flat[i]
        valid = rid >= 0
        if valid.sum() < 10:
            continue
        idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
        n_pairs = min(int(cfg.pairs_per_image), idx_valid.numel())
        anchors = idx_valid[torch.randperm(idx_valid.numel(), device=z.device)[:n_pairs]]

        zi = z_flat[i]  # (HW,C)
        li = []
        for a_idx in anchors:
            a_r = int(rid[a_idx].item())
            pos_pool = torch.nonzero(rid == a_r, as_tuple=False).squeeze(1)
            neg_pool = torch.nonzero((rid != a_r) & valid, as_tuple=False).squeeze(1)
            if pos_pool.numel() < 2 or neg_pool.numel() < 1:
                continue

            p_idx = pos_pool[torch.randint(0, pos_pool.numel(), (1,), device=z.device)].item()
            n_idx = neg_pool[torch.randint(0, neg_pool.numel(), (1,), device=z.device)].item()

            a = zi[a_idx]
            p = zi[p_idx]
            n = zi[n_idx]

            sp = F.cosine_similarity(a[None, :], p[None, :], dim=1) / tau
            sn = F.cosine_similarity(a[None, :], n[None, :], dim=1) / tau
            loss = -torch.log(torch.exp(sp) / (torch.exp(sp) + torch.exp(sn) + 1e-12) + 1e-12)
            li.append(loss)

        if len(li) > 0:
            losses.append(torch.stack(li).mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return torch.stack(losses).mean()
