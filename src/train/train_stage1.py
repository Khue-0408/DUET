# path: src/train/train_stage1.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets import PolypMetaDataset
from src.data.transforms import TransformConfig, build_transforms
from src.models.encoders.efficientnet_b4 import EfficientNetB4Backbone, EfficientNetB4Config
from src.models.domain.grl import GradientReversalLayer
from src.models.domain.domain_discriminator import DomainDiscriminator, DomainDiscriminatorConfig
from src.models.losses.contrastive_infonce import InfoNCEConfig, sampled_pixel_infonce
from src.models.losses.dice import dice_loss
from src.models.losses.region_weighting import RegionWeightConfig, make_region_weights
from src.utils.config import apply_overrides, load_yaml
from src.utils.distributed import init_distributed, is_main_process
from src.utils.seed import seed_all
from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointState, save_checkpoint, load_checkpoint


class Stage1Wrapper(nn.Module):
    """
    Stage1 model:
    - Encoder Eθ (EfficientNet-B4) (Sec.3.1)
    - Projection head for pixel embeddings (common contrastive practice; consistent with Fig.1 "Region Contrast Head")
    - Optional aux seg head to predict pseudo-mask with Dice (Eq. (2))
    - Domain discriminator with GRL (Eq. (3))
    """

    def __init__(self, enc: EfficientNetB4Backbone, proj_dim: int, aux_seg: bool, num_domains: int, grl_lambda: float, mlp_hidden: int) -> None:
        super().__init__()
        self.enc = enc
        # projection: 1x1 conv on chosen feature level
        self.proj = nn.Conv2d(0, 0, 1)  # placeholder set later via build
        self.proj_dim = proj_dim
        self.aux_seg = aux_seg
        self.aux_head = None
        self.grl = GradientReversalLayer(grl_lambda)
        self.domain_head = None
        self.num_domains = num_domains
        self.mlp_hidden = mlp_hidden

    def build_heads(self, feat_channels: int) -> None:
        self.proj = nn.Conv2d(feat_channels, self.proj_dim, kernel_size=1)
        if self.aux_seg:
            self.aux_head = nn.Sequential(
                nn.Conv2d(feat_channels, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
            )
        self.domain_head = DomainDiscriminator(DomainDiscriminatorConfig(in_dim=feat_channels, hidden_dim=self.mlp_hidden, num_domains=self.num_domains))

    def forward(self, x: torch.Tensor, feature_level: int) -> Dict[str, torch.Tensor]:
        feats = self.enc(x)  # list
        f = feats[feature_level]
        z = F.normalize(self.proj(f), dim=1)

        out: Dict[str, torch.Tensor] = {"feat": f, "z": z}

        if self.aux_head is not None:
            logits = self.aux_head(f)
            out["aux_logits"] = logits

        # domain logits from GAP
        gap = f.mean(dim=(2, 3))
        dom_logits = self.domain_head(self.grl(gap))
        out["dom_logits"] = dom_logits
        return out


def _downsample_mask(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(mask, size=size_hw, mode="nearest")


def make_stage1_regions_from_pseudomask(pm: torch.Tensor, re: int, rd: int) -> torch.Tensor:
    """
    Sec.3.1 pseudomask approach (p.9): core (erode), boundary (band), bg (outside).
    Returns region_id in {0:bg,1:core,2:boundary} with shape (B,1,H,W).
    """
    cfg = RegionWeightConfig(wB=3.0, wC=1.0, wBG=1.0, re=re, rd=rd)
    _, region_id = make_region_weights(pm, cfg)
    return region_id


def make_stage1_regions_weak(img: torch.Tensor) -> torch.Tensor:
    """
    Sec.3.1 weak/self-supervised approach (p.9).
    Baseline heuristic:
    - use simple edges (Sobel magnitude) to mark boundary-like pixels
    - use thresholded low-gradient as background
    - define "core" as non-edge foreground guess (very rough)

    TODO: replace with better heuristic (Canny + contours + fill) if needed.
    """
    # img: (B,3,H,W)
    gray = img.mean(dim=1, keepdim=True)
    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
    mag = torch.sqrt(gx * gx + gy * gy)

    thr_edge = mag.mean(dim=(2, 3), keepdim=True) + 1.0 * mag.std(dim=(2, 3), keepdim=True)
    boundary = (mag > thr_edge).float()

    thr_bg = mag.mean(dim=(2, 3), keepdim=True) - 0.5 * mag.std(dim=(2, 3), keepdim=True)
    bg = (mag < thr_bg).float()

    core = (1.0 - boundary) * (1.0 - bg)

    region_id = torch.zeros_like(gray)
    region_id = torch.where(core > 0.5, torch.ones_like(region_id), region_id)
    region_id = torch.where(boundary > 0.5, torch.full_like(region_id, 2.0), region_id)
    return region_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", type=str, nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    dist = init_distributed()
    seed_all(int(cfg["exp"]["seed"]), bool(cfg["exp"]["deterministic"]))

    logger = Logger(cfg["logging"]["out_dir"], cfg["exp"]["name"], bool(cfg["logging"]["tensorboard"])) if is_main_process(dist) else None

    # data
    tfm = build_transforms(TransformConfig(image_size=int(cfg["data"]["image_size"]), augment=True), for_stage1=True)
    train_ds = PolypMetaDataset(
        root=cfg["data"]["root"],
        meta_csv=cfg["data"]["meta_csv"],
        split_file=f'{cfg["data"]["split_dir"]}/{cfg["data"]["train_split"]}',
        transform=tfm,
        pseudo_mask_key=cfg["stage1"]["region_gen"]["pseudo_mask_key"],
    )
    val_ds = PolypMetaDataset(
        root=cfg["data"]["root"],
        meta_csv=cfg["data"]["meta_csv"],
        split_file=f'{cfg["data"]["split_dir"]}/{cfg["data"]["val_split"]}',
        transform=tfm,
        pseudo_mask_key=cfg["stage1"]["region_gen"]["pseudo_mask_key"],
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_ddp else None

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                            sampler=val_sampler, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    # model
    enc_cfg = EfficientNetB4Config(
        name=cfg["model"]["encoders"]["eh"]["name"] if "model" in cfg and "encoders" in cfg.get("model", {}) else "tf_efficientnet_b4",
        pretrained=True,
        out_indices=tuple(cfg.get("model", {}).get("encoders", {}).get("eh", {}).get("out_indices", [1, 2, 3, 4])),
    )
    enc = EfficientNetB4Backbone(enc_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = enc.to(device)

    # infer feature channel at desired level
    with torch.no_grad():
        dummy = torch.zeros(1, 3, int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"]), device=device)
        feats = enc(dummy)
        feat_level = int(cfg["stage1"]["contrastive"]["feature_level"])
        feat_ch = feats[feat_level].shape[1]

    aux_enabled = bool(cfg["stage1"]["aux_seg"]["enabled"])
    wrapper = Stage1Wrapper(
        enc=enc,
        proj_dim=int(cfg["stage1"]["contrastive"]["proj_dim"]),
        aux_seg=aux_enabled,
        num_domains=int(pd_read_num_domains(cfg["data"]["root"], cfg["data"]["meta_csv"])),
        grl_lambda=float(cfg["stage1"]["adv"]["grl_lambda"]),
        mlp_hidden=int(cfg["stage1"]["adv"]["mlp_hidden"]),
    )
    wrapper.build_heads(feat_ch)
    wrapper = wrapper.to(device)

    if dist.is_ddp:
        wrapper = torch.nn.parallel.DistributedDataParallel(wrapper, device_ids=[dist.local_rank], find_unused_parameters=True)

    optim = torch.optim.Adam(wrapper.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    start_epoch, step, best = 0, 0, 1e9
    if cfg["train"]["resume"]:
        st = load_checkpoint(cfg["train"]["resume"], wrapper, optim)
        start_epoch, step, best = st.epoch, st.step, st.best_metric

    contra_cfg = InfoNCEConfig(tau=float(cfg["stage1"]["contrastive"]["tau"]), pairs_per_image=int(cfg["stage1"]["contrastive"]["pairs_per_image"]))
    lambda_seg = float(cfg["stage1"]["aux_seg"]["lambda_seg"])
    lambda_adv = float(cfg["stage1"]["adv"]["lambda_adv"])
    adv_enabled = bool(cfg["stage1"]["adv"]["enabled"])

    region_mode = cfg["stage1"]["region_gen"]["mode"]
    re = int(cfg["stage1"]["region_gen"]["re"])
    rd = int(cfg["stage1"]["region_gen"]["rd"])

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        wrapper.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            pm = batch["pseudo_mask"]
            dom = torch.tensor(batch["domain_id"], device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                out = wrapper(img, feature_level=feat_level)

                z = out["z"]
                # build region ids at feature resolution
                if region_mode == "pseudomask" and pm is not None:
                    pm_t = pm.to(device).float()
                    rid = make_stage1_regions_from_pseudomask(pm_t, re=re, rd=rd)
                else:
                    rid = make_stage1_regions_weak(img)

                rid_ds = _downsample_mask(rid, z.shape[-2:]).long()
                rid_ds = torch.where(rid_ds < 0, torch.full_like(rid_ds, -1), rid_ds)

                l_contra = sampled_pixel_infonce(z, rid_ds, contra_cfg)

                l_seg = torch.tensor(0.0, device=device)
                if aux_enabled and ("aux_logits" in out) and (pm is not None):
                    aux = out["aux_logits"]
                    aux_up = torch.nn.functional.interpolate(aux, size=pm_t.shape[-2:], mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(aux_up)
                    l_seg = dice_loss(prob, pm_t)

                l_adv = torch.tensor(0.0, device=device)
                if adv_enabled:
                    l_adv = torch.nn.functional.cross_entropy(out["dom_logits"], dom)

                loss = l_contra + lambda_seg * l_seg + lambda_adv * l_adv

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if logger is not None and step % int(cfg["train"]["log_every"]) == 0:
                logger.log_scalars(step, {
                    "stage1/loss": float(loss.item()),
                    "stage1/l_contra": float(l_contra.item()),
                    "stage1/l_seg": float(l_seg.item()),
                    "stage1/l_adv": float(l_adv.item()),
                })
            step += 1

        # val: track avg loss (proxy)
        wrapper.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                pm = batch["pseudo_mask"]
                dom = torch.tensor(batch["domain_id"], device=device, dtype=torch.long)

                out = wrapper(img, feature_level=feat_level)
                z = out["z"]

                if region_mode == "pseudomask" and pm is not None:
                    pm_t = pm.to(device).float()
                    rid = make_stage1_regions_from_pseudomask(pm_t, re=re, rd=rd)
                else:
                    rid = make_stage1_regions_weak(img)

                rid_ds = _downsample_mask(rid, z.shape[-2:]).long()

                l_contra = sampled_pixel_infonce(z, rid_ds, contra_cfg)
                l_seg = torch.tensor(0.0, device=device)
                if aux_enabled and ("aux_logits" in out) and (pm is not None):
                    aux = out["aux_logits"]
                    aux_up = torch.nn.functional.interpolate(aux, size=pm_t.shape[-2:], mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(aux_up)
                    l_seg = dice_loss(prob, pm_t)
                l_adv = torch.tensor(0.0, device=device)
                if adv_enabled:
                    l_adv = torch.nn.functional.cross_entropy(out["dom_logits"], dom)

                loss = l_contra + lambda_seg * l_seg + lambda_adv * l_adv
                val_loss += float(loss.item())
                n += 1

        val_loss /= max(n, 1)
        if logger is not None:
            logger.log_scalars(step, {"stage1/val_loss": val_loss})

            # save best
            if val_loss < best:
                best = val_loss
                save_checkpoint(str(logger.ckpt_dir / "stage1_encoder_best.pt"), unwrap_ddp(wrapper).enc.backbone, optim, CheckpointState(epoch, step, best))
            save_checkpoint(str(logger.ckpt_dir / "stage1_last.pt"), unwrap_ddp(wrapper).enc.backbone, optim, CheckpointState(epoch, step, best))

    if logger is not None:
        logger.close()


def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def pd_read_num_domains(root: str, meta_csv: str) -> int:
    import pandas as pd
    from pathlib import Path
    meta = pd.read_csv(Path(root) / meta_csv)
    if "domain_id" not in meta.columns:
        return 1
    return int(meta["domain_id"].nunique())


if __name__ == "__main__":
    main()
