# path: src/train/train_duet.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets import PolypMetaDataset
from src.data.transforms import TransformConfig, build_transforms
from src.models.duet import DUETModel
from src.models.freq.fft_split import FFTSplit, FFTSplitConfig
from src.models.encoders.efficientnet_b4 import EfficientNetB4Backbone, EfficientNetB4Config
from src.models.encoders.vit_mae import ViTMAEBackbone, ViTMAEConfig, ViTMAEInit
from src.models.decoders.fpn_decoder import FPNDecoder, FPNDecoderConfig
from src.models.fusion.evidence_fusion import EvidenceFusion, EvidenceFusionConfig
from src.models.losses.edl import EDLConfig, edl_loss
from src.models.losses.region_weighting import RegionWeightConfig, make_region_weights
from src.models.losses.dice import dice_loss
from src.utils.config import apply_overrides, load_yaml
from src.utils.distributed import init_distributed, is_main_process
from src.utils.seed import seed_all
from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointState, save_checkpoint, load_checkpoint
from src.utils.metrics import dice_coeff, miou, precision_recall, expected_calibration_error, npv_image_level, ECEConfig


def onehot_binary(mask: torch.Tensor) -> torch.Tensor:
    # mask: (B,1,H,W) {0,1} -> (B,2,H,W) with [bg, polyp]
    polyp = (mask > 0.5).float()
    bg = 1.0 - polyp
    return torch.cat([bg, polyp], dim=1)


def build_model(cfg: Dict) -> DUETModel:
    fft_cfg = FFTSplitConfig(
        method=cfg["model"]["fft"]["method"],
        radius_ratio=float(cfg["model"]["fft"]["radius_ratio"]),
        energy_percentile=float(cfg["model"]["fft"]["energy_percentile"]),
    )
    fft = FFTSplit(fft_cfg)

    eh_cfg = EfficientNetB4Config(
        name=cfg["model"]["encoders"]["eh"]["name"],
        pretrained=bool(cfg["model"]["encoders"]["eh"]["pretrained"]),
        out_indices=tuple(cfg["model"]["encoders"]["eh"]["out_indices"]),
    )
    eh = EfficientNetB4Backbone(eh_cfg)

    # load stage1 weights if provided
    if cfg["model"]["stage1_ckpt"]:
        eh.load_stage1_weights(cfg["model"]["stage1_ckpt"], strict=False)

    mae_init = ViTMAEInit(
        mode=cfg["model"]["encoders"]["el"]["mae_init"]["mode"],
        timm_name=cfg["model"]["encoders"]["el"]["mae_init"]["timm_name"],
        ckpt_path=cfg["model"]["encoders"]["el"]["mae_init"]["ckpt_path"],
    )
    el_cfg = ViTMAEConfig(
        name=cfg["model"]["encoders"]["el"]["name"],
        img_size=int(cfg["model"]["encoders"]["el"]["img_size"]),
        patch_size=int(cfg["model"]["encoders"]["el"]["patch_size"]),
        feature_blocks=tuple(cfg["model"]["encoders"]["el"]["feature_blocks"]),
        mae_init=mae_init,
    )
    el = ViTMAEBackbone(el_cfg)

    # decoder config
    dec_cfg = FPNDecoderConfig(
        fpn_dim=int(cfg["model"]["decoders"]["fpn_dim"]),
        out_classes=2,
        dropout=float(cfg["model"]["decoders"]["dropout"]),
    )

    # infer in_channels for EH and EL by dummy forward
    size = int(cfg["data"]["image_size"])
    with torch.no_grad():
        dummy = torch.zeros(1, 3, size, size)
        ch_h = [t.shape[1] for t in eh(dummy)]
        ch_l = [t.shape[1] for t in el(dummy)]

    dh = FPNDecoder(ch_h, dec_cfg)
    dl = FPNDecoder(ch_l, dec_cfg)

    fusion_cfg = EvidenceFusionConfig(
        eps=float(cfg["model"]["fusion"]["eps"]),
        detach_weights=bool(cfg["model"]["fusion"]["detach_weights"]),
        type="avg_prob" if cfg.get("ablation") == "avg_prob_fusion" else "evidence_weighted",
    )
    fusion = EvidenceFusion(fusion_cfg)

    return DUETModel(fft=fft, eh=eh, el=el, dh=dh, dl=dl, fusion=fusion)


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
    tfm_train = build_transforms(TransformConfig(image_size=int(cfg["data"]["image_size"]), augment=True), for_stage1=False)
    tfm_val = build_transforms(TransformConfig(image_size=int(cfg["data"]["image_size"]), augment=False), for_stage1=False)

    train_ds = PolypMetaDataset(
        root=cfg["data"]["root"],
        meta_csv=cfg["data"]["meta_csv"],
        split_file=f'{cfg["data"]["split_dir"]}/{cfg["data"]["train_split"]}',
        transform=tfm_train,
    )
    val_ds = PolypMetaDataset(
        root=cfg["data"]["root"],
        meta_csv=cfg["data"]["meta_csv"],
        split_file=f'{cfg["data"]["split_dir"]}/{cfg["data"]["val_split"]}',
        transform=tfm_val,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_ddp else None

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                            sampler=val_sampler, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    # model
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if dist.is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank], find_unused_parameters=True)

    # ablations mapping (Sec.4.3)
    ab = cfg.get("ablation", "")
    if ab == "cnn_only":
        unwrap_ddp(model).el.requires_grad_(False)
        unwrap_ddp(model).dl.requires_grad_(False)
    elif ab == "vit_only":
        unwrap_ddp(model).eh.requires_grad_(False)
        unwrap_ddp(model).dh.requires_grad_(False)
    elif ab == "no_boundary_weight":
        cfg["model"]["evidential"]["boundary_weights"]["wB"] = 1.0
    elif ab == "no_edl":
        pass

    # optim with separate lr for encoders vs decoders (Training Details p.17)
    lr_enc = float(cfg["train"]["lr_encoder"])
    lr_dec = float(cfg["train"]["lr_decoder"])
    wd = float(cfg["train"]["weight_decay"])

    enc_params = list(unwrap_ddp(model).eh.parameters()) + list(unwrap_ddp(model).el.parameters())
    dec_params = list(unwrap_ddp(model).dh.parameters()) + list(unwrap_ddp(model).dl.parameters())

    optim = torch.optim.Adam(
        [{"params": enc_params, "lr": lr_enc}, {"params": dec_params, "lr": lr_dec}],
        weight_decay=wd,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    start_epoch, step, best = 0, 0, -1.0
    if cfg["train"]["resume"]:
        st = load_checkpoint(cfg["train"]["resume"], model, optim)
        start_epoch, step, best = st.epoch, st.step, st.best_metric

    edl_cfg = EDLConfig(lambda_kl=float(cfg["model"]["evidential"]["lambda_kl"]))
    reg_cfg = RegionWeightConfig(
        wB=float(cfg["model"]["evidential"]["boundary_weights"]["wB"]),
        wC=float(cfg["model"]["evidential"]["boundary_weights"]["wC"]),
        wBG=float(cfg["model"]["evidential"]["boundary_weights"]["wBG"]),
        re=int(cfg["model"]["evidential"]["boundary_weights"]["re"]),
        rd=int(cfg["model"]["evidential"]["boundary_weights"]["rd"]),
    )
    lambda_dice = float(cfg["model"]["evidential"]["lambda_dice"])

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            mask = batch["mask"]
            if mask is None:
                continue
            mask = mask.to(device).float()

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                out = model(img)
                y = onehot_binary(mask)

                # Eq.(7)(8)(9): region-weighted EDL for each decoder
                w, _ = make_region_weights(mask, reg_cfg)  # (B,1,H,W)
                l_h = edl_loss(out.alpha_h, y, edl_cfg, reduction="none") * w
                l_l = edl_loss(out.alpha_l, y, edl_cfg, reduction="none") * w
                l_edl = l_h.mean() + l_l.mean()

                # Dice on fused prob p* (paper Training Details: lambdaDice=0.5)
                l_dice = dice_loss(out.p_f, mask)

                if ab == "no_edl":
                    loss = lambda_dice * l_dice
                else:
                    loss = l_edl + lambda_dice * l_dice

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if logger is not None and step % int(cfg["train"]["log_every"]) == 0:
                logger.log_scalars(step, {
                    "train/loss": float(loss.item()),
                    "train/l_edl": float(l_edl.item()),
                    "train/l_dice": float(l_dice.item()),
                })
            step += 1

        # validation on Dice (paper: checkpoint best on val Dice)
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mask = batch["mask"]
                if mask is None:
                    continue
                mask = mask.to(device).float()

                out = model(img)
                dices.append(dice_coeff(out.p_f, mask).item())

        val_dice = float(sum(dices) / max(len(dices), 1))
        if logger is not None:
            logger.log_scalars(step, {"val/dice": val_dice})

            if val_dice > best:
                best = val_dice
                save_checkpoint(str(logger.ckpt_dir / "duet_best.pt"), model, optim, CheckpointState(epoch, step, best))
            save_checkpoint(str(logger.ckpt_dir / "duet_last.pt"), model, optim, CheckpointState(epoch, step, best))

    if logger is not None:
        logger.close()


def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


if __name__ == "__main__":
    main()
