# path: src/train/eval.py
from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.datasets import PolypMetaDataset
from src.data.transforms import TransformConfig, build_transforms
from src.data.samplers import build_loco_splits_from_meta
from src.models.duet import DUETModel
from src.train.train_duet import build_model
from src.utils.config import apply_overrides, load_yaml
from src.utils.metrics import dice_coeff, miou, precision_recall, expected_calibration_error, npv_image_level, ECEConfig


def eval_once(cfg: Dict, split_file: str) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = build_transforms(TransformConfig(image_size=int(cfg["data"]["image_size"]), augment=False))
    ds = PolypMetaDataset(
        root=cfg["data"]["root"],
        meta_csv=cfg["data"]["meta_csv"],
        split_file=split_file,
        transform=tfm,
    )
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    model = build_model(cfg).to(device)
    ckpt = cfg.get("eval", {}).get("ckpt", "") or f'{cfg["logging"]["out_dir"]}/{cfg["exp"]["name"]}/checkpoints/duet_best.pt'
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    dices, ious, precs, recs, eces, npvs = [], [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            mask = batch["mask"]
            if mask is None:
                continue
            mask = mask.to(device).float()
            out = model(img)

            dices.append(dice_coeff(out.p_f, mask).item())
            ious.append(miou(out.p_f, mask).item())
            p, r = precision_recall(out.p_f, mask)
            precs.append(p.item())
            recs.append(r.item())

            eces.append(expected_calibration_error(out.p_f, mask, ECEConfig(bins=int(cfg["eval"]["ece_bins"]))).item())
            npvs.append(npv_image_level(out.p_f, mask, threshold=float(cfg["eval"]["npv_threshold"])).item())

    return {
        "dice": float(sum(dices) / max(len(dices), 1)),
        "miou": float(sum(ious) / max(len(ious), 1)),
        "precision": float(sum(precs) / max(len(precs), 1)),
        "recall": float(sum(recs) / max(len(recs), 1)),
        "ece": float(sum(eces) / max(len(eces), 1)),
        "npv": float(sum(npvs) / max(len(npvs), 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", type=str, nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    if cfg["eval"]["loco"]["enabled"]:
        # build LOCO splits if missing
        folds = build_loco_splits_from_meta(cfg["data"]["root"], cfg["data"]["meta_csv"], out_dir="splits_loco")
        all_stats = []
        for heldout, tr, te in folds:
            stats = eval_once(cfg, split_file=te)
            stats["heldout_center"] = heldout
            all_stats.append(stats)
            print(f"[LOCO center={heldout}] {stats}")

        # average
        keys = ["dice", "miou", "precision", "recall", "ece", "npv"]
        avg = {k: sum(s[k] for s in all_stats) / max(len(all_stats), 1) for k in keys}
        print(f"[LOCO average] {avg}")
    else:
        split = cfg["eval"]["split"]
        split_file = f'{cfg["data"]["split_dir"]}/{cfg["data"][f"{split}_split"]}'
        stats = eval_once(cfg, split_file=split_file)
        print(stats)


if __name__ == "__main__":
    main()
