# path: docs/paper_trace.md
# Paper-trace map (DUET PDF only)

Notation in this doc:
- (p.X) = PDF page number
- Sec / Eq / Fig refer to paper numbering

## Stage 1 (Boundary-aware domain-adaptive pretraining)
Reference: Sec. 3.1 (p.7–10), Eq. (1)(2)(3), Fig.1 (p.8), Training Details (p.17)

Implemented in:
- src/train/train_stage1.py
- src/models/encoders/efficientnet_b4.py
- src/models/losses/contrastive_infonce.py   (Eq. (1), with sampled pairs)
- src/models/losses/dice.py                  (Eq. (2) Lseg Dice)
- src/models/domain/grl.py                   (GRL for Eq. (3))
- src/models/domain/domain_discriminator.py  (2-layer MLP domain classifier)
- src/data/datasets.py + region generation in train loop (Sec. 3.1 pseudomask vs weak)

Key hyperparameters (Training Details p.17):
- Adam, 100 epochs, lr=1e-4
- lambda_adv=0.5
- tau=0.2
- 64 pairs/image/iter

ASSUMPTION/TODO (paper does not fully specify):
- re/rd are "5–10 pixels" (p.9). Default re=5, rd=10.
- Exact sampling strategy for InfoNCE not fully specified beyond "64 pairs". Default: 64 anchors with 1 pos + 1 neg each.

## Stage 2 (Dual-stream + FFT split)
Reference: Sec. 3.2 (p.10–12), Fig.1 (p.8)

Implemented in:
- src/models/freq/fft_split.py
- src/models/encoders/efficientnet_b4.py (EH)
- src/models/encoders/vit_mae.py (EL)

ASSUMPTION/TODO:
- FFT cutoff radius not specified (p.11). Default radius_ratio=0.1; alternative energy_percentile.

ViT init:
- Paper: ViT-Base patch16 initialized from MAE (p.11). Repo supports:
  - timm MAE name (convenience; not a paper fact)
  - user-provided checkpoint path

## Stage 3 (Evidential + losses + fusion)
Reference: Sec. 3.3 (p.12–15), Eq. (4)–(12), Training Details (p.17)

Implemented in:
- src/models/decoders/fpn_decoder.py (lightweight FPN; paper says "lightweight FPN", p.12)
- src/models/evidential/evidence.py (Eq. (4)(5)(6))
- src/models/losses/edl.py (Eq. (7)(8), KL Dirichlet)
- src/models/losses/region_weighting.py (Eq. (9) masks RB/RC/RBG)
- src/models/fusion/evidence_fusion.py (Eq. (10)(11)(12))
- src/models/duet.py (full forward; p* and U)

Loss hyperparameters (Training Details p.17):
- lambda_KL=0.1
- wB=3, wC=1, wBG=1
- lambda_Dice=0.5
- Stage2/3: 50 epochs, lr_enc=2e-4, lr_dec=5e-4
- augmentations: flips/rotations/color jitter

Uncertainty U:
- Paper describes expected variance of Bernoulli from Dirichlet/Beta (end Sec. 3.3).
- Implemented as Var(p) = alpha_bg * alpha_polyp / (S^2 * (S+1)) for binary.

## Evaluation protocol
Reference: Sec. 4.1 (p.15–17)
Metrics:
- Dice, mIoU, Recall, Precision
- ECE with 10 bins
- NPV: image-level using max p* per image
LOCO PolypGen supported (Sec. 4.1).
