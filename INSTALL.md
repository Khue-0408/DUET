````md
# path: INSTALL.md

# DUET Installation & How to Run

This file explains:
1) how to set up the environment
2) where to place datasets + pseudo-masks
3) where to place pretrained weights / checkpoints
4) how to run Stage 1 (pretrain), Stage 2/3 (train), eval, inference, and ablations

Repo entrypoints (expected):
- Stage 1: `python -m src.train.train_stage1 ...`
- Stage 2/3: `python -m src.train.train_duet ...`
- Eval:     `python -m src.train.eval ...`
- Infer:    `python -m src.train.infer ...`

---

## 0) System prerequisites

- OS: Linux recommended
- Python: 3.10+ (3.10/3.11 tested best in most PyTorch stacks)
- CUDA: optional but recommended (Stage1/Stage2 training is heavy on GPU)
- GPUs: 1 GPU works; multi-GPU via torchrun (DDP) is supported (skeleton)

---

## 1) Create environment

### Option A: conda
```bash
conda create -n duet python=3.10 -y
conda activate duet
pip install -U pip
pip install -r requirements.txt
````

### Option B: venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Repository expected folders (you create these)

Create:

```bash
mkdir -p data weights outputs
```

Recommended layout:

```
DUET/
  data/                # YOU provide datasets here (no auto-download)
  weights/             # optional external pretrained weights (ViT-MAE etc.)
  outputs/             # all experiment runs, logs, checkpoints
  configs/             # yaml configs (duet_default.yaml, stage1_pretrain.yaml, ablations/*.yaml)
```

---

## 3) Dataset layout (no downloads)

### 3.1 Generic dataset structure (recommended default)

For each dataset (Kvasir, CVC, ETIS, PolypGen, ...), place:

```
data/<DATASET_NAME>/
  images/
    train/   *.png|*.jpg
    val/     *.png|*.jpg
    test/    *.png|*.jpg
  masks/
    train/   *.png  (binary mask, 0 background, 255 foreground recommended)
    val/     *.png
    test/    *.png
  meta/
    index.csv         # optional but recommended
    centers.csv       # only needed for LOCO (PolypGen)
  pseudomasks/        # optional; needed for Stage1 pseudomask mode
    train/ *.png
    val/   *.png
```

If you do NOT have `meta/index.csv`, the loader will fall back to filename matching between `images/<split>/` and `masks/<split>/`.

#### index.csv (optional)

Example columns (minimal):

* `image_relpath` (e.g., `images/train/0001.png`)
* `mask_relpath`  (e.g., `masks/train/0001.png`)
* `split`         (`train|val|test`)
* `center_id`     (optional; used by LOCO)

---

### 3.2 PolypGen LOCO protocol layout (two supported modes)

Mode A (recommended): center_id in a CSV

```
data/PolypGen/
  images/{train,val,test}/...
  masks/{train,val,test}/...
  meta/centers.csv
```

`meta/centers.csv` example columns:

* `image_id`   (filename without extension or full filename)
* `center_id`  (string, e.g., `C1`, `C2`, ...)
* `split`      (optional; if absent, split is derived from folder)

Mode B (fallback): folder-per-center

```
data/PolypGen/
  centers/
    C1/images/*.png
    C1/masks/*.png
    C2/images/*.png
    C2/masks/*.png
    ...
```

LOCO will treat one center as hold-out test, the remaining as train/val (val can be a small split from train).

---

## 4) Pseudo-mask placement (Stage 1 option)

If you run Stage 1 in pseudomask mode, provide pseudo masks here:

```
data/<DATASET_NAME>/pseudomasks/<split>/<same_filename_as_image>.png
```

Conventions:

* binary masks: 0 background, 255 foreground (recommended)
* loader will binarize automatically

If you do not have pseudo masks, you can:

* disable auxiliary segmentation head in Stage 1 (config switch), OR
* use weak/self-supervised region generation mode (baseline edge heuristic)

Placeholder script:

```bash
bash scripts/prepare_pseudomasks_placeholder.sh data/<DATASET_NAME>/
```

---

## 5) Pretrained weights and checkpoints placement

### 5.1 Stage 1 output (EfficientNet-B4 encoder weights)

Stage 1 will write to:

```
outputs/stage1/<RUN_NAME>/
  checkpoints/
    best.ckpt
    last.ckpt
  logs/
  config.yaml
```

You will reference `best.ckpt` in Stage 2 config:

* `model.eh_pretrained_ckpt: outputs/stage1/<RUN_NAME>/checkpoints/best.ckpt`

---

### 5.2 ViT-Base MAE initialization (Stage 2)

Two supported options:

Option (i) timm pretrained MAE (default if enabled)

* set: `model.vit_init: timm_mae`
* timm will download/cache weights automatically (internet required once), OR you can pre-populate the cache.

Option (ii) user-provided MAE checkpoint (offline-friendly)

* set: `model.vit_init: checkpoint`
* place file:

```
weights/vit_mae/vit_base_patch16_mae.pth
```

* set: `model.vit_ckpt_path: weights/vit_mae/vit_base_patch16_mae.pth`

Note: if you use timm pretrained, the cache is typically under `~/.cache/torch/hub/checkpoints/` (can be redirected via `TORCH_HOME`). See “Cache control” below.

---

### 5.3 Cache control (recommended for reproducibility)

Set a fixed cache directory (helpful on servers / clusters):

```bash
export TORCH_HOME=$PWD/.torch_cache
mkdir -p $TORCH_HOME
```

---

## 6) Run commands

All commands accept:

* `--config <yaml>`
* optional overrides: `key=value` (if your runner supports it), otherwise edit YAML.

### 6.1 Stage 1: Boundary-aware domain-adaptive pretraining

Single GPU:

```bash
python -m src.train.train_stage1 \
  --config configs/stage1_pretrain.yaml \
  --output_dir outputs/stage1/exp01
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=4 -m src.train.train_stage1 \
  --config configs/stage1_pretrain.yaml \
  --output_dir outputs/stage1/exp01
```

Expected artifacts:

* `outputs/stage1/exp01/checkpoints/best.ckpt`
* `outputs/stage1/exp01/logs/*`

---

### 6.2 Stage 2/3: DUET training (dual-stream + evidential + fusion)

Single GPU:

```bash
python -m src.train.train_duet \
  --config configs/duet_default.yaml \
  --output_dir outputs/duet/exp01
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 -m src.train.train_duet \
  --config configs/duet_default.yaml \
  --output_dir outputs/duet/exp01
```

Important: ensure Stage 2 config points to Stage 1 checkpoint:

* `model.eh_pretrained_ckpt: outputs/stage1/exp01/checkpoints/best.ckpt`

---

### 6.3 Evaluate

Single dataset (example: PolypGen test):

```bash
python -m src.train.eval \
  --config configs/duet_default.yaml \
  --ckpt outputs/duet/exp01/checkpoints/best.ckpt \
  --dataset PolypGen \
  --split test
```

LOCO (hold out one center):

```bash
python -m src.train.eval \
  --config configs/duet_default.yaml \
  --ckpt outputs/duet/exp01/checkpoints/best.ckpt \
  --dataset PolypGen \
  --protocol loco \
  --holdout_center C3
```

Outputs:

* metrics json/csv (Dice, mIoU, Precision, Recall)
* calibration metrics (ECE 10 bins)
* NPV image-level

---

### 6.4 Inference

Infer a single image:

```bash
python -m src.train.infer \
  --config configs/duet_default.yaml \
  --ckpt outputs/duet/exp01/checkpoints/best.ckpt \
  --input path/to/image.png \
  --output outputs/infer/single/
```

Infer a folder:

```bash
python -m src.train.infer \
  --config configs/duet_default.yaml \
  --ckpt outputs/duet/exp01/checkpoints/best.ckpt \
  --input path/to/folder_images/ \
  --output outputs/infer/folder/
```

Will save:

* predicted mask
* fused probability map
* uncertainty map (U)

---

## 7) Ablations

Ablation configs live in:

```
configs/ablations/
  no_ssl.yaml
  no_adv.yaml
  cnn_only.yaml
  vit_only.yaml
  early_fusion.yaml
  late_fusion.yaml
  no_edl.yaml
  no_boundary_weight.yaml
  avg_prob_fusion.yaml
```

Example run:

```bash
python -m src.train.train_duet \
  --config configs/ablations/cnn_only.yaml \
  --output_dir outputs/duet/abl_cnn_only
```

---

## 8) Reproducibility checklist

* Set `seed` in YAML (and do not change it across runs)
* Use `deterministic: true` if you need strict reproducibility (slower)
* Keep:

  * `outputs/<run>/config.yaml` (auto-copied from your config)
  * git commit hash (log it)
  * full environment (`pip freeze > outputs/<run>/pip_freeze.txt`)

---

## 9) Common pitfalls

1. Masks not binary

* ensure mask is {0,255} or {0,1}; loader will binarize but ambiguous palettes can break.

2. Filename mismatch

* `images/train/0001.png` must match `masks/train/0001.png` unless you provide `meta/index.csv`.

3. ViT weights missing

* if `model.vit_init=checkpoint`, make sure `weights/vit_mae/...` exists.
* if `model.vit_init=timm_mae`, ensure the machine can access the cached weights (or has internet once).

4. LOCO center mapping missing

* provide `meta/centers.csv` OR use the folder-per-center layout.

---

```

Ghi chú nguồn tham khảo cho các điểm “có thể thay đổi theo hệ thống” đã dùng trong INSTALL.md:
- torchrun/DDP và tham số `--nproc_per_node`: :contentReference[oaicite:0]{index=0}  
- Torch Hub cache directory + biến môi trường `TORCH_HOME`: :contentReference[oaicite:1]{index=1}  
- timm pretrained weights thường cache ở `~/.cache/torch/hub/checkpoints`: :contentReference[oaicite:2]{index=2}  
- tên model MAE trong timm (ví dụ `vit_base_patch16_224.mae`): :contentReference[oaicite:3]{index=3}  

**Q1:** Mình muốn mặc định hoá luôn 2 layout data (theo `index.csv` và theo folder) bằng một cơ chế auto-detect, bạn thiết kế rule + ưu tiên như nào để ít bug nhất?  
**Q2:** Mình muốn “offline fully reproducible”: bạn đề xuất quy trình đóng gói weights (EH pretrained, ViT MAE, bất kỳ backbone khác) vào `weights/` thế nào để không phụ thuộc cache?  
**Q3:** Với PolypGen LOCO, mình muốn chuẩn hoá `centers.csv` để vừa support LOCO vừa support cross-dataset eval, bạn đề xuất schema + validator (script) ra sao?
::contentReference[oaicite:4]{index=4}
```
