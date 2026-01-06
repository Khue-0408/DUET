# INSTALL.md — DUET (Domain-adaptive Uncertainty-aware Evidential Two-stream) Polyp Segmentation

Tài liệu này là “how to run” dạng template cho codebase DUET theo đúng 3-stage pipeline trong paper:
- Stage 1: Boundary-aware + domain-adversarial encoder pretraining (unlabeled images)
- Stage 2–3: Dual-stream CNN–ViT + evidential decoders + evidence fusion (labeled images)

Nếu repo của bạn dùng tên file/script khác, giữ nguyên logic và thay bằng entrypoints tương ứng.

---

## 0) Repo layout khuyến nghị

```
<repo_root>/
  configs/
    stage1.yaml
    stage3.yaml
  data/
    unlabeled/                # tập DU (~15k frames) cho Stage 1
    Kvasir-SEG/
    PolypGen/
    PolypDB/
    CVC-ClinicDB/
    ColonDB/
    ETIS-LaribDB/
  splits/
    polypgen_loco/            # LOCO split files (train/val/test per center)
    polypdb_cross_modality/   # split theo modality nếu bạn dùng
  weights/
    pretrained/
      efficientnet_b4.pth     # optional, hoặc dùng timm
      vit_b16_mae.pth         # MAE pretrained weights
    stage1/
      encoder_stage1.pth      # output Stage 1
    stage3/
      duet_best.pth           # output Stage 3 (full model)
  outputs/
    logs/
    checkpoints/
    preds/
  src/
    ...
  requirements.txt
  README.md
```

---

## 1) Environment

### Option A — Conda (khuyến nghị)
```bash
conda create -n duet python=3.10 -y
conda activate duet
pip install -U pip
pip install -r requirements.txt
```

### Option B — venv
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -U pip
pip install -r requirements.txt
```

Kiểm tra CUDA:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2) Data placement

### 2.1 Unlabeled set (Stage 1)
Đặt toàn bộ frames “unlabeled” vào:
```
data/unlabeled/
  <domain_1>/*.jpg
  <domain_2>/*.jpg
  ...
```

Gợi ý: mỗi domain/center/dataset tách thành 1 folder riêng để lấy domain label `d(i)` dễ dàng.

### 2.2 Labeled datasets (Stage 3)
Mỗi dataset nên có cấu trúc tối thiểu:
```
data/<DATASET_NAME>/
  images/
  masks/
```

Ví dụ:
```
data/Kvasir-SEG/images/*.jpg
data/Kvasir-SEG/masks/*.png
```

### 2.3 Splits
- PolypGen LOCO: đặt các file split (ví dụ json/csv) vào `splits/polypgen_loco/`.
- PolypDB cross-modality: đặt split vào `splits/polypdb_cross_modality/`.

---

## 3) Pretrained weights placement

Đặt tất cả weights tải sẵn vào:
```
weights/pretrained/
```

Khuyến nghị tối thiểu:
- ViT-Base patch16 MAE pretrained weights: `weights/pretrained/vit_b16_mae.pth`
- (Tuỳ chọn) EfficientNet-B4 pretrained weights: `weights/pretrained/efficientnet_b4.pth`
  - Nếu code dùng `timm`, có thể không cần file này vì `timm` tự load.

---

## 4) Running the pipeline

Các lệnh dưới đây là template. Thay tên script / args cho khớp repo của bạn.

### 4.1 Stage 1 — boundary-aware + domain-adversarial pretraining (unlabeled)
```bash
python -m src.train_stage1 \
  --config configs/stage1.yaml \
  --data_root data/unlabeled \
  --save_dir weights/stage1 \
  --out_name encoder_stage1.pth
```

Output mong đợi:
```
weights/stage1/encoder_stage1.pth
```

### 4.2 Stage 3 — supervised training (dual-stream + evidential + fusion)
Stage 3 sẽ:
- load CNN encoder từ Stage 1
- load ViT MAE pretrained
- train decoders + evidence fusion trên labeled sets

```bash
python -m src.train_stage3 \
  --config configs/stage3.yaml \
  --data_root data \
  --splits_root splits \
  --cnn_ckpt weights/stage1/encoder_stage1.pth \
  --vit_ckpt weights/pretrained/vit_b16_mae.pth \
  --save_dir weights/stage3
```

Output mong đợi:
```
weights/stage3/duet_best.pth
```

### 4.3 Evaluation
```bash
python -m src.eval \
  --ckpt weights/stage3/duet_best.pth \
  --data_root data \
  --splits_root splits \
  --dataset PolypGen \
  --protocol LOCO
```

### 4.4 Inference (single folder)
```bash
python -m src.infer \
  --ckpt weights/stage3/duet_best.pth \
  --input_dir <path_to_images> \
  --output_dir outputs/preds
```

---

## 5) Reproducing paper-ish settings (tham số gợi ý)

Các settings phổ biến theo paper:
- Stage 1: Adam, 100 epochs, lr 1e-4; contrastive sampling 64 pairs/image/iter; temperature τ=0.2; λadv=0.5
- Stage 2/3: 50 epochs; lr encoder 2e-4, decoder 5e-4; λKL=0.1; boundary weight wB=3; λDice=0.5

Bạn nên encode các tham số này trong `configs/stage1.yaml` và `configs/stage3.yaml`.

---

## 6) Troubleshooting

- Out-of-memory: giảm `batch_size`, giảm `img_size`, bật AMP, hoặc dùng gradient accumulation.
- Dataset path mismatch: kiểm tra `data_root` và tên folder `images/`, `masks/`.
- LOCO split not found: kiểm tra `splits/polypgen_loco/` và format file split.

---

## 7) Quick checklist

- [ ] `pip install -r requirements.txt` chạy OK
- [ ] `data/unlabeled/` có subfolders theo domain
- [ ] `data/<dataset>/images` và `data/<dataset>/masks` đúng
- [ ] `weights/pretrained/vit_b16_mae.pth` tồn tại
- [ ] Stage 1 tạo ra `weights/stage1/encoder_stage1.pth`
- [ ] Stage 3 tạo ra `weights/stage3/duet_best.pth`
