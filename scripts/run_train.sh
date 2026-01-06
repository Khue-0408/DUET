# path: scripts/run_train.sh
#!/usr/bin/env bash
python -m src.train.train_duet --config configs/duet_default.yaml --override model.stage1_ckpt=runs/stage1_pretrain/checkpoints/stage1_encoder_best.pt
