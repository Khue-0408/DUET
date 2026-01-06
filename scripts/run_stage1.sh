# path: scripts/run_stage1.sh
#!/usr/bin/env bash
python -m src.train.train_stage1 --config configs/stage1_pretrain.yaml
