# path: scripts/run_infer.sh
#!/usr/bin/env bash
python -m src.train.infer --config configs/duet_default.yaml --override infer.image_path=$1 --override infer.out_dir=outputs/infer
