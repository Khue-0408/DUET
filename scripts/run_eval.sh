# path: scripts/run_eval.sh
#!/usr/bin/env bash
python -m src.train.eval --config configs/duet_default.yaml --override eval.split=test
