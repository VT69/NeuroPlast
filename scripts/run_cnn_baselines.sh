#!/usr/bin/env bash
# Priority 2: CNN + PPO baselines, 2 tasks x 2 seeds, in parallel (1 thread each).
cd "$(dirname "$0")/.."
mkdir -p runs/logs
for s in 1 2; do
  python train.py --config configs/cnn_doorkey.yaml --set seed=$s run_name=cnn_doorkey5_s$s > runs/logs/cnn_doorkey5_s$s.log 2>&1 &
  python train.py --config configs/cnn_lavagap.yaml --set seed=$s run_name=cnn_lavagap5_s$s > runs/logs/cnn_lavagap5_s$s.log 2>&1 &
done
wait
