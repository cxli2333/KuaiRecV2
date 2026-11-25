#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p log

python -u backtest_first_day_checkpoints.py \
  --checkpoint_root ../streaming_data/click_seed2026_ewclora2/model_checkpoint \
  --test_file ../streaming_data/train3/Qwen2.5-0.5B/streaming_round_0/test_T.csv \
  --output_dir ../streaming_data/backtest_first_day/ewc_Qwen2.5-0.5B \
  --num_gpus 8 \
  --batch_size 32 \
  --num_beams 50 \
  --negative_sample_rate 1 \
  > ./log/backtest_first_day_$(date +%Y%m%d_%H%M).log 2>&1
