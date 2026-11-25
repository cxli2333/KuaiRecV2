#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p log

python -u streaming_ewc_lora_sft.py "$@" \
  > "./log/training_sft_click_ewc_lora_$(date +%Y%m%d_%H%M%S).log" 2>&1
