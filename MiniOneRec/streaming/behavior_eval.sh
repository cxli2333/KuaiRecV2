#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT_DIR="${RESULTS_ROOT_DIR:-${PROJECT_DIR}/streaming_data/train5/Qwen2.5-0.5B}"
FILE_PATHS_JSON="${FILE_PATHS_JSON:-${SCRIPT_DIR}/Full_Month_file_paths_relative.json}"
MASTER_INFO_DIR="${MASTER_INFO_DIR:-${PROJECT_DIR}/streaming_data/info}"
METRIC_PY="${METRIC_PY:-${PROJECT_DIR}/metric.py}"
mkdir -p "${RESULTS_ROOT_DIR}"

python "${SCRIPT_DIR}/behavior_eval.py" \
  --results_root_dir "${RESULTS_ROOT_DIR}" \
  --result_filename results_neg_1.json \
  --file_paths_json "${FILE_PATHS_JSON}" \
  --master_item_json "${MASTER_INFO_DIR}/master.item.json" \
  --master_index_json "${MASTER_INFO_DIR}/master.index.json" \
  --metric_py "${METRIC_PY}" \
  --item_info_txt "${MASTER_INFO_DIR}/master.info.txt" \
  --filter_behaviors is_click \
  --merged_summary_json "${RESULTS_ROOT_DIR}/all_round_behavior_metrics.json" \
  2>&1 | tee "${RESULTS_ROOT_DIR}/behavior_eval.log"
