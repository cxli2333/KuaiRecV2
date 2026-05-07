import json
import os
import subprocess
import sys
import argparse
import re
import pandas as pd

from streaming_data_prep import load_master_lookups, prepare_list_eval_from_two_days

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

FILE_PATHS_JSON = os.path.join(PARENT_DIR, "streaming", "Full_Month_file_paths_relative.json")
MASTER_INFO_DIR = os.path.join(PARENT_DIR, "streaming_data", "info")
MASTER_ITEM_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.item.json")
MASTER_INDEX_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.index.json")
MASTER_INFO_TXT_PATH = os.path.join(MASTER_INFO_DIR, "master.info.txt")

EVALUATE_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "streaming_eval.py")
LIST_METRIC_SCRIPT_PATH = os.path.join(PARENT_DIR, "list_metric.py")

PYTHON_EXECUTABLE = sys.executable

DEFAULT_TRAINED_MODEL_PATH = os.path.join(PARENT_DIR, "streaming_data", "train_click_e10")
DEFAULT_OUTPUT_PATH = os.path.join(PARENT_DIR, "streaming_data", "list_eval_runs")


def parse_metric_output(stdout_str):
    match = re.search(r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___", stdout_str, re.DOTALL)
    if not match:
        raise RuntimeError(f"Unable to parse metric output:\n{stdout_str[-1000:]}")
    return json.loads(match.group(1).strip())


def resolve_day_idx(idx, total_len):
    return idx if idx >= 0 else total_len + idx


def backfill_output_field(result_json_path, test_csv_path):
    """
    如果 EvalSidDataset 没把 output 原样透传到 results.json，这里补回去。
    """
    if not os.path.exists(result_json_path):
        return

    with open(result_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if len(results) == 0:
        return

    if "output" in results[0]:
        return

    df = pd.read_csv(test_csv_path, dtype={"user_id": str})
    user_to_output = {str(row["user_id"]): row["output"] for _, row in df.iterrows()}
    user_to_output_titles = {}
    if "output_title_list" in df.columns:
        user_to_output_titles = {str(row["user_id"]): row["output_title_list"] for _, row in df.iterrows()}

    updated = 0
    for row in results:
        uid = str(row.get("user_id", ""))
        if uid in user_to_output:
            row["output"] = user_to_output[uid]
            if uid in user_to_output_titles:
                row["output_title_list"] = user_to_output_titles[uid]
            updated += 1

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Backfill] Added output field for {updated} rows")


def main(args):
    valid_behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]

    with open(args.file_paths_json, "r") as f:
        paths = json.load(f)

    interaction_files = paths["interaction_files"]
    total_days = len(interaction_files)

    history_day_idx = resolve_day_idx(args.history_day_idx, total_days)
    target_day_idx = resolve_day_idx(args.target_day_idx, total_days)

    if not (0 <= history_day_idx < total_days):
        raise ValueError(f"history_day_idx out of range: {args.history_day_idx}")
    if not (0 <= target_day_idx < total_days):
        raise ValueError(f"target_day_idx out of range: {args.target_day_idx}")
    if history_day_idx >= target_day_idx:
        raise ValueError("history_day_idx must be smaller than target_day_idx")

    history_day_file = interaction_files[history_day_idx]
    target_day_file = interaction_files[target_day_idx]

    item_sid_map, item_title_map, valid_items = load_master_lookups(
        MASTER_ITEM_JSON_PATH, MASTER_INDEX_JSON_PATH
    )
    if item_sid_map is None:
        return

    eval_dir = os.path.join(
        args.output_path,
        args.model_name,
        f"ckpt_round_{args.ckpt_round}_hist_{history_day_idx}_target_{target_day_idx}"
    )
    os.makedirs(eval_dir, exist_ok=True)

    test_csv_path = os.path.join(eval_dir, "test_list_eval.csv")
    if args.rebuild_test or not os.path.exists(test_csv_path):
        generated_path = prepare_list_eval_from_two_days(
            history_day_file=history_day_file,
            target_day_file=target_day_file,
            output_dir=eval_dir,
            item_sid_map=item_sid_map,
            item_title_map=item_title_map,
            valid_items=valid_items,
            max_history_len=args.max_history_len,
            target_len=args.target_len,
            valid_behaviors=valid_behaviors,
            strict_target_len=bool(args.strict_target_len),
        )
        if generated_path is None:
            print("Failed to build list-eval test set")
            return
        test_csv_path = generated_path
    else:
        print(f"[Reuse] existing test file: {test_csv_path}")

    model_checkpoint = os.path.join(
        args.trained_model_path,
        args.model_name,
        f"streaming_round_{args.ckpt_round}",
        "model_checkpoint"
    )
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint}")

    result_json_path = os.path.join(eval_dir, "results.json")

    if args.force_eval or not os.path.exists(result_json_path):
        eval_command = [
            "accelerate", "launch", f"--num_processes={args.num_gpus}", EVALUATE_SCRIPT_PATH,
            "--base_model", model_checkpoint,
            "--test_data_path", test_csv_path,
            "--info_file", MASTER_INFO_TXT_PATH,
            "--result_json_data", result_json_path,
            "--category", args.category,
            "--batch_size", str(args.batch_size),
            "--num_beams", str(args.num_beams),
            "--negative_sample_rate", str(args.negative_sample_rate),
            "--min_diversity", str(args.min_diversity),
        ]
        print("[Eval] " + " ".join(eval_command))
        subprocess.run(eval_command, check=True)
    else:
        print(f"[Reuse] existing result file: {result_json_path}")

    backfill_output_field(result_json_path, test_csv_path)

    metric_command = [
        PYTHON_EXECUTABLE, LIST_METRIC_SCRIPT_PATH,
        "--path", result_json_path,
        "--item_path", MASTER_INFO_TXT_PATH,
        "--topk", "1,5,10,20,50"
    ]
    print("[Metric] " + " ".join(metric_command))
    process_metrics = subprocess.run(metric_command, capture_output=True, text=True, check=False)

    print(process_metrics.stdout)
    if process_metrics.returncode != 0:
        print(process_metrics.stderr)
        raise RuntimeError("list_metric.py failed")

    metrics = parse_metric_output(process_metrics.stdout)

    summary_path = os.path.join(eval_dir, "list_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[Done] Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_round", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--trained_model_path", type=str, default=DEFAULT_TRAINED_MODEL_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--history_day_idx", type=int, default=-2)
    parser.add_argument("--target_day_idx", type=int, default=-1)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--target_len", type=int, default=50)
    parser.add_argument("--strict_target_len", type=int, default=1)

    parser.add_argument("--behaviors", type=str, default="is_click")
    parser.add_argument("--category", type=str, default="Streaming")
    parser.add_argument("--file_paths_json", type=str, default=FILE_PATHS_JSON)

    parser.add_argument("--num_gpus", type=str, default="8")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--negative_sample_rate", type=float, default=0.0)
    parser.add_argument("--min_diversity", type=int, default=5)

    parser.add_argument("--rebuild_test", action="store_true")
    parser.add_argument("--force_eval", action="store_true")

    args = parser.parse_args()
    main(args)
