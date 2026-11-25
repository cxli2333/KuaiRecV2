"""Streaming entry point for continual EWC-LoRA supervised fine-tuning.

Place this file beside the existing streaming_sft.py and place
ewc_lora_sft.py beside sft.py.  Existing data preparation, evaluation, and
metric scripts are reused unchanged.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

FILE_PATHS_JSON = os.path.join(
    PARENT_DIR, "streaming", "Full_Month_file_paths_relative.json"
)
MASTER_INFO_DIR = os.path.join(PARENT_DIR, "streaming_data", "info")
MASTER_ITEM_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.item.json")
MASTER_INDEX_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.index.json")
MASTER_INFO_TXT_PATH = os.path.join(MASTER_INFO_DIR, "master.info.txt")

TRAIN_SCRIPT_PATH = os.path.join(PARENT_DIR, "ewc_lora_sft.py")
EVALUATE_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "streaming_eval.py")
METRIC_SCRIPT_PATH = os.path.join(PARENT_DIR, "metric.py")

BASE_MODEL = "../streaming_data/click_seed2026/Qwen2.5-0.5B/streaming_round_0/model_checkpoint"
NUM_GPUS = 8
TASKS_TO_TRAIN = "streaming_sid2sid"
START_DAY = 0
WINDOW_DAYS = 7
CATEGORY_NAME = "Streaming"
WANDB_PROJECT = "MiniOneRec-Streaming-EWCLoRA"
WANDB_RUN_NAME = "train-ewc-lora"

SAVED_PATH = os.path.join(
    SCRIPT_DIR, "..", "streaming_data", "click_seed2026_ewclora2"
)
NEGATIVE_EVAL_SAVED_PATH = os.path.join(
    SCRIPT_DIR, "..", "streaming_data", "negative_eval_ewclora2"
)
TOPK_LIST_FOR_METRIC = [1, 5, 10, 20, 50]


def parse_metric_output(stdout_str: str, topk_list: List[int]) -> Dict:
    metrics = {}
    match = re.search(
        r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___",
        stdout_str,
        re.DOTALL,
    )
    if not match:
        print("Error: metric.py did not emit a metric JSON block")
        print(f"metric.py stdout tail: {stdout_str[-500:]}")
        return metrics

    try:
        raw_metrics = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        print(f"Error: invalid metric JSON: {exc}")
        return metrics

    for index, k in enumerate(topk_list):
        for metric_name in ("NDCG", "HR", "Recall", "MRR", "AUC", "Coverage", "Gini"):
            values = raw_metrics.get(metric_name)
            if isinstance(values, list) and index < len(values):
                metrics[f"{metric_name}@{k}"] = values[index]

    for key in (
        "Avg_Path_Diversity",
        "Avg_Layer_Unique_Tokens",
        "Avg_Layer_Entropy",
        "Avg_Intra_List_Diversity",
        "TopK_Layer_Diversity",
    ):
        if key in raw_metrics:
            metrics[key] = raw_metrics[key]
    metrics["_raw_all_metrics"] = raw_metrics
    return metrics


def _require_files(paths: List[str]) -> None:
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Required project files are missing:\n  {formatted}")


def _run(command: List[str], env=None, capture_output=False):
    print("Command:", " ".join(command))
    return subprocess.run(
        command,
        check=False,
        env=env,
        capture_output=capture_output,
        text=capture_output,
    )


def _merge_distributed_results(result_json_path: str) -> bool:
    """Merge part files only when the evaluator did not create the final file."""
    if os.path.isfile(result_json_path):
        return True

    stem = os.path.splitext(os.path.basename(result_json_path))[0]
    pattern = os.path.join(os.path.dirname(result_json_path), f"{stem}_part_*.json")
    part_files = sorted(glob.glob(pattern))
    if not part_files:
        return False

    merged = []
    for part_file in part_files:
        with open(part_file, "r", encoding="utf-8") as handle:
            merged.extend(json.load(handle))
    with open(result_json_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)
    for part_file in part_files:
        os.remove(part_file)
    print(f"Merged {len(merged)} distributed evaluation samples")
    return True


def main(args) -> None:
    from streaming_data_prep import load_master_lookups, prepare_data_for_day

    master_item_path = args.master_item_json_path
    master_index_path = args.master_index_json_path
    master_info_path = args.master_info_txt_path
    _require_files(
        [
            args.file_paths_json,
            master_item_path,
            master_index_path,
            TRAIN_SCRIPT_PATH,
        ]
        + ([] if args.skip_eval else [master_info_path, EVALUATE_SCRIPT_PATH, METRIC_SCRIPT_PATH])
    )

    valid_behaviors = [
        behavior.strip() for behavior in args.behaviors.split(",") if behavior.strip()
    ]
    item_sid_map, item_title_map, valid_items = load_master_lookups(
        master_item_path, master_index_path
    )
    if item_sid_map is None:
        raise RuntimeError("Failed to load master item lookups")

    with open(args.file_paths_json, "r", encoding="utf-8") as handle:
        path_config = json.load(handle)
    interaction_files = path_config["interaction_files"]
    item_files = path_config["item_files"]

    model_name = os.path.basename(os.path.normpath(args.base_model))
    current_model_checkpoint = args.base_model
    current_ewc_state = None
    all_metrics = {}
    metrics_summary_path = os.path.join(
        args.negative_eval_saved_path,
        f"metrics_summary_neg_{args.negative_sample_rate}.json",
    )

    run_id = args.wandb_run_id or uuid.uuid4().hex[:8]
    env = os.environ.copy()
    env.update(
        {
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
            "WANDB_NAME": args.wandb_run_name,
        }
    )
    print(
        f"Starting streaming EWC-LoRA: tasks={args.tasks}, GPUs={args.num_gpus}, "
        f"lambda={args.ewc_lambda:g}, gamma={args.ewc_gamma:g}, run_id={run_id}"
    )

    rounds_completed = 0
    for round_index in range(args.start_day, len(interaction_files)):
        if round_index + args.window_days > len(item_files):
            break
        if args.max_rounds > 0 and rounds_completed >= args.max_rounds:
            break

        interaction_file = interaction_files[round_index]
        round_output_dir = os.path.join(
            args.saved_path, model_name, f"streaming_round_{round_index}"
        )
        training_output_dir = os.path.join(round_output_dir, "model_checkpoint")
        test_path = os.path.join(round_output_dir, "test_T.csv")
        state_candidate = os.path.join(training_output_dir, "ewc_state.pt")
        checkpoint_complete = (
            os.path.isfile(os.path.join(training_output_dir, "config.json"))
            and os.path.isfile(state_candidate)
        )

        os.makedirs(round_output_dir, exist_ok=True)
        print(
            f"\n--- [Round {round_index}] "
            f"{os.path.basename(interaction_file)} ---"
        )

        if checkpoint_complete and not args.force_retrain:
            print(f"Using completed EWC-LoRA checkpoint: {training_output_dir}")
            current_model_checkpoint = training_output_dir
            current_ewc_state = state_candidate
            if not args.skip_eval and not os.path.isfile(test_path):
                _, test_path = prepare_data_for_day(
                    interaction_file,
                    round_output_dir,
                    item_sid_map,
                    item_title_map,
                    valid_items,
                    valid_behaviors=valid_behaviors,
                )
        else:
            print(f"[Round {round_index}] Preparing data")
            train_path, test_path = prepare_data_for_day(
                interaction_file,
                round_output_dir,
                item_sid_map,
                item_title_map,
                valid_items,
                valid_behaviors=valid_behaviors,
            )
            if train_path is None:
                print(f"[Round {round_index}] No training data; skipping round")
                continue

            train_command = [
                "accelerate",
                "launch",
                f"--num_processes={args.num_gpus}",
                TRAIN_SCRIPT_PATH,
                "--base_model",
                current_model_checkpoint,
                "--train_file",
                train_path,
                "--eval_file",
                test_path,
                "--sid_index_path",
                master_index_path,
                "--item_meta_path",
                master_item_path,
                "--category",
                args.category,
                "--output_dir",
                training_output_dir,
                "--num_epochs",
                str(args.num_epochs),
                "--learning_rate",
                str(args.learning_rate),
                "--batch_size",
                str(args.batch_size),
                "--micro_batch_size",
                str(args.micro_batch_size),
                "--eval_strategy",
                "no",
                "--save_strategy",
                "no",
                "--save_total_limit",
                "1",
                "--report_to",
                args.report_to,
                "--tasks",
                args.tasks,
                "--wandb_project",
                args.wandb_project,
                "--wandb_run_name",
                args.wandb_run_name,
                "--target_modules",
                args.target_modules,
                "--lora_rank",
                str(args.lora_rank),
                "--lora_alpha",
                str(args.lora_alpha),
                "--lora_dropout",
                str(args.lora_dropout),
                "--train_semantic_embeddings",
                str(not args.freeze_semantic_embeddings),
                "--ewc_lambda",
                str(args.ewc_lambda),
                "--ewc_gamma",
                str(args.ewc_gamma),
                "--fisher_max_batches",
                str(args.fisher_max_batches),
                "--fisher_batch_size",
                str(args.fisher_batch_size),
            ]
            if current_ewc_state:
                train_command.extend(["--ewc_state_path", current_ewc_state])

            started = time.time()
            train_process = _run(train_command, env=env)
            print(
                f"[Round {round_index}] Training process finished in "
                f"{time.time() - started:.2f}s"
            )
            if train_process.returncode != 0:
                raise RuntimeError(
                    f"Round {round_index} training failed with exit code "
                    f"{train_process.returncode}"
                )
            if not os.path.isfile(state_candidate):
                raise RuntimeError(
                    f"Round {round_index} did not produce {state_candidate}"
                )
            current_model_checkpoint = training_output_dir
            current_ewc_state = state_candidate

        rounds_completed += 1
        if args.skip_eval:
            continue

        round_eval_dir = os.path.join(
            args.negative_eval_saved_path,
            model_name,
            f"streaming_round_{round_index}",
        )
        os.makedirs(round_eval_dir, exist_ok=True)
        result_name = f"results_neg_{args.negative_sample_rate}.json"
        result_path = os.path.join(round_eval_dir, result_name)

        if not os.path.isfile(result_path) or args.force_eval:
            eval_command = [
                "accelerate",
                "launch",
                f"--num_processes={args.num_gpus}",
                EVALUATE_SCRIPT_PATH,
                "--base_model",
                current_model_checkpoint,
                "--test_data_path",
                test_path,
                "--info_file",
                master_info_path,
                "--result_json_data",
                result_path,
                "--category",
                args.category,
                "--batch_size",
                str(args.eval_batch_size),
                "--num_beams",
                str(args.num_beams),
                "--negative_sample_rate",
                str(args.negative_sample_rate),
                "--min_diversity",
                str(args.min_diversity),
            ]
            eval_process = _run(eval_command)
            if eval_process.returncode != 0:
                raise RuntimeError(
                    f"Round {round_index} evaluation failed with exit code "
                    f"{eval_process.returncode}"
                )
            if not _merge_distributed_results(result_path):
                raise FileNotFoundError(
                    f"Evaluation result was not produced: {result_path}"
                )
        else:
            print(f"Using existing evaluation result: {result_path}")

        metric_process = _run(
            [
                sys.executable,
                METRIC_SCRIPT_PATH,
                "--path",
                result_path,
                "--item_path",
                master_info_path,
            ],
            capture_output=True,
        )
        if metric_process.returncode != 0:
            raise RuntimeError(
                f"metric.py failed for round {round_index}: {metric_process.stderr}"
            )

        metrics = parse_metric_output(metric_process.stdout, TOPK_LIST_FOR_METRIC)
        all_metrics[os.path.basename(interaction_file)] = metrics
        os.makedirs(os.path.dirname(metrics_summary_path), exist_ok=True)
        with open(metrics_summary_path, "w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2, ensure_ascii=False)
        print(f"[Round {round_index}] Metrics: {metrics}")

    print(f"\nStreaming EWC-LoRA finished; completed rounds={rounds_completed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming EWC-LoRA trainer")
    parser.add_argument("--file_paths_json", default=FILE_PATHS_JSON)
    parser.add_argument("--master_item_json_path", default=MASTER_ITEM_JSON_PATH)
    parser.add_argument("--master_index_json_path", default=MASTER_INDEX_JSON_PATH)
    parser.add_argument("--master_info_txt_path", default=MASTER_INFO_TXT_PATH)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--saved_path", default=SAVED_PATH)
    parser.add_argument(
        "--negative_eval_saved_path", default=NEGATIVE_EVAL_SAVED_PATH
    )
    parser.add_argument("--behaviors", default="is_click")
    parser.add_argument("--category", default=CATEGORY_NAME)
    parser.add_argument("--tasks", default=TASKS_TO_TRAIN)
    parser.add_argument("--num_gpus", type=int, default=NUM_GPUS)
    parser.add_argument("--start_day", type=int, default=START_DAY)
    parser.add_argument("--window_days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--max_rounds", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--micro_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--target_modules", default="q_proj,v_proj")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--freeze_semantic_embeddings", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=1e7)
    parser.add_argument("--ewc_gamma", type=float, default=1.0)
    parser.add_argument("--fisher_max_batches", type=int, default=100)
    parser.add_argument("--fisher_batch_size", type=int, default=8)

    parser.add_argument("--wandb_project", default=WANDB_PROJECT)
    parser.add_argument("--wandb_run_name", default=WANDB_RUN_NAME)
    parser.add_argument("--wandb_run_id", default="")
    parser.add_argument("--report_to", default="wandb")

    parser.add_argument("--negative_sample_rate", type=int, default=1)
    parser.add_argument("--min_diversity", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
