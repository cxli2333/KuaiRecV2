"""Backtest every streaming checkpoint on one fixed first-day test set.

Place this script in ``MiniOneRec/streaming``. By default it scans the original
``streaming_data/train3/Qwen2.5-0.5B`` run, evaluates every round checkpoint on
``streaming_round_0/test_T.csv``, runs ``metric.py``, and writes a consolidated
JSON/CSV report. Existing valid per-round outputs are reused.
"""

import argparse
import csv
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CHECKPOINT_ROOT = os.path.join(
    PROJECT_ROOT, "streaming_data", "train3", "Qwen2.5-0.5B"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "streaming_data", "backtest_first_day", "train3_Qwen2.5-0.5B"
)
DEFAULT_EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "backtest_streaming_eval.py")
DEFAULT_METRIC_SCRIPT = os.path.join(PROJECT_ROOT, "metric.py")
DEFAULT_INFO_FILE = os.path.join(
    PROJECT_ROOT, "streaming_data", "info", "master.info.txt"
)
DEFAULT_TOPK = [1, 5, 10, 20, 50]
ROUND_PATTERN = re.compile(r"^streaming_round_(\d+)$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def require_file(path: str, description: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def require_dir(path: str, description: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def is_model_checkpoint(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def checkpoint_step(path: str) -> int:
    match = CHECKPOINT_PATTERN.match(os.path.basename(path))
    return int(match.group(1)) if match else -1


def select_round_checkpoint(round_dir: str) -> Optional[str]:
    model_dir = os.path.join(round_dir, "model_checkpoint")
    preferred = [
        model_dir,
        os.path.join(model_dir, "final_checkpoint"),
        os.path.join(round_dir, "final_checkpoint"),
        round_dir,
    ]
    for candidate in preferred:
        if is_model_checkpoint(candidate):
            return os.path.abspath(candidate)

    nested = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    nested.extend(glob.glob(os.path.join(round_dir, "checkpoint-*")))
    valid = [path for path in nested if is_model_checkpoint(path)]
    if valid:
        return os.path.abspath(max(valid, key=checkpoint_step))
    return None


def discover_round_checkpoints(
    checkpoint_root: str,
    start_round: int,
    end_round: int,
) -> List[Tuple[int, str]]:
    discovered = []
    for name in os.listdir(checkpoint_root):
        match = ROUND_PATTERN.match(name)
        if not match:
            continue
        round_index = int(match.group(1))
        if round_index < start_round or (end_round >= 0 and round_index > end_round):
            continue
        checkpoint = select_round_checkpoint(os.path.join(checkpoint_root, name))
        if checkpoint is None:
            print(f"[Skip] round={round_index}: no checkpoint with config.json", flush=True)
            continue
        discovered.append((round_index, checkpoint))
    discovered.sort(key=lambda value: value[0])
    if not discovered:
        raise RuntimeError(
            f"No usable streaming_round_N checkpoints found under {checkpoint_root}"
        )
    return discovered


def valid_json_list(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return isinstance(value, list) and len(value) > 0
    except (OSError, json.JSONDecodeError):
        return False


def merge_distributed_parts(result_path: str) -> bool:
    if valid_json_list(result_path):
        return True
    stem = os.path.splitext(result_path)[0]
    part_paths = glob.glob(f"{stem}_part_*.json")
    if not part_paths:
        return False

    def part_rank(path: str) -> int:
        match = re.search(r"_part_(\d+)\.json$", path)
        return int(match.group(1)) if match else sys.maxsize

    merged = []
    for part_path in sorted(part_paths, key=part_rank):
        with open(part_path, "r", encoding="utf-8") as handle:
            part = json.load(handle)
        if not isinstance(part, list):
            raise ValueError(f"Distributed result is not a list: {part_path}")
        merged.extend(part)
    if not merged:
        return False
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
    print(f"[Merge] {len(merged):,} predictions -> {result_path}", flush=True)
    return True


def parse_metric_stdout(stdout: str, topk: Sequence[int]) -> Tuple[Dict, Dict]:
    match = re.search(
        r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___", stdout, re.DOTALL
    )
    if not match:
        raise ValueError(
            "metric.py did not emit ___METRIC_JSON_START___/END markers; "
            f"stdout tail: {stdout[-1000:]}"
        )
    raw = json.loads(match.group(1).strip())
    flattened = {}
    for metric_name, value in raw.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                if index < len(topk) and isinstance(item, (int, float)):
                    flattened[f"{metric_name}@{topk[index]}"] = float(item)
        elif isinstance(value, (int, float)):
            flattened[metric_name] = float(value)
    return raw, flattened


def run_command(command: Sequence[str], dry_run: bool = False, capture: bool = False):
    print("[CMD] " + shlex.join(list(command)), flush=True)
    if dry_run:
        return None
    return subprocess.run(
        list(command),
        check=False,
        text=capture,
        capture_output=capture,
    )


def load_metrics(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict) and isinstance(value.get("flattened_metrics"), dict):
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return None


def write_aggregate(records: List[Dict], output_dir: str) -> None:
    json_path = os.path.join(output_dir, "backtest_first_day_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    metric_names = sorted(
        {
            name
            for record in records
            for name in record.get("flattened_metrics", {}).keys()
        }
    )
    csv_path = os.path.join(output_dir, "backtest_first_day_summary.csv")
    fields = [
        "round_index",
        "checkpoint",
        "test_file",
        "prediction_count",
        "status",
        "error",
        *metric_names,
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            row.update(record.get("flattened_metrics", {}))
            writer.writerow(row)
    print(f"[Summary] JSON={json_path}", flush=True)
    print(f"[Summary] CSV={csv_path}", flush=True)


def prediction_count(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return len(value) if isinstance(value, list) else 0


def main(args: argparse.Namespace) -> None:
    checkpoint_root = require_dir(args.checkpoint_root, "Checkpoint root")
    eval_script = require_file(args.eval_script, "Evaluation script")
    metric_script = require_file(args.metric_script, "Metric script")
    info_file = require_file(args.info_file, "Master info file")
    test_file = args.test_file or os.path.join(
        checkpoint_root, f"streaming_round_{args.test_round}", "test_T.csv"
    )
    test_file = require_file(test_file, "Fixed backtest file")
    checkpoints = discover_round_checkpoints(
        checkpoint_root, args.start_round, args.end_round
    )
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"[Backtest] checkpoints={len(checkpoints)}, fixed_test={test_file}, "
        f"rounds={checkpoints[0][0]}..{checkpoints[-1][0]}",
        flush=True,
    )

    records = []
    for position, (round_index, checkpoint) in enumerate(checkpoints, start=1):
        round_dir = os.path.join(output_dir, f"streaming_round_{round_index}")
        os.makedirs(round_dir, exist_ok=True)
        result_path = os.path.join(
            round_dir, f"predictions_on_round_{args.test_round}.json"
        )
        metrics_path = os.path.join(round_dir, "metrics.json")
        print(
            f"\n[Round {round_index}] {position}/{len(checkpoints)} checkpoint={checkpoint}",
            flush=True,
        )
        record = {
            "round_index": round_index,
            "checkpoint": checkpoint,
            "test_round": args.test_round,
            "test_file": test_file,
            "prediction_count": 0,
            "status": "pending",
            "error": "",
            "flattened_metrics": {},
        }
        try:
            if args.force_eval or not valid_json_list(result_path):
                if args.force_eval:
                    for stale_path in [result_path, *glob.glob(
                        f"{os.path.splitext(result_path)[0]}_part_*.json"
                    )]:
                        if os.path.isfile(stale_path):
                            os.remove(stale_path)
                eval_command = [
                    args.accelerate,
                    "launch",
                    f"--num_processes={args.num_gpus}",
                ]
                if args.main_process_port > 0:
                    eval_command.append(f"--main_process_port={args.main_process_port}")
                eval_command.extend(
                    [
                        eval_script,
                        "--base_model",
                        checkpoint,
                        "--test_data_path",
                        test_file,
                        "--info_file",
                        info_file,
                        "--result_json_data",
                        result_path,
                        "--category",
                        args.category,
                        "--seed",
                        str(args.seed),
                        "--K",
                        str(args.k),
                        "--batch_size",
                        str(args.batch_size),
                        "--length_penalty",
                        str(args.length_penalty),
                        "--max_new_tokens",
                        str(args.max_new_tokens),
                        "--num_beams",
                        str(args.num_beams),
                        "--negative_sample_rate",
                        str(args.negative_sample_rate),
                        "--min_diversity",
                        str(args.min_diversity),
                    ]
                )
                process = run_command(eval_command, dry_run=args.dry_run)
                if args.dry_run:
                    record["status"] = "dry_run"
                    records.append(record)
                    continue
                if process.returncode != 0 and not merge_distributed_parts(result_path):
                    raise RuntimeError(
                        f"streaming_eval.py exited with code {process.returncode}"
                    )
            if not merge_distributed_parts(result_path):
                raise RuntimeError(f"No valid predictions produced at {result_path}")
            record["prediction_count"] = prediction_count(result_path)

            existing = (
                None
                if args.force_eval or args.force_metric
                else load_metrics(metrics_path)
            )
            if existing is None:
                metric_command = [
                    args.python_executable,
                    metric_script,
                    "--path",
                    result_path,
                    "--item_path",
                    info_file,
                ]
                metric_process = run_command(metric_command, capture=True)
                if metric_process.returncode != 0:
                    raise RuntimeError(
                        f"metric.py exited with code {metric_process.returncode}: "
                        f"{metric_process.stderr[-2000:]}"
                    )
                print(metric_process.stdout, end="", flush=True)
                raw_metrics, flattened = parse_metric_stdout(
                    metric_process.stdout, args.topk
                )
                existing = {
                    **record,
                    "status": "complete",
                    "raw_metrics": raw_metrics,
                    "flattened_metrics": flattened,
                }
                with open(metrics_path, "w", encoding="utf-8") as handle:
                    json.dump(existing, handle, ensure_ascii=False, indent=2)
            record.update(existing)
            record["status"] = "complete"
            record["error"] = ""
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[Round {round_index}] ERROR: {exc}", flush=True)
            if args.fail_fast:
                records.append(record)
                write_aggregate(records, output_dir)
                raise
        records.append(record)
        write_aggregate(records, output_dir)

    write_aggregate(records, output_dir)
    completed = sum(record["status"] == "complete" for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    print(
        f"[Done] complete={completed}, failed={failed}, total={len(records)}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate every streaming checkpoint on the fixed first-day test set"
    )
    parser.add_argument("--checkpoint_root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--test_round", type=int, default=0)
    parser.add_argument("--test_file", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_script", default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--metric_script", default=DEFAULT_METRIC_SCRIPT)
    parser.add_argument("--info_file", default=DEFAULT_INFO_FILE)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--accelerate", default="accelerate")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--main_process_port", type=int, default=0)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument("--end_round", type=int, default=-1)
    parser.add_argument("--category", default="Streaming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--negative_sample_rate", type=float, default=1.0)
    parser.add_argument("--min_diversity", type=int, default=5)
    parser.add_argument("--topk", nargs="+", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--force_metric", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
