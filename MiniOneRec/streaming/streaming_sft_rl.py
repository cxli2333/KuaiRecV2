import json
import os
import subprocess
import uuid
import argparse
import re
import sys
import time
from datetime import datetime, timedelta

from streaming_data_prep import load_master_lookups, prepare_rl_data_with_sampling, prepare_data_for_day

# --- Path configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

FILE_PATHS_JSON = os.path.join(SCRIPT_DIR, "Full_Month_file_paths_relative.json")
MASTER_INFO_DIR = os.path.join(PARENT_DIR, "streaming_data", "info")

MASTER_ITEM_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.item.json")
MASTER_INDEX_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.index.json")
MASTER_INFO_TXT_PATH = os.path.join(MASTER_INFO_DIR, "master.info.txt")

RL_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "..", "streaming_rl_run.py")
SFT_SCRIPT_PATH = os.path.join(PARENT_DIR, "sft.py")
EVAL_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "streaming_eval.py")
METRIC_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "..", "metric.py")

WANDB_PROJECT = "MiniOneRec-Streaming-SFT-RL"
BASE_MODEL = "../../../models/Qwen2.5-0.5B"
CATEGORY_NAME = "Streaming"
WINDOW_DAYS = 7

PYTHON_EXECUTABLE = sys.executable
TOPK_LIST_FOR_METRIC = [1, 5, 10, 20, 50]


def extract_yyyymmdd_from_path(path_str):
    m = re.search(r"(\d{8})\.tsv$", str(path_str))
    if m:
        return m.group(1)
    m = re.search(r"(\d{8})", str(path_str))
    if m:
        return m.group(1)
    return None


def get_item_files_for_interaction_date(interaction_file, item_files, window_days=7):
    interaction_date_str = extract_yyyymmdd_from_path(interaction_file)
    if interaction_date_str is None:
        raise ValueError(f"Cannot extract date from interaction file: {interaction_file}")

    interaction_date = datetime.strptime(interaction_date_str, "%Y%m%d")
    start_date = interaction_date - timedelta(days=window_days - 1)

    selected = []
    for fp in item_files:
        item_date_str = extract_yyyymmdd_from_path(fp)
        if item_date_str is None:
            continue
        item_date = datetime.strptime(item_date_str, "%Y%m%d")
        if start_date <= item_date <= interaction_date:
            selected.append(fp)

    selected = sorted(selected, key=lambda x: extract_yyyymmdd_from_path(x))
    return selected


def is_valid_ckpt(ckpt_dir: str) -> bool:
    if not ckpt_dir:
        return False
    return os.path.exists(os.path.join(ckpt_dir, "config.json"))


def is_eval_done(eval_out_path: str) -> bool:
    if not eval_out_path:
        return False
    return os.path.exists(eval_out_path) and os.path.getsize(eval_out_path) > 0


def get_metrics_path(eval_out_path: str) -> str:
    if eval_out_path.endswith(".json"):
        return eval_out_path[:-5] + ".metrics.json"
    return eval_out_path + ".metrics.json"


def is_metrics_done(metrics_path: str) -> bool:
    if not metrics_path:
        return False
    return os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0


def parse_metric_output(stdout_str, topk_list):
    """Parse the output of metric.py."""
    metrics = {}
    try:
        match = re.search(r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___", stdout_str, re.DOTALL)

        if not match:
            print("Error: Unable to load metric JSON")
            return metrics

        json_str = match.group(1).strip()
        raw_metrics = json.loads(json_str)

        for i, k in enumerate(topk_list):
            if "NDCG" in raw_metrics and i < len(raw_metrics["NDCG"]):
                metrics[f"NDCG@{k}"] = raw_metrics["NDCG"][i]
            if "HR" in raw_metrics and i < len(raw_metrics["HR"]):
                metrics[f"HR@{k}"] = raw_metrics["HR"][i]
            if "Recall" in raw_metrics and i < len(raw_metrics["Recall"]):
                metrics[f"Recall@{k}"] = raw_metrics["Recall"][i]
            if "MRR" in raw_metrics and i < len(raw_metrics["MRR"]):
                metrics[f"MRR@{k}"] = raw_metrics["MRR"][i]

        metrics["_raw_all_metrics"] = raw_metrics

    except Exception as e:
        print(f"Warning: Failed to parse metric JSON: {e}")

    return metrics


def run_eval(
    model_ckpt_dir: str,
    test_path: str,
    eval_out_path: str,
    category: str,
    args,
):
    """Run evaluation."""
    if not os.path.exists(EVAL_SCRIPT_PATH):
        raise FileNotFoundError(f"EVAL script not found: {EVAL_SCRIPT_PATH}")
    if not is_valid_ckpt(model_ckpt_dir):
        raise FileNotFoundError(f"Model checkpoint invalid: {model_ckpt_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Eval test file not found: {test_path}")

    os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)

    eval_command = [
        "accelerate", "launch", f"--num_processes={args.eval_num_gpus}",
        EVAL_SCRIPT_PATH,
        "--base_model", model_ckpt_dir,
        "--test_data_path", test_path,
        "--result_json_data", eval_out_path,
        "--category", category,
        "--seed", str(args.seed),
        "--K", str(args.eval_k),
        "--batch_size", str(args.eval_batch_size),
        "--length_penalty", str(args.eval_length_penalty),
        "--max_new_tokens", str(args.eval_max_new_tokens),
        "--num_beams", str(args.eval_num_beams),
        "--negative_sample_rate", str(args.eval_negative_sample_rate),
        "--min_diversity", str(args.eval_min_diversity),
    ]

    print(f"[EVAL CMD] {' '.join(eval_command)}\n")

    process_eval = subprocess.run(eval_command, check=False)

    if process_eval.returncode != 0:
        print(f"Warning: streaming_eval.py failed with code={process_eval.returncode}")
        return False

    if not is_eval_done(eval_out_path):
        print(f"Warning: Eval finished but output missing/empty: {eval_out_path}")
        return False

    print(f"[EVAL] Done. Output: {eval_out_path}")
    return True


def run_metric(eval_out_path: str, item_path: str):
     
    if not os.path.exists(METRIC_SCRIPT_PATH):
        raise FileNotFoundError(f"Metric script not found: {METRIC_SCRIPT_PATH}")
    if not os.path.exists(eval_out_path):
        raise FileNotFoundError(f"Eval json not found: {eval_out_path}")
    if not os.path.exists(item_path):
        raise FileNotFoundError(f"Metric item_path not found: {item_path}")

    metric_command = [
        PYTHON_EXECUTABLE,
        METRIC_SCRIPT_PATH,
        "--path", eval_out_path,
        "--item_path", item_path,
    ]

    print(f"[METRIC CMD] {' '.join(metric_command)}\n")

    process_metrics = subprocess.run(
        metric_command,
        capture_output=True,
        text=True,
        check=False
    )

    print(process_metrics.stdout)
    if process_metrics.stderr:
        print(process_metrics.stderr)

    if process_metrics.returncode != 0:
        print(f"Warning: metric.py failed with code={process_metrics.returncode}")
        return None, {}

    metrics = parse_metric_output(process_metrics.stdout, TOPK_LIST_FOR_METRIC)
    metrics_path = get_metrics_path(eval_out_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[METRIC] Saved metrics to: {metrics_path}")
    return metrics_path, metrics


def ensure_eval_and_metrics(
    model_ckpt_dir: str,
    test_path: str,
    eval_out_path: str,
    category: str,
    args,
):
     
    if not is_eval_done(eval_out_path):
        print(f"[EVAL] Running eval...")
        success = run_eval(
            model_ckpt_dir=model_ckpt_dir,
            test_path=test_path,
            eval_out_path=eval_out_path,
            category=category,
            args=args,
        )
        if not success:
            return None
    else:
        print(f"[EVAL] Eval json exists, skip: {eval_out_path}")

    metrics_path = get_metrics_path(eval_out_path)
    if not is_metrics_done(metrics_path):
        print(f"[METRIC] Running metric.py...")
        metrics_path, metrics = run_metric(
            eval_out_path=eval_out_path,
            item_path=MASTER_INFO_TXT_PATH,
        )
        if metrics:
            print(f"[METRIC] Summary: {metrics}")
        return metrics
    else:
        print(f"[METRIC] Metrics already exist, skip: {metrics_path}")
        with open(metrics_path, "r") as f:
            return json.load(f)


def run_rl_training(
    round_idx,
    interaction_file,
    item_files,
    current_model_checkpoint,
    output_dir,
    data_dir,
    args,
    env,
):
    
    print(f"\n{'='*60}")
    print(f"[Round {round_idx}] RL Training Phase")
    print(f"{'='*60}")

    rl_output_dir = os.path.join(output_dir, "rl_checkpoint")

     
    if is_valid_ckpt(rl_output_dir):
        print(f"[Round {round_idx}] RL checkpoint already exists, skipping training: {rl_output_dir}")
        return rl_output_dir

    train_day = os.path.splitext(os.path.basename(interaction_file))[0]

    
    train_rl_path = os.path.join(data_dir, "train_rl_sampled.tsv")
    test_path = os.path.join(data_dir, "test_T.csv")

    if (not os.path.exists(train_rl_path)) or (not os.path.exists(test_path)):
        print(f"[Round {round_idx}] Preparing RL data...")
        train_rl_path_ret, test_path_ret = prepare_rl_data_with_sampling(
            interaction_file=interaction_file,
            master_item_path=MASTER_ITEM_JSON_PATH,
            master_index_path=MASTER_INDEX_JSON_PATH,
            output_dir=data_dir,
            train_day=train_day,
            sample_rate=args.rl_sample_rate,
            max_history_len=args.max_history_len,
            valid_behaviors=args.valid_behaviors,
        )
        if train_rl_path_ret is None:
            print(f"[Round {round_idx}] No RL data available, skipping RL.")
            return None
        train_rl_path = train_rl_path_ret
        test_path = test_path_ret
    else:
        print(f"[Round {round_idx}] Reusing existing RL data")

     
    current_window_item_files = get_item_files_for_interaction_date(
        interaction_file=interaction_file,
        item_files=item_files,
        window_days=WINDOW_DAYS,
    )
    item_files_arg = ",".join(current_window_item_files)

    os.makedirs(rl_output_dir, exist_ok=True)

    
    rl_command = (
        f"accelerate launch "
        f"\"{RL_SCRIPT_PATH}\" "
        f"--train_file \"{train_rl_path}\" "
        f"--eval_file \"{test_path}\" "
        f"--model_path \"{current_model_checkpoint}\" "
        f"--output_dir \"{rl_output_dir}\" "
        f"--item_files \"{item_files_arg}\" "
        f"--reward_type {args.reward_type} "
        f"--num_train_epochs {args.rl_num_epochs} "
        f"--train_batch_size {args.rl_train_batch_size} "
        f"--eval_batch_size {args.rl_eval_batch_size} "
        f"--gradient_accumulation_steps {args.gradient_accumulation_steps} "
        f"--temperature {args.temperature} "
        f"--num_generations {args.num_generations} "
        f"--learning_rate {args.rl_learning_rate} "
        f"--beta {args.beta} "
        f"--w_exact {args.w_exact} "
        f"--w_prefix {args.w_prefix} "
        f"--w_sem {args.w_sem} "
        f"--subtree_sample_ratio {args.subtree_sample_ratio} "
        f"--w_pos {args.w_pos} "
        f"--w_inter {args.w_inter} "
        f"--w_neg {args.w_neg} "
        f"--positive_behavior_cols {' '.join(args.positive_behavior_cols)} "
        f"--interaction_behavior_cols {' '.join(args.interaction_behavior_cols)} "
        f"--negative_behavior_cols {' '.join(args.negative_behavior_cols)} "
    )

    if args.add_gt:
        rl_command += "--add_gt "
    if args.beam_search:
        rl_command += "--beam_search "
    if args.test_during_training:
        rl_command += "--test_during_training "
    if args.dynamic_sampling:
        rl_command += "--dynamic_sampling "
    if args.sync_ref_model:
        rl_command += "--sync_ref_model "
    if args.dapo:
        rl_command += "--dapo "
    if args.gspo:
        rl_command += "--gspo "

    print(f"[Round {round_idx}] Running RL training...")
    print(f"[CMD] {rl_command}\n")

    result = subprocess.run(rl_command, shell=True, env=env)
    if result.returncode != 0:
        print(f"[Round {round_idx}] RL training failed!")
        return None

    if not is_valid_ckpt(rl_output_dir):
        print(f"[Round {round_idx}] RL checkpoint not found!")
        return None

    print(f"[Round {round_idx}] RL training completed. Checkpoint: {rl_output_dir}")
    return rl_output_dir


def run_sft_training(
    round_idx,
    interaction_file,
    current_model_checkpoint,
    output_dir,
    data_dir,
    args,
    env,
    item_sid_map,
    item_title_map,
    valid_items,
):
    
    print(f"\n{'='*60}")
    print(f"[Round {round_idx}] SFT Training Phase")
    print(f"{'='*60}")

    sft_output_dir = os.path.join(output_dir, "sft_checkpoint")

    if is_valid_ckpt(sft_output_dir):
        print(f"[Round {round_idx}] SFT checkpoint already exists, skipping training: {sft_output_dir}")
        return sft_output_dir

     
    train_sft_path = os.path.join(data_dir, "train_sft.tsv")
    test_sft_path = os.path.join(data_dir, "test_sft.csv")

    if (not os.path.exists(train_sft_path)) or (not os.path.exists(test_sft_path)):
        print(f"[Round {round_idx}] Preparing SFT data...")
        train_sft_path_ret, test_sft_path_ret = prepare_data_for_day(
            interaction_file,
            data_dir,
            item_sid_map,
            item_title_map,
            valid_items,
            valid_behaviors=args.valid_behaviors,
        )
        if train_sft_path_ret is None:
            print(f"[Round {round_idx}] No SFT data available, skipping SFT.")
            return None
        train_sft_path = train_sft_path_ret
        test_sft_path = test_sft_path_ret
    else:
        print(f"[Round {round_idx}] Reusing existing SFT data")

    os.makedirs(sft_output_dir, exist_ok=True)

     
    sft_command = [
        "accelerate", "launch", f"--num_processes={args.num_gpus}",
        SFT_SCRIPT_PATH,
        "--base_model", current_model_checkpoint,
        "--train_file", train_sft_path,
        "--eval_file", test_sft_path,
        "--sid_index_path", MASTER_INDEX_JSON_PATH,
        "--item_meta_path", MASTER_ITEM_JSON_PATH,
        "--category", CATEGORY_NAME,
        "--output_dir", sft_output_dir,
        "--num_epochs", str(args.sft_num_epochs),
        "--learning_rate", str(args.sft_learning_rate),
        "--eval_strategy", "no",
        "--save_strategy", "epoch",
        "--save_total_limit", "1",
        "--report_to", "wandb",
        "--tasks", "streaming_sid2sid",
        "--wandb_project", WANDB_PROJECT,
    ]

    print(f"[Round {round_idx}] Running SFT training...")
    print(f"[CMD] {' '.join(sft_command)}\n")

    result = subprocess.run(sft_command, env=env)
    if result.returncode != 0:
        print(f"[Round {round_idx}] SFT training failed!")
        return None

    if not is_valid_ckpt(sft_output_dir):
        print(f"[Round {round_idx}] SFT checkpoint not found!")
        return None

    print(f"[Round {round_idx}] SFT training completed. Checkpoint: {sft_output_dir}")
    return sft_output_dir


def main(args):
    item_sid_map, item_title_map, valid_items = load_master_lookups(
        MASTER_ITEM_JSON_PATH, MASTER_INDEX_JSON_PATH
    )
    if item_sid_map is None:
        return

    try:
        with open(args.file_paths_json, "r") as f:
            paths = json.load(f)
        interaction_files = paths["interaction_files"]
        item_files = paths["item_files"]
    except Exception as e:
        print(f"Error loading {args.file_paths_json}: {e}")
        return

    model_name = os.path.basename(BASE_MODEL)
    sft_model_dir = args.sft_model_dir
    sft_round = args.sft_round

   
    sft_ckpt_path = os.path.join(sft_model_dir, f"streaming_round_{sft_round}", "model_checkpoint")
    if not is_valid_ckpt(sft_ckpt_path):
        print(f"Error: SFT checkpoint not found at {sft_ckpt_path}")
        return

    print(f"\n{'#'*70}")
    print(f"# Alternating SFT-RL Training with Eval")
    print(f"{'#'*70}")
    print(f"Starting from SFT round {sft_round}")
    print(f"Initial checkpoint: {sft_ckpt_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Auto eval: {args.auto_eval}")
    print(f"{'#'*70}\n")

    base_env = os.environ.copy()
    base_env["WANDB_PROJECT"] = WANDB_PROJECT
    base_env["WANDB_RESUME"] = "allow"

    current_model_checkpoint = sft_ckpt_path
    output_root = args.output_dir
    all_metrics = {}

     
    for i in range(sft_round, len(interaction_files)):
        if i + WINDOW_DAYS > len(item_files):
            break

        interaction_file = interaction_files[i]
        round_output_dir = os.path.join(output_root, model_name, f"round_{i}")
        data_dir = os.path.join(round_output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"# Round {i}: {os.path.basename(interaction_file)}")
        print(f"{'#'*60}")

        
        test_path = os.path.join(data_dir, "test_T.csv")

        
        sft_ckpt = run_sft_training(
            round_idx=i,
            interaction_file=interaction_file,
            current_model_checkpoint=rl_ckpt,
            output_dir=round_output_dir,
            data_dir=data_dir,
            args=args,
            env=base_env,
            item_sid_map=item_sid_map,
            item_title_map=item_title_map,
            valid_items=valid_items,
        )

        if sft_ckpt is None:
            print(f"[Round {i}] SFT training failed, using RL checkpoint for next round")
            current_model_checkpoint = rl_ckpt
        else:
            current_model_checkpoint = sft_ckpt

             
            if args.auto_eval:
                print(f"\n[Round {i}] Running eval after SFT training...")
                sft_eval_out_path = os.path.join(round_output_dir, f"eval_sft_round_{i}.json")
                try:
                    metrics = ensure_eval_and_metrics(
                        model_ckpt_dir=sft_ckpt,
                        test_path=test_path,
                        eval_out_path=sft_eval_out_path,
                        category=CATEGORY_NAME,
                        args=args,
                    )
                    if metrics:
                        all_metrics[f"round_{i}_sft"] = metrics
                except Exception as e:
                    print(f"[Round {i}] SFT eval failed: {e}")

       
        rl_ckpt = run_rl_training(
            round_idx=i,
            interaction_file=interaction_file,
            item_files=item_files,
            current_model_checkpoint=current_model_checkpoint,
            output_dir=round_output_dir,
            data_dir=data_dir,
            args=args,
            env=base_env,
        )

        if rl_ckpt is None:
            print(f"[Round {i}] RL training failed, skipping to next round")
            continue

         
        if args.auto_eval:
            print(f"\n[Round {i}] Running eval after RL training...")
            rl_eval_out_path = os.path.join(round_output_dir, f"eval_rl_round_{i}.json")
            try:
                metrics = ensure_eval_and_metrics(
                    model_ckpt_dir=rl_ckpt,
                    test_path=test_path,
                    eval_out_path=rl_eval_out_path,
                    category=CATEGORY_NAME,
                    args=args,
                )
                if metrics:
                    all_metrics[f"round_{i}_rl"] = metrics
            except Exception as e:
                print(f"[Round {i}] RL eval failed: {e}")

        
        if all_metrics:
            summary_path = os.path.join(output_root, "metrics_summary.json")
            with open(summary_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"[Round {i}] Metrics summary saved to: {summary_path}")

        print(f"[Round {i}] Completed. Next checkpoint: {current_model_checkpoint}")

    print(f"\n{'#'*70}")
    print(f"# Training Summary")
    print(f"{'#'*70}")
    print(f"Final checkpoint: {current_model_checkpoint}")
    if all_metrics:
        print(f"Metrics summary saved to: {os.path.join(output_root, 'metrics_summary.json')}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alternating SFT-RL Streaming Trainer with Eval")

    parser.add_argument("--file_paths_json", type=str, default=FILE_PATHS_JSON)
    parser.add_argument(
        "--sft_model_dir",
        type=str,
        default="../streaming_data/train_click/Qwen2.5-0.5B",
        help="SFT ",
    )
    parser.add_argument(
        "--sft_round",
        type=int,
        default=20,
        help=" ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../streaming_data/sft_rl_alternating",
        help=" ",
    )

    
    parser.add_argument("--rl_sample_rate", type=float, default=0.05, help=" ")
    parser.add_argument("--max_history_len", type=int, default=20, help=" ")

    
    parser.add_argument("--rl_num_epochs", type=int, default=1, help="RL")
    parser.add_argument("--rl_train_batch_size", type=int, default=256, help=" batch size")
    parser.add_argument("--rl_eval_batch_size", type=int, default=32, help="RL eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help=" ")
    parser.add_argument("--temperature", type=float, default=1.0, help=" ")
    parser.add_argument("--num_generations", type=int, default=8, help=" ")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-6, help=" ")
    parser.add_argument("--beta", type=float, default=0.08, help="KL")

     
    parser.add_argument(
        "--reward_type",
        type=str,
        default="exact_prefix_behavior",
        choices=["exact", "prefix", "semantic", "exact_prefix_semantic", "exact_prefix_behavior"],
        help=" ",
    )
    parser.add_argument("--w_exact", type=float, default=5.0, help=" ")
    parser.add_argument("--w_prefix", type=float, default=0.7, help=" ")
    parser.add_argument("--w_sem", type=float, default=0, help=" ")
    parser.add_argument("--subtree_sample_ratio", type=float, default=0.02, help=" ")
    parser.add_argument("--w_pos", type=float, default=5.0, help=" ")
    parser.add_argument("--w_inter", type=float, default=5.0, help=" ")
    parser.add_argument("--w_neg", type=float, default=1.0, help=" ")

     
    parser.add_argument(
        "--positive_behavior_cols",
        nargs="+",
        default=["is_long_view"],
        help=" ",
    )
    parser.add_argument(
        "--interaction_behavior_cols",
        nargs="+",
        default=["is_like", "is_follow", "is_forward", "is_comment", "is_search", "is_profile_enter"],
        help=" ",
    )
    parser.add_argument(
        "--negative_behavior_cols",
        nargs="+",
        default=["is_hate"],
        help=" ",
    )

    
    parser.add_argument("--add_gt", action="store_true", help=" ")
    parser.add_argument("--beam_search", action="store_true", help=" ")
    parser.add_argument("--test_during_training", action="store_true", help=" ")
    parser.add_argument("--dynamic_sampling", action="store_true", help=" ")
    parser.add_argument("--sync_ref_model", action="store_true", help=" ")
    parser.add_argument("--dapo", action="store_true", help=" ")
    parser.add_argument("--gspo", action="store_true", help=" ")

     
    parser.add_argument("--sft_num_epochs", type=int, default=1, help=" ")
    parser.add_argument("--sft_learning_rate", type=float, default=1e-4, help=" ")
    parser.add_argument("--num_gpus", type=str, default="8", help=" ")

     
    parser.add_argument(
        "--valid_behaviors",
        nargs="+",
        default=["is_click"],
        help=" ",
    )

    
    parser.add_argument("--auto_eval", dest="auto_eval", action="store_true", help=" ")
    parser.add_argument("--no_auto_eval", dest="auto_eval", action="store_false", help=" ")
    parser.set_defaults(auto_eval=True)

    parser.add_argument("--eval_num_gpus", type=int, default=8, help=" ")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="eval batch size")
    parser.add_argument("--eval_num_beams", type=int, default=50, help="eval beam")
    parser.add_argument("--eval_max_new_tokens", type=int, default=32, help=" ")
    parser.add_argument("--eval_k", type=int, default=50, help="eval top-k")
    parser.add_argument("--eval_length_penalty", type=float, default=0.0, help=" ")
    parser.add_argument("--eval_negative_sample_rate", type=float, default=0.0, help=" ")
    parser.add_argument("--eval_min_diversity", type=int, default=5, help=" ")
    parser.add_argument("--seed", type=int, default=42, help=" ")

    args = parser.parse_args()
    main(args)

