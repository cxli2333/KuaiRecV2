import json
import os
import subprocess
import uuid
import argparse
import re
import math
import sys
from datetime import datetime, timedelta

from streaming_data_prep import load_master_lookups, prepare_rl_data_with_sampling

# --- Path configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

FILE_PATHS_JSON = "Full_Month_file_paths_relative.json"
MASTER_INFO_DIR = os.path.join(PARENT_DIR, "streaming_data", "info")

MASTER_ITEM_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.item.json")
MASTER_INDEX_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.index.json")

RL_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "..", "streaming_rl_run.py")
EVAL_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "streaming_eval.py")
METRIC_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "..", "metric.py")

# --- W&B configuration ---
WANDB_PROJECT = "MiniOneRec-Streaming-RL"
WANDB_RUN_NAME = "streaming-rl"

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


def is_valid_final_ckpt(ckpt_dir: str) -> bool:
    if not ckpt_dir:
        return False
    return os.path.exists(os.path.join(ckpt_dir, "config.json"))


def is_eval_done(eval_out_path: str) -> bool:
    if not eval_out_path:
        return False
    return os.path.exists(eval_out_path) and os.path.getsize(eval_out_path) > 0


def run_cmd(cmd, env=None, cwd=None, check=True):
    print(f"[CMD]\n{cmd}\n")
    result = subprocess.run(cmd, shell=True, env=env, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (code={result.returncode}): {cmd}")
    return result.returncode


def get_round_tag(sft_round: int, rl_round: int) -> str:
    return f"sft_base_{sft_round}_rl_round_{rl_round}"


def get_wandb_state_path(rl_output_dir: str, sft_round: int) -> str:
    return os.path.join(rl_output_dir, f"wandb_state_sft_base_{sft_round}.json")


def load_wandb_state(rl_output_dir: str, sft_round: int):
    state_path = get_wandb_state_path(rl_output_dir, sft_round)
    if not os.path.exists(state_path):
        return {
            "group_name": f"{WANDB_RUN_NAME}-sft_base_{sft_round}",
            "round_runs": {}
        }
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_wandb_state(rl_output_dir: str, sft_round: int, state):
    state_path = get_wandb_state_path(rl_output_dir, sft_round)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_or_create_round_wandb_run(rl_output_dir: str, sft_round: int, round_tag: str, train_day: str):
     
    state = load_wandb_state(rl_output_dir, sft_round)
    round_runs = state.setdefault("round_runs", {})
    group_name = state.setdefault("group_name", f"{WANDB_RUN_NAME}-sft_base_{sft_round}")

    if round_tag not in round_runs:
        round_runs[round_tag] = {
            "run_id": uuid.uuid4().hex[:16],
            "run_name": f"{WANDB_RUN_NAME}-{round_tag}-{train_day}"
        }
        save_wandb_state(rl_output_dir, sft_round, state)

    return group_name, round_runs[round_tag]["run_id"], round_runs[round_tag]["run_name"]


def detect_resume_state(rl_output_dir, model_name, sft_model_dir, sft_round, interaction_files):
     
    last_completed_round = -1
    last_ckpt = None

    for i in range(sft_round, len(interaction_files)):
        round_tag = get_round_tag(sft_round, i)
        round_output_dir = os.path.join(rl_output_dir, model_name, round_tag)
        ckpt_dir = os.path.join(round_output_dir, "model_checkpoint")
        if is_valid_final_ckpt(ckpt_dir):
            last_completed_round = i
            last_ckpt = ckpt_dir

    if last_completed_round >= 0:
        start_round = last_completed_round + 1
        current_model_checkpoint = last_ckpt
        print(f"[Resume] Found existing RL checkpoint at round {last_completed_round}")
        print(f"[Resume] Resume from round {start_round}")
        print(f"[Resume] Current model checkpoint: {current_model_checkpoint}")
        return start_round, current_model_checkpoint

    current_model_checkpoint = os.path.join(
        sft_model_dir, f"streaming_round_{sft_round}", "model_checkpoint"
    )
    print(f"[Resume] No RL checkpoint found.")
    print(f"[Resume] Start from SFT round {sft_round}")
    print(f"[Resume] Current model checkpoint: {current_model_checkpoint}")
    return sft_round, current_model_checkpoint


def get_metrics_path(eval_out_path: str) -> str:
    if eval_out_path.endswith(".json"):
        return eval_out_path[:-5] + ".metrics.json"
    return eval_out_path + ".metrics.json"


def is_metrics_done(metrics_path: str) -> bool:
    if not metrics_path:
        return False
    return os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0


def parse_metric_output(stdout_str, topk_list):
     
    metrics = {}
    try:
        match = re.search(r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___", stdout_str, re.DOTALL)

        if not match:
            print("Error: Unable to load metric JSON")
            print(f"Debug stdout (last 500 chars): {stdout_str[-500:]}")
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

            if "AUC" in raw_metrics and i < len(raw_metrics["AUC"]):
                metrics[f"AUC@{k}"] = raw_metrics["AUC"][i]

            if "Coverage" in raw_metrics and i < len(raw_metrics["Coverage"]):
                metrics[f"Coverage@{k}"] = raw_metrics["Coverage"][i]

            if "Gini" in raw_metrics and i < len(raw_metrics["Gini"]):
                metrics[f"Gini@{k}"] = raw_metrics["Gini"][i]

        diversity_keys = [
            "Avg_Path_Diversity",
            "Avg_Layer_Unique_Tokens",
            "Avg_Layer_Entropy",
            "Avg_Intra_List_Diversity",
            "TopK_Layer_Diversity",
        ]
        for key in diversity_keys:
            if key in raw_metrics:
                metrics[key] = raw_metrics[key]

        metrics["_raw_all_metrics"] = raw_metrics

    except Exception as e:
        print(f"Warning: Failed to parse metric JSON: {e}")

    return metrics


def resolve_metric_item_path(model_ckpt_dir: str, round_output_dir: str = None) -> str:
     
    candidates = [
        os.path.join(model_ckpt_dir, "info_file.tsv"),
    ]
    if round_output_dir is not None:
        candidates.append(os.path.join(round_output_dir, "info_file.tsv"))

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Cannot find info_file.tsv for metric calculation. Tried: {candidates}"
    )

def run_eval(
    model_ckpt_dir: str,
    test_path: str,
    eval_out_path: str,
    category: str,
    eval_num_gpus: int = 8,
    eval_batch_size: int = 8,
    eval_num_beams: int = 50,
    eval_max_new_tokens: int = 32,
    eval_k: int = 50,
    eval_length_penalty: float = 0.0,
    eval_negative_sample_rate: float = 0.0,
    eval_min_diversity: int = 5,
    seed: int = 42,
):
     
    if not os.path.exists(EVAL_SCRIPT_PATH):
        raise FileNotFoundError(f"EVAL script not found: {EVAL_SCRIPT_PATH}")
    if not is_valid_final_ckpt(model_ckpt_dir):
        raise FileNotFoundError(f"Model checkpoint invalid (missing config.json): {model_ckpt_dir}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Eval test file not found: {test_path}")

    os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)

    eval_command = [
        "accelerate",
        "launch",
        f"--num_processes={int(eval_num_gpus)}",
        EVAL_SCRIPT_PATH,
        "--base_model", model_ckpt_dir,
        "--test_data_path", test_path,
        "--result_json_data", eval_out_path,
        "--category", category,
        "--seed", str(int(seed)),
        "--K", str(int(eval_k)),
        "--batch_size", str(int(eval_batch_size)),
        "--length_penalty", str(float(eval_length_penalty)),
        "--max_new_tokens", str(int(eval_max_new_tokens)),
        "--num_beams", str(int(eval_num_beams)),
        "--negative_sample_rate", str(float(eval_negative_sample_rate)),
        "--min_diversity", str(int(eval_min_diversity)),
    ]

    print("[CMD]")
    print(" ".join(eval_command))
    print()

    process_eval = subprocess.run(eval_command, check=False)

    if process_eval.returncode != 0:
        raise RuntimeError(f"streaming_eval.py failed with code={process_eval.returncode}")

    if not is_eval_done(eval_out_path):
        raise RuntimeError(f"Eval finished but output missing/empty: {eval_out_path}")

    print(f"[EVAL] Done. Output: {eval_out_path}")


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

    print("[CMD]")
    print(" ".join(metric_command))
    print()

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
        raise RuntimeError(f"metric.py failed with code={process_metrics.returncode}")

    metrics = parse_metric_output(process_metrics.stdout, TOPK_LIST_FOR_METRIC)
    metrics_path = get_metrics_path(eval_out_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[METRIC] Saved metrics to: {metrics_path}")
    return metrics_path, metrics


def ensure_eval_and_metrics(
    model_ckpt_dir: str,
    round_output_dir: str,
    test_path: str,
    eval_out_path: str,
    category: str,
    args,
):
     
    if not is_eval_done(eval_out_path):
        print(f"[EVAL] Eval output missing, running eval...")
        run_eval(
            model_ckpt_dir=model_ckpt_dir,
            test_path=test_path,
            eval_out_path=eval_out_path,
            category=category,
            eval_num_gpus=args.eval_num_gpus,
            eval_batch_size=args.eval_batch_size,
            eval_num_beams=args.eval_num_beams,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_k=args.eval_k,
            eval_length_penalty=args.eval_length_penalty,
            eval_negative_sample_rate=args.eval_negative_sample_rate,
            eval_min_diversity=args.eval_min_diversity,
            seed=args.seed,
        )
    else:
        print(f"[EVAL] Eval json exists, skip eval: {eval_out_path}")

    metrics_path = get_metrics_path(eval_out_path)
    if not is_metrics_done(metrics_path):
        print(f"[METRIC] Metrics missing, running metric.py ...")
        item_path = resolve_metric_item_path(
            model_ckpt_dir=model_ckpt_dir,
            round_output_dir=round_output_dir,
        )
        metrics_path, metrics = run_metric(
            eval_out_path=eval_out_path,
            item_path=item_path,
        )
        print(f"[METRIC] Summary: {metrics}")
    else:
        print(f"[METRIC] Metrics already exist, skip metric: {metrics_path}")



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

    
    start_round, current_model_checkpoint = detect_resume_state(
        rl_output_dir=args.rl_output_dir,
        model_name=model_name,
        sft_model_dir=args.sft_model_dir,
        sft_round=args.sft_round,
        interaction_files=interaction_files,
    )
    if args.auto_eval:
        backfill_eval_and_metrics_for_completed_rounds(
            args=args,
            interaction_files=interaction_files,
            item_files=item_files,
            model_name=model_name,
        )


    if not is_valid_final_ckpt(current_model_checkpoint):
        raise FileNotFoundError(f"Initial checkpoint not found/invalid: {current_model_checkpoint}")

    print(f"--- Start Streaming RL (Resume+AutoEval+WandbGroup Enabled) ---")
    print(f"[Start Round] {start_round}")

    base_env = os.environ.copy()

    
    for i in range(start_round, len(interaction_files)):
        round_tag = get_round_tag(args.sft_round, i)
        day_T_interaction_file = interaction_files[i]
        train_day = os.path.splitext(os.path.basename(day_T_interaction_file))[0]

         
        data_round_dir = os.path.join(args.rl_output_dir, "data", round_tag)
        os.makedirs(data_round_dir, exist_ok=True)

        round_output_dir = os.path.join(args.rl_output_dir, model_name, round_tag)
        training_output_dir = os.path.join(round_output_dir, "model_checkpoint")
        os.makedirs(round_output_dir, exist_ok=True)

        eval_out_name = f"eval_{round_tag}.json"
        eval_out_path = os.path.join(round_output_dir, eval_out_name)

        current_window_item_files = get_item_files_for_interaction_date(
            interaction_file=day_T_interaction_file,
            item_files=item_files,
            window_days=WINDOW_DAYS,
        )
        item_files_arg = ",".join(current_window_item_files)

        print(f"\n=== [Round {i}] Interaction: {os.path.basename(day_T_interaction_file)} ===")
        print(f"[Round {i}] round_tag={round_tag}")
        print(f"[Round {i}] train_day={train_day}")
        print(f"[Round {i}] item_window_days={WINDOW_DAYS}, item_files={len(current_window_item_files)}")
        if args.print_item_files:
            for fp in current_window_item_files:
                print(f"  [item_file] {fp}")

        train_rl_path = os.path.join(data_round_dir, "train_rl_sampled.tsv")
        test_path = os.path.join(data_round_dir, "test_T.csv")

        if (not os.path.exists(train_rl_path)) or (not os.path.exists(test_path)) or args.force_rebuild_data:
            print(f"[Round {i}] Preparing RL Data...")
            train_rl_path_ret, test_path_ret = prepare_rl_data_with_sampling(
                interaction_file=day_T_interaction_file,
                master_item_path=MASTER_ITEM_JSON_PATH,
                master_index_path=MASTER_INDEX_JSON_PATH,
                output_dir=data_round_dir,
                train_day=train_day,
                sample_rate=args.sample_rate,
                max_history_len=args.max_history_len,
                valid_behaviors=args.valid_behaviors,
            )
            if train_rl_path_ret is None:
                print(f"[Round {i}] No data, skipping.")
                continue
            train_rl_path = train_rl_path_ret
            test_path = test_path_ret
        else:
            print(f"[Round {i}] Reuse existing data: {train_rl_path} | {test_path}")

         
        if is_valid_final_ckpt(training_output_dir):
            print(f"[Round {i}] Final checkpoint exists, skipping training: {training_output_dir}")

            if args.auto_eval:
                ensure_eval_and_metrics(
                    model_ckpt_dir=training_output_dir,
                    round_output_dir=round_output_dir,
                    test_path=test_path,
                    eval_out_path=eval_out_path,
                    category=CATEGORY_NAME,
                    args=args,
                )


            current_model_checkpoint = training_output_dir
            continue

         
        if not is_valid_final_ckpt(current_model_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found/invalid: {current_model_checkpoint}")

        print(f"[Round {i}] Running RL Training...")
        if not os.path.exists(RL_SCRIPT_PATH):
            raise FileNotFoundError(f"RL script not found: {RL_SCRIPT_PATH}")

        
        group_name, round_wandb_run_id, round_wandb_run_name = get_or_create_round_wandb_run(
            rl_output_dir=args.rl_output_dir,
            sft_round=args.sft_round,
            round_tag=round_tag,
            train_day=train_day,
        )

        round_env = base_env.copy()
        round_env["WANDB_PROJECT"] = WANDB_PROJECT
        round_env["WANDB_GROUP"] = group_name
        round_env["WANDB_JOB_TYPE"] = "rl_round"
        round_env["WANDB_NAME"] = round_wandb_run_name
        round_env["WANDB_RUN_ID"] = round_wandb_run_id
        round_env["WANDB_RESUME"] = "allow"

        training_command = (
            f"accelerate launch "
            f"\"{RL_SCRIPT_PATH}\" "
            f"--train_file \"{train_rl_path}\" "
            f"--eval_file \"{test_path}\" "
            f"--model_path \"{current_model_checkpoint}\" "
            f"--output_dir \"{training_output_dir}\" "
            f"--item_files \"{item_files_arg}\" "
            f"--reward_type {args.reward_type} "
            f"--num_train_epochs {args.num_train_epochs} "
            f"--train_batch_size {args.train_batch_size} "
            f"--eval_batch_size {args.eval_batch_size_rl} "
            f"--gradient_accumulation_steps {args.gradient_accumulation_steps} "
            f"--temperature {args.temperature} "
            f"--num_generations {args.num_generations} "
            f"--learning_rate {args.learning_rate} "
            f"--beta {args.beta} "
            f"--subtree_sample_ratio {args.subtree_sample_ratio} "
            f"--w_pos {args.w_pos} "
            f"--w_inter {args.w_inter} "
            f"--w_neg {args.w_neg} "
            f"--positive_behavior_cols {' '.join(args.positive_behavior_cols)} "
            f"--interaction_behavior_cols {' '.join(args.interaction_behavior_cols)} "
            f"--negative_behavior_cols {' '.join(args.negative_behavior_cols)} "
        )

        if args.add_gt:
            training_command += " --add_gt"
        if args.beam_search:
            training_command += " --beam_search"
        if args.test_during_training:
            training_command += " --test_during_training"
        if args.dynamic_sampling:
            training_command += " --dynamic_sampling"
        if args.sync_ref_model:
            training_command += " --sync_ref_model"
        if args.dapo:
            training_command += " --dapo"
        if args.gspo:
            training_command += " --gspo"

        run_cmd(training_command, env=round_env, check=True)

        if not is_valid_final_ckpt(training_output_dir):
            raise RuntimeError(
                f"[Round {i}] Training finished but final checkpoint missing/invalid: {training_output_dir}"
            )

        current_model_checkpoint = training_output_dir

         
        if args.auto_eval:
            ensure_eval_and_metrics(
                model_ckpt_dir=training_output_dir,
                round_output_dir=round_output_dir,
                test_path=test_path,
                eval_out_path=eval_out_path,
                category=CATEGORY_NAME,
                args=args,
            )

    print("\n--- Streaming RL Finished ---")

def backfill_eval_and_metrics_for_completed_rounds(
    args,
    interaction_files,
    item_files,
    model_name,
):
     
    print("\n--- Backfill Eval/Metrics For Completed Rounds ---")

    for i in range(args.sft_round, len(interaction_files)):
        round_tag = get_round_tag(args.sft_round, i)

        day_T_interaction_file = interaction_files[i]
        train_day = os.path.splitext(os.path.basename(day_T_interaction_file))[0]

        data_round_dir = os.path.join(args.rl_output_dir, "data", round_tag)
        round_output_dir = os.path.join(args.rl_output_dir, model_name, round_tag)
        training_output_dir = os.path.join(round_output_dir, "model_checkpoint")
        eval_out_name = f"eval_{round_tag}.json"
        eval_out_path = os.path.join(round_output_dir, eval_out_name)

        
        if not is_valid_final_ckpt(training_output_dir):
            continue

        train_rl_path = os.path.join(data_round_dir, "train_rl_sampled.tsv")
        test_path = os.path.join(data_round_dir, "test_T.csv")

         
        if (not os.path.exists(train_rl_path)) or (not os.path.exists(test_path)):
            print(f"[Backfill Round {i}] Missing data files, preparing data...")
            os.makedirs(data_round_dir, exist_ok=True)

            train_rl_path_ret, test_path_ret = prepare_rl_data_with_sampling(
                interaction_file=day_T_interaction_file,
                master_item_path=MASTER_ITEM_JSON_PATH,
                master_index_path=MASTER_INDEX_JSON_PATH,
                output_dir=data_round_dir,
                train_day=train_day,
                sample_rate=args.sample_rate,
                max_history_len=args.max_history_len,
                valid_behaviors=args.valid_behaviors,
            )
            if train_rl_path_ret is None:
                print(f"[Backfill Round {i}] No data, skip.")
                continue
            train_rl_path = train_rl_path_ret
            test_path = test_path_ret

        need_eval = not is_eval_done(eval_out_path)
        need_metric = not is_metrics_done(get_metrics_path(eval_out_path))

        if not need_eval and not need_metric:
            continue

        print(f"[Backfill Round {i}] ckpt exists,补检查 eval/metric ...")
        ensure_eval_and_metrics(
            model_ckpt_dir=training_output_dir,
            round_output_dir=round_output_dir,
            test_path=test_path,
            eval_out_path=eval_out_path,
            category=CATEGORY_NAME,
            args=args,
        )

    print("--- Backfill Eval/Metrics Finished ---\n")

if __name__ == "__main__":
    VALID_BEHAVIOR_CHOICES = [
        "is_click",
        "is_long_view",
        "is_like",
        "is_follow",
        "is_forward",
        "is_comment",
        "is_hate",
        "is_search",
        "is_profile_enter",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_paths_json", type=str, default=FILE_PATHS_JSON)

    parser.add_argument(
        "--sft_model_dir",
        type=str,
        default="../streaming_data/train_click/Qwen2.5-0.5B",
        help="SFT outputdir, .../streaming_data/test/output/Qwen2.5-0.5B",
    )
    parser.add_argument(
        "--sft_round",
        type=int,
        default=20,
        help="init RL",
    )

    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="RL data sample rate(0,1]",
    )
    parser.add_argument("--rl_output_dir", type=str, default="../streaming_data/rl_long_view")
    parser.add_argument("--max_history_len", type=int, default=20)

    parser.add_argument(
        "--valid_behaviors",
        nargs="+",
        choices=VALID_BEHAVIOR_CHOICES,
        default=["is_long_view"],
        help="One or more behavior flags to keep (logical OR)",
    )

     
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size_rl", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--add_gt", action="store_true")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--test_during_training", action="store_true")
    parser.add_argument("--dynamic_sampling", action="store_true")
    parser.add_argument("--sync_ref_model", action="store_true")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="exact_prefix_behavior",
        choices=[
            "exact",
            "prefix",
            "semantic",
            "exact_prefix_semantic",
            "exact_prefix_behavior",
        ]
    )
    parser.add_argument("--w_exact", type=float, default=1.0)
    parser.add_argument("--w_prefix", type=float, default=0.7)
    
    parser.add_argument("--w_sem", type=float, default=0.2)

    parser.add_argument("--subtree_sample_ratio", type=float, default=0.1,
                    help=" ")
    parser.add_argument("--w_pos", type=float, default=0.5,
                        help=" ")
    parser.add_argument("--w_inter", type=float, default=1.0)
    parser.add_argument("--w_neg", type=float, default=1,
                        help=" ")
    parser.add_argument(
        "--positive_behavior_cols",
        nargs="+",
        default=[
            #"is_click",
            "is_long_view",
        ]
    )
    parser.add_argument(
        "--interaction_behavior_cols",
        nargs="+",
        default=[
            #"is_click",
            "is_like",
            "is_follow",
            "is_forward",
            "is_comment",
            "is_search",
            "is_profile_enter",
        ]
    )
    parser.add_argument(
        "--negative_behavior_cols",
        nargs="+",
        default=["is_hate"]
    )

    parser.add_argument("--dapo", action="store_true")
    parser.add_argument("--gspo", action="store_true")

    # ========= Automatic evaluation =========
    parser.add_argument(
        "--auto_eval",
        dest="auto_eval",
        action="store_true",
        help=" ",
    )
    parser.add_argument(
        "--no_auto_eval",
        dest="auto_eval",
        action="store_false",
        help=" ",
    )
    parser.set_defaults(auto_eval=True)

    parser.add_argument("--eval_num_gpus", type=int, default=8, help=" ")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_num_beams", type=int, default=50)
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument("--eval_k", type=int, default=50)
    parser.add_argument("--eval_length_penalty", type=float, default=0.0)
    parser.add_argument("--eval_negative_sample_rate", type=float, default=0.0)
    parser.add_argument("--eval_min_diversity", type=int, default=5)

    # Other controls
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rebuild_data", action="store_true")
    parser.add_argument("--print_item_files", action="store_true")


    args = parser.parse_args()
    main(args)
