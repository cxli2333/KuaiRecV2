import json
import os
import subprocess
import time
import re
import numpy as np
import sys
import argparse
import uuid
import glob

from streaming_data_prep import load_master_lookups, prepare_data_for_day

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


FILE_PATHS_JSON = os.path.join(PARENT_DIR,'streaming','Full_Month_file_paths_relative.json')
MASTER_INFO_DIR = os.path.join(PARENT_DIR, 'streaming_data', 'info')

MASTER_ITEM_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.item.json")
MASTER_INDEX_JSON_PATH = os.path.join(MASTER_INFO_DIR, "master.index.json")
MASTER_INFO_TXT_PATH = os.path.join(MASTER_INFO_DIR, "master.info.txt")

TRAIN_SCRIPT_PATH = os.path.join(PARENT_DIR, "sft.py")
EVALUATE_SCRIPT_PATH = os.path.join(PARENT_DIR, "streaming", "streaming_eval.py")
METRIC_SCRIPT_PATH = os.path.join(PARENT_DIR, "metric.py")

RUN_EVAL = True
NUM_GPUS = "8"
TASKS_TO_TRAIN = "streaming_sid2sid"

WANDB_PROJECT = "MiniOneRec-Streaming_click" 

WANDB_RUN_NAME = "train_click" 
WANDB_RUN_ID = uuid.uuid4().hex[:8]

BASE_MODEL = "../../../models/Qwen2.5-0.5B"  

START_DAY = 0                         
WINDOW_DAYS = 7
CATEGORY_NAME = "Streaming"         

SAVED_PATH = os.path.join(SCRIPT_DIR,'..','streaming_data',' ')

MODEL_NAME = os.path.basename(BASE_MODEL)

TOPK_LIST_FOR_METRIC = [1, 5, 10, 20, 50]

PYTHON_EXECUTABLE = sys.executable

MIN_DIVERSITY = "5"

EVAL_NEGATIVE_SAMPLE_RATE = 1
NEGATIVE_EVAL_BASE_NAME = f" "
NEGATIVE_EVAL_SAVED_PATH = os.path.join(SCRIPT_DIR,'..','streaming_data', NEGATIVE_EVAL_BASE_NAME)



def parse_metric_output(stdout_str, topk_list):
     
    metrics = {}
    try:
        match = re.search(r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___", stdout_str, re.DOTALL)
        
        if not match:
            print(f"Error: Unable to load metric JSON")
            print(f"Debug stdout (last 500 chars): {stdout_str[-500:]}")
            return metrics

        json_str = match.group(1).strip()
        raw_metrics = json.loads(json_str)
        
        # 
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
            "TopK_Layer_Diversity"
        ]
        
        for key in diversity_keys:
            if key in raw_metrics:
                metrics[key] = raw_metrics[key]
        
         
        metrics["_raw_all_metrics"] = raw_metrics
            
    except Exception as e:
        print(f"Warning: Failed to parse the metrics JSON: {e}")
        import traceback
        traceback.print_exc()
        
    return metrics


def main(args):

    valid_behaviors_list = [b.strip() for b in args.behaviors.split(',') if b.strip()]
    
    item_sid_map, item_title_map, valid_items = load_master_lookups(
        MASTER_ITEM_JSON_PATH, MASTER_INDEX_JSON_PATH 
    )
    if item_sid_map is None: return

    if args.saved_path:
        final_metrics_path = os.path.join(args.saved_path, "output",'metric.json')
        output_dir = os.path.dirname(final_metrics_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
    else:
        final_metrics_path = os.path.join(SCRIPT_DIR, 'streaming_metrics.json')

    try:
        with open(args.file_paths_json, 'r') as f: 
            paths = json.load(f)
        interaction_files = paths['interaction_files']
        item_files = paths['item_files']
    except Exception as e:
        print(f"Error: Unable to load {args.file_paths_json}: {e}")
        return
    
    current_model_checkpoint = BASE_MODEL
    all_metrics = {}
    all_negative_metrics = {}

    print(f"--- Start streaming training (Tasks: {TASKS_TO_TRAIN}, Number of GPUs: {NUM_GPUS}) ---")
    print(f"  WandB Project: {WANDB_PROJECT}")
    print(f"  WandB Run ID:  {WANDB_RUN_ID} ")

    env = os.environ.copy()
    env["WANDB_API_KEY"] = ' '
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["WANDB_RUN_ID"] = WANDB_RUN_ID
    env["WANDB_RESUME"] = "allow"
    env["WANDB_NAME"]    = WANDB_RUN_NAME
    
    for i in range(len(interaction_files)):

        if i + START_DAY + WINDOW_DAYS > len(item_files): break
        day_T_interaction_file = interaction_files[i]
        
        round_output_dir = os.path.join(SAVED_PATH, MODEL_NAME, f"streaming_round_{i}")
        training_output_dir = os.path.join(round_output_dir, f"model_checkpoint")
        test_T_path = os.path.join(round_output_dir, "test_T.csv")

        round_negative_eval_dir = os.path.join(NEGATIVE_EVAL_SAVED_PATH, MODEL_NAME, f"streaming_round_{i}")
        os.makedirs(round_negative_eval_dir, exist_ok=True)
        os.makedirs(round_output_dir, exist_ok=True)

        need_training = True

        if os.path.exists(training_output_dir) and os.path.exists(os.path.join(training_output_dir, "config.json")):
            print(f"\n--- [Jump] Round {i} already finished. Skipping training. ---")
            current_model_checkpoint = training_output_dir 
            need_training = False
        
        os.makedirs(round_output_dir, exist_ok=True)
        print(f"\n---  [Round {i}] interaction: {os.path.basename(day_T_interaction_file)} ---")
        
        print("need_training", need_training)
        if need_training:

            print(f"[Round {i}] preparing data...")
            train_T_path, test_T_path = prepare_data_for_day(
                day_T_interaction_file,
                round_output_dir,
                item_sid_map, item_title_map, valid_items,
                valid_behaviors=valid_behaviors_list 
            )
            if train_T_path is None: continue

            print(f"[Round {i}] are training (Base: {os.path.basename(current_model_checkpoint)})...")
            start_t = time.time()
            
            train_command = [
                "accelerate", "launch", f"--num_processes={NUM_GPUS}", TRAIN_SCRIPT_PATH,
                "--base_model", current_model_checkpoint,
                "--train_file", train_T_path,
                "--eval_file", test_T_path,
                "--sid_index_path", MASTER_INDEX_JSON_PATH,
                "--item_meta_path", MASTER_ITEM_JSON_PATH,
                "--category", CATEGORY_NAME,
                "--output_dir", training_output_dir,
                "--num_epochs", "10",
                "--learning_rate", "1e-4",
                "--eval_strategy", "no",
                "--save_strategy", "epoch",
                "--save_total_limit", "1",
                "--report_to", "wandb",
                "--tasks", TASKS_TO_TRAIN,
                "--wandb_project", WANDB_PROJECT,
                "--wandb_run_name", WANDB_RUN_NAME,
            ]
            
            process = subprocess.run(train_command, check=False,env = env) 
            
            print(f"[Round {i}] training finished  (Time taken: {time.time() - start_t:.2f}s)。")
            if process.returncode != 0:
                print(f"[Round {i}] training failed!")
            else:
                print(f"[Round {i}] mdoel saved to {training_output_dir}")
                current_model_checkpoint = training_output_dir
        
        elif RUN_EVAL and not os.path.exists(test_T_path):
            _, test_path_real = prepare_data_for_day(
                day_T_interaction_file,
                round_output_dir,
                item_sid_map, item_title_map, valid_items,
                valid_behaviors=valid_behaviors_list 
                )
            test_T_path = test_path_real
            print(f"[Round {i}] Data restored: {test_T_path}")
        
        if RUN_EVAL:
            print(f"[Round {i}] Generating predictions (calling evaluate.py).")
            start_t = time.time()
            #result_json_path = os.path.join(round_output_dir, "results.json")
            result_json_name = f"results_neg_{EVAL_NEGATIVE_SAMPLE_RATE}.json"
            result_json_path = os.path.join(round_negative_eval_dir, result_json_name)

            needs_metric_calc = False

            if os.path.exists(result_json_path):
                print(f"--- [Jump] Round {i} Eval results already exist: {result_json_name}. Skipping Eval. ---")
                if os.path.basename(day_T_interaction_file) not in all_negative_metrics:
                    needs_metric_calc = True
            else:
                print(f"[Round {i}] Starting Negative Sampling Evaluation (Rate: {EVAL_NEGATIVE_SAMPLE_RATE})...")
                needs_metric_calc = True
            
                eval_command = [
                    "accelerate", "launch", f"--num_processes={NUM_GPUS}", EVALUATE_SCRIPT_PATH, 
                    "--base_model", current_model_checkpoint,
                    "--test_data_path", test_T_path,
                    "--info_file", MASTER_INFO_TXT_PATH,
                    "--result_json_data", result_json_path,
                    "--category", CATEGORY_NAME,
                    "--batch_size", "4", 
                    "--num_beams", "50",
                    "--negative_sample_rate",str(EVAL_NEGATIVE_SAMPLE_RATE),
                    "--min_diversity", MIN_DIVERSITY ,
                ]
                
                process_eval = subprocess.run(eval_command, check=False)

                if process_eval.returncode != 0:
                    print(f"[Round {i}] evaluate.py failed!")
                    continue
                
                # print(f"[Round {i}] Merging distributed results...")
                
                # base_name_no_ext = os.path.splitext(os.path.basename(result_json_path))[0]
                # search_pattern = os.path.join(os.path.dirname(result_json_path), f"{base_name_no_ext}_part_*.json")
                # part_files = glob.glob(search_pattern)
                
                # full_results = []
                # for p_file in part_files:
                #     try:
                #         with open(p_file, 'r') as f:
                #             chunk = json.load(f)
                #             full_results.extend(chunk)
                #     except Exception as e:
                #         print(f"Error reading part file {p_file}: {e}")
                

                # with open(result_json_path, 'w', encoding='utf-8') as f:
                #     json.dump(full_results, f, indent=2, ensure_ascii=False)
                
                # print(f"[Round {i}] Merged {len(full_results)} samples. Cleaning up temp files.")
                
                # for p_file in part_files:
                #     try:
                #         os.remove(p_file)
                #     except: pass

                # print(f"[Round {i}] Prediction generation completed (Time taken: {time.time() - start_t:.2f}s)。")

            if os.path.exists(result_json_path):
                with open(result_json_path, 'r') as f:
                    results = json.load(f)
                print(f"[Round {i}] Evaluation completed with {len(results)} samples")
            else:
                print(f"[Round {i}] ERROR: Result file not found!")
                
            if needs_metric_calc:
                process_metrics = subprocess.run([
                    PYTHON_EXECUTABLE, METRIC_SCRIPT_PATH,
                    "--path", result_json_path,
                    "--item_path", MASTER_INFO_TXT_PATH
                ], capture_output=True, text=True, check=False)

                if process_metrics.returncode == 0:
                    metrics = parse_metric_output(process_metrics.stdout, TOPK_LIST_FOR_METRIC)
                    print(f"[Round {i}] Negative Eval Result ({EVAL_NEGATIVE_SAMPLE_RATE}): {metrics}")
                    all_negative_metrics[os.path.basename(day_T_interaction_file)] = metrics
                    
                    final_summary_name = f'metrics_summary_neg_{EVAL_NEGATIVE_SAMPLE_RATE}.json'
                    final_metrics_path = os.path.join(NEGATIVE_EVAL_SAVED_PATH, final_summary_name)
                    with open(final_metrics_path, 'w') as f:
                        json.dump(all_negative_metrics, f, indent=2)

    print("\n--- Streaming training finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming trainer")
    parser.add_argument(
        '--file_paths_json', 
        type=str, 
        default=FILE_PATHS_JSON
    )
    
    parser.add_argument(
        '--behaviors', 
        type=str, 
        default="is_click"
    )
    parser.add_argument('--saved_path', type=str, default=SAVED_PATH)
    args = parser.parse_args()
    main(args)
