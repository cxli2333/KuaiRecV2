import os
import re
import json
import glob
import shutil
import argparse
import subprocess
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm


DEFAULT_TARGET_BEHAVIORS = [
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

TOPK_LIST_FOR_METRIC = [1, 5, 10, 20, 50]


def resolve_relative_path(base_json_path: str, relative_path: str) -> str:
    """
    按 file_paths_json 所在目录解析相对路径
    """
    if os.path.isabs(relative_path):
        return relative_path
    base_dir = os.path.dirname(os.path.abspath(base_json_path))
    return os.path.abspath(os.path.join(base_dir, relative_path))


def get_interaction_file_by_round(file_paths_json: str, round_idx: int) -> str:
    with open(file_paths_json, "r", encoding="utf-8") as f:
        paths = json.load(f)

    interaction_files = paths["interaction_files"]
    if round_idx < 0 or round_idx >= len(interaction_files):
        raise IndexError(f"round_idx={round_idx} out of range, total={len(interaction_files)}")

    return resolve_relative_path(file_paths_json, interaction_files[round_idx])


def load_master_lookups(master_item_json_path, master_index_json_path):
    print("Loading master.item.json / master.index.json ...")
    try:
        with open(master_index_json_path, 'r', encoding='utf-8') as f:
            item_to_semantic = json.load(f)
        with open(master_item_json_path, 'r', encoding='utf-8') as f:
            item_metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load master files: {e}")

    item_id_to_sid_str = {}
    valid_item_ids = set()

    for item_id, tokens in item_to_semantic.items():
        if item_id in item_metadata:
            item_id_to_sid_str[item_id] = "".join(tokens)
            valid_item_ids.add(item_id)

    print(f"Loaded {len(valid_item_ids)} valid items.")
    return item_id_to_sid_str, valid_item_ids


def parse_metric_output(stdout_str: str, topk_list: List[int]) -> Dict[str, Any]:
    metrics = {}
    try:
        match = re.search(
            r"___METRIC_JSON_START___(.*?)___METRIC_JSON_END___",
            stdout_str,
            re.DOTALL
        )

        if not match:
            print("Warning: metric JSON block not found in metric.py stdout")
            print(stdout_str[-1000:])
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
            "TopK_Layer_Diversity"
        ]
        for key in diversity_keys:
            if key in raw_metrics:
                metrics[key] = raw_metrics[key]

        metrics["_raw_all_metrics"] = raw_metrics
    except Exception as e:
        print(f"Warning: failed to parse metric output: {e}")

    return metrics


def is_positive_flag(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def build_eval_input(history_sids: List[str]) -> str:
    """
    按你当前 results.json 的 input 模板重建
    """
    return (
        "Can you predict the next possible item the user may expect, "
        "given the following chronological interaction history: "
        + "".join(history_sids)
    )


def normalize_output_sid(output_text: str) -> str:
    if output_text is None:
        return ""
    return str(output_text).strip()


def normalize_input_text(input_text: str) -> str:
    if input_text is None:
        return ""
    return str(input_text).strip()


def rebuild_labeled_test_samples(
    interaction_file: str,
    master_item_json_path: str,
    master_index_json_path: str,
    filter_behaviors: List[str],
    target_behaviors: List[str],
    max_history_len: int = 50
) -> List[Dict[str, Any]]:
    """
    按 prepare_data_for_day 的逻辑重建 test 样本，但保留 target 行行为标签
    """
    item_sid_map, valid_items = load_master_lookups(
        master_item_json_path, master_index_json_path
    )

    header_df = pd.read_csv(interaction_file, sep='\t', nrows=1)
    existing_cols = header_df.columns.tolist()

    if 'user_str' in existing_cols:
        user_col = 'user_str'
    elif 'user' in existing_cols:
        user_col = 'user'
    else:
        raise ValueError(f"Neither 'user' nor 'user_str' exists. cols={existing_cols}")

    all_needed_behavior_cols = []
    for c in filter_behaviors + target_behaviors:
        if c not in all_needed_behavior_cols:
            all_needed_behavior_cols.append(c)

    cols_to_read = [user_col, 'item', 'time_ms'] + [
        c for c in all_needed_behavior_cols if c in existing_cols
    ]

    df = pd.read_csv(
        interaction_file,
        sep='\t',
        usecols=cols_to_read,
        dtype={user_col: str, 'item': str}
    )
    df.rename(columns={user_col: 'user'}, inplace=True)

    for c in all_needed_behavior_cols:
        if c not in df.columns:
            df[c] = 0

    original_count = len(df)

    mask = pd.Series([False] * len(df), index=df.index)
    for col in filter_behaviors:
        if col in df.columns:
            mask = mask | (df[col] == 1)

    df = df[mask]
    print(f"[Filtering] {os.path.basename(interaction_file)} original={original_count}, after_filter={len(df)}")

    df['item'] = df['item'].astype(str).str.replace(r'\.0$', '', regex=True)
    df = df[df['item'].isin(valid_items)]
    df = df.sort_values(by=['user', 'time_ms'])

    rebuilt = []
    grouped = df.groupby('user')

    for user_id, group in tqdm(grouped, desc=f"Rebuilding {os.path.basename(interaction_file)}"):
        interactions = []
        for _, row in group.iterrows():
            item_id_str = str(row['item'])
            sid = item_sid_map.get(item_id_str)
            if not sid:
                continue

            behavior_dict = {b: int(row.get(b, 0)) for b in target_behaviors}
            interactions.append({
                "sid": sid,
                "behaviors": behavior_dict
            })

        if len(interactions) < 2:
            continue

        test_item = interactions[-1]
        train_items = interactions[:-1]

        hist_sids = [x["sid"] for x in train_items][-max_history_len:]
        target_sid = test_item["sid"]

        sample = {
            "user_id": user_id,
            "input": build_eval_input(hist_sids),
            "output": target_sid,
            "history_item_sid": hist_sids,
            "item_sid": target_sid,
        }
        sample.update(test_item["behaviors"])
        rebuilt.append(sample)

    print(f"[Rebuild] total labeled test samples = {len(rebuilt)}")
    return rebuilt


def build_label_index(labeled_samples: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index = defaultdict(list)
    for row in labeled_samples:
        key = (
            normalize_input_text(row["input"]),
            normalize_output_sid(row["output"])
        )
        index[key].append(row)
    return index


def merge_labels_into_results(
    results: List[Dict[str, Any]],
    labeled_samples: List[Dict[str, Any]],
    target_behaviors: List[str]
):
    label_index = build_label_index(labeled_samples)

    merged = []
    matched = 0
    unmatched = 0

    for r in results:
        result_input = normalize_input_text(r.get("input", ""))
        result_output = normalize_output_sid(r.get("output", ""))
        key = (result_input, result_output)

        if key in label_index and len(label_index[key]) > 0:
            label_row = label_index[key].pop(0)
            rr = dict(r)
            for b in target_behaviors:
                rr[b] = int(label_row.get(b, 0))
            rr["merged_user_id"] = label_row.get("user_id")
            rr["merged_item_sid"] = label_row.get("item_sid")
            merged.append(rr)
            matched += 1
        else:
            rr = dict(r)
            for b in target_behaviors:
                rr[b] = None
            rr["merge_failed"] = True
            merged.append(rr)
            unmatched += 1

    print(f"[Merge] matched={matched}, unmatched={unmatched}")
    return merged, matched, unmatched


def run_metric(metric_py_path: str, result_json_path: str, item_path: str) -> Optional[Dict[str, Any]]:
    cmd = [
        "python", metric_py_path,
        "--path", result_json_path,
        "--item_path", item_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        print(f"[ERROR] metric.py failed for {result_json_path}")
        print("STDOUT:")
        print(proc.stdout)
        print("STDERR:")
        print(proc.stderr)
        return None

    return parse_metric_output(proc.stdout, TOPK_LIST_FOR_METRIC)


def filter_results_by_behavior(
    results: List[Dict[str, Any]],
    behavior: str,
    require_cols: List[str] = None
) -> List[Dict[str, Any]]:
    """
    过滤结果：
    - 排除 merge_failed 的行
    - 检查 require_cols 中的所有列都为正
    - 检查目标 behavior 为正
    """
    if require_cols is None:
        require_cols = []
    
    filtered = []
    for row in results:
        if row.get("merge_failed", False):
            continue
        
        # 检查所有 require_cols 是否都为正
        all_required_positive = True
        for col in require_cols:
            if not is_positive_flag(row.get(col, 0)):
                all_required_positive = False
                break
        
        if not all_required_positive:
            continue
        
        # 检查目标 behavior 是否为正
        if not is_positive_flag(row.get(behavior, 0)):
            continue
        
        filtered.append(row)
    
    return filtered


def compute_behavior_metrics(
    merged_results: List[Dict[str, Any]],
    metric_py_path: str,
    item_path: str,
    output_dir: str,
    target_behaviors: List[str],
    require_cols: List[str] = None,
    keep_temp: bool = False
) -> Dict[str, Any]:
    """
    计算行为指标
    require_cols: 需要为正的列列表（例如 ["is_long_view"] 或 ["is_click", "is_like"]）
    """
    if require_cols is None:
        require_cols = []
    
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp_behavior_metric_files")
    os.makedirs(tmp_dir, exist_ok=True)

    summary = {
        "total_results": len(merged_results),
        "matched_results": sum(1 for x in merged_results if not x.get("merge_failed", False)),
        "require_cols": require_cols,
        "overall": None,
        "by_behavior": {}
    }

    valid_results = [x for x in merged_results if not x.get("merge_failed", False)]

    overall_path = os.path.join(tmp_dir, "overall.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)

    overall_metrics = run_metric(metric_py_path, overall_path, item_path)
    summary["overall"] = {
        "sample_count": len(valid_results),
        "metrics": overall_metrics
    }

    for behavior in target_behaviors:
        sub = filter_results_by_behavior(
            valid_results,
            behavior=behavior,
            require_cols=require_cols
        )

        info = {
            "sample_count": len(sub),
            "ratio_vs_matched": (len(sub) / len(valid_results)) if len(valid_results) > 0 else 0.0,
            "metrics": None
        }

        if len(sub) == 0:
            summary["by_behavior"][behavior] = info
            print(f"[Behavior] {behavior}: 0 samples")
            continue

        sub_path = os.path.join(tmp_dir, f"{behavior}.json")
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False, indent=2)

        sub_metrics = run_metric(metric_py_path, sub_path, item_path)
        info["metrics"] = sub_metrics
        summary["by_behavior"][behavior] = info
        print(f"[Behavior] {behavior}: {len(sub)} samples")

    if not keep_temp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return summary


def process_one_round(
    results_json: str,
    round_idx: int,
    file_paths_json: str,
    master_item_json: str,
    master_index_json: str,
    metric_py: str,
    item_info_txt: str,
    output_json: str,
    filter_behaviors: List[str],
    target_behaviors: List[str],
    max_history_len: int,
    require_cols: List[str],
    save_merged_results: bool,
    keep_temp: bool
):
    interaction_file = get_interaction_file_by_round(file_paths_json, round_idx)
    print(f"[INFO] round_idx={round_idx}")
    print(f"[INFO] results_json={results_json}")
    print(f"[INFO] interaction_file={interaction_file}")

    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[INFO] Loaded results: {len(results)}")

    labeled_samples = rebuild_labeled_test_samples(
        interaction_file=interaction_file,
        master_item_json_path=master_item_json,
        master_index_json_path=master_index_json,
        filter_behaviors=filter_behaviors,
        target_behaviors=target_behaviors,
        max_history_len=max_history_len
    )

    merged_results, matched, unmatched = merge_labels_into_results(
        results=results,
        labeled_samples=labeled_samples,
        target_behaviors=target_behaviors
    )

    output_dir = os.path.dirname(output_json) or "."
    summary = compute_behavior_metrics(
        merged_results=merged_results,
        metric_py_path=metric_py,
        item_path=item_info_txt,
        output_dir=output_dir,
        target_behaviors=target_behaviors,
        require_cols=require_cols,
        keep_temp=keep_temp
    )

    summary["merge_stat"] = {
        "matched": matched,
        "unmatched": unmatched
    }
    summary["results_json"] = results_json
    summary["interaction_file"] = interaction_file
    summary["round_idx"] = round_idx
    summary["filter_behaviors"] = filter_behaviors
    summary["target_behaviors"] = target_behaviors

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved summary to: {output_json}")

    if save_merged_results:
        merged_path = os.path.splitext(output_json)[0] + ".merged_results.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved merged results to: {merged_path}")


def scan_round_result_files(results_root_dir: str, result_filename: str) -> List[Tuple[int, str]]:
    """
    扫描 root_dir 下的 streaming_round_x/result_filename
    返回 [(round_idx, path), ...]
    """
    pattern = os.path.join(results_root_dir, "**", result_filename)
    files = glob.glob(pattern, recursive=True)

    pairs = []
    for f in files:
        m = re.search(r"streaming_round_(\d+)", f)
        if m:
            pairs.append((int(m.group(1)), f))

    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Behavior-wise evaluation by round")
    parser.add_argument("--file_paths_json", type=str, required=True)
    parser.add_argument("--master_item_json", type=str, required=True)
    parser.add_argument("--master_index_json", type=str, required=True)
    parser.add_argument("--metric_py", type=str, required=True)
    parser.add_argument("--item_info_txt", type=str, required=True)

    # 单轮模式
    parser.add_argument("--results_json", type=str, default="")
    parser.add_argument("--round_idx", type=int, default=-1)
    parser.add_argument("--output_json", type=str, default="")

    # 批量模式
    parser.add_argument("--results_root_dir", type=str, default="")
    parser.add_argument("--result_filename", type=str, default="results_neg_1.json")
    parser.add_argument("--merged_summary_json", type=str, default="")

    parser.add_argument("--filter_behaviors", type=str, default="is_long_view")
    parser.add_argument("--target_behaviors", type=str, default=",".join(DEFAULT_TARGET_BEHAVIORS))
    parser.add_argument("--max_history_len", type=int, default=50)
    
    # 新参数：指定需要为正的列
    parser.add_argument("--require_cols", type=str, default="is_long_view",
                        help="逗号分隔的列名，这些列必须为正才会被包含在行为指标计算中。留空表示不做额外过滤。")
    parser.add_argument("--no_require_long_view", action="store_true",
                        help="（已弃用）使用 --require_cols='' 代替")
    
    parser.add_argument("--save_merged_results", action="store_true")
    parser.add_argument("--keep_temp", action="store_true")
    args = parser.parse_args()

    filter_behaviors = [x.strip() for x in args.filter_behaviors.split(",") if x.strip()]
    target_behaviors = [x.strip() for x in args.target_behaviors.split(",") if x.strip()]
    
    # 处理 require_cols
    if args.no_require_long_view:
        print("[WARNING] --no_require_long_view is deprecated. Use --require_cols='' instead.")
        require_cols = []
    else:
        require_cols = [x.strip() for x in args.require_cols.split(",") if x.strip()]

    # 单轮模式
    if args.results_json and args.round_idx >= 0:
        output_json = args.output_json
        if not output_json:
            output_json = os.path.splitext(args.results_json)[0] + ".behavior_metrics.json"

        process_one_round(
            results_json=args.results_json,
            round_idx=args.round_idx,
            file_paths_json=args.file_paths_json,
            master_item_json=args.master_item_json,
            master_index_json=args.master_index_json,
            metric_py=args.metric_py,
            item_info_txt=args.item_info_txt,
            output_json=output_json,
            filter_behaviors=filter_behaviors,
            target_behaviors=target_behaviors,
            max_history_len=args.max_history_len,
            require_cols=require_cols,
            save_merged_results=args.save_merged_results,
            keep_temp=args.keep_temp
        )
        return

    # 批量模式
    if args.results_root_dir:
        pairs = scan_round_result_files(args.results_root_dir, args.result_filename)
        if not pairs:
            raise FileNotFoundError(f"No result files found under: {args.results_root_dir}")

        merged_summary = {}
        for round_idx, results_json in pairs:
            print("\n" + "=" * 80)
            print(f"[BATCH] Processing round {round_idx}")
            print("=" * 80)

            output_json = os.path.splitext(results_json)[0] + ".behavior_metrics.json"

            process_one_round(
                results_json=results_json,
                round_idx=round_idx,
                file_paths_json=args.file_paths_json,
                master_item_json=args.master_item_json,
                master_index_json=args.master_index_json,
                metric_py=args.metric_py,
                item_info_txt=args.item_info_txt,
                output_json=output_json,
                filter_behaviors=filter_behaviors,
                target_behaviors=target_behaviors,
                max_history_len=args.max_history_len,
                require_cols=require_cols,
                save_merged_results=args.save_merged_results,
                keep_temp=args.keep_temp
            )

            with open(output_json, "r", encoding="utf-8") as f:
                merged_summary[f"round_{round_idx}"] = json.load(f)

        merged_summary_json = args.merged_summary_json
        if not merged_summary_json:
            merged_summary_json = os.path.join(args.results_root_dir, "all_round_behavior_metrics.json")

        with open(merged_summary_json, "w", encoding="utf-8") as f:
            json.dump(merged_summary, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Saved merged summary to: {merged_summary_json}")
        return

    raise ValueError("You must use either single-round mode (--results_json + --round_idx) or batch mode (--results_root_dir).")


if __name__ == "__main__":
    main()
