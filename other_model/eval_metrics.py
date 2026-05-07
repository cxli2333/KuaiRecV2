# eval_metrics.py

import os
import math
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re
from typing import Dict, List, Any, Optional


def parse_sid_tokens(sid_str: str) -> List[str]:
    """解析 SID token 字符串。"""
    clean_str = sid_str.replace("<sid_start_flag>", "").replace("<sid_end_flag>", "")
    clean_str = clean_str.replace("<sid_begin>", "").replace("<sid_end>", "")
    tokens = re.findall(r'<([^>]+)>', clean_str)
    return tokens


def get_path_diversity(predict_list: List[str], max_depth: int = 8) -> np.ndarray:
    """
    计算单个样本在每一层的累计唯一路径数 (Prefix Counts)。
    """
    if not predict_list:
        return np.zeros(max_depth)

    parsed_batch = [parse_sid_tokens(p) for p in predict_list]
    counts = np.zeros(max_depth)
    
    for d in range(max_depth):
        unique_prefixes = set()
        for tokens in parsed_batch:
            if d < len(tokens):
                prefix = tuple(tokens[:d+1])
                unique_prefixes.add(prefix)
        counts[d] = len(unique_prefixes)
        
    return counts


def calculate_layer_metrics(predict_list: List[str], max_depth: int = 8):
    """
    计算每一层的 token-level 多样性指标。
    """
    if not predict_list:
        return np.zeros(max_depth), np.zeros(max_depth), [Counter() for _ in range(max_depth)]
    
    parsed_batch = [parse_sid_tokens(p) for p in predict_list]
    
    layer_token_counts = [Counter() for _ in range(max_depth)]
    
    for tokens in parsed_batch:
        for d in range(min(len(tokens), max_depth)):
            layer_token_counts[d][tokens[d]] += 1
    
    unique_counts = np.zeros(max_depth)
    entropy = np.zeros(max_depth)
    
    for d in range(max_depth):
        counter = layer_token_counts[d]
        if len(counter) == 0:
            continue
            
        unique_counts[d] = len(counter)
        
        total = sum(counter.values())
        if total > 0:
            probs = np.array([count / total for count in counter.values()])
            entropy[d] = -np.sum(probs * np.log2(probs + 1e-10))
    
    return unique_counts, entropy, layer_token_counts


def calculate_topk_layer_diversity(predict_list: List[str], topk: int, max_depth: int = 8) -> np.ndarray:
    """
    计算 top-K 结果中每一层的唯一 token 数。
    """
    if not predict_list or topk > len(predict_list):
        return np.zeros(max_depth)
    
    topk_predictions = predict_list[:topk]
    parsed_batch = [parse_sid_tokens(p) for p in topk_predictions]
    
    unique_counts = np.zeros(max_depth)
    
    for d in range(max_depth):
        unique_tokens = set()
        for tokens in parsed_batch:
            if d < len(tokens):
                unique_tokens.add(tokens[d])
        unique_counts[d] = len(unique_tokens)
    
    return unique_counts


def calculate_intra_list_diversity(predict_list: List[str], max_depth: int = 8) -> float:
    """
    计算 Intra-List Diversity (ILD)。
    """
    if len(predict_list) < 2:
        return 0.0
    
    parsed_batch = [parse_sid_tokens(p) for p in predict_list]
    
    dissimilarities = []
    n = len(parsed_batch)
    
    for i in range(n):
        for j in range(i + 1, n):
            seq1 = parsed_batch[i]
            seq2 = parsed_batch[j]
            
            set1 = set(seq1[:max_depth])
            set2 = set(seq2[:max_depth])
            
            union = len(set1.union(set2))
            if union > 0:
                jaccard_sim = len(set1.intersection(set2)) / union
                dissimilarity = 1 - jaccard_sim
            else:
                dissimilarity = 0
            
            dissimilarities.append(dissimilarity)
    
    return np.mean(dissimilarities) if dissimilarities else 0.0


def gini_coefficient(x: List[float]) -> float:
    """计算基尼系数，只基于传入的频次列表（被推荐过的物品）"""
    x = np.array(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    x += 1e-7  # 避免除零
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x))  # 降序排名
    return 1 - (2.0 * (r * x).sum() + s) / (n * s)


def calculate_metrics(
    path: str, 
    item_path: str,
    topk_list: List[int] = None,
    max_depth: int = 8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    计算推荐系统的各种评估指标。
    
    Args:
        path: 预测结果文件路径（JSON 格式）
        item_path: 物品文件路径
        topk_list: Top-K 列表
        max_depth: 最大深度
        verbose: 是否打印详细信息
        
    Returns:
        Dict[str, Any]: 评估指标字典
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    
    if isinstance(path, str):
        path = [path]
    
    if item_path.endswith(".txt"):
        item_path_txt = item_path
    else:
        item_path_txt = f"{item_path}.txt"
    
    # 加载物品文件
    if os.path.exists(item_path_txt):
        with open(item_path_txt, 'r') as f:
            items = f.readlines()
        item_names = [line.split('\t')[0].strip() for line in items]
    else:
        print(f"Warning: Item file not found: {item_path_txt}")
        item_names = []
    
    num_total_items = len(item_names) if item_names else 1
    
    if verbose:
        print(f"Item Size: {num_total_items}")

    # 构建物品字典
    item_dict = {}
    for i, name in enumerate(item_names):
        if name not in item_dict:
            item_dict[name] = [i]
        else:   
            item_dict[name].append(i)
    
    result_dict = {}
    
    for p in path:
        if not os.path.exists(p):
            print(f"Warning: Prediction file not found: {p}")
            continue
            
        with open(p, 'r') as f:
            test_data = json.load(f)
        
        if not test_data:
            print(f"Warning: Empty prediction file: {p}")
            continue
        
        # 检查数据格式
        if 'predict' not in test_data[0]:
            print(f"Error: 'predict' key not found in {p}")
            continue

        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        num_samples = len(text)
        n_beam = len(text[0]) if text else 0
        valid_topk = [k for k in topk_list if k <= n_beam]
        
        if verbose:
            print(f"n_beam: {n_beam}")
            print(f"K: {valid_topk}")
        
        # 初始化指标数组
        ALLNDCG = np.zeros(len(valid_topk))
        ALLHR = np.zeros(len(valid_topk))
        ALLRECALL = np.zeros(len(valid_topk))
        ALLMRR = np.zeros(len(valid_topk)) 
        ALLAUC = np.zeros(len(valid_topk))
        
        ALL_PATH_COUNTS = np.zeros(max_depth)
        ALL_LAYER_UNIQUE = np.zeros(max_depth)
        ALL_LAYER_ENTROPY = np.zeros(max_depth)
        ALL_ILD = 0.0
        
        TOPK_LAYER_DIVERSITY = {k: np.zeros(max_depth) for k in valid_topk}
        global_counts = {k: defaultdict(int) for k in valid_topk}
        
        CC = 0  # 未找到的物品计数

        for index, sample in enumerate(tqdm(text, desc="Calculating metrics", disable=not verbose)):
            # 路径多样性
            sample_path_counts = get_path_diversity(sample, max_depth=max_depth)
            ALL_PATH_COUNTS += sample_path_counts
            
            # Token-level 多样性
            layer_unique, layer_entropy, _ = calculate_layer_metrics(sample, max_depth=max_depth)
            ALL_LAYER_UNIQUE += layer_unique
            ALL_LAYER_ENTROPY += layer_entropy
            
            # Intra-List Diversity
            ild = calculate_intra_list_diversity(sample, max_depth=max_depth)
            ALL_ILD += ild
            
            # Top-K layer diversity
            for k in valid_topk:
                topk_div = calculate_topk_layer_diversity(sample, k, max_depth=max_depth)
                TOPK_LAYER_DIVERSITY[k] += topk_div

            # 获取目标物品
            if 'output' not in test_data[index]:
                continue
                
            target = test_data[index]['output']
            if isinstance(target, list):
                target_item = target[0].strip("\"").strip(" ")
            else:
                target_item = target.strip(" \n\"")
            
            # 找到目标物品在预测中的位置
            minID = 1000000
            for i, pred in enumerate(sample):
                if pred not in item_dict:
                    CC += 1
                
                if pred == target_item:
                    minID = i 
                    break

            # 计算各 Top-K 指标
            for index_k, topk in enumerate(valid_topk):
                if minID < topk:
                    ALLNDCG[index_k] += 1 / math.log(minID + 2, 2) 
                    ALLHR[index_k] += 1
                    ALLRECALL[index_k] += 1 
                    ALLMRR[index_k] += 1 / (minID + 1)

                    if topk > 1:
                        auc_score = (topk - 1 - minID) / (topk - 1)
                        ALLAUC[index_k] += auc_score
                    else:
                        ALLAUC[index_k] += 1.0
                        
                # 统计全局物品出现次数
                for item in sample[:topk]:
                    global_counts[topk][item] += 1

        # 计算 Coverage 和 Gini
        ALLCOVERAGE = []
        ALLGINI = []

        for topk in valid_topk:
            unique_recommended_items = len(global_counts[topk])
            coverage = unique_recommended_items / num_total_items if num_total_items > 0 else 0
            ALLCOVERAGE.append(coverage)
            
            counts_list = list(global_counts[topk].values())
            gini = gini_coefficient(counts_list) if counts_list else 0
            ALLGINI.append(gini)

        # 构建最终结果
        final_metrics = {
            # 基础指标
            "NDCG": (ALLNDCG / num_samples).tolist(),
            "HR": (ALLHR / num_samples).tolist(),
            "Recall": (ALLRECALL / num_samples).tolist(),
            "MRR": (ALLMRR / num_samples).tolist(),
            "AUC": (ALLAUC / num_samples).tolist(),
            
            # 系统级多样性
            "Coverage": ALLCOVERAGE,
            "Gini": ALLGINI,
            
            # Token-level 多样性指标
            "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
            "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
            "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / num_samples).tolist(),
            "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
            
            # Top-K 的 layer diversity
            "TopK_Layer_Diversity": {
                f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist() 
                for k in valid_topk
            },
            
            # 元信息
            "num_samples": num_samples,
            "n_beam": n_beam,
            "valid_topk": valid_topk,
            "unknown_items": CC
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION METRICS SUMMARY")
            print("="*60)
            
            print("\nAccuracy Metrics:")
            for idx, k in enumerate(valid_topk):
                print(f"  Top-{k}:")
                print(f"    NDCG: {final_metrics['NDCG'][idx]:.4f}")
                print(f"    HR: {final_metrics['HR'][idx]:.4f}")
                print(f"    MRR: {final_metrics['MRR'][idx]:.4f}")
            
            print(f"\nDiversity Metrics:")
            print(f"  Avg Intra-List Diversity: {final_metrics['Avg_Intra_List_Diversity']:.4f}")
            print(f"  Coverage: {ALLCOVERAGE}")
            
            print("="*60 + "\n")
        
        result_dict[p] = final_metrics
    
    # 如果只有一个文件，直接返回结果
    if len(result_dict) == 1:
        return list(result_dict.values())[0]
    
    return result_dict


def calculate_metrics_from_predictions(
    predictions: List[Dict[str, Any]],
    item_list: List[str] = None,
    topk_list: List[int] = None,
    max_depth: int = 8
) -> Dict[str, Any]:
    """
    直接从预测列表计算指标（不需要文件）。
    
    Args:
        predictions: 预测结果列表，每个元素包含 'predict' 和 'output'
        item_list: 物品列表
        topk_list: Top-K 列表
        max_depth: 最大深度
        
    Returns:
        Dict[str, Any]: 评估指标字典
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    
    num_total_items = len(item_list) if item_list else 1
    
    text = [sample['predict'] for sample in predictions]
    
    num_samples = len(text)
    n_beam = len(text[0]) if text else 0
    valid_topk = [k for k in topk_list if k <= n_beam]
    
    # 初始化指标
    ALLNDCG = np.zeros(len(valid_topk))
    ALLHR = np.zeros(len(valid_topk))
    ALLMRR = np.zeros(len(valid_topk))
    
    ALL_PATH_COUNTS = np.zeros(max_depth)
    ALL_LAYER_UNIQUE = np.zeros(max_depth)
    ALL_ILD = 0.0
    
    global_counts = {k: defaultdict(int) for k in valid_topk}

    for index, sample in enumerate(text):
        # 多样性指标
        ALL_PATH_COUNTS += get_path_diversity(sample, max_depth)
        layer_unique, _, _ = calculate_layer_metrics(sample, max_depth)
        ALL_LAYER_UNIQUE += layer_unique
        ALL_ILD += calculate_intra_list_diversity(sample, max_depth)
        
        # 准确性指标
        target = predictions[index].get('output', '')
        if isinstance(target, list):
            target = target[0]
        
        minID = 1000000
        for i, pred in enumerate(sample):
            if pred == target:
                minID = i
                break
        
        for idx, topk in enumerate(valid_topk):
            if minID < topk:
                ALLNDCG[idx] += 1 / math.log(minID + 2, 2)
                ALLHR[idx] += 1
                ALLMRR[idx] += 1 / (minID + 1)
            
            for item in sample[:topk]:
                global_counts[topk][item] += 1
    
    # Coverage
    ALLCOVERAGE = [len(global_counts[k]) / num_total_items for k in valid_topk]
    
    return {
        "NDCG": (ALLNDCG / num_samples).tolist(),
        "HR": (ALLHR / num_samples).tolist(),
        "MRR": (ALLMRR / num_samples).tolist(),
        "Coverage": ALLCOVERAGE,
        "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
        "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
        "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
        "valid_topk": valid_topk
    }

# eval_metrics.py 末尾添加

def calculate_metrics_from_model_outputs(
    predictions: List[List[str or int]],
    targets: List[List[str or int]],
    item_list: List[str] = None,
    topk_list: List[int] = None,
    max_depth: int = 8
) -> Dict[str, Any]:
    """
    从模型输出直接计算指标（适配训练过程调用）。
    
    Args:
        predictions: 预测结果列表，每个元素是一个候选列表 [candidate1, candidate2, ...]
        targets: 目标物品列表
        item_list: 所有物品列表（用于计算 coverage）
        topk_list: Top-K 列表
        max_depth: 最大深度
        
    Returns:
        Dict[str, Any]: 评估指标字典
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    
    num_total_items = len(item_list) if item_list else 1
    num_samples = len(predictions)
    
    if num_samples == 0:
        return {"error": "No samples to evaluate"}
    
    n_beam = len(predictions[0]) if predictions and predictions[0] else 0
    valid_topk = [k for k in topk_list if k <= n_beam]
    
    if not valid_topk:
        valid_topk = [min(topk_list[0], n_beam)] if n_beam > 0 else [1]
    
    # 初始化指标
    ALLNDCG = np.zeros(len(valid_topk))
    ALLHR = np.zeros(len(valid_topk))
    ALLMRR = np.zeros(len(valid_topk))
    ALLRECALL = np.zeros(len(valid_topk))
    
    ALL_PATH_COUNTS = np.zeros(max_depth)
    ALL_LAYER_UNIQUE = np.zeros(max_depth)
    ALL_LAYER_ENTROPY = np.zeros(max_depth)
    ALL_ILD = 0.0
    
    TOPK_LAYER_DIVERSITY = {k: np.zeros(max_depth) for k in valid_topk}
    global_counts = {k: defaultdict(int) for k in valid_topk}

    for index, sample in enumerate(predictions):
        if not sample:
            continue
            
        # 多样性指标
        sample_path_counts = get_path_diversity(sample, max_depth)
        ALL_PATH_COUNTS += sample_path_counts
        
        layer_unique, layer_entropy, _ = calculate_layer_metrics(sample, max_depth)
        ALL_LAYER_UNIQUE += layer_unique
        ALL_LAYER_ENTROPY += layer_entropy
        
        ALL_ILD += calculate_intra_list_diversity(sample, max_depth)
        
        for k in valid_topk:
            topk_div = calculate_topk_layer_diversity(sample, k, max_depth)
            TOPK_LAYER_DIVERSITY[k] += topk_div
        
        # 准确性指标
        target = targets[index] if index < len(targets) else ""
        if isinstance(target, list):
            target = target[0] if target else ""
        
        # 找到目标在预测中的位置
        minID = 1000000
        for i, pred in enumerate(sample):
            if pred == target:
                minID = i
                break
        
        for idx, topk in enumerate(valid_topk):
            if minID < topk:
                ALLNDCG[idx] += 1 / math.log(minID + 2, 2)
                ALLHR[idx] += 1
                ALLRECALL[idx] += 1
                ALLMRR[idx] += 1 / (minID + 1)
            
            for item in sample[:topk]:
                global_counts[topk][item] += 1
    
    # Coverage 和 Gini
    ALLCOVERAGE = []
    ALLGINI = []
    
    for topk in valid_topk:
        unique_items = len(global_counts[topk])
        coverage = unique_items / num_total_items if num_total_items > 0 else 0
        ALLCOVERAGE.append(coverage)
        
        counts_list = list(global_counts[topk].values())
        gini = gini_coefficient(counts_list) if counts_list else 0
        ALLGINI.append(gini)
    
    return {
        # 准确性指标
        "NDCG": (ALLNDCG / num_samples).tolist(),
        "HR": (ALLHR / num_samples).tolist(),
        "Recall": (ALLRECALL / num_samples).tolist(),
        "MRR": (ALLMRR / num_samples).tolist(),
        
        # 系统级多样性
        "Coverage": ALLCOVERAGE,
        "Gini": ALLGINI,
        
        # Token-level 多样性
        "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
        "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
        "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / num_samples).tolist(),
        "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
        
        # Top-K layer diversity
        "TopK_Layer_Diversity": {
            f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist() 
            for k in valid_topk
        },
        
        # 元信息
        "num_samples": num_samples,
        "n_beam": n_beam,
        "valid_topk": valid_topk
    }

def calculate_multi_target_metrics_from_model_outputs(
    predictions: List[List[Any]],
    targets: List[List[Any]],
    topk_list: List[int] = None,
) -> Dict[str, Any]:
    """
    full_user 模式下的多目标 ranking 指标计算。

    Args:
        predictions:
            每个样本的 beam search 候选列表，按排名排序
            例如:
                [
                    [item1, item2, ..., item50],
                    [item3, item8, ..., item20],
                    ...
                ]

        targets:
            每个样本的真实目标列表（未来交互序列）
            例如:
                [
                    [gt1, gt2, ..., gt80],
                    [gt3, gt5, ..., gt60],
                    ...
                ]

        topk_list:
            要计算的 topK 列表，例如 [10, 20, 50]

    Returns:
        dict
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]

    num_samples = len(predictions)
    if num_samples == 0:
        return {"error": "No predictions provided"}

    # 自动过滤非法 topk
    max_pred_len = max((len(x) for x in predictions if x), default=0)
    valid_topk = [k for k in topk_list if k <= max_pred_len]
    if not valid_topk:
        valid_topk = [max_pred_len] if max_pred_len > 0 else [1]

    all_ndcg = np.zeros(len(valid_topk), dtype=np.float64)
    all_hr = np.zeros(len(valid_topk), dtype=np.float64)
    all_recall = np.zeros(len(valid_topk), dtype=np.float64)
    all_mrr = np.zeros(len(valid_topk), dtype=np.float64)

    valid_samples = 0

    for idx, pred_list in enumerate(predictions):
        if not pred_list:
            continue

        target_list = targets[idx] if idx < len(targets) else []
        if not target_list:
            continue

        valid_samples += 1

        for topk_idx, topk in enumerate(valid_topk):
            pred_topk = pred_list[:topk]
            target_topk = target_list[:topk]
            target_set = set(target_topk)

            if len(target_set) == 0:
                continue

            hits = 0
            dcg = 0.0
            mrr = 0.0

            for rank, pred in enumerate(pred_topk, start=1):
                if pred in target_set:
                    hits += 1
                    dcg += 1.0 / math.log2(rank + 1)
                    if mrr == 0.0:
                        mrr = 1.0 / rank

            ideal_hits = min(len(target_set), topk)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            hr = 1.0 if hits > 0 else 0.0
            recall = hits / len(target_set)

            all_ndcg[topk_idx] += ndcg
            all_hr[topk_idx] += hr
            all_recall[topk_idx] += recall
            all_mrr[topk_idx] += mrr

    if valid_samples == 0:
        return {
            "NDCG": [0.0 for _ in valid_topk],
            "HR": [0.0 for _ in valid_topk],
            "Recall": [0.0 for _ in valid_topk],
            "MRR": [0.0 for _ in valid_topk],
            "num_samples": num_samples,
            "valid_samples": 0,
            "valid_topk": valid_topk,
        }

    return {
        "NDCG": (all_ndcg / valid_samples).tolist(),
        "HR": (all_hr / valid_samples).tolist(),
        "Recall": (all_recall / valid_samples).tolist(),
        "MRR": (all_mrr / valid_samples).tolist(),
        "num_samples": num_samples,
        "valid_samples": valid_samples,
        "valid_topk": valid_topk,
    }



if __name__ == '__main__':
    import fire
    fire.Fire(calculate_metrics)


