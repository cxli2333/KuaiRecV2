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
    """Parse a SID token string."""
    clean_str = sid_str.replace("<sid_start_flag>", "").replace("<sid_end_flag>", "")
    clean_str = clean_str.replace("<sid_begin>", "").replace("<sid_end>", "")
    tokens = re.findall(r'<([^>]+)>', clean_str)
    return tokens


def get_path_diversity(predict_list: List[str], max_depth: int = 8) -> np.ndarray:
    """
    Calculate cumulative unique prefix counts at each level for one sample.
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
    Calculate token-level diversity metrics at each level.
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
    Calculate the number of unique tokens at each level in the top-K results.
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
    Calculate Intra-List Diversity (ILD).
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
    """Calculate Gini using only the supplied frequency list of recommended items."""
    x = np.array(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    x += 1e-7  # Avoid division by zero
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x))  # Descending ranking
    return 1 - (2.0 * (r * x).sum() + s) / (n * s)


def calculate_metrics(
    path: str, 
    item_path: str,
    topk_list: List[int] = None,
    max_depth: int = 8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate recommendation evaluation metrics.
    
    Args:
        path: path to the prediction results file in JSON format
        item_path: path to the item file
        topk_list: list of Top-K values
        max_depth: maximum depth
        verbose: whether to print detailed information
        
    Returns:
        Dict[str, Any]: evaluation metric dictionary
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    
    if isinstance(path, str):
        path = [path]
    
    if item_path.endswith(".txt"):
        item_path_txt = item_path
    else:
        item_path_txt = f"{item_path}.txt"
    
    # Load item files
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

    # Build the item dictionary
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
        
        # Check the data format
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
        
        # Initialize metric arrays
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
        
        CC = 0  # Count items not found

        for index, sample in enumerate(tqdm(text, desc="Calculating metrics", disable=not verbose)):
            # Path diversity
            sample_path_counts = get_path_diversity(sample, max_depth=max_depth)
            ALL_PATH_COUNTS += sample_path_counts
            
            # Token-level diversity
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

            # Get the target item
            if 'output' not in test_data[index]:
                continue
                
            target = test_data[index]['output']
            if isinstance(target, list):
                target_item = target[0].strip("\"").strip(" ")
            else:
                target_item = target.strip(" \n\"")
            
            # Find the target item's position in the predictions
            minID = 1000000
            for i, pred in enumerate(sample):
                if pred not in item_dict:
                    CC += 1
                
                if pred == target_item:
                    minID = i 
                    break

            # Calculate metrics for each Top-K
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
                        
                # Count global item occurrences
                for item in sample[:topk]:
                    global_counts[topk][item] += 1

        # Calculate Coverage and Gini
        ALLCOVERAGE = []
        ALLGINI = []

        for topk in valid_topk:
            unique_recommended_items = len(global_counts[topk])
            coverage = unique_recommended_items / num_total_items if num_total_items > 0 else 0
            ALLCOVERAGE.append(coverage)
            
            counts_list = list(global_counts[topk].values())
            gini = gini_coefficient(counts_list) if counts_list else 0
            ALLGINI.append(gini)

        # Build the final result
        final_metrics = {
            # Basic metrics
            "NDCG": (ALLNDCG / num_samples).tolist(),
            "HR": (ALLHR / num_samples).tolist(),
            "Recall": (ALLRECALL / num_samples).tolist(),
            "MRR": (ALLMRR / num_samples).tolist(),
            "AUC": (ALLAUC / num_samples).tolist(),
            
            # System-level diversity
            "Coverage": ALLCOVERAGE,
            "Gini": ALLGINI,
            
            # Token-level diversity metrics
            "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
            "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
            "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / num_samples).tolist(),
            "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
            
            # Top-K layer diversity
            "TopK_Layer_Diversity": {
                f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist() 
                for k in valid_topk
            },
            
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
    
    # Return the result directly when there is only one file
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
    Calculate metrics directly from prediction lists without reading files.
    
    Args:
        predictions: result list whose entries contain 'predict' and 'output'
        item_list: list of items
        topk_list: list of Top-K values
        max_depth: maximum depth
        
    Returns:
        Dict[str, Any]: evaluation metric dictionary
    """
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]
    
    num_total_items = len(item_list) if item_list else 1
    
    text = [sample['predict'] for sample in predictions]
    
    num_samples = len(text)
    n_beam = len(text[0]) if text else 0
    valid_topk = [k for k in topk_list if k <= n_beam]
    
    ALLNDCG = np.zeros(len(valid_topk))
    ALLHR = np.zeros(len(valid_topk))
    ALLMRR = np.zeros(len(valid_topk))
    
    ALL_PATH_COUNTS = np.zeros(max_depth)
    ALL_LAYER_UNIQUE = np.zeros(max_depth)
    ALL_ILD = 0.0
    
    global_counts = {k: defaultdict(int) for k in valid_topk}

    for index, sample in enumerate(text):
        # Diversity metrics
        ALL_PATH_COUNTS += get_path_diversity(sample, max_depth)
        layer_unique, _, _ = calculate_layer_metrics(sample, max_depth)
        ALL_LAYER_UNIQUE += layer_unique
        ALL_ILD += calculate_intra_list_diversity(sample, max_depth)
        
        # Accuracy metrics
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

# Add at the end of eval_metrics.py

def calculate_metrics_from_model_outputs(
    predictions: List[List[str]],
    targets: List[str],
    item_list: List[str] = None,
    topk_list: List[int] = None,
    max_depth: int = 8
) -> Dict[str, Any]:
    """
    Calculate metrics directly from model outputs for use during training.
    
    Args:
        predictions: list of candidate lists [candidate1, candidate2, ...]
        targets: list of target items
        item_list: full item list used to calculate coverage
        topk_list: list of Top-K values
        max_depth: maximum depth
        
    Returns:
        Dict[str, Any]: evaluation metric dictionary
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
            
        # Diversity metrics
        sample_path_counts = get_path_diversity(sample, max_depth)
        ALL_PATH_COUNTS += sample_path_counts
        
        layer_unique, layer_entropy, _ = calculate_layer_metrics(sample, max_depth)
        ALL_LAYER_UNIQUE += layer_unique
        ALL_LAYER_ENTROPY += layer_entropy
        
        ALL_ILD += calculate_intra_list_diversity(sample, max_depth)
        
        for k in valid_topk:
            topk_div = calculate_topk_layer_diversity(sample, k, max_depth)
            TOPK_LAYER_DIVERSITY[k] += topk_div
        
        # Accuracy metrics
        target = targets[index] if index < len(targets) else ""
        if isinstance(target, list):
            target = target[0] if target else ""
        
        # Find the target's position in the predictions
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
    
    # Coverage and Gini
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
        # Accuracy metrics
        "NDCG": (ALLNDCG / num_samples).tolist(),
        "HR": (ALLHR / num_samples).tolist(),
        "Recall": (ALLRECALL / num_samples).tolist(),
        "MRR": (ALLMRR / num_samples).tolist(),
        
        # System-level diversity
        "Coverage": ALLCOVERAGE,
        "Gini": ALLGINI,
        
        # Token-level diversity
        "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
        "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
        "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / num_samples).tolist(),
        "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
        
        # Top-K layer diversity
        "TopK_Layer_Diversity": {
            f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist() 
            for k in valid_topk
        },
        
        "num_samples": num_samples,
        "n_beam": n_beam,
        "valid_topk": valid_topk
    }


if __name__ == '__main__':
    import fire
    fire.Fire(calculate_metrics)


