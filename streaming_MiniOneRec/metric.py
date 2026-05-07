import os
import fire
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re

def parse_sid_tokens(sid_str):
    clean_str = sid_str.replace("<sid_start_flag>", "").replace("<sid_end_flag>", "")
    tokens = re.findall(r'<([^>]+)>', clean_str)
    return tokens


def get_path_diversity(predict_list, max_depth=8):
     
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


def calculate_layer_metrics(predict_list, max_depth=8):
     
    if not predict_list:
        return np.zeros(max_depth), np.zeros(max_depth), [Counter() for _ in range(max_depth)]
    
    parsed_batch = [parse_sid_tokens(p) for p in predict_list]
    
    # 
    layer_token_counts = [Counter() for _ in range(max_depth)]
    
    for tokens in parsed_batch:
        for d in range(min(len(tokens), max_depth)):
            layer_token_counts[d][tokens[d]] += 1
    
    # 
    unique_counts = np.zeros(max_depth)
    entropy = np.zeros(max_depth)
    
    for d in range(max_depth):
        counter = layer_token_counts[d]
        if len(counter) == 0:
            continue
            
        # 
        unique_counts[d] = len(counter)
        
        #  
        total = sum(counter.values())
        if total > 0:
            probs = np.array([count / total for count in counter.values()])
            entropy[d] = -np.sum(probs * np.log2(probs + 1e-10))
    
    return unique_counts, entropy, layer_token_counts


def calculate_topk_layer_diversity(predict_list, topk, max_depth=8):
    
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


def calculate_intra_list_diversity(predict_list, max_depth=8):
     
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


def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        return 0

    x += 0.0000001  
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x))  
    return 1 - (2.0 * (r * x).sum() + s) / (n * s)


def calculate_metrics(path, item_path):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0

    if os.path.exists(item_path):
        f = open(item_path, 'r')
    elif os.path.exists(f"{item_path}.txt"):
        f = open(f"{item_path}.txt", 'r')
    else:
        raise FileNotFoundError(...)
    items = f.readlines()
    item_names = [line.split('\t')[0].strip() for line in items]
    
    num_total_items = len(item_names)
    print(f"Item Size: {num_total_items}")

    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    result_dict = dict()
    topk_list = [1, 5, 10, 20, 50]
    n_beam = -1
    
    MAX_DEPTH = 8

    for p in path:
        result_dict[p] = {}
        f = open(p, 'r')
        test_data = json.load(f)
        f.close()
        
        if not test_data or 'predict' not in test_data[0]:
            print(f"Error: The 'predict' key was not found in {p}.")
            continue

        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        num_samples = len(text)
        valid_topk = [] 
        
        ALLNDCG = None
        ALLHR = None
        ALLRECALL = None
        ALLMRR = None
        ALLAUC = None

        ALL_PATH_COUNTS = np.zeros(MAX_DEPTH)
        
        
        ALL_LAYER_UNIQUE = np.zeros(MAX_DEPTH)
        ALL_LAYER_ENTROPY = np.zeros(MAX_DEPTH)
        ALL_ILD = 0.0
        
 
        TOPK_LAYER_DIVERSITY = {k: np.zeros(MAX_DEPTH) for k in topk_list}

        global_counts = {k: defaultdict(int) for k in topk_list}

        for index, sample in tqdm(enumerate(text), total=num_samples, desc="Calculating metrics"):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
                ALLRECALL = np.zeros(len(valid_topk))
                ALLMRR = np.zeros(len(valid_topk)) 
                ALLAUC = np.zeros(len(valid_topk))   
            
  
            sample_path_counts = get_path_diversity(sample, max_depth=MAX_DEPTH)
            ALL_PATH_COUNTS += sample_path_counts
            
      
            layer_unique, layer_entropy, _ = calculate_layer_metrics(sample, max_depth=MAX_DEPTH)
            ALL_LAYER_UNIQUE += layer_unique
            ALL_LAYER_ENTROPY += layer_entropy
            
            # Intra-List Diversity
            ild = calculate_intra_list_diversity(sample, max_depth=MAX_DEPTH)
            ALL_ILD += ild
            
            # Top-K layer diversity
            for k in valid_topk:
                topk_div = calculate_topk_layer_diversity(sample, k, max_depth=MAX_DEPTH)
                TOPK_LAYER_DIVERSITY[k] += topk_div

            if 'output' not in test_data[index]:
                continue
                
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            
            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    CC += 1
                    pass
                
                if sample[i] == target_item:
                    minID = i 
                    break

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
                        
                current_topk_items = sample[:topk]

                for item in current_topk_items:
                    global_counts[topk][item] += 1

        ALLCOVERAGE = []
        ALLGINI = []

        for topk in valid_topk:
            unique_recommended_items = len(global_counts[topk])
            coverage = unique_recommended_items / num_total_items
            ALLCOVERAGE.append(coverage)
            
            counts_list = list(global_counts[topk].values())

            # num_zeros = num_total_items - len(counts_list)
            # if num_zeros > 0:
            #     counts_list.extend([0] * num_zeros)
            
            gini = gini_coefficient(counts_list)
            ALLGINI.append(gini)

        print(f"n_beam: {n_beam}")
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(f"K: {valid_topk}")
        
        final_metrics = {
            # 
            "NDCG": (ALLNDCG / num_samples).tolist(),
            "HR": (ALLHR / num_samples).tolist(),
            "Recall": (ALLRECALL / num_samples).tolist(),
            "MRR": (ALLMRR / num_samples).tolist(),
            "AUC": (ALLAUC / num_samples).tolist(),
            
            #  
            "Coverage": ALLCOVERAGE,
            "Gini": ALLGINI,
            
     
            "Avg_Path_Diversity": (ALL_PATH_COUNTS / num_samples).tolist(),
            "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / num_samples).tolist(),
            "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / num_samples).tolist(),
            "Avg_Intra_List_Diversity": ALL_ILD / num_samples,
            
            
            "TopK_Layer_Diversity": {
                f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist() 
                for k in valid_topk
            }
        }
        
        print("\n" + "="*60)
        print("DIVERSITY METRICS SUMMARY")
        print("="*60)
        print(f"Average Unique Tokens per Layer (across all beams):")
        for d, count in enumerate((ALL_LAYER_UNIQUE / num_samples).tolist()):
            print(f"  Layer {d}: {count:.2f}")
        
        print(f"\nAverage Intra-List Diversity: {ALL_ILD / num_samples:.4f}")
        
        print(f"\nTop-K Layer Diversity:")
        for k in valid_topk:
            div_values = (TOPK_LAYER_DIVERSITY[k] / num_samples).tolist()
            print(f"  Top-{k}: Layer 0={div_values[0]:.2f}, Layer 1={div_values[1]:.2f}")
        print("="*60 + "\n")
        
        print("___METRIC_JSON_START___")
        print(json.dumps(final_metrics))
        print("___METRIC_JSON_END___")

if __name__=='__main__':
    fire.Fire(calculate_metrics)
