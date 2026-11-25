import os
import fire
import math
import json
import ast
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import re


def normalize_item(x):
    if x is None:
        return ""
    return str(x).strip().strip('"').strip("'")


def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = normalize_item(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_output_list(raw_output):
    """
    Supported formats:
    - list
    - JSON string
    - python list string
    - single string

    Deduplicate by default and evaluate item-level recommendation lists.
    Change dedup_keep_order to evaluate 50 interaction events rather than unique items.
    """
    if raw_output is None:
        return []

    if isinstance(raw_output, list):
        arr = raw_output
    elif isinstance(raw_output, str):
        s = raw_output.strip()
        if not s:
            return []
        try:
            arr = json.loads(s)
        except Exception:
            try:
                arr = ast.literal_eval(s)
            except Exception:
                arr = [s]
    else:
        arr = [raw_output]

    return dedup_keep_order(arr)


def parse_sid_tokens(sid_str):
    clean_str = str(sid_str).replace("<sid_start_flag>", "").replace("<sid_end_flag>", "")
    tokens = re.findall(r"<([^>]+)>", clean_str)
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
                prefix = tuple(tokens[:d + 1])
                unique_prefixes.add(prefix)
        counts[d] = len(unique_prefixes)

    return counts


def calculate_layer_metrics(predict_list, max_depth=8):
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
                dissimilarity = 0.0

            dissimilarities.append(dissimilarity)

    return float(np.mean(dissimilarities)) if dissimilarities else 0.0


def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    if np.amin(x) < 0:
        return 0.0

    x += 0.0000001
    n = len(x)
    s = x.sum()
    if s <= 0:
        return 0.0
    r = np.argsort(np.argsort(-x))
    return 1 - (2.0 * (r * x).sum() + s) / (n * s)


def parse_topk(topk):
    if isinstance(topk, str):
        return [int(x.strip()) for x in topk.split(",") if x.strip()]
    if isinstance(topk, (list, tuple)):
        return [int(x) for x in topk]
    return [1, 5, 10, 20, 50]


def calculate_metrics(path, item_path, topk="1,5,10,20,50", max_depth=8):
    if type(path) != list:
        path = [path]

    topk_list = parse_topk(topk)

    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    if os.path.exists(item_path):
        f = open(item_path, "r", encoding="utf-8")
    elif os.path.exists(f"{item_path}.txt"):
        f = open(f"{item_path}.txt", "r", encoding="utf-8")
    else:
        raise FileNotFoundError(item_path)

    items = f.readlines()
    f.close()

    item_names = [line.split("\t")[0].strip() for line in items]
    num_total_items = len(item_names)
    print(f"Item Size: {num_total_items}")

    for p in path:
        with open(p, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if not test_data or "predict" not in test_data[0]:
            print(f"Error: 'predict' not found in {p}")
            continue

        text = [[normalize_item(x) for x in sample["predict"]] for sample in test_data]

        num_samples = len(text)
        num_eval_samples = 0

        valid_topk = []
        n_beam = -1

        ALLNDCG = None
        ALLHR = None
        ALLRECALL = None
        ALLMRR = None
        ALLAUC = None

        ALL_PATH_COUNTS = np.zeros(max_depth)
        ALL_LAYER_UNIQUE = np.zeros(max_depth)
        ALL_LAYER_ENTROPY = np.zeros(max_depth)
        ALL_ILD = 0.0
        TOPK_LAYER_DIVERSITY = {k: np.zeros(max_depth) for k in topk_list}
        global_counts = {k: defaultdict(int) for k in topk_list}

        for index, sample in tqdm(enumerate(text), total=num_samples, desc="Calculating list metrics"):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
                ALLRECALL = np.zeros(len(valid_topk))
                ALLMRR = np.zeros(len(valid_topk))
                ALLAUC = np.zeros(len(valid_topk))

            sample = [x for x in sample if x]
            ranked_preds = dedup_keep_order(sample)

            sample_path_counts = get_path_diversity(ranked_preds, max_depth=max_depth)
            ALL_PATH_COUNTS += sample_path_counts

            layer_unique, layer_entropy, _ = calculate_layer_metrics(ranked_preds, max_depth=max_depth)
            ALL_LAYER_UNIQUE += layer_unique
            ALL_LAYER_ENTROPY += layer_entropy

            ild = calculate_intra_list_diversity(ranked_preds, max_depth=max_depth)
            ALL_ILD += ild

            for k in valid_topk:
                topk_div = calculate_topk_layer_diversity(ranked_preds, k, max_depth=max_depth)
                TOPK_LAYER_DIVERSITY[k] += topk_div

                current_topk_items = ranked_preds[:k]
                for item in current_topk_items:
                    global_counts[k][item] += 1

            raw_output = test_data[index].get("output", None)
            if raw_output is None:
                raw_output = test_data[index].get("target_item_sid_list", None)

            target_items = parse_output_list(raw_output)
            if len(target_items) == 0:
                continue

            num_eval_samples += 1
            target_set = set(target_items)
            num_targets = len(target_set)

            for index_k, topk_value in enumerate(valid_topk):
                pred_k = ranked_preds[:topk_value]
                hit_positions = [pos for pos, item in enumerate(pred_k) if item in target_set]
                hit_cnt = len(hit_positions)

                if hit_cnt > 0:
                    ALLHR[index_k] += 1.0
                    ALLMRR[index_k] += 1.0 / (hit_positions[0] + 1)

                    dcg = sum(1.0 / math.log(pos + 2, 2) for pos in hit_positions)
                    ideal_hits = min(num_targets, topk_value)
                    idcg = sum(1.0 / math.log(rank + 2, 2) for rank in range(ideal_hits))
                    if idcg > 0:
                        ALLNDCG[index_k] += dcg / idcg

                ALLRECALL[index_k] += hit_cnt / num_targets

                # List-level metrics do not reuse the old AUC definition; retain zero for parser compatibility.
                ALLAUC[index_k] += 0.0

        ALLCOVERAGE = []
        ALLGINI = []

        for topk_value in valid_topk:
            unique_recommended_items = len(global_counts[topk_value])
            coverage = unique_recommended_items / num_total_items if num_total_items > 0 else 0.0
            ALLCOVERAGE.append(coverage)

            counts_list = list(global_counts[topk_value].values())
            gini = gini_coefficient(counts_list)
            ALLGINI.append(gini)

        sample_denom = max(num_samples, 1)
        eval_denom = max(num_eval_samples, 1)

        final_metrics = {
            "NDCG": (ALLNDCG / eval_denom).tolist(),
            "HR": (ALLHR / eval_denom).tolist(),
            "Recall": (ALLRECALL / eval_denom).tolist(),
            "MRR": (ALLMRR / eval_denom).tolist(),
            "AUC": (ALLAUC / eval_denom).tolist(),
            "Coverage": ALLCOVERAGE,
            "Gini": ALLGINI,
            "Avg_Path_Diversity": (ALL_PATH_COUNTS / sample_denom).tolist(),
            "Avg_Layer_Unique_Tokens": (ALL_LAYER_UNIQUE / sample_denom).tolist(),
            "Avg_Layer_Entropy": (ALL_LAYER_ENTROPY / sample_denom).tolist(),
            "Avg_Intra_List_Diversity": ALL_ILD / sample_denom,
            "TopK_Layer_Diversity": {
                f"Top{k}": (TOPK_LAYER_DIVERSITY[k] / sample_denom).tolist()
                for k in valid_topk
            },
            "EvalSampleCount": num_eval_samples,
            "TotalSampleCount": num_samples
        }

        print("\n" + "=" * 60)
        print("LIST-LEVEL METRICS SUMMARY")
        print("=" * 60)
        print(f"n_beam: {n_beam}")
        print(f"K: {valid_topk}")
        print(f"EvalSampleCount: {num_eval_samples}")
        print(f"TotalSampleCount: {num_samples}")
        print("=" * 60 + "\n")

        print("___METRIC_JSON_START___")
        print(json.dumps(final_metrics, ensure_ascii=False))
        print("___METRIC_JSON_END___")


if __name__ == "__main__":
    fire.Fire(calculate_metrics)
