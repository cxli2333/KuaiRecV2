# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k,
            'mrr':  self.mrr_at_k,   
        }

        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()
    
    def mrr_at_k(self, pos_index, k):
        """
        pos_index: boolean tensor of shape (B, maxk), with at most one True per row
        MRR@k = 1 / rank   if hit in top-k
                0          else
        """
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        mrr = 1.0 / ranks 
        mrr = torch.where(pos_index, mrr, 0)
        return mrr[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        results.update(self.calculate_code_hit_ratios(preds, labels))
        return results

    def calculate_code_hit_ratios(self, preds, labels):
        """
        Count the distribution of matching GT codes among top-K predictions.
        Return each sample's fraction of top-K predictions matching c codes, for c=0..M.
        """
        if preds.ndim != 3 or labels.ndim != 2:
            return {}
        if preds.size(-1) != labels.size(-1):
            return {}

        results = {}
        num_codes = labels.size(-1)
        labels_exp = labels.unsqueeze(1)
        for k in self.config['topk']:
            k = min(int(k), preds.size(1))
            match_counts = (preds[:, :k, :] == labels_exp).sum(dim=-1)  # (B, k)
            one_hot = F.one_hot(match_counts, num_classes=num_codes + 1).float()  # (B, k, M+1)
            ratios = one_hot.mean(dim=1).cpu().float()  # (B, M+1)
            for c in range(num_codes + 1):
                results[f"code_hit@{k}/{c}"] = ratios[:, c]
        return results
    
    def pretty_print_legacy(self, output_results, logger=None):
        """
        Print in the original evaluate() format:
        - ACC row: HR@1, NDCG@1, MRR@1
        - HR/NDCG/MRR @5,10,20,50
        output_results: { "recall@k": float, "ndcg@k": float, "mrr@k": float }
        """
        topk = self.config['topk']   # e.g. [1,5,10,20,50]

        hr_list = [output_results[f"recall@{k}"] for k in topk]
        ndcg_list = [output_results[f"ndcg@{k}"]  for k in topk]
        mrr_list = [output_results[f"mrr@{k}"]    for k in topk]

        def _print(msg):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
                
        def _to_float(x):
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.item()
                else:
                    # Choose an aggregation for vector/matrix values; mean is used here as an example
                    return x.mean().item()
            return float(x)

        # First row: ACC, equivalent to HR@1 / NDCG@1 / MRR@1
        _print('{:<10s} {:<10s} {:<10s}'.format("ACC", "ACC", "ACC"))
        _print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(
            _to_float(hr_list[0]), _to_float(ndcg_list[0]), _to_float(mrr_list[0])
        ))

        # HR@5,10,20,50
        _print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
            f'HR@{topk[1]}', f'HR@{topk[2]}', f'HR@{topk[3]}', f'HR@{topk[4]}'
        ))
        _print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            _to_float(hr_list[1]), _to_float(hr_list[2]),
            _to_float(hr_list[3]), _to_float(hr_list[4])
        ))

        # NDCG@5,10,20,50
        _print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
            f'NDCG@{topk[1]}', f'NDCG@{topk[2]}', f'NDCG@{topk[3]}', f'NDCG@{topk[4]}'
        ))
        _print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            _to_float(ndcg_list[1]), _to_float(ndcg_list[2]),
            _to_float(ndcg_list[3]), _to_float(ndcg_list[4])
        ))

        # MRR@5,10,20,50
        _print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
            f'MRR@{topk[1]}', f'MRR@{topk[2]}', f'MRR@{topk[3]}', f'MRR@{topk[4]}'
        ))
        _print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
            _to_float(mrr_list[1]), _to_float(mrr_list[2]),
            _to_float(mrr_list[3]), _to_float(mrr_list[4])
        ))

    def pretty_print_code_hit(self, output_results, logger=None):
        keys = [k for k in output_results.keys() if k.startswith("code_hit@")]
        if not keys:
            return

        def _print(msg):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        def _parse_key(key):
            # code_hit@{k}/{c}
            body = key.split("@", 1)[1]
            k_str, c_str = body.split("/", 1)
            return int(k_str), int(c_str)

        grouped = {}
        for key in keys:
            k, c = _parse_key(key)
            grouped.setdefault(k, {})[c] = output_results[key]

        for k in sorted(grouped.keys()):
            parts = []
            for c in sorted(grouped[k].keys()):
                parts.append(f"{c}:{grouped[k][c]:.6f}")
            _print(f"CodeHit@{k} (hit_count:ratio) " + " ".join(parts))
