# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming Tokenizer - 专为流式训练设计的分词器
直接从 StreamingInteractionDataset 构建 item-to-SID 映射，无需预先生成 pkl 文件
"""

import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from logging import getLogger
from torch.utils.data import Dataset

BEHAVIOR_COLS = [
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


class StreamingTokenizer:
    """
    流式训练专用分词器
    从 StreamingInteractionDataset 动态构建词表和映射
    """
    
    # 特殊 token
    SPECIAL_TOKENS = {
        '[PAD]': 0,
        '[BOS]': 1,
        '[EOS]': 2,
        '[SEP]': 3,
        '[UNK]': 4,
    }
    
    # SID 相关 token
    SID_TOKENS = {
        '<sid_begin>': 5,
        '<sid_end>': 6,
    }
    
    def __init__(
        self,
        config: dict,
        dataset,  # StreamingInteractionDataset
        n_digits: int = 8,
        codebook_size: int = 512
    ):
        """
        初始化分词器。
        
        Args:
            config: 配置字典
            dataset: StreamingInteractionDataset 实例
            n_digits: SID 层数（默认 8 层）
            codebook_size: 每层的 codebook 大小（默认 512）
        """
        self.config = config
        self.dataset = dataset
        self.n_digits = n_digits
        self.codebook_size = codebook_size
        self.logger = getLogger()
        
        # 序列长度配置
        self.max_item_seq_len = config.get('max_item_seq_len', 50)
        self.max_token_seq_len = config.get('max_token_seq_len', 512)
        self.pad_order = config.get('pad_order', 'left')

        self.max_seq_len = dataset.max_seq_len if hasattr(dataset, 'max_seq_len') else 50
        self.min_seq_len = dataset.min_seq_len if hasattr(dataset, 'min_seq_len') else 1
        
        # 构建词表
        self._build_vocabulary()
        
        # 构建 item-to-SID 映射
        self._build_item_to_sid_mapping()
        self.dataset = None  # 释放 dataset 引用

        # collate 函数
        self.collate_fn = {
            'train': self._collate_train,
            'val': self._collate_eval,
            'test': self._collate_eval,
        }
        
        self.logger.info(f"StreamingTokenizer initialized:")
        self.logger.info(f"  - Vocab size: {self.vocab_size}")
        self.logger.info(f"  - Num items: {len(self.item2sid)}")
        self.logger.info(f"  - N digits: {self.n_digits}")
        self.logger.info(f"  - Codebook size: {self.codebook_size}")

    def _build_vocabulary(self):
        """构建词表。"""
        self.token2id = {}
        self.id2token = {}
        
        # 1. 添加特殊 token
        for token, idx in self.SPECIAL_TOKENS.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        # 2. 添加 SID 特殊 token
        for token, idx in self.SID_TOKENS.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        # 3. SID 层级 token 的起始位置
        self.sid_offset = len(self.SPECIAL_TOKENS) + len(self.SID_TOKENS)
        
        # 4. 添加 SID 层级 token: <a_0>, <a_1>, ..., <h_511>
        level_chars = 'abcdefgh'
        current_idx = self.sid_offset
        
        for level_idx, level_char in enumerate(level_chars[:self.n_digits]):
            for code in range(self.codebook_size):
                token = f'<{level_char}_{code}>'
                self.token2id[token] = current_idx
                self.id2token[current_idx] = token
                current_idx += 1
        
        self.vocab_size = current_idx
        self.padding_token = self.SPECIAL_TOKENS['[PAD]']

    def _build_item_to_sid_mapping(self):
        self.item2sid = {}  # item_id -> List[token_id]
        self.sid2item = {}  # tuple(token_ids) -> List[item_id]  # 改为列表！
        
        for item_id in self.dataset.item_data.keys():
            sid_tokens = self.dataset.get_item_sid_tokens(item_id)
            if not sid_tokens:
                continue
            
            token_ids = self._convert_sid_tokens_to_ids(sid_tokens)
            self.item2sid[item_id] = token_ids
            
            key = tuple(token_ids)
            if key not in self.sid2item:
                self.sid2item[key] = []
            self.sid2item[key].append(item_id)  # 追加而不是覆盖
        
        # 统计冲突
        conflicts = sum(1 for v in self.sid2item.values() if len(v) > 1)
        self.logger.info(f"Built item-to-SID mapping for {len(self.item2sid)} items")
        self.logger.info(f"SID collisions: {conflicts} ({conflicts*100/len(self.sid2item):.2f}%)")


    def _convert_sid_tokens_to_ids(self, sid_tokens: List[str]) -> List[int]:
        """将 SID token 字符串列表转换为 token ID 列表。"""
        token_ids = []
        
        for token in sid_tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
            else:
                # 未知 token，使用 UNK
                token_ids.append(self.SPECIAL_TOKENS['[UNK]'])
        
        return token_ids

    def tokenize(self, datasets):
        tokenized = {}
        for split, ds in datasets.items():
            self.logger.info(f"Tokenizing {split} ({len(ds)} samples)...")
            tokenized[split] = self._tokenize_split(ds, split)
        return tokenized

    def _tokenize_split(self, dataset, split_name: str):
        """对单个数据集划分进行分词。"""
        tokenized_samples = []
        
        for sample in dataset:
            tokenized_sample = self._tokenize_sample(sample, split_name)
            if tokenized_sample is not None:
                tokenized_samples.append(tokenized_sample)
        
        return tokenized_samples

    def _tokenize_sample(self, sample: Dict, split_name: str) -> Optional[Dict]:
        history = sample.get('history_item_ids', [])
        target_item = sample.get('target_item')

        # 先按 item 维度截断，保留最近的历史
        max_history_items = max(1, (self.max_token_seq_len - self.n_digits) // self.n_digits)
        if len(history) > max_history_items:
            history = history[-max_history_items:]

        if len(history) < self.min_seq_len:
            return None

        if target_item not in self.item2sid:
            return None

        input_token_ids = []
        for item_id in history:
            if item_id in self.item2sid:
                input_token_ids.extend(self.item2sid[item_id])

        if len(input_token_ids) == 0:
            return None

        # 再做一次 token 级保险截断
        max_history_tokens = self.max_token_seq_len - self.n_digits
        if len(input_token_ids) > max_history_tokens:
            input_token_ids = input_token_ids[-max_history_tokens:]

        labels = self.item2sid[target_item].copy()

        tokenized = {
            'input_ids': input_token_ids,
            'labels': labels,
            'target_item': target_item,
            'user': sample.get('user_id'),
            'user_id': sample.get('user_id'),
        }

        # 保留 sample_idx（如果上游已经加了）
        if 'sample_idx' in sample:
            tokenized['sample_idx'] = sample['sample_idx']

        # 保留行为列（如果原 sample 里有）
        for col in BEHAVIOR_COLS:
            if col in sample:
                tokenized[col] = int(sample.get(col, 0))

        return tokenized




    def _collate_train(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """训练数据的 collate 函数。"""
        return self._collate_batch(batch, include_meta=False)

    def _collate_eval(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """评估数据的 collate 函数。"""
        return self._collate_batch(batch, include_meta=True)

    def _collate_batch(self, batch, include_meta: bool = False):
        """
        兼容 user_id / user 字段
        只在这里动态生成 attention_mask

        include_meta=False: 训练时只返回模型需要的字段
        include_meta=True: 评估时额外返回 sample_idx / target_item / behavior labels
        """
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        max_input_len = max(len(s['input_ids']) for s in batch)
        max_label_len = max(len(s['labels']) for s in batch)

        input_ids = []
        attention_mask = []
        labels = []
        users = []

        for s in batch:
            inp = s['input_ids']
            lab = s['labels']

            # pad input
            pad_len = max_input_len - len(inp)
            input_ids.append(inp + [self.SPECIAL_TOKENS['[PAD]']] * pad_len)
            attention_mask.append([1] * len(inp) + [0] * pad_len)

            # pad labels with -100
            lab_pad = max_label_len - len(lab)
            labels.append(lab + [-100] * lab_pad)

            uid = s.get('user_id', s.get('user'))
            users.append(uid if uid is not None else -1)

        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'users': torch.tensor(users, dtype=torch.long),
        }

        # 只有 eval/val/test 才透传 metadata，避免影响训练
        if include_meta:
            if 'sample_idx' in batch[0]:
                output['sample_idx'] = torch.tensor(
                    [int(s['sample_idx']) for s in batch],
                    dtype=torch.long
                )

            if 'target_item' in batch[0]:
                output['target_item'] = [s.get('target_item') for s in batch]

            for col in BEHAVIOR_COLS:
                if col in batch[0]:
                    output[col] = torch.tensor(
                        [int(s.get(col, 0)) for s in batch],
                        dtype=torch.long
                    )

        return output




    def decode_item_tokens(self, token_ids) -> str:
        """
        将 token IDs 解码为物品的 SID 字符串。
        
        Args:
            token_ids: token ID 序列（可以是 Tensor 或 List）
            
        Returns:
            str: SID 字符串，如 '<sid_begin><a_171><b_450>...<sid_end>'
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        decoded_tokens = []
        for tid in token_ids:
            if tid in self.id2token:
                token = self.id2token[tid]
                # 跳过 padding 和其他非 SID token
                if token not in ['[PAD]', '[BOS]', '[EOS]', '[SEP]', '[UNK]']:
                    decoded_tokens.append(token)
        
        return ''.join(decoded_tokens)

    def decode_input_tokens(self, token_ids) -> str:
        """
        将输入 token IDs 解码为可读字符串。
        
        Args:
            token_ids: token ID 序列
            
        Returns:
            str: 解码后的字符串
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        decoded_tokens = []
        for tid in token_ids:
            if tid in self.id2token:
                token = self.id2token[tid]
                if token != '[PAD]':
                    decoded_tokens.append(token)
        
        return ' '.join(decoded_tokens)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """
        通用解码方法。
        
        Args:
            token_ids: token ID 序列
            skip_special_tokens: 是否跳过特殊 token
            
        Returns:
            str: 解码后的字符串
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        special_tokens = {'[PAD]', '[BOS]', '[EOS]', '[SEP]', '[UNK]'}
        
        decoded_tokens = []
        for tid in token_ids:
            if tid in self.id2token:
                token = self.id2token[tid]
                if skip_special_tokens and token in special_tokens:
                    continue
                decoded_tokens.append(token)
        
        return ''.join(decoded_tokens)

    def encode_item(self, item_id: int) -> List[int]:
        """
        将物品 ID 编码为 SID token IDs。
        
        Args:
            item_id: 物品 ID
            
        Returns:
            List[int]: token ID 列表
        """
        if item_id in self.item2sid:
            return self.item2sid[item_id]
        return [self.SPECIAL_TOKENS['[UNK]']]

    def get_item_from_sid(self, sid_token_ids: List[int]) -> Optional[int]:
        key = tuple(sid_token_ids)
        items = self.sid2item.get(key, [])
        if items:
            return items[0]  # 返回第一个，或者可以返回所有
        return None


    def get_all_item_sids(self) -> Dict[int, List[int]]:
        """获取所有物品的 SID 映射。"""
        return self.item2sid.copy()

    def save(self, path: str):
        """保存分词器到文件。"""
        import pickle
        
        data = {
            'token2id': self.token2id,
            'id2token': self.id2token,
            'item2sid': self.item2sid,
            'sid2item': self.sid2item,
            'vocab_size': self.vocab_size,
            'n_digits': self.n_digits,
            'codebook_size': self.codebook_size,
            'sid_offset': self.sid_offset,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str, config: dict):
        """从文件加载分词器。"""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # 创建一个空的 tokenizer 实例
        tokenizer = cls.__new__(cls)
        tokenizer.config = config
        tokenizer.logger = getLogger()
        
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = data['id2token']
        tokenizer.item2sid = data['item2sid']
        tokenizer.sid2item = data['sid2item']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.n_digits = data['n_digits']
        tokenizer.codebook_size = data['codebook_size']
        tokenizer.sid_offset = data['sid_offset']
        tokenizer.padding_token = cls.SPECIAL_TOKENS['[PAD]']
        
        tokenizer.max_item_seq_len = config.get('max_item_seq_len', 50)
        tokenizer.max_token_seq_len = config.get('max_token_seq_len', 512)
        tokenizer.pad_order = config.get('pad_order', 'left')
        
        tokenizer.collate_fn = {
            'train': tokenizer._collate_train,
            'val': tokenizer._collate_eval,
            'test': tokenizer._collate_eval,
        }
        
        return tokenizer


    def decode_to_sid(self, token_ids) -> str:
        """
        将 token IDs 解码为 SID 字符串（用于评估指标计算）。
        
        Args:
            token_ids: token ID 序列
            
        Returns:
            str: SID 字符串，如 '<a_171><b_450><c_123>...'
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # 过滤掉特殊 token 和 padding
        special_ids = set(self.SPECIAL_TOKENS.values())
        
        decoded_tokens = []
        for tid in token_ids:
            if tid in special_ids or tid == -100:
                continue
            if tid in self.id2token:
                token = self.id2token[tid]
                decoded_tokens.append(token)
        
        return ''.join(decoded_tokens)

    def get_all_item_sid_strings(self) -> list:
        """
        获取所有物品的 SID 字符串列表（用于计算 Coverage）。
        
        Returns:
            List[str]: 所有物品的 SID 字符串
        """
        sid_strings = []
        for item_id, token_ids in self.item2sid.items():
            sid_str = self.decode_to_sid(token_ids)
            sid_strings.append(sid_str)
        return sid_strings

    def get_target_sid_string(self, item_id: int) -> str:
        """
        获取目标物品的 SID 字符串。
        
        Args:
            item_id: 物品 ID
            
        Returns:
            str: SID 字符串
        """
        if item_id in self.item2sid:
            return self.decode_to_sid(self.item2sid[item_id])
        return ""

    # 为了兼容性，提供一个别名
VanillaStreamingTokenizer = StreamingTokenizer

