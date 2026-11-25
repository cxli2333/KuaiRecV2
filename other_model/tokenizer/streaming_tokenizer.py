# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming Tokenizer: a tokenizer designed for streaming training
Build item-to-SID mappings directly from StreamingInteractionDataset without precomputed pickle files
"""

import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from logging import getLogger


class StreamingTokenizer:
    """
    Tokenizer for streaming training
    Build vocabulary and mappings dynamically from StreamingInteractionDataset
    """
    
    SPECIAL_TOKENS = {
        '[PAD]': 0,
        '[BOS]': 1,
        '[EOS]': 2,
        '[SEP]': 3,
        '[UNK]': 4,
    }
    
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
        Initialize the tokenizer.
        
        Args:
            config: configuration dictionary
            dataset: StreamingInteractionDataset instance
            n_digits: number of SID levels, default 8
            codebook_size: codebook size at each level, default 512
        """
        self.config = config
        self.dataset = dataset
        self.n_digits = n_digits
        self.codebook_size = codebook_size
        self.logger = getLogger()
        
        self.max_item_seq_len = config.get('max_item_seq_len', 50)
        self.max_token_seq_len = config.get('max_token_seq_len', 512)
        self.pad_order = config.get('pad_order', 'left')
        
        self._build_vocabulary()
        
        self._build_item_to_sid_mapping()
        
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
        """Build the vocabulary."""
        self.token2id = {}
        self.id2token = {}
        
        # 1. Add special tokens
        for token, idx in self.SPECIAL_TOKENS.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        # 2. Add SID special tokens
        for token, idx in self.SID_TOKENS.items():
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        # 3. Starting offset for hierarchical SID tokens
        self.sid_offset = len(self.SPECIAL_TOKENS) + len(self.SID_TOKENS)
        
        # 4. Add hierarchical SID tokens: <a_0>, <a_1>, ..., <h_511>
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
        self.sid2item = {}  # tuple(token_ids) -> List[item_id]  # Store a list
        
        for item_id in self.dataset.item_data.keys():
            sid_tokens = self.dataset.get_item_sid_tokens(item_id)
            if not sid_tokens:
                continue
            
            token_ids = self._convert_sid_tokens_to_ids(sid_tokens)
            self.item2sid[item_id] = token_ids
            
            key = tuple(token_ids)
            if key not in self.sid2item:
                self.sid2item[key] = []
            self.sid2item[key].append(item_id)  # Append rather than overwrite
        
        conflicts = sum(1 for v in self.sid2item.values() if len(v) > 1)
        self.logger.info(f"Built item-to-SID mapping for {len(self.item2sid)} items")
        conflict_ratio = conflicts * 100 / len(self.sid2item) if self.sid2item else 0.0
        self.logger.info(f"SID collisions: {conflicts} ({conflict_ratio:.2f}%)")


    def _convert_sid_tokens_to_ids(self, sid_tokens: List[str]) -> List[int]:
        """Convert a list of SID token strings to token IDs."""
        token_ids = []
        
        for token in sid_tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
            else:
                token_ids.append(self.SPECIAL_TOKENS['[UNK]'])
        
        return token_ids

    def tokenize(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize datasets.
        
        Args:
            datasets: dictionary containing train, val, and test datasets
            
        Returns:
            Dictionary of tokenized datasets
        """
        tokenized = {}
        
        for split_name, dataset in datasets.items():
            self.logger.info(f"Tokenizing {split_name} set ({len(dataset)} samples)...")
            tokenized[split_name] = self._tokenize_split(dataset, split_name)
        
        return tokenized

    def _tokenize_split(self, dataset, split_name: str):
        """Tokenize one dataset split."""
        tokenized_samples = []
        
        for sample in dataset:
            tokenized_sample = self._tokenize_sample(sample, split_name)
            if tokenized_sample is not None:
                tokenized_samples.append(tokenized_sample)
        
        return tokenized_samples

    def _tokenize_sample(self, sample: Dict, split_name: str, debug_sample_id=-1) -> Optional[Dict]:
        """
        Tokenize one sample.
        
        Input format:
            sample = {
                'user': user_id,
                'seq': [item_id, item_id, ...],  # Historical sequence
                'len_seq': int,
                'next': item_id,  # Target item
                'target_item': item_id
            }
        
        Output format:
            {
                'input_ids': [token_id, ...],  # Input sequence token IDs
                'attention_mask': [1, 1, ...],
                'labels': [token_id, ...],  # Target item's SID token IDs
                'target_item': item_id,
                'user': user_id
            }
        """
        user = sample['user']
        seq = sample['seq']
        target_item = sample['target_item']


        # 1. Convert history to token IDs
        input_token_ids = [self.SPECIAL_TOKENS['[BOS]']]
        
        for item_id in seq:
            if item_id == 0:  # Skip padding
                continue
            
            if item_id in self.item2sid:
                # Add the item's SID tokens
                item_sid_ids = self.item2sid[item_id]
                input_token_ids.extend(item_sid_ids)
                input_token_ids.append(self.SPECIAL_TOKENS['[SEP]'])
        
        # 2. Truncate or pad the input sequence
        if len(input_token_ids) > self.max_token_seq_len - 1:
            input_token_ids = input_token_ids[-(self.max_token_seq_len - 1):]
        
        input_token_ids.append(self.SPECIAL_TOKENS['[EOS]'])
        
        # 3. Padding
        attention_mask = [1] * len(input_token_ids)
        
        if len(input_token_ids) < self.max_token_seq_len:
            pad_len = self.max_token_seq_len - len(input_token_ids)
            if self.pad_order == 'left':
                input_token_ids = [self.padding_token] * pad_len + input_token_ids
                attention_mask = [0] * pad_len + attention_mask
            else:
                input_token_ids = input_token_ids + [self.padding_token] * pad_len
                attention_mask = attention_mask + [0] * pad_len
        
        # 4. Target item's SID tokens as labels
        if target_item in self.item2sid:
            labels = self.item2sid[target_item].copy()  # Make a copy
            #labels.append(self.SPECIAL_TOKENS['[EOS]'])
        else:
            # Skip samples whose target item is outside the vocabulary
            return None
        

        return {
            'input_ids': input_token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_item': target_item,
            'user': user
        }

    def _collate_train(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Training data collate function."""
        return self._collate_batch(batch)

    def _collate_eval(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Evaluation data collate function."""
        return self._collate_batch(batch)

    def _collate_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples into tensors."""
        input_ids = torch.tensor([s['input_ids'] for s in batch], dtype=torch.long)
        attention_mask = torch.tensor([s['attention_mask'] for s in batch], dtype=torch.long)
        
        # Pad labels to equal lengths
        max_label_len = max(len(s['labels']) for s in batch)
        labels = []
        for s in batch:
            label = s['labels']
            if len(label) < max_label_len:
                # Pad with -100, which CrossEntropyLoss ignores
                label = label + [-100] * (max_label_len - len(label))
            labels.append(label)
        labels = torch.tensor(labels, dtype=torch.long)
        
        target_items = torch.tensor([s['target_item'] for s in batch], dtype=torch.long)
        users = torch.tensor([s['user'] for s in batch], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_item': target_items,
            'user': users
        }

    def decode_item_tokens(self, token_ids) -> str:
        """
        Decode token IDs to an item's SID string.
        
        Args:
            token_ids: token ID sequence as a tensor or list
            
        Returns:
            str: SID string such as '<sid_begin><a_171><b_450>...<sid_end>'
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        decoded_tokens = []
        for tid in token_ids:
            if tid in self.id2token:
                token = self.id2token[tid]
                # Skip padding and other non-SID tokens
                if token not in ['[PAD]', '[BOS]', '[EOS]', '[SEP]', '[UNK]']:
                    decoded_tokens.append(token)
        
        return ''.join(decoded_tokens)

    def decode_input_tokens(self, token_ids) -> str:
        """
        Decode input token IDs to a readable string.
        
        Args:
            token_ids: token ID sequence
            
        Returns:
            str: decoded string
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
        General decoding method.
        
        Args:
            token_ids: token ID sequence
            skip_special_tokens: whether to skip special tokens
            
        Returns:
            str: decoded string
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
        Encode an item ID as SID token IDs.
        
        Args:
            item_id: item ID
            
        Returns:
            List[int]: token ID list
        """
        if item_id in self.item2sid:
            return self.item2sid[item_id]
        return [self.SPECIAL_TOKENS['[UNK]']]

    def get_item_from_sid(self, sid_token_ids: List[int]) -> Optional[int]:
        key = tuple(sid_token_ids)
        items = self.sid2item.get(key, [])
        if items:
            return items[0]  # Return the first item, or optionally all items
        return None


    def get_all_item_sids(self) -> Dict[int, List[int]]:
        """Get SID mappings for all items."""
        return self.item2sid.copy()

    def state_dict(self) -> dict:
        """Return vocabulary and mapping state without retaining the dataset."""
        return {
            'token2id': self.token2id,
            'id2token': self.id2token,
            'item2sid': self.item2sid,
            'sid2item': self.sid2item,
            'vocab_size': self.vocab_size,
            'n_digits': self.n_digits,
            'codebook_size': self.codebook_size,
            'sid_offset': self.sid_offset,
            'max_item_seq_len': self.max_item_seq_len,
            'max_token_seq_len': self.max_token_seq_len,
            'pad_order': self.pad_order,
        }

    def save(self, path: str):
        """Save the tokenizer to a file."""
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self.state_dict(), f)

        self.logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str, config: dict):
        """Load the tokenizer from a file."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return cls.from_state_dict(data, config)

    @classmethod
    def from_state_dict(cls, data: dict, config: dict):
        """Restore vocabulary and mappings from a checkpoint."""
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
        
        tokenizer.max_item_seq_len = data.get('max_item_seq_len', config.get('max_item_seq_len', 50))
        tokenizer.max_token_seq_len = data.get('max_token_seq_len', config.get('max_token_seq_len', 512))
        tokenizer.pad_order = data.get('pad_order', config.get('pad_order', 'left'))
        
        tokenizer.collate_fn = {
            'train': tokenizer._collate_train,
            'val': tokenizer._collate_eval,
            'test': tokenizer._collate_eval,
        }
        
        return tokenizer


    def decode_to_sid(self, token_ids) -> str:
        """
        Decode token IDs to a SID string for metric calculation.
        
        Args:
            token_ids: token ID sequence
            
        Returns:
            str: SID string such as '<a_171><b_450><c_123>...'
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Filter special tokens and padding
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
        Get all item SID strings for calculating Coverage.
        
        Returns:
            List[str]: SID strings for all items
        """
        sid_strings = []
        for item_id, token_ids in self.item2sid.items():
            sid_str = self.decode_to_sid(token_ids)
            sid_strings.append(sid_str)
        return sid_strings

    def get_target_sid_string(self, item_id: int) -> str:
        """
        Get the target item's SID string.
        
        Args:
            item_id: item ID
            
        Returns:
            str: SID string
        """
        if item_id in self.item2sid:
            return self.decode_to_sid(self.item2sid[item_id])
        return ""

    # Provide an alias for compatibility
VanillaStreamingTokenizer = StreamingTokenizer

