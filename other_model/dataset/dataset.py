# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from logging import getLogger
from datasets import Dataset
from typing import List, Tuple, Any, Dict

from torch import norm

def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')

class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        pass

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        for user in self.all_item_seqs:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def _sliding_train(self, min_hist=2, max_hist=50):
        """
        Augment only train; keep val/test in leave-one-out form.
        Prevent training labels from overlapping validation/test targets (the final interaction).
        
        Args:
            min_hist (int): minimum sliding-window history length before the label
            max_hist (int): maximum sliding-window history length
            
        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}

        for user, seq in self.all_item_seqs.items():
            L = len(seq)

            # test: use the complete sequence
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(seq)

            # val: remove the final item
            if L > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(seq[:-1])

            # train: sliding windows
            if L > 2:  # At least three items are required to generate training samples
                # Limit hi_max to L-3 so that label <= L-3
                # Validation and test histories are seq[:-1], positions 0...L-2
                # Limiting training labels to L-3 avoids overlap with that history
                hi_max = min(L - 3, max_hist)
                
                if hi_max >= min_hist:
                    for t in range(min_hist, hi_max + 1):
                        datasets['train']['user'].append(user)
                        datasets['train']['item_seq'].append(seq[:t + 1])  # Length t+1; use the final item as the label

        # Convert to a Hugging Face Dataset
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None:
            return self.split_data

        # Keep split as the global switch
        strategy = self.config.get('split', 'leave_one_out')
        
        # Improve boolean parsing
        val = self.config.get('train_sliding', False)
        slide = val if isinstance(val, bool) else str(val).lower() == "true"

        if strategy in ['leave_one_out', 'last_out']:
            if slide:  # Leave-one-out with sliding-window training augmentation
                datasets = self._sliding_train(
                    min_hist=self.config.get('min_hist_len', 2),
                    max_hist=self.config.get('max_hist_len', 50)
                )
            else:  # Plain leave-one-out
                datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
    
class SeqAugDataset(AbstractDataset):
    """
    Read user-aggregated raw_data.pkl with at least ['user', 'seq'],
    split users in their existing order using an 8:1:1 ratio, then augment sequences,
    and return Hugging Face Datasets with columns user, seq, len_seq, next.
    seq/next contain mapped item IDs: [PAD]=0 and other IDs start at 1.
    """
    def __init__(self, config: dict):
        """
        Required config:
            raw_pkl_path: str   # User-aggregated pickle with user and seq columns, one click sequence per user
        Optional:
            user_col: str = 'user'
            seq_col:  str = 'seq'
            max_item_seq_len: int = 10
            pad_value: int = 0           # Left-padding value; align with [PAD]=0
            split_ratio: tuple = (0.8, 0.1, 0.1)   # Split users in their existing order
            item_mapping_order: str = 'first'      # Mapping strategy: 'first' | 'freq_first'
        """
        super().__init__(config)
        self.user_col: str = self.config.get('user_col', 'user')
        self.seq_col:  str = self.config.get('seq_col',  'seq')

        self.max_item_seq_len: int    = int(self.config.get('max_item_seq_len', 10))
        self.pad_value: int     = int(self.config.get('pad_value', 0))
        self.pad_order: str     = str(self.config.get('pad_order', 'right'))
        self.split_ratio: tuple = tuple(self.config.get('split_ratio', (0.8, 0.1, 0.1)))
        self.item_mapping_order: str = self.config.get('item_mapping_order', 'first')

        # Intermediate cache
        self._raw_df: pd.DataFrame | None = None
        self._expanded_df: pd.DataFrame | None = None
        self._user_splits: Dict[str, set] | None = None  # {'train': set(...), ...}

        # Used by statistics interfaces
        self.all_item_seqs: Dict[Any, List[Any]] = {}

        # Read and normalize once
        self._download_and_process_raw()

        # Basic statistics from raw data
        self.num_users: int = len(self._raw_df)
        # len_seq is generated dynamically; use raw sequence lengths for these statistics
        self.num_interactions: int = int(sum(len(seq) for seq in self._raw_df[self.seq_col]))
        self.num_avg_seq_len: float = (
            float(self.num_interactions) / max(1, self.num_users)
        )

        # Build mappings and global frequencies from the original _raw_df
        self._build_user_mapping()
        self._build_item_mapping_and_counts()

        # Map raw sequences to item_id; preserve raw user values and add user_id separately if needed
        self._raw_df[self.seq_col] = self._raw_df[self.seq_col].apply(
            lambda seq: [self.id_mapping['item2id'][it] for it in seq if it in self.id_mapping['item2id']]
        )

        self.log(f"[SeqAugDataset] Loaded users={self.num_users}, interactions={self.num_interactions}, "
                f"avg_len={self.num_avg_seq_len:.4f}", level='info')

    # ---------- Utilities ----------
    @staticmethod
    def _pad_left(seq: List[int], target_len: int, pad: int) -> List[int]:
        L = len(seq)
        if L < target_len:
            return [pad] * (target_len - L) + seq 
        return seq
    
    # ---------- Utilities ----------
    @staticmethod
    def _pad_right(seq: List[int], target_len: int, pad: int) -> List[int]:
        L = len(seq)
        if L < target_len:
            return seq + [pad] * (target_len - L)
        return seq

    @staticmethod
    def _split_users_by_order(users_sorted: List[Any], ratios=(0.8, 0.1, 0.1)) -> Tuple[set, set, set]:
        assert abs(sum(ratios) - 1.0) < 1e-8, f"split ratios must sum to 1.0, got {ratios}"
        n = len(users_sorted)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train_users = set(users_sorted[:n_train])
        val_users   = set(users_sorted[n_train:n_train + n_val])
        test_users  = set(users_sorted[n_train + n_val:])
        return train_users, val_users, test_users

    # ---------- Reading and normalization ----------
    def _download_and_process_raw(self):
        """Read raw_pkl, requiring user and seq, with seq as List[int]/List[str]."""
        raw_path = self.config['raw_seq_path']
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"raw_seq_path not found: {raw_path}")

        df = pd.read_pickle(raw_path)

        if self.user_col not in df.columns or self.seq_col not in df.columns:
            raise ValueError(f"raw data must contain columns: '{self.user_col}', '{self.seq_col}'")

        # Normalize seq to a list
        norm_rows = []
        for _, row in df.iterrows():
            user = row[self.user_col]
            seq  = row[self.seq_col]
            if isinstance(seq, (list, tuple)):
                seq_list = list(seq)
            elif isinstance(seq, str):
                # Extend config or provide a custom parser for space- or comma-separated strings
                seq_list = row[self.seq_col].split()
            else:
                # Skip unrecognized values
                continue

            # Filter empty sequences
            if len(seq_list) == 0:
                continue

            norm_rows.append({self.user_col: user, self.seq_col: seq_list})

        self._raw_df = pd.DataFrame(norm_rows).reset_index(drop=True)

        # all_item_seqs for statistics display
        self.all_item_seqs = {row[self.user_col]: row[self.seq_col] for _, row in self._raw_df.iterrows()}

        self.log(f"[SeqAugDataset] Raw users loaded: {len(self._raw_df)}", level='info')

    # ---------- Mappings and statistics ----------
    def _build_user_mapping(self):
        """Build user2id / id2user in first-occurrence order; [PAD]=0 and IDs start at 1."""
        users_in_order = list(dict.fromkeys(self._raw_df[self.user_col].tolist()))
        user2id = {'[PAD]': 0}
        id2user = ['[PAD]']
        for i, u in enumerate(users_in_order, start=1):
            user2id[u] = i
            id2user.append(u)
        self.id_mapping['user2id'] = user2id
        self.id_mapping['id2user'] = id2user
        
    def _load_full_item_corpus(self) -> dict:
        """
        Read the full mapping: item_key -> SID(list[int]).
        Load self.config['pid2sid_path'] by default (pickle/JSON), overridden by self.config['item2codes_path'].
        """
        import pickle

        path = self.config.get('item2codes_path')
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"item2codes file not found. Set 'item2codes_path' or 'pid2sid_path'. Got: {path}")

        with open(path, "rb") as f:
            item2tokens = pickle.load(f)
        if not isinstance(item2tokens, dict):
            raise ValueError(f"item2codes file must contain a dict. Got: {type(item2tokens)}")
        return list(item2tokens.keys())

    def _build_item_mapping_and_counts(self):
        """
        Build item2id/id2item from the full item2codes mapping; [PAD]=0 and IDs start at 1.
        Count global frequencies from item_key occurrences in _raw_df; unseen items have zero frequency.
        """
        all_item_keys = self._load_full_item_corpus()

        # 2) Count occurrences in _raw_df, which may be a subset
        # Assume _raw_df[self.seq_col] still contains raw item_key lists, not mapped IDs
        # If already mapped, restore keys with id2item or call this function before mapping
        all_seqs = self._raw_df[self.seq_col].tolist()
        flat_items = list(chain.from_iterable(all_seqs))
        appear_counter = Counter(flat_items)   # Count only keys that occur

        # 3) Determine item mapping order
        #    - 'freq_first': descending frequency, with stable original-order ties
        #    - 'first' or others: original full-set order from item2codes keys or input order
        if getattr(self, "item_mapping_order", "first") == "freq_first":
            first_pos = {k: i for i, k in enumerate(all_item_keys)}
            item_keys_ordered = sorted(
                all_item_keys,
                key=lambda k: (-appear_counter.get(k, 0), first_pos[k])
            )
        else:
            item_keys_ordered = all_item_keys  # Preserve original full-set order for stable reproducibility
            
        assert '[PAD]' not in all_item_keys, "item2codes 中不应含 [PAD]"

        # 4) Build item2id / id2item for the full set
        item2id = {'[PAD]': 0}
        id2item = ['[PAD]']
        for i, key in enumerate(item_keys_ordered, start=1):
            item2id[key] = i
            id2item.append(key)

        # 5) Build an ID-aligned frequency array for the full set; unseen items have zero frequency
        max_item_id = len(id2item) - 1
        item_freq_arr = [0] * (max_item_id + 1)
        for key, cnt in appear_counter.items():
            iid = item2id.get(key, None)
            if iid is not None and 0 < iid <= max_item_id:
                item_freq_arr[iid] = cnt
        # Also retain a dictionary mapping IDs to counts
        item_freq_dict = {i: item_freq_arr[i] for i in range(1, max_item_id + 1)}

        # 6) Write back
        self.id_mapping['item2id'] = item2id
        self.id_mapping['id2item'] = id2item
        self.item_freq_arr = item_freq_arr
        self.item_freq_dict = item_freq_dict
        self.num_items = max_item_id

    # ---------- Splitting and augmentation ----------
    def _build_user_splits(self):
        """Split users once in their existing order using an 8:1:1 ratio."""
        users_sorted = sorted(self._raw_df[self.user_col].unique())
        tr, va, te = self._split_users_by_order(users_sorted, self.split_ratio)
        self._user_splits = {'train': tr, 'val': va, 'test': te}
        self.log(f"[SeqAugDataset] Split users -> train:{len(tr)}, val:{len(va)}, test:{len(te)}", level='info')

    def _augment_sequences(self):
        """
        Augment sequences according to max_item_seq_len/pad_value:
        - L <= max_item_seq_len and L>1: create one sample (hist=items[:-1], next=items[-1])
        - L > max_item_seq_len: use sliding windows (hist=items[i:i+max_item_seq_len], next=items[i+max_item_seq_len])
        items already contains item IDs.
        """
        rows = []
        for _, row in self._raw_df.iterrows():
            user = row[self.user_col]
            items = row[self.seq_col]
            if not items or len(items) <= 1:
                continue
            L = len(items)

            if L <= self.max_item_seq_len:
                hist = items[:-1]
                target = items[-1]
                len_hist = len(hist)
                if self.pad_order == "right":
                    hist_pad = self._pad_right(hist, target_len=self.max_item_seq_len, pad=self.pad_value)
                elif self.pad_order == "left":
                    hist_pad = self._pad_left(hist, target_len=self.max_item_seq_len, pad=self.pad_value)
                else:
                    raise NotImplementedError(f'Padding strategy [{self.pad_order}] not implemented.') 
                rows.append({'user': user, 'seq': hist_pad, 'len_seq': len_hist, 'next': target})
            else:
                # Sliding windows
                for i in range(0, L - self.max_item_seq_len):
                    hist = items[i:i + self.max_item_seq_len]
                    target = items[i + self.max_item_seq_len]
                    rows.append({'user': user, 'seq': hist, 'len_seq': self.max_item_seq_len, 'next': target})

        self._expanded_df = pd.DataFrame(rows)
        self.log(f"[SeqAugDataset] Augmented samples: {len(self._expanded_df)}", level='info')

    # ---------- Main entry point ----------
    def build_augmented_datasets(self) -> Dict[str, Dataset]:
        """
        Return three Hugging Face Datasets: {'train', 'val', 'test'},
        with columns ['user', 'seq', 'len_seq', 'next']; seq/next contain item IDs.
        """
        if self._user_splits is None:
            self._build_user_splits()
        if self._expanded_df is None:
            self._augment_sequences()

        tr_users = self._user_splits['train']
        va_users = self._user_splits['val']
        te_users = self._user_splits['test']

        train_df = self._expanded_df[self._expanded_df[self.user_col].isin(tr_users)].reset_index(drop=True)
        val_df   = self._expanded_df[self._expanded_df[self.user_col].isin(va_users)].reset_index(drop=True)
        test_df  = self._expanded_df[self._expanded_df[self.user_col].isin(te_users)].reset_index(drop=True)

        self.log(f"[SeqAugDataset] Split samples -> train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}", level='info')

        datasets = {
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'val':   Dataset.from_pandas(val_df,   preserve_index=False),
            'test':  Dataset.from_pandas(test_df,  preserve_index=False),
        }
        return datasets

    # Override the parent split() to return augmented data directly
    def split(self):
        if self.split_data is not None:
            return self.split_data
        self.split_data = self.build_augmented_datasets()
        return self.split_data

    def __str__(self) -> str:
        return (f'[Dataset] {self.__class__.__name__}\n'
                f'\tNumber of users: {self.num_users}\n'
                f'\tNumber of items: {self.num_items}\n'
                f'\tNumber of interactions: {self.num_interactions}\n'
                f'\tAverage item sequence length: {self.num_avg_seq_len:.6f}')

    # ---------- Convenience methods ----------
    def get_item_count(self, item_id: int) -> int:
        """Return the global frequency for an item_id."""
        if 0 <= item_id < len(self.item_freq_arr):
            return self.item_freq_arr[item_id]
        return 0

    def inverse_items(self, id_seq: List[int]) -> List[Any]:
        """Restore item_id sequences to original items."""
        id2item = self.id_mapping['id2item']
        return [id2item[i] for i in id_seq]

    def inverse_user(self, user_id: int) -> Any:
        """Restore user_id to the original user."""
        return self.id_mapping['id2user'][user_id]

