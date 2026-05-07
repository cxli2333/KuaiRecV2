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
        只扩充 train，val / test 仍保持 leave-one-out。
        保证训练样本的 label 不与 val/test 的 target（最后一条交互）重叠
        
        Args:
            min_hist (int): 滑窗最短history长度（含label前的序列长度）
            max_hist (int): 滑窗最长history长度
            
        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}

        for user, seq in self.all_item_seqs.items():
            L = len(seq)

            # test: 用完整序列
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(seq)

            # val: 去掉最后 1 个
            if L > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(seq[:-1])

            # train: 滑窗
            if L > 2:  # 至少需要3个item才能生成训练样本
                # 关键修复：hi_max只能到L-3，确保label ≤ L-3
                # val和test的历史都是seq[:-1]（0...L-2）
                # 把训练label最远限制到L-3，就保证label ∉ 这段history
                hi_max = min(L - 3, max_hist)
                
                if hi_max >= min_hist:
                    for t in range(min_hist, hi_max + 1):
                        datasets['train']['user'].append(user)
                        datasets['train']['item_seq'].append(seq[:t + 1])  # 长度 t+1，最后 1 个当 label

        # 转成 HF Dataset
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

        # ① 保留 split 字段作总开关
        strategy = self.config.get('split', 'leave_one_out')
        
        # 改进布尔值解析
        val = self.config.get('train_sliding', False)
        slide = val if isinstance(val, bool) else str(val).lower() == "true"

        if strategy in ['leave_one_out', 'last_out']:
            if slide:  # leave-one-out + train 滑窗
                datasets = self._sliding_train(
                    min_hist=self.config.get('min_hist_len', 2),
                    max_hist=self.config.get('max_hist_len', 50)
                )
            else:  # 纯 leave-one-out
                datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
    
class SeqAugDataset(AbstractDataset):
    """
    读取按用户聚合的 raw_data.pkl（至少包含 ['user','seq']），
    按用户顺序 8:1:1 划分，再做序列增强，
    最终返回 HuggingFace Dataset（列：user, seq, len_seq, next）。
    其中 seq/next 为 **映射后的 item_id**（[PAD]=0，其余从1开始）。
    """
    def __init__(self, config: dict):
        """
        必要 config：
            raw_pkl_path: str   # 原始按用户聚合的数据 pkl，至少两列：user, seq（每行一个用户及其点击序列）
        可选：
            user_col: str = 'user'
            seq_col:  str = 'seq'
            max_item_seq_len: int = 10
            pad_value: int = 0           # 左侧padding值，建议与 [PAD] 对齐为0
            split_ratio: tuple = (0.8, 0.1, 0.1)   # 仅按用户顺序划分
            item_mapping_order: str = 'first'      # 'first' | 'freq_first'  映射策略
        """
        super().__init__(config)
        self.user_col: str = self.config.get('user_col', 'user')
        self.seq_col:  str = self.config.get('seq_col',  'seq')

        self.max_item_seq_len: int    = int(self.config.get('max_item_seq_len', 10))
        self.pad_value: int     = int(self.config.get('pad_value', 0))
        self.pad_order: str     = str(self.config.get('pad_order', 'right'))
        self.split_ratio: tuple = tuple(self.config.get('split_ratio', (0.8, 0.1, 0.1)))
        self.item_mapping_order: str = self.config.get('item_mapping_order', 'first')

        # 中间缓存
        self._raw_df: pd.DataFrame | None = None
        self._expanded_df: pd.DataFrame | None = None
        self._user_splits: Dict[str, set] | None = None  # {'train': set(...), ...}

        # 供统计接口使用
        self.all_item_seqs: Dict[Any, List[Any]] = {}

        # 读入与标准化（只做一次）
        self._download_and_process_raw()

        # 基本统计（按原始数据）
        self.num_users: int = len(self._raw_df)
        # 注意：len_seq 动态生成，这里以原序列长度统计
        self.num_interactions: int = int(sum(len(seq) for seq in self._raw_df[self.seq_col]))
        self.num_avg_seq_len: float = (
            float(self.num_interactions) / max(1, self.num_users)
        )

        # 构建映射与全局频次（基于原始 _raw_df）
        self._build_user_mapping()
        self._build_item_mapping_and_counts()

        # 将原始序列映射为 item_id（保留 user 原值；如需 user_id 可自行添加列）
        self._raw_df[self.seq_col] = self._raw_df[self.seq_col].apply(
            lambda seq: [self.id_mapping['item2id'][it] for it in seq if it in self.id_mapping['item2id']]
        )

        self.log(f"[SeqAugDataset] Loaded users={self.num_users}, interactions={self.num_interactions}, "
                f"avg_len={self.num_avg_seq_len:.4f}", level='info')

    # ---------- 工具 ----------
    @staticmethod
    def _pad_left(seq: List[int], target_len: int, pad: int) -> List[int]:
        L = len(seq)
        if L < target_len:
            return [pad] * (target_len - L) + seq 
        return seq
    
    # ---------- 工具 ----------
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

    # ---------- 读取与标准化 ----------
    def _download_and_process_raw(self):
        """读取 raw_pkl，确保至少包含 user、seq，且 seq 为 List[int]/List[str]。"""
        raw_path = self.config['raw_seq_path']
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"raw_seq_path not found: {raw_path}")

        df = pd.read_pickle(raw_path)

        if self.user_col not in df.columns or self.seq_col not in df.columns:
            raise ValueError(f"raw data must contain columns: '{self.user_col}', '{self.seq_col}'")

        # 统一把 seq 转成 list
        norm_rows = []
        for _, row in df.iterrows():
            user = row[self.user_col]
            seq  = row[self.seq_col]
            if isinstance(seq, (list, tuple)):
                seq_list = list(seq)
            elif isinstance(seq, str):
                # 若你的字符串是用空格或逗号拼接，可在 config 中拓展，或自定义解析
                seq_list = row[self.seq_col].split()
            else:
                # 不可识别则跳过
                continue

            # 过滤空序列
            if len(seq_list) == 0:
                continue

            norm_rows.append({self.user_col: user, self.seq_col: seq_list})

        self._raw_df = pd.DataFrame(norm_rows).reset_index(drop=True)

        # all_item_seqs（用于统计展示）
        self.all_item_seqs = {row[self.user_col]: row[self.seq_col] for _, row in self._raw_df.iterrows()}

        self.log(f"[SeqAugDataset] Raw users loaded: {len(self._raw_df)}", level='info')

    # ---------- 映射与统计 ----------
    def _build_user_mapping(self):
        """构建 user2id / id2user（[PAD]=0，从1开始；按首次出现顺序）"""
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
        读取全集映射：item_key -> SID(list[int])
        默认从 self.config['pid2sid_path'] 加载（pkl/json均可），也可用 self.config['item2codes_path'] 覆盖。
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
        基于全集 item2codes（item_key->SID）构建 item2id/id2item（[PAD]=0，从1开始）。
        全局频次用 _raw_df 中出现过的 item_key 统计；未出现的频次为 0。
        """
        all_item_keys = self._load_full_item_corpus()

        # 2) 统计 _raw_df 中的出现频次（注意 _raw_df 可能是子集）
        # 这里假设 _raw_df[self.seq_col] 仍是“原始 item_key”列表（尚未映射为 id）
        # 如果你前面已经把 _raw_df 映射为 id，这里请改用 id2item 还原或在映射前调用本函数。
        all_seqs = self._raw_df[self.seq_col].tolist()
        flat_items = list(chain.from_iterable(all_seqs))
        appear_counter = Counter(flat_items)   # 仅统计出现过的 key

        # 3) 确定 item 映射顺序
        #    - 'freq_first'：按出现频次降序；同频按“全集原始顺序”稳定排序
        #    - 'first' 或其他：按“全集原始顺序”（即 item2codes 的键顺序/读入顺序）
        if getattr(self, "item_mapping_order", "first") == "freq_first":
            first_pos = {k: i for i, k in enumerate(all_item_keys)}
            item_keys_ordered = sorted(
                all_item_keys,
                key=lambda k: (-appear_counter.get(k, 0), first_pos[k])
            )
        else:
            item_keys_ordered = all_item_keys  # 保持全集的原始顺序（稳定可复现）
            
        assert '[PAD]' not in all_item_keys, "item2codes 中不应含 [PAD]"

        # 4) 构建 item2id / id2item（全集）
        item2id = {'[PAD]': 0}
        id2item = ['[PAD]']
        for i, key in enumerate(item_keys_ordered, start=1):
            item2id[key] = i
            id2item.append(key)

        # 5) 构建与 id 对齐的频次数组（全集长度；未出现的=0）
        max_item_id = len(id2item) - 1
        item_freq_arr = [0] * (max_item_id + 1)
        for key, cnt in appear_counter.items():
            iid = item2id.get(key, None)
            if iid is not None and 0 < iid <= max_item_id:
                item_freq_arr[iid] = cnt
        # 同时保留 dict 形式（id->count）
        item_freq_dict = {i: item_freq_arr[i] for i in range(1, max_item_id + 1)}

        # 6) 回写
        self.id_mapping['item2id'] = item2id
        self.id_mapping['id2item'] = id2item
        self.item_freq_arr = item_freq_arr
        self.item_freq_dict = item_freq_dict
        self.num_items = max_item_id

    # ---------- 划分与增强 ----------
    def _build_user_splits(self):
        """按用户顺序 8:1:1 划分用户集合（只做一次）"""
        users_sorted = sorted(self._raw_df[self.user_col].unique())
        tr, va, te = self._split_users_by_order(users_sorted, self.split_ratio)
        self._user_splits = {'train': tr, 'val': va, 'test': te}
        self.log(f"[SeqAugDataset] Split users -> train:{len(tr)}, val:{len(va)}, test:{len(te)}", level='info')

    def _augment_sequences(self):
        """
        根据 max_item_seq_len/pad_value 进行序列增强：
        - L <= max_item_seq_len 且 L>1：生成 1 条 (hist=items[:-1], next=items[-1])
        - L >  max_item_seq_len        ：滑窗生成 (hist=items[i:i+max_item_seq_len], next=items[i+max_item_seq_len])
        这里的 items 已是 **item_id** 序列。
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
                # 滑窗
                for i in range(0, L - self.max_item_seq_len):
                    hist = items[i:i + self.max_item_seq_len]
                    target = items[i + self.max_item_seq_len]
                    rows.append({'user': user, 'seq': hist, 'len_seq': self.max_item_seq_len, 'next': target})

        self._expanded_df = pd.DataFrame(rows)
        self.log(f"[SeqAugDataset] Augmented samples: {len(self._expanded_df)}", level='info')

    # ---------- 主入口 ----------
    def build_augmented_datasets(self) -> Dict[str, Dataset]:
        """
        返回 {'train','val','test'} 三个 HF Dataset，
        列：['user','seq','len_seq','next']（其中 seq/next 已是 item_id）。
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

    # 覆盖父类 split()（可直接拿增强数据）
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

    # ---------- 便捷方法 ----------
    def get_item_count(self, item_id: int) -> int:
        """返回某个 item_id 的全局频次"""
        if 0 <= item_id < len(self.item_freq_arr):
            return self.item_freq_arr[item_id]
        return 0

    def inverse_items(self, id_seq: List[int]) -> List[Any]:
        """将 item_id 序列还原为原始 item"""
        id2item = self.id_mapping['id2item']
        return [id2item[i] for i in id_seq]

    def inverse_user(self, user_id: int) -> Any:
        """将 user_id 还原为原始 user"""
        return self.id_mapping['id2user'][user_id]

