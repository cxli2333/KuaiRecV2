# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
流式交互数据集

支持:
- 按天加载数据
- 累积用户历史
- 自动构建训练/验证/测试集

数据格式:
- Item 文件: sid_list, emb_vec, item_id, tag_list
  - sid_list 格式: "417,224,387,175,414,125,334,499" (逗号分隔的数字)
- Interaction 文件: user, item, is_click, ...
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from logging import getLogger
from collections import defaultdict,deque

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


class StreamingInteractionDataset:
    """
    流式交互数据集
    
    支持:
    - 按天加载数据
    - 累积用户历史
    - 自动构建训练/验证/测试集
    """
    
    def __init__(   
        self,
        config: dict,
        task: str = 'is_click',
        max_seq_len: int = 50,
        min_seq_len: int = 5,
        max_history_len: int = 999,
        accumulate_history: bool = True,
        splits: List[str] = None,
    ):
        """
        初始化数据集。
        
        Args:
            config: 配置字典
            task: 任务类型（用于过滤交互）
            max_seq_len: 最大序列长度
            min_seq_len: 最小序列长度（小于此长度的用户会被过滤）
            max_history_len: 最大历史长度
            accumulate_history: 是否累积历史
        """
        super().__init__()  
        self.config = config
        self.task = task
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.max_history_len = max_history_len
        self.accumulate_history = accumulate_history
        self._accumulated_loaded = False
        self.logger = getLogger()
        self.splits = splits if splits else ['train', 'val', 'test']

        # 数据存储
        
        self.item_data = {}  # item_id -> {sid_list, sid_tokens, ...}
        self.user_sequences = {}  # user_id -> [item_id, ...]
        
        self.user_seq_len_before_round = {}
        # 统计信息
        self._num_items = 0
        self._num_users = 0
        
        # 调试模式
        self.debug = config.get('debug', False)
    
    def _debug_log(self, message: str):
        """调试输出。"""
        if self.debug:
            self.logger.info(f"[DEBUG] {message}")
    
    @property
    def num_items(self) -> int:
        return self._num_items
    
    @property
    def num_users(self) -> int:
        return self._num_users
    
    # def load_from_files(
    #     self,
    #     interaction_files: List[str],
    #     item_files: List[str],
    #     accumulated_user_sequences: Optional[Dict[int, List[int]]] = None
    # ) -> Dict[str, List[Dict]]:
    #     """
    #     从文件加载数据。
        
    #     Args:
    #         interaction_files: 交互文件路径列表
    #         item_files: 物品文件路径列表
    #         accumulated_user_sequences: 累积的用户序列（可选）
            
    #     Returns:
    #         Dict[str, List[Dict]]: 包含 train, val, test 的数据集
    #     """
    #     self.logger.info(f"Loading {len(interaction_files)} interaction files, {len(item_files)} item files")
    #     self.logger.info(f"Max history length: {self.max_history_len}, Accumulate: {self.accumulate_history}")
        
    #     # 1. 初始化用户历史
    #     if accumulated_user_sequences is not None and self.accumulate_history:
    #         self.user_sequences = {k: list(v) for k, v in accumulated_user_sequences.items()}
    #         self._debug_log(f"Loaded {len(self.user_sequences)} users from accumulated history")
    #     else:
    #         self.user_sequences = {}

    #     ####################
    #     if not self.user_seq_len_before_round:
    #         self.user_seq_len_before_round = {
    #             user_id: len(seq) for user_id, seq in self.user_sequences.items()
    #         }

    #     # 2. 加载 item 文件
    #     self._debug_log("=" * 50)
    #     self._debug_log("Loading item files...")
    #     for item_file in item_files:
    #         self._load_item_file(item_file)
        
    #     self._num_items = len(self.item_data)
    #     self.logger.info(f"Loaded {self._num_items} items with SID tokens")
        
    #     # 3. 加载 interaction 文件
    #     self._debug_log("=" * 50)
    #     self._debug_log("Loading interaction files...")
    #     new_interactions = []
    #     for inter_file in interaction_files:
    #         interactions = self._load_interaction_file(inter_file)
    #         new_interactions.extend(interactions)
        
    #     unique_users = len(set(i['user_id'] for i in new_interactions)) if new_interactions else 0
    #     self.logger.info(f"Loaded {len(new_interactions)} new interactions from {unique_users} users")
        
    #     # 4. 更新用户序列
    #     self._debug_log("=" * 50)
    #     self._debug_log("Updating user sequences...")
    #     self._update_user_sequences(new_interactions)
        
    #     self._num_users = len(self.user_sequences)
    #     self.logger.info(f"Built sequences for {self._num_users} users")
        
    #     # 5. 构建数据集
    #     self._debug_log("=" * 50)
    #     self._debug_log("Building datasets...")
    #     datasets = self._build_datasets(self.splits)
        
    #     self.logger.info(f"Built datasets: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")
        
    #     self.user_seq_len_before_round = {
    #         user_id: len(seq) for user_id, seq in self.user_sequences.items()
    #     }

    #     return datasets

    def load_from_files(self, interaction_files, item_files, history_store, splits=None):
        """
        Load interactions/items for one round, build round datasets,
        and update history_store in-place.

        Args:
            interaction_files (list[str]): list of interaction file paths for this round
            item_files (list[str]): list of item meta file paths for this round (rolling window)
            history_store (dict[user_id -> deque]): global history store (single copy)
            splits (list[str]): splits to build, default ['train', 'val', 'test']

        Returns:
            dict: {split_name: list of samples}
                sample = {
                    'user_id': ...,
                    'history_item_ids': [...],
                    'target_item': ...,
                    'is_click': ...,
                    'is_long_view': ...,
                    ...
                }
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        # 1) load item meta & interactions
        self.item_meta = self._load_item_meta(item_files)
        interactions = self._load_interactions(interaction_files)

        # 2) group new interactions by user (preserve order)
        per_user_new = defaultdict(list)
        for inter in interactions:
            per_user_new[inter['user_id']].append(inter)

        datasets = {s: [] for s in splits}

        # 3) build round samples per user
        for user_id, new_inters in per_user_new.items():
            if not new_inters:
                continue

            # history before this round (truncate by max_seq_len for input)
            base_seq = list(history_store.get(user_id, []))
            history = deque(base_seq, maxlen=self.max_seq_len)

            n = len(new_inters)
            for i, inter in enumerate(new_inters):
                item = inter['item_id']

                # build sample BEFORE appending current item
                if len(history) >= self.min_seq_len:
                    sample = {
                        'user_id': user_id,
                        'history_item_ids': list(history),
                        'target_item': item,
                    }

                    for col in BEHAVIOR_COLS:
                        sample[col] = inter.get(col, 0)

                    if i == n - 1 and 'test' in splits:
                        datasets['test'].append(sample)
                    elif i == n - 2 and n > 1 and 'val' in splits:
                        datasets['val'].append(sample)
                    elif 'train' in splits:
                        datasets['train'].append(sample)

                # update history for next step
                history.append(item)

            # 4) update global history_store (single copy)
            if self.accumulate_history:
                dq = history_store.setdefault(user_id, deque(maxlen=self.max_history_len))
                dq.extend([x['item_id'] for x in new_inters])

        # optional stats
        self._num_items = len(self.item_data) if hasattr(self, "item_data") else 0
        self._num_users = len(history_store)

        return datasets


    def remove_old_items(self, item_ids_to_remove: set):
        """
        移除指定的旧 items，释放内存。
        
        Args:
            item_ids_to_remove: 要移除的 item ID 集合
        """
        removed_count = 0
        for item_id in item_ids_to_remove:
            if item_id in self.item_data:
                del self.item_data[item_id]
                removed_count += 1
        
        self.logger.info(f"Removed {removed_count} old items from memory")
        self._num_items = len(self.item_data)

        
    def _load_item_file(self, filepath: str):
        """
        加载单个 item 文件。
        
        预期格式（TSV）:
            sid_list          emb_vec    item_id    tag_list
            417,224,387,...   ...        1398888    [1,2,3]
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Item file not found: {filepath}")
            return
        
        try:
            df = pd.read_csv(filepath, sep='\t')
            self._debug_log(f"Loading item file: {os.path.basename(filepath)}, rows: {len(df)}")
            self._debug_log(f"  Columns: {list(df.columns)}")
            
            # 查找 item_id 列
            item_id_col = self._find_column(df.columns, ['item_id', 'item', 'id', 'itemid'])
            if item_id_col is None:
                self.logger.warning(f"Cannot find item_id column in {filepath}")
                self.logger.warning(f"  Available columns: {list(df.columns)}")
                return
            
            # 查找 SID 列 - 支持 sid_list 列名
            sid_col = self._find_column(df.columns, [
                'sid_list', 'semantic_id', 'sid', 'sid_tokens', 
                'semantic_tokens', 'codes', 'semantic_code'
            ])
            if sid_col is None:
                self.logger.warning(f"Cannot find SID column in {filepath}")
                self.logger.warning(f"  Available columns: {list(df.columns)}")
                return
            
            self._debug_log(f"  Item ID col: '{item_id_col}', SID col: '{sid_col}'")
            
            # 解析每一行
            loaded_count = 0
            for idx, row in df.iterrows():
                try:
                    # 获取 item_id（可能是浮点数，需要转换为整数）
                    item_id_raw = row[item_id_col]
                    if pd.isna(item_id_raw):
                        continue
                    item_id = int(float(item_id_raw))
                    
                    # 获取 SID
                    sid_str = str(row[sid_col])
                    if pd.isna(row[sid_col]) or not sid_str:
                        continue
                    
                    # 解析 SID tokens
                    sid_tokens = self._parse_sid_tokens(sid_str)
                    
                    if sid_tokens:
                        self.item_data[item_id] = {
                            'sid_list': sid_str,
                            'sid_tokens': sid_tokens
                        }
                        loaded_count += 1
                        
                        # 打印前几个样本
                        #if loaded_count <= 1 and self.debug:
                        #    self._debug_log(f"    Sample: item_id={item_id}, tokens={sid_tokens}")
                
                except Exception as e:
                    if self.debug:
                        self._debug_log(f"    Error parsing row {idx}: {e}")
            
            self._debug_log(f"  Loaded {loaded_count} items from {os.path.basename(filepath)}")
            
        except Exception as e:
            self.logger.error(f"Error reading item file {filepath}: {e}")
    
    def _parse_sid_tokens(self, sid_str: str) -> List[str]:
        """
        解析 SID 字符串为 token 列表。
        
        支持的格式:
        1. "417,224,387,175,414,125,334,499" (逗号分隔的纯数字) - 你的数据格式
        2. "<a_417><b_224><c_387>..." (已格式化的 token)
        3. "a_417,b_224,c_387,..." (带前缀的逗号分隔)
        """
        if pd.isna(sid_str) or not sid_str:
            return []
        
        sid_str = str(sid_str).strip()
        
        # 格式 1: 逗号分隔的纯数字 "417,224,387,175,414,125,334,499"
        # 这是你的数据格式
        if re.match(r'^[\d,\s]+$', sid_str):
            numbers = [n.strip() for n in sid_str.split(',') if n.strip()]
            if numbers:
                level_chars = 'abcdefgh'
                tokens = []
                for i, num in enumerate(numbers[:8]):  # 最多 8 层
                    if i < len(level_chars):
                        tokens.append(f'<{level_chars[i]}_{num}>')
                return tokens
        
        # 格式 2: 已格式化的 token "<a_417><b_224>..."
        tokens = re.findall(r'<[a-h]_\d+>', sid_str)
        if tokens:
            return tokens
        
        # 格式 3: 带前缀的逗号分隔 "a_417,b_224,..."
        tokens = re.findall(r'[a-h]_\d+', sid_str)
        if tokens:
            return [f'<{t}>' for t in tokens]
        
        # 格式 4: 空格分隔的数字
        numbers = sid_str.split()
        if all(n.isdigit() for n in numbers) and len(numbers) >= 4:
            level_chars = 'abcdefgh'
            tokens = []
            for i, num in enumerate(numbers[:8]):
                if i < len(level_chars):
                    tokens.append(f'<{level_chars[i]}_{num}>')
            return tokens
        
        return []

    def _find_column(self, columns, candidates):
        """
        在 columns 中查找第一个匹配的候选列名。
        大小写不敏感。
        """
        if columns is None:
            return None

        col_map = {str(c).strip().lower(): c for c in columns}
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in col_map:
                return col_map[key]
        return None

    
    def _load_interaction_file(self, filepath: str) -> List[Dict]:
        """
        加载单个 interaction 文件。

        预期格式（TSV）:
            user    item    is_click    ...
            0       0       1           ...
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"Interaction file not found: {filepath}")
            return []

        try:
            df = pd.read_csv(filepath, sep='\t')
            self._debug_log(f"Loading interaction file: {os.path.basename(filepath)}, rows: {len(df)}")
            self._debug_log(f"  Columns: {list(df.columns)}")

            # 查找必需的列
            user_col = self._find_column(df.columns, ['user_id', 'user', 'uid', 'userid'])
            item_col = self._find_column(df.columns, ['item_id', 'item', 'itemid', 'iid'])

            if user_col is None:
                self.logger.warning(f"Cannot find user column in {filepath}")
                return []
            if item_col is None:
                self.logger.warning(f"Cannot find item column in {filepath}")
                return []

            self._debug_log(f"  User col: '{user_col}', Item col: '{item_col}'")

            # 查找任务列（可选）
            task_col = self._find_column(df.columns, [self.task, 'label', 'click', 'action'])
            if task_col:
                self._debug_log(f"  Task col: '{task_col}'")
                original_len = len(df)
                df = df[df[task_col] == 1]
                self._debug_log(f"  After filtering ({task_col}=1): {len(df)} rows (from {original_len})")

            interactions = []
            skipped_no_sid = 0

            for _, row in df.iterrows():
                try:
                    user_id = int(row[user_col])
                    item_id = int(row[item_col])
                except Exception:
                    continue

                # 只保留有 SID 的 item
                if item_id not in self.item_data:
                    skipped_no_sid += 1
                    continue

                inter = {
                    'user_id': user_id,
                    'item_id': item_id,
                }

                for col in BEHAVIOR_COLS:
                    if col in df.columns:
                        inter[col] = row[col]
                    else:
                        inter[col] = 0

                interactions.append(inter)

            self._debug_log(f"  Valid interactions: {len(interactions)}, Skipped (no SID): {skipped_no_sid}")

            if len(interactions) == 0 and len(df) > 0:
                sample_items = df[item_col].head(10).tolist()
                self._debug_log(f"  WARNING: No valid interactions!")
                self._debug_log(f"    Sample item IDs from interaction file: {sample_items}")
                self._debug_log(f"    Sample item IDs in item_data: {list(self.item_data.keys())[:10]}")
                self._debug_log(f"    Total items in item_data: {len(self.item_data)}")

            return interactions

        except Exception as e:
            self.logger.error(f"Error reading interaction file {filepath}: {e}")
            return []

    
    def _find_column(self, columns, possible_names: List[str]) -> Optional[str]:
        """查找列名（大小写不敏感）。"""
        columns_lower = {c.lower(): c for c in columns}
        for name in possible_names:
            if name.lower() in columns_lower:
                return columns_lower[name.lower()]
        return None
    
    def _update_user_sequences(self):
        """截断过长的用户序列。"""
        truncated_count = 0
        for user_id in self.user_sequences:
            if len(self.user_sequences[user_id]) > self.max_history_len:
                self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]
                truncated_count += 1
        
        if truncated_count > 0:
            self._debug_log(f"Truncated {truncated_count} users by max_history_len={self.max_history_len}")


    # def _update_user_sequences(self, interactions: List[Dict]):
    #     """更新用户序列。"""
    #     for inter in interactions:
    #         user_id = inter['user_id']
    #         item_id = inter['item_id']
            
    #         if user_id not in self.user_sequences:
    #             self.user_sequences[user_id] = []
            
    #         self.user_sequences[user_id].append(item_id)
        
    #     # ========== 添加调试输出 ==========
    #     # truncated_count = 0
    #     # for user_id in self.user_sequences:
    #     #     original_len = len(self.user_sequences[user_id])
    #     #     if original_len > self.max_history_len:
    #     #         self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]
    #     #         truncated_count += 1
    #     #         # 打印第一个被截断的用户作为示例
    #     #         if truncated_count == 1:
    #     #             print(f"\n[max_history_len 作用示例]")
    #     #             print(f"  用户 {user_id} 原始历史长度: {original_len}")
    #     #             print(f"  max_history_len: {self.max_history_len}")
    #     #             print(f"  截断后长度: {len(self.user_sequences[user_id])}")
        
    #     # if truncated_count > 0:
    #     #     print(f"  共 {truncated_count} 个用户的历史被 max_history_len 截断")
    #     # ========== 调试输出结束 ==========
        
    #     # 截断过长的序列
    #     # for user_id in self.user_sequences:
    #     #     if len(self.user_sequences[user_id]) > self.max_history_len:
    #     #         self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]

    #     truncated_count = 0
    #     for user_id in self.user_sequences:
    #         if len(self.user_sequences[user_id]) > self.max_history_len:
    #             self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]
    #             truncated_count += 1
        
    #     if truncated_count > 0:
    #         self._debug_log(f"Truncated {truncated_count} users by max_history_len={self.max_history_len}")

    #     # self.user_seq_len_before_round = {
    #     #     user_id: len(seq) for user_id, seq in self.user_sequences.items()
    #     # }
    #     # self._debug_log(f"Recorded sequence lengths for {len(self.user_seq_len_before_round)} users")


    # def _build_datasets(self, splits: List[str] = None) -> Dict[str, List[Dict]]:
    #     """构建训练/验证/测试集。"""
    #     if splits is None:
    #         splits = ['train', 'val', 'test']
        
    #     train_data = []
    #     val_data = []
    #     test_data = []
        
    #     for user_id, seq in self.user_sequences.items():
    #         prev_len = self.user_seq_len_before_round.get(user_id, 0)
    #         current_len = len(seq)
    #         new_items_count = current_len - prev_len
    def _build_datasets(
        self,
        splits: List[str] = None,
        user_seq_len_before_loading: Dict[int, int] = None,
        user_seq_len_after_loading: Dict[int, int] = None
    ) -> Dict[str, List[Dict]]:
        """构建数据集（只保存索引，不保存完整序列）。"""
        
        if splits is None:
            splits = ['train', 'val', 'test']
        
        datasets = {split: [] for split in splits}
        
        for user_id, seq in self.user_sequences.items():
            prev_len = user_seq_len_before_loading.get(user_id, 0) if user_seq_len_before_loading else 0
            current_len = user_seq_len_after_loading.get(user_id, len(seq)) if user_seq_len_after_loading else len(seq)
            
            new_items_count = current_len - prev_len
            
            if new_items_count <= 0:
                continue
            
            # 为每个新增交互生成样本
            for i in range(new_items_count):
                target_idx = prev_len + i
                target_item = seq[target_idx]
                
                # ===== 关键修改：只保存索引，不保存完整序列 =====
                sample = {
                    'user_id': user_id,
                    'target_idx': target_idx,  # 目标位置的索引
                    'target_item': target_item,
                    'user': user_id,  # 兼容旧代码
                    'next': target_item,  # 兼容旧代码
                }
                
                # 分配到train/val/test
                if i == new_items_count - 1 and 'test' in splits:
                    datasets['test'].append(sample)
                elif i == new_items_count - 2 and new_items_count > 1 and 'val' in splits:
                    datasets['val'].append(sample)
                elif 'train' in splits:
                    datasets['train'].append(sample)
        
        return datasets



        #     if 'test' in splits:
        #         test_data.append({
        #             'user': user_id,
        #             'seq': seq[:-1][-self.max_seq_len:],
        #             'len_seq': min(len(seq) - 1, self.max_seq_len),
        #             'next': seq[-1],
        #             'target_item': seq[-1]
        #         })
            
        #     # 验证集
        #     if 'val' in splits:
        #         val_data.append({
        #             'user': user_id,
        #             'seq': seq[:-2][-self.max_seq_len:],
        #             'len_seq': min(len(seq) - 2, self.max_seq_len),
        #             'next': seq[-2],
        #             'target_item': seq[-2]
        #         })

        #     # 训练集
        #     if 'train' in splits:
        #         train_item_ids = seq[:-2]
        #         if len(train_item_ids) >= 2:
        #             for i in range(self.min_seq_len, len(train_item_ids)):
        #                 input_seq = train_item_ids[:i]
        #                 if len(input_seq) > self.max_seq_len:
        #                     input_seq = input_seq[-self.max_seq_len:]
                        
        #                 train_data.append({
        #                     'user': user_id,
        #                     'seq': input_seq,
        #                     'len_seq': len(input_seq),
        #                     'next': train_item_ids[i],
        #                     'target_item': train_item_ids[i]
        #                 })
                    
        #     debug_printed = True  # 只打印第一个用户的样本
        
        # return {
        #     'train': train_data,
        #     'val': val_data,
        #     'test': test_data
        # }

    
    def get_accumulated_sequences(self) -> Dict[int, List[int]]:
        """获取累积的用户序列。"""
        return {k: list(v) for k, v in self.user_sequences.items()}
    
    def get_item_sid_tokens(self, item_id: int) -> List[str]:
        """
        获取物品的 SID tokens。
        
        Args:
            item_id: 物品 ID
            
        Returns:
            List[str]: SID token 列表，如 ['<a_417>', '<b_224>', '<c_387>', ...]
        """
        if item_id in self.item_data:
            return self.item_data[item_id].get('sid_tokens', [])
        return []


    def _load_item_meta(self, item_files: List[str]) -> Dict:
        """
        加载所有 item 文件并返回 item 数据字典。
        这是 load_from_files 缺失的辅助方法。
        """
        self._debug_log("=" * 50)
        self._debug_log("Loading item files...")
        
        # 遍历文件列表，调用已有的单文件处理方法
        for item_file in item_files:
            self._load_item_file(item_file)
            
        self._num_items = len(self.item_data)
        self.logger.info(f"Loaded {self._num_items} items with SID tokens")
        
        # 返回加载后的 item_data
        return self.item_data

    def _load_interactions(self, interaction_files: List[str]) -> List[Dict]:
        """
        加载所有 interaction 文件并返回合并后的交互列表。
        这是 load_from_files 缺失的辅助方法。
        """
        self._debug_log("=" * 50)
        self._debug_log("Loading interaction files...")
        
        all_interactions = []
        for inter_file in interaction_files:
            # 调用已有的单文件处理方法
            interactions = self._load_interaction_file(inter_file)
            all_interactions.extend(interactions)
            
        unique_users = len(set(i['user_id'] for i in all_interactions)) if all_interactions else 0
        self.logger.info(f"Loaded {len(all_interactions)} interactions from {unique_users} unique users")
        
        return all_interactions
    
    def _load_item_files(self, item_files: List[str]):
        """
        加载 item 文件，兼容多种列名：
        - item_id / item / itemid / iid
        - sid_list / sid / semantic_id
        - emb_vec / emb / embedding
        - tag_list / tag / tags
        """
        import pandas as pd
        import numpy as np

        self.logger.info(f"Loading {len(item_files)} item files...")
        self.item_data = {}

        total_loaded = 0
        total_skipped = 0

        for item_file in item_files:
            if not os.path.exists(item_file):
                self.logger.warning(f"Item file not found: {item_file}")
                continue

            self.logger.info(f"Loading item file: {item_file}")

            preview_df = pd.read_csv(item_file, sep="\t", nrows=5)
            cols = list(preview_df.columns)
            self.logger.info(f"[DEBUG] item file columns: {cols}")

            item_col = self._find_column(cols, ['item_id', 'item', 'itemid', 'iid'])
            sid_col = self._find_column(cols, ['sid_list', 'sid', 'semantic_id'])
            tag_col = self._find_column(cols, ['tag_list', 'tag', 'tags'])
            emb_col = self._find_column(cols, ['emb_vec', 'emb', 'embedding'])

            if item_col is None:
                raise ValueError(f"{item_file} missing item column, columns={cols}")
            if sid_col is None:
                raise ValueError(f"{item_file} missing sid column, columns={cols}")

            usecols = [item_col, sid_col]
            if tag_col is not None:
                usecols.append(tag_col)
            if emb_col is not None:
                usecols.append(emb_col)

            df = pd.read_csv(item_file, sep="\t", usecols=usecols)

            # item_id -> int
            df[item_col] = pd.to_numeric(df[item_col], errors="coerce")
            bad_item_rows = int(df[item_col].isna().sum())
            if bad_item_rows > 0:
                self.logger.warning(f"{item_file}: bad item rows = {bad_item_rows}")

            df = df.dropna(subset=[item_col]).copy()
            df[item_col] = df[item_col].astype(np.int64)

            loaded_cnt = 0
            skipped_cnt = 0

            for row in df.itertuples(index=False):
                row_dict = row._asdict()

                try:
                    item_id = int(row_dict[item_col])
                except Exception:
                    skipped_cnt += 1
                    continue

                sid_raw = str(row_dict[sid_col]).strip()
                if sid_raw == "" or sid_raw.lower() == "nan":
                    skipped_cnt += 1
                    continue

                # 兼容 sid 格式：
                # 1) "417,224,387,175,..."
                # 2) "<a_417><b_224>..."
                if "<a_" in sid_raw or "<b_" in sid_raw or "<c_" in sid_raw:
                    # 已经是 token 串
                    sid_tokens = []
                    import re
                    sid_tokens = re.findall(r"<[a-z]_\d+>", sid_raw)
                    if len(sid_tokens) == 0:
                        skipped_cnt += 1
                        continue
                else:
                    sid_codes = [x.strip() for x in sid_raw.split(",") if x.strip() != ""]
                    if len(sid_codes) == 0:
                        skipped_cnt += 1
                        continue

                    level_chars = "abcdefghijklmnopqrstuvwxyz"
                    sid_tokens = []

                    for idx, code in enumerate(sid_codes):
                        if idx >= len(level_chars):
                            break
                        try:
                            code_int = int(code)
                        except Exception:
                            continue
                        sid_tokens.append(f"<{level_chars[idx]}_{code_int}>")

                    if len(sid_tokens) == 0:
                        skipped_cnt += 1
                        continue

                item_info = {
                    "sid_list": sid_raw,
                    "sid_tokens": sid_tokens,
                }

                if tag_col is not None:
                    item_info["tag"] = row_dict.get(tag_col, None)
                if emb_col is not None:
                    item_info["emb"] = row_dict.get(emb_col, None)

                self.item_data[item_id] = item_info
                loaded_cnt += 1

            total_loaded += loaded_cnt
            total_skipped += skipped_cnt

            self.logger.info(
                f"Loaded item file {os.path.basename(item_file)}: "
                f"loaded={loaded_cnt}, skipped={skipped_cnt}"
            )

        self._num_items = len(self.item_data)

        self.logger.info(f"Total loaded items: {self._num_items}")
        self.logger.info(f"Total loaded rows: {total_loaded}, total skipped rows: {total_skipped}")

        if self._num_items > 0:
            sample_keys = list(self.item_data.keys())[:5]
            self.logger.info(f"[DEBUG] sample item_data keys: {sample_keys}")
            self.logger.info(f"[DEBUG] sample key types: {[type(x) for x in sample_keys]}")
            for k in sample_keys[:3]:
                self.logger.info(f"[DEBUG] item {k}: {self.item_data[k]}")


    def load_train_only_from_files(
        self,
        interaction_files: List[str],
        item_files: List[str],
        history_store: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        full_user 模式专用：
        - 加载 item 文件
        - 按 chunk 流式读取 interaction 文件
        - interaction 文件已按 user + time 排序
        - 每个用户只构造 1 条训练样本
        - 随机截断一段前缀作为 history，下一项作为 target
        - 兼容多种列名
        """
        self.logger.info("Loading full_user train-only data by chunk (one random sample per user)...")

        import gc
        import pandas as pd
        import numpy as np

        seed = self.config.get("train_sample_seed", 2025)
        rng = np.random.default_rng(seed)

        # 1. 加载 item 文件
        self._load_item_files(item_files)
        valid_item_set = set(self.item_data.keys())

        self.logger.info(f"[full_user] Loaded item vocab size: {len(valid_item_set)}")
        if len(valid_item_set) == 0:
            self.logger.warning("Item vocab is empty.")
            return {"train": []}

        # 2. 参数
        chunk_size = int(self.config.get("interaction_chunk_size", 2_000_000))
        min_required_len = self.min_seq_len + 1

        train_samples = []
        skipped_too_short = 0
        sampled_users = 0

        total_rows_read = 0
        total_rows_after_positive = 0
        total_rows_after_item_filter = 0

        all_users_seen = set()

        current_user = None
        current_seq = []

        def flush_one_user(user_id, item_seq):
            nonlocal skipped_too_short, sampled_users, train_samples

            if user_id is None:
                return

            seq_len = len(item_seq)
            if seq_len < min_required_len:
                skipped_too_short += 1
                return

            cut = int(rng.integers(self.min_seq_len, seq_len))
            history = item_seq[:cut]
            target_item = item_seq[cut]

            train_samples.append({
                "user_id": int(user_id),
                "history_item_ids": history[-self.max_seq_len:],
                "target_item": int(target_item),
            })
            sampled_users += 1

        for interaction_file in interaction_files:
            if not os.path.exists(interaction_file):
                raise FileNotFoundError(f"Interaction file not found: {interaction_file}")

            self.logger.info(f"[full_user] Streaming interaction file: {interaction_file}")

            preview_df = pd.read_csv(interaction_file, sep="\t", nrows=5)
            cols = list(preview_df.columns)
            self.logger.info(f"[DEBUG] interaction file columns: {cols}")

            user_col = self._find_column(cols, ['user_id', 'user', 'uid', 'userid'])
            item_col = self._find_column(cols, ['item_id', 'item', 'itemid', 'iid'])
            task_col = None

            if self.task in cols:
                task_col = self.task
            else:
                task_col = self._find_column(cols, ['is_click', 'label', 'target'])

            if user_col is None or item_col is None:
                raise ValueError(
                    f"{interaction_file} missing user/item column, "
                    f"columns={cols}, user_col={user_col}, item_col={item_col}"
                )

            usecols = [user_col, item_col]
            if task_col is not None:
                usecols.append(task_col)

            reader = pd.read_csv(
                interaction_file,
                sep="\t",
                usecols=usecols,
                chunksize=chunk_size,
            )

            for chunk_idx, chunk in enumerate(reader):
                total_rows_read += len(chunk)

                # 正样本过滤
                if task_col is not None:
                    chunk = chunk[chunk[task_col] == 1]

                total_rows_after_positive += len(chunk)

                if len(chunk) == 0:
                    continue

                # 重命名成统一列名
                rename_map = {
                    user_col: "user",
                    item_col: "item",
                }
                chunk = chunk.rename(columns=rename_map)

                chunk = chunk[["user", "item"]].copy()

                chunk["user"] = pd.to_numeric(chunk["user"], errors="coerce")
                chunk["item"] = pd.to_numeric(chunk["item"], errors="coerce")
                chunk = chunk.dropna(subset=["user", "item"])

                if len(chunk) == 0:
                    continue

                chunk["user"] = chunk["user"].astype(np.int64)
                chunk["item"] = chunk["item"].astype(np.int64)

                before_item_filter = len(chunk)
                chunk = chunk[chunk["item"].isin(valid_item_set)]
                after_item_filter = len(chunk)
                total_rows_after_item_filter += after_item_filter

                if chunk_idx % 10 == 0:
                    self.logger.info(
                        f"[full_user] file={os.path.basename(interaction_file)}, "
                        f"chunk={chunk_idx}, rows={before_item_filter}, kept={after_item_filter}"
                    )

                if len(chunk) == 0:
                    continue

                users_np = chunk["user"].to_numpy()
                items_np = chunk["item"].to_numpy()

                for u, it in zip(users_np, items_np):
                    all_users_seen.add(int(u))

                    if current_user is None:
                        current_user = u
                        current_seq = [int(it)]
                    elif u == current_user:
                        current_seq.append(int(it))
                    else:
                        flush_one_user(current_user, current_seq)
                        current_user = u
                        current_seq = [int(it)]

                del chunk
                del users_np
                del items_np
                gc.collect()

        flush_one_user(current_user, current_seq)

        self._num_users = len(all_users_seen)
        self._num_items = len(self.item_data)

        self.logger.info("[full_user] Built one-sample-per-user training dataset (chunk mode)")
        self.logger.info(f"  - Total rows read: {total_rows_read}")
        self.logger.info(f"  - Total rows after positive filter: {total_rows_after_positive}")
        self.logger.info(f"  - Total rows after item filter: {total_rows_after_item_filter}")
        self.logger.info(f"  - Users in current part: {len(all_users_seen)}")
        self.logger.info(f"  - Sampled users: {sampled_users}")
        self.logger.info(f"  - Train samples: {len(train_samples)}")
        self.logger.info(f"  - Skipped too short users: {skipped_too_short}")
        self.logger.info(f"  - Num items loaded: {self._num_items}")
        self.logger.info(
            f"  - Skipped (no SID): {total_rows_after_positive - total_rows_after_item_filter}"
        )

        return {"train": train_samples}
