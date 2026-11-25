# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming interaction dataset

Supported formats:
- Load data by day
- Accumulate user histories
- Automatically build training, validation, and test datasets

Data format:
- Item files: sid_list, emb_vec, item_id, tag_list
  - sid_list: "417,224,387,175,414,125,334,499" (comma-separated numbers)
- Interaction files: user, item, is_click, ...
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from logging import getLogger
from collections import defaultdict


class StreamingInteractionDataset:
    """
    Streaming interaction dataset
    
    Supported formats:
    - Load data by day
    - Accumulate user histories
    - Automatically build training, validation, and test datasets
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
        Initialize the dataset.
        
        Args:
            config: configuration dictionary
            task: task type used to filter interactions
            max_seq_len: maximum sequence length
            min_seq_len: minimum sequence length; filter users with shorter histories
            max_history_len: maximum history length
            accumulate_history: whether to accumulate history
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

        # Data storage
        
        self.item_data = {}  # item_id -> {sid_list, sid_tokens, ...}
        self.user_sequences = {}  # user_id -> [item_id, ...]
        
        self.user_seq_len_before_round = {}
        # Statistics
        self._num_items = 0
        self._num_users = 0
        
        # Debug mode
        self.debug = config.get('debug', False)
    
    def _debug_log(self, message: str):
        """Debug output."""
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
    # Load data from files.
        
    #     Args:
    # interaction_files: list of interaction file paths
    # item_files: list of item file paths
    # accumulated_user_sequences: optional accumulated user sequences
            
    #     Returns:
    # Dict[str, List[Dict]]: datasets containing train, val, and test
    #     """
    #     self.logger.info(f"Loading {len(interaction_files)} interaction files, {len(item_files)} item files")
    #     self.logger.info(f"Max history length: {self.max_history_len}, Accumulate: {self.accumulate_history}")
        

    #     if not self.user_seq_len_before_round:
    #         self.user_seq_len_before_round = {
    #             user_id: len(seq) for user_id, seq in self.user_sequences.items()
    #         }

        
        
        
        
        
        
        
    #     self.logger.info(f"Built datasets: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")
        

    #     return datasets

    def load_from_files(
        self,
        interaction_files: List[str],
        item_files: List[str],
        accumulated_user_sequences: Optional[Dict[str, List[str]]] = None,
        splits: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load data from files and build datasets.
        
        Args:
            interaction_files: list of interaction files
            item_files: list of item files
            accumulated_user_sequences: accumulated user sequences from the previous round
            splits: dataset splits to build; defaults to ['train', 'val', 'test']
        
        Returns:
            Dictionary containing data for each split
        """
        # 1. Initialize user history
        if accumulated_user_sequences is not None and self.accumulate_history:
            if not self._accumulated_loaded:
                self.user_sequences = {k: list(v) for k, v in accumulated_user_sequences.items()}
                self._debug_log(f"Loaded {len(self.user_sequences)} users from accumulated history")
                self._accumulated_loaded = True
        else:
            self.user_sequences = {}
            self._accumulated_loaded = False
        
        # 2. Record sequence lengths at the start of this round, after previous-round truncation
        user_seq_len_before_loading = {
            user_id: len(seq) for user_id, seq in self.user_sequences.items()
        }
        
        # 3. Load item metadata
        self.item_meta = self._load_item_meta(item_files)
        self._debug_log(f"Loaded {len(self.item_meta)} items")
        
        # 4. Load interaction data
        new_interactions = self._load_interactions(interaction_files)
        self._debug_log(f"Loaded {len(new_interactions)} new interactions")
        
        # 5. Append new interactions to user sequences without truncation
        for inter in new_interactions:
            user_id = inter['user_id']
            item_id = inter['item_id']
            
            if user_id not in self.user_sequences:
                self.user_sequences[user_id] = []
            
            self.user_sequences[user_id].append(item_id)
        
        # 6. Record lengths after loading new interactions and before truncation
        user_seq_len_after_loading = {
            user_id: len(seq) for user_id, seq in self.user_sequences.items()
        }

        # 8. Build datasets using pre-truncation lengths to calculate new_items_count
        datasets = self._build_datasets(
            splits=splits,
            user_seq_len_before_loading=user_seq_len_before_loading,
            user_seq_len_after_loading=user_seq_len_after_loading
        )
        
        # 7. Truncate long sequences
        self._update_user_sequences()

        # 9. Save truncated lengths for the next round
        self.user_seq_len_before_round = {
            user_id: len(seq) for user_id, seq in self.user_sequences.items()
        }
        
        return datasets
        
    def _load_item_file(self, filepath: str):
        """
        Load one item file.
        
        Expected TSV format:
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
            
            # Find the item_id column
            item_id_col = self._find_column(df.columns, ['item_id', 'item', 'id', 'itemid'])
            if item_id_col is None:
                self.logger.warning(f"Cannot find item_id column in {filepath}")
                self.logger.warning(f"  Available columns: {list(df.columns)}")
                return
            
            # Find the SID column, including sid_list
            sid_col = self._find_column(df.columns, [
                'sid_list', 'semantic_id', 'sid', 'sid_tokens', 
                'semantic_tokens', 'codes', 'semantic_code'
            ])
            if sid_col is None:
                self.logger.warning(f"Cannot find SID column in {filepath}")
                self.logger.warning(f"  Available columns: {list(df.columns)}")
                return
            
            self._debug_log(f"  Item ID col: '{item_id_col}', SID col: '{sid_col}'")
            
            # Parse each row
            loaded_count = 0
            for idx, row in df.iterrows():
                try:
                    # Get item_id, converting floating-point values to integers if necessary
                    item_id_raw = row[item_id_col]
                    if pd.isna(item_id_raw):
                        continue
                    item_id = int(float(item_id_raw))
                    
                    # Get the SID
                    sid_str = str(row[sid_col])
                    if pd.isna(row[sid_col]) or not sid_str:
                        continue
                    
                    # Parse SID tokens
                    sid_tokens = self._parse_sid_tokens(sid_str)
                    
                    if sid_tokens:
                        self.item_data[item_id] = {
                            'sid_list': sid_str,
                            'sid_tokens': sid_tokens
                        }
                        loaded_count += 1
                        
                
                except Exception as e:
                    if self.debug:
                        self._debug_log(f"    Error parsing row {idx}: {e}")
            
            self._debug_log(f"  Loaded {loaded_count} items from {os.path.basename(filepath)}")
            
        except Exception as e:
            self.logger.error(f"Error reading item file {filepath}: {e}")
    
    def _parse_sid_tokens(self, sid_str: str) -> List[str]:
        """
        Parse a SID string into a token list.
        
        Supported formats:
        1. "417,224,387,175,414,125,334,499" (comma-separated numbers)
        2. "<a_417><b_224><c_387>..." (formatted tokens)
        3. "a_417,b_224,c_387,..." (prefixed comma-separated values)
        """
        if pd.isna(sid_str) or not sid_str:
            return []
        
        sid_str = str(sid_str).strip()
        
        # Format 1: comma-separated numbers "417,224,387,175,414,125,334,499"
        # This is the expected data format
        if re.match(r'^[\d,\s]+$', sid_str):
            numbers = [n.strip() for n in sid_str.split(',') if n.strip()]
            if numbers:
                level_chars = 'abcdefgh'
                tokens = []
                for i, num in enumerate(numbers[:8]):  # At most eight levels
                    if i < len(level_chars):
                        tokens.append(f'<{level_chars[i]}_{num}>')
                return tokens
        
        # Format 2: formatted tokens "<a_417><b_224>..."
        tokens = re.findall(r'<[a-h]_\d+>', sid_str)
        if tokens:
            return tokens
        
        # Format 3: prefixed comma-separated values "a_417,b_224,..."
        tokens = re.findall(r'[a-h]_\d+', sid_str)
        if tokens:
            return [f'<{t}>' for t in tokens]
        
        # Format 4: space-separated numbers
        numbers = sid_str.split()
        if all(n.isdigit() for n in numbers) and len(numbers) >= 4:
            level_chars = 'abcdefgh'
            tokens = []
            for i, num in enumerate(numbers[:8]):
                if i < len(level_chars):
                    tokens.append(f'<{level_chars[i]}_{num}>')
            return tokens
        
        return []
    
    def _load_interaction_file(self, filepath: str) -> List[Dict]:
        """
        Load one interaction file.
        
        Expected TSV format:
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
            
            # Find required columns
            user_col = self._find_column(df.columns, ['user_id', 'user', 'uid', 'userid'])
            item_col = self._find_column(df.columns, ['item_id', 'item', 'itemid', 'iid'])
            
            if user_col is None:
                self.logger.warning(f"Cannot find user column in {filepath}")
                return []
            if item_col is None:
                self.logger.warning(f"Cannot find item column in {filepath}")
                return []
            
            self._debug_log(f"  User col: '{user_col}', Item col: '{item_col}'")
            
            # Find the optional task column
            task_col = self._find_column(df.columns, [self.task, 'label', 'click', 'action'])
            if task_col:
                self._debug_log(f"  Task col: '{task_col}'")
                # Filter positive samples
                original_len = len(df)
                df = df[df[task_col] == 1]
                self._debug_log(f"  After filtering ({task_col}=1): {len(df)} rows (from {original_len})")
            
            # Convert to an interaction list
            interactions = []
            skipped_no_sid = 0
            
            for _, row in df.iterrows():
                user_id = int(row[user_col])
                item_id = int(row[item_col])
                
                # Keep only items with SIDs
                if item_id in self.item_data:
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item_id
                    })
                else:
                    skipped_no_sid += 1
            
            self._debug_log(f"  Valid interactions: {len(interactions)}, Skipped (no SID): {skipped_no_sid}")
            
            if len(interactions) == 0 and len(df) > 0:
                # Debug information
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
        """Find a column name case-insensitively."""
        columns_lower = {c.lower(): c for c in columns}
        for name in possible_names:
            if name.lower() in columns_lower:
                return columns_lower[name.lower()]
        return None
    
    def _update_user_sequences(self):
        """Truncate long user sequences."""
        truncated_count = 0
        for user_id in self.user_sequences:
            if len(self.user_sequences[user_id]) > self.max_history_len:
                self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]
                truncated_count += 1
        
        if truncated_count > 0:
            self._debug_log(f"Truncated {truncated_count} users by max_history_len={self.max_history_len}")


    # def _update_user_sequences(self, interactions: List[Dict]):
    # """Update user sequences."""
    #     for inter in interactions:
    #         user_id = inter['user_id']
    #         item_id = inter['item_id']
            
            
    #         self.user_sequences[user_id].append(item_id)
        
    # # ========== Add debug output ==========
    #     # truncated_count = 0
    #     # for user_id in self.user_sequences:
    #     #     original_len = len(self.user_sequences[user_id])
    #     #     if original_len > self.max_history_len:
    #     #         self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]
    #     #         truncated_count += 1
    # #         # Print the first truncated user as an example
    #     #         if truncated_count == 1:
    # #             print(f"\n[max_history_len example]")
    # #             print(f"  User {user_id} original history length: {original_len}")
    #     #             print(f"  max_history_len: {self.max_history_len}")
    # #             print(f"  Truncated length: {len(self.user_sequences[user_id])}")
        
    #     # if truncated_count > 0:
    # #     print(f"  Histories truncated by max_history_len: {truncated_count} users")
    # # ========== End debug output ==========
        
    # # Truncate long sequences
    #     # for user_id in self.user_sequences:
    #     #     if len(self.user_sequences[user_id]) > self.max_history_len:
    #     #         self.user_sequences[user_id] = self.user_sequences[user_id][-self.max_history_len:]

        

    #     # self.user_seq_len_before_round = {
    #     #     user_id: len(seq) for user_id, seq in self.user_sequences.items()
    #     # }
    #     # self._debug_log(f"Recorded sequence lengths for {len(self.user_seq_len_before_round)} users")


    # def _build_datasets(self, splits: List[str] = None) -> Dict[str, List[Dict]]:
    # """Build training, validation, and test datasets."""
    #     if splits is None:
    #         splits = ['train', 'val', 'test']
        
        
    def _build_datasets(
        self, 
        splits: List[str] = None,
        user_seq_len_before_loading: Dict[str, int] = None,
        user_seq_len_after_loading: Dict[str, int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Build training, validation, and test datasets.
        
        Args:
            splits: dataset splits to build
            user_seq_len_before_loading: lengths before loading new interactions
            user_seq_len_after_loading: lengths after loading new interactions, before truncation
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        if user_seq_len_before_loading is None:
            user_seq_len_before_loading = {}
        if user_seq_len_after_loading is None:
            user_seq_len_after_loading = {}
        
        train_data = []
        val_data = []
        test_data = []
        
        for user_id, seq in self.user_sequences.items():
            # Use pre-truncation lengths to calculate new_items_count
            prev_len = user_seq_len_before_loading.get(user_id, 0)
            current_len = user_seq_len_after_loading.get(user_id, len(seq))
            new_items_count = current_len - prev_len
                
            if new_items_count <= 0:
                continue
            
            # Calculate the history start position
            # Use only the last max_seq_len items from the previous round as the history base
            history_start = max(0, prev_len - self.max_seq_len)
            
            # Generate a sample for each new interaction
            for i in range(new_items_count):
                target_idx = prev_len + i
                target_item = seq[target_idx]
                
                # History = previous round's last max_seq_len items + interactions processed in this round
                history = seq[history_start:target_idx]

                if len(history) > self.max_seq_len:
                    history = history[-self.max_seq_len:]
                    
                if len(history) < self.min_seq_len:
                    continue
                
                sample = {
                    'user': user_id,
                    'seq': list(history),
                    'len_seq': len(history),
                    'next': target_item,
                    'target_item': target_item
                }
                
                # Assign to train/val/test
                if i == new_items_count - 1 and 'test' in splits:
                    test_data.append(sample)
                elif i == new_items_count - 2 and new_items_count > 1 and 'val' in splits:
                    val_data.append(sample)
                elif 'train' in splits:
                    train_data.append(sample)
        
        return {'train': train_data, 'val': val_data, 'test': test_data}


            
        # # Validation set
        #     if 'val' in splits:
        #         val_data.append({
        #             'user': user_id,
        #             'seq': seq[:-2][-self.max_seq_len:],
        #             'len_seq': min(len(seq) - 2, self.max_seq_len),
        #             'next': seq[-2],
        #             'target_item': seq[-2]
        #         })

        # # Training set
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
                    
        
        # return {
        #     'train': train_data,
        #     'val': val_data,
        #     'test': test_data
        # }

    
    def get_accumulated_sequences(self) -> Dict[int, List[int]]:
        """Get accumulated user sequences."""
        return {k: list(v) for k, v in self.user_sequences.items()}
    
    def get_item_sid_tokens(self, item_id: int) -> List[str]:
        """
        Get SID tokens for an item.
        
        Args:
            item_id: item ID
            
        Returns:
            List[str]: SID tokens such as ['<a_417>', '<b_224>', '<c_387>', ...]
        """
        if item_id in self.item_data:
            return self.item_data[item_id].get('sid_tokens', [])
        return []


    def _load_item_meta(self, item_files: List[str]) -> Dict:
        """
        Load all item files and return the item dictionary.
        Helper method required by load_from_files.
        """
        self._debug_log("=" * 50)
        self._debug_log("Loading item files...")
        
        # Iterate over files using the existing single-file loader
        for item_file in item_files:
            self._load_item_file(item_file)
            
        self._num_items = len(self.item_data)
        self.logger.info(f"Loaded {self._num_items} items with SID tokens")
        
        # Return loaded item_data
        return self.item_data

    def _load_interactions(self, interaction_files: List[str]) -> List[Dict]:
        """
        Load all interaction files and return the merged interaction list.
        Helper method required by load_from_files.
        """
        self._debug_log("=" * 50)
        self._debug_log("Loading interaction files...")
        
        all_interactions = []
        for inter_file in interaction_files:
            # Call the existing single-file loader
            interactions = self._load_interaction_file(inter_file)
            all_interactions.extend(interactions)
            
        unique_users = len(set(i['user_id'] for i in all_interactions)) if all_interactions else 0
        self.logger.info(f"Loaded {len(all_interactions)} interactions from {unique_users} unique users")
        
        return all_interactions