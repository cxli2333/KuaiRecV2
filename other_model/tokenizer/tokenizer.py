import os
import torch
import pickle
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from logging import getLogger

from dataset.dataset import AbstractDataset

def stack_to_tensor(seq, dtype=None):
    """将列表 stack 成 Tensor；若传入 dtype 则转换类型。"""
    out = torch.stack(seq) if torch.is_tensor(seq[0]) else torch.as_tensor(seq)
    return out.to(dtype) if dtype else out


def collate_fn_with_labels(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    训练时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - decoder_input_ids: decoder输入 [n_digit]
            - decoder_labels: decoder标签 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'input_ids': stack_to_tensor([b['history_tokens'] for b in batch]),                      # [B, S, n_digit]
        'attention_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),  # [B, S]
        'labels': stack_to_tensor([b['target_tokens'] for b in batch]),                # [B, n_digit]
        'sid_labels': stack_to_tensor([b['target_sid'] for b in batch]),
        'id_labels': stack_to_tensor([b['target_id'] for b in batch]),
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    验证时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - labels: 真标签序列 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'input_ids': stack_to_tensor([b['history_tokens'] for b in batch]),
        'attention_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'labels': stack_to_tensor([b['target_tokens'] for b in batch]),
        'sid_labels': stack_to_tensor([b['target_sid'] for b in batch]),
    }


class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.mask_token = -1
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self):
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError('Maximum token sequence length not implemented.')

    def log(self, message, level='info'):
        from utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)

class VanillaTokenizer(AbstractTokenizer):
    """
    Vanilla Tokenizer that uses precomputed id2sid (item_id -> SID list) directly..
    - SID list length = n_digit (auto-infer from id2sid, expected 8)
    - Each SID entry is in [0, codebook_size-1]
    Special tokens:
        PAD=0, BOS=1, EOS=2, SID tokens start at SID_OFFSET=3 with per-digit offset.
    """

    def __init__(self, config: dict, dataset: AbstractDataset, n_digits: int, codebook_size: int, item2sid_map: Dict[int, List[int]] = None):
        # defaults
        config.setdefault('accelerator', 'cpu')
        config.setdefault('num_proc', 1)
        super(VanillaTokenizer, self).__init__(config, dataset)

        self.dataset = dataset
        self.pad_order = config.get('pad_order', 'right')
        self.pad_token = 0
        #self.bos_token = 1
        #self.eos_token = 1
        self.sid_offset = 1
        self.n_digit = n_digits
        self.codebook_size = codebook_size
        self.sid_order = self._parse_sid_order(config.get('sid_order'))

        if item2sid_map is None:
            item2sid_map_path = config.get('item2codes_path', 'item2codes_raw.pkl')
            with open(item2sid_map_path, 'rb') as f:
                item2sid_map = pickle.load(f)
            self.log(f'[VANILLA] Loaded id2sid_map from {item2sid_map_path}, size={len(item2sid_map)}')
        else:
            self.log(f'[VANILLA] Using id2sid_map from memory, size={len(item2sid_map)}')

        self.log(f'[VANILLA] n_digit={self.n_digit}, codebook_size={self.codebook_size}, sid_order={self.sid_order}')

        self.id2sids: Dict[Any, Tuple[int, ...]] = {}
        id2item = dataset.id_mapping['id2item']
        num_items = dataset.num_items  

        # 仅为 1..num_items-1 的有效 item_id 构建
        not_found = 0
        range_err = 0
        for iid in range(1, num_items+1):
            item_key = id2item[iid]
            if item_key not in item2sid_map:
                not_found += 1
                continue
            sid_list = item2sid_map[item_key]
            # 合法性检查：长度一致、取值范围在 [0, codebook_size-1]
            if len(sid_list) != self.n_digit:
                raise ValueError(f'item_id={iid} has SID length {len(sid_list)} != n_digit {self.n_digit}')
            if any((s < 0 or s >= self.codebook_size) for s in sid_list):
                range_err += 1
                continue

            # tokens = [
            #     int(sid_list[d]) + self.sid_offset + d * self.codebook_size
            #           for d in range(self.n_digit) ]
            #self.id2tokens[iid] = tuple(tokens)
            self.id2sids[iid] = self._reorder_sids(tuple(sid_list))
            
        self.build_id2tokens_tensor(num_items)
            
        if not_found > 0:
            self.log(f'[VANILLA] Warning: {not_found} / {num_items} items missing in id2sid_map')
        if range_err > 0:
            self.log(f'[VANILLA] Warning: {range_err} items have SID out of inferred codebook range')

        self.sids2ids: Dict[Tuple[int, ...], List[int]] = {}

        for iid, sids in self.id2sids.items():
            if sids not in self.sids2ids:
                    self.sids2ids[sids] = []
            self.sids2ids[sids].append(iid)

        # 可选：统计冲突情况
        num_collisions = sum(1 for v in self.sids2ids.values() if len(v) > 1)
        self.log(f'[VANILLA] Built mappings: item2tokens={len(self.id2sids)}, '
                    f'sids2ids={len(self.sids2ids)}, collisions={num_collisions}')

        # collate，由你项目中的 collate 保持一致（可选）
        try:    
            self.collate_fn = {'train': collate_fn, 'val': collate_fn_with_labels, 'test': collate_fn_with_labels}
        except Exception:  
            self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _parse_sid_order(self, sid_order: Optional[List[int]]) -> Tuple[int, ...]:
        """
        支持在配置中指定 SID 的维度顺序。
        - 支持 0/1 开始的索引，例如 [2,1,3] 会交换前两位（按 1-based 解析）。
        """
        if sid_order is None:
            return tuple(range(self.n_digit))

        order_list = list(sid_order)
        if len(order_list) != self.n_digit:
            raise ValueError(f'sid_order length {len(order_list)} must equal n_digit {self.n_digit}')

        # 先尝试 0-based，再尝试 1-based
        zero_based_candidate = list(order_list)
        one_based_candidate = [x - 1 for x in order_list]

        def is_valid(order):
            return set(order) == set(range(self.n_digit))

        if is_valid(zero_based_candidate):
            return tuple(zero_based_candidate)
        if is_valid(one_based_candidate):
            return tuple(one_based_candidate)

        raise ValueError(f'sid_order must be a permutation of 0..{self.n_digit-1} or 1..{self.n_digit}, got {sid_order}')

    def _reorder_sids(self, sid_list: Tuple[int, ...]) -> Tuple[int, ...]:
        """按照 sid_order 重新排列 SID 维度。"""
        return tuple(sid_list[idx] for idx in self.sid_order)

    @property
    def vocab_size(self) -> int:
        return 1 + self.n_digit * self.codebook_size  # 0,1,2 + per-digit buckets

    @property
    def max_token_seq_len(self) -> int:
        max_len = self.config.get('max_item_seq_len')
        return max_len * self.n_digit  # [BOS] + SID digits + [EOS]
    
    def sids_to_id(self, sids):
        """
        将codebook ID序列转换为item_id，检查合法性
        
        Args:
            cb_ids: List[int] 长度 n_digit, 原始 codebook ID (0-255)
            
        Returns:
            item_id(int) 或 None（如果非法）
        """
        if len(sids) != self.n_digit:
            return None
        
        return self.sids2ids.get(tuple(sids), [None])

    def build_id2tokens_tensor(self, num_items):
        """
        构建查表张量 table: (num_items+1, n_digit)
        第 0 行为 PAD（全 -1），其余行为对应 iid 的 sids（无则也填 -1）。
        """
        sids2tokens_offset = torch.arange(self.n_digit, dtype=torch.long) * self.codebook_size + int(self.sid_offset)
        table_id2sids = torch.full((num_items + 1, self.n_digit), int(self.pad_token), dtype=torch.long)
        table_id2tokens = torch.full((num_items + 1, self.n_digit), int(self.pad_token), dtype=torch.long)
        for iid, sids in self.id2sids.items():
            if 0 < iid <= num_items:
                sids = torch.as_tensor(sids, dtype=torch.long)
                table_id2sids[iid] = sids
                table_id2tokens[iid] = sids + sids2tokens_offset  
        self._id2sids_table = table_id2sids
        self._id2tokens_table = table_id2tokens
        
    def add_bos_eos(self,
                sids_flat: torch.Tensor,
                len_seq: torch.Tensor,
                padding_order: str = "right",
                return_mask: bool = True):
        """
        Args:
            sids_flat: (B, T) = (B, L * n_digit)
                已展平的 code tokens（前 len_seq[b]*n_digit 为有效 token，其余可视为 PAD/占位）
            len_seq:   (B,) 每个样本的有效 item 数（不是 token 数）
            padding_side: "right" 或 "left"
                - "right": [BOS] tok tok tok EOS PAD PAD ...
                - "left" : PAD PAD ... [BOS] tok tok tok EOS
            return_mask: 是否返回 mask

        Returns:
            tokens_out: (B, T+2)  # +BOS +EOS
            mask_out  : (B, T+2)  # True 表示非 PAD（含 BOS/EOS）
        """

        B, T = sids_flat.shape              # T = L * n_digit
        M    = int(self.n_digit)            # 每个 item 的 code 数
        device = sids_flat.device
        dtype  = sids_flat.dtype

        # 有效 token 总数（每行）：eff_T[b] = len_seq[b] * M
        eff_T = (len_seq * M)      # (B,)
        eff_T_unsq = eff_T.unsqueeze(1)           # (B,1)
        rows = torch.arange(B, device=device)     # (B,)

        # 1) 预分配输出：全 PAD
        tokens_out = torch.full((B, T + 2), int(self.pad_token), dtype=dtype, device=device)

        inner = tokens_out[:, 1:1+T]             # (B, T)

        col_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        if padding_order == "right":
            # [BOS] tok tok tok EOS PAD PAD ...
            bos_pos = torch.zeros(B, dtype=torch.long, device=device)   # (B,)
            eos_pos = 1 + eff_T                                         # (B,)
            body_mask_inner = (col_T < eff_T_unsq)                      # (B, T)

        elif padding_order == "left":
            # PAD PAD ... [BOS] tok tok tok EOS
            bos_pos = T - eff_T                                         # (B,)
            eos_pos = torch.full((B,), T + 1, dtype=torch.long, device=device)  # 固定最后一列
            body_mask_inner = (col_T >= (T - eff_T_unsq))               # (B, T)
        else:
            raise ValueError(f"padding_side must be 'right' or 'left', got: {padding_order}")

        tokens_out[rows, bos_pos] = int(self.bos_token)
        tokens_out[rows, eos_pos] = int(self.eos_token)

        src_col = torch.arange(T, device=device).unsqueeze(0)           # (1, T)
        src_mask = (src_col < eff_T_unsq)                               # (B, T)
        sids_active = sids_flat[src_mask]                               # (sum(eff_T),)
        inner[body_mask_inner] = sids_active

        if not return_mask:
            return tokens_out

        mask_out = (tokens_out != int(self.pad_token))                  # (B, T+2)
        return tokens_out, mask_out

    def add_eos(self, sids_flat: torch.Tensor, len_seq: torch.Tensor, padding_order: str = "right", return_mask: bool = True):
        """
        只添加 EOS（不再添加 BOS）

        Args:
            sids_flat: (B, T) = (B, L * n_digit)
                已展平的 code tokens（前 len_seq[b] * n_digit 为有效 token，其余可视为占位）
            len_seq:   (B,) 每个样本的有效 item 数（不是 token 数）
            padding_order: "right" 或 "left"
                - "right": tok tok tok EOS PAD PAD ...
                - "left" : PAD PAD ... tok tok tok EOS
            return_mask: 是否返回 mask

        Returns:
            tokens_out: (B, T+1)  # 只多出一个 EOS
            mask_out  : (B, T+1)  # True 表示非 PAD（含 EOS）
        """

        B, T = sids_flat.shape              # T = L * n_digit
        M    = int(self.n_digit)            # 每个 item 的 code 数
        device = sids_flat.device
        dtype  = sids_flat.dtype

        # 有效 token 总数（每行）：eff_T[b] = len_seq[b] * M
        eff_T = (len_seq * M)                      # (B,)
        eff_T_unsq = eff_T.unsqueeze(1)            # (B, 1)
        rows = torch.arange(B, device=device)      # (B,)

        # 1) 预分配输出：全 PAD，长度为 T+1（只多一个 EOS）
        tokens_out = torch.full(
            (B, T + 1),
            int(self.pad_token),
            dtype=dtype,
            device=device,
        )

        # inner 是“主体 token 区域”，最后一列保留给 EOS（对 right padding 来说）
        inner = tokens_out[:, :T]                  # (B, T)
        col_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        if padding_order == "right":
            # tok tok tok EOS PAD PAD ...
            # 有效 body 从位置 0 开始连续放 eff_T 个
            body_mask_inner = (col_T < eff_T_unsq)         # (B, T)
            # EOS 紧跟在最后一个有效 token 之后
            eos_pos = eff_T                                # (B,)

        elif padding_order == "left":
            # PAD PAD ... tok tok tok EOS
            # 有效 body 靠右对齐：放在 [T - eff_T, T)
            body_mask_inner = (col_T >= (T - eff_T_unsq))  # (B, T)
            # EOS 固定在 tokens_out 的最后一列（索引 T）
            eos_pos = torch.full(
                (B,),
                T,
                dtype=torch.long,
                device=device,
            )
        else:
            raise ValueError(
                f"padding_order must be 'right' or 'left', got: {padding_order}"
            )

        # 写入 EOS
        tokens_out[rows, eos_pos] = int(self.eos_token)

        # 把 sids_flat 前 eff_T 个有效 token 填入 body
        src_col = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        src_mask = (src_col < eff_T_unsq)                       # (B, T)
        sids_active = sids_flat[src_mask]                       # (sum(eff_T),)
        inner[body_mask_inner] = sids_active                    # 填充 body 区

        if not return_mask:
            return tokens_out

        mask_out = (tokens_out != int(self.pad_token))          # (B, T+1)
        return tokens_out, mask_out

    def encode_history(self, id_seq, len_seq, mask_flag=True, sids_flag=False):
        tokens_table = getattr(self, "_id2tokens_table")
        sids_table = getattr(self, "_id2sids_table")
        idx = torch.as_tensor(id_seq, dtype=torch.long)
        #len_seq = torch.as_tensor(len_seq, dtype=torch.long)
        # return self.add_eos(tokens, len_seq,
        #                     padding_order=self.pad_order,
        #                     return_mask=mask_flag)
        
        if sids_flag:
            sids = sids_table[idx]      # (B, L, M)
            if not mask_flag:
                return sids
            id_mask = (idx != int(self.pad_token))          # (B, L)
            return sids, id_mask
        else:
            tokens = tokens_table[idx].flatten(1)  # (B, L * M)
            if not mask_flag:
                return tokens
            tokens_mask = (tokens != int(self.pad_token))          # (B, L * M)
            return tokens, tokens_mask

    def encode_target_as_tokens(self, target_id, ar_flag=False):
        table = getattr(self, "_id2tokens_table")
        idx = torch.as_tensor(target_id, dtype=torch.long)
        tokens = table[idx]                                 # (B, M)

        if not ar_flag:
            return tokens
        # -------------------------------
        #   AR 模式：前加 BOS，后加 EOS
        # -------------------------------
        B, M = tokens.shape
        dtype = tokens.dtype
        device = tokens.device
        #bos = torch.full((B, 1), int(self.bos_token), dtype=dtype, device=device)
        eos = torch.full((B, 1), int(self.eos_token), dtype=dtype, device=device)
        #return torch.cat([bos, tokens, eos], dim=1)
        return torch.cat([tokens, eos], dim=1)
    
    def encode_target_as_sids(self, target_id):
        table = getattr(self, "_id2sids_table")
        idx = torch.as_tensor(target_id, dtype=torch.long)
        return table[idx]

    def tokenize_function(self, example: dict, split: str) -> dict:

        id_seq = example['seq']
        #self.logger.info(id_seq)
        len_seq = example['len_seq']
        #self.logger.info(len_seq)
        target_id = example['next']
        #self.logger.info(target_id)
        
        if self.config.get('is_diffusion_model', False):
            seq, mask = self.encode_history(id_seq, len_seq, mask_flag=True, sids_flag=True)
        else:
            seq, mask = self.encode_history(id_seq, len_seq, mask_flag=True)
        if split == 'train':
            target_tokens = self.encode_target_as_tokens(target_id, ar_flag = False)
            target_sids = self.encode_target_as_sids(target_id)
            return {
                'history_tokens': seq,
                'history_mask': mask,
                'target_tokens': target_tokens,
                'target_sid': target_sids,
            }
        else:
            target_tokens = self.encode_target_as_tokens(target_id, ar_flag = False)
            target_sids = self.encode_target_as_sids(target_id)
            return {
                'history_tokens': seq,
                'history_mask': mask,
                'target_tokens': target_tokens,
                'target_sid': target_sids,
                'target_id': target_id
            }

    def tokenize(self, datasets: dict) -> dict:
        """datasets: {'train': HF-Dataset, 'val': ..., 'test': ...} with column 'seq' 'next' """
        tokenized = {}
        for split in datasets:
            tokenized[split] = datasets[split].map(
                lambda ex: self.tokenize_function(ex, split),
                batched=True,
                batch_size=1024, 
                remove_columns=datasets[split].column_names,
                num_proc=self.config.get('num_proc', 1),
                desc=f'[VANILLA] Tokenizing {split}'
            )
        for split in tokenized:
            try:
                tokenized[split].set_format(type='torch')
            except Exception:
                pass
        return tokenized

    # --------- 可选：保存/加载映射，方便复用 ----------
    # def save_mappings(self, out_dir: str):
    #     os.makedirs(out_dir, exist_ok=True)
    #     # item_id -> sid_tokens（带 offset）
    #     item_id2tokens = np.zeros((self.dataset.num_items, self.n_digit), dtype=np.int64)
    #     for item, toks in self.item2tokens.items():
    #         iid = self.dataset.item2id.get(item, 0)
    #         if iid > 0:
    #             item_id2tokens[iid] = np.asarray(toks, dtype=np.int64)
    #     np.save(os.path.join(out_dir, 'item_id2tokens.npy'), item_id2tokens)
    #     with open(os.path.join(out_dir, 'tokens2item.pkl'), 'wb') as f:
    #         pickle.dump(self.tokens2item, f)
    #     self.log(f'[VANILLA] Saved mappings to {out_dir}')
    
from typing import Dict, List, Any, Tuple
import torch
from logging import getLogger

class IdentTokenizer(AbstractTokenizer):
    """
    IdentTokenizer: 不使用 SID 映射，直接用 item_id 作为 token。
    - n_digit 固定为 1
    - codebook_size = dataset.num_items
    - diffusion 用的 code/sid 取值为 (item_id - 1) ∈ [0, codebook_size-1]
    - token 取值为 item_id（PAD=0），通过 sid_offset=1 保证 token = sid + 1

    Special tokens:
        PAD=0
        token(item) = item_id ∈ [1..num_items]
        sid(code)   = item_id-1 ∈ [0..num_items-1]
    """

    def __init__(self, config: dict, dataset: "AbstractDataset"):
        config.setdefault('accelerator', 'cpu')
        config.setdefault('num_proc', 1)
        super(IdentTokenizer, self).__init__(config, dataset)

        self.logger = getLogger()
        self.dataset = dataset

        self.pad_order = config.get('pad_order', 'right')
        self.pad_token = 0

        # 与 Vanilla 的设计保持一致：token = sid + sid_offset
        self.sid_offset = 1

        # 关键：不用 SID，固定 1 digit
        self.n_digit = 1
        self.codebook_size = int(dataset.num_items)

        # 映射表：iid -> (sid,)
        # sid = iid - 1，保证 sid 范围在 [0, codebook_size-1]
        self.id2sids: Dict[Any, Tuple[int, ...]] = {}
        for iid in range(1, self.codebook_size + 1):
            self.id2sids[iid] = (iid - 1,)

        self.build_id2tokens_tensor(self.codebook_size)

        # 反查表：(sid,) -> [iid]
        self.sids2ids: Dict[Tuple[int, ...], List[int]] = {}
        for iid, sids in self.id2sids.items():
            self.sids2ids.setdefault(sids, []).append(iid)

        num_collisions = sum(1 for v in self.sids2ids.values() if len(v) > 1)
        self.log(
            f'[IDENT] n_digit={self.n_digit}, codebook_size={self.codebook_size}, '
            f'item2sids={len(self.id2sids)}, sids2ids={len(self.sids2ids)}, collisions={num_collisions}'
        )

        # collate：保持 Vanilla 的逻辑
        try:
            self.collate_fn = {
                'train': collate_fn,
                'val': collate_fn_with_labels,
                'test': collate_fn_with_labels
            }
        except Exception:
            self.collate_fn = {'train': None, 'val': None, 'test': None}

    # --------------------
    # Properties
    # --------------------
    @property
    def vocab_size(self) -> int:
        # PAD(0) + item tokens(1..num_items) => num_items+1
        return 1 + self.n_digit * self.codebook_size

    @property
    def max_token_seq_len(self) -> int:
        max_len = self.config.get('max_item_seq_len')
        return int(max_len) * self.n_digit

    # --------------------
    # Core mapping helpers
    # --------------------
    def sids_to_id(self, sids):
        """输入 (sid,) 返回对应 item_id 列表（或 [None]）"""
        if len(sids) != self.n_digit:
            return [None]
        return self.sids2ids.get(tuple(sids), [None])

    def build_id2tokens_tensor(self, num_items: int):
        """
        table_id2sids  : (num_items+1, 1)  sid = iid-1
        table_id2tokens: (num_items+1, 1)  token = sid + sid_offset = iid
        第 0 行为 PAD（全 0）
        """
        # n_digit=1 时偏移就是 sid_offset
        sids2tokens_offset = torch.tensor([int(self.sid_offset)], dtype=torch.long)

        table_id2sids = torch.full((num_items + 1, self.n_digit), int(self.pad_token), dtype=torch.long)
        table_id2tokens = torch.full((num_items + 1, self.n_digit), int(self.pad_token), dtype=torch.long)

        for iid, sids in self.id2sids.items():
            if 0 < iid <= num_items:
                sids_t = torch.as_tensor(sids, dtype=torch.long)               # (1,)
                table_id2sids[iid] = sids_t
                table_id2tokens[iid] = sids_t + sids2tokens_offset            # token = iid

        self._id2sids_table = table_id2sids
        self._id2tokens_table = table_id2tokens

    # --------------------
    # Encode APIs (keep same behavior as Vanilla)
    # --------------------
    def encode_history(self, id_seq, len_seq, mask_flag=True, sids_flag=False):
        tokens_table = getattr(self, "_id2tokens_table")
        sids_table = getattr(self, "_id2sids_table")

        idx = torch.as_tensor(id_seq, dtype=torch.long)  # (B, L)

        if sids_flag:
            # diffusion 用：返回 sid(code) = iid-1
            sids = sids_table[idx]  # (B, L, 1)
            if not mask_flag:
                return sids
            id_mask = (idx != int(self.pad_token))  # (B, L)
            return sids, id_mask
        else:
            # 非 diffusion 用：返回 token = iid
            tokens = tokens_table[idx].flatten(1)  # (B, L*1) => (B, L)
            if not mask_flag:
                return tokens
            tokens_mask = (tokens != int(self.pad_token))  # (B, L)
            return tokens, tokens_mask

    def encode_target_as_tokens(self, target_id, ar_flag=False):
        table = getattr(self, "_id2tokens_table")
        idx = torch.as_tensor(target_id, dtype=torch.long)
        tokens = table[idx]  # (B, 1) token = iid

        if not ar_flag:
            return tokens

        # AR 模式保持 Vanilla 的写法（如果你后续需要）
        B, M = tokens.shape
        dtype = tokens.dtype
        device = tokens.device
        eos = torch.full((B, 1), int(self.eos_token), dtype=dtype, device=device)
        return torch.cat([tokens, eos], dim=1)

    def encode_target_as_sids(self, target_id):
        table = getattr(self, "_id2sids_table")
        idx = torch.as_tensor(target_id, dtype=torch.long)
        return table[idx]  # (B, 1) sid = iid-1

    def tokenize_function(self, example: dict, split: str) -> dict:
        id_seq = example['seq']
        len_seq = example['len_seq']
        target_id = example['next']

        if self.config.get('is_diffusion_model', False):
            seq, mask = self.encode_history(id_seq, len_seq, mask_flag=True, sids_flag=True)
        else:
            seq, mask = self.encode_history(id_seq, len_seq, mask_flag=True, sids_flag=False)

        target_tokens = self.encode_target_as_tokens(target_id, ar_flag=False)
        target_sids = self.encode_target_as_sids(target_id)

        if split == 'train':
            return {
                'history_tokens': seq,
                'history_mask': mask,
                'target_tokens': target_tokens,
                'target_sid': target_sids,
            }
        else:
            return {
                'history_tokens': seq,
                'history_mask': mask,
                'target_tokens': target_tokens,
                'target_sid': target_sids,
                'target_id': target_id
            }

    def tokenize(self, datasets: dict) -> dict:
        tokenized = {}
        for split in datasets:
            tokenized[split] = datasets[split].map(
                lambda ex: self.tokenize_function(ex, split),
                batched=True,
                batch_size=1024,
                remove_columns=datasets[split].column_names,
                num_proc=self.config.get('num_proc', 1),
                desc=f'[IDENT] Tokenizing {split}'
            )
        for split in tokenized:
            try:
                tokenized[split].set_format(type='torch')
            except Exception:
                pass
        return tokenized

