import os
import torch
import pickle
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from logging import getLogger

from dataset.dataset import AbstractDataset

def stack_to_tensor(seq, dtype=None):
    """Stack a list into a tensor and convert its type if dtype is supplied."""
    out = torch.stack(seq) if torch.is_tensor(seq[0]) else torch.as_tensor(seq)
    return out.to(dtype) if dtype else out


def collate_fn_with_labels(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Training collate function.
    
    Args:
        batch: list of dictionaries containing these fields:
            - history_sid: historical SID sequence [seq_len, n_digit]
            - history_mask: history mask [seq_len]
            - decoder_input_ids: decoder inputs [n_digit]
            - decoder_labels: decoder labels [n_digit]
    
    Returns:
        Collated dictionary
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
    Validation collate function.
    
    Args:
        batch: list of dictionaries containing these fields:
            - history_sid: historical SID sequence [seq_len, n_digit]
            - history_mask: history mask [seq_len]
            - labels: ground-truth label sequence [n_digit]
    
    Returns:
        Collated dictionary
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

        # Build mappings only for valid item IDs in 1..num_items-1
        not_found = 0
        range_err = 0
        for iid in range(1, num_items+1):
            item_key = id2item[iid]
            if item_key not in item2sid_map:
                not_found += 1
                continue
            sid_list = item2sid_map[item_key]
            # Validate equal lengths and values in [0, codebook_size-1]
            if len(sid_list) != self.n_digit:
                raise ValueError(f'item_id={iid} has SID length {len(sid_list)} != n_digit {self.n_digit}')
            if any((s < 0 or s >= self.codebook_size) for s in sid_list):
                range_err += 1
                continue

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

        # Optionally count collisions
        num_collisions = sum(1 for v in self.sids2ids.values() if len(v) > 1)
        self.log(f'[VANILLA] Built mappings: item2tokens={len(self.id2sids)}, '
                    f'sids2ids={len(self.sids2ids)}, collisions={num_collisions}')

        # Use the project's existing collate functions for consistency, optionally
        try:    
            self.collate_fn = {'train': collate_fn, 'val': collate_fn_with_labels, 'test': collate_fn_with_labels}
        except Exception:  
            self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _parse_sid_order(self, sid_order: Optional[List[int]]) -> Tuple[int, ...]:
        """
        Support configuring the order of SID dimensions.
        - Support zero- and one-based indices; [2,1,3] swaps the first two dimensions using one-based indices.
        """
        if sid_order is None:
            return tuple(range(self.n_digit))

        order_list = list(sid_order)
        if len(order_list) != self.n_digit:
            raise ValueError(f'sid_order length {len(order_list)} must equal n_digit {self.n_digit}')

        # Try zero-based indices first, then one-based indices
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
        """Reorder SID dimensions according to sid_order."""
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
        Convert a codebook ID sequence to item_id with validity checks.
        
        Args:
            cb_ids: List[int] of length n_digit containing raw codebook IDs (0-255)
            
        Returns:
            item_id (int), or None for invalid values
        """
        if len(sids) != self.n_digit:
            return None
        
        return self.sids2ids.get(tuple(sids), [None])

    def build_id2tokens_tensor(self, num_items):
        """
        Build a lookup tensor table: (num_items+1, n_digit).
        Row 0 is PAD (all -1); other rows contain the corresponding iid SIDs, or -1 when missing.
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
                Flattened code tokens; the first len_seq[b]*n_digit tokens are valid, and the rest are padding/placeholders.
            len_seq: (B,) number of valid items per sample, not the number of tokens
            padding_side: "right" or "left"
                - "right": [BOS] tok tok tok EOS PAD PAD ...
                - "left" : PAD PAD ... [BOS] tok tok tok EOS
            return_mask: whether to return a mask

        Returns:
            tokens_out: (B, T+2)  # +BOS +EOS
            mask_out: (B, T+2)  # True indicates non-PAD positions, including BOS/EOS
        """

        B, T = sids_flat.shape              # T = L * n_digit
        M    = int(self.n_digit)            # Number of codes per item
        device = sids_flat.device
        dtype  = sids_flat.dtype

        # Total valid tokens per row: eff_T[b] = len_seq[b] * M
        eff_T = (len_seq * M)      # (B,)
        eff_T_unsq = eff_T.unsqueeze(1)           # (B,1)
        rows = torch.arange(B, device=device)     # (B,)

        # 1) Preallocate output filled with PAD
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
            eos_pos = torch.full((B,), T + 1, dtype=torch.long, device=device)  # Always use the final column
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
        Append only EOS, without BOS.

        Args:
            sids_flat: (B, T) = (B, L * n_digit)
                Flattened code tokens; the first len_seq[b]*n_digit tokens are valid, and the rest are placeholders.
            len_seq: (B,) number of valid items per sample, not the number of tokens
            padding_order: "right" or "left"
                - "right": tok tok tok EOS PAD PAD ...
                - "left" : PAD PAD ... tok tok tok EOS
            return_mask: whether to return a mask

        Returns:
            tokens_out: (B, T+1)  # One additional EOS
            mask_out: (B, T+1)  # True indicates non-PAD positions, including EOS
        """

        B, T = sids_flat.shape              # T = L * n_digit
        M    = int(self.n_digit)            # Number of codes per item
        device = sids_flat.device
        dtype  = sids_flat.dtype

        # Total valid tokens per row: eff_T[b] = len_seq[b] * M
        eff_T = (len_seq * M)                      # (B,)
        eff_T_unsq = eff_T.unsqueeze(1)            # (B, 1)
        rows = torch.arange(B, device=device)      # (B,)

        # 1) Preallocate PAD-filled output of length T+1 for one extra EOS
        tokens_out = torch.full(
            (B, T + 1),
            int(self.pad_token),
            dtype=dtype,
            device=device,
        )

        # inner is the body-token region; reserve the final column for EOS with right padding
        inner = tokens_out[:, :T]                  # (B, T)
        col_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        if padding_order == "right":
            # tok tok tok EOS PAD PAD ...
            # Place eff_T valid body tokens consecutively starting at position 0
            body_mask_inner = (col_T < eff_T_unsq)         # (B, T)
            # Place EOS immediately after the last valid token
            eos_pos = eff_T                                # (B,)

        elif padding_order == "left":
            # PAD PAD ... tok tok tok EOS
            # Right-align valid body tokens in [T - eff_T, T)
            body_mask_inner = (col_T >= (T - eff_T_unsq))  # (B, T)
            # Place EOS in the final tokens_out column at index T
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

        # Write EOS
        tokens_out[rows, eos_pos] = int(self.eos_token)

        # Copy the first eff_T valid tokens from sids_flat into the body
        src_col = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        src_mask = (src_col < eff_T_unsq)                       # (B, T)
        sids_active = sids_flat[src_mask]                       # (sum(eff_T),)
        inner[body_mask_inner] = sids_active                    # Fill the body region

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
        #   AR mode: prepend BOS and append EOS
        B, M = tokens.shape
        dtype = tokens.dtype
        device = tokens.device
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

    # --------- Optional: save/load mappings for reuse ----------
    # def save_mappings(self, out_dir: str):
    #     os.makedirs(out_dir, exist_ok=True)
    # # item_id -> sid_tokens with offsets
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
    IdentTokenizer: use item_id directly as the token without SID mappings.
    - n_digit is fixed to 1
    - codebook_size = dataset.num_items
    - Diffusion code/sid values are (item_id - 1) in [0, codebook_size-1]
    - token = item_id with PAD=0; sid_offset=1 ensures token = sid + 1

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

        # Match Vanilla's design: token = sid + sid_offset
        self.sid_offset = 1

        # Do not use SIDs; fix the number of digits to one
        self.n_digit = 1
        self.codebook_size = int(dataset.num_items)

        # Mapping table: iid -> (sid,)
        # sid = iid - 1, ensuring sid is in [0, codebook_size-1]
        self.id2sids: Dict[Any, Tuple[int, ...]] = {}
        for iid in range(1, self.codebook_size + 1):
            self.id2sids[iid] = (iid - 1,)

        self.build_id2tokens_tensor(self.codebook_size)

        # Reverse mapping: (sid,) -> [iid]
        self.sids2ids: Dict[Tuple[int, ...], List[int]] = {}
        for iid, sids in self.id2sids.items():
            self.sids2ids.setdefault(sids, []).append(iid)

        num_collisions = sum(1 for v in self.sids2ids.values() if len(v) > 1)
        self.log(
            f'[IDENT] n_digit={self.n_digit}, codebook_size={self.codebook_size}, '
            f'item2sids={len(self.id2sids)}, sids2ids={len(self.sids2ids)}, collisions={num_collisions}'
        )

        # collate: retain Vanilla's logic
        try:
            self.collate_fn = {
                'train': collate_fn,
                'val': collate_fn_with_labels,
                'test': collate_fn_with_labels
            }
        except Exception:
            self.collate_fn = {'train': None, 'val': None, 'test': None}

    # Properties
    @property
    def vocab_size(self) -> int:
        # PAD(0) + item tokens(1..num_items) => num_items+1
        return 1 + self.n_digit * self.codebook_size

    @property
    def max_token_seq_len(self) -> int:
        max_len = self.config.get('max_item_seq_len')
        return int(max_len) * self.n_digit

    # Core mapping helpers
    def sids_to_id(self, sids):
        """Return item IDs corresponding to (sid,), or [None]."""
        if len(sids) != self.n_digit:
            return [None]
        return self.sids2ids.get(tuple(sids), [None])

    def build_id2tokens_tensor(self, num_items: int):
        """
        table_id2sids  : (num_items+1, 1)  sid = iid-1
        table_id2tokens: (num_items+1, 1)  token = sid + sid_offset = iid
        Row 0 is PAD (all zeros).
        """
        # For n_digit=1, the offset is sid_offset
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

    # Encode APIs (keep same behavior as Vanilla)
    def encode_history(self, id_seq, len_seq, mask_flag=True, sids_flag=False):
        tokens_table = getattr(self, "_id2tokens_table")
        sids_table = getattr(self, "_id2sids_table")

        idx = torch.as_tensor(id_seq, dtype=torch.long)  # (B, L)

        if sids_flag:
            # For diffusion: return sid(code) = iid-1
            sids = sids_table[idx]  # (B, L, 1)
            if not mask_flag:
                return sids
            id_mask = (idx != int(self.pad_token))  # (B, L)
            return sids, id_mask
        else:
            # For non-diffusion: return token = iid
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

        # Keep Vanilla's AR mode implementation for optional future use
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

