import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.model import AbstractModel
from .diffusion import MaskDiffusion
from .ablate_decode import decode_ablate_confidence
from .beam import fast_beam_search_for_eval

from tokenizer.tokenizer import AbstractTokenizer

def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or "layernorm").lower() == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)

# ===============================
# 1. 标准 Transformer 封装
# ===============================
class TransformerEncoderWrapper(nn.Module):
    """
    T5 风格命名：d_model / num_heads / d_ff / dropout_rate / activation_function
    """

    def __init__(self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    num_layers: int,
                    dropout_rate: float = 0.1,
                    activation_function: str = "relu"):
        super().__init__()
        activation = activation_function  # "relu" / "gelu"

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, S, d_model)
        # src_key_padding_mask: (B, S) bool, True = padding
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoderWrapper(nn.Module):
    """
    T5 风格命名：d_model / num_heads / d_ff / num_decoder_layers / dropout_rate / activation_function
    """

    def __init__(self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    num_decoder_layers: int,
                    dropout_rate: float = 0.1,
                    activation_function: str = "relu"):
        super().__init__()
        activation = activation_function

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)

    def forward(self, tgt, memory, memory_key_padding_mask=None, tgt_mask=None):
        # tgt: (B, T_dec, d_model)
        # memory: (B, S, d_model)
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

class ModelOutput:
    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None

# ===============================
# 3. 重构后的 DiffGRM
# ===============================

class DiffGRM(AbstractModel):

    def __init__(self, config: dict, tokenizer: AbstractTokenizer):
        super().__init__(config, tokenizer)

        self.n_digit = config['n_digits']
        self.codebook_size = config['codebook_size']
        self.vocab_size = config['vocab_size']
        self.sid_offset = config['sid_offset']

        # Model dimensions
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.d_ff = config['d_ff']
        self.dropout_rate = config['dropout_rate']
        self.activation_function = config['activation_function']
        self.feed_projector = config.get('feed_forward_proj', 'relu')

        # Encoder / Decoder 层数
        self.num_layers = config['num_layers']
        self.num_decoder_layers = config['num_decoder_layers']

        # Normalization
        self.norm_type = (config.get('norm_type', 'layernorm') or 'layernorm').lower()
        self.norm_eps = float(config.get('norm_eps', 1e-6 if self.norm_type == 'rmsnorm' else 1e-5))

        # ==== 原来的 mask 策略配置（保留逻辑）====
        self.masking_strategy = config.get('masking_strategy', 'random')  # random | sequential | guided

        if self.masking_strategy == 'sequential':
            seq_cfg = config.get('sequential_steps', 'auto')
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            assert 1 <= self.seq_steps <= self.n_digit

            self.sequential_paths = config.get('sequential_paths', 1)
            assert self.sequential_paths >= 1

            self.augment_factor = self.seq_steps * self.sequential_paths
            print(f"[MODEL] ▶ use SEQUENTIAL views: steps={self.seq_steps}, "
                    f"paths={self.sequential_paths}, augment_factor={self.augment_factor}")
            self.mask_probs = None
        elif self.masking_strategy == 'guided':
            guided_cfg = config.get('guided_steps', 'auto')
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = config.get('guided_conf_metric', 'msp')
            assert self.guided_conf_metric in ('msp', 'entropy')
            self.guided_select = config.get('guided_select', 'most')
            assert self.guided_select in ('most', 'least')
            self.augment_factor = self.guided_steps
            print(f"[MODEL] ▶ GUIDED: steps={self.guided_steps}, metric={self.guided_conf_metric}, "
                    f"select={self.guided_select}, augment_factor={self.augment_factor}")
            self.mask_probs = None
        else:
            # random
            self.mask_prob_random = bool(config.get('mask_prob_random', False))
            if self.mask_prob_random:
                low = float(config.get('mask_prob_random_min', 0.0))
                high = float(config.get('mask_prob_random_max', 1.0))
                if not (0.0 <= low <= high <= 1.0):
                    raise ValueError(
                        f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                    )
                sampled_prob = float(np.random.uniform(low, high))
                self.augment_factor = 1
                self.mask_probs = [sampled_prob]
                self.sampled_mask_prob = sampled_prob
                print(
                    f"[MODEL] Using RANDOMLY-SAMPLED masking prob: {sampled_prob:.4f} "
                    f"(range [{low}, {high}]); augment_factor=1"
                )
            elif 'mask_probs' in config and config['mask_probs'] is not None:
                mask_probs_raw = config['mask_probs']
                if isinstance(mask_probs_raw, str):
                    self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
                elif isinstance(mask_probs_raw, (list, tuple)):
                    self.mask_probs = [float(p) for p in mask_probs_raw]
                elif isinstance(mask_probs_raw, (int, float)):
                    self.mask_probs = [float(mask_probs_raw)]
                else:
                    try:
                        mask_probs_str = str(mask_probs_raw)
                        self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                    except (ValueError, AttributeError):
                        raise ValueError(
                            f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)})"
                        )
                self.augment_factor = len(self.mask_probs)
                print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")
            else:
                mask_prob = config.get('mask_prob', 0.5)
                self.augment_factor = config.get('augment_factor', 4)
                self.mask_probs = [float(mask_prob)] * self.augment_factor
                print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")

        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        # item_mlp：将 n_digit 个 SID token 压成 1 个 item token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # 位置编码（encoder）
        self.max_history_len = config.get('max_history_len', 50)
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.d_model)

        # === 新：用标准 Transformer 封装 Encoder / Decoder ===
        self.encoder = TransformerEncoderWrapper(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            activation_function=self.activation_function,
        )

        self.decoder = TransformerDecoderWrapper(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_decoder_layers=self.num_decoder_layers,
            dropout_rate=self.dropout_rate,
            activation_function=self.activation_function,
        )

        # LayerNorm
        self.ln_f = make_norm(self.norm_type, self.d_model, self.norm_eps)

        # 输出层：共享 embedding dot-product
        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:
            self.output_adapter = nn.Identity()
            print(f"[DIFF_GRM] Using shared embedding dot-product output layer")
        else:
            self.output_adapter = nn.Linear(self.d_model, self.d_model, bias=False)
            print(f"[DIFF_GRM] Using independent Linear output adapter")

        # Dropout
        self.drop = nn.Dropout(self.dropout_rate)

        # === 新：MaskDiffusion 实例（只负责视图构造 + loss） ===
        self.masker = MaskDiffusion(self.config, n_digit=self.n_digit)
        self.masker.sync_from_model_config(self)
        # per-digit mask embedding
        self.mask_emb_table = nn.Embedding(self.n_digit, self.d_model)

        # ablation decode 插入 mode（保持原逻辑）
        ab_cfg = self.config.get('ablate_decode', {}) or {}
        if bool(ab_cfg.get('enabled', False)):
            modes = list(self.config.get('beam_search_modes', []) or [])
            to_add = ['confidence_s1', 'confidence_s2', 'confidence_s3']
            if 'confidence' in modes:
                base = modes.index('confidence')
                for i, m in enumerate(to_add, 1):
                    if m not in modes:
                        modes.insert(base + i, m)
            else:
                for m in reversed(to_add):
                    if m not in modes:
                        modes.insert(0, m)
            self.config['beam_search_modes'] = modes
            
        # 初始化参数
        self.apply(self._init_weights)

    # ----------------------------------------------------
    # 一些工具属性 / 方法
    # ----------------------------------------------------

    @property
    def n_parameters(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def _compute_digit_logits(self, hidden_last, digit: int):
        """
        使用共享 embedding dot-product 计算第 digit 位的 logits
        hidden_last: (B, d_model)
        返回: (B, codebook_size)
        """
        if digit is None:
            raise ValueError("digit 不能为 None")

        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围 [0, {self.n_digit-1}]")

        start = self.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size
        E_sub = self.embedding.weight[start:end]    # (K, d)

        h = self.output_adapter(hidden_last)        # (B, d)
        logits = torch.matmul(h, E_sub.t())         # (B, K)
        return logits

    # ----------------------------------------------------
    # mask 策略在 epoch 之间切换（保留原接口，内部同步给 masker）
    # ----------------------------------------------------

    def resample_mask_prob_if_needed(self):
        """
        当采用 random + mask_prob_random=true 时，在训练每个 epoch 开始重新采样一次掩码概率
        """
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            low = float(self.config.get('mask_prob_random_min', 0.0))
            high = float(self.config.get('mask_prob_random_max', 1.0))
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(
                    f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                )
            sampled_prob = float(np.random.uniform(low, high))
            self.mask_probs = [sampled_prob]
            self.sampled_mask_prob = sampled_prob
            self.augment_factor = 1
            print(f"[MODEL] [Epoch-Resample] RANDOM masking prob resampled to {sampled_prob:.4f} "
                    f"(range [{low}, {high}]); augment_factor=1")

        # 同步给 masker
        self.masker.sync_from_model_config(self)

    def set_masking_mode(self, strategy: str, **kw):
        """
        训练过程中的热切换：
        - strategy: 'guided' | 'sequential' | 'random'
        - kw: 对应策略的额外超参（尽量复用你原来的逻辑）
        """
        self.masking_strategy = strategy

        if strategy == 'sequential':
            seq_cfg = kw.get('sequential_steps', self.config.get('sequential_steps', 'auto'))
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            self.sequential_paths = int(kw.get('sequential_paths', self.config.get('sequential_paths', 1)))
            self.augment_factor = self.seq_steps * self.sequential_paths
            self.mask_probs = None
            print(f"[SCHEDULE] → SEQUENTIAL: steps={self.seq_steps}, paths={self.sequential_paths}, "
                    f"augment_factor={self.augment_factor}")

        elif strategy == 'guided':
            guided_cfg = kw.get('guided_steps', self.config.get('guided_steps', 'auto'))
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = kw.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp'))
            self.guided_select = kw.get('guided_select', self.config.get('guided_select', 'least'))
            self.config['guided_refresh_each_step'] = bool(kw.get(
                'guided_refresh_each_step',
                self.config.get('guided_refresh_each_step', False)
            ))
            self.augment_factor = self.guided_steps
            self.mask_probs = None
            print(f"[SCHEDULE] → GUIDED({self.guided_select}): steps={self.guided_steps}, "
                  f"metric={self.guided_conf_metric}, refresh={self.config['guided_refresh_each_step']}, "
                  f"augment_factor={self.augment_factor}")

        elif strategy == 'random':
            self.mask_prob_random = bool(kw.get('mask_prob_random', self.config.get('mask_prob_random', False)))
            if self.mask_prob_random:
                low = float(kw.get('mask_prob_random_min', self.config.get('mask_prob_random_min', 0.0)))
                high = float(kw.get('mask_prob_random_max', self.config.get('mask_prob_random_max', 1.0)))
                self.mask_probs = [float(np.random.uniform(low, high))]
                self.augment_factor = 1
            else:
                if 'mask_probs' in kw and kw['mask_probs'] is not None:
                    raw = kw['mask_probs']
                    if isinstance(raw, (list, tuple)):
                        self.mask_probs = [float(p) for p in raw]
                    else:
                        self.mask_probs = [float(p) for p in str(raw).split(',')]
                    self.augment_factor = len(self.mask_probs)
                else:
                    mp = float(kw.get('mask_prob', self.config.get('mask_prob', 0.5)))
                    af = int(kw.get('augment_factor', self.config.get('augment_factor', 4)))
                    self.mask_probs = [mp] * af
                    self.augment_factor = af
            print(f"[SCHEDULE] → RANDOM: mask_probs={self.mask_probs}, augment_factor={self.augment_factor}")
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

        # 同步给 masker
        self.masker.sync_from_model_config(self)

    # ----------------------------------------------------
    # forward：Encoder + MaskDiffusion + Decoder
    # ----------------------------------------------------

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        device = next(self.parameters()).device

        # 仅为安全起见，每次 forward 前同步一次 mask 配置（防止外部直接改属性）
        self.masker.sync_from_model_config(self)

        # --- 调试一次 ---
        if not hasattr(self, '_debug_printed'):
            print(f"[DIFF_GRM] Using RPG_ED-style encoder: MLP compression + positional enc")
            print(f"[DIFF_GRM] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            print(f"[DIFF_GRM] masking_strategy: {self.masking_strategy}")
            if self.masking_strategy == 'random' and getattr(self, 'mask_probs', None) is not None:
                print(f"[DIFF_GRM] mask_probs: {self.mask_probs}")
            self._debug_printed = True

        # ================= Encoder =================
        history_tokens, item_mask, L = self._prepare_encoder_inputs(batch, device)
        B = history_tokens.size(0)
        
        tok_emb = self.embedding(history_tokens)        # (B, S, n_digit, d)
        
        # n_digit * d → d
        item_emb = tok_emb.reshape(B, L, self.n_digit * self.d_model)
        item_emb = self.item_mlp(item_emb)              # (B, S, d_model)

        # 位置编码
        pos_ids = torch.arange(L, device=device)
        pos_emb = self.pos_emb_enc(pos_ids).unsqueeze(0).expand(B, -1, -1)  # (B, S, d_model)

        encoder_hidden = item_emb + pos_emb
        encoder_hidden = self.drop(encoder_hidden)
        
        # 标准 TransformerEncoder
        # src_key_padding_mask: True = padding
        src_kpm = ~item_mask
        encoder_hidden = self.encoder(encoder_hidden, src_key_padding_mask=src_kpm)
        encoder_hidden = self.ln_f(encoder_hidden)

        encoder_hidden = encoder_hidden * item_mask.unsqueeze(-1).float()

        if not return_loss:
            out = ModelOutput()
            out.hidden_states = encoder_hidden
            return out

        # ================= Decoder-side Masking =================
        # labels: diffusion models should use codebook ids (sid_labels)
        if 'sid_labels' in batch:
            sid_labels = batch['sid_labels'].to(device)
        else:
            sid_labels = batch['labels'].to(device)
            # Fallback: if labels are token ids, convert back to codebook ids
            if int(sid_labels.max().item()) >= int(self.codebook_size):
                offsets = (
                    torch.arange(self.n_digit, device=device, dtype=sid_labels.dtype) * self.codebook_size
                    + self.sid_offset
                ).view(1, self.n_digit)
                sid_labels = sid_labels - offsets
        sid_labels = sid_labels.clamp(0, self.codebook_size - 1)
        decoder_input_sids = sid_labels.clone()          # (B, n_digit)

        # --- guided 策略需要的打分函数 ---
        def guided_score_fn(cur_inp, cur_mask, enc_hidden):
            """
            cur_inp:  (B, n_digit)  当前视图 decoder_input_ids
            cur_mask:(B, n_digit)  True=被掩盖
            enc_hidden: (B, S, d)
            返回 conf: (B, n_digit)
            """
            was_training = self.training
            self.eval()
            with torch.no_grad():
                inner_batch = {
                    'decoder_input_ids': cur_inp,
                    'encoder_hidden': enc_hidden,
                }
                # history_mask: True=valid item, False=padding
                inner_batch['history_mask'] = item_mask

                inner_batch['mask_positions'] = cur_mask.float()

                out = self.forward_decoder_only(
                    inner_batch,
                    return_loss=False,
                    digit=None,
                    past_key_values=None,
                    use_cache=False,
                )
                logits = out.logits  # (B, n_digit, K)
                probs = F.softmax(logits, dim=-1)

                metric = getattr(self, 'guided_conf_metric', 'msp')
                if metric == 'entropy':
                    ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)
                    conf = -ent
                else:
                    conf = probs.max(dim=-1).values
            if was_training:
                self.train()
            return conf

        score_fn = guided_score_fn if self.masking_strategy == 'guided' else None

        # 由 MaskDiffusion 构造多视图
        (
            decoder_input_ids_exp,
            decoder_labels_exp,
            mask_positions_exp,
            encoder_hidden_exp,
            augment_factor,
        ) = self.masker.build_views(
            decoder_input_ids=decoder_input_sids,
            decoder_labels=sid_labels,
            encoder_hidden=encoder_hidden,
            strategy=self.masking_strategy,
            score_fn=score_fn,
        )

        B_expanded = decoder_input_ids_exp.size(0)
        self.augment_factor = augment_factor  # 更新当前视图数（方便外部读取）

        # guided 模式下检查掩码单调递增
        if self.masking_strategy == 'guided':
            m = mask_positions_exp.view(B, augment_factor, self.n_digit).sum(-1)
            assert torch.all(m[:, 1:] >= m[:, :-1]), "guided views should increase masked count monotonically"

        # ================= Decoder =================
        decoder_emb = torch.zeros(B_expanded, self.n_digit, self.d_model, device=device)

        for d in range(self.n_digit):
            codebook_ids = decoder_input_ids_exp[:, d]
            token_ids = codebook_ids + self.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb = self.embedding(token_ids)                       # (B_exp, d_model)

            mask_emb = self.mask_emb_table.weight[d].unsqueeze(0)      # (1, d_model)
            mask_emb = mask_emb.expand(B_expanded, -1)                 # (B_exp, d_model)

            is_masked = mask_positions_exp[:, d].unsqueeze(-1)         # (B_exp, 1)
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)

        decoder_emb = self.drop(decoder_emb)

        # memory_key_padding_mask: True = padding
        item_mask_exp = item_mask.repeat(augment_factor, 1)
        memory_kpm = ~item_mask_exp

        decoder_hidden = self.decoder(
            tgt=decoder_emb,
            memory=encoder_hidden_exp,
            memory_key_padding_mask=memory_kpm,
            tgt_mask=None,    # 非自回归
        )
        decoder_hidden = self.ln_f(decoder_hidden)

        # ================= Loss =================
        label_smoothing = self.config.get('label_smoothing', 0.1)

        def digit_logits_fn(hidden_last, digit: int):
            return self._compute_digit_logits(hidden_last, digit=digit)

        total_loss = self.masker.compute_loss(
            decoder_hidden=decoder_hidden,
            decoder_labels=decoder_labels_exp,
            mask_positions=mask_positions_exp,
            digit_logits_fn=digit_logits_fn,
            label_smoothing=label_smoothing,
            strategy=self.masking_strategy,
            mask_prob_random=getattr(self, 'mask_prob_random', False),
        )

        out = ModelOutput()
        out.loss = total_loss
        out.hidden_states = decoder_hidden
        out.logits = None
        return out

    # ----------------------------------------------------
    # forward_decoder_only：仅 decoder，用于推理 / guided 打分
    # ----------------------------------------------------

    def forward_decoder_only(
        self,
        batch: dict,
        return_loss: bool = False,
        digit: int = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """
        仅运行 decoder 部分（非自回归），接口保持不变：
            - decoder_input_ids: (B, n_digit)
            - encoder_hidden:    (B, S, d)
            - mask_positions:    (B, n_digit) (0/1)，可选
        """
        device = next(self.parameters()).device

        decoder_input_ids = batch['decoder_input_ids'].to(device)      # (B, n_digit)
        encoder_hidden = batch['encoder_hidden'].to(device)            # (B, S, d)
        B, n_digit = decoder_input_ids.shape
        assert n_digit == self.n_digit

        if 'mask_positions' in batch:
            mask_positions = batch['mask_positions'].to(device)        # (B, n_digit)
        else:
            mask_positions = torch.zeros(B, n_digit, device=device)

        # 构造 decoder_emb（与 forward 保持一致）
        decoder_emb = torch.zeros(B, self.n_digit, self.d_model, device=device)
        for d in range(self.n_digit):
            token_ids = decoder_input_ids[:, d] + self.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb = self.embedding(token_ids)

            mask_emb = self.mask_emb_table.weight[d].unsqueeze(0).expand(B, -1)
            is_masked = mask_positions[:, d].unsqueeze(-1)
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)

        decoder_emb = self.drop(decoder_emb)

        memory_kpm = None
        history_mask = batch.get('history_mask', None)
        if history_mask is None:
            history_mask = batch.get('attention_mask', None)
        if history_mask is not None:
            history_mask = history_mask.to(device)
            if history_mask.dtype != torch.bool:
                history_mask = history_mask != 0

            # history_mask is expected to be "valid mask" (True=valid item).
            # Reduce token-level masks to item-level masks if needed.
            S = encoder_hidden.size(1)
            if history_mask.dim() == 2 and history_mask.shape[1] != S:
                if history_mask.shape[1] == S * self.n_digit:
                    history_mask = history_mask.view(B, S, self.n_digit)[:, :, 0]
                else:
                    raise ValueError(
                        f"history_mask shape {tuple(history_mask.shape)} does not match encoder_hidden length S={S}"
                    )
            elif history_mask.dim() == 3 and history_mask.shape[:2] == (B, S):
                history_mask = history_mask[:, :, 0]

            # memory_key_padding_mask: True = padding
            memory_kpm = ~history_mask.bool()

        decoder_hidden = self.decoder(
            tgt=decoder_emb,
            memory=encoder_hidden,
            memory_key_padding_mask=memory_kpm,
            tgt_mask=None,
        )
        decoder_hidden = self.ln_f(decoder_hidden)

        # digit=None → 所有位；否则只算指定 digit 的 logits
        if digit is not None:
            logits = self._compute_digit_logits(decoder_hidden[:, digit, :], digit=digit)
        else:
            logits_list = []
            for d in range(self.n_digit):
                logits_list.append(self._compute_digit_logits(decoder_hidden[:, d, :], digit=d))
            logits = torch.stack(logits_list, dim=1)    # (B, n_digit, K)

        out = ModelOutput()
        out.hidden_states = decoder_hidden
        out.logits = logits
        out.past_key_values = None   # 接口保留，但非自回归不再使用 cache
        return out

    # ----------------------------------------------------
    # generate：保持原有逻辑（调用 fast_beam_search_for_eval）
    # ----------------------------------------------------

    def generate(self, batch, n_return_sequences=1, mode="confidence"):

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                encoder_outputs = self.forward(batch, return_loss=False)
                encoder_hidden = encoder_outputs.hidden_states

                if mode in ("confidence", "random"):
                    generated_sequences = fast_beam_search_for_eval(
                        model=self,
                        encoder_hidden=encoder_hidden,
                        beam_size=n_return_sequences,
                        max_len=self.n_digit,
                        tokenizer=self.tokenizer,
                        mode=mode,
                        rand_cfg=self.config.get("random_beam", {}),
                    )
                    return generated_sequences

                if mode.startswith("confidence_s") and bool(self.config.get('ablate_decode', {}).get('enabled', False)):
                    try:
                        steps = int(mode.split("confidence_s")[-1])
                    except Exception:
                        steps = int(self.config.get('ablate_decode', {}).get('steps_default', 3))
                    if steps >= 4:
                        generated_sequences = fast_beam_search_for_eval(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            beam_size=n_return_sequences,
                            max_len=self.n_digit,
                            tokenizer=self.tokenizer,
                            mode="confidence",
                            rand_cfg=self.config.get("random_beam", {}),
                        )
                    else:
                        generated_sequences = decode_ablate_confidence(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            tokenizer=self.tokenizer,
                            steps=steps,
                            n_return_sequences=n_return_sequences,
                        )
                    return generated_sequences

                generated_sequences = fast_beam_search_for_eval(
                    model=self,
                    encoder_hidden=encoder_hidden,
                    beam_size=n_return_sequences,
                    max_len=self.n_digit,
                    tokenizer=self.tokenizer,
                    mode="confidence",
                    rand_cfg=self.config.get("random_beam", {}),
                )
                return generated_sequences
        finally:
            if was_training:
                self.train()

    # ----------------------------------------------------
    # 初始化权重
    # ----------------------------------------------------

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
        # output_adapter=Identity() 不需要初始化

    def _prepare_encoder_inputs(self, batch: dict, device):
        """
        Normalize encoder inputs to:
          - history_token_ids: (B, L, n_digit) token ids with per-digit offset
          - item_mask:         (B, L) bool, True=valid item, False=padding
        Supports both:
          - token-level input_ids: (B, L*n_digit)
          - codebook-level input_ids: (B, L, n_digit)
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # ---- reshape input_ids to (B, L, n_digit) ----
        if input_ids.dim() == 2:
            B, total_len = input_ids.shape
            if total_len % self.n_digit != 0:
                raise ValueError(
                    f"input_ids length must be divisible by n_digit={self.n_digit}, got {total_len}"
                )
            L = total_len // self.n_digit
            hist = input_ids.view(B, L, self.n_digit)
        elif input_ids.dim() == 3:
            B, L, D = input_ids.shape
            if D != self.n_digit:
                raise ValueError(f"input_ids last dim must be n_digit={self.n_digit}, got {D}")
            hist = input_ids
        else:
            raise ValueError(f"input_ids must be 2D or 3D, got shape={tuple(input_ids.shape)}")

        # ---- build item_mask (B, L) with True=valid ----
        item_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.shape == (B, L):
                    item_mask = attention_mask.bool()
                elif input_ids.dim() == 2 and attention_mask.shape == input_ids.shape:
                    item_mask = attention_mask.view(B, L, self.n_digit)[:, :, 0].bool()
                elif attention_mask.shape == (B, L * self.n_digit):
                    item_mask = attention_mask.view(B, L, self.n_digit)[:, :, 0].bool()
            elif attention_mask.dim() == 3 and attention_mask.shape[:2] == (B, L):
                item_mask = attention_mask[:, :, 0].bool()

        if item_mask is None:
            # Fallback: treat token-id padding (0) as padding.
            # If hist is codebook ids, this may be ambiguous for cb_id==0; but tokenizer always provides attention_mask.
            item_mask = hist[:, :, 0].ne(0)

        # ---- detect whether hist is already token ids or raw codebook ids ----
        needs_codebook_to_token = False
        if bool(item_mask.any().item()):
            # For token ids, valid positions at digit d should be >= sid_offset + d*codebook_size.
            # If we see values smaller than that on any digit>0, it's almost surely codebook ids.
            check_digits = range(1, self.n_digit) if self.n_digit > 1 else range(1)
            for d in check_digits:
                vals = hist[:, :, d][item_mask]
                if vals.numel() == 0:
                    continue
                if int(vals.min().item()) < int(self.sid_offset + d * self.codebook_size):
                    needs_codebook_to_token = True
                    break

        if needs_codebook_to_token:
            offsets = (
                torch.arange(self.n_digit, device=device, dtype=torch.long) * self.codebook_size + self.sid_offset
            ).view(1, 1, self.n_digit)
            hist = (hist.long() + offsets).clamp(0, self.vocab_size - 1)
            hist = hist.masked_fill(~item_mask.unsqueeze(-1), 0)
        else:
            hist = hist.long()

        return hist, item_mask.bool(), L
