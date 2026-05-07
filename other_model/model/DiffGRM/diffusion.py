import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 2. Mask + Loss 抽象：MaskDiffusion
# ===============================
class MaskDiffusion(nn.Module):
    """
    负责：
    1) 根据策略构造多视图 masked decoder_input_ids / labels / mask_positions
        - random
        - sequential
        - guided （需要外部提供 score_fn）
    2) 根据 digit_logits_fn 计算 loss
    """

    def __init__(self, config: dict, n_digit: int):
        super().__init__()
        self.config = config
        self.n_digit = n_digit

        # 以下属性由 DiffGRM.sync_from_model_config() 注入
        self.masking_strategy = 'random'
        self.mask_prob_random = False
        self.mask_probs = None
        self.augment_factor = 1
        self.seq_steps = None
        self.sequential_paths = 1
        self.guided_steps = None
        self.guided_conf_metric = 'msp'
        self.guided_select = 'least'
        self.guided_refresh_each_step = bool(self.config.get('guided_refresh_each_step', False))

    # ---- 与 DiffGRM 的配置同步（在 __init__ / set_masking_mode / resample 中调用） ----

    def sync_from_model_config(self, model: "DiffGRM"):
        """
        从 DiffGRM 实例同步当前 mask 策略配置。
        """
        self.masking_strategy = getattr(model, 'masking_strategy', 'random')
        self.mask_prob_random = bool(getattr(model, 'mask_prob_random', False))
        self.mask_probs = getattr(model, 'mask_probs', None)
        self.augment_factor = int(getattr(model, 'augment_factor', 1))

        self.seq_steps = getattr(model, 'seq_steps', None)
        self.sequential_paths = int(getattr(model, 'sequential_paths', 1))

        self.guided_steps = getattr(model, 'guided_steps', None)
        self.guided_conf_metric = getattr(model, 'guided_conf_metric', 'msp')
        self.guided_select = getattr(model, 'guided_select', 'least')
        self.guided_refresh_each_step = bool(
            model.config.get('guided_refresh_each_step', False)
        )

    # ----------- 1. 构造 multi-view 训练样本 -----------

    def build_views(
        self,
        decoder_input_ids: torch.Tensor,   # (B, n_digit)
        decoder_labels: torch.Tensor,      # (B, n_digit)
        encoder_hidden: torch.Tensor,      # (B, S, d)
        strategy: str = None,
        score_fn=None,                     # guided 时：score_fn(cur_inp, cur_mask, enc_hidden) → conf (B, n_digit)
    ):
        """
        返回：
            decoder_input_ids_exp: (B * n_views, n_digit)
            decoder_labels_exp:    (B * n_views, n_digit)
            mask_positions_exp:    (B * n_views, n_digit)
            encoder_hidden_exp:    (B * n_views, S, d)
            augment_factor:        n_views
        """
        device = decoder_input_ids.device
        B, n_digit = decoder_input_ids.shape
        assert n_digit == self.n_digit

        strategy = strategy or self.masking_strategy

        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []

        # -------- sequential --------
        if strategy == 'sequential':
            seq_steps = self.seq_steps
            if seq_steps in (None, 'auto'):
                seq_steps = self.n_digit
            seq_steps = int(seq_steps)
            paths = int(self.sequential_paths)
            assert 1 <= seq_steps <= self.n_digit
            assert paths >= 1

            augment_factor = seq_steps * paths

            for p in range(paths):
                # 每条路径一个随机顺序
                orders = torch.argsort(torch.rand(B, n_digit, device=device), dim=1)

                # step-0: 全掩码
                full_mask = torch.ones(B, n_digit, dtype=torch.bool, device=device)
                inp0 = decoder_input_ids.new_zeros(B, n_digit)
                all_masked_input_ids.append(inp0)
                all_labels.append(decoder_labels)
                all_mask_positions.append(full_mask.float())

                # step-1..seq_steps-1：逐步揭开
                for reveal in range(1, seq_steps):
                    mask_pos = torch.ones_like(full_mask)   # 先全 mask
                    reveal_idx = orders[:, :reveal]         # 本步揭开的列
                    mask_pos.scatter_(1, reveal_idx, 0)     # 0 = 不 mask

                    inp = decoder_input_ids.clone()
                    inp[mask_pos] = 0                       # 被 mask 的位写 0

                    all_masked_input_ids.append(inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(mask_pos.float())

        # -------- guided --------
        elif strategy == 'guided':
            if score_fn is None:
                raise ValueError("guided 策略需要提供 score_fn")

            guided_steps = self.guided_steps
            if guided_steps in (None, 'auto'):
                guided_steps = self.n_digit
            guided_steps = min(int(guided_steps), self.n_digit, 4)

            select = self.guided_select  # 'most' or 'least'
            refresh = bool(self.guided_refresh_each_step)

            def _score_with_mask(cur_mask: torch.Tensor):
                """
                cur_mask: (B, n_digit) bool, True=被掩盖
                score_fn 负责：
                    decoder_inp, cur_mask, encoder_hidden → conf (B, n_digit)
                """
                cur_inp = decoder_input_ids.new_zeros(B, n_digit)
                cur_inp[~cur_mask] = decoder_labels[~cur_mask]
                conf = score_fn(cur_inp, cur_mask, encoder_hidden)
                return conf  # (B, n_digit)

            if not refresh:
                # 一次性排序，不刷新：先用全掩盖打分得到顺序
                full_mask = torch.ones(B, n_digit, dtype=torch.bool, device=device)
                conf = _score_with_mask(full_mask)  # (B, n_digit)

                if select == 'most':
                    order = torch.argsort(conf, dim=1, descending=True)
                else:
                    order = torch.argsort(conf, dim=1, descending=False)

                for t in range(1, guided_steps + 1):
                    cur_mask = torch.zeros(B, n_digit, dtype=torch.bool, device=device)
                    cols = order[:, :t]   # 每个样本选择前 t 个位置掩盖
                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
            else:
                # 每一步刷新：每次根据当前 cur_mask 打分，选择一个新位置
                cur_mask = torch.zeros(B, n_digit, dtype=torch.bool, device=device)
                for t in range(1, guided_steps + 1):
                    conf = _score_with_mask(cur_mask)  # (B, n_digit)

                    if select == 'most':
                        conf = conf.masked_fill(cur_mask, -1e9)
                        cols = torch.argmax(conf, dim=1, keepdim=True)
                    else:
                        conf = conf.masked_fill(cur_mask, 1e9)
                        cols = torch.argmin(conf, dim=1, keepdim=True)

                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())

            augment_factor = guided_steps

        # -------- random (默认) --------
        else:
            # 若采用 batch-level random prob（LLaDA 风格）
            if self.mask_prob_random:
                low = float(self.config.get('mask_prob_random_min', 0.0))
                high = float(self.config.get('mask_prob_random_max', 1.0))
                prob = float(torch.empty(1, device=device).uniform_(low, high).item())
                mask_probs = [prob]
                augment_factor = 1
            else:
                if self.mask_probs is not None:
                    mask_probs = [float(p) for p in self.mask_probs]
                    augment_factor = len(mask_probs)
                else:
                    mask_prob = float(self.config.get('mask_prob', 0.5))
                    augment_factor = int(self.config.get('augment_factor', 4))
                    mask_probs = [mask_prob] * augment_factor

            for prob in mask_probs:
                mask_positions = (torch.rand(B, n_digit, device=device) < prob)
                # 保证每个样本至少有一个位置被 mask
                no_mask_samples = ~mask_positions.any(dim=1)
                if no_mask_samples.any():
                    mask_positions[no_mask_samples, 0] = True

                masked_input_ids = decoder_input_ids.clone()
                masked_input_ids[mask_positions] = 0

                all_masked_input_ids.append(masked_input_ids)
                all_labels.append(decoder_labels)
                all_mask_positions.append(mask_positions.float())

        # ---- 拼接 & repeat encoder_hidden ----
        decoder_input_ids_exp = torch.cat(all_masked_input_ids, dim=0)   # (B*views, n_digit)
        decoder_labels_exp = torch.cat(all_labels, dim=0)                # (B*views, n_digit)
        mask_positions_exp = torch.cat(all_mask_positions, dim=0)        # (B*views, n_digit)

        encoder_hidden_exp = encoder_hidden.repeat(augment_factor, 1, 1)

        # 保存 augment_factor，便于外部读取
        self.augment_factor = augment_factor

        return decoder_input_ids_exp, decoder_labels_exp, mask_positions_exp, encoder_hidden_exp, augment_factor

    # ----------- 2. loss 计算 -----------

    def compute_loss(
        self,
        decoder_hidden: torch.Tensor,   # (B_exp, n_digit, d)
        decoder_labels: torch.Tensor,   # (B_exp, n_digit)
        mask_positions: torch.Tensor,   # (B_exp, n_digit)
        digit_logits_fn,                # callable: (hidden_d, digit:int) -> logits (B_exp, K)
        label_smoothing: float = 0.0,
        strategy: str = None,
        mask_prob_random: bool = False,
    ):
        """
        根据 digit_logits_fn 计算总 loss。
        - 对 random + mask_prob_random=True 使用 LLaDA 风格按样本归一化
        - 其他情况：在掩码位上做 token 平均
        """
        strategy = strategy or self.masking_strategy
        B_exp, n_digit, _ = decoder_hidden.shape
        device = decoder_hidden.device

        # LLaDA 风格：按样本汇总，再除以 t
        if strategy == 'random' and mask_prob_random:
            per_sample_loss = torch.zeros(B_exp, device=device)

            for d in range(n_digit):
                logits_d = digit_logits_fn(decoder_hidden[:, d, :], d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()

                loss_d = F.cross_entropy(
                    logits_d,
                    labels_d,
                    reduction='none',
                    label_smoothing=label_smoothing,
                )
                per_sample_loss += loss_d * mask_d

            t_actual = mask_positions.float().mean(dim=1)  # (B_exp,)
            t_actual = torch.clamp(t_actual, min=1e-6)
            total_loss = (per_sample_loss / t_actual).mean()
            return total_loss

        # 其他策略：对所有掩码 token 总和 / 掩码 token 总数
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)

        for d in range(n_digit):
            logits_d = digit_logits_fn(decoder_hidden[:, d, :], d)
            labels_d = decoder_labels[:, d]
            mask_d = mask_positions[:, d].float()

            loss_d = F.cross_entropy(
                logits_d,
                labels_d,
                reduction='none',
                label_smoothing=label_smoothing,
            )

            total_loss += (loss_d * mask_d).sum()
            total_weight += mask_d.sum()

        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss