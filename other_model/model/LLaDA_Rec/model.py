import os
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Tuple
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import AbstractModel
from tokenizer.tokenizer import AbstractTokenizer
from model.LLaDA import LLaDAConfig, LLaDAModelLM
from model.LLaDA.modeling_llada import (CausalLMOutputWithPast, ModuleType, init_weights)

@dataclass
class LLaDARecOutPut:
    loss: torch.Tensor
    his_mask_loss: torch.Tensor
    target_mask_loss: torch.Tensor

def load_llada_rec_yaml_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class LLaDA_Rec(AbstractModel):
    """
    LLaDA_Rec 适配流式训练流水线的版本 (保留原始掩码 Beam Search 逻辑)
    """
    def __init__(self, config: dict, tokenizer: AbstractTokenizer):
        super(LLaDA_Rec, self).__init__(config, tokenizer)

        yaml_config = load_llada_rec_yaml_config()
        merged_config = yaml_config.copy()
        merged_config.update(config)   # 外部传入优先级更高
        config = merged_config

        self.logger = getLogger()
        self.n_digit = config.get('n_digits', 8)
        
        # 忽略标签
        self.ignored_label = -100
        # 将 MASK token 设置为词表大小，并扩展词表容量
        self.mask_token_id = tokenizer.vocab_size 
        vocab_size = tokenizer.vocab_size + 1

        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.posValidTokens = self.get_posValidToken().to(device)

        lladaconfig = LLaDAConfig(
            activation_type=config.get('activation_type', 'silu'),
            attention_dropout=config.get('dropout_rate', 0.0),
            bias_for_layer_norm=config.get('bias_for_layer_norm', False),
            block_type=config.get('block_type', 'sequential'),
            d_model=config.get('d_model', 512),
            mlp_ratio=config.get('mlp_ratio', 4),
            embedding_dropout=config.get('dropout_rate', 0.0),
            init_fn=config.get('init_fn', 'normal'),
            layer_norm_type=config.get('layer_norm_type', 'rms'),
            max_sequence_length=config.get('max_token_seq_len', 512),
            n_heads=config.get('num_heads', 8),
            n_kv_heads=config.get('n_kv_heads', 8),
            n_layers=config.get('num_layers', 6),
            residual_dropout=config.get('dropout_rate', 0.0),
            rope=config.get('rope', True),
            rope_theta=config.get('rope_theta', 10000.0),
            weight_tying=config.get('weight_tying', False),
            vocab_size=vocab_size,
            embedding_size=vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.SPECIAL_TOKENS.get('[EOS]', 2),
            mask_token_id=self.mask_token_id,
            #init_device=None
        )
        # print("[DEBUG BEFORE BUILD] lladaconfig.init_device =", getattr(lladaconfig, "init_device", None), type(getattr(lladaconfig, "init_device", None)))
        # print("[DEBUG BEFORE BUILD] lladaconfig.precision =", getattr(lladaconfig, "precision", None), type(getattr(lladaconfig, "precision", None)))
        # print("[DEBUG BEFORE BUILD] lladaconfig.layer_norm_type =", getattr(lladaconfig, "layer_norm_type", None))
        # print("[DEBUG BEFORE BUILD] lladaconfig.block_type =", getattr(lladaconfig, "block_type", None))
        # print("[DEBUG BEFORE BUILD] lladaconfig.activation_type =", getattr(lladaconfig, "activation_type", None))
        # print("[DEBUG BEFORE BUILD] lladaconfig.__dict__.get('init_device') =", lladaconfig.__dict__.get("init_device", "NOT_FOUND"))

        self.llada = LLaDAModelLM(lladaconfig)
        self.llada.to(device)

        self.item_pos_emb = nn.Embedding(
            num_embeddings=config.get('max_item_seq_len', 50) + 2,
            embedding_dim=config.get('d_model', 512)
        )
        init_weights(self.llada.model.config, self.item_pos_emb, type_of_module=ModuleType.emb)

        self.temperature = self.config.get('temperature', 1.0)
        self.logger.info(f"Scale logits by temperature: {self.temperature}")

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignored_label, reduction='none')
        self.his_mask_w = self.config.get('his_mask_w', 1.0)

        self.gen_steps = config.get('gen_steps', 4)

    def get_posValidToken(self) -> torch.Tensor:
        """从流式 tokenizer 中动态构建有效 token 的位置约束"""
        posValidTokens = {}

        for item_id, tokens in self.tokenizer.item2sid.items():
            for pos, token in enumerate(tokens):
                if pos not in posValidTokens:
                    posValidTokens[pos] = set()
                posValidTokens[pos].add(token)
                
        max_token_num = 0
        for pos in posValidTokens:
            posValidTokens[pos] = sorted(list(posValidTokens[pos]))
            max_token_num = max(max_token_num, len(posValidTokens[pos]))

        posValidTokens_pt = torch.zeros((len(posValidTokens), max_token_num), dtype=torch.long)

        for pos in posValidTokens:
            posValidTokens_pt[pos, :len(posValidTokens[pos])] = torch.LongTensor(posValidTokens[pos])

        return posValidTokens_pt

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.llada.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n#Non-embedding parameters: {total_params - emb_params}\n#Total trainable parameters: {total_params}\n'

    def add_item_pos_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_pos_idx = torch.arange(input_ids.shape[1], device=input_ids.device)
        item_pos_idx = token_pos_idx // (self.n_digit)
        item_pos_idx = torch.clamp(item_pos_idx, max=self.config.get('max_item_seq_len', 50))

        token_pos_emb = self.item_pos_emb(item_pos_idx)
        token_pos_emb = token_pos_emb.unsqueeze(0).expand((input_ids.shape[0], -1, -1))

        input_embeds = self.llada.model.transformer.wte(input_ids)
        return input_embeds + token_pos_emb

    def llada_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.add_item_pos_emb(input_ids)
        return self.llada(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )

    def his_mask_loss(self, input_tokens, input_attention_mask) -> CausalLMOutputWithPast:
        masked_input_tokens, masked_indices, p_mask = self.get_masked_seqs(input_tokens, input_attention_mask)
        outputs = self.llada_forward(input_ids=masked_input_tokens, attention_mask=input_attention_mask.long())

        #############################
        # print("[DEBUG RANGE CHECK HIS]")
        # print(f"  masked_input_tokens.min={masked_input_tokens.min().item()}, max={masked_input_tokens.max().item()}")
        # assert masked_input_tokens.min().item() >= 0
        # assert masked_input_tokens.max().item() <= self.mask_token_id, (
        #     f"masked_input_tokens overflow: {masked_input_tokens.max().item()} > {self.mask_token_id}"
        # )

        logits = outputs.logits / self.temperature
        input_tokens = input_tokens.masked_fill(~input_attention_mask, self.ignored_label)

        token_loss = self.loss_fct(logits[masked_indices], input_tokens[masked_indices]) / p_mask[masked_indices]
        outputs.loss = token_loss.sum() / (input_tokens.shape[0] * input_tokens.shape[1])
        return outputs

    def target_mask_loss(self, input_tokens, input_attention_mask, label_tokens, label_attention_mask) -> CausalLMOutputWithPast:
        masked_label_tokens, masked_indices, p_mask = self.get_masked_seqs(label_tokens, label_attention_mask, is_target=True)

        ##############################
        # print("[DEBUG RANGE CHECK TARGET]")
        # print(f"  masked_label_tokens.min={masked_label_tokens.min().item()}, max={masked_label_tokens.max().item()}")
        # assert masked_label_tokens.min().item() >= 0
        # assert masked_label_tokens.max().item() <= self.mask_token_id, (
        #     f"masked_label_tokens overflow: {masked_label_tokens.max().item()} > {self.mask_token_id}"
        # )



        final_inputs_ids = torch.cat([input_tokens, masked_label_tokens], dim=1)
        final_attention_mask = torch.cat([input_attention_mask, label_attention_mask], dim=1)

        outputs = self.llada_forward(input_ids=final_inputs_ids, attention_mask=final_attention_mask.long())

        logits = outputs.logits[:, -masked_label_tokens.shape[1]:, :] / self.temperature
        label_tokens = label_tokens.masked_fill(~label_attention_mask, self.ignored_label)

        token_loss = self.loss_fct(logits[masked_indices], label_tokens[masked_indices]) / p_mask[masked_indices]
        outputs.loss = token_loss.sum() / (label_tokens.shape[0] * label_tokens.shape[1])
        return outputs

    def forward(self, input_ids=None, attention_mask=None, labels=None, users=None, batch=None, **kwargs):
        # 移除原先的 item_id2tokens 查表逻辑，直接接收 Token ID


        #####################################
        # print("[DEBUG RANGE CHECK BEFORE FORWARD]")
        # print(f"  tokenizer.vocab_size={self.tokenizer.vocab_size}")
        # print(f"  model mask_token_id={self.mask_token_id}")
        # print(f"  model embedding_size={self.llada.model.transformer.wte.weight.shape[0]}")
        # print(f"  input_ids.shape={tuple(input_ids.shape)}")
        # print(f"  input_ids.min={input_ids.min().item()}, input_ids.max={input_ids.max().item()}")
        # valid_labels = labels[labels != -100]
        # if valid_labels.numel() > 0:
        #     print(f"  labels(valid).min={valid_labels.min().item()}, labels(valid).max={valid_labels.max().item()}")
        # else:
        #     print("  labels(valid)=EMPTY")
        # # 强检查
        # assert input_ids.min().item() >= 0, f"input_ids has negative value: min={input_ids.min().item()}"
        # assert input_ids.max().item() <= self.mask_token_id, (
        #     f"input_ids overflow: max={input_ids.max().item()}, mask_token_id={self.mask_token_id}"
        # )
        # if valid_labels.numel() > 0:
        #     assert valid_labels.min().item() >= 0, f"labels has negative valid value: min={valid_labels.min().item()}"
        #     assert valid_labels.max().item() < self.mask_token_id, (
        #         f"labels overflow: max={valid_labels.max().item()}, tokenizer vocab max={self.mask_token_id - 1}"
        #     )
            

        
        if batch is not None:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
        elif isinstance(input_ids, dict):
            batch = input_ids
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

        input_tokens = input_ids
        input_attention_mask = attention_mask.bool()

        label_tokens = labels.clone()
        label_attention_mask = (label_tokens != -100)
        label_tokens = label_tokens.masked_fill(~label_attention_mask, 0)

        his_outputs = self.his_mask_loss(input_tokens, input_attention_mask)
        his_loss = his_outputs.loss

        target_outputs = self.target_mask_loss(input_tokens, input_attention_mask, label_tokens, label_attention_mask)
        target_loss = target_outputs.loss

        loss = target_loss + self.his_mask_w * his_loss

        return LLaDARecOutPut(loss=loss, his_mask_loss=his_loss, target_mask_loss=target_loss)

    def get_masked_seqs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_target: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_target:
            return self.get_masked_targets(input_ids, attention_mask)
        else:
            return self.get_masked_seqs_his(input_ids, attention_mask)

    def get_masked_seqs_his(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape
        masked_num = torch.randint(low=0, high=self.n_digit + 1, size=(b, max(1, l // self.n_digit)), device=input_ids.device)
        masked_prob = masked_num / self.n_digit
        masked_prob = torch.repeat_interleave(masked_prob, self.n_digit, dim=1)[:, :l]

        masked_indices = torch.rand((b, l), device=input_ids.device) < masked_prob
        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)

        noisy_batch = noisy_batch.masked_fill(~attention_mask, 0)
        masked_indices = masked_indices.masked_fill(~attention_mask, False)
        return noisy_batch, masked_indices, masked_prob

    def get_masked_targets(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l = input_ids.shape
        mask_num = torch.randint(1, self.n_digit + 1, (b, ), device=input_ids.device)
        p_mask = mask_num / self.n_digit

        rand_scores = torch.rand((b, l), device=input_ids.device)
        rand_scores = rand_scores.masked_fill(~attention_mask, 2.0)
        sorted_idx = torch.argsort(rand_scores, dim=1)

        sort_input_ids = input_ids.clone()
        sort_input_ids = sort_input_ids[torch.arange(b).unsqueeze(1), sorted_idx]
        sort_mask_pos = torch.arange(l, device=input_ids.device)[None, :] < mask_num[:, None]
        sort_input_ids[sort_mask_pos] = self.mask_token_id

        masked_input_ids = input_ids.clone()
        masked_input_ids[torch.arange(b).unsqueeze(1), sorted_idx] = sort_input_ids

        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        mask_positions[masked_input_ids == self.mask_token_id] = True

        return masked_input_ids, mask_positions, p_mask[:, None].repeat(1, l)

    def get_transfer_index(self, input_ids, logits, num_transfer_tokens):
        mask_index = (input_ids == self.mask_token_id)

        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = p.gather(index=x0.unsqueeze(-1), dim=-1).squeeze(-1)

        x0 = torch.where(mask_index, x0, input_ids)
        confidence = torch.where(mask_index, x0_p, -float('inf'))

        _, transfer_index = torch.topk(confidence, k=num_transfer_tokens, dim=-1)
        return transfer_index

    def prepare_beam_search_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size, num_beams):
        device = input_ids.device
        beam_label_ids = torch.ones(
            (batch_size * num_beams, self.n_digit),
            device=device,
            dtype=torch.long) * self.mask_token_id
        
        input_ids_rep = input_ids.repeat_interleave(num_beams, dim=0)
        all_input_ids = torch.cat([input_ids_rep, beam_label_ids], dim=-1)

        beam_label_attention_mask = torch.ones_like(beam_label_ids, device=device, dtype=torch.long)
        attention_mask_rep = attention_mask.repeat_interleave(num_beams, dim=0)
        all_attention_mask = torch.cat([attention_mask_rep, beam_label_attention_mask], dim=-1)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams, ))

        beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams

        return all_input_ids, all_attention_mask, initial_beam_scores, beam_idx_offset

    def beam_search_step(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        transfer_index: torch.Tensor,
        beam_scores: torch.Tensor,
        beam_idx_offset: torch.Tensor,
        batch_size,
        num_beams,
    ):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits.gather(
            index=transfer_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, vocab_size),
            dim=1).squeeze(1)

        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        his_len = input_ids.shape[1] - self.n_digit
        valid_tokens = self.posValidTokens[transfer_index - his_len]

        valid_tokens = valid_tokens.masked_fill(valid_tokens == 0, self.mask_token_id)
        next_token_scores = next_token_scores.gather(index=valid_tokens, dim=1)
        next_token_scores = next_token_scores.masked_fill(valid_tokens == self.mask_token_id, -float('inf'))
        vocab_size = next_token_scores.shape[-1]

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        input_ids = input_ids[beam_idx + beam_idx_offset]
        transfer_index = transfer_index[beam_idx + beam_idx_offset]

        valid_tokens = self.posValidTokens[transfer_index - his_len]
        beam_next_tokens = valid_tokens.gather(index=beam_next_tokens.unsqueeze(1), dim=1).squeeze(1)

        x_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        input_ids[x_index, transfer_index] = beam_next_tokens

        return input_ids, beam_scores

    def beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=9,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        """
        保留原始的置信度驱动的多步 Beam Search 逻辑
        增加了对 return_score 的兼容支持
        """
        assert num_beams >= num_return_sequences

        batch_size = input_ids.shape[0]
        n_digit = self.n_digit

        # Prepare beam search inputs
        all_input_ids, all_attention_mask, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(input_ids, attention_mask, batch_size, num_beams)

        num_transfer_tokens = max(1, n_digit // self.gen_steps)
        num_transfered = 0

        for i in range(self.gen_steps):
            if i == (self.gen_steps - 1):
                num_transfer_tokens = n_digit - num_transfered

            outputs = self.llada_forward(input_ids=all_input_ids, attention_mask=all_attention_mask.long())
            logits = outputs.logits / self.temperature

            transfer_index = self.get_transfer_index(all_input_ids, logits, num_transfer_tokens)

            for j in range(num_transfer_tokens):
                all_input_ids, beam_scores = self.beam_search_step(
                    logits, all_input_ids, transfer_index[:, j], beam_scores,
                    beam_idx_offset, batch_size, num_beams)

            num_transfered += num_transfer_tokens

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        label_ids = all_input_ids[:, -n_digit:]
        label_ids = label_ids[selection_mask.view(-1), :].reshape(-1, n_digit)
        
        final_scores = beam_scores[selection_mask.view(-1)]

        # 适配 pipeline 的切片预期: decoder_input_ids[:, 1:1+n_digit]
        # 在最前面插入一个 dummy token
        dummy_start = torch.zeros((batch_size * num_return_sequences, 1), dtype=torch.long, device=input_ids.device)
        decoder_input_ids = torch.cat([dummy_start, label_ids], dim=1)

        if return_score:
            return decoder_input_ids, final_scores
        return decoder_input_ids