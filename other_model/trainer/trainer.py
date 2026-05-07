# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from model.model import AbstractModel
from tokenizer.tokenizer import AbstractTokenizer
from eval.evaluator import Evaluator
from utils import get_file_name, get_total_steps, config_for_log, log

import time

class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )
        
        # ==================== DEBUG HOOK（可选）====================
        # 如果 config 里设置了 debug_first_batch=True，
        # 在真正训练前，用 train_dataloader 的第一个 batch 做一轮 debug：
        # 1) teacher-forcing 预测
        # 2) HF generate vs 自己的 generate
        # if True:
        #     if self.accelerator.is_main_process:
        #         try:
        #             first_batch = next(iter(train_dataloader))
        #             self.log("Running debug_one_batch_teacher_forcing on first train batch")
        #             self.debug_one_batch_teacher_forcing(first_batch, max_print_len=8)

        #             self.log("Running debug_one_batch_generate on first train batch")
        #             # n_return_sequences 可以用 evaluator 的 maxk，或者自己指定一个小数
        #             n_ret = min(2, getattr(self.evaluator, "maxk", 2))
        #             self.debug_one_batch_generate(first_batch, n_return_sequences=n_ret, max_print_len=8)
        #         except StopIteration:
        #             self.log("train_dataloader is empty, skip debug_first_batch", level="warning")
        #     # 让所有进程在这里同步一下
        #     self.accelerator.wait_for_everyone()
        # ==================== DEBUG HOOK END =====================

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            # Training
            # self.print_recovery_accuracy_over_k(train_dataloader)
            # self.print_recovery_accuracy_over_k_with_conf_stats(train_dataloader)
            # self.print_recovery_accuracy_over_k_with_conf_stats(val_dataloader)
            # self.print_recovery_accuracy_over_k(val_dataloader)
            # self.print_recovery_accuracy_over_alpha_grid_for_each_k(train_dataloader)
            # self.print_recovery_accuracy_over_alpha_grid_for_each_k(val_dataloader)
            self.model.train()
            total_loss = 0.0
            # total_length = len(train_dataloader)
            # train_progress_bar = tqdm(
            #     train_dataloader,
            #     total=total_length,
            #     miniters=int(total_length * 0.5),
            #     desc=f"Training - [Epoch {epoch + 1}]",
            #     disable=False
            # )
            
            start_time = time.time()
            for batch in train_dataloader:
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                # logits = outputs.logits  # (B, T, V)
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()
                
            # self.model.debug_one_batch_teacher_forcing(batch, topk=5, max_print_len=4)

            self.log(
                f"[Epoch {epoch+1}] Time: {time.time()-start_time:.1f}s Train Loss={total_loss / len(train_dataloader):.4f}"
            )
            # self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            # self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                #_ = self.evaluate(train_dataloader, split='train')
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score

    def debug_one_batch_teacher_forcing(self, batch, max_print_len=8):
        """
        在 teacher-forcing 条件下检查 T5 的逐 token 预测情况：
        - 打印 input_ids[0]
        - 打印 labels[0]
        - 打印 logits.argmax 后的 pred_ids[0]
        """

        # 关键：以 model 当前所在 device 为准，避免 CPU/GPU 不一致
        model_device = next(self.model.parameters()).device
        batch = {k: v.to(model_device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,   # 让 HF 自动 shift_right
            )
            logits = outputs.logits  # (B, T, V)
            pred_ids = logits.argmax(dim=-1)  # (B, T)

        b0_inp = input_ids[0].detach().cpu()
        b0_att = attention_mask[0].detach().cpu()
        b0_lab = labels[0].detach().cpu()
        b0_pred = pred_ids[0].detach().cpu()

        # 只打印前 max_print_len 个，避免太长
        self.logger.info("=== [Teacher Forcing Debug] ===")
        self.logger.info("Input IDs[0]      : %s", b0_inp[:max_print_len].tolist())
        self.logger.info("Attention Mask[0] : %s", b0_att[:max_print_len].tolist())
        self.logger.info("Labels[0]         : %s", b0_lab[:max_print_len].tolist())
        self.logger.info("Pred IDs[0]       : %s", b0_pred[:max_print_len].tolist())

        same = (b0_lab[:max_print_len] == b0_pred[:max_print_len]).float()
        self.logger.info("Match ratio (labels==pred) on sample[0]: %.4f", same.mean().item())

        return {
            "input_ids": b0_inp,
            "attention_mask": b0_att,
            "labels": b0_lab,
            "pred_ids": b0_pred,
        }


    def debug_one_batch_generate(self, batch, n_return_sequences=2, max_print_len=8):
        """
        对比:
        - HF 自带 self.model.t5.generate
        - 你自己写的 self.model.generate(batch, n_return_sequences)

        前提:
        - batch 至少包含 input_ids, attention_mask, labels
        - self.model.generate(batch, n_return_sequences) 返回 (B, n_return_sequences, n_digit)
        """

        model_device = next(self.model.parameters()).device
        batch = {k: v.to(model_device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        B = input_ids.size(0)
        n_digit = labels.shape[1]

        max_length = 1 + n_digit
        num_beams = self.config.get('num_beams', 1)

        self.model.eval()
        with torch.no_grad():
            # 1) HF generate
            gen_hf = self.model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=n_return_sequences,
                early_stopping=False,
                do_sample=False,
            )  # (B * n_return_sequences, max_length)

            gen_hf = gen_hf.view(B, n_return_sequences, max_length)
            gen_hf_digits = gen_hf[:, :, 1:1+n_digit]

            # 2) 你自己的 generate
            gen_custom = self.model.generate(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                n_return_sequences=n_return_sequences,
            )  # (B, n_return_sequences, n_digit)

        self.logger.info("=== [Generate Debug] ===")
        self.logger.info("Labels[0]              : %s", labels[0, :max_print_len].detach().cpu().tolist())

        self.logger.info("HF generate digits[0]:")
        for i in range(min(n_return_sequences, 3)):
            self.logger.info(
                "  seq %d: %s", i,
                gen_hf_digits[0, i, :max_print_len].detach().cpu().tolist()
            )

        self.logger.info("Custom generate digits[0]:")
        for i in range(min(n_return_sequences, 3)):
            self.logger.info(
                "  seq %d: %s", i,
                gen_custom[0, i, :max_print_len].detach().cpu().tolist()
            )

        return {
            "labels": labels[0].detach().cpu(),
            "hf_digits": gen_hf_digits[0].detach().cpu(),
            "custom_digits": gen_custom[0].detach().cpu(),
        }


    def run_debug_on_dataloader(self, dataloader, n_batches=1, n_return_sequences=2):
        """
        从 dataloader 里抽 n_batches 个 batch，依次做：
        1) teacher-forcing 下的预测对比
        2) HF generate vs 自定义 generate 对比
        """
        from itertools import islice

        for step, batch in enumerate(islice(dataloader, n_batches)):
            self.logger.info("========== Debug batch %d ==========", step)

            # 1) teacher-forcing 调试
            self.debug_one_batch_teacher_forcing(batch, max_print_len=8)

            # 2) generate 调试
            self.debug_one_batch_generate(batch, n_return_sequences=n_return_sequences, max_print_len=8)

    def overfit_one_batch(self, train_dataloader, steps=500, log_every=50):
        """
        单 batch 过拟合实验：
        - 从 train_dataloader 取出第一个 batch
        - 只在这个 batch 上训练若干步
        - 持续打印 loss 和 teacher-forcing 下的预测情况

        用法示例：
            trainer.overfit_one_batch(train_dataloader, steps=500, log_every=50)
        """

        # 1) 选设备：直接用 accelerator 的 device，并把模型搬上去
        device = self.accelerator.device
        self.model.to(device)

        # 2) 取一个 batch（不会走 accelerator.prepare）
        try:
            batch = next(iter(train_dataloader))
        except StopIteration:
            self.log("train_dataloader is empty, cannot overfit one batch.", level="warning")
            return

        # 把 batch 搬到同一个 device 上
        batch = {k: v.to(device) for k, v in batch.items()}

        # 3) 建一个临时 optimizer（不用 scheduler / accelerator，单机调试就好）
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 1e-3),
            weight_decay=self.config.get('weight_decay', 0.0),
        )

        self.log(f"[Overfit] Start overfitting 1 batch for {steps} steps", level="info")
        self.model.train()

        for step in range(steps):
            optimizer.zero_grad()

            # 和正常训练保持一致：直接调用 self.model(batch)
            outputs = self.model(batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            if (step + 1) % log_every == 0 or step == 0:
                self.log(f"[Overfit] step {step+1}/{steps}, loss={loss.item():.6f}", level="info")
                # 看一下同一个 batch 在 teacher-forcing 下的预测有没有变好
                self.debug_one_batch_teacher_forcing(batch, max_print_len=8)

        self.log("[Overfit] Finished overfitting 1 batch.", level="info")

    @torch.no_grad()
    def print_recovery_accuracy_over_k(
        self,
        dataloader,
        device=None,
        personalized_strength: float = 1.0,
        max_batches: int = None,
    ):
        """
        遍历 k=0..M，测量从 x_t 还原 x_0 的精度，并直接打印结果表格（不返回）。

        指标：
        - acc_all       : 全位置 accuracy
        - acc_perturbed : 仅在被加噪位置上的 accuracy（更关键）
        - pert_ratio    : 被加噪位置占比（用于理解难度）
        - acc_seq       : sequence-level accuracy（整条 SID 全对）

        额外 confidence 统计（logits 已经是 softmax 后 probs）：
        - correct / wrong 的 entropy & pmax（全位置）
        - correct / wrong 的 entropy & pmax（仅 perturbed 位置）
        """

        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()

        # --- 拿 M ---
        first_batch = None
        for b in dataloader:
            first_batch = b
            break
        assert first_batch is not None, "Empty dataloader."
        M = int(first_batch["sid_labels"].shape[1])

        # 统计：每个 k 的 correct/total
        stats_all = defaultdict(lambda: [0, 0])         # k -> [correct, total]
        stats_pert = defaultdict(lambda: [0, 0])        # k -> [correct, total]
        stats_pert_ratio = defaultdict(lambda: [0, 0])  # k -> [pert_count, total_count]
        stats_seq = defaultdict(lambda: [0, 0])         # k -> [correct_seq, total_seq]

        # confidence stats: k -> [sum_entropy, sum_pmax, count]
        stats_conf_correct = defaultdict(lambda: [0.0, 0.0, 0])
        stats_conf_wrong   = defaultdict(lambda: [0.0, 0.0, 0])
        stats_conf_correct_pert = defaultdict(lambda: [0.0, 0.0, 0])
        stats_conf_wrong_pert   = defaultdict(lambda: [0.0, 0.0, 0])

        batch_count = 0
        eps = 1e-12

        for batch in dataloader:
            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                break

            user_history_sids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            x0 = batch["sid_labels"].to(device)  # (B, M)
            B = x0.size(0)

            # probs_fn 闭包（你自己的推理函数）
            probs_fn = self.model._get_infer_func_wo_cfg(
                user_history_sids, attention_mask, personalized_strength
            )

            for kk in range(M + 1):
                k = torch.full((B,), kk, device=device, dtype=torch.long)

                # 1) x0 -> xt
                xt, sigma_t = self.model.DiscreteDiffusion.forward_perturb_conjugate(x0, k)  # xt: (B,M)

                # 2) predict x0_hat from (xt, sigma_t)
                try:
                    out = probs_fn(xt, sigma_t, num_beams=1)
                except TypeError:
                    out = probs_fn(user_history_sids, attention_mask, xt, sigma_t)

                if isinstance(out, dict):
                    if "logits" in out:
                        out = out["logits"]
                    elif "probs" in out:
                        out = out["probs"]
                    elif "log_score" in out:
                        out = out["log_score"]
                    else:
                        raise ValueError(f"Unknown output dict keys: {list(out.keys())}")

                # reshape to (B, M, V)
                if out.dim() != 3:
                    raise ValueError(f"Expect 3D output, got dim={out.dim()} shape={tuple(out.shape)}")
                if out.shape[1] == M:
                    logits = out
                elif out.shape[2] == M:
                    logits = out.transpose(1, 2)
                else:
                    raise ValueError(f"Cannot infer M in out.shape={tuple(out.shape)}, M={M}")

                # logits 已经是 softmax 后概率
                probs = logits  # (B, M, V)
                x0_hat = probs.argmax(dim=-1)  # (B, M)

                # 3) acc_all
                correct_all = (x0_hat == x0)
                stats_all[kk][0] += int(correct_all.sum().item())
                stats_all[kk][1] += int(correct_all.numel())

                # 5) sequence-level accuracy（整条 SID）
                seq_correct = correct_all.all(dim=1)  # (B,)
                stats_seq[kk][0] += int(seq_correct.sum().item())
                stats_seq[kk][1] += int(seq_correct.numel())

                # 4) acc_perturbed + pert_ratio
                pert_mask = (xt != x0)

                pert_cnt = int(pert_mask.sum().item())
                tot_cnt = int(pert_mask.numel())
                stats_pert_ratio[kk][0] += pert_cnt
                stats_pert_ratio[kk][1] += tot_cnt

                if pert_cnt > 0:
                    pert_correct = int((correct_all & pert_mask).sum().item())
                    stats_pert[kk][0] += pert_correct
                    stats_pert[kk][1] += pert_cnt

                # -------- confidence: entropy & pmax (all vs perturbed, correct vs wrong) --------
                probs_clamped = probs.clamp(min=eps, max=1.0)
                pmax = probs_clamped.max(dim=-1).values                         # (B, M)
                entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)    # (B, M)

                correct_mask = correct_all
                wrong_mask = ~correct_mask

                # all positions
                c_cnt = int(correct_mask.sum().item())
                w_cnt = int(wrong_mask.sum().item())
                if c_cnt > 0:
                    stats_conf_correct[kk][0] += float(entropy[correct_mask].sum().item())
                    stats_conf_correct[kk][1] += float(pmax[correct_mask].sum().item())
                    stats_conf_correct[kk][2] += c_cnt
                if w_cnt > 0:
                    stats_conf_wrong[kk][0] += float(entropy[wrong_mask].sum().item())
                    stats_conf_wrong[kk][1] += float(pmax[wrong_mask].sum().item())
                    stats_conf_wrong[kk][2] += w_cnt

                # perturbed-only positions
                correct_mask_pert = correct_mask & pert_mask
                wrong_mask_pert = wrong_mask & pert_mask

                c_cnt_p = int(correct_mask_pert.sum().item())
                w_cnt_p = int(wrong_mask_pert.sum().item())
                if c_cnt_p > 0:
                    stats_conf_correct_pert[kk][0] += float(entropy[correct_mask_pert].sum().item())
                    stats_conf_correct_pert[kk][1] += float(pmax[correct_mask_pert].sum().item())
                    stats_conf_correct_pert[kk][2] += c_cnt_p
                if w_cnt_p > 0:
                    stats_conf_wrong_pert[kk][0] += float(entropy[wrong_mask_pert].sum().item())
                    stats_conf_wrong_pert[kk][1] += float(pmax[wrong_mask_pert].sum().item())
                    stats_conf_wrong_pert[kk][2] += w_cnt_p

        # --- 打印表格 ---
        self.log("\n=== Recovery Accuracy over k (0..M) ===")
        self.log(f"Evaluated batches: {batch_count}  |  M={M} ")
        self.log("-" * 170)
        self.log(
            f"{'k':>4} | {'pert_ratio':>10} | {'acc_perturbed':>14} | {'acc_all':>10} | {'acc_seq':>8} || "
            f"{'Hc_all':>8} {'Pc_all':>8} | {'Hw_all':>8} {'Pw_all':>8} || "
            f"{'Hc_pert':>8} {'Pc_pert':>8} | {'Hw_pert':>8} {'Pw_pert':>8}"
        )
        self.log("-" * 170)

        for kk in range(M + 1):
            # pert ratio
            pc, pt = stats_pert_ratio[kk]
            pert_ratio = (pc / pt) if pt > 0 else 0.0

            # acc_perturbed
            c, t = stats_pert[kk]
            acc_pert = (c / t) if t > 0 else 0.0

            # acc_all
            c, t = stats_all[kk]
            acc_all = (c / t) if t > 0 else 0.0

            # acc_seq
            c, t = stats_seq[kk]
            acc_seq = (c / t) if t > 0 else 0.0

            # confidence (all positions)
            se, sp, cnt = stats_conf_correct[kk]
            Hc_all = (se / cnt) if cnt > 0 else 0.0
            Pc_all = (sp / cnt) if cnt > 0 else 0.0

            se, sp, cnt = stats_conf_wrong[kk]
            Hw_all = (se / cnt) if cnt > 0 else 0.0
            Pw_all = (sp / cnt) if cnt > 0 else 0.0

            # confidence (perturbed only)
            se, sp, cnt = stats_conf_correct_pert[kk]
            Hc_pert = (se / cnt) if cnt > 0 else 0.0
            Pc_pert = (sp / cnt) if cnt > 0 else 0.0

            se, sp, cnt = stats_conf_wrong_pert[kk]
            Hw_pert = (se / cnt) if cnt > 0 else 0.0
            Pw_pert = (sp / cnt) if cnt > 0 else 0.0

            self.log(
                f"{kk:4d} | {pert_ratio:10.4f} | {acc_pert:14.4f} | {acc_all:10.4f} | {acc_seq:8.4f} || "
                f"{Hc_all:8.4f} {Pc_all:8.4f} | {Hw_all:8.4f} {Pw_all:8.4f} || "
                f"{Hc_pert:8.4f} {Pc_pert:8.4f} | {Hw_pert:8.4f} {Pw_pert:8.4f}"
            )

        self.log("-" * 170)
        self.log(
            "Notes: acc_perturbed 只统计被加噪位置；acc_all 会被未扰动位置拉高。\n"
            "      H 越小 / pmax 越大 => 越自信；重点关注 wrong@pert 是否出现高 pmax（过度自信）。\n"
        )

    @torch.no_grad()
    def print_recovery_accuracy_over_k_with_conf_stats(
        self,
        dataloader,
        device=None,
        personalized_strength: float = 1.0,
        max_batches: int = None,
        eps: float = 1e-12,
    ):
        """
        遍历 k=0..M，测量从 x_t 还原 x_0 的精度，并打印表格。

        指标：
        - acc_all       : 全位置 accuracy
        - acc_perturbed : 仅在被加噪位置上的 accuracy
        - pert_ratio    : 被加噪位置占比
        - acc_seq       : sequence-level accuracy（整条 SID 全对）

        confidence 统计（probs 已是 softmax 后）：
        - correct / wrong 的 entropy & pmax（全位置）
        - correct / wrong 的 entropy & pmax（仅 perturbed 位置）
        输出 mean ± std
        """

        if device is None:
            device = next(self.model.parameters()).device
        self.model.eval()

        # --- 拿 M ---
        first_batch = None
        for b in dataloader:
            first_batch = b
            break
        assert first_batch is not None, "Empty dataloader."
        M = int(first_batch["sid_labels"].shape[1])

        # --- accuracy stats ---
        stats_all = defaultdict(lambda: [0, 0])         # k -> [correct, total]
        stats_pert = defaultdict(lambda: [0, 0])        # k -> [correct, total(pert)]
        stats_pert_ratio = defaultdict(lambda: [0, 0])  # k -> [pert_count, total_count]
        stats_seq = defaultdict(lambda: [0, 0])         # k -> [correct_seq, total_seq]

        # --- confidence stats with mean±std: store sum, sumsq, count ---
        def _init_moments():
            return [0.0, 0.0, 0]  # sum, sumsq, count

        # key: (k) -> moments
        Hc_all = defaultdict(_init_moments)
        Pc_all = defaultdict(_init_moments)
        Hw_all = defaultdict(_init_moments)
        Pw_all = defaultdict(_init_moments)

        Hc_pert = defaultdict(_init_moments)
        Pc_pert = defaultdict(_init_moments)
        Hw_pert = defaultdict(_init_moments)
        Pw_pert = defaultdict(_init_moments)

        def _accum(mom, x: torch.Tensor):
            # x: 1D tensor
            if x.numel() == 0:
                return
            s = float(x.sum().item())
            ss = float((x * x).sum().item())
            c = int(x.numel())
            mom[0] += s
            mom[1] += ss
            mom[2] += c

        batch_count = 0

        for batch in dataloader:
            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                break

            user_history_sids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            x0 = batch["sid_labels"].to(device)  # (B, M)
            B = x0.size(0)

            # probs_fn 闭包（你自己的推理函数）
            probs_fn = self.model._get_infer_func_wo_cfg(
                user_history_sids, attention_mask, personalized_strength
            )

            for kk in range(M + 1):
                k = torch.full((B,), kk, device=device, dtype=torch.long)

                # 1) x0 -> xt
                xt, sigma_t = self.model.DiscreteDiffusion.forward_perturb_conjugate(x0, k)  # xt: (B,M)

                # 2) predict p(x0 | xt, sigma_t)
                try:
                    out = probs_fn(xt, sigma_t, num_beams=1)
                except TypeError:
                    out = probs_fn(user_history_sids, attention_mask, xt, sigma_t)

                if isinstance(out, dict):
                    if "logits" in out:
                        out = out["logits"]
                    elif "probs" in out:
                        out = out["probs"]
                    elif "log_score" in out:
                        out = out["log_score"]
                    else:
                        raise ValueError(f"Unknown output dict keys: {list(out.keys())}")

                if out.dim() != 3:
                    raise ValueError(f"Expect 3D output, got dim={out.dim()} shape={tuple(out.shape)}")

                if out.shape[1] == M:
                    probs = out
                elif out.shape[2] == M:
                    probs = out.transpose(1, 2)
                else:
                    raise ValueError(f"Cannot infer M in out.shape={tuple(out.shape)}, M={M}")

                # probs 已经是 softmax 后概率
                probs = probs.clamp(min=eps, max=1.0)        # (B, M, V)
                x0_hat = probs.argmax(dim=-1)                # (B, M)

                # accuracy masks
                correct_all = (x0_hat == x0)                 # (B, M)
                pert_mask = (xt != x0)                       # (B, M)

                # acc_all
                stats_all[kk][0] += int(correct_all.sum().item())
                stats_all[kk][1] += int(correct_all.numel())

                # acc_seq
                seq_correct = correct_all.all(dim=1)         # (B,)
                stats_seq[kk][0] += int(seq_correct.sum().item())
                stats_seq[kk][1] += int(seq_correct.numel())

                # pert_ratio
                pert_cnt = int(pert_mask.sum().item())
                stats_pert_ratio[kk][0] += pert_cnt
                stats_pert_ratio[kk][1] += int(pert_mask.numel())

                # acc_perturbed
                if pert_cnt > 0:
                    stats_pert[kk][0] += int((correct_all & pert_mask).sum().item())
                    stats_pert[kk][1] += pert_cnt

                # -------- confidence stats (entropy & pmax) --------
                pmax = probs.max(dim=-1).values                          # (B, M)
                entropy = -(probs * probs.log()).sum(dim=-1)             # (B, M)

                wrong_all = ~correct_all

                # all positions: correct / wrong
                _accum(Hc_all[kk], entropy[correct_all])
                _accum(Pc_all[kk], pmax[correct_all])
                _accum(Hw_all[kk], entropy[wrong_all])
                _accum(Pw_all[kk], pmax[wrong_all])

                # pert-only positions
                correct_pert = correct_all & pert_mask
                wrong_pert = wrong_all & pert_mask
                _accum(Hc_pert[kk], entropy[correct_pert])
                _accum(Pc_pert[kk], pmax[correct_pert])
                _accum(Hw_pert[kk], entropy[wrong_pert])
                _accum(Pw_pert[kk], pmax[wrong_pert])

        def _mean_std(mom):
            s, ss, c = mom
            if c <= 0:
                return 0.0, 0.0
            mean = s / c
            var = max(ss / c - mean * mean, 0.0)
            std = var ** 0.5
            return mean, std

        # --- 打印表格 ---
        self.log("\n=== Recovery Accuracy over k (0..M) with Confidence mean±std ===")
        self.log(f"Evaluated batches: {batch_count}  |  M={M}")
        self.log("-" * 220)
        self.log(
            f"{'k':>4} | {'pert_ratio':>10} | {'acc_pert':>10} | {'acc_all':>10} | {'acc_seq':>10} || "
            f"{'Hc_all':>18} {'Pc_all':>18} | {'Hw_all':>18} {'Pw_all':>18} || "
            f"{'Hc_pert':>18} {'Pc_pert':>18} | {'Hw_pert':>18} {'Pw_pert':>18}"
        )
        self.log("-" * 220)

        for kk in range(M + 1):
            pc, pt = stats_pert_ratio[kk]
            pert_ratio = (pc / pt) if pt > 0 else 0.0

            c, t = stats_pert[kk]
            acc_pert = (c / t) if t > 0 else 0.0

            c, t = stats_all[kk]
            acc_all = (c / t) if t > 0 else 0.0

            c, t = stats_seq[kk]
            acc_seq = (c / t) if t > 0 else 0.0

            hc_m, hc_s = _mean_std(Hc_all[kk])
            pc_m, pc_s = _mean_std(Pc_all[kk])
            hw_m, hw_s = _mean_std(Hw_all[kk])
            pw_m, pw_s = _mean_std(Pw_all[kk])

            hcp_m, hcp_s = _mean_std(Hc_pert[kk])
            pcp_m, pcp_s = _mean_std(Pc_pert[kk])
            hwp_m, hwp_s = _mean_std(Hw_pert[kk])
            pwp_m, pwp_s = _mean_std(Pw_pert[kk])

            self.log(
                f"{kk:4d} | {pert_ratio:10.4f} | {acc_pert:10.4f} | {acc_all:10.4f} | {acc_seq:10.4f} || "
                f"{hc_m:8.4f}±{hc_s:8.4f} {pc_m:8.4f}±{pc_s:8.4f} | "
                f"{hw_m:8.4f}±{hw_s:8.4f} {pw_m:8.4f}±{pw_s:8.4f} || "
                f"{hcp_m:8.4f}±{hcp_s:8.4f} {pcp_m:8.4f}±{pcp_s:8.4f} | "
                f"{hwp_m:8.4f}±{hwp_s:8.4f} {pwp_m:8.4f}±{pwp_s:8.4f}"
            )

        self.log("-" * 220)
        self.log(
            "Notes: acc_pert 只统计被加噪位置；acc_all 会被未扰动位置拉高。\n"
            "      H 越小 / pmax 越大 => 越自信；重点关注 wrong@pert 的 pmax 是否异常偏大（过度自信）。\n"
        )

    @torch.no_grad()
    def print_recovery_accuracy_over_alpha_grid_for_each_k(
        self,
        dataloader,
        alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        device=None,
        personalized_strength: float = 1.0,
        max_batches: int = None,
        alpha_eps: float = 1e-6,
    ):
        """
        对每个 k=0..M:
        - 用 forward_perturb_conjugate(x0,k) 生成 xt（保持噪声位置/幅度与k一致）
        - 但在推理时，手工指定 sigma_t = -log(alpha_t)，alpha_t 从 alpha_grid 扫描
        - 统计 acc_all / acc_pert / pert_ratio / acc_seq
        不计算 confidence。

        目的：研究 sigma_t（即 timestep embedding/conditioning）的影响，与 k 解耦观察。
        """

        if device is None:
            device = next(self.model.parameters()).device
        self.model.eval()

        # --- 拿 M ---
        first_batch = None
        for b in dataloader:
            first_batch = b
            break
        assert first_batch is not None, "Empty dataloader."
        M = int(first_batch["sid_labels"].shape[1])

        # stats[(k, aidx)] -> [correct_all, total_all, correct_pert, total_pert, correct_seq, total_seq, pert_cnt, total_cnt]
        stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])

        batch_count = 0

        for batch in dataloader:
            batch_count += 1
            if max_batches is not None and batch_count > max_batches:
                break

            user_history_sids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            x0 = batch["sid_labels"].to(device)  # (B,M)
            B = x0.size(0)

            probs_fn = self.model._get_infer_func_wo_cfg(
                user_history_sids, attention_mask, personalized_strength
            )

            for kk in range(M + 1):
                k = torch.full((B,), kk, device=device, dtype=torch.long)

                # xt 的生成仍依赖 k
                xt, _sigma_from_forward = self.model.DiscreteDiffusion.forward_perturb_conjugate(x0, k)  # (B,M)
                pert_mask = (xt != x0)
                pert_cnt = int(pert_mask.sum().item())
                tot_cnt = int(pert_mask.numel())

                for aidx, a in enumerate(alpha_grid):
                    alpha = torch.tensor(float(a), device=device)
                    alpha = alpha.clamp(min=alpha_eps, max=1.0)
                    sigma_t = (-alpha.log()).expand(B)  # (B,)
                    
                    sigma_t = sigma_t.unsqueeze(-1).expand(-1, M)
                    # 如果你模型必须 [B,1,1]，就改成：
                    # sigma_t = (-alpha.log()).view(1,1,1).expand(B,1,1)

                    # 推理
                    try:
                        out = probs_fn(xt, sigma_t, num_beams=1)
                    except TypeError:
                        out = probs_fn(user_history_sids, attention_mask, xt, sigma_t)

                    if isinstance(out, dict):
                        if "logits" in out:
                            out = out["logits"]
                        elif "probs" in out:
                            out = out["probs"]
                        elif "log_score" in out:
                            out = out["log_score"]
                        else:
                            raise ValueError(f"Unknown output dict keys: {list(out.keys())}")

                    if out.dim() != 3:
                        raise ValueError(f"Expect 3D output, got dim={out.dim()} shape={tuple(out.shape)}")
                    if out.shape[1] == M:
                        probs = out
                    elif out.shape[2] == M:
                        probs = out.transpose(1, 2)
                    else:
                        raise ValueError(f"Cannot infer M in out.shape={tuple(out.shape)}, M={M}")

                    x0_hat = probs.argmax(dim=-1)          # (B,M)
                    correct_all = (x0_hat == x0)           # (B,M)

                    key = (kk, aidx)
                    st = stats[key]

                    # acc_all
                    st[0] += int(correct_all.sum().item())
                    st[1] += int(correct_all.numel())

                    # acc_pert
                    st[6] += pert_cnt
                    st[7] += tot_cnt
                    if pert_cnt > 0:
                        st[2] += int((correct_all & pert_mask).sum().item())
                        st[3] += pert_cnt

                    # acc_seq
                    seq_correct = correct_all.all(dim=1)   # (B,)
                    st[4] += int(seq_correct.sum().item())
                    st[5] += int(seq_correct.numel())

        # --- 打印 ---
        self.log("\n=== Recovery Accuracy over alpha_grid for each k ===")
        self.log(f"Evaluated batches: {batch_count} | M={M} | alpha_grid={list(alpha_grid)} (clamped with eps={alpha_eps})")
        self.log("-" * 120)

        # header
        header = f"{'k':>3} | {'alpha':>6} | {'sigma=-log(alpha)':>14} | {'pert_ratio':>10} | {'acc_pert':>10} | {'acc_all':>10} | {'acc_seq':>10}"
        self.log(header)
        self.log("-" * 120)

        for kk in range(M + 1):
            for aidx, a in enumerate(alpha_grid):
                alpha = max(min(float(a), 1.0), alpha_eps)
                sigma = float(-torch.log(torch.tensor(alpha)).item())

                st = stats[(kk, aidx)]
                correct_all, total_all, correct_pert, total_pert, correct_seq, total_seq, pert_cnt_sum, tot_cnt_sum = st

                pert_ratio = (pert_cnt_sum / tot_cnt_sum) if tot_cnt_sum > 0 else 0.0
                acc_pert = (correct_pert / total_pert) if total_pert > 0 else 0.0
                acc_all = (correct_all / total_all) if total_all > 0 else 0.0
                acc_seq = (correct_seq / total_seq) if total_seq > 0 else 0.0

                self.log(
                    f"{kk:3d} | {float(a):6.2f} | {sigma:14.6f} | {pert_ratio:10.4f} | {acc_pert:10.4f} | {acc_all:10.4f} | {acc_seq:10.4f}"
                )

            self.log("-" * 120)

        self.log(
            "Notes: xt 仍由 k 决定（噪声位置/强度模式）；但推理条件 sigma_t 被手工设为 -log(alpha)。\n"
            "      这能帮助你判断模型到底依赖 k 所决定的 xt 本身，还是强依赖 sigma_t 的 conditioning。\n"
        )


    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        # val_progress_bar = tqdm(
        #     dataloader,
        #     total=len(dataloader),
        #     miniters=int(len(dataloader) * 0.5),
        #     desc=f"Eval - {split}",
        #     disable=False
        # )
        start_time = time.time()
        for batch in dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                label_key = "sid_labels" if self.config.get("is_diffusion_model", False) else "labels"
                labels = batch[label_key]
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, labels))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    #self.logger.info("\nInput IDs %s", batch['input_ids'][23])
                    #self.logger.info("\nAttention Mask %s", batch['attention_mask'][23])
                    #self.logger.info("\nlabels %s", batch['labels'][23])
                    #self.logger.info("\nGenerated SIDs %s", preds[23, :2])
                    results = self.evaluator.calculate_metrics(preds, labels)

                for key, value in results.items():
                    all_results[key].append(value)
                    
        self.log(f"[Eval - {split}] Time: {time.time()-start_time:.1f}s")

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        code_hit_keys = [k for k in all_results.keys() if k.startswith("code_hit@")]
        def _code_hit_sort_key(key):
            # code_hit@{k}/{c}
            body = key.split("@", 1)[1]
            k_str, c_str = body.split("/", 1)
            return (int(k_str), int(c_str))
        for key in sorted(code_hit_keys, key=_code_hit_sort_key):
            output_results[key] = torch.cat(all_results[key]).mean().item()
        if self.accelerator.is_main_process:
            self.evaluator.pretty_print_legacy(output_results, logger=self.logger)
            self.evaluator.pretty_print_code_hit(output_results, logger=self.logger)
        return output_results

    def case_evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        diff2gap = defaultdict(list)

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model.forward(batch, return_loss=False)
                states = outputs.final_states.gather(
                    dim=1,
                    index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.model.n_pred_head, self.model.config['n_embd'])
                )
                states = F.normalize(states, dim=-1)

                token_emb = self.model.gpt2.wte.weight[1:-1]
                token_emb = F.normalize(token_emb, dim=-1)
                token_embs = torch.chunk(token_emb, self.model.n_pred_head, dim=0)
                logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.model.temperature for i in range(self.model.n_pred_head)]
                logits = [F.log_softmax(logit, dim=-1) for logit in logits]
                token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

                sampled_items = torch.randint(1, self.model.item_id2tokens.shape[0], (token_logits.shape[0], 10))

                item_logits = torch.gather(
                    input=token_logits.unsqueeze(-2).expand(-1, sampled_items.shape[1], -1),              # (batch_size, n_items, n_tokens)
                    dim=-1,
                    index=(self.model.item_id2tokens[sampled_items,:] - 1)  # (batch_size, n_items, code_dim)
                ).mean(dim=-1)

                for batch_id in range(item_logits.shape[0]):
                    logit_list = item_logits[batch_id].cpu().tolist()
                    for i in range(len(logit_list)):
                        for j in range(i + 1, len(logit_list)):
                            item_a = sampled_items[batch_id, i]
                            item_b = sampled_items[batch_id, j]
                            gap = abs(logit_list[i] - logit_list[j])
                            diff = (self.model.item_id2tokens[item_a] != self.model.item_id2tokens[item_b]).sum().item()
                            diff2gap[diff].append(gap)
        return diff2gap

    def evaluate_cold_start(self, dataloader, token2item, item2group, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        group2results = {
            '0': defaultdict(list),
            '1': defaultdict(list),
            '2': defaultdict(list),
            '3': defaultdict(list),
            '4': defaultdict(list)
        }
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for i, label in enumerate(batch['labels'].cpu().tolist()):
                    if self.config['model'] == 'TIGER':
                        item_id = token2item[' '.join(list(map(str, tuple(label[:-1]))))]
                    else:
                        item_id = token2item[str(label[0])]
                    if item_id not in item2group:
                        continue
                    group = item2group[item_id]
                    for key, value in results.items():
                        group2results[group][key].append(value.cpu().tolist()[i])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results, group2results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
