"""Supervised fine-tuning with online EWC in LoRA weight-update space.

This adapts the EWC-LoRA formulation from yaoyz96/low-rank-cl to a
Transformers causal/seq2seq language model.  EWC is applied to the induced
dense update ``delta_W = scaling * B @ A``, not independently to the two
low-rank factors.  After each streaming task the empirical Fisher is folded
into an online importance matrix and the LoRA update is merged into the base
model, leaving a regular Hugging Face checkpoint for inference.
"""

import inspect
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from data import (
    StreamingSid2SidDataset,
    StreamingSid2TitleDataset,
    StreamingTitle2SidDataset,
    ValidationDataset,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
EWC_STATE_VERSION = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenExtender:
    def __init__(self, index_path: str):
        self.index_path = index_path

    def get_new_tokens(self) -> List[str]:
        with open(self.index_path, "r", encoding="utf-8") as handle:
            indices = json.load(handle)
        return sorted({token for index in indices.values() for token in index})


class EWCLoRALinear(nn.Module):
    """Frozen Linear layer plus a trainable LoRA update with explicit delta_W."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base_layer.weight.device
        # Keeping adapters in fp32 matches common PEFT behavior and prevents
        # underflow while the frozen model itself remains bf16.
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.capture_fisher = False
        self._last_delta_weight: Optional[torch.Tensor] = None

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        lora_inputs = self.lora_dropout(inputs).to(self.lora_A.dtype)

        if self.capture_fisher:
            # Fisher needs the gradient with respect to the induced dense
            # update. This branch is enabled only during the Fisher pass.
            delta_weight = self.delta_weight()
            if delta_weight.requires_grad:
                delta_weight.retain_grad()
                self._last_delta_weight = delta_weight
            lora_output = F.linear(lora_inputs, delta_weight)
        else:
            # During normal training, preserve the low-rank computation:
            # x @ A.T @ B.T. This is algebraically equivalent to using B @ A
            # but avoids an additional dense projection for every token.
            low_rank_hidden = F.linear(lora_inputs, self.lora_A)
            lora_output = F.linear(low_rank_hidden, self.lora_B) * self.scaling

        return base_output + lora_output.to(base_output.dtype)


class SemanticTokenEmbedding(nn.Module):
    """Train a compact delta only for semantic-token rows of an embedding."""

    def __init__(
        self,
        base_layer: nn.Embedding,
        token_ids: List[int],
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.delta = delta
        mapping = torch.full(
            (base_layer.num_embeddings,),
            -1,
            dtype=torch.long,
            device=base_layer.weight.device,
        )
        mapping[torch.tensor(token_ids, dtype=torch.long, device=mapping.device)] = (
            torch.arange(len(token_ids), dtype=torch.long, device=mapping.device)
        )
        self.register_buffer("token_id_to_row", mapping, persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        # Some Transformers helpers inspect this attribute. Training forwards
        # use the compact delta below so the frozen full matrix is not copied.
        return self.base_layer.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(input_ids)
        row_ids = self.token_id_to_row[input_ids]
        mask = row_ids.ge(0)
        delta_output = F.embedding(row_ids.clamp_min(0), self.delta)
        delta_output = delta_output * mask.unsqueeze(-1)
        return output + delta_output.to(output.dtype)


class SemanticTokenOutput(nn.Module):
    """Apply the same compact semantic-token delta to an LM output head."""

    def __init__(
        self,
        base_layer: nn.Linear,
        token_ids: List[int],
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        if base_layer.out_features <= max(token_ids):
            raise ValueError("Semantic token ID exceeds the output vocabulary size")
        self.base_layer = base_layer
        self.delta = delta
        self.register_buffer(
            "token_ids",
            torch.tensor(token_ids, dtype=torch.long, device=base_layer.weight.device),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base_layer(hidden_states)
        semantic_logits = F.linear(hidden_states.to(self.delta.dtype), self.delta)
        return torch.index_add(
            logits,
            dim=-1,
            index=self.token_ids,
            source=semantic_logits.to(logits.dtype),
        )


def inject_semantic_token_adapter(
    model: nn.Module,
    token_ids: List[int],
) -> Optional[nn.Parameter]:
    if not token_ids:
        return None
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not isinstance(input_layer, nn.Embedding) or not isinstance(output_layer, nn.Linear):
        print(
            "Warning: semantic-token adapter requires nn.Embedding input and "
            "nn.Linear output layers; leaving both frozen"
        )
        return None
    if input_layer.embedding_dim != output_layer.in_features:
        print("Warning: input/output embedding dimensions differ; adapter disabled")
        return None

    delta = nn.Parameter(
        torch.zeros(
            len(token_ids),
            input_layer.embedding_dim,
            dtype=torch.float32,
            device=input_layer.weight.device,
        )
    )
    model.set_input_embeddings(SemanticTokenEmbedding(input_layer, token_ids, delta))
    model.set_output_embeddings(SemanticTokenOutput(output_layer, token_ids, delta))
    return delta


def merge_and_unload_semantic_token_adapter(model: nn.Module) -> int:
    input_wrapper = model.get_input_embeddings()
    output_wrapper = model.get_output_embeddings()
    if not isinstance(input_wrapper, SemanticTokenEmbedding):
        return 0
    if not isinstance(output_wrapper, SemanticTokenOutput):
        raise RuntimeError("Semantic input adapter exists without its output adapter")

    token_ids = output_wrapper.token_ids
    delta = input_wrapper.delta.detach()
    with torch.no_grad():
        input_wrapper.base_layer.weight.index_add_(
            0,
            token_ids,
            delta.to(
                device=input_wrapper.base_layer.weight.device,
                dtype=input_wrapper.base_layer.weight.dtype,
            ),
        )
        input_weight = input_wrapper.base_layer.weight
        output_weight = output_wrapper.base_layer.weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            output_weight.index_add_(
                0,
                token_ids,
                delta.to(device=output_weight.device, dtype=output_weight.dtype),
            )

    model.set_input_embeddings(input_wrapper.base_layer)
    model.set_output_embeddings(output_wrapper.base_layer)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    return token_ids.numel()


def _module_matches(name: str, targets: Tuple[str, ...]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def _normalize_name_list(value, argument_name: str) -> Tuple[str, ...]:
    """Normalize Fire's string/tuple parsing for comma-separated arguments."""
    if isinstance(value, str):
        raw_values = (value,)
    elif isinstance(value, (tuple, list, set)):
        raw_values = value
    else:
        raise TypeError(
            f"--{argument_name} must be a comma-separated string or a list, "
            f"got {type(value).__name__}"
        )

    normalized = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise TypeError(
                f"Every --{argument_name} value must be a string, "
                f"got {type(raw_value).__name__}"
            )
        normalized.extend(part.strip() for part in raw_value.split(",") if part.strip())
    return tuple(dict.fromkeys(normalized))


def _parent_and_child(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_lora_layers(model: nn.Module) -> Iterable[Tuple[str, EWCLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, EWCLoRALinear):
            yield name, module


def inject_ewc_lora(
    model: nn.Module,
    target_modules: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    targets = _normalize_name_list(target_modules, "target_modules")
    if not targets:
        raise ValueError("--target_modules must contain at least one module name")

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _module_matches(name, targets):
            replacements.append((name, module))

    if not replacements:
        raise ValueError(
            f"No nn.Linear modules matched {targets}. "
            "Inspect model.named_modules() and adjust --target_modules."
        )

    for name, base_layer in replacements:
        parent, child_name = _parent_and_child(model, name)
        setattr(parent, child_name, EWCLoRALinear(base_layer, rank, alpha, dropout))

    return [name for name, _ in replacements]


def merge_and_unload_lora(model: nn.Module) -> List[str]:
    """Merge every LoRA update into its frozen base layer and remove wrappers."""
    layers = list(iter_lora_layers(model))
    with torch.no_grad():
        for name, layer in layers:
            merged = layer.delta_weight().to(
                device=layer.base_layer.weight.device,
                dtype=layer.base_layer.weight.dtype,
            )
            layer.base_layer.weight.add_(merged)
            parent, child_name = _parent_and_child(model, name)
            setattr(parent, child_name, layer.base_layer)
    return [name for name, _ in layers]


def _safe_torch_load(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_previous_fisher(
    state_path: Optional[str], model: nn.Module
) -> Tuple[Dict[str, torch.Tensor], int]:
    if not state_path:
        return {}, 0
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"EWC state does not exist: {state_path}")

    state = _safe_torch_load(state_path)
    if state.get("version") != EWC_STATE_VERSION:
        raise ValueError(
            f"Unsupported EWC state version {state.get('version')}; "
            f"expected {EWC_STATE_VERSION}"
        )

    expected_shapes = {
        name: tuple(layer.delta_weight().shape) for name, layer in iter_lora_layers(model)
    }
    fisher = {}
    ignored = []
    for name, tensor in state.get("fisher", {}).items():
        if name not in expected_shapes or tuple(tensor.shape) != expected_shapes[name]:
            ignored.append(name)
            continue
        fisher[name] = tensor.detach().float().cpu()

    missing = sorted(set(expected_shapes) - set(fisher))
    if ignored:
        print(f"Warning: ignored {len(ignored)} incompatible Fisher tensors")
    if missing:
        print(f"Warning: previous Fisher is missing {len(missing)} current LoRA layers")
    print(
        f"Loaded EWC state from {state_path}: "
        f"tasks={state.get('tasks_seen', 0)}, tensors={len(fisher)}"
    )
    return fisher, int(state.get("tasks_seen", 0))


def ewc_penalty(
    model: nn.Module,
    previous_fisher: Dict[str, torch.Tensor],
) -> torch.Tensor:
    penalty = None
    for name, layer in iter_lora_layers(model):
        omega = previous_fisher.get(name)
        if omega is None:
            continue
        delta_weight = layer.delta_weight()
        term = torch.sum(
            omega.to(device=delta_weight.device, dtype=delta_weight.dtype)
            * delta_weight.square()
        )
        penalty = term if penalty is None else penalty + term

    if penalty is not None:
        return penalty
    parameter = next(model.parameters())
    return torch.zeros((), device=parameter.device, dtype=torch.float32)


class EWCLoRATrainer(transformers.Trainer):
    def __init__(self, *args, previous_fisher=None, ewc_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        previous_fisher = previous_fisher or {}
        lora_layers = dict(iter_lora_layers(self.model))
        # Trainer has placed the model on its training device by this point.
        # Cache omega once instead of copying every dense matrix every batch.
        self.previous_fisher = {
            name: omega.to(
                device=lora_layers[name].lora_A.device,
                dtype=torch.float32,
            )
            for name, omega in previous_fisher.items()
            if name in lora_layers
        }
        self.ewc_lambda = float(ewc_lambda)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if model.training and self.previous_fisher and self.ewc_lambda != 0:
            regularizer = ewc_penalty(self.model, self.previous_fisher)
            loss = loss + (self.ewc_lambda / 2.0) * regularizer

        return (loss, outputs) if return_outputs else loss


def _batch_size(batch: Dict[str, torch.Tensor]) -> int:
    for key in ("input_ids", "labels", "attention_mask"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value.shape[0]
    raise ValueError("Unable to infer Fisher batch size")


def _build_fisher_dataloader(
    trainer: transformers.Trainer,
    batch_size: int,
) -> DataLoader:
    """Build an unprepared per-rank loader with a Fisher-specific batch size."""
    if batch_size <= 0:
        raise ValueError(f"fisher_batch_size must be positive, got {batch_size}")

    dataset = trainer.train_dataset
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=int(trainer.args.seed),
            drop_last=False,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(int(trainer.args.seed))
        sampler = RandomSampler(dataset, generator=generator)

    num_workers = int(trainer.args.dataloader_num_workers)
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "collate_fn": trainer.data_collator,
        "num_workers": num_workers,
        "pin_memory": bool(trainer.args.dataloader_pin_memory),
        "drop_last": False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            getattr(trainer.args, "dataloader_persistent_workers", False)
        )
    return DataLoader(**dataloader_kwargs)


def _release_optimizer_memory(trainer: transformers.Trainer) -> None:
    """Release state that is no longer needed after the training epoch."""
    trainer.model.zero_grad(set_to_none=True)
    if trainer.optimizer is not None:
        trainer.optimizer.zero_grad(set_to_none=True)
        optimizer_state = getattr(trainer.optimizer, "state", None)
        if optimizer_state is None and hasattr(trainer.optimizer, "optimizer"):
            optimizer_state = getattr(trainer.optimizer.optimizer, "state", None)
        if optimizer_state is not None:
            optimizer_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_empirical_fisher(
    trainer: transformers.Trainer,
    max_batches: int,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal empirical Fisher with respect to each delta_W."""
    raw_model = trainer.model
    wrapped_model = trainer.model_wrapped
    layers = dict(iter_lora_layers(raw_model))
    if not layers:
        raise RuntimeError("No EWC-LoRA layers found while computing Fisher")

    fisher_sums: Dict[str, Optional[torch.Tensor]] = {name: None for name in layers}
    sample_count = 0
    was_training = wrapped_model.training
    wrapped_model.eval()
    for layer in layers.values():
        layer.capture_fisher = True

    dataloader = _build_fisher_dataloader(trainer, batch_size)
    print(
        f"Computing empirical Fisher over {len(dataloader)} local batches "
        f"(per-device batch size={batch_size})..."
    )
    for batch_index, batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        prepared = trainer._prepare_inputs(batch)
        wrapped_model.zero_grad(set_to_none=True)
        sync_context = (
            wrapped_model.no_sync() if hasattr(wrapped_model, "no_sync") else nullcontext()
        )
        with sync_context:
            with trainer.compute_loss_context_manager():
                outputs = wrapped_model(**prepared)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()

        current_batch_size = _batch_size(prepared)
        sample_count += current_batch_size
        for name, layer in layers.items():
            delta_weight = layer._last_delta_weight
            grad = None if delta_weight is None else delta_weight.grad
            if grad is None:
                raise RuntimeError(f"Fisher gradient was not captured for {name}")
            contribution = grad.detach().float().square() * current_batch_size
            if fisher_sums[name] is None:
                fisher_sums[name] = contribution
            else:
                fisher_sums[name].add_(contribution)
            layer._last_delta_weight = None

        if (batch_index + 1) % 50 == 0:
            print(f"  Fisher batches processed: {batch_index + 1}")

    for layer in layers.values():
        layer.capture_fisher = False
        layer._last_delta_weight = None

    if sample_count == 0:
        raise RuntimeError("Fisher dataloader produced no samples")

    count_device = next(value.device for value in fisher_sums.values() if value is not None)
    count_tensor = torch.tensor(float(sample_count), device=count_device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        for value in fisher_sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    result = {
        name: (value / count_tensor).detach().cpu()
        for name, value in fisher_sums.items()
    }
    wrapped_model.zero_grad(set_to_none=True)
    if was_training:
        wrapped_model.train()
    print(f"Empirical Fisher complete: global samples={int(count_tensor.item())}")
    return result


def update_online_fisher(
    previous: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
    gamma: float,
) -> Dict[str, torch.Tensor]:
    updated = {}
    for name, current_value in current.items():
        previous_value = previous.get(name)
        if previous_value is None:
            updated[name] = current_value.float().cpu()
        else:
            updated[name] = (
                gamma * previous_value.float().cpu() + current_value.float().cpu()
            )
    return updated


def _training_arguments(**kwargs):
    """Handle the evaluation_strategy -> eval_strategy Transformers rename."""
    parameters = inspect.signature(transformers.TrainingArguments.__init__).parameters
    if "eval_strategy" not in parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "gradient_checkpointing_kwargs" not in parameters:
        kwargs.pop("gradient_checkpointing_kwargs", None)
    return transformers.TrainingArguments(**kwargs)


def _build_datasets(
    tasks: str,
    train_file: str,
    eval_file: str,
    tokenizer,
    cutoff_len: int,
    sample: int,
    seed: int,
    category: str,
):
    task_factories = {
        "streaming_sid2sid": StreamingSid2SidDataset,
        "streaming_title2sid": StreamingTitle2SidDataset,
        "streaming_sid2title": StreamingSid2TitleDataset,
    }
    train_datasets = []
    for task in _normalize_name_list(tasks, "tasks"):
        if task not in task_factories:
            raise ValueError(f"Unsupported streaming task: {task}")
        print(f"[{time.strftime('%H:%M:%S')}] Loading {task}...")
        train_datasets.append(
            task_factories[task](
                train_file=train_file,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category,
            )
        )

    if not train_datasets:
        raise ValueError("No streaming training tasks were selected")

    train_data = ConcatDataset(train_datasets)
    val_data = ValidationDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    return train_data, val_data


def train(
    base_model: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    sid_index_path: str = "",
    item_meta_path: str = "",
    category: str = "Streaming",
    tasks: str = "streaming_sid2sid",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 1024,
    micro_batch_size: int = 128,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    eval_strategy: str = "no",
    save_strategy: str = "no",
    save_total_limit: int = 1,
    report_to: str = "none",
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: Optional[str] = None,
    target_modules: str = "q_proj,v_proj",
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    train_semantic_embeddings: bool = True,
    ewc_lambda: float = 1e7,
    ewc_gamma: float = 1.0,
    ewc_state_path: Optional[str] = None,
    fisher_max_batches: int = -1,
    fisher_batch_size: int = 8,
    dataloader_num_workers: int = 8,
):
    del item_meta_path  # Retained for command compatibility with the original sft.py.
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if not base_model:
        raise ValueError("--base_model is required")
    if batch_size < micro_batch_size:
        raise ValueError("batch_size must be >= micro_batch_size")
    if not 0.0 <= ewc_gamma <= 1.0:
        raise ValueError("ewc_gamma must be in [0, 1]")
    if fisher_batch_size <= 0:
        raise ValueError("fisher_batch_size must be positive")

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name

    config = AutoConfig.from_pretrained(base_model)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_class.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if not config.is_encoder_decoder:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    semantic_tokens: List[str] = []
    if sid_index_path and os.path.isfile(sid_index_path):
        semantic_tokens = TokenExtender(sid_index_path).get_new_tokens()
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": semantic_tokens}
        )
        if num_added:
            model.resize_token_embeddings(len(tokenizer))
        print(
            f"Tokenizer semantic IDs: requested={len(semantic_tokens)}, "
            f"added={num_added}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    injected_names = inject_ewc_lora(
        model=model,
        target_modules=target_modules,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    semantic_token_ids = []
    if train_semantic_embeddings and semantic_tokens:
        converted_ids = tokenizer.convert_tokens_to_ids(semantic_tokens)
        semantic_token_ids = sorted(
            {
                int(token_id)
                for token_id in converted_ids
                if token_id is not None and token_id != tokenizer.unk_token_id
            }
        )
        semantic_delta = inject_semantic_token_adapter(model, semantic_token_ids)
        if semantic_delta is not None:
            print(
                f"Enabled compact semantic-token adapter for "
                f"{len(semantic_token_ids)} vocabulary rows"
            )

    # Re-entrant gradient checkpointing needs at least one input requiring a
    # gradient when the base model and its embeddings are frozen.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Injected EWC-LoRA into {len(injected_names)} layers; "
        f"trainable={trainable:,}/{total:,} ({100.0 * trainable / total:.4f}%)"
    )

    previous_fisher, tasks_seen = load_previous_fisher(ewc_state_path, model)
    category_names = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "Streaming": "short video",
    }
    if category not in category_names:
        raise ValueError(f"Unknown category: {category}")

    train_data, val_data = _build_datasets(
        tasks=tasks,
        train_file=train_file,
        eval_file=eval_file,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category_names[category],
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if world_size > 1:
        gradient_accumulation_steps //= world_size
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    training_args = _training_arguments(
        run_name=wandb_run_name or None,
        output_dir=output_dir,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        optim="adamw_torch",
        gradient_checkpointing=True,
        # Reentrant checkpointing conflicts with DDP when LoRA parameters also
        # participate in the EWC penalty outside the checkpointed forward.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        group_by_length=group_by_length,
        report_to=report_to,
        remove_unused_columns=False,
    )
    trainer = EWCLoRATrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        previous_fisher=previous_fisher,
        ewc_lambda=ewc_lambda,
    )

    model.config.use_cache = False
    print(f"[{time.strftime('%H:%M:%S')}] Starting EWC-LoRA training")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"[{time.strftime('%H:%M:%S')}] Training finished; updating Fisher")

    _release_optimizer_memory(trainer)
    current_fisher = compute_empirical_fisher(
        trainer,
        max_batches=fisher_max_batches,
        batch_size=fisher_batch_size,
    )
    online_fisher = update_online_fisher(previous_fisher, current_fisher, ewc_gamma)

    merged_names = merge_and_unload_lora(model)
    merged_token_rows = merge_and_unload_semantic_token_adapter(model)
    model.config.use_cache = True
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        state = {
            "version": EWC_STATE_VERSION,
            "tasks_seen": tasks_seen + 1,
            "gamma": float(ewc_gamma),
            "fisher": online_fisher,
            "target_modules": list(
                _normalize_name_list(target_modules, "target_modules")
            ),
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "semantic_token_rows": int(merged_token_rows),
            "fisher_batch_size": int(fisher_batch_size),
            "fisher_max_batches": int(fisher_max_batches),
            "source_model": base_model,
        }
        state_path = os.path.join(output_dir, "ewc_state.pt")
        torch.save(state, state_path)
        print(
            f"Saved merged model and EWC state to {output_dir}; "
            f"merged_layers={len(merged_names)}, "
            f"semantic_rows={merged_token_rows}, tasks_seen={tasks_seen + 1}"
        )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(train)
