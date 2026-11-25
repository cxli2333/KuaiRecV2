"""Distributed generation evaluator with robust variable-length left padding."""

import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import fire
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import EvalSidDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_1d_long(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(dtype=torch.long, device="cpu")
    else:
        tensor = torch.as_tensor(value, dtype=torch.long)
    return tensor.reshape(-1)


def left_pad_batch(
    batch_encodings: Sequence[Dict],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_rows = [as_1d_long(encoding["input_ids"]) for encoding in batch_encodings]
    mask_rows = [
        as_1d_long(
            encoding.get("attention_mask", torch.ones_like(input_rows[index]))
        )
        for index, encoding in enumerate(batch_encodings)
    ]
    if not input_rows:
        raise ValueError("Cannot pad an empty evaluation batch")
    for index, (input_ids, attention_mask) in enumerate(zip(input_rows, mask_rows)):
        if len(input_ids) != len(attention_mask):
            raise ValueError(
                f"input_ids/attention_mask length mismatch at row {index}: "
                f"{len(input_ids)} != {len(attention_mask)}"
            )

    max_length = max(len(row) for row in input_rows)
    padded_ids: List[torch.Tensor] = []
    padded_masks: List[torch.Tensor] = []
    for input_ids, attention_mask in zip(input_rows, mask_rows):
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_padding = torch.full(
                (padding_length,), pad_token_id, dtype=torch.long
            )
            mask_padding = torch.zeros(padding_length, dtype=torch.long)
            input_ids = torch.cat((input_padding, input_ids), dim=0)
            attention_mask = torch.cat((mask_padding, attention_mask), dim=0)
        padded_ids.append(input_ids)
        padded_masks.append(attention_mask)
    return (
        torch.stack(padded_ids, dim=0).to(device, non_blocking=True),
        torch.stack(padded_masks, dim=0).to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate_batch(
    batch_encodings: Sequence[Dict],
    batch_raws: Sequence[Dict],
    model,
    tokenizer,
    generation_config: GenerationConfig,
    accelerator: Accelerator,
) -> List[Dict]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_ids, attention_mask = left_pad_batch(
        batch_encodings, int(pad_token_id), accelerator.device
    )
    actual_model = accelerator.unwrap_model(model)
    output = actual_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    return_count = int(generation_config.num_return_sequences)
    prompt_width = input_ids.shape[1]
    results = []
    for row_index, raw in enumerate(batch_raws):
        start = row_index * return_count
        end = min(start + return_count, len(sequences))
        predictions = []
        for sequence in sequences[start:end]:
            generated = sequence[prompt_width:]
            text = tokenizer.decode(
                generated, skip_special_tokens=False
            ).strip()
            if tokenizer.eos_token:
                text = text.replace(tokenizer.eos_token, "").strip()
            predictions.append(text)
        result = dict(raw)
        result["predict"] = predictions
        results.append(result)
    return results


def main(
    base_model: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    info_file: str = "",
    category: str = "Streaming",
    seed: int = 42,
    K: int = 50,
    batch_size: int = 4,
    length_penalty: float = 0.0,
    max_new_tokens: int = 32,
    num_beams: int = 50,
    negative_sample_rate: float = 1.0,
    debug: bool = False,
    min_diversity: int = 5,
) -> None:
    del info_file, negative_sample_rate, debug, min_diversity
    if not base_model or not test_data_path or not result_json_data:
        raise ValueError("base_model, test_data_path, and result_json_data are required")
    if batch_size < 1 or num_beams < 1 or K < 1:
        raise ValueError("batch_size, num_beams, and K must be positive")

    accelerator = Accelerator()
    set_seed(seed)
    if accelerator.is_main_process:
        print(
            f"[Eval] model={base_model}, test={test_data_path}, "
            f"beams={num_beams}, batch_size={batch_size}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    model = accelerator.prepare(model)
    model.eval()

    dataset = EvalSidDataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=4096,
        category=category,
        test=True,
        K=K,
        seed=seed,
    )
    raw_rows = dataset.get_all()
    indices = list(
        range(accelerator.process_index, len(dataset), accelerator.num_processes)
    )
    encodings = [dataset[index] for index in indices]
    rank_raw_rows = [raw_rows[index] for index in indices]
    total_batches = (len(encodings) + batch_size - 1) // batch_size
    print(
        f"[Rank {accelerator.process_index}] samples={len(encodings):,}, "
        f"batches={total_batches:,}",
        flush=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        length_penalty=length_penalty,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=(
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        ),
        return_dict_in_generate=True,
        output_scores=False,
    )
    rank_results = []
    for batch_number, start in enumerate(
        range(0, len(encodings), batch_size), start=1
    ):
        rank_results.extend(
            evaluate_batch(
                encodings[start : start + batch_size],
                rank_raw_rows[start : start + batch_size],
                model,
                tokenizer,
                generation_config,
                accelerator,
            )
        )
        if batch_number % 10 == 0 or batch_number == total_batches:
            print(
                f"[Rank {accelerator.process_index}] "
                f"batch={batch_number:,}/{total_batches:,}",
                flush=True,
            )

    result_stem = os.path.splitext(result_json_data)[0]
    part_path = f"{result_stem}_part_{accelerator.process_index}.json"
    os.makedirs(os.path.dirname(os.path.abspath(part_path)), exist_ok=True)
    with open(part_path, "w", encoding="utf-8") as handle:
        json.dump(rank_results, handle, ensure_ascii=False, indent=2)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        indexed_results = []
        for rank in range(accelerator.num_processes):
            rank_path = f"{result_stem}_part_{rank}.json"
            with open(rank_path, "r", encoding="utf-8") as handle:
                values = json.load(handle)
            for local_index, value in enumerate(values):
                original_index = rank + local_index * accelerator.num_processes
                indexed_results.append((original_index, value))
        indexed_results.sort(key=lambda value: value[0])
        merged = [value for _, value in indexed_results]
        with open(result_json_data, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
        for rank in range(accelerator.num_processes):
            os.remove(f"{result_stem}_part_{rank}.json")
        print(
            f"[Eval] saved {len(merged):,} predictions to {result_json_data}",
            flush=True,
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
