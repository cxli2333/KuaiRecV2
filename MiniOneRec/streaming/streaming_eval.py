import os
import sys
import json
import random
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)
from accelerate import Accelerator
import fire

 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import EvalSidDataset


def set_seed(seed):
     
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_batch(batch_encodings, batch_raws, model, tokenizer, gen_config, accelerator):
 
    input_ids_list = []
    attention_mask_list = []
    for enc in batch_encodings:
        if isinstance(enc['input_ids'], torch.Tensor):
            input_ids_list.append(enc['input_ids'])
        elif isinstance(enc['input_ids'], list):
            input_ids_list.append(torch.tensor(enc['input_ids'], dtype=torch.long))
        else:
            input_ids_list.append(torch.tensor(enc['input_ids'], dtype=torch.long))
        
        if isinstance(enc['attention_mask'], torch.Tensor):
            attention_mask_list.append(enc['attention_mask'])
        elif isinstance(enc['attention_mask'], list):
            attention_mask_list.append(torch.tensor(enc['attention_mask'], dtype=torch.long))
        else:
            attention_mask_list.append(torch.tensor(enc['attention_mask'], dtype=torch.long))
    
     
    max_len = max(t.shape[0] for t in input_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
     
    
   
    
    # # 
    # input_tensor = torch.stack(padded_input_ids).to(model.device)
    # mask_tensor = torch.stack(padded_attention_mask).to(model.device)
    padded_input_ids = []
    padded_attention_mask = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        pad_length = max_len - ids.shape[0]

        if pad_length > 0:
            id_padding = torch.full(
                (pad_length,),
                pad_token_id,
                dtype=ids.dtype,
                device=ids.device,
            )
            mask_padding = torch.zeros(
                pad_length,
                dtype=mask.dtype,
                device=mask.device,
            )

            padded_ids = torch.cat([id_padding, ids], dim=0)
            padded_mask = torch.cat([mask_padding, mask], dim=0)
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_mask.append(padded_mask)

    input_tensor = torch.stack(padded_input_ids).to(model.device)
    mask_tensor = torch.stack(padded_attention_mask).to(model.device)
    
    
    with torch.no_grad():
         
        actual_model = model.module if hasattr(model, 'module') else model
        out = actual_model.generate(
            input_tensor, 
            attention_mask=mask_tensor, 
            generation_config=gen_config
        )
    
     
    num_beams = gen_config.num_return_sequences
    batch_size_actual = len(batch_encodings)
    results = []
    for i in range(batch_size_actual):
         
        beam_start = i * num_beams
        beam_end = beam_start + num_beams
        
        predict_list = []
        for beam_idx in range(beam_start, min(beam_end, len(out.sequences))):
            seq = out.sequences[beam_idx]
            
             
            if seq.shape[0] > max_len:
                generated = seq[max_len:]
            else:
                 
                original_input_len = input_ids_list[i].shape[0]
                generated = seq[original_input_len:]
            
            generated_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
            generated_text = generated_text.replace('<|endoftext|>', '').strip()
            predict_list.append(generated_text)

             
        result = dict(batch_raws[i])
        result["predict"] = predict_list  
        results.append(result)

     
    
    return results



def main(
    base_model: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    info_file: str = "",
    category: str = "",
    seed: int = 42,
    K: int = 50,
    batch_size: int = 8,
    length_penalty: float = 0.0,
    max_new_tokens: int = 32, 
    num_beams: int = 50,
    negative_sample_rate: float = 0.0,
    debug: bool = False,
    min_diversity: int = 5
): 
     
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(seed)
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"[INFO] Running evaluation WITHOUT Trie constraints")
        print(f"[INFO] Model: {os.path.basename(base_model)}")
        print(f"[INFO] Test data: {test_data_path}")
        print(f"[INFO] Num beams: {num_beams}")
        print(f"[INFO] Max new tokens: {max_new_tokens}")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
 
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
     
    if accelerator.is_main_process:
        print(f"[INFO] Loading model from {base_model}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None
    )
    model = accelerator.prepare(model)
    model.eval()
    
    if accelerator.is_main_process:
        print(f"[INFO] Model loaded successfully")
    
     
    if accelerator.is_main_process:
        print(f"[INFO] Loading dataset from {test_data_path}...")
    
    val_dataset = EvalSidDataset(
        train_file=test_data_path, 
        tokenizer=tokenizer, 
        max_len=4096, 
        category=category, 
        test=True, 
        K=K, 
        seed=seed
    )
    
    if accelerator.is_main_process:
        print(f"[INFO] Dataset loaded: {len(val_dataset)} samples")
    
 
    full_data_list = val_dataset.get_all()
    my_indices = list(range(accelerator.process_index, len(val_dataset), accelerator.num_processes))
    my_encodings = [val_dataset[i] for i in my_indices]
    my_raw_data = [full_data_list[i] for i in my_indices]
    
    print(f"[Process {accelerator.process_index}] Handling {len(my_encodings)} samples")
    
 
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        length_penalty=length_penalty,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        return_dict_in_generate=True,  
        output_scores=False,
    )
        
 
    final_results = []
    num_batches = (len(my_encodings) + batch_size - 1) // batch_size
    
    print(f"[Process {accelerator.process_index}] Starting inference ({num_batches} batches)...")
    
    for batch_idx in range(0, len(my_encodings), batch_size):
        batch_enc = my_encodings[batch_idx:batch_idx+batch_size]
        batch_raw = my_raw_data[batch_idx:batch_idx+batch_size]
        
        batch_results = evaluate_batch(batch_enc, batch_raw, model, tokenizer, gen_config, accelerator)
        final_results.extend(batch_results)
        
        if (batch_idx // batch_size) % 10 == 0:
            print(f"[Process {accelerator.process_index}] Batch {batch_idx//batch_size + 1}/{num_batches} completed")
    
    print(f"[Process {accelerator.process_index}] Inference completed, {len(final_results)} results")
    
   
    part_path = f"{os.path.splitext(result_json_data)[0]}_part_{accelerator.process_index}.json"
    dir_path = os.path.dirname(part_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(part_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"[Process {accelerator.process_index}] Saved to {part_path}")
    
    
    accelerator.wait_for_everyone()
    
     
    if accelerator.is_main_process:
        print(f"\n[Main Process] Merging results from all processes...")
        all_results = []
        
        for rank in range(accelerator.num_processes):
            part_file = f"{os.path.splitext(result_json_data)[0]}_part_{rank}.json"
            if os.path.exists(part_file):
                with open(part_file, 'r', encoding='utf-8') as f:
                    part_data = json.load(f)
                    all_results.extend(part_data)
                print(f"  Loaded {len(part_data)} results from rank {rank}")
        
        
        with open(result_json_data, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Saved {len(all_results)} total results to {result_json_data}")
        
        
        for rank in range(accelerator.num_processes):
            part_file = f"{os.path.splitext(result_json_data)[0]}_part_{rank}.json"
            if os.path.exists(part_file):
                os.remove(part_file)
        
        print(f"[INFO] Cleaned up temporary part files")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"[DONE] Evaluation completed successfully")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    fire.Fire(main)
