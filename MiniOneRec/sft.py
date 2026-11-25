import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig,AutoModelForSeq2SeqLM, TrainerCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset, BaseStreamingDataset, StreamingSid2SidDataset, StreamingTitle2SidDataset, StreamingSid2TitleDataset, ValidationDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset,Dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings("ignore", category=UserWarning)

class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 1024,
    #1024,128
    micro_batch_size: int = 128,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    cutoff_len: int =512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    tasks: str = "streaming_sid2sid",
    eval_strategy: str = 'no',
    save_strategy: str = "epoch",
    save_total_limit: int = 1,
    report_to: str = "none",
):
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name

    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books","Streaming":"short video"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if "t5" in base_model.lower():
        model_class = AutoModelForSeq2SeqLM
        print(f"Detected T5 model, using {model_class.__name__}")
    else:
        model_class = AutoModelForCausalLM
        print(f"Detected Causal model, using {model_class.__name__}")

    if not train_from_scratch:
        model = model_class.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = model_class.from_config(config)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True, 
        fix_mistral_regex=True
        )

    

    if "t5" not in base_model.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        print(f"T5 mode: Keeping original pad_token_id={tokenizer.pad_token_id}")
        
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            
            
            num_added = tokenizer.add_special_tokens({
                'additional_special_tokens': new_tokens
            })
            print(f"Successfully added {num_added} tokens")
            
            # Validation
            # sample_tokens = new_tokens[:5]
            # all_correct = True
            # for token in sample_tokens:
            #     token_ids = tokenizer.encode(token, add_special_tokens=False)
            #     if len(token_ids) == 1:
            #     else:
            #         all_correct = False
            
            
            # Expand model embeddings
            model.resize_token_embeddings(len(tokenizer))
            
            # Save the tokenizer
            tokenizer.save_pretrained(output_dir)
            print(f"Tokenizer saved to {output_dir}")

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
#     train_datasets = []
#     # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
#     train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
#     train_datasets.append(train_data1)
# #     train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
# #     train_datasets.append(train_data2)
# #     train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
# #     train_datasets.append(train_data3)
# #     train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
# #     train_datasets.append(train_data4)
# #     train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
# #     train_datasets.append(train_data5)
#     train_data = ConcatDataset(train_datasets)
#     val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
#     # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    


    train_datasets = []
    
    task_list = tasks.split(',')
    print(f"[{time.strftime('%H:%M:%S')}] train.py: The following tasks: {task_list} are being loaded")
    
    if "streaming_sid2sid" in task_list:
        print(f"[{time.strftime('%H:%M:%S')}] train.py: Loading StreamingSid2SidDataset...")
        train_datasets.append(StreamingSid2SidDataset(
            train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, 
            sample=sample, seed=seed, category=category
        ))


        # # --- Print training details for the first three samples ---
        # ds = StreamingSid2SidDataset(
        #     train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, 
        #     sample=sample, seed=seed, category=category
        # )
        # for i in range(min(3, len(ds))):
        #     item = ds[i]
        #     input_ids = item['input_ids']
        #     labels = item['labels']
            
                
        #         # Print only non-padding content for readability
        #         if tid == tokenizer.pad_token_id:
        #             continue
                



    if "streaming_title2sid" in task_list:
        print(f"[{time.strftime('%H:%M:%S')}] train.py: Loading StreamingTitle2SidDataset...")
        train_datasets.append(StreamingTitle2SidDataset(
            train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, 
            sample=sample, seed=seed, category=category,
        ))

    if "streaming_sid2title" in task_list:
        print(f"[{time.strftime('%H:%M:%S')}] train.py: Loading StreamingSid2TitleDataset...")
        train_datasets.append(StreamingSid2TitleDataset(
            train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, 
            sample=sample, seed=seed, category=category
        ))
    
    if not train_datasets:
        raise ValueError("No valid tasks have been selected! Please check the --tasks parameter.")
    
    print(f"[{time.strftime('%H:%M:%S')}] train.py: Merging ConcatDataset...")
    train_data = ConcatDataset(train_datasets)

    print(f"[{time.strftime('%H:%M:%S')}] train.py: Loading Val Dataset...")
    val_data = ValidationDataset(
        train_file=eval_file, 
        tokenizer=tokenizer, 
        max_len=cutoff_len, 
        sample=sample, 
        seed=seed, 
        category=category
    )
    
    print(f"[{time.strftime('%H:%M:%S')}] LOAD DATA FINISHED")    
    
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    

    

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
            dataloader_num_workers=8,
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
            eval_strategy="no",
            save_strategy="epoch", 
            save_total_limit=save_total_limit,
            save_steps = 1.0,
            output_dir=output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb", 
        ),
        data_collator=data_collator,
        callbacks = [], 
    )
    model.config.use_cache = False
    
    print(f"[{time.strftime('%H:%M:%S')}] train.py: trainer.train() will start ")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"[{time.strftime('%H:%M:%S')}] train.py: training finished。")

    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    #trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    

    
#     trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#     trainer.save_model(output_dir)
    



if __name__ == "__main__":
    fire.Fire(train)
