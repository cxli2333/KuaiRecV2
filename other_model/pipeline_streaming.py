# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import os
import json
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from logging import getLogger
import logging
from collections import defaultdict
from tqdm import tqdm
from eval_metrics import (
    calculate_metrics_from_model_outputs,
    calculate_multi_target_metrics_from_model_outputs,
)
from collections import defaultdict, deque

BEHAVIOR_COLS = [
    "is_click",
    "is_long_view",
    "is_like",
    "is_follow",
    "is_forward",
    "is_comment",
    "is_hate",
    "is_search",
    "is_profile_enter",
]


EVAL_METRICS_AVAILABLE = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Run: pip install wandb")


try:
    from dataset.streaming_dataset import StreamingInteractionDataset
except ImportError:
    try:
        from streaming_dataset import StreamingInteractionDataset
    except ImportError:
        StreamingInteractionDataset = None

class StreamingPipeline:
    
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        category: str = None,
        json_config_path: str = None,
        checkpoint_path: str = None,
        tokenizer = None,
        config_dict: dict = None,
        task: str = 'is_click',
        epochs_per_round: int = 5,
        item_history_days: int = 7,
        max_seq_len: int = 999,
        min_seq_len: int = 5,
        max_history_len: int = 999,
        accumulate_history: bool = False,
        max_train_samples_per_user: int = 999,
        training_mode: str = 'continual',
        data_mode: str = 'streaming',
        trainer_class = None,
        
        use_wandb: bool = True,
        wandb_project: str = 'tiger-streaming',
        wandb_run_name: str = None,
        wandb_tags: List[str] = None,
    ):

        self.logger = getLogger()

        try:
            from accelerate import Accelerator
            self.accelerator = Accelerator()
        except ImportError:
            self.accelerator = None
            self.log("Accelerate not available, using single GPU mode")
        

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.category = category
        self.json_config_path = json_config_path
        self.checkpoint_path = checkpoint_path
        self.tokenizer_class = tokenizer
        self.task = task
        self.epochs_per_round = epochs_per_round
        self.item_history_days = item_history_days
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.max_history_len = max_history_len
        self.accumulate_history = accumulate_history
        self.max_train_samples_per_user = max_train_samples_per_user
        self.training_mode = training_mode
        self.data_mode = data_mode
        self.trainer_class = trainer_class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.full_user_eval_data = None

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_tags = wandb_tags or [model_name, dataset_name, task]
        self.wandb_run = None

        self.config = config_dict.copy() if config_dict else {}
        self.config['model_name'] = model_name
        self.config['dataset_name'] = dataset_name
        self.config['accelerator'] = self.accelerator
        self.config['max_train_samples_per_user'] = max_train_samples_per_user
        self.config['task'] = task
        self.config['epochs'] = epochs_per_round
        self.config['use_wandb'] = self.use_wandb
        

        self.output_dir = self.config.get('output_dir', './outputs/streaming')
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.raw_dataset = None
        self.history_store = defaultdict(lambda: deque(maxlen=self.max_history_len))
        self.carryover_len = int(getattr(self, "carryover_len", 0) or self.max_seq_len)


        self.training_results = []
        self.best_val_score = float('-inf')
        self.best_round = 0

        self.global_step = 0
        

        self.data_config = None
        self.interaction_files = []
        self.item_files = []
        self.rounds = []
        

        if self.json_config_path:
            self._load_json_config()

        if self.data_mode == "full_user":
            self._load_full_user_eval_data()

        
        if self.use_wandb:
            self._init_wandb()
        
       
        self._log_config()
    
    def _init_wandb(self):
        
        if not WANDB_AVAILABLE:
            self.log("wandb not available, skipping initialization", level='warning')
            return
        
        try:
            
            wandb_config = {
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'task': self.task,
                'training_mode': self.training_mode,
                'epochs_per_round': self.epochs_per_round,
                'item_history_days': self.item_history_days,
                'max_seq_len': self.max_seq_len,
                'max_history_len': self.max_history_len,
                'accumulate_history': self.accumulate_history,
                'max_train_samples_per_user': self.max_train_samples_per_user,
                'model_size': self.config.get('model_size', 'small'),
                'lr': self.config.get('lr', 5e-4),
                'train_batch_size': self.config.get('train_batch_size', 64),
                'eval_batch_size': self.config.get('eval_batch_size', 128),
            }
            

            if self.accelerator.is_main_process and self.use_wandb:
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    tags=self.wandb_tags,
                    config=wandb_config,
                    reinit=True
                )
            self.log(f"wandb initialized: {self.wandb_project}/{self.wandb_run_name}")
            self.log(f"  View at: {self.wandb_run.url}")
        except Exception as e:
            self.log(f"Failed to initialize wandb: {e}", level='warning')
            self.use_wandb = False
    
    def log(self, message: str, level: str = 'info'):
        """记录日志消息。"""
        if self.accelerator is None or self.accelerator.is_main_process:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
            else:
                self.logger.info(message)
    
    def _load_json_config(self):
        
        if not os.path.exists(self.json_config_path):
            raise FileNotFoundError(f"Config file not found: {self.json_config_path}")
        
        with open(self.json_config_path, 'r') as f:
            self.data_config = json.load(f)
        
        # 
        if 'interaction_files' in self.data_config:
            raw_inter_files = self.data_config['interaction_files']
            if raw_inter_files and os.path.dirname(raw_inter_files[0]):
                self.interaction_files = raw_inter_files
            else:
                data_dir = self.data_config.get('data_dir', './data')
                inter_subdir = self.data_config.get('interaction_subdir', 'interactions')
                inter_dir = os.path.join(data_dir, inter_subdir)
                self.interaction_files = [os.path.join(inter_dir, f) for f in raw_inter_files]
        
        if 'item_files' in self.data_config:
            raw_item_files = self.data_config['item_files']
            if raw_item_files and os.path.dirname(raw_item_files[0]):
                self.item_files = raw_item_files
            else:
                data_dir = self.data_config.get('data_dir', './data')
                item_subdir = self.data_config.get('item_subdir', 'items')
                item_dir = os.path.join(data_dir, item_subdir)
                self.item_files = [os.path.join(item_dir, f) for f in raw_item_files]
        
        self.interaction_files = sorted(self.interaction_files)
        self.item_files = sorted(self.item_files)
        
        if 'task' in self.data_config:
            self.task = self.data_config['task']
        
        self._build_training_rounds()
        
        self.log(f"Loaded config from {self.json_config_path}")
        self.log(f"  - Interaction files: {len(self.interaction_files)}")
        self.log(f"  - Item files: {len(self.item_files)}")
        self.log(f"  - Training rounds: {len(self.rounds)}")
    
    def _build_training_rounds(self):
        """
        """
        self.rounds = []

        if self.data_mode == "full_user":
            for i, interaction_file in enumerate(self.interaction_files):
                self.rounds.append({
                    "round": i + 1,
                    "interaction_file": interaction_file,
                    "item_files": self.item_files,   
                })

            self.total_rounds = len(self.rounds)
            self.log(f"Built {self.total_rounds} full_user training rounds")

            if self.rounds:
                r = self.rounds[0]
                item_names = [os.path.basename(f) for f in r["item_files"]]
                self.log(f"  - Round 1 interaction: {os.path.basename(r['interaction_file'])}")
                self.log(f"  - Fixed item files ({len(item_names)}): {item_names}")

            return

        self.total_rounds = len(self.interaction_files)

        for i in range(self.total_rounds):
            interaction_file, item_files = self._get_round_files(i + 1)
            self.rounds.append({
                'round': i + 1,
                'interaction_file': interaction_file[0] if isinstance(interaction_file, list) else interaction_file,
                'item_files': item_files
            })

        self.log(f"Built {len(self.rounds)} training rounds")
        if self.rounds:
            for r in self.rounds[:3]:
                item_names = [os.path.basename(f) for f in r['item_files']]
                self.log(f"  - Round {r['round']}: {os.path.basename(r['interaction_file'])} + {item_names}")

    
    def _log_config(self):
        
        self.log(f"\n{'='*60}")
        self.log("Streaming Pipeline Configuration")
        self.log(f"{'='*60}")
        self.log(f"Model: {self.model_name}")
        self.log(f"Dataset: {self.dataset_name}")
        self.log(f"Task: {self.task}")
        self.log(f"Training Mode: {self.training_mode}")
        self.log(f"Epochs per Round: {self.epochs_per_round}")
        self.log(f"Item History Days: {self.item_history_days}")
        self.log(f"Max Seq Length: {self.max_seq_len}")
        self.log(f"Max History Length: {self.max_history_len}")
        self.log(f"Accumulate History: {self.accumulate_history}")
        self.log(f"Output Dir: {self.output_dir}")
        self.log(f"Use wandb: {self.use_wandb}")
        self.log(f"{'='*60}\n")
    
    def _load_data_for_round(self, round_num, splits=None):
        if splits is None:
            splits = ['train', 'val', 'test']

        interaction_files, item_files = self._get_round_files(round_num)

        
        if self.data_mode == "full_user":
            self.raw_dataset = StreamingInteractionDataset(
                config={**self.config, 'max_train_samples_per_user': self.max_train_samples_per_user, 'debug': True},
                task=self.task,
                max_seq_len=self.max_seq_len,
                min_seq_len=self.min_seq_len,
                max_history_len=self.max_history_len,
                accumulate_history=False,   
                splits=['train'],           
            )

            
            self.raw_dataset.carryover_len = 0

            datasets = self.raw_dataset.load_train_only_from_files(
                interaction_files=interaction_files,
                item_files=item_files,
                history_store=None,         
                #splits=None
            )

            train_size = len(datasets.get('train', []))
            self.log(
                f"[full_user] Round {round_num}: "
                f"interaction={interaction_files}, item_files={len(item_files)}, train_samples={train_size}"
            )

            return datasets

        
        self._load_carryover_store_for_round(round_num)

        self.raw_dataset = StreamingInteractionDataset(
            config={**self.config, 'max_train_samples_per_user': self.max_train_samples_per_user, 'debug': True},
            task=self.task,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            max_history_len=self.max_history_len,
            accumulate_history=self.accumulate_history,
            splits=splits,
        )

        self.raw_dataset.carryover_len = self.carryover_len

        datasets = self.raw_dataset.load_from_files(
            interaction_files=interaction_files,
            item_files=item_files,
            history_store=self.history_store,
            splits=splits
        )

        
        total_items = sum(len(v) for v in self.history_store.values())
        self.log(f"Round {round_num}: history_store users={len(self.history_store)}, total_items={total_items}")

        return datasets

    
    def _create_new_model(self):
       
        import yaml
        import os
        import importlib
        
        self.log(f"Creating new model for '{self.model_name}' (case-insensitive)...")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized before creating model")

        
        model_base_dir = "model"
        actual_model_folder = None
        
        if os.path.exists(model_base_dir):
            for folder in os.listdir(model_base_dir):
                if os.path.isdir(os.path.join(model_base_dir, folder)) and folder.lower() == self.model_name.lower():
                    actual_model_folder = folder
                    break
                    
        if not actual_model_folder:
            raise FileNotFoundError(f"Model directory for '{self.model_name}' not found in '{model_base_dir}/' (case-insensitive search).")

        
        model_size = self.config.get('model_size', 'small')
        vocab_size = self.config.get('vocab_size', 4096)
        self.config['max_token_seq_len'] = self.config.get('max_token_seq_len', 512)
        self.config['sid_offset'] = self.config.get('sid_offset', 7)
        self.config['n_digits'] = self.config.get('n_digits', 8)
        self.config['padding_token'] = self.tokenizer.padding_token

        
        config_path = os.path.join(model_base_dir, actual_model_folder, "config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            
        
        for k, v in yaml_config.items():
            if k != 'presets' and k not in self.config:
                self.config[k] = v

        
        presets = yaml_config.get('presets', {})
        if model_size not in presets:
            self.log(f"Warning: model_size '{model_size}' not found in presets. Falling back to 'small'.", level='warning')
            model_size = 'small'
            
        model_config = presets.get(model_size, {})
        
        
        self.config.update(model_config)
        self.config['vocab_size'] = vocab_size
        
        
        try:
            module_name = f"model.{actual_model_folder}.model"
            model_module = importlib.import_module(module_name)
            
           
            ModelClass = None
            for attr_name in dir(model_module):
                if attr_name.lower() == self.model_name.lower():
                    attr = getattr(model_module, attr_name)
                    if isinstance(attr, type):  
                        ModelClass = attr
                        break
                        
            if ModelClass is None:
                raise RuntimeError(f"Cannot find class matching '{self.model_name}' in module '{module_name}'")
                
        except Exception as e:
            raise RuntimeError(f"Failed to dynamically import model '{self.model_name}': {e}")
            
        self.model = ModelClass(self.config, self.tokenizer)
        
        
        actual_class_name = ModelClass.__name__
        
        self.log(f"Using {actual_class_name} model (Size: {model_size})")
        self.log(f"  - num_layers: {self.config.get('num_layers')}")
        self.log(f"  - d_model: {self.config.get('d_model')}")
        self.log(f"  - num_heads: {self.config.get('num_heads')}")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model created: {model_size}, Parameters: {num_params:,}")
        
        
        if self.use_wandb and self.wandb_run:
            wandb.config.update({
                'model_params': num_params,
                'vocab_size': vocab_size,
                **model_config
            }, allow_val_change=True)
    
    def _create_default_model(self, vocab_size: int, config: dict):
        
        import torch.nn as nn
        
        d_model = config.get('d_model', 512)
        nhead = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 6)
        dim_feedforward = config.get('d_ff', 1024)
        
        class DefaultTransformerModel(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoder = nn.Embedding(1024, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_layer = nn.Linear(d_model, vocab_size)
            
            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                
                x = self.embedding(input_ids)
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                x = x + self.pos_encoder(positions)
                
                if attention_mask is not None:
                    mask = attention_mask == 0
                else:
                    mask = None
                
                x = self.transformer(x, src_key_padding_mask=mask)
                logits = self.output_layer(x)
                
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    if labels.dim() == 2:
                        loss = loss_fn(logits[:, -1, :], labels[:, 0])
                    else:
                        loss = loss_fn(logits[:, -1, :], labels)
                
                return type('Output', (), {'loss': loss, 'logits': logits})()
            
            def generate(self, batch, n_return_sequences=1):
                input_ids = batch['input_ids']
                outputs = self.forward(input_ids=input_ids, attention_mask=batch.get('attention_mask'))
                logits = outputs.logits[:, -1, :]
                _, top_indices = torch.topk(logits, n_return_sequences, dim=-1)
                return top_indices.unsqueeze(-1)
        
        return DefaultTransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    
    def _load_checkpoint(self, path: str, round_num: int):
        
        if not os.path.exists(path):
            self.log(f"Checkpoint not found: {path}", level='warning')
            return

        self.log(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location='cpu')

        
        if 'tokenizer' in checkpoint:
            if self.tokenizer is None:
                self.tokenizer = checkpoint['tokenizer']
            else:
                self.log("Tokenizer already initialized, skip loading from checkpoint")
        else:
            
            if self.tokenizer is None:
                
                raise RuntimeError(
                    "Checkpoint has no tokenizer and tokenizer not initialized. "
                    "Please call _initialize_tokenizer_only() before loading checkpoint."
                )
            else:
                self.log("Checkpoint has no tokenizer, using existing tokenizer")

        
        if self.model is None:
            self._initialize_model_only()

        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)


    
    def _load_model_state_only(self, path: str):
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.log(f"Loading model state only from {path}")
        checkpoint = torch.load(path, map_location='cpu')

        if self.model is None:
            self._initialize_model_only()

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


    
    def _save_checkpoint(self, path: str):
        
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        
        self.log(f"Saving checkpoint to {path}")
        
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            model_state = unwrapped_model.state_dict()
        else:
            model_state = self.model.state_dict()

        tokenizer_config = {
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'max_token_seq_len': self.tokenizer.max_token_seq_len if self.tokenizer else None,
            'pad_order': self.tokenizer.pad_order if self.tokenizer else 'left',
        }
        
        torch.save({
            'model_state_dict': model_state,
            'tokenizer_config': tokenizer_config, 
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None
        }, path)
        
    def _initialize_trainer(self):
        
        if self.trainer_class is not None:
            self.trainer = self.trainer_class(self.config, self.model, self.tokenizer)
        else:
            self.trainer = DefaultTrainer(
                self.config, 
                self.model, 
                self.tokenizer, 
                self.accelerator,
                use_wandb=self.use_wandb,
                global_step_ref=lambda: self.global_step,
                global_step_setter=self._set_global_step
            )
    
    def _set_global_step(self, step: int):
         
        self.global_step = step
        
    def _run_single_round(self, round_num: int) -> Dict[str, Any]:
         
        print("run_single_round start")

        tokenized_data = None
        datasets = None
        train_loader = None
        val_loader = None
        test_loader = None

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if self.raw_dataset is not None:
            self.raw_dataset._accumulated_loaded = False

        from torch.utils.data import DataLoader

        self.log(f"\n{'='*60}")
        self.log(f"Round {round_num}/{len(self.rounds)}")
        self.log(f"{'='*60}")

        round_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num}.pt")
        eval_output_path = os.path.join(self.output_dir, f"eval_outputs_round_{round_num}.pt")
        eval_metrics_path = os.path.join(self.output_dir, f"eval_metrics_round_{round_num}.json")
        val_metrics_path = os.path.join(self.output_dir, f"val_metrics_round_{round_num}.json")

         
        if self.data_mode == "full_user":
            skip_training = os.path.exists(round_ckpt_path)
            skip_eval_metrics = os.path.exists(eval_metrics_path)

            tokenized_train_count = 0
            best_epoch = 0
            best_val_score = 0.0
            val_results = {}
            test_results = {}

            
            datasets = self._load_data_for_round(round_num, ['train'])

           
            self._initialize_tokenizer_only(round_num)

            if skip_training:
                self.log(f"[full_user] Round {round_num}: loading checkpoint...")
                self._load_checkpoint(round_ckpt_path, round_num)
            else:
                if self.training_mode == 'from_scratch':
                    self._create_new_model()

                elif self.training_mode == 'from_checkpoint':
                    if round_num == 1:
                        self._create_new_model()
                        if self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path)
                    else:
                        prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                        self._create_new_model()
                        if os.path.exists(prev_ckpt):
                            self._load_checkpoint(prev_ckpt, round_num - 1)

                else:  # continual
                    if self.model is None:
                        self._create_new_model()
                        if round_num > 1:
                            prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                            if os.path.exists(prev_ckpt):
                                self._load_checkpoint(prev_ckpt, round_num - 1)
                                self.log(f"[full_user] loaded checkpoint from round {round_num - 1}")
                            elif self.checkpoint_path:
                                self._load_checkpoint(self.checkpoint_path)
                        elif self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path)

            self._initialize_trainer()

            # ---------- 3. tokenize train ----------
            tokenized_data = self.tokenizer.tokenize(datasets)

            if self.raw_dataset is not None:
                self.raw_dataset.item_data = {}
                if hasattr(self.raw_dataset, "item_meta"):
                    self.raw_dataset.item_meta = {}
                self.raw_dataset = None

            del datasets
            gc.collect()

            train_samples = tokenized_data.get('train', []) if tokenized_data else []
            tokenized_train_count = len(train_samples)

            self.log(f"[full_user] Round {round_num}: tokenized train={tokenized_train_count}")

            if train_samples:
                train_loader = DataLoader(
                    train_samples,
                    num_workers=8,
                    batch_size=self.config.get('train_batch_size', 64),
                    shuffle=True,
                    collate_fn=self.tokenizer.collate_fn['train']
                )

                if self.trainer.accelerator is not None:
                    train_loader = self.trainer.accelerator.prepare(train_loader)

            del tokenized_data
            gc.collect()

            # ---------- 4. train ----------
            if not skip_training and train_loader is not None:
                self.log(f"[full_user] Round {round_num}: training for {self.epochs_per_round} epochs...")
                best_epoch, best_val_score, val_results = self.trainer.fit(
                    train_loader,
                    None,
                    epochs=self.epochs_per_round,
                    round_num=round_num
                )

                if self.trainer.accelerator is not None:
                    self.trainer.accelerator.wait_for_everyone()

                self._save_checkpoint(round_ckpt_path)

                if not val_results:
                    val_results = {}

            # ---------- 5. full_user eval ----------
            if skip_eval_metrics:
                self.log(f"[full_user] Round {round_num}: loading existing eval metrics...")
                with open(eval_metrics_path, 'r') as f:
                    test_results = json.load(f)
            else:
                self.log(f"[full_user] Round {round_num}: evaluating with full_user eval cache...")

                eval_loader = self._build_full_user_eval_loader()
                if eval_loader is None:
                    self.log(f"[full_user] Round {round_num}: no valid eval samples")
                    test_results = {}
                else:
                    test_results = self.trainer._evaluate_with_full_metrics(
                        eval_loader, round_num=round_num, split='test'
                    )

                    with open(eval_metrics_path, 'w') as f:
                        json.dump(test_results, f, indent=2)
                    self.log(f"[full_user] Round {round_num}: eval metrics saved to {eval_metrics_path}")

            
            if test_results:
                ndcg_list = test_results.get("NDCG", [])
                valid_topk = test_results.get("valid_topk", [])

                if 50 in valid_topk:
                    best_val_score = ndcg_list[valid_topk.index(50)]
                elif len(ndcg_list) > 0:
                    best_val_score = ndcg_list[-1]
                else:
                    best_val_score = 0.0

            # ---------- 7.   round result ----------
            round_result = {
                'round': round_num,
                'best_epoch': best_epoch,
                'best_val_score': best_val_score,
                'val_results': val_results,
                'test_results': test_results,
                'train_samples': tokenized_train_count,
                'val_samples': 0,
                'test_samples': test_results.get('num_samples', 0) if isinstance(test_results, dict) else 0
            }

            self.training_results.append(round_result)

            # ---------- 8. wandb ----------
            if self.use_wandb and self.wandb_run:
                log_dict = {
                    'round_summary/round': round_num,
                    'round_summary/best_epoch': best_epoch,
                    'round_summary/best_val_score': best_val_score,
                    'round_summary/train_samples': tokenized_train_count,
                }

                if test_results:
                    for k, v in test_results.items():
                        if isinstance(v, (int, float)):
                            log_dict[f'round_summary/test_{k}'] = v

                    valid_topk = test_results.get("valid_topk", [])
                    for metric_name in ["NDCG", "HR", "Recall", "MRR"]:
                        vals = test_results.get(metric_name, [])
                        if isinstance(vals, list):
                            for k, val in zip(valid_topk, vals):
                                if isinstance(val, (int, float)):
                                    log_dict[f'round_summary/test_{metric_name}@{k}'] = val

                self._log_wandb(log_dict, step=self.global_step)

            # ---------- 9.   ----------
            if best_val_score > self.best_val_score:
                self.best_val_score = best_val_score
                self.best_round = round_num
                best_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")

                best_model_in_training = None
                if self.trainer is not None:
                    best_model_in_training = getattr(self.trainer, "saved_model_ckpt", None)

                if best_model_in_training and os.path.exists(best_model_in_training):
                    import shutil
                    shutil.copy2(best_model_in_training, best_ckpt_path)
                    self.log(f"[full_user] New best model saved! Round {round_num}, score: {best_val_score:.4f}")
                elif os.path.exists(round_ckpt_path):
                    import shutil
                    shutil.copy2(round_ckpt_path, best_ckpt_path)
                    self.log(f"[full_user] New best model saved from round ckpt! Round {round_num}, score: {best_val_score:.4f}")
                else:
                    self.log(f"[full_user] Warning: Cannot save best model, no checkpoint available")

            self.log(f"[full_user] Round {round_num}: Complete")
            self.log(f"  - Best score (NDCG@50): {best_val_score:.4f}")
            if test_results:
                self.log(f"  - Test results: {test_results}")

            if self.trainer is not None and self.trainer.accelerator is not None:
                self.trainer.accelerator.wait_for_everyone()

            if 'datasets' in locals():
                del datasets
            if 'tokenized_data' in locals():
                del tokenized_data
            if train_loader is not None:
                del train_loader
            if val_loader is not None:
                del val_loader
            if test_loader is not None:
                del test_loader

            gc.collect()
            torch.cuda.empty_cache()

            print("run_single_round end")
            return round_result

        
        tokenized_train_count = 0
        tokenized_val_count = 0
        tokenized_test_count = 0

        skip_training = os.path.exists(round_ckpt_path)
        skip_eval_output = os.path.exists(eval_output_path)
        skip_eval_metrics = os.path.exists(eval_metrics_path)
        skip_val_metrics = os.path.exists(val_metrics_path)

        need_train_data = not skip_training
        need_val_data = not skip_val_metrics
        need_test_data = not skip_eval_metrics

        train_loader = None
        val_loader = None
        test_loader = None

        if need_train_data or need_val_data or need_test_data:
            needed_splits = []
            if need_train_data:
                needed_splits.append('train')
            if need_val_data:
                needed_splits.append('val')
            if need_test_data:
                needed_splits.append('test')

            datasets = self._load_data_for_round(round_num, needed_splits)

            self._initialize_tokenizer_only(round_num)

            if skip_training:
                self.log(f"Round {round_num}: Loading checkpoint...")
                self._load_checkpoint(round_ckpt_path, round_num)
            else:
                if self.training_mode == 'from_scratch':
                    self._create_new_model()

                elif self.training_mode == 'from_checkpoint':
                    if round_num == 1:
                        self._create_new_model()
                        if self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path)
                    else:
                        prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                        self._create_new_model()
                        if os.path.exists(prev_ckpt):
                            self._load_checkpoint(prev_ckpt)

                else:  # continual
                    if self.model is None:
                        self._create_new_model()
                        if round_num > 1:
                            prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                            if os.path.exists(prev_ckpt):
                                self._load_checkpoint(prev_ckpt, round_num - 1)
                                self.log(f"Loaded checkpoint from round {round_num - 1} for continual training")
                            elif self.checkpoint_path:
                                self._load_checkpoint(self.checkpoint_path)
                                self.log(f"Loaded initial checkpoint for continual training")
                        elif self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path)
                            self.log(f"Loaded initial checkpoint for continual training")

            self._initialize_trainer()

            tokenized_data = self.tokenizer.tokenize(datasets)

            if self.raw_dataset is not None:
                self.raw_dataset.item_data = {}
                if hasattr(self.raw_dataset, "item_meta"):
                    self.raw_dataset.item_meta = {}
                self.raw_dataset = None

            del datasets
            gc.collect()

            tokenized_train_count = len(tokenized_data['train']) if tokenized_data.get('train') else 0
            tokenized_val_count = len(tokenized_data['val']) if tokenized_data.get('val') else 0
            tokenized_test_count = len(tokenized_data['test']) if tokenized_data.get('test') else 0

            self.log(f"Tokenized: train={tokenized_train_count}, val={tokenized_val_count}, test={tokenized_test_count}")

            if need_train_data and tokenized_data.get('train'):
                train_loader = DataLoader(
                    tokenized_data['train'],
                    num_workers=8,
                    batch_size=self.config.get('train_batch_size', 64),
                    shuffle=True,
                    collate_fn=self.tokenizer.collate_fn['train']
                )

            if need_val_data and tokenized_data.get('val'):
                val_loader = DataLoader(
                    tokenized_data['val'],
                    num_workers=8,
                    batch_size=self.config.get('eval_batch_size', 128),
                    shuffle=False,
                    collate_fn=self.tokenizer.collate_fn['val']
                ) if tokenized_val_count > 0 else None

            if need_test_data and tokenized_data.get('test'):
                test_loader = DataLoader(
                    tokenized_data['test'],
                    num_workers=8,
                    batch_size=self.config.get('eval_batch_size', 128),
                    shuffle=False,
                    collate_fn=self.tokenizer.collate_fn['test']
                )

            if self.trainer.accelerator is not None:
                if val_loader is not None:
                    val_loader = self.trainer.accelerator.prepare(val_loader)
                    self.log(f"Val loader prepared for {self.trainer.accelerator.num_processes} GPUs")

                if test_loader is not None:
                    test_loader = self.trainer.accelerator.prepare(test_loader)
                    self.log(f"Test loader prepared for {self.trainer.accelerator.num_processes} GPUs")

        del tokenized_data
        gc.collect()

        self.log(f"Round {round_num}: DataLoaders created, tokenized_data deleted from memory")

        best_epoch = 0
        best_val_score = 0.0
        val_results = {}

        if not skip_training and train_loader is not None:
            self.log(f"Round {round_num}: Training for {self.epochs_per_round} epochs...")
            best_epoch, best_val_score, val_results = self.trainer.fit(
                train_loader,
                val_loader,
                epochs=self.epochs_per_round,
                round_num=round_num
            )

            if self.trainer.accelerator is not None:
                self.trainer.accelerator.wait_for_everyone()

            self._save_checkpoint(round_ckpt_path)

            if val_results:
                with open(val_metrics_path, 'w') as f:
                    json.dump(val_results, f, indent=2)
                self.log(f"Round {round_num}: Val metrics saved to {val_metrics_path}")

        if not val_results:
            if skip_val_metrics and os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'r') as f:
                    val_results = json.load(f)
                best_val_score = val_results.get('primary_score', 0)
            elif skip_training and val_loader is not None:
                val_results = self.trainer._evaluate_with_full_metrics(val_loader, round_num=round_num, split='val')
                best_val_score = val_results.get('primary_score', 0)
                with open(val_metrics_path, 'w') as f:
                    json.dump(val_results, f, indent=2)

        test_results = {}

        if skip_eval_metrics:
            self.log(f"Round {round_num}: Loading existing test metrics...")
            with open(eval_metrics_path, 'r') as f:
                test_results = json.load(f)
        elif test_loader is not None:
            if skip_eval_output:
                self.log(f"Round {round_num}: Loading existing eval outputs...")
                eval_outputs = torch.load(eval_output_path)
                test_results = self._calculate_metrics_from_outputs(eval_outputs)
            else:
                self.log(f"Round {round_num}: Evaluating on test set...")
                test_results, eval_outputs = self.trainer.evaluate_with_outputs(
                    test_loader, round_num=round_num, split='test'
                )
                torch.save(eval_outputs, eval_output_path)
                self.log(f"Round {round_num}: Eval outputs saved to {eval_output_path}")

            if self.trainer.accelerator is not None:
                self.trainer.accelerator.wait_for_everyone()
                self.log(f"Round {round_num}: All GPUs finished test evaluation")

            with open(eval_metrics_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            self.log(f"Round {round_num}: Test metrics saved to {eval_metrics_path}")

        round_result = {
            'round': round_num,
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'val_results': val_results,
            'test_results': test_results,
            'train_samples': tokenized_train_count,
            'val_samples': tokenized_val_count,
            'test_samples': tokenized_test_count
        }

        self.training_results.append(round_result)

        if self.use_wandb and self.wandb_run:
            log_dict = {
                'round_summary/round': round_num,
                'round_summary/best_epoch': best_epoch,
                'round_summary/best_val_score': best_val_score,
            }
            if val_results:
                for k, v in val_results.items():
                    if isinstance(v, (int, float)):
                        log_dict[f'round_summary/val_{k}'] = v
            if test_results:
                for k, v in test_results.items():
                    if isinstance(v, (int, float)):
                        log_dict[f'round_summary/test_{k}'] = v
            self._log_wandb(log_dict, step=self.global_step)

        if best_val_score > self.best_val_score:
            self.best_val_score = best_val_score
            self.best_round = round_num
            best_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")

            best_model_in_training = None
            if self.trainer is not None:
                best_model_in_training = getattr(self.trainer, "saved_model_ckpt", None)

            if best_model_in_training and os.path.exists(best_model_in_training):
                import shutil
                shutil.copy2(best_model_in_training, best_ckpt_path)
                self.log(f"New best model saved! Round {round_num}, Val score: {best_val_score:.4f}")
            elif os.path.exists(round_ckpt_path):
                import shutil
                shutil.copy2(round_ckpt_path, best_ckpt_path)
                self.log(f"New best model saved (from round ckpt)! Round {round_num}, Val score: {best_val_score:.4f}")
            else:
                self.log(f"Warning: Cannot save best model, no checkpoint available")

        self.log(f"Round {round_num}: Complete")
        self.log(f"  - Best val score: {best_val_score:.4f}")
        if test_results:
            self.log(f"  - Test results: {test_results}")

 
        self._save_carryover_store(round_num)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        print("run_single_round end")

        if self.trainer is not None and self.trainer.accelerator is not None:
            self.trainer.accelerator.wait_for_everyone()

        if 'datasets' in locals():
            del datasets
        if 'tokenized_data' in locals():
            del tokenized_data
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        if test_loader is not None:
            del test_loader

        gc.collect()
        torch.cuda.empty_cache()

        return round_result

    def _save_training_results(self):
       
        results_file = os.path.join(self.output_dir, 'training_results.json')
        
        data = {
            'config': {
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'task': self.task,
                'epochs_per_round': self.epochs_per_round,
                'training_mode': self.training_mode,
                'max_train_samples_per_user': self.max_train_samples_per_user,
                'max_history_len': self.max_history_len,
            },
            'results': self.training_results,
            'best_round': self.best_round,
            'best_val_score': self.best_val_score
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log(f"Training results saved to {results_file}")
         
        if self.use_wandb and self.wandb_run:
            try:
                artifact = wandb.Artifact(
                    name=f"{self.model_name}_results",
                    type='results'
                )
                artifact.add_file(results_file)
                wandb.log_artifact(artifact)
            except Exception as e:
                self.log(f"Failed to log results artifact: {e}", level='warning')
    
    
    def _generate_summary(self) -> Dict[str, Any]:
         
        if not self.training_results:
            return {'error': 'No training results'}
        
        valid_results = [r for r in self.training_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid training results'}
         
        val_scores = [r['best_val_score'] for r in valid_results]
        avg_val_score = np.mean(val_scores)
        std_val_score = np.std(val_scores)
        
        total_train_samples = sum(r.get('train_samples', 0) for r in valid_results)
        
        summary = {
            'total_rounds': len(self.training_results),
            'successful_rounds': len(valid_results),
            'failed_rounds': len(self.training_results) - len(valid_results),
            'avg_val_score': float(avg_val_score),
            'std_val_score': float(std_val_score),
            'best_round': self.best_round,
            'best_val_score': float(self.best_val_score),
            'total_train_samples': total_train_samples,
        }
        
        test_results_list = [r.get('test_results', {}) for r in valid_results if r.get('test_results')]
        
        if test_results_list:
            # 
            test_keys = test_results_list[0].keys()
            
            for key in test_keys:
                 
                #  
                values = [r.get(key) for r in test_results_list if key in r]
                
                if not values:
                    continue

                try:
                    first_val = values[0]
                    
                     
                    if isinstance(first_val, dict):
                        avg_dict = {}
                         
                        for sub_key in first_val.keys():
                             
                            sub_values = [v.get(sub_key) for v in values if isinstance(v, dict) and sub_key in v]
                            
                            if not sub_values:
                                continue
                                
                             
                            sub_first = sub_values[0]
                            if isinstance(sub_first, (int, float, np.number)):
                                avg_dict[sub_key] = float(np.mean(sub_values))
                            elif isinstance(sub_first, list) and len(sub_first) > 0 and isinstance(sub_first[0], (int, float, np.number)):
                                mean_arr = np.mean(sub_values, axis=0)
                                avg_dict[sub_key] = mean_arr.tolist()
                        
                         
                        summary[f'avg_test_{key}'] = avg_dict

                     
                    elif isinstance(first_val, (int, float, np.number)):
                        summary[f'avg_test_{key}'] = float(np.mean(values))
                        
                     
                    elif isinstance(first_val, list):
                        if len(first_val) > 0 and isinstance(first_val[0], (int, float, np.number)):
                            mean_val = np.mean(values, axis=0)
                            if isinstance(mean_val, np.ndarray):
                                summary[f'avg_test_{key}'] = mean_val.tolist()
                            else:
                                summary[f'avg_test_{key}'] = mean_val
                        else:
                            pass  
                    
                    else:
                        pass 

                except Exception as e:
                    self.log(f"Warning: Could not calculate average for metric '{key}': {e}", level='warning')
        
        
        self.log("\n" + "="*60)
        self.log("Training Summary")
        self.log("="*60)
        for k, v in summary.items():
            if isinstance(v, float):
                self.log(f"  {k}: {v:.4f}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], float):
                formatted_list = [f"{x:.4f}" for x in v[:5]]
                self.log(f"  {k}: {formatted_list}..." if len(v)>5 else f"  {k}: {formatted_list}")
            elif isinstance(v, dict):
                 
                keys_show = list(v.keys())[:3]
                self.log(f"  {k}: Dict containing {keys_show}...")
            else:
                self.log(f"  {k}: {v}")
        self.log("="*60 + "\n")
        
        return summary

    def _log_wandb(self, metrics: dict, step=None):
         
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)

    def _initialize_model_only(self):
         
        self.log("Initializing model only (without tokenizer)...")
        
        
        if 'vocab_size' not in self.config:
            
            self.config['vocab_size'] = self.config.get('default_vocab_size', 4103)
            self.config['padding_token'] = 0
            self.config['sid_offset'] = self.config.get('sid_offset', 7)
            self.config['max_token_seq_len'] = self.config.get('max_token_seq_len', 512)
        
        self._create_new_model()


    def load_from_files(self, interaction_files, item_files, history_store, splits=None):
        if splits is None:
            splits = ['train', 'val', 'test']
 
        carryover_len = int(getattr(self, "carryover_len", 0) or self.max_seq_len)
        carryover_len = min(carryover_len, self.max_seq_len)

        
        self.item_meta = self._load_item_meta(item_files)
        interactions = self._load_interactions(interaction_files)

       
        per_user_new = defaultdict(list)
        for inter in interactions:
            per_user_new[inter['user_id']].append(inter['item_id'])

        datasets = {s: [] for s in splits}

         
        for user_id, new_items in per_user_new.items():
            if not new_items:
                continue

             
            base_seq = list(history_store.get(user_id, []))
            history = deque(base_seq, maxlen=self.max_seq_len)

            n = len(new_items)
            for i, item in enumerate(new_items):
             
                if len(history) >= self.min_seq_len:
                    sample = {
                        'user_id': user_id,
                        'history_item_ids': list(history),
                        'target_item': item,
                    }
                    if i == n - 1 and 'test' in splits:
                        datasets['test'].append(sample)
                    elif i == n - 2 and n > 1 and 'val' in splits:
                        datasets['val'].append(sample)
                    elif 'train' in splits:
                        datasets['train'].append(sample)

                history.append(item)

 
            tail = list(history)[-carryover_len:]  
            history_store[user_id] = deque(tail, maxlen=carryover_len)

        return datasets

    def _get_round_files(self, round_num: int):
        
        if round_num < 1:
            raise ValueError(f"round_num must be >= 1, got {round_num}")

        if self.data_mode == "full_user":
            idx = round_num - 1
            if idx >= len(self.interaction_files):
                raise IndexError(f"round_num={round_num} out of range, total={len(self.interaction_files)}")
            return [self.interaction_files[idx]], self.item_files

        
        interaction_idx = round_num - 1
        if interaction_idx >= len(self.interaction_files):
            raise IndexError(f"round_num={round_num} out of range, total={len(self.interaction_files)}")

        interaction_files = [self.interaction_files[interaction_idx]]

        item_window_days = self.item_history_days
        item_start_idx = interaction_idx
        item_end_idx = interaction_idx + item_window_days

        if item_end_idx > len(self.item_files):
            raise IndexError(
                f"Round {round_num}: item window [{item_start_idx}:{item_end_idx}] "
                f"exceeds available item files ({len(self.item_files)})"
            )

        item_files = self.item_files[item_start_idx:item_end_idx]
        return interaction_files, item_files


    def run(self,start_round: int = 1) -> Dict[str, Any]:
       
        self.log("\n" + "="*60)
        self.log("Starting Streaming Training")
        self.log("="*60 + "\n")
        
        total_rounds = len(self.rounds)
        self.log(f"Total rounds: {total_rounds}")

         
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_name = (
            f"{self.model_name}_"
            f"lr{self.config.get('learning_rate', 0.001)}_"
            f"bs{self.config.get('train_batch_size', 64)}_"
            f"maxlen{self.max_history_len}_"
            f"{timestamp}"
        )
        
      
        if self.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,  
                tags=self.wandb_tags,
                config={
                    'model': self.model_name,
                    'learning_rate': self.config.get('learning_rate'),
                    'batch_size': self.config.get('train_batch_size'),
                    'max_history_len': self.max_history_len,
                    'timestamp': timestamp,
                }
            )
            self.logger.info(f"Initialized wandb run: {run_name}")
        
        for round_num in range(start_round, total_rounds + 1):
            try:
                round_result = self._run_single_round(round_num)
                 
                if self.use_wandb and self.wandb_run:
                    self._log_wandb({
                        'training/completed_rounds': round_num,
                        'training/progress': round_num / total_rounds
                    }, step=self.global_step)
                    
            except Exception as e:
                self.log(f"Round {round_num} Error: {str(e)}", level='error')
                import traceback
                error_trace = traceback.format_exc()
                self.log(error_trace, level='error')
                
                # 
                if self.use_wandb and self.wandb_run:
                    self._log_wandb({
                        'training/error_round': round_num,
                        'training/error_message': str(e)
                    }, step=self.global_step)
                continue
        
        #  
        self._save_training_results()
         
        summary = self._generate_summary()
        
         
        if self.use_wandb and self.wandb_run:
            self._log_wandb({
                'training/status': 'completed',
                'training/final_best_round': self.best_round,
                'training/final_best_val_score': self.best_val_score,
            }, step=self.global_step)
            
            
            wandb.run.summary.update(summary)
            
             
            wandb.finish()
            self.log("wandb run finished")
        
        return {
            'total_rounds': total_rounds,
            'best_round': self.best_round,
            'best_val_score': self.best_val_score,
            'results': self.training_results,
            'summary': summary
        }

    def resume_training(self, from_round: int = 1) -> Dict[str, Any]:
         
        total_rounds = len(self.rounds)
        if from_round < 1 or from_round > total_rounds:
            raise ValueError(f"from_round must be between 1 and {total_rounds}, got {from_round}")
        
        self.log(f"Resuming training from round {from_round}")
        
        # 
        if self.training_mode == 'continual' and from_round > 1:
            prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{from_round - 1}.pt")
            if os.path.exists(prev_ckpt):
                self._create_new_model()
                self._load_checkpoint(prev_ckpt, from_round - 1)
                self.log(f"Loaded model from round {from_round - 1} for continual training")
            else:
                self.log(f"Warning: No checkpoint found for round {from_round - 1}, starting with fresh model")
        
        return self.run(start_round=from_round)

    def _initialize_tokenizer_only(self, round_num: int):
        
        self.log(f"Round {round_num}: Initializing tokenizer only...")
        
        if self.tokenizer_class is not None:
            if isinstance(self.tokenizer_class, type):
                self.tokenizer = self.tokenizer_class(
                    self.config,
                    self.raw_dataset,
                    n_digits=self.config.get('n_digits', 8),
                    codebook_size=self.config.get('codebook_size', 512)
                )
            else:
                self.tokenizer = self.tokenizer_class
        else:
            from tokenizer.streaming_tokenizer import StreamingTokenizer
            self.tokenizer = StreamingTokenizer(
                self.config,
                self.raw_dataset,
                n_digits=self.config.get('n_digits', 8),
                codebook_size=self.config.get('codebook_size', 512)
            )
         
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.config['padding_token'] = self.tokenizer.padding_token

    def evaluate_only(self, round_num: int = None) -> Dict[str, Any]:
         
        if round_num is None:
            round_num = len(self.rounds)

        round_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num}.pt")
        if not os.path.exists(round_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {round_ckpt_path}")
 
        if self.data_mode == "full_user":
            self._load_checkpoint(round_ckpt_path, round_num)
            self._load_data_for_round(round_num, ['train'])
            self._initialize_tokenizer_only(round_num)
            self._initialize_trainer()

            eval_loader = self._build_full_user_eval_loader()
            if eval_loader is None:
                self.log(f"[full_user] No valid eval samples for round {round_num}")
                return {'round': round_num, 'test_results': {}}

            test_results = self.trainer._evaluate_with_full_metrics(
                eval_loader, round_num=round_num, split='test'
            )

            self.log(f"[full_user] Evaluation results for round {round_num}: {test_results}")
            return {'round': round_num, 'test_results': test_results}
 
        datasets = self._load_data_for_round(round_num, ['test'])
        self._initialize_tokenizer_only(round_num)
        self._load_checkpoint(round_ckpt_path, round_num)
        self._initialize_trainer()

        tokenized_data = self.tokenizer.tokenize(datasets)

        self._attach_test_metadata_to_tokenized_data(
            raw_test_samples=datasets['test'],
            tokenized_test_samples=tokenized_data['test']
        )

        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            tokenized_data['test'],
            batch_size=self.config.get('eval_batch_size', 8),
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        if self.trainer.accelerator is not None:
            test_loader = self.trainer.accelerator.prepare(test_loader)

        test_results, eval_outputs = self.trainer.evaluate_with_outputs(
            test_loader, round_num=round_num, split='test'
        )

        eval_output_path = os.path.join(self.output_dir, f"eval_outputs_round_{round_num}.pt")
        eval_metrics_path = os.path.join(self.output_dir, f"eval_metrics_round_{round_num}.json")
        behavior_metrics_path = os.path.join(self.output_dir, f"eval_behavior_metrics_round_{round_num}.json")

        if self.trainer.accelerator is None or self.trainer.accelerator.is_main_process:
            torch.save(eval_outputs, eval_output_path)

            with open(eval_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self._to_jsonable(test_results), f, ensure_ascii=False, indent=2)

            behavior_results = self._calculate_behavior_metrics_from_outputs(eval_outputs)
            with open(behavior_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self._to_jsonable(behavior_results), f, ensure_ascii=False, indent=2)

            self.log(f"Eval outputs saved to {eval_output_path}")
            self.log(f"Eval metrics saved to {eval_metrics_path}")
            self.log(f"Behavior metrics saved to {behavior_metrics_path}")
        else:
            behavior_results = {}

        if self.trainer.accelerator is not None:
            self.trainer.accelerator.wait_for_everyone()

        self.log(f"Evaluation results for round {round_num}: {test_results}")
        return {
            'round': round_num,
            'test_results': test_results,
            'behavior_results': behavior_results,
        }




    def evaluate_all_rounds(self) -> Dict[str, Any]:
 
        all_results = {}
        for round_num in range(1, len(self.rounds) + 1):
            round_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num}.pt")
            if not os.path.exists(round_ckpt_path):
                self.log(f"Round {round_num}: No checkpoint found, skipping")
                continue
            try:
                result = self.evaluate_only(round_num=round_num)
                all_results[round_num] = result
            except Exception as e:
                self.log(f"Round {round_num} evaluation failed: {e}", level='error')
        return all_results

    def _carryover_store_path(self, round_num: int) -> str:
 
        return os.path.join(self.output_dir, f"carryover_store_round_{round_num}.pt")

    def _load_carryover_store_for_round(self, round_num: int):
 
        if round_num <= 1:
            self.history_store = defaultdict(lambda: deque(maxlen=self.carryover_len))
            return

        path = self._carryover_store_path(round_num - 1)
        if not os.path.exists(path):
 
            self.history_store = defaultdict(lambda: deque(maxlen=self.carryover_len))
            return

        obj = torch.load(path, map_location="cpu")
        store = defaultdict(lambda: deque(maxlen=self.carryover_len))
        for u, seq in obj.items():
           
            store[u] = deque(list(seq), maxlen=self.carryover_len)
        self.history_store = store

    def _save_carryover_store(self, round_num: int):
        
        if not self.accelerator.is_main_process:
            return
 
        payload = {u: list(dq) for u, dq in self.history_store.items()}
        torch.save(payload, self._carryover_store_path(round_num))

    def _load_full_user_eval_data(self):
        eval_cfg = self.data_config.get("eval_config", {})
        cache_path = eval_cfg.get("cache_path")
        if cache_path is None:
            raise ValueError("eval_config.cache_path is required for full_user mode")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Full user eval cache not found: {cache_path}")
        import pickle
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        self.full_user_eval_data = payload
        self.log(f"Loaded full_user eval cache from {cache_path}")
        self.log(f"Num eval users: {len(payload.get('eval_users', []))}")


    def _build_full_user_eval_samples(self):
         
        if not getattr(self, "full_user_eval_data", None):
            return []

        eval_samples = []
        for sample in self.full_user_eval_data:
            eval_samples.append({
                "user_id": sample.get("user_id", -1),
                "history_item_ids": sample.get("history_item_ids", []),
                "target_items": sample.get("target_items", []),
            })

        self.log(f"[full_user eval] loaded samples from cache: {len(eval_samples)}")
        return eval_samples


    def _initialize_tokenizer_from_item_files_only(self, round_num: int):
        interaction_files, item_files = self._get_round_files(round_num)

        self.raw_dataset = StreamingInteractionDataset(
            config={**self.config, 'max_train_samples_per_user': self.max_train_samples_per_user, 'debug': True},
            task=self.task,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            max_history_len=self.max_history_len,
            accumulate_history=self.accumulate_history,
            splits=['test'],
        )

        self.raw_dataset.carryover_len = self.carryover_len

 
        self.raw_dataset._load_item_meta(item_files)

        self._initialize_tokenizer_only(round_num)


    def _build_last_day_prev_ckpt_eval_samples(self, eval_round: int, target_limit: int = 50):
         
        if eval_round <= 1:
            raise ValueError("eval_prev_ckpt_last_day requires eval_round >= 2")

        if self.data_mode != "streaming":
            raise ValueError("eval_prev_ckpt_last_day currently supports streaming mode only")

       
        self._load_carryover_store_for_round(eval_round)

         
        interaction_files, item_files = self._get_round_files(eval_round)

         
        self.raw_dataset = StreamingInteractionDataset(
            config={**self.config, 'max_train_samples_per_user': self.max_train_samples_per_user, 'debug': True},
            task=self.task,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            max_history_len=self.max_history_len,
            accumulate_history=False,
            splits=['test'],
        )
        self.raw_dataset.carryover_len = self.carryover_len
 
        self.raw_dataset._load_item_meta(item_files)

     
        interactions = self.raw_dataset._load_interactions(interaction_files)
 
        per_user_targets = defaultdict(list)
        for inter in interactions:
            per_user_targets[inter['user_id']].append(inter['item_id'])

        eval_samples = []
        skipped_short_history = 0
        skipped_empty_targets = 0

        for user_id, today_items in per_user_targets.items():
            if not today_items:
                skipped_empty_targets += 1
                continue

            history = list(self.history_store.get(user_id, []))
            if len(history) < self.min_seq_len:
                skipped_short_history += 1
                continue

            target_items = today_items[:target_limit]
            if len(target_items) == 0:
                skipped_empty_targets += 1
                continue

            eval_samples.append({
                "user_id": int(user_id),
                "history_item_ids": history[-self.max_seq_len:],
                "target_items": [int(x) for x in target_items],
            })

        self.log(
            f"[prev_ckpt_last_day eval] round={eval_round}, "
            f"interaction_files={interaction_files}, "
            f"num_eval_users={len(eval_samples)}, "
            f"skipped_short_history={skipped_short_history}, "
            f"skipped_empty_targets={skipped_empty_targets}"
        )

        return eval_samples

    def _build_last_day_prev_ckpt_eval_loader_from_samples(self, eval_samples):
        tokenized_eval_samples = self._tokenize_full_user_eval_samples(eval_samples)

        if len(tokenized_eval_samples) == 0:
            return None

        from torch.utils.data import DataLoader
        import torch

        pad_id = self.tokenizer.SPECIAL_TOKENS['[PAD]']

        def collate_fn(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None

            max_input_len = max(len(x['input_ids']) for x in batch)

            input_ids = []
            attention_mask = []
            users = []
            target_items = []

            for sample in batch:
                inp = sample['input_ids']
                pad_len = max_input_len - len(inp)

                input_ids.append(inp + [pad_id] * pad_len)
                attention_mask.append([1] * len(inp) + [0] * pad_len)
                users.append(sample.get('user_id', -1))
                target_items.append(sample.get('target_items', []))

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'users': torch.tensor(users, dtype=torch.long),
                'target_items': target_items,
            }

        eval_loader = DataLoader(
            tokenized_eval_samples,
            batch_size=self.config.get('eval_batch_size', 8),
            shuffle=False,
            collate_fn=collate_fn
        )

        if self.trainer is not None and self.trainer.accelerator is not None:
            eval_loader = self.trainer.accelerator.prepare(eval_loader)

        return eval_loader


    
    def evaluate_last_day_with_prev_ckpt(self, eval_round: int = None, target_limit: int = 50) -> Dict[str, Any]:
        if eval_round is None:
            eval_round = len(self.rounds)

        if eval_round <= 1:
            raise ValueError("Need at least 2 rounds for evaluate_last_day_with_prev_ckpt")

        prev_round = eval_round - 1
        prev_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{prev_round}.pt")
        if not os.path.exists(prev_ckpt_path):
            raise FileNotFoundError(f"Previous round checkpoint not found: {prev_ckpt_path}")

        self.log(f"{'='*60}")
        self.log(f"Special Eval: prev_ckpt_last_day")
        self.log(f"Eval day(round): {eval_round}")
        self.log(f"Checkpoint round: {prev_round}")
        self.log(f"Target limit per user: {target_limit}")
        self.log(f"{'='*60}")
 
        eval_samples = self._build_last_day_prev_ckpt_eval_samples(
            eval_round=eval_round,
            target_limit=target_limit,
        )

        self._initialize_tokenizer_only(eval_round)
        self._load_model_state_only(prev_ckpt_path)
        self._initialize_trainer()

        eval_loader = self._build_last_day_prev_ckpt_eval_loader_from_samples(eval_samples)

        if eval_loader is None:
            self.log("[prev_ckpt_last_day eval] no valid eval samples")
            return {
                "eval_round": eval_round,
                "ckpt_round": prev_round,
                "target_limit": target_limit,
                "metrics": {},
                "num_eval_users": 0,
            }

        old_topk_list = list(self.trainer.topk_list)
        self.trainer.topk_list = [50]

        try:
            test_results = self.trainer._evaluate_with_full_metrics(
                eval_loader,
                round_num=eval_round,
                split='prev_ckpt_last_day'
            )
        finally:
            self.trainer.topk_list = old_topk_list

        save_path = os.path.join(
            self.output_dir,
            f"eval_prev_ckpt_last_day_round_{eval_round}.json"
        )

        payload = {
            "mode": "eval_prev_ckpt_last_day",
            "eval_round": eval_round,
            "ckpt_round": prev_round,
            "target_limit": target_limit,
            "num_eval_users": len(eval_samples),
            "metrics": test_results,
        }

        if self.accelerator is None or self.accelerator.is_main_process:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self._to_jsonable(payload), f, ensure_ascii=False, indent=2)
            self.log(f"[prev_ckpt_last_day eval] metrics saved to {save_path}")

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        self.log(f"[prev_ckpt_last_day eval] results: {test_results}")
        return payload



    def _tokenize_full_user_eval_samples(self, eval_samples):
    
        tokenized_samples = []
        skipped_empty_input = 0

        for sample in eval_samples:
            history = sample.get("history_item_ids", [])
            input_token_ids = []

            for item_id in history:
                if item_id in self.tokenizer.item2sid:
                    input_token_ids.extend(self.tokenizer.item2sid[item_id])

            if len(input_token_ids) == 0:
                skipped_empty_input += 1
                continue

            tokenized_samples.append({
                "input_ids": input_token_ids,
                "user_id": sample.get("user_id", -1),
                "target_items": sample.get("target_items", []),
            })

        self.log(
            f"[full_user eval] tokenized samples={len(tokenized_samples)}, "
            f"skipped_empty_input={skipped_empty_input}"
        )
        return tokenized_samples


    def _build_full_user_eval_samples(self):
        
        if not getattr(self, "full_user_eval_data", None):
            return []

        eval_samples = []
        skipped_short_history = 0
        skipped_empty_target = 0

        for sample in self.full_user_eval_data:
            history = sample.get("history_item_ids", [])
            targets = sample.get("target_items", [])

            if len(history) < self.min_seq_len:
                skipped_short_history += 1
                continue

            if not targets:
                skipped_empty_target += 1
                continue

            eval_samples.append({
                "user_id": sample.get("user_id", -1),
                "history_item_ids": history[-self.max_seq_len:],
                "target_items": targets,
            })

        self.log(
            f"[full_user eval] samples={len(eval_samples)}, "
            f"skipped_short_history={skipped_short_history}, "
            f"skipped_empty_target={skipped_empty_target}"
        )

        return eval_samples


    def _build_full_user_eval_loader(self):
         
        eval_samples = self._build_full_user_eval_samples()
        tokenized_eval_samples = self._tokenize_full_user_eval_samples(eval_samples)

        if len(tokenized_eval_samples) == 0:
            return None

        from torch.utils.data import DataLoader
        import torch

        pad_id = self.tokenizer.SPECIAL_TOKENS['[PAD]']

        def collate_fn(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None

            max_input_len = max(len(x['input_ids']) for x in batch)

            input_ids = []
            attention_mask = []
            users = []
            target_items = []

            for sample in batch:
                inp = sample['input_ids']
                pad_len = max_input_len - len(inp)

                input_ids.append(inp + [pad_id] * pad_len)
                attention_mask.append([1] * len(inp) + [0] * pad_len)
                users.append(sample.get('user_id', -1))
                target_items.append(sample.get('target_items', []))

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'users': torch.tensor(users, dtype=torch.long),
                'target_items': target_items,   # 保持 python list，不转 tensor
            }

        eval_loader = DataLoader(
            tokenized_eval_samples,
            batch_size=self.config.get('eval_batch_size', 8),
            shuffle=False,
            collate_fn=collate_fn
        )

        if self.trainer.accelerator is not None:
            eval_loader = self.trainer.accelerator.prepare(eval_loader)

        return eval_loader

    def _attach_test_metadata_to_tokenized_data(self, raw_test_samples, tokenized_test_samples):
        if len(raw_test_samples) != len(tokenized_test_samples):
            raise ValueError(
                f"raw_test_samples != tokenized_test_samples: "
                f"{len(raw_test_samples)} vs {len(tokenized_test_samples)}"
            )

        for idx, (raw_sample, tokenized_sample) in enumerate(zip(raw_test_samples, tokenized_test_samples)):
            tokenized_sample["sample_idx"] = idx
            tokenized_sample["target_item"] = raw_sample.get("target_item")

            for col in BEHAVIOR_COLS:
                tokenized_sample[col] = int(raw_sample.get(col, 0))

    def _calculate_behavior_metrics_from_outputs(self, eval_outputs, behavior_cols=None):
        if behavior_cols is None:
            behavior_cols = BEHAVIOR_COLS

        predictions = eval_outputs["predictions"]
        targets = eval_outputs["targets"]
        behavior_labels = eval_outputs.get("behavior_labels", {})

        item_list = None
        if hasattr(self.tokenizer, "get_all_item_sid_strings"):
            try:
                item_list = self.tokenizer.get_all_item_sid_strings()
            except Exception:
                item_list = None

        results = {}

        for behavior_col in behavior_cols:
            labels = behavior_labels.get(behavior_col)
            if labels is None:
                results[behavior_col] = {
                    "num_samples": 0,
                    "message": f"missing behavior labels: {behavior_col}"
                }
                continue

            mask = [int(x) > 0 for x in labels]
            sub_predictions = [p for p, m in zip(predictions, mask) if m]
            sub_targets = [t for t, m in zip(targets, mask) if m]

            if len(sub_predictions) == 0:
                results[behavior_col] = {
                    "num_samples": 0,
                    "message": "no samples"
                }
                continue

            metrics = calculate_metrics_from_model_outputs(
                predictions=sub_predictions,
                targets=sub_targets,
                item_list=item_list,
                topk_list=[1, 5, 10, 20, 50],
                max_depth=8,
            )
            metrics["num_samples"] = len(sub_predictions)
            results[behavior_col] = metrics

        return results

    def _to_jsonable(self, obj):
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj


class DefaultTrainer:
     
    
    def __init__(
        self, 
        config: dict, 
        model, 
        tokenizer, 
        accelerator=None,
        use_wandb: bool = False,
        global_step_ref=None,
        global_step_setter=None,
        output_dir=None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.logger = getLogger()
        self.saved_model_ckpt = None
        self.output_dir = config.get('output_dir', './output')
 
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.global_step_ref = global_step_ref or (lambda: 0)
        self.global_step_setter = global_step_setter or (lambda x: None)
        self.local_step = 0
         
        self.num_beams = config.get('num_beams', 50)
        self.topk_list = config.get('topk_list', [1, 5, 10, 20, 50])
        self.max_depth = config.get('max_depth', 8)
         
        self._item_sid_list = None
    
    def _get_item_sid_list(self) -> list:
         
        if self._item_sid_list is None:
            if hasattr(self.tokenizer, 'get_all_item_sid_strings'):
                self._item_sid_list = self.tokenizer.get_all_item_sid_strings()
            elif hasattr(self.tokenizer, 'item2sid'):
                self._item_sid_list = []
                for item_id, token_ids in self.tokenizer.item2sid.items():
                    if hasattr(self.tokenizer, 'decode_to_sid'):
                        sid_str = self.tokenizer.decode_to_sid(token_ids)
                    else:
                        sid_str = self.tokenizer.decode(token_ids)
                    self._item_sid_list.append(sid_str)
            else:
                self._item_sid_list = []
        return self._item_sid_list
    
    def _get_device(self):
         
        if self.accelerator is not None:
            return self.accelerator.device
        return next(self.model.parameters()).device
    
    def _get_unwrapped_model(self):
         
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model
    
    def fit(self, train_loader, val_loader, epochs=None, round_num=1):
         
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        lr = self.config.get('lr', 5e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        
        if self.accelerator is not None:
             
            if not getattr(self.model, '_is_accelerate_prepared', False):
                self.model, optimizer, train_loader, scheduler = \
                    self.accelerator.prepare(self.model, optimizer, train_loader, scheduler)
                self.model._is_accelerate_prepared = True
            else:
                 
                optimizer, train_loader, scheduler = \
                    self.accelerator.prepare(optimizer, train_loader, scheduler)
             

            self.log(f"Train loader: {len(train_loader)} batches per GPU")
            self.log(f"Total batches across {self.accelerator.num_processes} GPUs: "
                     f"{len(train_loader) * self.accelerator.num_processes}")
            
        best_val_score = float('-inf')
        best_epoch = 0
        best_val_metrics = {}
        device = next(self.model.parameters()).device
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            is_main_process = self.accelerator is None or self.accelerator.is_main_process
            progress_bar = tqdm(train_loader, desc=f"Round {round_num} Epoch {epoch}/{epochs}",
                                disable=not is_main_process)
            
            for batch in progress_bar:
                if self.accelerator is None:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                if hasattr(self.model, 'module'):
                    inner_model = self.model.module
                else:
                    inner_model = self.model

                if hasattr(inner_model, 't5') or inner_model.__class__.__name__ == 'TIGER':
                    outputs = self.model(batch)
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                
                loss = outputs.loss
                
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.local_step += 1
                
                global_step = self.global_step_ref() + 1
                self.global_step_setter(global_step)
                
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                if self.use_wandb and num_batches % 10 == 0:
                    self._log_wandb({
                        'train/loss': loss.item(),
                        'train/avg_loss': epoch_loss / num_batches,
                        'train/lr': current_lr,
                        'train/epoch': epoch,
                        'train/round': round_num,
                    }, step=global_step)
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            if self.use_wandb:
                self._log_wandb({
                    'epoch/train_loss': avg_epoch_loss,
                    'epoch/epoch': epoch,
                    'epoch/round': round_num,
                }, step=self.global_step_ref())
            
            #  
            if val_loader is not None:
                val_metrics = self._evaluate_with_full_metrics(val_loader, round_num, epoch, split='val')
                val_score = val_metrics.get('primary_score', -val_metrics.get('loss', 0))
                val_loss = val_metrics.get('loss', 0)
                
                if is_main_process:
                    valid_topk = val_metrics.get('valid_topk', [])
                    idx_10 = valid_topk.index(10) if 10 in valid_topk else -1
                    hr_10 = val_metrics.get('HR', [])[idx_10] if idx_10 >= 0 else 0
                    ndcg_10 = val_metrics.get('NDCG', [])[idx_10] if idx_10 >= 0 else 0

                    self.logger.info(f"Round {round_num} Epoch {epoch} - Loss: {avg_epoch_loss:.4f}, "
                                   f"Val Loss: {val_loss:.4f}, HR@10: {hr_10:.4f}, NDCG@10: {ndcg_10:.4f}")
                
                if self.use_wandb:
                    self._log_metrics_to_wandb(val_metrics, prefix='val', round_num=round_num, epoch=epoch)
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch
                    best_val_metrics = val_metrics
                    self._save_best_model()

                    if self.accelerator is not None:
                        self.accelerator.wait_for_everyone()

                    if self.use_wandb:
                        self._log_wandb({
                            'best/val_score': best_val_score,
                            'best/epoch': best_epoch,
                            'best/round': round_num,
                        }, step=self.global_step_ref())

            else:
                best_epoch = epoch
                best_val_score = -avg_epoch_loss
        
        return best_epoch, best_val_score, best_val_metrics


    
    def _evaluate_with_full_metrics(self, data_loader, round_num=1, epoch=1, split='val') -> Dict[str, Any]:
     
        inference_path = os.path.join(
            self.output_dir, 'inference_outputs',
            f'inference_{split}_r{round_num}_e{epoch}.pt'
        )
        os.makedirs(os.path.dirname(inference_path), exist_ok=True)
  
        self._inference_and_save(data_loader, inference_path, round_num, split)
     
        metrics = self._calculate_metrics_from_file(inference_path)
        return metrics

    def _generate_with_scores(self, batch, model) -> Dict:
         
        predictions = []
        probs = []
        scores = []
        batch_size = batch['input_ids'].shape[0]

        try:
            
            if hasattr(model, 'beam_search'):
                n_digit = self.config.get('n_digits', 8)
                
               
                decoder_input_ids, beam_scores = model.beam_search(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=n_digit + 1,  # +1 for initial decoder token
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    return_score=True
                )
                
                
                generated = decoder_input_ids[:, 1:1+n_digit].reshape(batch_size, self.num_beams, -1)
                seq_scores = beam_scores.reshape(batch_size, self.num_beams)
                
                for i in range(batch_size):
                    sample_preds = []
                    for j in range(self.num_beams):
                        tokens = generated[i, j].tolist()
                        sid_str = self.tokenizer.decode_to_sid(tokens)
                        sample_preds.append(sid_str)
                    
                    predictions.append(sample_preds)
                    
                    sample_scores = seq_scores[i]
                    sample_probs = torch.softmax(sample_scores, dim=0)
                    probs.append(sample_probs.cpu().tolist())
                    scores.append(sample_scores.cpu().tolist())

 
            elif hasattr(model, 't5') or hasattr(model, 'generate'):
                self.logger.warning("Falling back to native generate, bypassing custom beam_search.")
                
                
        except Exception as e:
            self.logger.warning(f"Error in _generate_with_scores: {e}")
            predictions = [[""] * self.num_beams for _ in range(batch_size)]
            probs = [[1.0 / self.num_beams] * self.num_beams for _ in range(batch_size)]
            scores = [[0.0] * self.num_beams for _ in range(batch_size)]
        
        return {
            'predictions': predictions,
            'probs': probs,
            'scores': scores
        }
    
    def _get_targets_from_batch(self, batch):
         
        targets = []

         
        if 'target_items' in batch and hasattr(self.tokenizer, 'get_target_sid_string'):
            target_items_batch = batch['target_items']

            for item_list in target_items_batch:
                sid_list = []

                if item_list is None:
                    targets.append(sid_list)
                    continue
 
                if hasattr(item_list, 'tolist'):
                    item_list = item_list.tolist()

                for item_id in item_list:
                    try:
                        item_id = int(item_id)
                    except Exception:
                        continue

                    sid_str = self.tokenizer.get_target_sid_string(item_id)
                    if sid_str:
                        sid_list.append(sid_str)

                targets.append(sid_list)

            return targets

 
        if 'target_item' in batch and hasattr(self.tokenizer, 'get_target_sid_string'):
            target_items = batch['target_item']

            if hasattr(target_items, 'tolist'):
                target_items = target_items.tolist()

            for item_id in target_items:
                try:
                    item_id = int(item_id)
                except Exception:
                    targets.append("")
                    continue

                sid_str = self.tokenizer.get_target_sid_string(item_id)
                targets.append(sid_str if sid_str else "")

            return targets
 
        if 'labels' in batch:
            labels = batch['labels']

            for i in range(labels.shape[0]):
                if labels.dim() == 2:
                    label_tokens = labels[i].tolist()
                else:
                    label_tokens = [labels[i].item()]

                label_tokens = [t for t in label_tokens if t != -100]
                sid_str = self.tokenizer.decode_to_sid(label_tokens)
                targets.append(sid_str)

            return targets

        # =========================
        # fallback
        # =========================
        batch_size = batch['input_ids'].shape[0]
        return [""] * batch_size

    
    def _calculate_basic_metrics(self, predictions: List[List[str]], targets: List[str]) -> Dict[str, Any]:
         
        import math
        
        num_samples = len(predictions)
        if num_samples == 0:
            return {'HR': [0], 'NDCG': [0], 'MRR': [0], 'valid_topk': [1]}
        
        n_beam = len(predictions[0]) if predictions[0] else 0
        valid_topk = [k for k in self.topk_list if k <= n_beam]
        if not valid_topk:
            valid_topk = [max(1, n_beam)]
        
        ALLHR = np.zeros(len(valid_topk))
        ALLNDCG = np.zeros(len(valid_topk))
        ALLMRR = np.zeros(len(valid_topk))
        
        for preds, target in zip(predictions, targets):
            if not target:
                continue
                
            minID = 1000000
            for j, pred in enumerate(preds):
                if pred == target:
                    minID = j
                    break
            
            for idx, k in enumerate(valid_topk):
                if minID < k:
                    ALLHR[idx] += 1
                    ALLNDCG[idx] += 1 / math.log2(minID + 2)
                    ALLMRR[idx] += 1 / (minID + 1)

        ndcg_list = (ALLNDCG / num_samples).tolist()
        
        primary_score = ndcg_list[2] if len(ndcg_list) > 2 else (ndcg_list[0] if ndcg_list else 0.0)

        return {
            'HR': (ALLHR / num_samples).tolist(),
            'NDCG': (ALLNDCG / num_samples).tolist(),
            'MRR': (ALLMRR / num_samples).tolist(),
            'valid_topk': valid_topk,
            'primary_score': primary_score
        }
    
    def _log_metrics_to_wandb(self, metrics: Dict[str, Any], prefix: str, round_num: int, epoch: int = None):
 
        if not self.use_wandb:
            return
        
        log_dict = {f'{prefix}/round': round_num}
        if epoch is not None:
            log_dict[f'{prefix}/epoch'] = epoch
        
        valid_topk = metrics.get('valid_topk', self.topk_list)
        
        for metric_name in ['HR', 'NDCG', 'MRR', 'Recall', 'Coverage', 'Gini']:
            if metric_name in metrics and metrics[metric_name]:
                values = metrics[metric_name]
                for idx, k in enumerate(valid_topk):
                    if idx < len(values):
                        log_dict[f'{prefix}/{metric_name}@{k}'] = values[idx]
        
        for key in ['loss', 'Avg_Intra_List_Diversity', 'primary_score']:
            if key in metrics:
                log_dict[f'{prefix}/{key}'] = metrics[key]
        
        if 'Avg_Path_Diversity' in metrics and metrics['Avg_Path_Diversity']:
            log_dict[f'{prefix}/avg_path_diversity'] = np.mean(metrics['Avg_Path_Diversity'])
        if 'Avg_Layer_Unique_Tokens' in metrics and metrics['Avg_Layer_Unique_Tokens']:
            log_dict[f'{prefix}/avg_layer_unique'] = np.mean(metrics['Avg_Layer_Unique_Tokens'])
        if 'Avg_Layer_Entropy' in metrics and metrics['Avg_Layer_Entropy']:
            log_dict[f'{prefix}/avg_layer_entropy'] = np.mean(metrics['Avg_Layer_Entropy'])
        
        self._log_wandb(log_dict, step=self.global_step_ref())

    def _print_evaluation_results(self, metrics: Dict[str, Any], round_num: int):
 
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Round {round_num} Evaluation Results")
        self.logger.info(f"{'='*60}")
        
        valid_topk = metrics.get('valid_topk', [])
        
 
        self.logger.info("Accuracy Metrics:")
        for idx, k in enumerate(valid_topk):
            hr = metrics['HR'][idx] if idx < len(metrics.get('HR', [])) else 0
            ndcg = metrics['NDCG'][idx] if idx < len(metrics.get('NDCG', [])) else 0
            mrr = metrics['MRR'][idx] if idx < len(metrics.get('MRR', [])) else 0
            self.logger.info(f"  Top-{k}: HR={hr:.4f}, NDCG={ndcg:.4f}, MRR={mrr:.4f}")
        
        if 'Coverage' in metrics and metrics['Coverage']:
            self.logger.info("Diversity Metrics:")
            for idx, k in enumerate(valid_topk):
                if idx < len(metrics['Coverage']):
                    cov = metrics['Coverage'][idx]
                    gini = metrics['Gini'][idx] if idx < len(metrics.get('Gini', [])) else 0
                    self.logger.info(f"  Top-{k}: Coverage={cov:.4f}, Gini={gini:.4f}")
         
        if 'Avg_Intra_List_Diversity' in metrics:
            self.logger.info(f"  Avg Intra-List Diversity (ILD): {metrics['Avg_Intra_List_Diversity']:.4f}")
         
        if 'Avg_Path_Diversity' in metrics and metrics['Avg_Path_Diversity']:
            avg_path_div = np.mean(metrics['Avg_Path_Diversity'])
            self.logger.info(f"  Avg Path Diversity: {avg_path_div:.4f}")
         
        if 'Avg_Layer_Entropy' in metrics and metrics['Avg_Layer_Entropy']:
            avg_entropy = np.mean(metrics['Avg_Layer_Entropy'])
            self.logger.info(f"  Avg Layer Entropy: {avg_entropy:.4f}")
         
        self.logger.info(f"Other Info:")
        self.logger.info(f"  Loss: {metrics.get('loss', 0):.4f}")
        self.logger.info(f"  Num Samples: {metrics.get('num_samples', 0)}")
        self.logger.info(f"  Num Beams: {metrics.get('n_beam', 0)}")
        
        self.logger.info(f"{'='*60}\n")

    def _save_best_model(self):
         
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        output_dir = self.config.get('output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        self.saved_model_ckpt = os.path.join(output_dir, 'best_model_in_training.pt')
        
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
        else:
            torch.save(self.model.state_dict(), self.saved_model_ckpt)
        
        self.logger.info(f"Best model saved to {self.saved_model_ckpt}")



    def _save_evaluation_results(self, metrics: Dict[str, Any], round_num: int, split: str = 'test'):
         
        import json
        from datetime import datetime
        
        output_dir = self.config.get('output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        eval_file = os.path.join(output_dir, f'eval_results_round_{round_num}_{split}.json')
        
        results_to_save = {
            'round': round_num,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_beams': self.num_beams,
                'topk_list': self.topk_list,
                'max_depth': self.max_depth,
            },
            'metrics': metrics
        }
        
        with open(eval_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {eval_file}")
        
        return eval_file
    
    def evaluate(self, test_loader, round_num=1) -> Dict[str, Any]:
         
        metrics = self._evaluate_with_full_metrics(test_loader, round_num, split='test')
        
        
        self._print_evaluation_results(metrics, round_num)
        
        
        if self.use_wandb:
            self._log_metrics_to_wandb(metrics, prefix='test', round_num=round_num)
         
        self._save_evaluation_results(metrics, round_num, split='test')
        
        return metrics

    def _gather_predictions(self, predictions: List) -> List:
        
        if self.accelerator is None:
            return predictions
        
         
        import pickle
        local_data = pickle.dumps(predictions)
        local_tensor = torch.tensor(list(local_data), dtype=torch.uint8, device=self._get_device())
        
         
        local_size = torch.tensor([len(local_data)], device=self._get_device())
        all_sizes = self.accelerator.gather(local_size)
        max_size = all_sizes.max().item()
         
        padded = torch.zeros(max_size, dtype=torch.uint8, device=self._get_device())
        padded[:len(local_data)] = local_tensor
         
        all_padded = self.accelerator.gather(padded)
         
        if self.accelerator.is_main_process:
            all_predictions = []
            world_size = self.accelerator.num_processes
            for i in range(world_size):
                size = all_sizes[i].item()
                data = bytes(all_padded[i * max_size: i * max_size + size].cpu().tolist())
                preds = pickle.loads(data)
                all_predictions.extend(preds)
            return all_predictions
        return []

    def _gather_list(self, data: List) -> List:
         
        if self.accelerator is None:
            return data
        
        import pickle
        local_data = pickle.dumps(data)
        local_tensor = torch.tensor(list(local_data), dtype=torch.uint8, device=self._get_device())
        
        local_size = torch.tensor([len(local_data)], device=self._get_device())
        all_sizes = self.accelerator.gather(local_size)
        max_size = all_sizes.max().item()
        
        padded = torch.zeros(max_size, dtype=torch.uint8, device=self._get_device())
        padded[:len(local_data)] = local_tensor
        all_padded = self.accelerator.gather(padded)
        
        if self.accelerator.is_main_process:
            result = []
            world_size = self.accelerator.num_processes
            for i in range(world_size):
                size = all_sizes[i].item()
                chunk = bytes(all_padded[i * max_size: i * max_size + size].cpu().tolist())
                result.extend(pickle.loads(chunk))
            return result
        return []

    def _gather_losses(self, losses: List[float]) -> List[float]:
     
        if self.accelerator is None:
            return losses
        
        loss_tensor = torch.tensor(losses, device=self._get_device())
        all_losses = self.accelerator.gather(loss_tensor)
        
        if self.accelerator.is_main_process:
            return all_losses.cpu().tolist()
        return []
    
    def _to_python_list(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist") and not isinstance(value, list):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    def _extract_batch_metadata(self, batch):
        meta = {}
        meta["sample_idx"] = self._to_python_list(batch.get("sample_idx"))
        meta["target_item"] = self._to_python_list(batch.get("target_item"))

        behavior_labels = {}
        for col in BEHAVIOR_COLS:
            if col in batch:
                behavior_labels[col] = self._to_python_list(batch[col])

        meta["behavior_labels"] = behavior_labels
        return meta


    def evaluate_with_outputs(self, data_loader, round_num=1, split='test'):
    
        inference_path = os.path.join(
            self.output_dir, 'inference_outputs',
            f'inference_{split}_r{round_num}.pt'
        )

        self._inference_and_save(data_loader, inference_path, round_num, split)
 
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
 
        if self.accelerator is None or self.accelerator.is_main_process:
            metrics = self._calculate_metrics_from_file(inference_path)
            outputs = torch.load(inference_path, map_location="cpu")
            outputs["metrics"] = metrics
        else:
            metrics = {}
            outputs = {}

 
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        return metrics, outputs



    def _calculate_metrics(self, predictions, targets):
        
        from eval_metrics import (
            calculate_metrics_from_model_outputs,
            calculate_multi_target_metrics_from_model_outputs,
        )

        topk_list = [1, 5, 10, 20, 50]

        is_multi_target = (
            isinstance(targets, list)
            and len(targets) > 0
            and isinstance(targets[0], list)
        )

        if is_multi_target:
            return calculate_multi_target_metrics_from_model_outputs(
                predictions=predictions,
                targets=targets,
                topk_list=topk_list,
            )

        item_list = None
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "get_all_item_sid_strings"):
            try:
                item_list = self.tokenizer.get_all_item_sid_strings()
            except Exception:
                item_list = None

        return calculate_metrics_from_model_outputs(
            predictions=predictions,
            targets=targets,
            item_list=item_list,
            topk_list=topk_list,
            max_depth=8,
        )


    def _calculate_metrics(self, predictions, targets):
         
        from eval_metrics import (
            calculate_metrics_from_model_outputs,
            calculate_multi_target_metrics_from_model_outputs,
        )

        topk_list = [1, 5, 10, 20, 50]

        is_multi_target = (
            isinstance(targets, list)
            and len(targets) > 0
            and isinstance(targets[0], list)
        )

        if is_multi_target:
            metrics = calculate_multi_target_metrics_from_model_outputs(
                predictions=predictions,
                targets=targets,
                topk_list=topk_list,
            )

            self.logger.info(
                f"[full_user eval] multi-target metrics: "
                f"NDCG={metrics.get('NDCG')}, "
                f"HR={metrics.get('HR')}, "
                f"Recall={metrics.get('Recall')}, "
                f"MRR={metrics.get('MRR')}"
            )
            return metrics

        item_list = None
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "get_all_item_sid_strings"):
            try:
                item_list = self.tokenizer.get_all_item_sid_strings()
            except Exception as e:
                self.logger.warning(f"Failed to get item_list from tokenizer: {e}")
                item_list = None

        metrics = calculate_metrics_from_model_outputs(
            predictions=predictions,
            targets=targets,
            item_list=item_list,
            topk_list=topk_list,
            max_depth=8,
        )

        self.logger.info(
            f"[single-target eval] metrics: "
            f"NDCG={metrics.get('NDCG')}, "
            f"HR={metrics.get('HR')}, "
            f"Recall={metrics.get('Recall')}, "
            f"MRR={metrics.get('MRR')}"
        )
        return metrics

    def _log_wandb(self, metrics: dict, step=None):
        
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)

    def _inference_and_save(self, data_loader, output_path: str, round_num=1, split='test', save_interval=500):
         
        self.model.eval()
        device = self._get_device()
        model_to_use = self._get_unwrapped_model()
        is_distributed = self.accelerator is not None
        is_main_process = (not is_distributed) or self.accelerator.is_main_process

        process_index = self.accelerator.process_index if is_distributed else 0

        temp_dir = output_path + '.tmp'
        process_temp_dir = os.path.join(temp_dir, f'process_{process_index}')

        if is_main_process and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

        if is_distributed:
            self.accelerator.wait_for_everyone()

        os.makedirs(process_temp_dir, exist_ok=True)

        batch_predictions = []
        batch_targets = []
        batch_probs = []
        batch_scores = []
        batch_sample_idx = []
        batch_target_items = []
        batch_behavior_labels = {col: [] for col in BEHAVIOR_COLS}
        all_losses = []
        chunk_idx = 0

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(
                    data_loader,
                    desc=f"Inference ({split}) GPU{process_index}",
                    disable=not is_main_process
                )
            ):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                outputs = self.model(batch)
                all_losses.append(outputs.loss.item())

                gen_outputs = self._generate_with_scores(batch, model_to_use)

                current_predictions = gen_outputs['predictions']
                current_probs = gen_outputs['probs']
                current_scores = gen_outputs['scores']
                current_targets = self._get_targets_from_batch(batch)

                batch_predictions.extend(current_predictions)
                batch_probs.extend(current_probs)
                batch_scores.extend(current_scores)
                batch_targets.extend(current_targets)

                meta = self._extract_batch_metadata(batch)

                if meta["sample_idx"] is None:
                    raise ValueError("sample_idx not found in batch during inference")

                batch_sample_idx.extend(meta["sample_idx"])

                if meta["target_item"] is not None:
                    batch_target_items.extend(meta["target_item"])
                else:
                    batch_target_items.extend([None] * len(current_predictions))

                for col in BEHAVIOR_COLS:
                    values = meta["behavior_labels"].get(col)
                    if values is None:
                        batch_behavior_labels[col].extend([0] * len(current_predictions))
                    else:
                        batch_behavior_labels[col].extend(values)

                if (i + 1) % save_interval == 0:
                    chunk_path = os.path.join(process_temp_dir, f'chunk_{chunk_idx}.pt')
                    torch.save({
                        "predictions": batch_predictions,
                        "targets": batch_targets,
                        "probs": batch_probs,
                        "scores": batch_scores,
                        "sample_idx": batch_sample_idx,
                        "target_item": batch_target_items,
                        "behavior_labels": batch_behavior_labels,
                        "losses": all_losses,
                    }, chunk_path)
                    self.log(f"GPU{process_index} saved chunk {chunk_idx} ({len(batch_predictions)} samples)")

                    batch_predictions = []
                    batch_targets = []
                    batch_probs = []
                    batch_scores = []
                    batch_sample_idx = []
                    batch_target_items = []
                    batch_behavior_labels = {col: [] for col in BEHAVIOR_COLS}
                    all_losses = []
                    chunk_idx += 1

        if batch_predictions:
            chunk_path = os.path.join(process_temp_dir, f'chunk_{chunk_idx}.pt')
            torch.save({
                "predictions": batch_predictions,
                "targets": batch_targets,
                "probs": batch_probs,
                "scores": batch_scores,
                "sample_idx": batch_sample_idx,
                "target_item": batch_target_items,
                "behavior_labels": batch_behavior_labels,
                "losses": all_losses,
            }, chunk_path)

        if is_distributed:
            self.accelerator.wait_for_everyone()
            self.log(f"GPU{process_index}: All GPUs finished inference")

        if is_main_process:
            merged_rows = []
            merged_losses = []

            num_processes = self.accelerator.num_processes if is_distributed else 1
            for proc_idx in range(num_processes):
                proc_dir = os.path.join(temp_dir, f'process_{proc_idx}')
                if not os.path.exists(proc_dir):
                    continue

                chunk_files = sorted([f for f in os.listdir(proc_dir) if f.startswith('chunk_')])
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(proc_dir, chunk_file)
                    try:
                        chunk = torch.load(chunk_path, map_location="cpu")
                    except Exception as e:
                        self.log(f"Warning: Failed to load {chunk_path}: {e}", level='warning')
                        continue

                    merged_losses.extend(chunk.get("losses", []))

                    n = len(chunk["predictions"])
                    sample_idx_list = chunk.get("sample_idx")
                    target_items = chunk.get("target_item", [None] * n)
                    behavior_labels = chunk.get("behavior_labels", {})

                    if sample_idx_list is None:
                        raise ValueError(f"sample_idx missing in chunk: {chunk_path}")

                    for i in range(n):
                        row = {
                            "sample_idx": sample_idx_list[i],
                            "prediction": chunk["predictions"][i],
                            "target": chunk["targets"][i],
                            "prob": chunk["probs"][i],
                            "score": chunk["scores"][i],
                            "target_item": target_items[i] if i < len(target_items) else None,
                            "behavior_labels": {},
                        }

                        for col in BEHAVIOR_COLS:
                            values = behavior_labels.get(col)
                            row["behavior_labels"][col] = values[i] if values is not None and i < len(values) else 0

                        merged_rows.append(row)

            merged_rows.sort(key=lambda x: x["sample_idx"])

            final_outputs = {
                "round_num": round_num,
                "split": split,
                "sample_idx": [x["sample_idx"] for x in merged_rows],
                "target_item": [x["target_item"] for x in merged_rows],
                "predictions": [x["prediction"] for x in merged_rows],
                "targets": [x["target"] for x in merged_rows],
                "behavior_labels": {
                    col: [x["behavior_labels"][col] for x in merged_rows]
                    for col in BEHAVIOR_COLS
                },
                "loss": sum(merged_losses) / max(len(merged_losses), 1),
            }


            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(final_outputs, output_path)

            import shutil
            shutil.rmtree(temp_dir)
            self.log(f"Saved to {output_path} ({len(merged_rows)} samples from {num_processes} GPUs)")

        if is_distributed:
            self.accelerator.wait_for_everyone()


    
    def log(self, message, level='info'):
         
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

    def _calculate_metrics_from_file(self, inference_path: str) -> Dict[str, Any]:
         
        outputs = torch.load(inference_path)
        metrics = self._calculate_metrics(outputs['predictions'], outputs['targets'])
        metrics['loss'] = outputs.get('loss', 0.0)
        metrics['num_samples'] = len(outputs['predictions'])
        return metrics