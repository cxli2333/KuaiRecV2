# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Streaming training pipeline with wandb support
Support daily or periodic incremental training of recommendation models
"""

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
from eval_metrics import calculate_metrics_from_model_outputs

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
    """
    Streaming training pipeline
    
    Supported formats:
    - Load data by day
    - Accumulate user histories
    - Continual training, training from scratch, or training from checkpoints
    - Automatic saving and restoration
    - wandb visualization
    """
    
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
        accumulate_history: bool = True,
        max_train_samples_per_user: int = 999,
        training_mode: str = 'continual',
        trainer_class = None,
        use_wandb: bool = True,
        wandb_project: str = 'tiger-streaming',
        wandb_run_name: str = None,
        wandb_tags: List[str] = None,
    ):
        """
        Initialize the streaming training pipeline.
        """
        self.logger = getLogger()
        
        # Initialize the accelerator if available
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
        self.trainer_class = trainer_class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
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
        self.accumulated_user_sequences = {}
        
        self.training_results = []
        self.best_val_score = float('-inf')
        self.best_round = 0
        
        # Global step for wandb
        self.global_step = 0
        
        self.data_config = None
        self.interaction_files = []
        self.item_files = []
        self.rounds = []
        
        if json_config_path:
            self._load_json_config()
        
        # Initialize wandb
        if self.use_wandb:
            self._init_wandb()
        
        # Print configuration
        self._log_config()
    
    def _init_wandb(self):
        """Initialize wandb."""
        if not WANDB_AVAILABLE:
            self.log("wandb not available, skipping initialization", level='warning')
            return
        
        try:
            # Prepare wandb configuration
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
            

            if self.accelerator is None or self.accelerator.is_main_process:
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
        """Record a log message."""
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
        """Load the JSON configuration file."""
        if not os.path.exists(self.json_config_path):
            raise FileNotFoundError(f"Config file not found: {self.json_config_path}")
        
        with open(self.json_config_path, 'r') as f:
            self.data_config = json.load(f)
        
        # Get file lists; support two formats
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
        """Build training rounds."""
        self.rounds = []
        
        for i, interaction_file in enumerate(self.interaction_files):
            start_idx = i
            end_idx = i + self.item_history_days
            
            if end_idx > len(self.item_files):
                self.log(f"Warning: Round {i+1} skipped - needs items[{start_idx}:{end_idx}], "
                        f"but only {len(self.item_files)} item files available.", 
                        level='warning')
                continue
            
            item_files_for_round = self.item_files[start_idx:end_idx]
            
            self.rounds.append({
                'round': i + 1,
                'interaction_file': interaction_file,
                'item_files': item_files_for_round
            })
        
        self.log(f"Built {len(self.rounds)} training rounds")
        self.log(f"  - Item history days: {self.item_history_days}")
        
        if self.rounds:
            r = self.rounds[0]
            item_names = [os.path.basename(f) for f in r['item_files']]
            self.log(f"  - Round 1: {os.path.basename(r['interaction_file'])}")
            self.log(f"    Items: {item_names[:3]}... ({len(item_names)} files)")
            
            if len(self.rounds) > 1:
                r = self.rounds[-1]
                item_names = [os.path.basename(f) for f in r['item_files']]
                self.log(f"  - Round {r['round']}: {os.path.basename(r['interaction_file'])}")
                self.log(f"    Items: {item_names[:3]}... ({len(item_names)} files)")
    
    def print_training_plan(self):
        """Print daily input files without loading data or creating a model."""
        for round_info in self.rounds:
            self.log(f"Round {round_info['round']}: {round_info['interaction_file']}")
            for item_file in round_info['item_files']:
                self.log(f"  Item file: {item_file}")

    def _log_config(self):
        """Print configuration information."""
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
    
    def _get_files_for_round(self, round_num: int) -> Tuple[List[str], List[str]]:
        """Get file paths for the specified round."""
        if round_num < 1 or round_num > len(self.rounds):
            raise ValueError(f"Invalid round number: {round_num}")
        
        round_info = self.rounds[round_num - 1]
        return [round_info['interaction_file']], round_info['item_files']
    
    def _load_data_for_round(self, round_num: int, splits:List[str] = None) -> Dict[str, Any]:
        """Load data for the specified round."""
        if splits is None:
            splits = ['train', 'val', 'test']
        interaction_files, item_files = self._get_files_for_round(round_num)
        
        self.log(f"Round {round_num}: Loading data...")
        self.log(f"  - Interaction: {os.path.basename(interaction_files[0])}")
        self.log(f"  - Items: {len(item_files)} files")
        
        for i, f in enumerate(item_files):
            self.log(f"    [{i}] {os.path.basename(f)}")
        
        if StreamingInteractionDataset is None:
            raise ImportError("StreamingInteractionDataset not found. Please check your imports.")
        
        self.raw_dataset = StreamingInteractionDataset(
            config={
                **self.config,
                'max_train_samples_per_user': self.max_train_samples_per_user,
                'debug': os.environ.get("KUAIRECV2_LOG_SAMPLES", "0") == "1"
            },
            task=self.task,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len,
            max_history_len=self.max_history_len,
            accumulate_history=self.accumulate_history,
            splits=splits,
        )
        
        datasets = self.raw_dataset.load_from_files(
            interaction_files=interaction_files,
            item_files=item_files,
            accumulated_user_sequences=self.accumulated_user_sequences if self.accumulate_history else None
        )
        
        self.log(f"Round {round_num}: Dataset loaded")
        self.log(f"  - Num items: {self.raw_dataset.num_items}")
        self.log(f"  - Num users: {self.raw_dataset.num_users}")
        self.log(f"  - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
        
        if os.environ.get("KUAIRECV2_LOG_SAMPLES", "0") == "1" and datasets['train']:
            sample = datasets['train'][0]
            self.log(f"  - Sample: user={sample.get('user')}, seq_len={len(sample.get('seq', []))}, target={sample.get('target_item')}")
        
        if self.accumulate_history:
            self.accumulated_user_sequences = self.raw_dataset.get_accumulated_sequences()
        
        if self.accelerator.is_main_process and self.use_wandb and self.wandb_run:
            self._log_wandb({
                'data/num_items': self.raw_dataset.num_items,
                'data/num_users': self.raw_dataset.num_users,
                'data/train_samples': len(datasets['train']),
                'data/val_samples': len(datasets['val']),
                'data/test_samples': len(datasets['test']),
                'round': round_num,
            }, step=self.global_step)
        
        return datasets
    
    def _initialize_model_and_tokenizer(self, round_num: int):
        """Initialize or update the model and tokenizer."""
        self.log(f"Round {round_num}: Initializing tokenizer and model...")
        
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
            try:
                from tokenizer.streaming_tokenizer import StreamingTokenizer
                self.tokenizer = StreamingTokenizer(
                    self.config,
                    self.raw_dataset,
                    n_digits=self.config.get('n_digits', 8),
                    codebook_size=self.config.get('codebook_size', 512)
                )
            except ImportError:
                raise ImportError("StreamingTokenizer not found. Please provide a tokenizer class.")
        
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.config['padding_token'] = self.tokenizer.padding_token
        self.config['sid_offset'] = getattr(self.tokenizer, 'sid_offset', 0)
        self.config['max_token_seq_len'] = getattr(self.tokenizer, 'max_token_seq_len', 512)
        
        self.log(f"  - Vocab size: {self.config['vocab_size']}")
        
        if self.training_mode == 'from_scratch':
            self._create_new_model()
        elif self.training_mode == 'from_checkpoint':
            if round_num == 1:
                self._create_new_model()
                if self.checkpoint_path:
                    self._load_checkpoint(self.checkpoint_path, round_num)
            else:
                prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                self._create_new_model()
                if os.path.exists(prev_ckpt):
                    self._load_checkpoint(prev_ckpt, round_num - 1)
        else:  # continual
            if self.model is None:
                self._create_new_model()
                if self.checkpoint_path:
                    self._load_checkpoint(self.checkpoint_path, round_num)
        
        self._initialize_trainer()
    
    def _create_new_model(self):
        """Create a new model."""
        self.log("Creating new model...")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized before creating model")

        model_size = self.config.get('model_size', 'small')
        vocab_size = self.config.get('vocab_size', 4096)
        self.config['max_token_seq_len'] = self.config.get('max_token_seq_len', 512)
        self.config['sid_offset'] = self.config.get('sid_offset', 7)
        self.config['n_digits'] = self.config.get('n_digits', 8)
        self.config['padding_token'] = self.tokenizer.padding_token
        
        tiger_configs = {
            'tiny': {
                'num_layers': 4, 
                'num_decoder_layers': 4,
                'd_model': 128,
                'd_ff': 1024,
                'num_heads': 6,
                'd_kv': 64,
                'dropout_rate': 0.1,
                'activation_function': 'relu',
                'feed_forward_proj': 'relu',
            },
            'small': {
                'num_layers': 6, 
                'num_decoder_layers': 6,
                'd_model': 512,
                'd_ff': 1024,
                'num_heads': 8,
                'd_kv': 64,
                'dropout_rate': 0.1,
                'activation_function': 'relu',
                'feed_forward_proj': 'relu',
            },
            'base': {
                'num_layers': 12, 
                'num_decoder_layers': 12,
                'd_model': 768,
                'd_ff': 3072,
                'num_heads': 12,
                'd_kv': 64,
                'dropout_rate': 0.1,
                'activation_function': 'relu',
                'feed_forward_proj': 'relu',
            },
            'large': {
                'num_layers': 24, 
                'num_decoder_layers': 24,
                'd_model': 1024,
                'd_ff': 4096,
                'num_heads': 16,
                'd_kv': 64,
                'dropout_rate': 0.1,
                'activation_function': 'relu',
                'feed_forward_proj': 'relu',
            },
        }
        
        model_config = tiger_configs.get(model_size, tiger_configs['small'])
        self.config.update(model_config)
        self.config['vocab_size'] = vocab_size
        
        try:
            from model.TIGER.model import TIGER
            self.model = TIGER(self.config, self.tokenizer)
            self.log(f"Using TIGER model (T5-based)")
            self.log(f"  - num_layers: {model_config['num_layers']}")
            self.log(f"  - d_model: {model_config['d_model']}")
            self.log(f"  - num_heads: {model_config['num_heads']}")
        except ImportError as e:
            self.log(f"TIGER import failed: {e}", level='warning')
            self.log("Using default Transformer model", level='warning')
            self.model = self._create_default_model(vocab_size, model_config)
        except Exception as e:
            self.log(f"TIGER initialization failed: {e}", level='warning')
            self.log("Using default Transformer model", level='warning')
            self.model = self._create_default_model(vocab_size, model_config)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model created: {model_size}, Parameters: {num_params:,}")
        
        # Log model information to wandb
        if self.use_wandb and self.wandb_run:
            wandb.config.update({
                'model_params': num_params,
                'vocab_size': vocab_size,
                **model_config
            }, allow_val_change=True)
    
    def _create_default_model(self, vocab_size: int, config: dict):
        """Create a fallback Transformer model."""
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
        """Load a checkpoint containing the model and tokenizer."""
        if not os.path.exists(path):
            self.log(f"Checkpoint not found: {path}", level='warning')
            return
        
        self.log(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        self.config.update(checkpoint.get('model_config', {}))

        if 'tokenizer_state' in checkpoint:
            from tokenizer.streaming_tokenizer import StreamingTokenizer
            tokenizer_class = self.tokenizer_class if isinstance(self.tokenizer_class, type) else StreamingTokenizer
            self.tokenizer = tokenizer_class.from_state_dict(checkpoint['tokenizer_state'], self.config)
        elif 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        elif self.tokenizer is None:
            # Rebuild mappings only when they were not loaded for this round.
            self.log("Rebuilding tokenizer from round data")
            self._load_data_for_round(round_num, ['train'])
            self._initialize_tokenizer_only(round_num)

        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.config['padding_token'] = self.tokenizer.padding_token
        self.config['sid_offset'] = self.tokenizer.sid_offset
        self.config['max_token_seq_len'] = self.tokenizer.max_token_seq_len

        # Recreate the model to avoid loading into a previously wrapped instance.
        self._initialize_model_only()
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Support legacy checkpoints containing a plain state_dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)

    
    def _save_checkpoint(self, path: str):
        """Save a checkpoint containing the model and tokenizer."""
        # Save only on the main process
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
        # Save the model and tokenizer
        torch.save({
            'model_state_dict': model_state,
            'tokenizer_config': tokenizer_config, 
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'tokenizer_state': self.tokenizer.state_dict(),
            'model_config': {
                key: self.config[key]
                for key in ('model_size', 'n_digits', 'codebook_size')
                if key in self.config
            },
        }, path)
        
    def _initialize_trainer(self):
        """Initialize the trainer."""
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
        """Set the global step."""
        self.global_step = step
    
    def _run_single_round(self, round_num: int) -> Dict[str, Any]:
        """Run one round with support for resuming from checkpoints."""
        print("run_single_round start")
        
        if self.raw_dataset is not None:
            self.raw_dataset._accumulated_loaded = False
        from torch.utils.data import DataLoader
        
        self.log(f"\n{'='*60}")
        self.log(f"Round {round_num}/{len(self.rounds)}")
        self.log(f"{'='*60}")
        
        train_loader = None
        val_loader = None
        test_loader = None
        tokenized_train_count = 0
        tokenized_val_count = 0
        tokenized_test_count = 0
        
        # ========== Checkpoint path definitions ==========
        round_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num}.pt")
        eval_output_path = os.path.join(self.output_dir, f"eval_outputs_round_{round_num}.pt")
        eval_metrics_path = os.path.join(self.output_dir, f"eval_metrics_round_{round_num}.json")
        val_metrics_path = os.path.join(self.output_dir, f"val_metrics_round_{round_num}.json")

        # ========== Check whether training should be skipped ==========
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
            # 2. Initialize the tokenizer and model
            self._initialize_tokenizer_only(round_num)

            if skip_training:
                self.log(f"Round {round_num}: Loading checkpoint...")
                self._load_checkpoint(round_ckpt_path, round_num)
            else:
                # Decide whether to create a new model based on training_mode
                if self.training_mode == 'from_scratch':
                    # Start each round from scratch
                    self._create_new_model()
                elif self.training_mode == 'from_checkpoint':
                    if round_num == 1:
                        self._create_new_model()
                        if self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path, round_num)
                    else:
                        prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                        self._create_new_model()
                        if os.path.exists(prev_ckpt):
                            self._load_checkpoint(prev_ckpt, round_num - 1)
                else:  # continual
                    # Create a model only in the first round or when no model exists
                    if self.model is None:
                        self._create_new_model()
                        # Try loading the previous round's checkpoint to resume training
                        if round_num > 1:
                            prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num - 1}.pt")
                            if os.path.exists(prev_ckpt):
                                self._load_checkpoint(prev_ckpt, round_num - 1)
                                self.log(f"Loaded checkpoint from round {round_num - 1} for continual training")
                            elif self.checkpoint_path:
                                self._load_checkpoint(self.checkpoint_path, round_num)
                                self.log(f"Loaded initial checkpoint for continual training")
                        elif self.checkpoint_path:
                            self._load_checkpoint(self.checkpoint_path, round_num)
                            self.log(f"Loaded initial checkpoint for continual training")

            # Refresh mappings for items introduced in the current round.
            self._initialize_tokenizer_only(round_num)
            self._initialize_trainer()

            # 3. Tokenize
            if need_train_data or need_val_data or need_test_data:
                tokenized_data = self.tokenizer.tokenize(datasets)

                tokenized_train_count = len(tokenized_data['train']) if tokenized_data['train'] else 0
                tokenized_val_count = len(tokenized_data['val']) if tokenized_data['val'] else 0
                tokenized_test_count = len(tokenized_data['test']) if tokenized_data['test'] else 0
                
                self.log(f"Tokenized: train={tokenized_train_count}, val={tokenized_val_count}, test={tokenized_test_count}")

                print(f"if need_train_data and tokenized_data.get('train'):{need_train_data and tokenized_data.get('train') is not None}")

                if need_train_data and tokenized_data.get('train'):
                    train_loader = DataLoader(
                        tokenized_data['train'],
                        batch_size=self.config.get('train_batch_size', 64),
                        shuffle=True,
                        collate_fn=self.tokenizer.collate_fn['train']
                    )

                print(f"if need_val_data and tokenized_data.get('val'):{need_val_data and tokenized_data.get('val') is not None}")
                if need_val_data and tokenized_data.get('val'):
                    val_loader = DataLoader(
                        tokenized_data['val'],
                        batch_size=self.config.get('eval_batch_size', 128),
                        shuffle=False,
                        collate_fn=self.tokenizer.collate_fn['val']
                    ) if tokenized_val_count > 0 else None
                print(f"if need_test_data and tokenized_data.get('test'):{need_val_data and tokenized_data.get('val') is not None}")
                if need_test_data and tokenized_data.get('test'):
                    test_loader = DataLoader(
                        tokenized_data['test'],
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
                    
        # ========== 5. Training ==========
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
            # Save the checkpoint
            if self.trainer.accelerator is not None:
                self.trainer.accelerator.wait_for_everyone()
            self._save_checkpoint(round_ckpt_path)

            # Save validation metrics returned by fit without repeating inference
            if val_results:
                with open(val_metrics_path, 'w') as f:
                    json.dump(val_results, f, indent=2)
                self.log(f"Round {round_num}: Val metrics saved to {val_metrics_path}")

        
        # ========== 6. Validation evaluation ==========
        if not val_results:
            if skip_val_metrics and os.path.exists(val_metrics_path):
                with open(val_metrics_path, 'r') as f:
                    val_results = json.load(f)
                best_val_score = val_results.get('primary_score', 0)
            elif skip_training and val_loader is not None:
                # Run separate validation only when training is skipped and validation metrics are missing
                val_results = self.trainer._evaluate_with_full_metrics(val_loader, round_num=round_num, split='val')
                best_val_score = val_results.get('primary_score', 0)
                with open(val_metrics_path, 'w') as f:
                    json.dump(val_results, f, indent=2)
        
        # ========== 7. Test evaluation ==========
        test_results = {}
        print(f"if skip_eval_metrics:{skip_eval_metrics is not None}")
        print(f"elif test_loader is not None:{test_loader is not None}")
        print(f"if skip_eval_output:{skip_eval_output is not None}")

        if skip_eval_metrics:
            self.log(f"Round {round_num}: Loading existing test metrics...")
            with open(eval_metrics_path, 'r') as f:
                test_results = json.load(f)
        elif test_loader is not None:
            # Check for saved prediction outputs
            
            if skip_eval_output:
                self.log(f"Round {round_num}: Loading existing eval outputs...")
                eval_outputs = torch.load(eval_output_path)
                test_results = self._calculate_metrics_from_outputs(eval_outputs)
            else:
                self.log(f"Round {round_num}: Evaluating on test set...")
                test_results, eval_outputs = self.trainer.evaluate_with_outputs(
                    test_loader, round_num=round_num, split='test'
                )
                # Save prediction outputs
                torch.save(eval_outputs, eval_output_path)
                self.log(f"Round {round_num}: Eval outputs saved to {eval_output_path}")

            if self.trainer.accelerator is not None:
                self.trainer.accelerator.wait_for_everyone()
                self.log(f"Round {round_num}: All GPUs finished test evaluation")
            # Save test metrics
            with open(eval_metrics_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            self.log(f"Round {round_num}: Test metrics saved to {eval_metrics_path}")

        # ========== 8. Save results ==========
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
        
        # ========== 9. Update the best model ==========
        print(f"判断条件 'best_val_score > self.best_val_score' 的结果: {best_val_score > self.best_val_score}")

        if best_val_score > self.best_val_score:
            self.best_val_score = best_val_score
            self.best_round = round_num
            best_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")
            
            # Copy best_model_in_training.pt saved by fit(), rather than saving the current model
            # The current self.model is from the last epoch, not necessarily the best epoch
            best_model_in_training = None
            if self.trainer is not None:
                best_model_in_training = self.trainer.saved_model_ckpt
            
            if best_model_in_training and os.path.exists(best_model_in_training):
                import shutil
                shutil.copy2(best_model_in_training, best_ckpt_path)
                self.log(f"New best model saved! Round {round_num}, Val score: {best_val_score:.4f}")
            elif os.path.exists(round_ckpt_path):
                # Fallback: copy the round checkpoint if best_model_in_training.pt is unavailable
                import shutil
                shutil.copy2(round_ckpt_path, best_ckpt_path)
                self.log(f"New best model saved (from round ckpt)! Round {round_num}, Val score: {best_val_score:.4f}")
            else:
                self.log(f"Warning: Cannot save best model, no checkpoint available")

        self.log(f"Round {round_num}: Complete")
        self.log(f"  - Best val score: {best_val_score:.4f}")
        if test_results:
            self.log(f"  - Test results: {test_results}")
        
        print("run_single_round end")

        if self.trainer is not None and self.trainer.accelerator is not None:
            self.trainer.accelerator.wait_for_everyone()

        if 'tokenized_data' in locals():
            del tokenized_data
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        if test_loader is not None:
            del test_loader
        
        import gc
        gc.collect()
        
        return round_result

    # def recalculate_metrics_from_file(self, eval_output_path: str) -> Dict[str, Any]:
    # """Recalculate metrics from saved eval_outputs files.
        
    #     Args:
    # eval_output_path: path to the eval_outputs file
            
    #     Returns:
    # Recalculated metric dictionary
    #     """
    #     if not os.path.exists(eval_output_path):
    #         self.log(f"Error: File not found: {eval_output_path}")
    #         return {}
        
    #     self.log(f"Loading eval outputs from {eval_output_path}")
    #     eval_outputs = torch.load(eval_output_path)
        
        
        
    #     self.log(f"Recalculating metrics for {len(predictions)} samples...")
    #     metrics = self._calculate_metrics(predictions, targets)
        
    # # Update metrics in eval_outputs
    #     eval_outputs['metrics'] = metrics
    #     torch.save(eval_outputs, eval_output_path)
    #     self.log(f"Updated metrics saved to {eval_output_path}")
        
    #     return metrics

    def _calculate_metrics_from_file(self, inference_path: str) -> Dict[str, Any]:
        """Calculate metrics from a file."""
        outputs = torch.load(inference_path)
        metrics = self._calculate_metrics(outputs['predictions'], outputs['targets'])
        metrics['loss'] = outputs.get('loss', 0.0)
        metrics['num_samples'] = len(outputs['predictions'])
        return metrics
    
    def _save_training_results(self):
        """Save training results to a JSON file."""
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
        
        # Save to wandb
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
        """
        Generate a training summary.
        Support averaging nested dictionary metrics such as 'TopK_Layer_Diversity'.
        """
        if not self.training_results:
            return {'error': 'No training results'}
        
        valid_results = [r for r in self.training_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid training results'}
        
        # 1. Aggregate basic scalar metrics
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
            # Use the first round's keys as the reference
            test_keys = test_results_list[0].keys()
            
            for key in test_keys:
                # values contains one metric's values across all rounds
                # For example, TopK_Layer_Diversity: [Dict_Round1, Dict_Round2, ...]
                values = [r.get(key) for r in test_results_list if key in r]
                
                if not values:
                    continue

                try:
                    first_val = values[0]
                    
                    # --- Case A: Dictionary values ---
                    # Handle "TopK_Layer_Diversity"
                    if isinstance(first_val, dict):
                        avg_dict = {}
                        # Iterate over dictionary subkeys such as 'Top1' and 'Top5'
                        for sub_key in first_val.keys():
                            # Collect each subkey's values across rounds
                            # Handle subkeys that may be missing in some rounds
                            sub_values = [v.get(sub_key) for v in values if isinstance(v, dict) and sub_key in v]
                            
                            if not sub_values:
                                continue
                                
                            # Determine the subvalue type and calculate its mean
                            sub_first = sub_values[0]
                            if isinstance(sub_first, (int, float, np.number)):
                                avg_dict[sub_key] = float(np.mean(sub_values))
                            elif isinstance(sub_first, list) and len(sub_first) > 0 and isinstance(sub_first[0], (int, float, np.number)):
                                mean_arr = np.mean(sub_values, axis=0)
                                avg_dict[sub_key] = mean_arr.tolist()
                        
                        # Store the averaged dictionary in summary
                        summary[f'avg_test_{key}'] = avg_dict

                    # --- Case B: Numeric values (int/float) ---
                    elif isinstance(first_val, (int, float, np.number)):
                        summary[f'avg_test_{key}'] = float(np.mean(values))
                        
                    # --- Case C: Numeric lists ---
                    elif isinstance(first_val, list):
                        if len(first_val) > 0 and isinstance(first_val[0], (int, float, np.number)):
                            mean_val = np.mean(values, axis=0)
                            if isinstance(mean_val, np.ndarray):
                                summary[f'avg_test_{key}'] = mean_val.tolist()
                            else:
                                summary[f'avg_test_{key}'] = mean_val
                        else:
                            pass # Skip non-numeric lists
                    
                    else:
                        pass # Skip other types

                except Exception as e:
                    self.log(f"Warning: Could not calculate average for metric '{key}': {e}", level='warning')
        
        # 3. Print logs with basic dictionary formatting
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
                # For nested metrics such as TopK_Layer_Diversity, print only Top1/Top5 examples
                keys_show = list(v.keys())[:3]
                self.log(f"  {k}: Dict containing {keys_show}...")
            else:
                self.log(f"  {k}: {v}")
        self.log("="*60 + "\n")
        
        return summary

    def _log_wandb(self, metrics: dict, step=None):
        """Log to wandb only on the main process."""
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)

    def _initialize_model_only(self):
        """Initialize only the model, not the tokenizer, when loading checkpoints."""
        self.log("Initializing model only (without tokenizer)...")
        
        # Read required configuration from the checkpoint
        # Otherwise use default configuration
        if 'vocab_size' not in self.config:
            # Use defaults or values from the configuration file
            self.config['vocab_size'] = self.config.get('default_vocab_size', 4103)
            self.config['padding_token'] = 0
            self.config['sid_offset'] = self.config.get('sid_offset', 7)
            self.config['max_token_seq_len'] = self.config.get('max_token_seq_len', 512)
        
        self._create_new_model()

    def _initialize_tokenizer_only(self, round_num: int):
        """Initialize only the tokenizer when the model already exists."""
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
        
        # Update config; these values should already exist if the model is initialized
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.config['padding_token'] = self.tokenizer.padding_token


    def run(self, start_round: int = 1) -> Dict[str, Any]:
        """Run the complete streaming training pipeline."""
        self.log("\n" + "="*60)
        self.log("Starting Streaming Training")
        self.log("="*60 + "\n")
        
        total_rounds = len(self.rounds)
        if start_round < 1 or start_round > max(total_rounds, 1):
            raise ValueError(f"start_round must be between 1 and {max(total_rounds, 1)}")
        self.log(f"Total rounds: {total_rounds}")
        
        # Log training start
        if self.use_wandb and self.wandb_run:
            self._log_wandb({
                'training/total_rounds': total_rounds,
                'training/status': 'started'
            }, step=0)
        
        for round_num in range(start_round, total_rounds + 1):
            try:
                round_result = self._run_single_round(round_num)
                
                # Log round completion
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
                
                # Log errors to wandb
                if self.use_wandb and self.wandb_run:
                    self._log_wandb({
                        'training/error_round': round_num,
                        'training/error_message': str(e)
                    }, step=self.global_step)
                continue
        
        # Save training results
        self._save_training_results()
        
        # Generate a summary
        summary = self._generate_summary()
        
        # Log training completion to wandb
        if self.use_wandb and self.wandb_run:
            self._log_wandb({
                'training/status': 'completed',
                'training/final_best_round': self.best_round,
                'training/final_best_val_score': self.best_val_score,
            }, step=self.global_step)
            
            # Update the summary
            wandb.run.summary.update(summary)
            
            # Finish the wandb run
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
        """Resume training from the specified round."""
        total_rounds = len(self.rounds)
        if from_round < 1 or from_round > total_rounds:
            raise ValueError(f"from_round must be between 1 and {total_rounds}, got {from_round}")
        
        self.log(f"Resuming training from round {from_round}")
        
        # In continual mode, load the previous checkpoint first to restore model state
        if self.training_mode == 'continual' and from_round > 1:
            prev_ckpt = os.path.join(self.output_dir, f"{self.model_name}_round_{from_round - 1}.pt")
            if os.path.exists(prev_ckpt):
                self._load_checkpoint(prev_ckpt, from_round - 1)
                self.log(f"Loaded model from round {from_round - 1} for continual training")
            else:
                self.log(f"Warning: No checkpoint found for round {from_round - 1}, starting with fresh model")
        
        return self.run(start_round=from_round)

    def evaluate_only(self, round_num: int = None) -> Dict[str, Any]:
        """Evaluate a round's checkpoint without training; default to the last round."""
        if round_num is None:
            round_num = len(self.rounds)
        
        round_ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_round_{round_num}.pt")
        if not os.path.exists(round_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {round_ckpt_path}")
        
        # Load test data before rebuilding legacy checkpoint mappings.
        datasets = self._load_data_for_round(round_num, ['test'])
        self._initialize_tokenizer_only(round_num)
        self._load_checkpoint(round_ckpt_path, round_num)
        self._initialize_trainer()
        
        tokenized_data = self.tokenizer.tokenize(datasets)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            tokenized_data['test'],
            batch_size=self.config.get('eval_batch_size', 8),
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )
        if self.trainer.accelerator is not None:
            test_loader = self.trainer.accelerator.prepare(test_loader)
        
        test_results = self.trainer._evaluate_with_full_metrics(
            test_loader, round_num=round_num, split='test'
        )
        
        self.log(f"Evaluation results for round {round_num}: {test_results}")
        return {'round': round_num, 'test_results': test_results}

    def evaluate_all_rounds(self) -> Dict[str, Any]:
        """Evaluate checkpoints from all rounds."""
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


class DefaultTrainer:
    """Default trainer with wandb support and full evaluation metrics."""
    
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

        # wandb settings
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.global_step_ref = global_step_ref or (lambda: 0)
        self.global_step_setter = global_step_setter or (lambda x: None)
        self.local_step = 0
        
        # Evaluation configuration
        self.num_beams = config.get('num_beams', 50)
        self.topk_list = config.get('topk_list', [1, 5, 10, 20, 50])
        self.max_depth = config.get('max_depth', 8)
        
        # Cache SID strings for all items
        self._item_sid_list = None
    
    def _get_item_sid_list(self) -> list:
        """Get SID strings for all items."""
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
        """Get the device."""
        if self.accelerator is not None:
            return self.accelerator.device
        return next(self.model.parameters()).device
    
    def _get_unwrapped_model(self):
        """Get the unwrapped model."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model
    
    def fit(self, train_loader, val_loader, epochs=None, round_num=1):
        """Train the model."""
        if epochs is None:
            epochs = self.config.get('epochs', 10)
        
        lr = self.config.get('lr', 5e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        
        if self.accelerator is not None:
            # Check whether the model is already prepared to avoid wrapping it twice
            if not getattr(self.model, '_is_accelerate_prepared', False):
                self.model, optimizer, train_loader, scheduler = \
                    self.accelerator.prepare(self.model, optimizer, train_loader, scheduler)
                self.model._is_accelerate_prepared = True
            else:
                # Prepare only the optimizer, loader, and scheduler when the model is already prepared
                optimizer, train_loader, scheduler = \
                    self.accelerator.prepare(optimizer, train_loader, scheduler)
            if val_loader is not None:
                val_loader = self.accelerator.prepare(val_loader)

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
            
            # Validation
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
        """Evaluate using the shared inference logic."""
        inference_path = os.path.join(
            self.output_dir, 'inference_outputs',
            f'inference_{split}_r{round_num}_e{epoch}.pt'
        )
        
        # Step 1: Run inference and save outputs
        self._inference_and_save(data_loader, inference_path, round_num, split)
        
        # Step 2: Calculate metrics
        metrics = self._calculate_metrics_from_file(inference_path)
        
        # Cache for use by other functions
        outputs = torch.load(inference_path)
        self._last_predictions = outputs['predictions']
        self._last_targets = outputs['targets']
        self._last_probs = outputs['probs']
        self._last_scores = outputs['scores']
        
        return metrics

    def _generate_with_scores(self, batch, model) -> Dict:
        """Generate predictions and return probabilities and scores."""
        predictions = []
        probs = []
        scores = []
        batch_size = batch['input_ids'].shape[0]

        # if not hasattr(self, '_debug_gen_input'):
        #     self._debug_gen_input = True
            
        # # Print the first sample's input
        #     input_ids_0 = batch['input_ids'][0].cpu().tolist()
        #     attn_mask_0 = batch['attention_mask'][0].cpu().tolist()
            
        #     if 'target_item' in batch:
        #         target_item = batch['target_item'][0].item() if torch.is_tensor(batch['target_item'][0]) else batch['target_item'][0]
        #         if target_item in self.tokenizer.item2sid:
        #             target_sid_tokens = self.tokenizer.item2sid[target_item]
        #             target_sid_str = self.tokenizer.decode_to_sid(target_sid_tokens)

        try:
            if hasattr(model, 't5'):
                # T5 model generation
                outputs = model.t5.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.config.get('n_digits', 8) + 1,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                sequences = outputs.sequences  # [batch * num_beams, seq_len]
                seq_scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None

                
                if not hasattr(self, '_debug_gen_output'):
                    self._debug_gen_output = True
                    print(f"\n{'='*80}")
                    print(f"DEBUG _generate_with_scores OUTPUT")
                    print(f"{'='*80}")
                    print(f"Generated sequences shape: {sequences.shape}")
                    print(f"Has sequence scores: {seq_scores is not None}")
                    if seq_scores is not None:
                        print(f"Sequence scores shape: {seq_scores.shape}")
                        print(f"Sequence scores (first 5): {seq_scores[:5].cpu().tolist()}")
                    print(f"{'='*80}\n")


                for i in range(batch_size):
                    start_idx = i * self.num_beams
                    end_idx = start_idx + self.num_beams
                    
                    sample_seqs = sequences[start_idx:end_idx, 1:]  # Remove BOS
                    sample_preds = []
                    for seq in sample_seqs:
                        tokens = seq.tolist()

                        # if len(predictions) == 0:  # Print only the first sample
                        #     self.log(f"DEBUG: Generated tokens: {tokens}")
                        #     self.log(f"DEBUG: Token in vocab: {[t in self.tokenizer.id2token for t in tokens]}")

                        sid_str = self.tokenizer.decode_to_sid(tokens)

                        # if len(predictions) == 0:
                        #     self.log(f"DEBUG: Decoded SID: '{sid_str}'")

                        sample_preds.append(sid_str)
                    predictions.append(sample_preds)
                    
                    if seq_scores is not None:
                        sample_scores = seq_scores[start_idx:end_idx]
                        sample_probs = torch.softmax(sample_scores, dim=0)
                        probs.append(sample_probs.cpu().tolist())
                        scores.append(sample_scores.cpu().tolist())
                    else:
                        probs.append([1.0 / self.num_beams] * self.num_beams)
                        scores.append([0.0] * self.num_beams)
            
            elif hasattr(model, 'generate'):
                generated = model.generate(batch, n_return_sequences=self.num_beams)
                
                for i in range(batch_size):
                    if generated.dim() == 3:
                        sample_preds = []
                        for j in range(min(self.num_beams, generated.shape[1])):
                            tokens = generated[i, j].tolist()
                            sid_str = self.tokenizer.decode_to_sid(tokens)
                            sample_preds.append(sid_str)
                        predictions.append(sample_preds)
                    else:
                        tokens = generated[i].tolist()
                        sid_str = self.tokenizer.decode_to_sid(tokens)
                        predictions.append([sid_str])
                    
                    probs.append([1.0 / self.num_beams] * len(predictions[-1]))
                    scores.append([0.0] * len(predictions[-1]))
            else:
                # Fallback: use logits
                predictions = self._generate_predictions(batch, model)
                probs = [[1.0 / len(p)] * len(p) for p in predictions]
                scores = [[0.0] * len(p) for p in predictions]
                
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
    
    def _get_targets_from_batch(self, batch) -> List[str]:
        """Get target SID strings from a batch."""
        targets = []
        
        if 'target_item' in batch and hasattr(self.tokenizer, 'get_target_sid_string'):
            target_items = batch['target_item']
            if hasattr(target_items, 'tolist'):
                target_items = target_items.tolist()
            
            for item_id in target_items:
                sid_str = self.tokenizer.get_target_sid_string(item_id)
                targets.append(sid_str)
        elif 'labels' in batch:
            labels = batch['labels']
            for i in range(labels.shape[0]):
                if labels.dim() == 2:
                    label_tokens = labels[i].tolist()
                else:
                    label_tokens = [labels[i].item()]
                
                label_tokens = [t for t in label_tokens if t != -100]
                sid_str = self.tokenizer.decode_to_sid(label_tokens)
                targets.append(sid_str)
        else:
            batch_size = batch['input_ids'].shape[0]
            targets = [""] * batch_size
        
        return targets
    
    def _calculate_basic_metrics(self, predictions: List[List[str]], targets: List[str]) -> Dict[str, Any]:
        """Calculate basic metrics."""
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
        """Log metrics to wandb."""
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
        """Print evaluation results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Round {round_num} Evaluation Results")
        self.logger.info(f"{'='*60}")
        
        valid_topk = metrics.get('valid_topk', [])
        
        # Print accuracy metrics
        self.logger.info("Accuracy Metrics:")
        for idx, k in enumerate(valid_topk):
            hr = metrics['HR'][idx] if idx < len(metrics.get('HR', [])) else 0
            ndcg = metrics['NDCG'][idx] if idx < len(metrics.get('NDCG', [])) else 0
            mrr = metrics['MRR'][idx] if idx < len(metrics.get('MRR', [])) else 0
            self.logger.info(f"  Top-{k}: HR={hr:.4f}, NDCG={ndcg:.4f}, MRR={mrr:.4f}")
        
        # Print diversity metrics
        if 'Coverage' in metrics and metrics['Coverage']:
            self.logger.info("Diversity Metrics:")
            for idx, k in enumerate(valid_topk):
                if idx < len(metrics['Coverage']):
                    cov = metrics['Coverage'][idx]
                    gini = metrics['Gini'][idx] if idx < len(metrics.get('Gini', [])) else 0
                    self.logger.info(f"  Top-{k}: Coverage={cov:.4f}, Gini={gini:.4f}")
        
        # Print ILD
        if 'Avg_Intra_List_Diversity' in metrics:
            self.logger.info(f"  Avg Intra-List Diversity (ILD): {metrics['Avg_Intra_List_Diversity']:.4f}")
        
        # Print path diversity
        if 'Avg_Path_Diversity' in metrics and metrics['Avg_Path_Diversity']:
            avg_path_div = np.mean(metrics['Avg_Path_Diversity'])
            self.logger.info(f"  Avg Path Diversity: {avg_path_div:.4f}")
        
        # Print layer entropy
        if 'Avg_Layer_Entropy' in metrics and metrics['Avg_Layer_Entropy']:
            avg_entropy = np.mean(metrics['Avg_Layer_Entropy'])
            self.logger.info(f"  Avg Layer Entropy: {avg_entropy:.4f}")
        
        # Print other information
        self.logger.info(f"Other Info:")
        self.logger.info(f"  Loss: {metrics.get('loss', 0):.4f}")
        self.logger.info(f"  Num Samples: {metrics.get('num_samples', 0)}")
        self.logger.info(f"  Num Beams: {metrics.get('n_beam', 0)}")
        
        self.logger.info(f"{'='*60}\n")

    def _save_best_model(self):
        """Save the best model."""
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
        """Save evaluation results to a JSON file."""
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
        """Evaluate the model."""
        metrics = self._evaluate_with_full_metrics(test_loader, round_num, split='test')
        
        # 1. Print to the console
        self._print_evaluation_results(metrics, round_num)
        
        # 2. Log to wandb
        if self.use_wandb:
            self._log_metrics_to_wandb(metrics, prefix='test', round_num=round_num)
        
        # 3. Save to a JSON file
        self._save_evaluation_results(metrics, round_num, split='test')
        
        return metrics

    def _gather_predictions(self, predictions: List) -> List:
        """Gather predictions from all processes."""
        if self.accelerator is None:
            return predictions
        
        # Serialize with pickle before gathering
        import pickle
        local_data = pickle.dumps(predictions)
        local_tensor = torch.tensor(list(local_data), dtype=torch.uint8, device=self._get_device())
        
        # Get data lengths from all processes
        local_size = torch.tensor([len(local_data)], device=self._get_device())
        all_sizes = self.accelerator.gather(local_size)
        max_size = all_sizes.max().item()
        
        # Pad to equal lengths
        padded = torch.zeros(max_size, dtype=torch.uint8, device=self._get_device())
        padded[:len(local_data)] = local_tensor
        
        # Gather all data
        all_padded = self.accelerator.gather(padded)
        
        # Decode only on the main process
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
        """Gather list data from all processes."""
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
        """Gather losses from all processes."""
        if self.accelerator is None:
            return losses
        
        loss_tensor = torch.tensor(losses, device=self._get_device())
        all_losses = self.accelerator.gather(loss_tensor)
        
        if self.accelerator.is_main_process:
            return all_losses.cpu().tolist()
        return []

    def evaluate_with_outputs(self, data_loader, round_num=1, split='test'):
        """Evaluate and return complete outputs."""
        # Reuse _evaluate_with_full_metrics without an epoch argument
        inference_path = os.path.join(
            self.output_dir, 'inference_outputs',
            f'inference_{split}_r{round_num}.pt'
        )
        self._inference_and_save(data_loader, inference_path, round_num, split)
        metrics = self._calculate_metrics_from_file(inference_path)
        
        outputs = torch.load(inference_path)
        return metrics, {
            'predictions': outputs['predictions'],
            'targets': outputs['targets'],
            'probs': outputs['probs'],
            'scores': outputs['scores'],
            'metrics': metrics
        }

    def _calculate_metrics_from_outputs(self, eval_outputs):
        """Calculate metrics from saved outputs."""
        # Return existing metrics directly
        if 'metrics' in eval_outputs:
            return eval_outputs['metrics']
        
        # Otherwise recalculate from predictions and targets
        predictions = eval_outputs.get('predictions', [])
        targets = eval_outputs.get('targets', [])
        
        if not predictions or not targets:
            self.logger.warning("No predictions or targets found in eval_outputs")
            return {}
        
        return self._calculate_metrics(predictions, targets)

    def _calculate_metrics(self, predictions: List[List[str]], targets: List[str]) -> Dict[str, Any]:
        """Calculate full evaluation metrics."""
        # Prefer the full metric calculation from eval_metrics
        if EVAL_METRICS_AVAILABLE:
            try:
                item_list = self._get_item_sid_list() if hasattr(self, '_get_item_sid_list') else None
                metrics = calculate_metrics_from_model_outputs(
                    predictions=predictions,
                    targets=targets,
                    item_list=item_list,
                    topk_list=self.topk_list,
                )
                return metrics
            except Exception as e:
                self.logger.warning(f"Error in calculate_metrics_from_model_outputs: {e}")
        
        # Fall back to basic metrics
        return self._calculate_basic_metrics(predictions, targets)

    def _log_wandb(self, metrics: dict, step=None):
        """Log to wandb only on the main process."""
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=step)

    def _inference_and_save(self, data_loader, output_path: str, round_num=1, split='test', save_interval=500):
        """Run inference and save in chunks to avoid exhausting memory."""
        self.model.eval()
        device = self._get_device()
        model_to_use = self._get_unwrapped_model()
        is_distributed = self.accelerator is not None
        is_main_process = not is_distributed or self.accelerator.is_main_process
        
        # Get the process ID
        process_index = self.accelerator.process_index if is_distributed else 0
        
        # Temporary paths with a separate directory for each process
        temp_dir = output_path + '.tmp'
        process_temp_dir = os.path.join(temp_dir, f'process_{process_index}')
        
        # Remove stale files
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
        all_losses = []
        chunk_idx = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f"Inference ({split}) GPU{process_index}", disable=not is_main_process)):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model(batch)
                all_losses.append(outputs.loss.item())
                
                gen_outputs = self._generate_with_scores(batch, model_to_use)
                batch_predictions.extend(gen_outputs['predictions'])
                batch_probs.extend(gen_outputs['probs'])
                batch_scores.extend(gen_outputs['scores'])
                batch_targets.extend(self._get_targets_from_batch(batch))

                if os.environ.get("KUAIRECV2_LOG_SAMPLES", "0") == "1" and i == 0 and is_main_process:
                    print(f"\n{'='*80}")
                    print(f"DEBUG MATCHING (First Batch)")
                    print(f"{'='*80}")
                    
                    current_preds = gen_outputs['predictions']
                    current_targets = self._get_targets_from_batch(batch)
                    
                    for idx in range(min(3, len(current_preds))):
                        pred_list = current_preds[idx]
                        target = current_targets[idx]
                        
                        print(f"\nSample {idx}:")
                        print(f"  Target: {target}")
                        print(f"  Target len: {len(target)}, repr: {repr(target)}")
                        
                        for beam_idx in range(min(3, len(pred_list))):
                            pred = pred_list[beam_idx]
                            match = (pred == target)
                            print(f"  Pred[{beam_idx}]: {pred}")
                            print(f"    Len: {len(pred)}, repr: {repr(pred)}, Match: {match}")
                            
                            # Compare characters
                            if not match and pred and target:
                                for char_idx in range(min(len(pred), len(target))):
                                    if pred[char_idx] != target[char_idx]:
                                        print(f"    First diff at pos {char_idx}: '{pred[char_idx]}' vs '{target[char_idx]}'")
                                        break
                    
                    print(f"{'='*80}\n")
                
                # Save independently in each process
                if (i + 1) % save_interval == 0:
                    chunk_path = os.path.join(process_temp_dir, f'chunk_{chunk_idx}.pt')
                    torch.save({
                        'predictions': batch_predictions,
                        'targets': batch_targets,
                        'probs': batch_probs,
                        'scores': batch_scores
                    }, chunk_path)
                    self.log(f"GPU{process_index} saved chunk {chunk_idx} ({len(batch_predictions)} samples)")
                    
                    batch_predictions = []
                    batch_targets = []
                    batch_probs = []
                    batch_scores = []
                    chunk_idx += 1
        
        # Save the final batch
        if batch_predictions:
            chunk_path = os.path.join(process_temp_dir, f'chunk_{chunk_idx}.pt')
            torch.save({
                'predictions': batch_predictions,
                'targets': batch_targets,
                'probs': batch_probs,
                'scores': batch_scores
            }, chunk_path)
        
        # Wait for all processes to finish inference
        if is_distributed:
            self.accelerator.wait_for_everyone()
            self.log(f"GPU{process_index}: All GPUs finished inference")
        # Merge results from all processes on the main process
        if is_main_process:
            all_predictions = []
            all_targets = []
            all_probs = []
            all_scores = []
            
            # Iterate over all process directories
            num_processes = self.accelerator.num_processes if is_distributed else 1
            for proc_idx in range(num_processes):
                proc_dir = os.path.join(temp_dir, f'process_{proc_idx}')
                if not os.path.exists(proc_dir):
                    continue
                
                # Load all chunks for this process
                chunk_files = sorted([f for f in os.listdir(proc_dir) if f.startswith('chunk_')])
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(proc_dir, chunk_file)
                    try:
                        chunk = torch.load(chunk_path)
                        all_predictions.extend(chunk['predictions'])
                        all_targets.extend(chunk['targets'])
                        all_probs.extend(chunk['probs'])
                        all_scores.extend(chunk['scores'])
                    except Exception as e:
                        self.log(f"Warning: Failed to load {chunk_path}: {e}", level='warning')
            
            # Save the final result
            torch.save({
                'predictions': all_predictions,
                'targets': all_targets,
                'probs': all_probs,
                'scores': all_scores,
                'loss': sum(all_losses) / max(len(all_losses), 1),
                'round_num': round_num,
                'split': split
            }, output_path)
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir)
            self.log(f"Saved to {output_path} ({len(all_predictions)} samples from {num_processes} GPUs)")
            
        if is_distributed:
            self.accelerator.wait_for_everyone()

    
    def log(self, message, level='info'):
        """Log output"""
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
        """Calculate metrics from a file."""
        outputs = torch.load(inference_path)
        metrics = self._calculate_metrics(outputs['predictions'], outputs['targets'])
        metrics['loss'] = outputs.get('loss', 0.0)
        metrics['num_samples'] = len(outputs['predictions'])
        return metrics