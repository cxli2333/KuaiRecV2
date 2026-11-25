# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset.dataset import AbstractDataset
from model.model import AbstractModel
from tokenizer.tokenizer import AbstractTokenizer, VanillaTokenizer
from utils import get_config, init_seed, init_logger, init_device, \
                    get_dataset, get_tokenizer, get_model, get_trainer, log
                    
import numpy as np

def build_train_mask_from_split(raw_dataset, split_datasets):
    """
    Build train_mask from split_datasets['train'].
    - item_id starts at 1; 0 is PAD.
    - mask[i-1] == True means item_id == i appears in the training set.
    
    Args:
        raw_dataset: SeqAugDataset or a similar dataset with n_items/num_items/id_mapping
        split_datasets: dictionary containing at least 'train' with seq and next columns
    
    Returns:
        np.ndarray(bool) shape=(n_items,)
    """
    # ---------- Get the total item count ----------
    n_items = getattr(raw_dataset, 'n_items', None)
    if n_items is None:
        n_items = getattr(raw_dataset, 'num_items', None)
    if n_items is None and hasattr(raw_dataset, 'id_mapping'):
        id2item = raw_dataset.id_mapping.get('id2item', [])
        n_items = len(id2item) - 1
    assert n_items is not None and n_items > 0, \
        "Cannot infer n_items from dataset."

    mask = np.zeros((n_items,), dtype=bool)
    train_set = split_datasets['train']

    # ---------- Iterate over the training set ----------
    for ex in train_set:
        # Historical sequence seq
        seq = ex.get('seq', [])
        if hasattr(seq, 'tolist'):
            seq = seq.tolist()
        for iid in seq:
            iid = int(iid)
            if iid > 0:  # Skip PAD=0
                mask[iid - 1] = True

        # Next target item
        tgt = ex.get('next', None)
        if tgt is not None:
            try:
                tgt = int(tgt)
            except Exception:
                tgt = int(getattr(tgt, 'item', lambda: tgt)())
            if tgt > 0:
                mask[tgt - 1] = True

    print(f"[build_train_mask_from_split] True={mask.sum()} / {n_items} "
            f"({mask.sum()/n_items:.2%} of items in train)")
    return mask

class Pipeline:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            category=category,
            config_file=config_file,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device() 
        self.checkpoint_path = checkpoint_path

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        # Dataset
        self.raw_dataset = get_dataset(dataset_name, self.config)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            if isinstance(tokenizer, type):
                # pid2sid requires an additional argument
                #self.tokenizer = tokenizer(self.config, self.raw_dataset)
                self.tokenizer = tokenizer(self.config, self.raw_dataset, n_digits=self.config['n_digits'], codebook_size=self.config['codebook_size'])
            else:
                self.tokenizer = tokenizer
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            tokenizer_class = get_tokenizer(model_name)
            self.tokenizer = tokenizer_class(self.config, self.raw_dataset)
        
        self.config['vocab_size'] = self.tokenizer.vocab_size
        self.config['padding_token'] = self.tokenizer.padding_token
        self.config['sid_offset'] = self.tokenizer.sid_offset
        self.config['max_token_seq_len'] = self.tokenizer.max_token_seq_len
            
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        # self.log(f"model_name = {model_name} get_model = {get_model(model_name)}")

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.tokenizer)
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.config['device']))
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader
        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        # Run a single-batch overfitting experiment first
        #self.trainer.overfit_one_batch(train_dataloader, steps=500, log_every=50)
        best_epoch, best_val_score = self.trainer.fit(train_dataloader, val_dataloader)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        if self.checkpoint_path is None:
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process and self.checkpoint_path is None:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        # Enable graph-constrained decoding for model inference
        self.trainer.model.generate_w_decoding_graph = True
        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()
        return {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'test_results': test_results
        }

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
