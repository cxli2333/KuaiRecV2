# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from pipeline import Pipeline
from utils import parse_command_line_args
from tokenizer.tokenizer import VanillaTokenizer, IdentTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RPG', help='Model name')
    parser.add_argument('--dataset', type=str, default='KSID', choices=['KSID', 'Amazon2014'], help='Dataset name')
    parser.add_argument('--category', type=str, default=None, help='Amazon sub-dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    return parser.parse_known_args()


if __name__ == '__main__':
    
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        category=args.category,
        tokenizer=VanillaTokenizer,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    pipeline.run()
