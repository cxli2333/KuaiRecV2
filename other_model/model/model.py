# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from tokenizer.tokenizer import AbstractTokenizer


class AbstractModel(nn.Module):
    def __init__(self, config: dict, tokenizer: AbstractTokenizer):
        super(AbstractModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def calculate_loss(self, batch):
        raise NotImplementedError('calculate_loss method must be implemented.')

    def generate_samples(self, batch, n_return_sequences=1):
        raise NotImplementedError('generate_samples method must be implemented.')
