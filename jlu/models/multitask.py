from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .vgg_m import VggMBase


class JLUMultitaskModel(pl.LightningModule):
    def __init__(self, n_classes: List[int], feature_length=4096):
        super().__init__()
        self.feature_length = feature_length

        assert len(n_classes) >= 2
        n_primary, *n_secondaries = n_classes

        self.feature_base = VggMBase()
        self.primary_task = nn.Linear(self.feature_length, n_primary)
        self.secondary_tasks = self.construct_secondaries(n_secondaries)

    def construct_secondaries(self, n_classes: List[int]) -> nn.ModuleList:
        secondary_tasks: List[nn.Linear] = []
        for n in n_classes:
            secondary_task = nn.Linear(self.feature_length, n)
            secondary_tasks.append(secondary_task)
        return nn.ModuleList(secondary_tasks)

    def forward(self, x):
        features = self.feature_base(x)
        primary_output = self.primary_task(features)
        secondary_outputs: List[torch.Tensor] = []
        for secondary_task in self.secondary_tasks:
            secondary_output = secondary_task(features)
            secondary_outputs.append(secondary_output)

        return primary_output, *secondary_outputs
