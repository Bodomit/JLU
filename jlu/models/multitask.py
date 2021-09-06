from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from .vgg_m import VggMBase


class JLUMultitaskModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: List[int],
        pretrained_base: Optional[pl.LightningModule],
        dropout=0.5,
        feature_length=4096,
    ):
        super().__init__()
        self.feature_length = feature_length

        try:
            assert len(n_classes) >= 2
            n_primary, *n_secondaries = n_classes
        except AssertionError:
            n_primary = n_classes[0]
            n_secondaries = []

        if pretrained_base:
            self.feature_base = pretrained_base

            assert isinstance(self.feature_base.features, Module)

            # Freeze all the CNN layers of the pretrained model.
            for p in self.feature_base.features.parameters():
                p.requires_grad = False

            # Reset all the parameters in the fully connected layers.
            assert isinstance(self.feature_base.fully_connected, Module)
            for c in self.feature_base.fully_connected.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()  # type: ignore

            # Set the dropout layers in the pretrained model to use new probability.
            self.feature_base.fully_connected[3].p = dropout  # type: ignore
            self.feature_base.fully_connected[5].p = dropout  # type: ignore
        else:
            self.feature_base = VggMBase(dropout=dropout)
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
