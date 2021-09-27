from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from jlu.models.resnet import ResNetBase
from torch.nn.modules.module import Module

from .vgg_m import VggMBase


class JLUMultitaskModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: List[int],
        pretrained_base: Optional[pl.LightningModule],
        dropout=0.5,
        feature_model: str = "vggm"
    ):
        super().__init__()
        self.feature_model = feature_model

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
            if self.feature_model == "vggm":
                self.feature_base.fully_connected[3].p = dropout  # type: ignore
                self.feature_base.fully_connected[5].p = dropout  # type: ignore
            elif feature_model == "resnet50":
                self.feature_base.fully_connected[0].p = dropout  # type: ignore

        else:
            if self.feature_model == "vggm":
                self.feature_base = VggMBase(dropout=dropout)
            elif self.feature_model == "resnet50":
                self.feature_base = ResNetBase(dropout=dropout, resnet_type="resnet50")
            else:
                raise ValueError(f"feature_model: {self.feature_model}")

        assert isinstance(self.feature_base.FEATURE_LENGTH, int)
        self.primary_task = nn.Linear(self.feature_base.FEATURE_LENGTH, n_primary)
        self.secondary_tasks = self.construct_secondaries(n_secondaries)

    def construct_secondaries(self, n_classes: List[int]) -> nn.ModuleList:
        assert isinstance(self.feature_base.FEATURE_LENGTH, int)

        secondary_tasks: List[nn.Linear] = []
        for n in n_classes:
            secondary_task = nn.Linear(self.feature_base.FEATURE_LENGTH, n)
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
