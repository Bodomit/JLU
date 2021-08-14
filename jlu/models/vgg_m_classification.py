from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .vgg_m import VggMBase


class VggMClassification(pl.LightningModule):
    def __init__(self, n_classes: int, base: Optional[VggMBase] = None) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.fc_classification = nn.Linear(4096, n_classes)

        if base is not None:
            self.base = base
        else:
            self.base = VggMBase()

    def forward(self, x):
        x = self.base(x)
        x = self.fc_classification(x)
        return x
