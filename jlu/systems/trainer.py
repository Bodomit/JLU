from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.models import JLUMultitaskModel
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torchmetrics.functional import accuracy


class Trainer(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        primary_task: str,
        datamodule_n_classes: np.ndarray,
        datamodule_labels: List[str],
        pretrained: str,
        pretrained_base: Optional[pl.LightningModule],
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.primary_task = primary_task
        self.datamodule_n_classes = datamodule_n_classes

        # Get label indexes per task.
        self.primary_idx: int = datamodule_labels.index(self.primary_task)
        all_idx: List[int] = [self.primary_idx]

        # Get the model.
        n_classes = self.datamodule_n_classes[all_idx]
        self.model = JLUMultitaskModel(n_classes, pretrained_base, dropout=dropout)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = F.cross_entropy(y_, y)
        acc = accuracy(y_.softmax(dim=-1), y)
        self.log("loss", loss)
        self.log("acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = F.cross_entropy(y_, y)
        acc = accuracy(y_.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {
                    "params": self.model.primary_task.parameters(),
                    "lr": self.learning_rate * 10,
                },
                {
                    "params": self.model.feature_base.parameters(),
                    "lr": self.learning_rate,
                },
            ],
            lr=self.learning_rate,
        )

    def configure_callbacks(self):
        early_stopping = EarlyStopping(monitor="val_loss", patience=20)
        checkpoint = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1)
        lr_monitor = LearningRateMonitor()
        return [early_stopping, checkpoint, lr_monitor]
