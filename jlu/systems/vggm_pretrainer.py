import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.data.vggface2 import VGGFace2
from jlu.models import VggMClassification
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy


class VggMPretrainer(pl.LightningModule):
    def __init__(self, learning_rate: float, n_classes: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = VggMClassification(n_classes)

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
        super().configure_optimizers()
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=10),
                "monitor": "loss",
            },
        }

    def configure_callbacks(self):
        early_stopping = EarlyStopping(monitor="val_loss", patience=20)
        checkpoint = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1)
        lr_monitor = LearningRateMonitor()
        return [early_stopping, checkpoint, lr_monitor]
