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
    def __init__(self, datamodule: str, learning_rate: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = self.load_datamodule(datamodule, **kwargs)
        self.datamodule.setup("fit")
        self.learning_rate = learning_rate
        self.model = VggMClassification(self.datamodule.n_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = F.cross_entropy(y_, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)
        loss = F.cross_entropy(y_, y)
        acc = accuracy(y_.softmax(dim=-1), y)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss)
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
        checkpoint = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=3)
        lr_monitor = LearningRateMonitor()
        return [early_stopping, checkpoint, lr_monitor]

    @staticmethod
    def load_datamodule(datamodule: str, **kwargs) -> pl.LightningDataModule:
        if datamodule.lower() == "vggface2":
            return VGGFace2(**kwargs)
        else:
            raise ValueError
