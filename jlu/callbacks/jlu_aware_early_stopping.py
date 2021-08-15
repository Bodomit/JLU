from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


class JLUAwareEarlyStopping(EarlyStopping):
    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args
    ) -> None:

        if pl_module.train_lp_lconf:
            return super().on_train_epoch_end(trainer, pl_module)
        else:
            return

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.train_lp_lconf:
            return super().on_validation_end(trainer, pl_module)
        else:
            return
