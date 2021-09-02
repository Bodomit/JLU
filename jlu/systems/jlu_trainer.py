from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.callbacks import JLUAwareEarlyStopping
from jlu.losses import UniformTargetKLDivergence
from jlu.models import JLUMultitaskModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy


class JLUTrainer(pl.LightningModule):
    def __init__(
        self,
        alpha: float,
        learning_rate: float,
        primary_task: str,
        secondary_task: List[str],
        bootstrap_epochs: int,
        datamodule_n_classes: np.ndarray,
        datamodule_labels: List[str],
        pretrained: str,
        pretrained_base: Optional[pl.LightningModule],
        dropout: float,
        *args,
        ls_average_n_steps: int = 5,
        ls_is_best_patentice: int = 50,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pretrained_base")

        assert alpha >= 0

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.primary_task = primary_task
        self.secondary_task = secondary_task
        self.bootstrap_epochs = bootstrap_epochs
        self.datamodule_n_classes = datamodule_n_classes

        self.ls_average_n_steps = ls_average_n_steps
        self.ls_is_best_patentice = ls_is_best_patentice

        self.ls_is_best = False
        self.train_lp_lconf = False
        self.automatic_optimization = False

        # Get label indexes per task.
        self.primary_idx: int = datamodule_labels.index(self.primary_task)
        self.secondary_idx: List[int] = [
            datamodule_labels.index(s) for s in self.secondary_task
        ]
        all_idx: List[int] = [self.primary_idx] + self.secondary_idx

        # Get the model.
        n_classes = self.datamodule_n_classes[all_idx]
        self.model = JLUMultitaskModel(n_classes, pretrained_base, dropout=dropout)

        # Store the loss object.
        self.uniform_kldiv = UniformTargetKLDivergence()

    def on_fit_start(self) -> None:
        # Get the weights.
        self.y_primary_weights = torch.tensor(
            self.trainer.datamodule.train.weights_per_label[self.primary_idx],
            dtype=torch.float,
        )
        self.y_secondary_weights = [
            torch.tensor(
                self.trainer.datamodule.train.weights_per_label[i], dtype=torch.float
            )
            for i in self.secondary_idx
        ]

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # Reset ls early stopping.
        self.ls_loss_average_buffer = torch.zeros(
            self.ls_average_n_steps, device=self.device
        )
        self.ls_is_best_patentice_counter = 0
        self.ls_is_best_current_best = None

        # If the ls is best possible train lp and lconf.
        if self.ls_is_best:
            self.train_lp_lconf = True
        else:
            # Else train ls until grads are near zero.
            self.train_lp_lconf = False

        if self.current_epoch < self.bootstrap_epochs:
            self.train_lp_lconf = True

    def on_train_epoch_end(self, unused=None):
        scheduler: ReduceLROnPlateau = self.lr_schedulers()  # type: ignore
        assert isinstance(scheduler, ReduceLROnPlateau)

        # If current epoch was training lp and backbone
        # reset ls_grads_are_near_zero flag.
        if self.train_lp_lconf:
            scheduler.step(self.trainer.callback_metrics["loss/lp+alconf"])
            self.ls_is_best = False

    def training_step(self, batch, batch_idx):
        if self.train_lp_lconf:
            self.training_step_Lp_Lconf(batch, batch_idx)
        else:
            self.training_step_Ls(batch, batch_idx)

    def on_train_batch_start(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Optional[int]:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        # Stops the epoch early grads are near zero.
        if self.ls_is_best and not self.train_lp_lconf:
            return -1

    def training_step_Ls(self, batch, batch_idx):
        opt = self.optimizers()[1]  # type: ignore
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, yb = batch

        y_secondary = yb[:, self.secondary_idx]
        _, *y_secondary_ = self.model(xb)

        y_secondary_losses = torch.zeros(len(y_secondary_), device=self.device)
        for i, ys_ in enumerate(y_secondary_):
            ys_loss = F.cross_entropy(
                ys_, y_secondary[:, i], self.y_secondary_weights[i].to(self.device)
            )
            y_secondary_losses[i] = ys_loss

        ls = torch.sum(y_secondary_losses)
        self.log("loss/ls", ls, prog_bar=True)

        self.manual_backward(ls)

        loss_average = self.average_loss_over_steps(ls, batch_idx)
        if batch_idx > self.ls_average_n_steps:
            self.ls_is_best = self.ls_is_best_check(loss_average)
        else:
            self.ls_is_best = False

        opt.step()

    def average_loss_over_steps(self, loss: torch.Tensor, batch_idx: int):
        idx = batch_idx % self.ls_average_n_steps
        self.ls_loss_average_buffer[idx] = loss
        return self.ls_loss_average_buffer.mean()

    def ls_is_best_check(self, average: torch.Tensor) -> bool:
        if self.ls_is_best_current_best is None:
            self.ls_is_best_current_best = average
            return False

        if average < self.ls_is_best_current_best:
            self.ls_is_best_patentice_counter = 0
            self.ls_is_best_current_best = average
            return False

        self.ls_is_best_patentice_counter += 1
        return self.ls_is_best_patentice_counter > self.ls_is_best_patentice

    def training_step_Lp_Lconf(self, batch, batch_idx):
        opt = self.optimizers()[0]  # type: ignore
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, yb = batch
        y_primary = yb[:, self.primary_idx]

        # Get the primary and secondary outputs.
        # Note that y_secondary_ is a list of outputs.
        y_primary_, *y_secondary_ = self.model(xb)

        if not self.y_primary_weights.is_cuda:
            self.y_primary_weights = self.y_primary_weights.to(self.device)

        # Calculate the primary loss
        lp = F.cross_entropy(y_primary_, y_primary, self.y_primary_weights)
        self.log(f"loss/lp", lp, prog_bar=True)

        # Calculate the confusion losses.
        lconfs = torch.stack([self.uniform_kldiv(ys_) for ys_ in y_secondary_])
        for i, lconf_m in enumerate(lconfs):
            self.log(f"loss/lconf/{i}", lconf_m)

        lconf = lconfs.sum()
        self.log(f"loss/lconf", lconf, prog_bar=True)

        # Get the total loss.
        total_loss = lp + (self.alpha * lconf)
        self.log(f"loss/lp+alconf", total_loss, prog_bar=True)

        acc = accuracy(y_primary_.softmax(dim=-1), y_primary)
        self.log(f"acc", acc, prog_bar=True)

        self.manual_backward(total_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        # Get the batch.
        xb, yb = batch
        y_primary = yb[:, self.primary_idx]

        # Get the primary and secondary outputs.
        # Note that y_secondary_ is a list of outputs.
        y_primary_, *y_secondary_ = self.model(xb)

        # Calculate the primary loss
        lp = F.cross_entropy(y_primary_, y_primary)
        self.log(f"val_loss/lp", lp)

        # Calculate the confusion losses.
        lconfs = torch.stack([self.uniform_kldiv(ys_) for ys_ in y_secondary_])
        for i, lconf_m in enumerate(lconfs):
            self.log(f"val_loss/lconf/{i}", lconf_m)

        lconf = lconfs.sum()
        self.log(f"val_loss/lconf", lconf)

        # Get the total loss.
        total_loss = lp + (self.alpha * lconf)
        self.log(f"val_loss/lp+alconf", total_loss)

        acc = accuracy(y_primary_.softmax(dim=-1), y_primary)
        self.log(f"val_acc", acc, prog_bar=True)

        return total_loss

    def configure_callbacks(self):
        super().configure_callbacks()
        early_stopping = JLUAwareEarlyStopping("val_loss/lp+alconf", patience=20)
        checkpoint = ModelCheckpoint(
            monitor="val_loss/lp+alconf", save_last=True, save_top_k=3
        )
        return [early_stopping, checkpoint]

    def configure_optimizers(self):
        optimizer_secondary = torch.optim.Adam(
            self.model.secondary_tasks.parameters(), lr=self.learning_rate * 10
        )
        optimizer_primary = torch.optim.Adam(
            [
                {
                    "params": self.model.primary_task.parameters(),
                    "lr": self.learning_rate * 10,
                },
                {
                    "params": self.model.feature_base.parameters(),
                    "lr": self.learning_rate,
                },
            ]
        )
        return [
            {
                "optimizer": optimizer_primary,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer_primary, patience=10),
                },
            },
            {"optimizer": optimizer_secondary,},
        ]
