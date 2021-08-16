import itertools
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.callbacks import JLUAwareEarlyStopping
from jlu.data import UTKFace
from jlu.losses import UniformTargetKLDivergence
from jlu.models import JLUMultitaskModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy


class JLUTrainer(pl.LightningModule):
    def __init__(
        self,
        datamodule: str,
        alpha: float,
        learning_rate: float,
        primary_task: str,
        secondary_task: List[str],
        bootstrap_epochs: int,
        grads_near_zero_threshold: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        assert alpha >= 0

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.grads_near_zero_threshold = grads_near_zero_threshold
        self.bootstrap_epochs = bootstrap_epochs
        self.ls_grads_are_near_zero = False
        self.train_lp_lconf = False
        self.automatic_optimization = False
        self.datamodule = self.load_datamodule(datamodule, **kwargs)
        self.datamodule.setup()

        # Get label indexes per task.
        self.primary_idx: int = self.get_label_index(primary_task)
        self.secondary_idx: List[int] = [
            self.get_label_index(s) for s in secondary_task
        ]
        all_idx: List[int] = [self.primary_idx] + self.secondary_idx

        # Get the model.
        n_classes = self.datamodule.n_classes[all_idx]
        self.model = JLUMultitaskModel(n_classes)

        # Store the loss object.
        self.uniform_kldiv = UniformTargetKLDivergence()

        # Get the weights.
        self.y_primary_weights = torch.tensor(
            self.datamodule.train.weights_per_label[self.primary_idx], dtype=torch.float
        )
        self.y_secondary_weights = [
            torch.tensor(self.datamodule.train.weights_per_label[i], dtype=torch.float)
            for i in self.secondary_idx
        ]

    def load_datamodule(self, datamodule: str, **kwargs) -> pl.LightningDataModule:
        if datamodule.lower() == "utkface":
            return UTKFace(**kwargs)
        else:
            raise ValueError

    def get_label_index(self, label: str):
        return self.datamodule.labels.index(label)

    def grads_near_zero(self) -> bool:
        grad_total = 0.0
        for param in (p for p in self.model.parameters() if p.grad is not None):
            assert param.grad is not None
            grad_total += param.grad.abs().sum()

        result = bool(grad_total < self.grads_near_zero_threshold)
        return result

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        # If the ls grads were are near zero on last epoch
        # train lp and lconf.
        if self.ls_grads_are_near_zero:
            self.train_lp_lconf = True
        else:
            # Else train ls until grads are near zero.
            self.train_lp_lconf = False

        if self.current_epoch < self.bootstrap_epochs:
            self.train_lp_lconf = True

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        # If current epoch was training lp and lconf
        # reset ls_grads_are_near_zero flag.
        if self.train_lp_lconf:
            self.ls_grads_are_near_zero = False

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.train_lp_lconf:
            self.training_step_Lp_Lconf(batch, batch_idx)
        else:
            self.training_step_Ls(batch, batch_idx)

    def on_train_batch_start(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Optional[int]:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        # Stops the epoch early grads are near zero.
        if self.ls_grads_are_near_zero and not self.train_lp_lconf:
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

        self.ls_grads_are_near_zero = self.grads_near_zero()

        opt.step()

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

        self.manual_backward(lp)
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
        return [optimizer_primary, optimizer_secondary]
