from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.data import UTKFace
from jlu.losses import UniformTargetKLDivergence
from jlu.models import JLUMultitaskModel


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
        self.primary_idx = self.get_label_index(primary_task)
        self.seconday_idx = [self.get_label_index(s) for s in secondary_task]
        all_idx: List[int] = [self.primary_idx] + self.seconday_idx

        # Get the model.
        n_classes = self.datamodule.n_classes[all_idx]
        self.model = JLUMultitaskModel(n_classes)

        # Store the loss object.
        self.uniform_kldiv = UniformTargetKLDivergence()

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
        if self.ls_grads_are_near_zero and not self.train_lp_lconf:
            return -1

    def training_step_Ls(self, batch, batch_idx):

        # Train attribute model only.
        self.model.requires_grad_(False)
        self.model.secondary_tasks.requires_grad_(True)

        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, yb = batch

        y_secondary = yb[:, self.seconday_idx]
        _, *y_secondary_ = self.model(xb)

        y_secondary_losses = torch.zeros(len(y_secondary_), device=self.device)
        for i, ys_ in enumerate(y_secondary_):
            ys_loss = F.cross_entropy(ys_, y_secondary[:, i])
            y_secondary_losses[i] = ys_loss

        ls = torch.sum(y_secondary_losses)
        self.log("loss/ls", ls, on_step=True, on_epoch=True, prog_bar=True)

        self.manual_backward(ls)

        self.ls_grads_are_near_zero = self.grads_near_zero()

        opt.step()
        opt.zero_grad()

    def training_step_Lp_Lconf(self, batch, batch_idx):
        self.model.requires_grad_(True)
        self.model.secondary_tasks.requires_grad_(False)

        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, yb = batch
        y_primary = yb[:, self.primary_idx]

        # Get the primary and secondary outputs.
        # Note that y_secondary_ is a list of outputs.
        y_primary_, *y_secondary_ = self.model(xb)

        # Calculate the primary loss
        lp = F.cross_entropy(y_primary_, y_primary)
        self.log(f"loss/lp", lp, on_step=True, on_epoch=True, prog_bar=True)

        # Calculate the confusion losses.
        lconfs = torch.stack([self.uniform_kldiv(ys_) for ys_ in y_secondary_])
        for i, lconf_m in enumerate(lconfs):
            self.log(f"loss/lconf/{i}", lconf_m, on_step=True, on_epoch=True)

        lconf = lconfs.sum()
        self.log(f"loss/lconf", lconf, on_step=True, on_epoch=True, prog_bar=True)

        # Get the total loss.
        total_loss = lp + (self.alpha * lconf)
        self.log(
            f"loss/lp+alconf", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        # Get the batch.
        xb, yb = batch
        y_primary = yb[:, self.primary_idx]

        # Get the primary and secondary outputs.
        # Note that y_secondary_ is a list of outputs.
        y_primary_, *y_secondary_ = self.model(xb)

        # Calculate the primary loss
        lp = F.cross_entropy(y_primary_, y_primary)
        self.log(f"val_loss/lp", lp, on_epoch=True)

        # Calculate the confusion losses.
        lconfs = torch.stack([self.uniform_kldiv(ys_) for ys_ in y_secondary_])
        for i, lconf_m in enumerate(lconfs):
            self.log(f"val_loss/lconf/{i}", lconf_m)

        lconf = lconfs.sum()
        self.log(f"val_loss/lconf", lconf)

        # Get the total loss.
        total_loss = lp + (self.alpha * lconf)
        self.log(f"val_loss/lp+alconf", total_loss)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {"params": self.model.feature_base.parameters()},
                {"params": self.model.primary_task.parameters(), "lr": 1e-3},
                {"params": self.model.secondary_tasks.parameters(), "lr": 1e-3},
            ],
            lr=1e-4,
        )
