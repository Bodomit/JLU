from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jlu.data.utkface import UTKFace


class JLUTrainer(pl.LightningModule):
    def __init__(
        self,
        datamodule: str,
        alpha: float,
        learning_rate: float,
        primary_task: str,
        secondary_task: List[str],
        grads_near_zero_threshold: float = 2e-9,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        assert alpha >= 0

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.grads_near_zero_threshold = grads_near_zero_threshold
        self.grads_are_near_zero = False
        self.automatic_optimization = False

        self.datamodule = self.load_datamodule(datamodule, **kwargs)

        # Get label indexes per task.
        self.primary_idx = self.get_label_index(primary_task)
        self.seconday_idx = [self.get_label_index(s) for s in secondary_task]

    def load_datamodule(self, datamodule: str, **kwargs) -> pl.LightningDataModule:
        if datamodule.lower() == "utkface":
            return UTKFace(**kwargs)
        else:
            raise ValueError

    def get_label_index(self, label: str):
        return self.datamodule.labels.index(label)

    def attribute_grads_near_zero(self) -> bool:
        grad_total = 0.0
        for param in self.model.parameters():
            assert param.grad
            grad_total += param.grad.abs().sum()

        assert isinstance(grad_total, float)
        return grad_total < self.grads_near_zero_threshold

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.grads_are_near_zero = False

    def on_train_batch_start(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Optional[int]:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        # Stops the epoch early if on an even (Ls training) epoch and grads
        # are near zero.
        if self.current_epoch % 2 == 0 and self.grads_are_near_zero:
            return -1

    def training_step_Ls(self, batch, batch_idx):

        # Train attribute model only.
        self.model.freeze()
        self.model.unfreeze()

        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, (_, ab) = batch
        ab = ab.squeeze()

        _, attribute_pred = self.model(xb)

        attribute_loss = F.cross_entropy(attribute_pred, ab, self.attribute_weights)
        self.log(
            "loss/stage1", attribute_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        self.manual_backward(attribute_loss)

        self.grads_are_near_zero = self.attribute_grads_near_zero()

        opt.step()
        opt.zero_grad()

    def training_step_Lp_Lconf(self, batch, batch_idx):

        # Train on the FC layers in the model only.
        self.model.unfreeze()
        self.model.attribute_model.freeze()

        # If using the pretrained model, only unfreeze the extra fc layers.
        if self.model.feature_model.use_pretrained:
            self.model.feature_model.resnet.requires_grad = False
            self.model.feature_model.extra_fc.requires_grad = True
        else:
            self.model.feature_model.unfreeze()

        opt = self.optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        opt.zero_grad()

        # Get the batch.
        xb, (yb, ab) = batch
        ab = ab.squeeze()

        logits, attribute_pred = self.model(xb)

        # Log metrics.
        metrics = self.training_step_attribute_metrics(ab, attribute_pred)
        self.log_dict(metrics)

        # Backprop the multi-task loss.
        assert isinstance(logits, torch.Tensor)
        total_loss, sub_losses = self.get_totalloss_with_sublosses(
            self.loss,
            yb,
            ab,
            logits,
            attribute_pred,
            prefix="loss/stage2/",
        )
        self.log("loss/stage2/total", total_loss, on_step=True, on_epoch=True)
        self.log_dict(sub_losses, on_step=True, on_epoch=True, prog_bar=True)
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()

    def training_step(self, batch, batch_idx):
        # If on an even Epoch (starting at 0), train Ls.
        # If on an odd Epoch (starting at 1), train combined loss.
        if self.current_epoch % 2 == 0:
            return self.training_step_Ls(batch, batch_idx)
        else:
            return self.training_step_Lp_Lconf(batch, batch_idx)

    def configure_optimizers(self):
        return (torch.optim.SGD(self.model.parameters(), lr=self.learning_rate),)
