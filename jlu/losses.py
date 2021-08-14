import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformTargetKLDivergence(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_distribution = None

    @staticmethod
    def calc_target_distribution(n_classes: int, device: torch.device):
        assert n_classes > 1
        avg_probability = 1 / n_classes
        probabilities = torch.tensor([avg_probability], device=device).repeat(n_classes)
        return probabilities

    def forward(self, pred: torch.Tensor):
        if self.target_distribution is None:
            n_attribute_classes = pred.shape[1]
            self.target_distribution = self.calc_target_distribution(
                n_attribute_classes, pred.device
            )

        expanded_target = self.target_distribution.expand_as(pred)
        pred_log_softmax = F.log_softmax(pred, dim=1)
        loss = F.kl_div(pred_log_softmax, expanded_target, reduction="none")

        loss_per_sample = loss.sum(1).squeeze()
        loss_reduced = loss_per_sample.sum() / loss_per_sample.size()[0]
        loss_reduced = loss_reduced.clamp(min=0.0)
        return loss_reduced
