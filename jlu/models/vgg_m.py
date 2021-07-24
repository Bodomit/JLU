import pytorch_lightning as pl
import torch
import torch.nn as nn


class VggM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, (7, 7), 2, padding=0),
            nn.LocalResponseNorm(96),
            nn.MaxPool2d(2, 2),
            # Conv2
            nn.Conv2d(96, 256, (5, 5), 2, padding=1),
            nn.LocalResponseNorm(256),
            nn.MaxPool2d(2, 2),
            # Conv3
            nn.Conv2d(256, 512, (3, 3), 1, padding=1),
            # Conv4
            nn.Conv2d(512, 512, (3, 3), 1, padding=1),
            # Conv5
            nn.Conv2d(512, 512, (3, 3), 1, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
