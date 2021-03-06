import pytorch_lightning as pl
import torch
import torch.nn as nn


class VggMBase(pl.LightningModule):

    FEATURE_LENGTH: int = 4096

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, (7, 7), 2, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(96),
            nn.MaxPool2d(2, 2),
            # Conv2
            nn.Conv2d(96, 256, (5, 5), 2, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(256),
            nn.MaxPool2d(2, 2),
            # Conv3
            nn.Conv2d(256, 512, (3, 3), 1, padding=1),
            nn.ReLU(),
            # Conv4
            nn.Conv2d(512, 512, (3, 3), 1, padding=1),
            nn.ReLU(),
            # Conv5
            nn.Conv2d(512, 512, (3, 3), 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            # FC6
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            # FC7
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fully_connected(x)
        return x
