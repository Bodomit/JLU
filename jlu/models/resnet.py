import pytorch_lightning as pl
import torch
import torch.nn as nn


class ResNetBase(pl.LightningModule):

    FEATURE_LENGTH: int = 2048

    def __init__(self, dropout: float = 0.5, resnet_type="resnet50") -> None:
        super().__init__()

        resnet = torch.hub.load("pytorch/vision:v0.9.0", resnet_type, pretrained=False)
        self.features = nn.Sequential(*(list(resnet.children())[:-1]))
        self.fully_connected = nn.Sequential(nn.Dropout(p=dropout), nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        return self.fully_connected(x)
