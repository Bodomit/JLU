import pytorch_lightning as pl
import torch.nn as nn


class AttributeExtractionModel(pl.LightningModule):
    def __init__(
        self, n_inputs=512, n_outputs=2, use_short_attribute_branch=False, **kwargs
    ):
        super().__init__()
        super().save_hyperparameters()
        if use_short_attribute_branch:
            self.full_model = nn.Sequential(
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(n_inputs, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 32),
                nn.LeakyReLU(),
                nn.Linear(32, n_outputs),
            )
        else:
            self.full_model = nn.Sequential(
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(n_inputs, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 16),
                nn.LeakyReLU(),
                nn.Linear(16, n_outputs),
            )

    def forward(self, x):
        return self.full_model(x)
