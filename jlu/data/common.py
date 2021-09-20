from typing import List

import numpy as np
import PIL
from torch.utils.data import Dataset


class AttributeDataset(Dataset):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, transform, labels: List[str]
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.support_per_label = self.calc_support_for_labels(self.y)
        self.weights_per_label = self.calc_weights_for_classes_per_label(
            self.support_per_label
        )
        self.labels = labels
        assert len(self.x) == len(self.y)
        assert self.y.shape[1] == len(self.labels)

    def __getitem__(self, index):
        x = PIL.Image.open(self.x[index])  # type: ignore
        x = self.transform(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

    @staticmethod
    def calc_support_for_labels(labels: np.ndarray) -> List[np.ndarray]:
        support_per_label: List[np.ndarray] = []
        for i in range(labels.shape[1]):
            _, c = np.unique(labels[:, i], return_counts=True)
            support_per_label.append(c)
        return support_per_label

    @staticmethod
    def calc_weights_for_classes_per_label(
        support_per_label: List[np.ndarray],
    ) -> List[np.ndarray]:
        weights_per_label: List[np.ndarray] = []
        for support in support_per_label:
            weights = 1 / support
            weights = weights / weights.sum() * len(weights)
            weights_per_label.append(weights)

        return weights_per_label
