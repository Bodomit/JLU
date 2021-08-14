import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL
import pytorch_lightning as pl
from jlu.data.utils import parse_dataset_dir, read_filenames
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class VGGFace2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir="vggface2",
        image_size=(224, 224),
        val_split=0.1,
        **kwargs
    ):

        self.batch_size = batch_size
        self.dataset_dir = parse_dataset_dir(dataset_dir)
        self.val_split = val_split
        self.train_split = 1 - val_split
        self.num_workers = min(32, len(os.sched_getaffinity(0)))

        # Define the transformations.
        common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Reads images scales to [0, 1]
                transforms.Lambda(lambda x: x * 2 - 1),  # Change range to [-1, 1]
                transforms.Resize(image_size),
            ]
        )
        train_transforms = transforms.Compose(
            [common_transforms, transforms.RandomHorizontalFlip(p=0.5)]
        )
        val_transforms = common_transforms
        test_transforms = common_transforms

        # Pass to base.
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=(3,) + image_size,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.trainval_filenames = list(
                sorted(glob.glob(os.path.join(self.dataset_dir, "train", "*", "*.jpg")))
            )

            trainval_x, trainval_y = zip(*self.parse_filenames(self.trainval_filenames))
            trainval_x = np.array(list(trainval_x))
            trainval_y = np.array(list(trainval_y))

            self.fit_label_map = self.get_label_map(trainval_y)
            trainval_y = np.array(list(self.fit_label_map[y] for y in trainval_y))

            # Calculate train and validation splits.
            train_x, val_x, train_y, val_y = train_test_split(
                trainval_x, trainval_y, train_size=self.train_split, random_state=42
            )

            assert isinstance(train_x, np.ndarray)
            assert isinstance(train_y, np.ndarray)
            assert isinstance(val_x, np.ndarray)
            assert isinstance(val_y, np.ndarray)

            self.n_classes = len(np.unique(np.concatenate((train_y, val_y))))
            self.train = VGGFace2Dataset(train_x, train_y, self.train_transforms)
            self.valid = VGGFace2Dataset(val_x, val_y, self.val_transforms)

        if stage == "test" or stage is None:
            self.test_filenames = list(
                sorted(glob.glob(os.path.join(self.dataset_dir, "test", "*", "*.jpg")))
            )

            test_x, test_y = zip(*self.parse_filenames(self.test_filenames))
            test_x = np.array(list(test_x))
            test_y = np.array(list(test_y))
            assert isinstance(test_x, np.ndarray)
            assert isinstance(test_y, np.ndarray)

            self.test_label_map = self.get_label_map(test_y)
            test_y = np.array(list(self.test_label_map[y] for y in test_y))

            self.test = VGGFace2Dataset(test_x, test_y, self.val_transforms)

    @staticmethod
    def parse_filenames(filenames: List[str]):
        parsed_filenames: List[Tuple[str, str]] = []
        for f in filenames:
            match = re.match(r".+/([\w]+)/[\w\.]+$", f)
            assert match
            label = match.groups()[0]
            parsed_filenames.append((f, label))

        return parsed_filenames

    @staticmethod
    def get_label_map(labels: np.ndarray) -> Dict[str, int]:
        unique_labels = np.unique(labels)
        new_labels = range(len(unique_labels))
        label_map = dict(zip(unique_labels, new_labels))
        return label_map

    def train_dataloader(self):
        return DataLoader(
            self.train, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, self.batch_size, shuffle=False, num_workers=self.num_workers
        )


class VGGFace2Dataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.n_classes = len(np.unique(y))
        self.support_per_class = self.calc_support(self.y)
        self.weights_per_class = self.calc_weights(self.support_per_class)

        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        x = PIL.Image.open(self.x[index])  # type: ignore
        x = self.transform(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

    @staticmethod
    def calc_support(labels: np.ndarray) -> np.ndarray:
        _, c = np.unique(labels, return_counts=True)
        return c

    @staticmethod
    def calc_weights(
        support: np.ndarray,
    ) -> np.ndarray:
        weights = 1 / support
        weights = weights / weights.sum() * len(weights)
        return weights