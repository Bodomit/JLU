import os
import re
from typing import List, Optional, Set, Tuple

import numpy as np
import PIL
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import parse_dataset_dir, read_filenames


class UTKFace(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir="UTKFace",
        image_size=(224, 224),
        test_split=0.1,
        val_split=0.1,
    ):
        self.batch_size = batch_size
        self.dataset_dir = parse_dataset_dir(dataset_dir)
        self.test_split = test_split
        self.val_split = val_split
        self.train_split = 1 - val_split - test_split

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
        self.filenames = list(sorted(read_filenames(self.dataset_dir)))

        full_dataset = self.parse_filenames(self.filenames)
        x, y = zip(*full_dataset)

        self.x = np.array(list(x))  # type: ignore
        self.y = np.array(list(y))  # type: ignore

        train_x, val_test_x, train_y, val_test_y = train_test_split(
            self.x, self.y, train_size=self.train_split, random_state=42
        )

        val_test_ratio = self.val_split / (self.val_split + self.test_split)
        valid_x, test_x, valid_y, test_y = train_test_split(
            val_test_x, val_test_y, train_size=val_test_ratio
        )

        assert len(train_x) + len(valid_x) + len(test_x) == len(self.x)
        assert len(train_y) + len(valid_y) + len(test_y) == len(self.y)

        assert isinstance(train_x, np.ndarray)
        assert isinstance(train_y, np.ndarray)
        assert isinstance(valid_x, np.ndarray)
        assert isinstance(valid_y, np.ndarray)
        assert isinstance(test_x, np.ndarray)
        assert isinstance(test_y, np.ndarray)

        self.train = UTKFaceDataset(train_x, train_y, self.train_transforms)
        self.valid = UTKFaceDataset(valid_x, valid_y, self.val_transforms)
        self.test = UTKFaceDataset(test_x, test_y, self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, shuffle=False)

    @staticmethod
    def parse_filenames(filenames: List[str]):
        parsed_filenames: List[Tuple[str, Tuple[int, int, int]]] = []
        for f in filenames:
            match = re.match(r"^(\d+)_(\d+)_(\d+)", os.path.basename(f))
            assert match
            age, gender, race = match.groups()
            parsed_filenames.append((f, (int(age), int(gender), int(race))))

        return parsed_filenames


class UTKFaceDataset(Dataset):
    AGE_BIN_BOUNDARIES = [0, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 2000]

    def __init__(self, x: np.ndarray, y: np.ndarray, transform, bin_age=True) -> None:
        super().__init__()
        self.x = x
        self.transform = transform
        self.bin_age = bin_age

        if bin_age:
            age, sex, race = y.T
            age_bin = np.digitize(age, self.AGE_BIN_BOUNDARIES) - 1
            self.y = np.vstack((age_bin, sex, race)).T
        else:
            self.y = y

        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        x = PIL.Image.open(self.x[index])  # type: ignore
        x = self.transform(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)
