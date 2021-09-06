import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import PIL
import pytorch_lightning as pl
import torch
from jlu.data.utils import get_unique_in_columns, parse_dataset_dir, read_filenames
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FairFace(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        random_affine: bool,
        random_crop: bool,
        stratification_label="age",
        dataset_dir="fairface-img-margin025-trainval",
        image_size=(224, 224),
        val_split=0.1,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.dataset_dir = parse_dataset_dir(dataset_dir)
        self.val_split = val_split
        self.train_split = 1 - val_split

        self.train_data, self.test_data = self.load_and_parse_data(self.dataset_dir)
        labels = self.train_data.columns.tolist()
        labels[labels.index("gender")] = "sex"

        self.stratification_index = self.train_data.columns.tolist().index(
            stratification_label
        )

        labels.remove("file")
        self.labels = labels

        self.num_workers = min(32, len(os.sched_getaffinity(0)))

        # Define the transformations.
        common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Reads images scales to [0, 1]
                transforms.Lambda(lambda x: x * 2 - 1),  # Change range to [-1, 1]
                transforms.Resize(image_size),
            ]
        )

        train_transforms_list: List[Any] = [common_transforms]
        if random_affine:
            train_transforms_list.append(
                transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1))
            )
        if random_crop:
            train_transforms_list.append(
                transforms.RandomResizedCrop(size=(image_size))
            )
        train_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        train_transforms = transforms.Compose(train_transforms_list)

        val_transforms = common_transforms
        test_transforms = common_transforms

        # Pass to base.
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=(3,) + image_size,
        )

    def load_and_parse_data(
        self, dataset_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Load in the data.
        train_data = (
            pd.read_csv(os.path.join(dataset_dir, "fairface_label_train.csv"))
            .sort_values(by=["file"])
            .reset_index(drop=True)
        )

        test_data = (
            pd.read_csv(os.path.join(dataset_dir, "fairface_label_val.csv"))
            .sort_values(by=["file"])
            .reset_index(drop=True)
        )

        # Transform the ages so they can be easily sorted.
        train_data["age"] = train_data["age"].transform(
            self._transform_age  # type: ignore
        )
        test_data["age"] = test_data["age"].transform(
            self._transform_age  # type: ignore
        )

        # Encode the dataset values using the train set as basis.
        self.unique_ages = sorted(train_data["age"].unique())
        self.unique_genders = sorted(train_data["gender"].unique())
        self.unique_races = sorted(train_data["race"].unique())

        train_data = self._encode_categorical(
            train_data, self.unique_ages, self.unique_genders, self.unique_races
        )
        test_data = self._encode_categorical(
            test_data, self.unique_ages, self.unique_genders, self.unique_races
        )

        return train_data, test_data

    @staticmethod
    def _transform_age(age: str) -> str:
        if age == "0-2":
            return "00-02"
        elif age == "3-9":
            return "03-09"
        else:
            return age

    @staticmethod
    def _encode_categorical(
        df: pd.DataFrame,
        unique_ages: List[str],
        unique_genders: List[str],
        unique_races: List[str],
    ):
        df["age"] = pd.Categorical(
            df["age"], categories=unique_ages, ordered=True  # type: ignore
        ).codes
        assert df["age"].notna().all()
        assert (df["age"] >= 0).all()
        assert (df["age"] < len(unique_ages)).all()

        df["gender"] = pd.Categorical(
            df["gender"], categories=unique_genders, ordered=False  # type: ignore
        ).codes
        assert df["age"].notna().all()
        assert (df["age"] >= 0).all()
        assert (df["age"] < len(unique_ages)).all()

        df["race"] = pd.Categorical(
            df["race"], categories=unique_races, ordered=False  # type: ignore
        ).codes
        assert df["age"].notna().all()
        assert (df["age"] >= 0).all()
        assert (df["age"] < len(unique_ages)).all()

        return df

    def setup(self, stage: Optional[str] = None):

        if stage is None or stage == "fit":
            train_path = os.path.join(self.dataset_dir, "train")
            self.train_filenames = list(sorted(read_filenames(train_path)))
            assert len(self.train_data) == len(self.train_filenames)

            x = np.array(self.train_filenames)
            y = np.array(self.train_data)

            stratification_labels = (
                y[:, self.stratification_index]
                if self.stratification_index is not None
                else None
            )

            train_x, val_x, train_y, val_y = train_test_split(
                x,
                y,
                train_size=self.train_split,
                random_state=42,
                stratify=stratification_labels,
            )

            assert len(train_x) + len(val_x) == len(x)
            assert len(train_y) + len(val_y) == len(y)

            assert isinstance(train_x, np.ndarray)
            assert isinstance(train_y, np.ndarray)
            assert isinstance(val_x, np.ndarray)
            assert isinstance(val_y, np.ndarray)

            self.train = FairFaceDataset(
                train_x, train_y[:, 1::], self.train_transforms
            )
            self.valid = FairFaceDataset(val_x, val_y[:, 1::], self.val_transforms)

            # Store the number of classes per label.
            self.n_classes = get_unique_in_columns(y[:, 1::])

        if stage is None or stage == "test":
            test_path = os.path.join(self.dataset_dir, "val")
            self.test_filenames = list(sorted(read_filenames(test_path)))
            assert len(self.test_data) == len(self.test_filenames)

            test_x = np.array(list(self.test_filenames))
            test_y = np.array(list(self.test_data.to_numpy()))

            assert isinstance(test_x, np.ndarray)
            assert isinstance(test_y, np.ndarray)

            self.test = FairFaceDataset(test_x, test_y[:, 1::], self.train_transforms)

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


class FairFaceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.support_per_label = self.calc_support_for_labels(self.y)
        self.weights_per_label = self.calc_weights_for_classes_per_label(
            self.support_per_label
        )

        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        x = PIL.Image.open(self.x[index])  # type: ignore
        x = self.transform(x)
        y = self.y[index]

        return x, y.astype(int)

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
