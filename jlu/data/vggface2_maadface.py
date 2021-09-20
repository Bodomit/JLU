import os
from functools import partial
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .common import AttributeDataset
from .utils import get_unique_in_columns, parse_dataset_dir


class VGGFace2WithMaadFace(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir="vggface2_MTCNN",
        image_size=(224, 224),
        val_split=0.1,
        random_affine=False,
        random_crop=False,
        **kwargs,
    ):

        self.batch_size = batch_size
        self.dataset_dir = parse_dataset_dir(dataset_dir)
        self.val_split = val_split
        self.train_split = 1 - val_split
        self.num_workers = min(32, len(os.sched_getaffinity(0)))
        self.val_split = val_split

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

    def setup(self, stage: Optional[str]) -> None:

        # Read the attributes. Usecols is specified to reduce memory usage
        # and parsing time.
        attributes = pandas.read_csv(  # type: ignore
            os.path.join(self.dataset_dir, "MAAD_Face.csv"),
            index_col=0,
            usecols=["Filename", "Male"],
        )
        attributes.columns = ["sex"]
        assert isinstance(attributes, pandas.DataFrame)

        self.labels = ["id", "sex"]

        if stage is None or stage == "fit":
            self.train, self.valid = self._process_train_valid(attributes)
            self.n_classes = get_unique_in_columns(self.train.y)

        if stage is None or stage == "test":
            self.test = self._process_test(attributes)

        super().setup(stage)

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

    def _process_train_valid(
        self, attributes: pandas.DataFrame
    ) -> Tuple[AttributeDataset, AttributeDataset]:
        real_split_dir = os.path.join(self.dataset_dir, "train")

        # Get the full training set.
        imgs_with_attrs: Set[str] = set(attributes.index.tolist())

        def is_valid_file(path: str):
            return os.path.relpath(path, real_split_dir) in imgs_with_attrs

        image_folder = ImageFolder(real_split_dir, is_valid_file=is_valid_file)

        # Get the classes for the validation set and mask.
        real_split_classes = set(
            [d.name for d in os.scandir(real_split_dir) if d.is_dir()]
        )
        val_classes = self.get_val_set_classes(real_split_classes, self.val_split, 42)
        val_classes_encoded = set([image_folder.class_to_idx[c] for c in val_classes])

        val_class_mask = [s[1] in val_classes_encoded for s in image_folder.samples]

        # Split into validation and training samples.
        val_samples: List[Tuple[str, int]] = []
        train_samples: List[Tuple[str, int]] = []
        for (s, cm,) in zip(image_folder.samples, val_class_mask):
            if cm:
                val_samples.append(s)
            else:
                train_samples.append(s)

        # Construct the dataset objects proper.
        construct_fn = partial(
            self.construct_dataset,
            real_split_dir=real_split_dir,
            attributes=attributes,
        )
        train_dataset = construct_fn(
            samples=train_samples, transform=self.train_transforms
        )
        val_dataset = construct_fn(samples=val_samples, transform=self.val_transforms)

        return train_dataset, val_dataset

    def _process_test(self, attributes: pandas.DataFrame) -> AttributeDataset:
        real_split_dir = os.path.join(self.dataset_dir, "test")

        # Get the full test set, removing those without attributes.
        imgs_with_attrs: Set[str] = set(attributes.index.tolist())

        def is_valid_file(path: str):
            return os.path.relpath(path, real_split_dir) in imgs_with_attrs

        image_folder = ImageFolder(real_split_dir, is_valid_file=is_valid_file)

        test_dataset = self.construct_dataset(
            real_split_dir, image_folder.samples, attributes, self.test_transforms,
        )

        return test_dataset

    def construct_dataset(
        self,
        real_split_dir: str,
        samples: List[Tuple[str, int]],
        attributes: pandas.DataFrame,
        transform: transforms.Compose,
    ) -> AttributeDataset:

        # Remove any attribute lines that are not in the dataset samples.
        real_imgs = [os.path.relpath(s[0], start=real_split_dir) for s in samples]
        diff = set(attributes.index.values) - set(real_imgs)
        attributes = attributes.drop(list(diff), errors="ignore")

        # Sort the attributes to match the sample order.
        sort_order = attributes.index.sort_values()
        attributes = attributes.loc[sort_order]  # type: ignore

        # Ensure the attribute file and samples are aligned.
        assert isinstance(attributes, pandas.DataFrame)
        for x, y in zip(real_imgs, attributes.index.tolist()):
            assert x == y

        identities = np.array([s[1] for s in samples])
        attributes_ = np.array(attributes.values)
        attributes_ = (attributes_ + 1) // 2  # map from {-1, 1} to {0, 1}

        # Remap the identities to be contiguous (for classification training).
        unique_identities, identity_map = np.unique(identities, return_inverse=True)
        local_unique_identities = np.arange(0, len(unique_identities), dtype=np.int64)
        local_identities = local_unique_identities[identity_map]

        # One last sanity check
        assert len(local_identities) == len(attributes_) == len(samples)

        # Separate filename into x.
        X = np.array([s[0] for s in samples])

        # Merge identity and attributes into y.
        Y = np.concatenate(
            (np.expand_dims(local_identities, axis=1), attributes_), axis=1
        )

        return AttributeDataset(X, Y, transform, self.labels)

    @staticmethod
    def get_val_set_classes(
        classes: Set[str], valid_split: float, valid_split_seed: int
    ) -> Set[str]:
        assert valid_split >= 0 and valid_split < 1.0
        n_valid_set_classes = int(len(classes) * valid_split)
        rng = np.random.default_rng(valid_split_seed)
        valid_classes = rng.choice(sorted(classes), size=n_valid_set_classes)
        return set(valid_classes)
