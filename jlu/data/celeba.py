import os
from functools import partial
from typing import Any, List, Optional

import numpy as np
import pandas
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .common import AttributeDataset
from .utils import get_unique_in_columns, parse_dataset_dir


class CelebA(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir: str = "CelebA_MTCNN",
        image_size=(224, 224),
        use_png: bool = True,
        random_affine=False,
        random_crop=False,
        **kwargs
    ):

        # Store attributes.
        self.batch_size = batch_size
        self.dataset_dir = parse_dataset_dir(dataset_dir)
        self.num_workers = min(32, len(os.sched_getaffinity(0)))
        self.image_dir = "img_align_celeba_png" if use_png else "img_align_celeba"
        self.ext = "png" if use_png else "jpg"

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

        fn = partial(os.path.join, self.dataset_dir)
        image_absdir = fn(self.image_dir)
        real_image_ids = set(
            [
                os.path.basename(f.path)
                for f in os.scandir(image_absdir)
                if f.name.split(".")[1] == self.ext
            ]
        )

        split_map = {"train": 0, "valid": 1, "test": 2}

        replace_ext_ = partial(self.replace_etx, self.ext)
        identities = pandas.read_csv(  # type: ignore
            fn("identity_CelebA.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        )

        sort_order = identities.sort_values(1).index
        identities = identities.loc[sort_order]

        splits = pandas.read_csv(  # type: ignore
            fn("list_eval_partition.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
            converters={0: replace_ext_},
        ).loc[sort_order]
        attrs = pandas.read_csv(
            fn("list_attr_celeba.txt"),
            delim_whitespace=True,
            header=1,
            converters={0: replace_ext_},
        ).loc[sort_order]

        assert isinstance(attrs, pandas.DataFrame)

        # Remove attribute lines that have no corresponding image.
        diff = list(sorted(set(splits.index.values) - real_image_ids))

        def rm_diff(df: pandas.DataFrame) -> pandas.DataFrame:
            return df.drop(diff)

        splits = rm_diff(splits)
        assert len(splits) == len(real_image_ids)
        assert isinstance(attrs, pandas.DataFrame)

        # Get the attribute names and coresponding indexes.
        attr_names = ["id"] + list(attrs.columns)
        attr_names[attr_names.index("Male")] = "sex"

        fullpath = np.vectorize(lambda filename: os.path.join(image_absdir, filename))

        # For each split, consturct a mask and create a correspondign dataset.
        for split in split_map:
            if stage == "fit" and split == "test":
                continue
            elif stage == "test" and split in ["train", "valid"]:
                continue

            mask = splits[1] == split_map[split]

            filenames = splits[mask].index.values
            filenames = fullpath(filenames)
            identities_ = rm_diff(identities)[mask].values.squeeze()
            attrs_ = rm_diff(attrs)[mask].values.squeeze()
            attrs_ = (attrs_ + 1) // 2  # map from {-1, 1} to {0, 1}

            # Remap the identities to be contiguous (for classification training).
            unique_identities, identity_map = np.unique(
                identities_, return_inverse=True
            )
            local_unique_identities = np.arange(
                0, len(unique_identities), dtype=np.int64
            )
            local_identities = local_unique_identities[identity_map]

            if split == "train":
                transform = self.train_transforms
            elif split == "valid":
                transform = self.val_transforms
            elif split == "test":
                transform = self.test_transforms
            else:
                raise ValueError()

            x = filenames
            y = np.concatenate(
                (np.expand_dims(local_identities, axis=1), attrs_), axis=1
            )

            split_dataset = AttributeDataset(x, y, transform, attr_names)

            if split == "train":
                self.train = split_dataset
            elif split == "valid":
                self.valid = split_dataset
            elif split == "test":
                self.test = split_dataset

        super().setup(stage)

    @staticmethod
    def replace_etx(ext: str, val: str) -> str:
        return val.split(".")[0] + "." + ext

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
