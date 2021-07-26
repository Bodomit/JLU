import os

import pytest
from jlu import data
from jlu.data import UTKFace

POSSIBLE_DATASET_ROOT_DIRS = ["/mnt/e/datasets", "~/datasets"]


def get_root_dir() -> str:
    for dir in POSSIBLE_DATASET_ROOT_DIRS:
        dir = os.path.abspath(os.path.expanduser(dir))
        if os.path.isdir(dir):
            return dir
    raise ValueError("No dataset directory found.")


# UKTFace Tests -----------------------------------------------------------------------
@pytest.fixture
def uktface_dm():
    dataset = UTKFace(32)
    dataset.setup()
    return dataset


def test_loading_full(uktface_dm):
    assert uktface_dm
    assert len(uktface_dm.filenames) == 23708
    assert len(uktface_dm.train) == 18966
    assert len(uktface_dm.valid) == 2371
    assert len(uktface_dm.test) == 2371


def dataloader_test(dataloader, batch_size, image_size=(3, 224, 224)):
    for x, y in dataloader:
        assert x.shape == (batch_size,) + image_size
        assert y.shape == (batch_size, 3)
        break


def test_dataloaders(uktface_dm):
    assert uktface_dm
    dataloader_test(uktface_dm.train_dataloader(), uktface_dm.batch_size)
    dataloader_test(uktface_dm.val_dataloader(), uktface_dm.batch_size)
    dataloader_test(uktface_dm.test_dataloader(), uktface_dm.batch_size)
