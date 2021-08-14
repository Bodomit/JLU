import os

import pytest
from jlu import data
from jlu.data import UTKFace, VGGFace2

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


def test_uktface_loading_full(uktface_dm):
    assert uktface_dm
    assert len(uktface_dm.filenames) == 23708
    assert len(uktface_dm.train) == 18966
    assert len(uktface_dm.valid) == 2371
    assert len(uktface_dm.test) == 2371


def uktface_dataloader_test(dataloader, batch_size, image_size=(3, 224, 224)):
    for x, y in dataloader:
        assert x.shape == (batch_size,) + image_size
        assert y.shape == (batch_size, 3)
        break


def test_uktface_dataloaders(uktface_dm):
    assert uktface_dm
    uktface_dataloader_test(uktface_dm.train_dataloader(), uktface_dm.batch_size)
    uktface_dataloader_test(uktface_dm.val_dataloader(), uktface_dm.batch_size)
    uktface_dataloader_test(uktface_dm.test_dataloader(), uktface_dm.batch_size)


# VGGFace2 Tests ----------------------------------------------------------------------
@pytest.fixture
def vggface2_dm():
    dataset = VGGFace2(32, val_split=0.1)
    dataset.setup()
    return dataset


def test_loading_full(vggface2_dm):
    assert vggface2_dm
    assert len(vggface2_dm.train) == 2827701
    assert len(vggface2_dm.valid) == 314189
    assert len(vggface2_dm.test) == 169396


def vggface2_dataloader_test(dataloader, batch_size, image_size=(3, 224, 224)):
    for x, y in dataloader:
        assert x.shape == (batch_size,) + image_size
        assert len(y.shape) == 1
        assert y.shape[0] == batch_size
        break


def test_vggface2_dataloaders(vggface2_dm):
    assert vggface2_dm
    vggface2_dataloader_test(vggface2_dm.train_dataloader(), vggface2_dm.batch_size)
    vggface2_dataloader_test(vggface2_dm.val_dataloader(), vggface2_dm.batch_size)
    vggface2_dataloader_test(vggface2_dm.test_dataloader(), vggface2_dm.batch_size)
