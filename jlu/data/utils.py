import itertools
import os
import warnings
from typing import List, Set, Tuple

import numpy as np
import ruyaml as yaml


def parse_dataset_dir(dataset_dir: str) -> str:
    dataset_dir = os.path.expanduser(dataset_dir)

    if os.path.isabs(dataset_dir):
        return dataset_dir

    # Try reading the appconfig.yaml file to get the root directory.
    appconfig_path = os.path.abspath(os.path.join(__file__, "../../../appconfig.yaml"))
    try:
        with open(appconfig_path, "r") as infile:
            config = yaml.safe_load(infile)

            if "root_datasets_directory" in config:
                return os.path.join(config["root_datasets_directory"], dataset_dir)
    except FileNotFoundError:
        warnings.warn(f"{appconfig_path} not found.")

    return dataset_dir


def read_filenames(path: str) -> Set[str]:
    image_absdir = os.path.join(path)
    filenames = set([f.path for f in os.scandir(image_absdir)])
    return filenames


def get_unique_in_columns(arr: np.ndarray) -> np.ndarray:
    n_classes: List[int] = []
    n_cols = arr.shape[1]
    for i in range(n_cols):
        n_classes.append(len(np.unique(arr[:, i])))
    return np.array(n_classes)


def limit_dataset_samples(
    x: np.ndarray, y: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Assumes that identity is colum 0 in y"""

    # If the limit is greater than the input data, just return the inputs.
    if n >= y.shape[0]:
        return x, y

    # Sort the inputs by identity for use in groupby.
    sort_indexs = np.argsort(y[:, 0])
    x = x[sort_indexs]
    y = y[sort_indexs, :]

    # Group all of the samples into a dict, with identity as the key.
    sample_iters_per_id = dict(
        (k, list(v)) for k, v in itertools.groupby(zip(x, y), lambda xy: xy[1][0])
    )

    new_x = []
    new_y = []

    # Iterate through the identies.
    # Get 1 sample at a time to ensure even distribution.
    while len(new_y) < n:
        for id in sample_iters_per_id:
            try:
                sample_x, sample_y = sample_iters_per_id[id].pop()
                new_x.append(sample_x)
                new_y.append(sample_y)
            except IndexError:
                continue

    assert len(new_x) == len(new_y)
    assert len(new_y) >= n

    # Cut the number of samples to the desired length.
    if len(new_y) > n:
        new_x = new_x[:n]
        new_y = new_y[:n]

    return np.array(new_x), np.array(new_y)
