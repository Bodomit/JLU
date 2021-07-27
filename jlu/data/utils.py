import os
import warnings
from typing import Set

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
