import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from jlu.callbacks import JLUAwareEarlyStopping
from jlu.data import UTKFace
from jlu.systems import JLUTrainer


def main(hparams):
    # Create results directory.
    os.makedirs(hparams.output_directory, exist_ok=True)

    # Get datamodelule.
    datamodule: UTKFace = load_datamodule(**vars(hparams))
    datamodule.setup()

    # Construct model.
    model = JLUTrainer(
        **vars(hparams),
        datamodule_n_classes=datamodule.n_classes,
        datamodule_labels=datamodule.labels
    )

    # Get Trainer.
    trainer = pl.Trainer(
        default_root_dir=hparams.output_directory,
        gpus=1,
        auto_select_gpus=True,
        log_every_n_steps=50,
        benchmark=True,
        logger=True,
    )

    # Train
    trainer.fit(model, datamodule)


def load_datamodule(datamodule: str, **kwargs) -> UTKFace:
    if datamodule.lower() == "utkface":
        return UTKFace(**kwargs)
    else:
        raise ValueError


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--datamodule", "-d", default="UTKFace")
    parser.add_argument("--primary-task", "-p", default="age")
    parser.add_argument("--secondary-task", "-s", default=["sex"], action="append")
    parser.add_argument("--alpha", "-a", default=0.1, type=float)
    parser.add_argument("--learning-rate", "-lr", default=1e-6, type=float)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--bootstrap-epochs", default=0, type=int)
    hyperparams = parser.parse_args()

    main(hyperparams)
