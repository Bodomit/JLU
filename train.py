import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

from jlu.data import UTKFace, load_datamodule
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
        max_epochs=sys.maxsize,
        resume_from_checkpoint=hparams.resume_from,
    )

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--datamodule", "-d", default="UTKFace")
    parser.add_argument("--primary-task", "-p", default="age")
    parser.add_argument("--secondary-task", "-s", default=["sex"], action="append")
    parser.add_argument("--alpha", "-a", default=0.1, type=float)
    parser.add_argument("--learning-rate", "-lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--bootstrap-epochs", default=0, type=int)
    parser.add_argument("--resume-from", default=None, type=str)
    hyperparams = parser.parse_args()

    main(hyperparams)
