import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from jlu.data import VGGFace2, load_datamodule
from jlu.systems import VggMPretrainer


def main(hparams):
    # Create results directory.
    os.makedirs(hparams.output_directory, exist_ok=True)

    # Get datamodelule.
    datamodule = load_datamodule(**vars(hparams))
    datamodule.setup()

    # Construct model.
    model = VggMPretrainer(**vars(hparams), n_classes=int(datamodule.n_classes))

    # Get Trainer.
    trainer = pl.Trainer(
        default_root_dir=hparams.output_directory,
        gpus=1,
        auto_select_gpus=True,
        log_every_n_steps=50,
        benchmark=True,
        logger=True,
        auto_lr_find=True,
    )

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--datamodule", "-d", default="VggFace2")
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    hyperparams = parser.parse_args()

    main(hyperparams)
