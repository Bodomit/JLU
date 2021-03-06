import os
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl

from jlu.data import load_datamodule
from jlu.systems import JLUTrainer
from jlu.systems.trainer import Trainer
from jlu.systems.vggm_pretrainer import VggMPretrainer


def main(hparams):
    # Create results directory.
    os.makedirs(hparams.output_directory, exist_ok=True)

    # Get datamodelule.
    datamodule = load_datamodule(**vars(hparams))
    datamodule.setup(None)

    if hparams.pretrained:
        pretrained_model = VggMPretrainer.load_from_checkpoint(hparams.pretrained)
        pretrained_base = pretrained_model.model.base
        pretrained_base = pretrained_base
    else:
        pretrained_base = None

    # Construct model.
    if hparams.primary_only:
        max_epochs = 1000
        system_class = Trainer
    else:
        max_epochs = sys.maxsize
        system_class = JLUTrainer

    system = system_class(
        **vars(hparams),
        pretrained_base=pretrained_base,
        datamodule_n_classes=datamodule.n_classes,  # type: ignore
        datamodule_labels=datamodule.labels  # type: ignore
    )

    # Get Trainer.
    trainer = pl.Trainer(
        default_root_dir=hparams.output_directory,
        gpus=1,
        auto_select_gpus=True,
        log_every_n_steps=50,
        benchmark=True,
        logger=True,
        max_epochs=max_epochs,
        resume_from_checkpoint=hparams.resume_from,
    )

    # Train
    trainer.fit(system, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_directory")
    parser.add_argument("--datamodule", "-d", default="vggface2_maadface")
    parser.add_argument("--primary-task", "-p", default="id")
    parser.add_argument("--secondary-task", "-s", default=["sex"], action="append")
    parser.add_argument("--alpha", "-a", default=0.1, type=float)
    parser.add_argument("--learning-rate", "-lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--bootstrap-epochs", default=0, type=int)
    parser.add_argument("--resume-from", default=None, type=str)
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--random-affine", action="store_true")
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--dropout", default=0.7, type=float)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--feature-model", default="vggm")
    hyperparams = parser.parse_args()

    main(hyperparams)
