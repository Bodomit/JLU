import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from ruyaml import YAML

from jlu.data import VGGFace2WithMaadFace
from jlu.systems import AttributeExtractionTask, JLUTrainer
from jlu.utils import find_last_epoch_path, find_last_path

pl.seed_everything(42, workers=True)


def main(experiment_path: str, batch_size: int, is_fast_dev_run: bool):
    # Get model class form config and path to the last checkpoint.
    try:
        feature_model_checkpoint_path = find_last_epoch_path(experiment_path)
    except ValueError:
        feature_model_checkpoint_path = find_last_path(experiment_path)

    print("Experiment Directory: ", experiment_path)
    print("Checkpoint Path; ", feature_model_checkpoint_path)

    # Load the jlu trainer and get the base (feature) model.
    jlu_trainer = JLUTrainer.load_from_checkpoint(
        feature_model_checkpoint_path, verifier_args=None, pretrained_base=None
    )

    feature_model = jlu_trainer.model.feature_base
    feature_model.eval()
    feature_model.freeze()
    del jlu_trainer

    # Construct the datamodules.
    datamodules = [
        VGGFace2WithMaadFace(batch_size, buffer_size=None),
        # CelebA(batch_size, buffer_size=None),
    ]

    yaml = YAML(typ="safe")

    for train_module in datamodules:

        # Get directory for this test.
        root_dir = os.path.join(
            experiment_path, "aems", os.path.basename(train_module.dataset_dir)
        )

        # Constuct loggers.
        loggers = [
            CSVLogger(save_dir=root_dir, name="csv_logs"),
            TensorBoardLogger(save_dir=root_dir, name="tb_logs"),
        ]

        # Train the attribute extraction model.
        aem_task = AttributeExtractionTask(feature_model, 0.001)
        trainer = pl.Trainer(
            logger=loggers,
            max_epochs=10,
            default_root_dir=root_dir,
            gpus=1,
            auto_select_gpus=True,
            benchmark=True,
            fast_dev_run=is_fast_dev_run,
        )
        trainer.fit(aem_task, train_module)
        aem_task.freeze()

        # Test each data module.
        for test_module in datamodules:
            results_path = os.path.join(
                root_dir, f"{os.path.basename(test_module.dataset_dir)}.yaml"
            )

            results = trainer.test(aem_task, datamodule=test_module)

            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as outfile:
                yaml.dump(results[0], outfile)

        del trainer
        del aem_task


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.experiment_path, args.batch_size, args.debug)
