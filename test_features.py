import os
from argparse import ArgumentParser
from typing import Generator, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import default_rng
from ruyaml import YAML
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from jlu.data import VGGFace2WithMaadFace
from jlu.metrics import CVThresholdingVerifier, ReidentificationTester
from jlu.systems import JLUTrainer
from jlu.utils import find_last_epoch_path, find_last_path, save_cv_verification_results

pl.seed_everything(42, workers=True)


def main(
    experiment_path: str, batch_size: int, is_debug: bool, cluster_only: bool,
):

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
        # CelebA(batch_size, ["Male"], buffer_size=None),
        VGGFace2WithMaadFace(batch_size, val_split=0),
    ]

    # Get the device.
    device = torch.device("cuda:0")
    feature_model.to(device)

    for test_module in datamodules:

        test_module.setup("test")
        test_dataloader = test_module.test_dataloader()
        assert isinstance(test_dataloader, DataLoader)
        test_dataset = test_dataloader.dataset
        dataset_name = os.path.basename(test_module.dataset_dir)

        print(f"Dataset Module: {dataset_name}")

        cluster(
            experiment_path,
            dataset_name,
            test_dataloader,
            feature_model,
            device,
            is_debug=is_debug,
        )

        if cluster_only:
            continue

        visualise(
            experiment_path,
            dataset_name,
            test_dataloader,
            feature_model,
            device,
            is_debug=is_debug,
        )

        id_index = test_dataset.labels.index("id")
        n_class_schedule = [10] + list(
            n_classes_scheduler(len(test_dataset.support_per_label[id_index]))
        )
        for n_classes in n_class_schedule:

            print(f"N Classes: {n_classes}")

            reid_scenario(
                experiment_path,
                dataset_name,
                n_classes,
                batch_size,
                is_debug,
                test_dataloader,
                feature_model,
                device,
            )

            verification_scenario(
                experiment_path,
                dataset_name,
                n_classes,
                batch_size,
                is_debug,
                test_dataloader,
                feature_model,
                device,
            )


def cluster(
    experiment_path: str,
    dataset_name: str,
    test_dataloader: DataLoader,
    feature_model: torch.nn.Module,
    device: torch.device,
    perplexity=50,
    is_debug=False,
):

    results_dir = os.path.join(
        experiment_path, "feature_tests", "clustering", dataset_name
    )
    os.makedirs(results_dir, exist_ok=True)

    embeddings_per_batch: List[np.ndarray] = []
    attributes_per_batch: List[np.ndarray] = []

    attribute_index = test_dataloader.dataset.labels.index("sex")

    for x, y in tqdm.tqdm(
        test_dataloader, desc="Clustering - Embeddings", dynamic_ncols=True
    ):
        a = y[:, attribute_index]
        x_ = feature_model(x.to(device))

        if isinstance(x_, tuple):
            x_ = x_[1]

        embeddings_per_batch.append(x_.cpu().detach().numpy())
        attributes_per_batch.append(a.cpu().detach().numpy())

        if is_debug:
            if len(np.unique(np.concatenate(attributes_per_batch))) > 1:
                break

    all_embeddings = np.concatenate(embeddings_per_batch)
    all_attributes = np.concatenate(attributes_per_batch).squeeze()

    if len(all_embeddings) > 10000:
        rng = default_rng()
        idxs = rng.integers(0, len(all_embeddings), size=10000)
        all_embeddings = all_embeddings[idxs]
        all_attributes = all_attributes[idxs]

    reduced_data = TSNE(
        n_components=2, perplexity=perplexity, n_iter=5000, random_state=42, verbose=1
    ).fit_transform(all_embeddings)
    scaled_data = StandardScaler().fit_transform(reduced_data)

    model = GaussianMixture(n_components=2)
    cluster_labels = model.fit_predict(scaled_data)

    full_data = pd.DataFrame.from_dict(
        {"cluster_id": cluster_labels, "true_attribute": all_attributes}
    )
    full_data.to_csv(os.path.join(results_dir, "raw_data.csv"))

    # Visualise
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = scaled_data[:, 0].min() - 1, scaled_data[:, 0].max() + 1
    y_min, y_max = scaled_data[:, 1].min() - 1, scaled_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = model.predict(
        np.c_[xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)]
    )

    # Put the result into a color plot and plot the mesh.
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    colourmap = ListedColormap(["plum", "palegreen"])
    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=colourmap,
        aspect="auto",
        origin="lower",
    )

    # Plot Male sample.
    male_mask = full_data["true_attribute"] == 1
    ax.plot(
        scaled_data[male_mask][:, 0],
        scaled_data[male_mask][:, 1],
        "x",
        markersize=2,
        c="blue",
        label="Male",
    )

    # Plot female.
    ax.plot(
        scaled_data[male_mask == np.array(False)][:, 0],
        scaled_data[male_mask == np.array(False)][:, 1],
        "+",
        markersize=2,
        c="red",
        label="Female",
    )

    # Add the legend.
    ax.legend()

    # Plot the centroids.
    # centroids = model.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
    #             color="w", zorder=10)

    # Save chart
    fig.savefig(os.path.join(results_dir, f"chart.png"))
    fig.savefig(os.path.join(results_dir, f"chart.eps"))

    def calc_metrics(cluster_ids: pd.Series, real_attribute_labels: pd.Series):
        metrics = {
            "accuracy": accuracy_score(real_attribute_labels, cluster_ids),
            "balanced_accuracy": balanced_accuracy_score(
                real_attribute_labels, cluster_ids
            ),
            "precision": precision_score(real_attribute_labels, cluster_ids),
            "recall": recall_score(real_attribute_labels, cluster_ids),
            "f1": f1_score(real_attribute_labels, cluster_ids),
        }

        metrics = {m: float(metrics[m]) for m in metrics}

        return metrics

    cluster_0_is_att_0_metrics = calc_metrics(
        full_data["cluster_id"], full_data["true_attribute"]
    )

    opposite_cluster_ids = (full_data["cluster_id"] == np.array(False)).astype(int)
    cluster_1_is_att_0_metrics = calc_metrics(
        opposite_cluster_ids, full_data["true_attribute"]
    )

    yaml = YAML(typ="safe")
    with open(os.path.join(results_dir, "cluster_0_is_att_0.yaml"), "w") as outfile:
        yaml.dump(cluster_0_is_att_0_metrics, outfile)

    with open(os.path.join(results_dir, "cluster_1_is_att_0.yaml"), "w") as outfile:
        yaml.dump(cluster_1_is_att_0_metrics, outfile)


def visualise(
    experiment_path: str,
    dataset_name: str,
    test_dataloader: DataLoader,
    feature_model: torch.nn.Module,
    device: torch.device,
    perplexity=30,
    is_debug=False,
):

    results_dir = os.path.join(
        experiment_path, "feature_tests", "visualisations", dataset_name
    )
    os.makedirs(results_dir, exist_ok=True)

    embeddings_per_batch: List[np.ndarray] = []
    attributes_per_batch: List[np.ndarray] = []
    attribute_index = test_dataloader.dataset.labels.index("sex")

    for x, y in tqdm.tqdm(
        test_dataloader, desc="Visualiser - Embeddings", dynamic_ncols=True
    ):
        a = y[:, attribute_index]
        x_ = feature_model(x.to(device))

        if isinstance(x_, tuple):
            x_ = x_[1]

        embeddings_per_batch.append(x_.cpu().detach().numpy())
        attributes_per_batch.append(a.cpu().detach().numpy())

        if is_debug:
            if len(np.unique(np.concatenate(attributes_per_batch))) > 1:
                break

    all_embeddings = np.concatenate(embeddings_per_batch)
    all_attributes = np.concatenate(attributes_per_batch)

    if len(all_attributes) > 10000:
        rng = default_rng()
        idxs = rng.integers(0, len(all_attributes), size=10000)
        all_embeddings = all_embeddings[idxs]
        all_attributes = all_attributes[idxs]

    tsne = TSNE(
        n_components=2, perplexity=perplexity, n_iter=5000, random_state=42, verbose=1
    )
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    full_data = np.concatenate(
        (reduced_embeddings, np.expand_dims(all_attributes, axis=1)), axis=1
    )

    # Save numpy array.
    np.save(os.path.join(results_dir, f"TSNE_p{perplexity}.npy"), reduced_embeddings)

    # Save chart.
    df = pd.DataFrame(data=full_data, columns=["x", "y", "h"])
    df["h"] = df["h"].astype(int)
    chart = sns.lmplot(
        data=df,
        x="x",
        y="y",
        hue="h",
        markers=["x", "+"],
        fit_reg=False,
        scatter_kws={"s": 10},
        legend=False,
        palette=["red", "blue"],
    )
    chart.savefig(os.path.join(results_dir, f"TSNE_p{perplexity}.png"))
    chart.savefig(os.path.join(results_dir, f"TSNE_p{perplexity}.svg"))


def reid_scenario(
    experiment_path: str,
    dataset_name: str,
    n_classes: int,
    batch_size: int,
    is_debug: bool,
    test_dataloader: DataLoader,
    feature_model: torch.nn.Module,
    device: torch.device,
):
    results_dir = os.path.join(
        experiment_path, "feature_tests", "reid", dataset_name, str(n_classes)
    )
    os.makedirs(results_dir, exist_ok=True)

    # Get the tester.
    tester = ReidentificationTester(
        batch_size, max_n_classes=n_classes, debug=is_debug, seed=42
    )

    # load the data.
    tester.setup(test_dataloader)

    # Get the avg rank over all probes.
    avg_ranks = tester.reidentification(feature_model, device)

    df = pd.DataFrame(
        avg_ranks, columns=["Cumulative Accuracy"], index=range(1, len(avg_ranks) + 1)
    )
    df.index.name = "Rank"

    df.to_csv(os.path.join(results_dir, "avg_ranks.csv"))


def verification_scenario(
    experiment_path: str,
    dataset_name: str,
    n_classes: int,
    batch_size: int,
    is_debug: bool,
    test_dataloader: DataLoader,
    feature_model: torch.nn.Module,
    device: torch.device,
):

    results_dir = os.path.join(
        experiment_path, "feature_tests", "verification", dataset_name, str(n_classes)
    )
    os.makedirs(results_dir, exist_ok=True)

    verifier = CVThresholdingVerifier(
        batch_size, max_n_classes=n_classes, debug=is_debug
    )

    verifier.setup(test_dataloader)

    (
        metrics_rocs,
        per_attribute_pair_metrics_rocs,
    ) = verifier.cv_thresholding_verification(feature_model, device)

    # Save the combined cv verification results.
    save_cv_verification_results(metrics_rocs, results_dir, "_all")

    # Save the attribuet pair results.
    for ap in per_attribute_pair_metrics_rocs:
        ap_suffix = f"_{ap[0]}_{ap[1]}"
        save_cv_verification_results(
            per_attribute_pair_metrics_rocs[ap], results_dir, ap_suffix
        )


def n_classes_scheduler(
    total_n_classes: int, start=25, step_f=lambda x: 2 * x
) -> Generator[int, None, None]:
    x = start
    yield x

    while x < total_n_classes:
        x = step_f(x)

        if x < total_n_classes:
            yield x
        else:
            yield total_n_classes
            return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_path")
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cluster-only", action="store_true")
    args = parser.parse_args()
    main(args.experiment_path, args.batch_size, args.debug, args.cluster_only)
