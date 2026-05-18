import os
from pathlib import Path

import cv2
import numpy as np
import yaml
import tifffile as tiff

import matplotlib.pyplot as plt

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# MODEL
config_path = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]


# VISUALIZE MULTIPLE SAMPLES
def visualize_samples(
    split="test",
    file_names=None,
    num_samples=5
):

    # AUTO FILE SELECTION
    if file_names is None:
        pre_dir = (
            PROJECT_ROOT
            / "dataset"
            / split
            / "pre-event"
        )

        file_names = sorted(
            os.listdir(pre_dir)
        )[:num_samples]

    # LOOP THROUGH FILES
    for file_name in file_names:

        # PATHS
        pre_path = (
            PROJECT_ROOT
            / "dataset"
            / split
            / "pre-event"
            / file_name
        )

        post_path = (
            PROJECT_ROOT
            / "dataset"
            / split
            / "post-event"
            / file_name
        )

        target_path = (
            PROJECT_ROOT
            / "dataset"
            / split
            / "target"
            / file_name
        )

        prediction_path = (
            PROJECT_ROOT
            / "outputs"
            / MODEL_NAME
            / "predictions"
            / file_name.replace(".tif", ".png")
        )
        # LOAD IMAGES
        eo_image = tiff.imread(
            str(pre_path)
        )

        sar_image = tiff.imread(
            str(post_path)
        )

        target_mask = tiff.imread(
            str(target_path)
        )

        prediction_mask = cv2.imread(
            str(prediction_path),
            cv2.IMREAD_GRAYSCALE
        )
        # REMAP TARGET
        binary_target = np.zeros_like(
            target_mask
        )

        binary_target[
            (target_mask == 2)
            |
            (target_mask == 3)
        ] = 1

        # PLOT
        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20,5)
        )

        # EO IMAGE
        axes[0].imshow(eo_image)
        axes[0].set_title("EO Image")
        axes[0].axis("off")

        # SAR IMAGE
        axes[1].imshow(
            sar_image,
            cmap="gray"
        )
        axes[1].set_title("SAR Image")
        axes[1].axis("off")

        # GROUND TRUTH
        axes[2].imshow(
            binary_target,
            cmap="gray"
        )
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # PREDICTION
        axes[3].imshow(
            prediction_mask,
            cmap="gray"
        )

        axes[3].set_title("Prediction")
        axes[3].axis("off")
        plt.suptitle(file_name)
        plt.tight_layout()
        plt.show()

def visualize_non_empty_samples(
    split="test",
    num_samples=5
):

    target_dir = (
        PROJECT_ROOT
        / "dataset"
        / split
        / "target"
    )

    selected_files = []

    for file_name in sorted(os.listdir(target_dir)):
        target_path = (
            target_dir
            / file_name
        )

        mask = tiff.imread(
            str(target_path)
        )

        binary_mask = np.zeros_like(mask)
        binary_mask[
            (mask == 2)
            |
            (mask == 3)
        ] = 1

        if binary_mask.sum() > 0:
            selected_files.append(file_name)

        if len(selected_files) >= num_samples:
            break

    visualize_samples(
        split=split,
        file_names=selected_files
    )