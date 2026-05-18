import yaml
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.datasets.change_detection_dataset import ChangeDetectionDataset

from src.models.unet import build_unet_model
from src.models.unetplusplus import UNetPlusPlusModel

from src.metrics.metrics import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix
)

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    required=True
)

parser.add_argument(
    "--weights",
    type=str,
    required=True
)

args = parser.parse_args()

# LOAD CONFIG
config_path = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# DATASET
test_dataset = ChangeDetectionDataset(
    root_dir=args.data_path,
    split="test",
    image_size=config["dataset"]["image_size"],
    augment=False
)

# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["dataset"]["num_workers"]
)

# MODEL
if config["model"]["name"] == "unet":

    model = build_unet_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    )

elif config["model"]["name"] == "unetplusplus":

    model = UNetPlusPlusModel()

else:
    raise ValueError("Invalid model name!")

# LOAD WEIGHTS
checkpoint = torch.load(
    args.weights,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")

# METRICS
test_iou = 0.0
test_precision = 0.0
test_recall = 0.0
test_f1 = 0.0

valid_batches = 0

all_preds = []
all_targets = []

with torch.no_grad():

    progress_bar = tqdm(test_loader)

    for batch in progress_bar:

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        valid_batches += 1

        test_iou += compute_iou(outputs, masks)
        test_precision += compute_precision(outputs, masks)
        test_recall += compute_recall(outputs, masks)
        test_f1 += compute_f1(outputs, masks)

        all_preds.append(
            preds.cpu().numpy()
        )

        all_targets.append(
            masks.cpu().numpy()
        )

# AVERAGES
test_iou /= valid_batches
test_precision /= valid_batches
test_recall /= valid_batches
test_f1 /= valid_batches

# CONFUSION MATRIX
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

cm = compute_confusion_matrix(
    all_targets,
    all_preds
)

# PRINT RESULTS
print("\nTEST RESULTS")

print(f"IoU       : {test_iou:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")

print("\nCONFUSION MATRIX")
print(cm)