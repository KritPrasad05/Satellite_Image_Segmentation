import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from src.datasets.change_detection_dataset import ChangeDetectionDataset
from src.models.unet import build_unet_model
from src.models.unetplusplus import UNetPlusPlusModel

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# CONFIG
IMAGE_SIZE = 512
BATCH_SIZE = 2
THRESHOLD = 0.35

#Parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "--weights",
    type=str,
    required=True
)

args = parser.parse_args()

# DATASET
test_dataset = ChangeDetectionDataset(
    root_dir=str(PROJECT_ROOT / "dataset"),
    split="test",
    image_size=IMAGE_SIZE,
    augment=False
)

# DATALOADER
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# MODEL
config_path = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]

if MODEL_NAME == "unet":

    model = build_unet_model(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=4,
        classes=1
    )

elif MODEL_NAME == "unetplusplus":

    model = UNetPlusPlusModel()

else:
    raise ValueError("Invalid   model name")

# OUTPUT DIRECTORY
prediction_dir = (
    PROJECT_ROOT
    / "outputs"
    / MODEL_NAME
    / "predictions"
)

os.makedirs(prediction_dir, exist_ok=True)

# LOAD CHECKPOINT
checkpoint_path = args.weights

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()
print("Model loaded successfully!")

# TTA FUNCTION
def tta_predict(model, images):
    pred1 = torch.sigmoid(model(images))
    flipped_images = torch.flip(images, dims=[3])
    pred2 = torch.sigmoid(model(flipped_images))
    pred2 = torch.flip(pred2, dims=[3])

    predictions = (
        pred1 + pred2
    ) / 2.0

    return predictions

# POST PROCESSING
def post_process(mask):
    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

# INFERENCE LOOP
with torch.no_grad():
    progress_bar = tqdm(test_loader)

    for batch in progress_bar:
        images = batch["image"].to(device)
        file_names = batch["file_name"]
        probabilities = tta_predict(
            model,
            images
        )

        predictions = (
            probabilities > THRESHOLD
        ).float()
        predictions = predictions.cpu().numpy()

        for i in range(len(predictions)):
            pred_mask = predictions[i][0]
            pred_mask = (
                pred_mask * 255
            ).astype(np.uint8)

            pred_mask = post_process(pred_mask)
            file_name = file_names[i]
            save_path = (
                prediction_dir
                / file_name.replace(".tif", ".png")
            )

            cv2.imwrite(
                str(save_path),
                pred_mask
            )

print("\nInference Completed!")