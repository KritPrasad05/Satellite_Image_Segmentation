import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LOAD CONFIG
config_path = PROJECT_ROOT / "configs" / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]

# PATH
csv_path = f"{PROJECT_ROOT}/outputs/{MODEL_NAME}/logs/{MODEL_NAME}_training_metrics.csv"

# LOAD CSV
df = pd.read_csv(csv_path)

# OUTPUT DIR
plot_dir = f"{PROJECT_ROOT}/outputs/{MODEL_NAME}/plots"

os.makedirs(plot_dir, exist_ok=True)

# LOSS CURVE
plt.figure(figsize=(8,5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    label="Train Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.savefig(
    os.path.join(plot_dir, "loss_curve.png")
)

plt.close()

# IOU CURVE
plt.figure(figsize=(8,5))

plt.plot(
    df["epoch"],
    df["train_iou"],
    label="Train IoU"
)

plt.plot(
    df["epoch"],
    df["val_iou"],
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU Curve")
plt.legend()

plt.savefig(
    os.path.join(plot_dir, "iou_curve.png")
)

plt.close()

print("Plots saved successfully!")