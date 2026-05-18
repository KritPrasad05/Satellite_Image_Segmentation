import os
import yaml

from pathlib import Path
from tqdm import tqdm
import gc
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.datasets.change_detection_dataset import ChangeDetectionDataset
from src.datasets.samplers import create_weighted_sampler
from src.models.unet import build_unet_model
from src.models.unetplusplus import UNetPlusPlusModel
from src.losses.losses import FocalDiceLoss
from src.metrics.metrics import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix
)
from src.utils.logger import (
    initialize_csv_logger,
    log_metrics
)

from src.utils.scheduler import get_scheduler
from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
from src.utils.seed import set_seed

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LOAD CONFIG
config_path = PROJECT_ROOT / "configs" / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]

model_output_dir = os.path.join(
    "outputs",
    MODEL_NAME
)

os.makedirs(model_output_dir, exist_ok=True)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

#SEED
set_seed(
    config["training"]["seed"]
)

# DATASETS
train_dataset = ChangeDetectionDataset(
    root_dir=str(PROJECT_ROOT / "dataset"),
    split="train",
    image_size=config["dataset"]["image_size"],
    augment=True
)

val_dataset = ChangeDetectionDataset(
    root_dir=str(PROJECT_ROOT / "dataset"),
    split="val",
    image_size=config["dataset"]["image_size"],
    augment=False
)

# WEIGHTED SAMPLER
sampler = create_weighted_sampler(
    target_dir=str(
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "target"
    ),
    file_names=train_dataset.file_names
)

# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    sampler=sampler,
    num_workers=config["dataset"]["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["dataset"]["num_workers"],
    pin_memory=True
)

# MODEL
if config["model"]["name"] == "unet":

    model = build_unet_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    )

elif config["model"]["name"] == "unetplusplus":

    model = UNetPlusPlusModel()

else:
    raise ValueError("Invalid model name!")

model = model.to(device)

# LOSS FUNCTION
criterion = FocalDiceLoss(
    focal_weight=config["loss"]["focal_weight"],
    dice_weight=config["loss"]["dice_weight"]
)

# OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

# LR SCHEDULER
scheduler = get_scheduler(
    optimizer=optimizer,
    scheduler_name=config["scheduler"]["name"],
    epochs=config["training"]["epochs"]
)

# CHECKPOINT DIRECTORY
checkpoint_dir = os.path.join(
    model_output_dir,
    "checkpoints"
)
os.makedirs(checkpoint_dir, exist_ok=True)

# LOG DIRECTORY
log_dir = os.path.join(
    model_output_dir,
    "logs"
)

os.makedirs(log_dir, exist_ok=True)

csv_log_path = os.path.join(
    log_dir,
    f"{MODEL_NAME}_training_metrics.csv"
)

initialize_csv_logger(csv_log_path)

# TRAINING FUNCTION
def main():
    best_iou = 0.0

    start_epoch = 0
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"best_{MODEL_NAME}_model.pth"
    )
    if os.path.exists(checkpoint_path):
        print("\nResuming from checkpoint...")
        start_epoch, best_iou = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        start_epoch += 1

        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(
        start_epoch,
        config["training"]["epochs"]
    ):

        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")

        # TRAINING
        model.train()
        
        train_loss = 0.0
        train_iou = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        
        valid_train_batches = 0
        train_progress = tqdm(train_loader)

        for batch in train_progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            if torch.isnan(loss):
                continue
                
            valid_train_batches += 1

            loss.backward()
            optimizer.step()
            
            # METRICS
            train_loss += loss.item()
            train_iou += compute_iou(outputs, masks)
            train_precision += compute_precision(outputs, masks)
            train_recall += compute_recall(outputs, masks)
            train_f1 += compute_f1(outputs, masks)
            train_progress.set_postfix({
                "Loss": loss.item()
            })

        # EPOCH AVERAGES
        train_loss /= valid_train_batches
        train_iou /= valid_train_batches
        train_precision /= valid_train_batches
        train_recall /= valid_train_batches
        train_f1 /= valid_train_batches

        # VALIDATION
        model.eval()

        val_loss = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        valid_val_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            val_progress = tqdm(val_loader)
            
            for batch in val_progress:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                loss = criterion(outputs, masks)
                if torch.isnan(loss):
                    continue
                    
                valid_val_batches += 1
                val_loss += loss.item()
                val_iou += compute_iou(outputs, masks)
                val_precision += compute_precision(outputs, masks)
                val_recall += compute_recall(outputs, masks)
                val_f1 += compute_f1(outputs, masks)
                all_preds.append(
                    preds.cpu().numpy()
                )
                
                all_targets.append(
                    masks.cpu().numpy()
                )

        # VALIDATION AVERAGES
        val_loss /= valid_val_batches
        val_iou /= valid_val_batches
        val_precision /= valid_val_batches
        val_recall /= valid_val_batches
        val_f1 /= valid_val_batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        cm = compute_confusion_matrix(
            all_targets,
            all_preds
        )
        
        print("\nCONFUSION MATRIX")
        print(cm)
        np.save(
            os.path.join(
                checkpoint_dir,
                f"confusion_matrix_epoch_{epoch+1}.npy"
            ),
            cm
        )

        # LR SCHEDULER
        if config["scheduler"]["name"] == "plateau":
            scheduler.step(val_iou)
        else:
            scheduler.step()
        torch.cuda.empty_cache()
        
        gc.collect()

        # PRINT RESULTS
        print("\nTRAIN RESULTS")

        print(f"Loss      : {train_loss:.4f}")
        print(f"IoU       : {train_iou:.4f}")
        print(f"Precision : {train_precision:.4f}")
        print(f"Recall    : {train_recall:.4f}")
        print(f"F1 Score  : {train_f1:.4f}")

        print("\nVALIDATION RESULTS")

        print(f"Loss      : {val_loss:.4f}")
        print(f"IoU       : {val_iou:.4f}")
        print(f"Precision : {val_precision:.4f}")
        print(f"Recall    : {val_recall:.4f}")
        print(f"F1 Score  : {val_f1:.4f}")

        # SAVE METRICS
        log_metrics(
            csv_path=csv_log_path,
            epoch=epoch + 1,

            train_loss=train_loss,
            train_iou=train_iou,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,

            val_loss=val_loss,
            val_iou=val_iou,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1
        )
        
        # SAVE BEST MODEL
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"best_{MODEL_NAME}_model.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_iou=best_iou,            
                checkpoint_path=checkpoint_path
            )
            print("\nBest model saved!")

# MAIN
if __name__ == "__main__":
    main()