import csv
import os

def initialize_csv_logger(csv_path):

    if not os.path.exists(csv_path):

        with open(csv_path, mode="w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "train_iou",
                "train_precision",
                "train_recall",
                "train_f1",
                "val_loss",
                "val_iou",
                "val_precision",
                "val_recall",
                "val_f1"
            ])

def log_metrics(
    csv_path,
    epoch,
    train_loss,
    train_iou,
    train_precision,
    train_recall,
    train_f1,
    val_loss,
    val_iou,
    val_precision,
    val_recall,
    val_f1
):

    with open(csv_path, mode="a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            train_iou,
            train_precision,
            train_recall,
            train_f1,
            val_loss,
            val_iou,
            val_precision,
            val_recall,
            val_f1
        ])