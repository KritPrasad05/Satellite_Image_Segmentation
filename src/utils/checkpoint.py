import torch

# SAVE CHECKPOINT
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_iou,
    checkpoint_path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(f"\nCheckpoint saved at: {checkpoint_path}")

# LOAD CHECKPOINT
def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cuda"
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
    epoch = checkpoint["epoch"]
    best_iou = checkpoint["best_iou"]
    print(f"\nCheckpoint loaded from: {checkpoint_path}")
    return epoch, best_iou