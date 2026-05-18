import torch

# GET SCHEDULER
def get_scheduler(
    optimizer,
    scheduler_name,
    epochs
):

    # COSINE ANNEALING
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    # STEP LR
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5
        )

    # REDUCE LR ON PLATEAU
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )
    return scheduler