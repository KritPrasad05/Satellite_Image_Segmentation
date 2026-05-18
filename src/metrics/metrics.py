import torch
from sklearn.metrics import confusion_matrix

# APPLY THRESHOLD
def threshold_predictions(
    predictions,
    threshold=0.5
):

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > threshold
    ).float()

    return predictions

# IoU SCORE
def compute_iou(
    predictions,
    targets,
    smooth=1e-6
):

    predictions = threshold_predictions(
        predictions
    )

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions.sum()
        +
        targets.sum()
        -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()

# PRECISION
def compute_precision(
    predictions,
    targets,
    smooth=1e-6
):

    predictions = threshold_predictions(
        predictions
    )

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    true_positive = (
        predictions * targets
    ).sum()

    false_positive = (
        predictions * (1 - targets)
    ).sum()

    precision = (
        true_positive + smooth
    ) / (
        true_positive
        +
        false_positive
        +
        smooth
    )

    return precision.item()

# RECALL
def compute_recall(
    predictions,
    targets,
    smooth=1e-6
):

    predictions = threshold_predictions(
        predictions
    )

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    true_positive = (
        predictions * targets
    ).sum()

    false_negative = (
        (1 - predictions) * targets
    ).sum()

    recall = (
        true_positive + smooth
    ) / (
        true_positive
        +
        false_negative
        +
        smooth
    )

    return recall.item()

# F1 SCORE
def compute_f1(
    predictions,
    targets,
    smooth=1e-6
):

    precision = compute_precision(
        predictions,
        targets,
        smooth
    )

    recall = compute_recall(
        predictions,
        targets,
        smooth
    )

    f1 = (
        2
        *
        precision
        *
        recall
    ) / (
        precision
        +
        recall
        +
        smooth
    )

    return f1

def compute_confusion_matrix(
    y_true,
    y_pred
):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return cm