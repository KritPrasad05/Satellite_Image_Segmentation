import torch
import torch.nn as nn
import torch.nn.functional as F

# DICE LOSS
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        predictions = predictions.float()
        targets = targets.float()
        intersection = (
            predictions * targets
        ).sum()

        denominator = (
            predictions.sum()
            +
            targets.sum()
        )

        dice_score = (
            (2.0 * intersection + self.smooth)
            /
            (denominator + self.smooth)
        )

        dice_score = torch.clamp(
            dice_score,
            min=0.0,
            max=1.0
        )

        return 1 - dice_score

# FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.8,
        gamma=2
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none"
        )

        probabilities = torch.sigmoid(predictions)

        pt = torch.where(
            targets == 1,
            probabilities,
            1 - probabilities
        )
        focal_loss = (
            self.alpha
            *
            (1 - pt) ** self.gamma
            *
            bce_loss
        )
        return focal_loss.mean()

# FOCAL + DICE LOSS
class FocalDiceLoss(nn.Module):

    def __init__(
        self,
        focal_weight=1.0,
        dice_weight=1.0
    ):

        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):

        focal_loss = self.focal(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        total_loss = (
            self.focal_weight * focal_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss