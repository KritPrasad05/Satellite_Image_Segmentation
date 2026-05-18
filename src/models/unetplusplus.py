import segmentation_models_pytorch as smp
import torch.nn as nn


class UNetPlusPlusModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=4,
            classes=1,
            activation=None
        )

    def forward(self, x):

        return self.model(x)