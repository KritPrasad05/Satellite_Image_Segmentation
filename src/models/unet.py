import segmentation_models_pytorch as smp

import torch
import torch.nn as nn

# BUILD UNET MODEL
def build_unet_model(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,
    classes=1
):

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None
    )
    return model