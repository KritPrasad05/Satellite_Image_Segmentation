import os

import numpy as np
import tifffile as tiff

import torch
from torch.utils.data import WeightedRandomSampler

# CREATE SAMPLE WEIGHTS
def create_sample_weights(
    target_dir,
    file_names
):

    sample_weights = []

    for file_name in file_names:

        mask_path = os.path.join(
            target_dir,
            file_name
        )

        mask = tiff.imread(mask_path)
        # Binary Remapping
        binary_mask = np.zeros_like(mask)

        binary_mask[
            (mask == 2) | (mask == 3)
        ] = 1

        # Count Changed Pixels
        changed_pixels = np.sum(binary_mask)

        # Weight Strategy
        if changed_pixels == 0:

            # Empty mask
            weight = 1.0

        else:

            # More damage -> higher weight
            damage_ratio = (
                changed_pixels
                /
                binary_mask.size
            )

            weight = 5.0 + (damage_ratio * 20.0)

        sample_weights.append(weight)

    return sample_weights

# CREATE WEIGHTED SAMPLER
def create_weighted_sampler(
    target_dir,
    file_names
):

    sample_weights = create_sample_weights(
        target_dir,
        file_names
    )

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler