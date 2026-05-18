import os

import numpy as np
import tifffile as tiff

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=512,
        augment=False
    ):

        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Folder Paths
        self.pre_dir = os.path.join(
            root_dir,
            split,
            "pre-event"
        )

        self.post_dir = os.path.join(
            root_dir,
            split,
            "post-event"
        )

        self.target_dir = os.path.join(
            root_dir,
            split,
            "target"
        )
        # File Names
        self.file_names = sorted(
            os.listdir(self.pre_dir)
        )
        # Transforms
        self.transforms = self.get_transforms()

    # TRANSFORMS
    def get_transforms(self):

        if self.augment and self.split == "train":

            transforms = A.Compose([

                A.Resize(
                    self.image_size,
                    self.image_size
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406, 0.5),
                    std=(0.229, 0.224, 0.225, 0.5)
                ),

                ToTensorV2()
            ])

        else:

            transforms = A.Compose([
                A.Resize(
                    self.image_size,
                    self.image_size
                ),

                A.Normalize(
                    mean=(0.485, 0.456, 0.406, 0.5),
                    std=(0.229, 0.224, 0.225, 0.5)
                ),

                ToTensorV2()
            ])

        return transforms

    # REMAP MASK
    def remap_mask(self, mask):

        binary_mask = np.zeros_like(mask)
        binary_mask[
            (mask == 2) | (mask == 3)
        ] = 1
        
        return binary_mask

    # DATASET LENGTH
    def __len__(self):

        return len(self.file_names)

    # GET ITEM
    def __getitem__(self, idx):
        # File Name
        file_name = self.file_names[idx]
        # Paths
        pre_path = os.path.join(
            self.pre_dir,
            file_name
        )

        post_path = os.path.join(
            self.post_dir,
            file_name
        )

        target_path = os.path.join(
            self.target_dir,
            file_name
        )

        # Load Images
        pre_image = tiff.imread(pre_path).astype(np.uint8)
        post_image = tiff.imread(post_path).astype(np.uint8)
        mask = tiff.imread(target_path).astype(np.uint8)

        # Convert SAR to Single Channel
        if len(post_image.shape) == 2:
            post_image = np.expand_dims(
                post_image,
                axis=-1
            )
        # Remap Mask
        mask = self.remap_mask(mask)
        # Combine EO + SAR
        combined_image = np.concatenate(
            [pre_image, post_image],
            axis=-1
        )
        # Apply Transforms
        transformed = self.transforms(
            image=combined_image,
            mask=mask
        )

        image = transformed["image"]
        mask = transformed["mask"]
        
        # Convert Mask Shape
        mask = mask.unsqueeze(0).float()

        return {
            "image": image.float(),
            "mask": mask,
            "file_name": file_name
        }