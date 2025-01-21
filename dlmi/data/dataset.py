import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (str): Directory containing the input images.
            mask_dir (str): Directory containing the segmentation masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # ensure both folders have matching file names
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        assert len(self.image_files) == len(
            self.mask_files
        ), "Mismatch between image and mask count"

        # for img_file, mask_file in zip(self.image_files, self.mask_files):
        #     assert img_file == mask_file, f"File mismatch: {img_file} and {mask_file}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # assumption: grayscale mask

        # normalize to [0, 1]
        image = (
            torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        mask = torch.tensor(np.array(mask) / 255, dtype=torch.long)  # binary mask

        return image, mask
