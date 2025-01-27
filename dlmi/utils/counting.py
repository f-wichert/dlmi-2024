import numpy as np
from PIL import Image
from skimage.measure import label


def count_cells(mask_path):
    """Count cells in binary mask using connected components"""
    mask = np.array(Image.open(mask_path))
    if mask.dtype == np.float32:
        mask = (mask > 0).astype(np.uint8)

    labeled = label(mask)
    return labeled.max()


