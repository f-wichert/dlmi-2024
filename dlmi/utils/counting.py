from pathlib import Path
import numpy as np
from PIL import Image
from skimage.measure import label
import pandas as pd
from dlmi.utils.utils import load_config
from tqdm import tqdm


def count_cells(mask_path):
    """Count cells in binary mask using connected components"""
    mask = np.array(Image.open(mask_path))
    if mask.dtype == np.float32:
        mask = (mask > 0).astype(np.uint8)

    labeled = label(mask)
    return labeled.max()


def main():
    config = load_config()
    cell_counts = {}

    for set_type in config["data"]["prepare_for"]:
        ic = config["data"][set_type]
        base_dir = Path(config["data"]["data_dir"]) / ic["dir"]
        binary_mask_dir = base_dir / ic["binary_masks"]

        mask_files = list(binary_mask_dir.glob(f"*.{config['data']['out_format']}"))
        for mask_path in tqdm(mask_files, desc=f"Counting cells in {set_type}"):
            if "-no-overlap" not in str(mask_path):
                num_cells = count_cells(mask_path)
                cell_counts[mask_path.stem] = num_cells

        # Save results
        df = pd.DataFrame.from_dict(cell_counts, orient='index', columns=['cell_count'])
        df.to_csv(base_dir / ic['cell_counts'] / "automated_cell_counts.csv")


if __name__ == "__main__":
    main()
