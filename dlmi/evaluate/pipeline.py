from argparse import ArgumentParser
import pandas as pd

from dlmi.utils.counting import count_cells
from dlmi.utils.utils import load_experiment_config
from tqdm import tqdm


def single_image(config):
    set_type = "train"
    image = "TCGA-A7-A13F-01Z-00-DX1.png"
    mask_dir = config["dir"] / set_type / "processed" / "binary_masks"

    mask_path = mask_dir / image
    count = count_cells(mask_path)
    print(f"Number of cells in {image}: {count}")


def main(config):
    set_type = "test"
    pred_dir = config["dir"] / set_type / "predictions"
    df = pd.read_csv(config["dir"] / set_type / "actual_cell_counts.csv")

    df['predicted_cell_count'] = [count_cells(f) for f in tqdm(pred_dir.glob("*.png"))]

    df.to_csv(config["dir"] / set_type / "predicted_cell_counts.csv", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    main(exp)
