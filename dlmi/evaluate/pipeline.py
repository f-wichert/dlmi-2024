from argparse import ArgumentParser
import pandas as pd

from dlmi.utils.counting import count_cells
from dlmi.utils.utils import load_experiment_config
from tqdm import tqdm


def main(config):
    cell_counts = {}
    pred_dir = config["data_dir"] / "test" / "predictions"
    mask_dir = config["data_dir"] / "test" / "processed" / "binary_masks"

    pred_files = list(pred_dir.glob("*.png"))
    mask_files = list(mask_dir.glob("*.png"))

    for pred_f, mask_f in tqdm(zip(pred_files, mask_files), desc="Counting cells", total=len(pred_files)):
        pred_num_cells = count_cells(pred_f)
        actual_num_cells = count_cells(mask_f)
        cell_counts[mask_f.stem] = {
            'predicted_count': pred_num_cells,
            'true_count': actual_num_cells
        }

    # Save results
    df = pd.DataFrame.from_dict(cell_counts, orient='index')
    df.to_csv(config["dir"] / "cell_counts.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    exp = load_experiment_config(args.experiment)
    main(exp)
