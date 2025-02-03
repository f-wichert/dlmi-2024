from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from dlmi.utils.utils import load_general_config


def calculate_qq_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = actual / predicted
    b = predicted / actual
    return np.max([a, b])


def analyze_experiments(base_path: Path) -> Dict:
    """Analyze all experiments and compute errors"""
    results = {}

    for exp_dir in base_path.rglob("predicted_cell_counts.csv", ):
        df = pd.read_csv(exp_dir)

        qq_error = calculate_qq_error(df["predicted_cell_count"].values, df["num_cells"].values)

        results[exp_dir.parent.parent.name] = {
            'qq_error': qq_error,
        }

    return results


def plot_results(results: Dict, output_dir: Path):
    sorted_exps = sorted(results.items(), key=lambda x: x[1]['qq_error'])

    plt.figure(figsize=(10, 5))
    exps = [x[0] for x in sorted_exps]
    errors = [x[1]['qq_error'] for x in sorted_exps]
    plt.bar(exps, errors)
    plt.xticks(rotation=45)
    plt.title('QQ Error by Experiment')
    plt.tight_layout()
    plt.savefig(output_dir / 'qq_errors.png')
    plt.close()


def main(config):
    """Main execution function"""
    base_path = config["experiments_dir"]
    output_dir = base_path / 'analysis'
    output_dir.mkdir(exist_ok=True)

    results = analyze_experiments(base_path)

    print("\nExperiment Results Summary:")
    print("-" * 50)
    for exp_name, metrics in sorted(results.items(), key=lambda x: x[1]['qq_error']):
        print(f"{exp_name}:")
        print(f"  QQ Error: {metrics['qq_error']:.4f}")

    plot_results(results, output_dir)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    exp = load_general_config()
    main(exp)
