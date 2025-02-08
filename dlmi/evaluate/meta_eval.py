from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dlmi.utils.utils import load_general_config


def calculate_qq_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = actual / predicted
    b = predicted / actual
    return np.max([a, b])


def calculate_image_metrics(actual: float, predicted: float) -> Dict:
    epsilon = 1e-6
    abs_error = abs(actual - predicted)
    rel_error = abs_error / actual
    qq_error = max(actual/(predicted + epsilon), predicted/(actual) + epsilon)

    return {
        'absolute_error': abs_error,
        'relative_error': rel_error,
        'qq_error': qq_error
    }


def analyze_experiments(base_path: Path) -> Tuple[Dict, pd.DataFrame]:
    results = {}
    all_image_results = []

    for exp_dir in base_path.rglob("predicted_cell_counts.csv"):
        df = pd.read_csv(exp_dir)
        exp_name = exp_dir.parent.parent.name

        mse = mean_squared_error(df["num_cells"], df["predicted_cell_count"])
        mae = mean_absolute_error(df["num_cells"], df["predicted_cell_count"])
        r2 = r2_score(df["num_cells"], df["predicted_cell_count"])
        qq_error = calculate_qq_error(df["num_cells"].values, df["predicted_cell_count"].values)

        results[exp_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'qq_error': qq_error
        }

        for _, row in df.iterrows():
            metrics = calculate_image_metrics(row["num_cells"], row["predicted_cell_count"])
            all_image_results.append({
                'experiment': exp_name,
                'image_id': row['id'],
                'actual_count': row['num_cells'],
                'predicted_count': row['predicted_cell_count'],
                'mean_cell_area': row['mean_cell_area'],
                **metrics
            })

    return results, pd.DataFrame(all_image_results)


def plot_results(results: Dict, image_results: pd.DataFrame, output_dir: Path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sorted_data = pd.DataFrame([(k, v['qq_error']) for k, v in results.items()],
                               columns=['experiment', 'qq_error'])
    sns.barplot(data=sorted_data, x='experiment', y='qq_error')
    plt.xticks(rotation=45)
    plt.title('QQ Error by Experiment')
    plt.tight_layout()
    plt.savefig(output_dir / 'qq_errors.png')
    plt.close()

    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=image_results, x='actual_count', y='predicted_count',
                      hue='experiment', kind='scatter', height=10)
    g.ax_joint.plot([0, max(image_results['actual_count'])],
                    [0, max(image_results['actual_count'])], 'k--', alpha=0.5)
    g.fig.suptitle('Actual vs Predicted Cell Counts', y=1.02)
    plt.savefig(output_dir / 'correlation_plot.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=image_results, x='experiment', y='relative_error')
    plt.xticks(rotation=45)
    plt.title('Distribution of Relative Errors')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lmplot(data=image_results, x='mean_cell_area', y='relative_error',
               hue='experiment', height=6, aspect=1.5)
    plt.title('Error vs Mean Cell Area')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_area.png')
    plt.close()


def main(config):
    base_path = Path(config["experiments_dir"])
    output_dir = base_path / 'analysis'
    output_dir.mkdir(exist_ok=True)

    results, image_results = analyze_experiments(base_path)

    print("\nExperiment Results Summary:")
    print("-" * 50)
    for exp_name, metrics in sorted(results.items(), key=lambda x: x[1]['qq_error']):
        print(f"{exp_name}:")
        print(f"  MSE: {metrics['mse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  QQ Error: {metrics['qq_error']:.4f}")

    print("\nImage-level Analysis:")
    print("-" * 50)
    print("Top 5 Most Accurate Predictions:")
    print(image_results.nsmallest(5, 'relative_error')[
              ['experiment', 'image_id', 'actual_count', 'predicted_count', 'relative_error']])

    print("\nTop 5 Least Accurate Predictions:")
    print(image_results.nlargest(5, 'relative_error')[
              ['experiment', 'image_id', 'actual_count', 'predicted_count', 'relative_error']])

    plot_results(results, image_results, output_dir)
    image_results.to_csv(output_dir / 'image_level_analysis.csv', index=False)


if __name__ == "__main__":
    exp = load_general_config()
    main(exp)
