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
    qq_error = max(actual / (predicted + epsilon), predicted / (actual + epsilon))
    error_ratio = (predicted + epsilon) / (actual + epsilon)

    return {
        'absolute_error': abs_error,
        'relative_error': rel_error,
        'qq_error': qq_error,
        'error_ratio': error_ratio
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

    val_dice_scores = {
        'augment_blur': 0.6297659277915955,
        'augment_flip': 0.5583277344703674,
        'augment_rotation': 0.5149551033973694,
        'no_augment_size_256_depth_6': 0.428,
        'augment_all': 0.4326605796813965,
        'no_augment_size_64': 0.7126829028129578,
        'no_augment_size_256': 0.52659010887146,
        'no_augment_depth_2': 0.5929386615753174,
        'no_augment_depth_6': 0.6439461708068848,
        'no_augment_default': 0.6410626173019409
    }

    plt.figure(figsize=(21, 9))

    ax1 = plt.gca()
    sns.swarmplot(data=image_results, x='experiment', y='relative_error',
                  hue='image_id', legend=False, ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Relative Error')
    plt.ylim(-0.1, 2.6)

    ax2 = ax1.twinx()
    dice_data = pd.DataFrame({
        'experiment': val_dice_scores.keys(),
        'dice_score': val_dice_scores.values()
    })
    ax2.plot(range(len(dice_data)), dice_data['dice_score'],
             color='blue', marker='s', linestyle=None, linewidth=0, label='Dice Score', )
    ax2.grid(False)
    ax2.set_ylabel('Dice Score', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.xlabel('')
    plt.savefig(output_dir / 'error_distribution_with_dice.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.swarmplot(data=image_results, x='experiment', y='relative_error', hue='image_id', legend=False)
    plt.xticks(rotation=45)
    plt.xlabel('')
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

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=image_results, x='experiment', y='error_ratio')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.title('Distribution of Error Ratios\n(>1: Over-prediction, <1: Under-prediction)')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_ratio_distribution.png')
    plt.close()

    # Add error ratio vs cell count plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=image_results, x='actual_count', y='error_ratio', hue='experiment', alpha=0.6)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.title('Error Ratio vs Actual Cell Count')
    plt.xlabel('Actual Cell Count')
    plt.ylabel('Error Ratio (Predicted/Actual)')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_ratio_vs_count.png')
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
