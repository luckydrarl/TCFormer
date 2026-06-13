"""
Generate result plots for EEG MI model comparison.
Phase 5: Visualization scripts.

Usage:
    python scripts/plot_results.py --results_dir results/tables/
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_loso_accuracy_bar(results_df, output_path):
    """Bar chart of LOSO accuracy per model with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = results_df['model_name'].unique()
    means = []
    stds = []
    for m in models:
        model_data = results_df[results_df['model_name'] == m]
        means.append(model_data['mean_acc'].values[0] * 100)
        stds.append(model_data['std_acc'].values[0] * 100)

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.viridis(np.linspace(0, 1, len(models))))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LOSO Accuracy Comparison (BCI IV 2a)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_loso_kappa_bar(results_df, output_path):
    """Bar chart of LOSO kappa per model with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = results_df['model_name'].unique()
    means = []
    stds = []
    for m in models:
        model_data = results_df[results_df['model_name'] == m]
        means.append(model_data['mean_kappa'].values[0])
        stds.append(model_data['std_kappa'].values[0])

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.plasma(np.linspace(0, 1, len(models))))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title('LOSO Kappa Comparison (BCI IV 2a)')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_latency(results_df, output_path):
    """Scatter plot: Accuracy vs Latency."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in results_df.iterrows():
        ax.scatter(row['mean_latency_ms'], row['mean_acc'] * 100, s=150,
                  label=row['model_name'])
        ax.annotate(row['model_name'],
                   (row['mean_latency_ms'], row['mean_acc'] * 100),
                   textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel('Latency (ms/trial)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Latency (BCI IV 2a LOSO)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_params(results_df, output_path):
    """Scatter plot: Accuracy vs Number of Parameters."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in results_df.iterrows():
        ax.scatter(row['params'], row['mean_acc'] * 100, s=150,
                  label=row['model_name'])
        ax.annotate(row['model_name'],
                   (row['params'], row['mean_acc'] * 100),
                   textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Parameters (BCI IV 2a LOSO)')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_subject_heatmap(per_subject_df, output_path):
    """Heatmap of per-subject accuracy for each model."""
    # Pivot: rows=models, cols=subjects, values=accuracy
    pivot = per_subject_df.pivot(index='model_name', columns='test_subject', values='accuracy')

    fig, ax = plt.subplots(figsize=(12, len(pivot) * 0.8 + 2))
    im = ax.imshow(pivot.values * 100, aspect='auto', cmap='YlOrRd', vmin=30, vmax=90)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'S{s}' for s in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)')

    ax.set_title('Per-Subject Accuracy Heatmap (BCI IV 2a)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="D:/EEGnet/results/tables")
    parser.add_argument("--plots_dir", type=str, default="D:/EEGnet/results/plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "loso_summary.csv"
    per_subject_path = results_dir / "loso_results.csv"

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if len(summary_df) > 0:
            plot_loso_accuracy_bar(summary_df, plots_dir / "loso_accuracy_bar.png")
            plot_loso_kappa_bar(summary_df, plots_dir / "loso_kappa_bar.png")
            plot_accuracy_vs_latency(summary_df, plots_dir / "accuracy_vs_latency.png")
            plot_accuracy_vs_params(summary_df, plots_dir / "accuracy_vs_params.png")
    else:
        print(f"Summary file not found: {summary_path}")

    if per_subject_path.exists():
        per_subject_df = pd.read_csv(per_subject_path)
        if len(per_subject_df) > 0:
            plot_per_subject_heatmap(per_subject_df, plots_dir / "per_subject_heatmap.png")
    else:
        print(f"Per-subject file not found: {per_subject_path}")


if __name__ == "__main__":
    main()
