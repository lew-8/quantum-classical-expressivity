"""Results table and figures.

The table sums the results from the training and testing, as in the conclusions of
the report: the rows are the datasets, the columns the architectures, and every cell
is train/test accuracy x 1000. With several seeds the average is shown.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["SNN", "DNN", "SE1", "SE2", "SE3", "DE1", "DE2", "DE3"]
DATASET_ORDER = ["linear", "difference", "product"]


def accuracy(y_true, y_pred):
    """Fraction of correctly classified points."""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("labels must be non-empty and of the same length")
    return float(np.mean(y_true == y_pred))


def _ordered(names, order):
    names = list(dict.fromkeys(names))
    return [n for n in order if n in names] + sorted(n for n in names if n not in order)


def aggregate_benchmarks(df):
    """Mean and standard deviation of the accuracies over the seeds."""
    out = df.groupby(["dataset", "model"], sort=False).agg(
        n_params=("n_params", "first"),
        runs=("seed", "nunique"),
        train_mean=("train_accuracy", "mean"),
        train_std=("train_accuracy", "std"),
        test_mean=("test_accuracy", "mean"),
        test_std=("test_accuracy", "std"),
    ).reset_index()
    return out.fillna({"train_std": 0.0, "test_std": 0.0})


def summary_table(df):
    """Table like the one in the report, cells 'train/test' x 1000."""
    agg = aggregate_benchmarks(df)
    agg["cell"] = [f"{round(1000 * a)}/{round(1000 * b)}"
                   for a, b in zip(agg["train_mean"], agg["test_mean"])]
    table = agg.pivot(index="dataset", columns="model", values="cell")
    return table.loc[_ordered(table.index, DATASET_ORDER),
                     _ordered(table.columns, MODEL_ORDER)].fillna("")


# ---------------------------------------------------------------- figures

def plot_data(x, y, fig=None, ax=None, title=None):
    """Plot data with red/blue values for a binary classification.

    Args:
        x (array): array of data points
        y (array[int]): array of data points labels
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    x, y = np.asarray(x), np.ravel(y)
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)
    return ax.figure


def plot_datasets(datasets, titles=None):
    """The datasets next to each other (Fig. 2 of the report)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.5), squeeze=False)
    for i, (data, ax) in enumerate(zip(datasets, axes[0])):
        plot_data(data.X, data.y, ax=ax,
                  title=titles[i] if titles else f"Artificial Dataset: {data.name}")
    fig.tight_layout()
    return fig


def plot_prediction_comparison(x, y_initial, y_trained, y_true, suptitle=None):
    """Predictions with random weights, after training and the true test data."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_data(x, y_initial, ax=axes[0], title="Predictions with random weights")
    plot_data(x, y_trained, ax=axes[1], title="Predictions after training")
    plot_data(x, y_true, ax=axes[2], title="True test data")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_loss_history(results, ax=None):
    """Loss of one or more trained models (PQC: per COBYLA step, NN: per epoch)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for res in results:
        ax.plot(res.loss_history, label=f"{res.model} ({res.n_params} params)")
    ax.set_xlabel("iteration / epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax.figure


def plot_accuracy_comparison(df, metric="test"):
    """Bar chart of the average accuracy per dataset and model (error bar = std)."""
    agg = aggregate_benchmarks(df)
    datasets = _ordered(agg["dataset"], DATASET_ORDER)
    models = _ordered(agg["model"], MODEL_ORDER)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(datasets) * len(models)), 4))
    for i, model in enumerate(models):
        sub = agg[agg["model"] == model].set_index("dataset").reindex(datasets)
        ax.bar(np.arange(len(datasets)) + (i - (len(models) - 1) / 2) * width,
               sub[f"{metric}_mean"], width, yerr=sub[f"{metric}_std"], capsize=2,
               label=model, color=plt.get_cmap("tab10")(i % 10))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(datasets)), datasets)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{metric} accuracy")
    ax.legend(ncol=len(models), fontsize="small", loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save_benchmark_figures(df, directory, n_points=1000, margin=0.3, seed=42):
    """Save datasets.png and accuracy_comparison.png in the given folder."""
    from .datasets import make_dataset

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    datasets = [make_dataset(n, n_points=n_points, margin=margin, seed=seed)
                for n in _ordered(df["dataset"], DATASET_ORDER)]
    figures = {"datasets.png": plot_datasets(datasets),
               "accuracy_comparison.png": plot_accuracy_comparison(df)}
    paths = []
    for filename, fig in figures.items():
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)
        paths.append(directory / filename)
    return paths
