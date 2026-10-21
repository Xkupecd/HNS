"""Training-curve and confusion-matrix plots, per the assignment.

Figure sizes / font sizes scale with the number of classes so labels stay
readable even for the larger datasets (e.g. 16+ classes) instead of
overlapping like in earlier runs.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})


def plot_training_curves(history_csv: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="val", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="train", linewidth=2)
    axes[1].plot(df["epoch"], df["val_acc"], label="val", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    """Confusion matrix heatmap with readable labels: figure/font size scale
    with the number of classes, labels are rotated and never truncated."""
    n = len(class_names)
    cell = max(0.55, min(1.0, 12 / n))
    fig_size = max(6.0, n * cell)
    label_fontsize = 11 if n <= 10 else (9 if n <= 20 else 7)
    annot_fontsize = 10 if n <= 10 else (8 if n <= 20 else 6)

    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(class_names, fontsize=label_fontsize)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title, fontsize=16)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=annot_fontsize,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
