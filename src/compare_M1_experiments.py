"""Compares the M1 baseline against its two required ablations:
augmentation on/off (M1_cnn vs M1_cnn_no_augmentation) and gradual
unfreezing vs a frozen backbone (M1_cnn vs M1_cnn_gradual_unfreeze).

Produces bar charts (test acc/precision/recall/f1, mean +/- std over the
3 runs) and overlaid best-run training curves for each pair, saved to
Results/comparisons/.

Usage:
    python -m src.compare_M1_experiments
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

OUT_DIR = config.RESULTS_DIR / "comparisons"

PAIRS = [
    ("augmentation", "M1_cnn", "M1_cnn_no_augmentation", "s augmentáciou", "bez augmentácie"),
    ("gradual_unfreeze", "M1_cnn", "M1_cnn_gradual_unfreeze", "zamrznutý backbone", "postupné rozmrazovanie"),
]

METRICS = ["acc", "precision", "recall", "f1"]


def load_summary(model_name: str) -> dict:
    path = config.RESULTS_DIR / model_name / "summary.json"
    return json.loads(path.read_text())


def best_run_history(model_name: str) -> pd.DataFrame:
    summary = load_summary(model_name)
    best_idx = int(np.argmax(summary["per_run_acc"])) + 1
    return pd.read_csv(config.RESULTS_DIR / model_name / f"run{best_idx}" / "history.csv")


def plot_metric_bars(name: str, model_a: str, model_b: str, label_a: str, label_b: str) -> None:
    sum_a, sum_b = load_summary(model_a), load_summary(model_b)

    x = np.arange(len(METRICS))
    width = 0.35
    means_a = [sum_a[f"{m}_mean"] for m in METRICS]
    stds_a = [sum_a[f"{m}_std"] for m in METRICS]
    means_b = [sum_b[f"{m}_mean"] for m in METRICS]
    stds_b = [sum_b[f"{m}_std"] for m in METRICS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, means_a, width, yerr=stds_a, capsize=4, label=f"{model_a} ({label_a})")
    ax.bar(x + width / 2, means_b, width, yerr=stds_b, capsize=4, label=f"{model_b} ({label_b})")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylabel("Testovacia hodnota metriky")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_a} vs {model_b} - test metriky (priemer 3 behov +/- std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = OUT_DIR / f"compare_{name}_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_curve_overlay(name: str, model_a: str, model_b: str, label_a: str, label_b: str) -> None:
    hist_a, hist_b = best_run_history(model_a), best_run_history(model_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist_a["epoch"], hist_a["val_loss"], label=f"{label_a} (val)", linewidth=2)
    axes[0].plot(hist_b["epoch"], hist_b["val_loss"], label=f"{label_b} (val)", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Validačný loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist_a["epoch"], hist_a["val_acc"], label=f"{label_a} (val)", linewidth=2)
    axes[1].plot(hist_b["epoch"], hist_b["val_acc"], label=f"{label_b} (val)", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Validačná accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"{model_a} vs {model_b} - priebeh trénovania (najlepší beh)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = OUT_DIR / f"compare_{name}_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_table(name: str, model_a: str, model_b: str) -> None:
    rows = []
    for model in (model_a, model_b):
        s = load_summary(model)
        rows.append({
            "model": model,
            **{f"{m}_mean": round(s[f"{m}_mean"], 4) for m in METRICS},
            **{f"{m}_std": round(s[f"{m}_std"], 4) for m in METRICS},
            "total_duration": s.get("total_duration", "n/a"),
        })
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"compare_{name}_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out_path}")


def main() -> None:
    for name, model_a, model_b, label_a, label_b in PAIRS:
        dir_a = config.RESULTS_DIR / model_a
        dir_b = config.RESULTS_DIR / model_b
        if not (dir_a / "summary.json").exists() or not (dir_b / "summary.json").exists():
            print(f"Skipping {name}: {model_a} or {model_b} not trained yet.")
            continue
        write_table(name, model_a, model_b)
        plot_metric_bars(name, model_a, model_b, label_a, label_b)
        plot_curve_overlay(name, model_a, model_b, label_a, label_b)


if __name__ == "__main__":
    main()
