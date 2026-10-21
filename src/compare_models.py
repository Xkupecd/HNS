"""Compare M1-M3 on the test set: summary table (mean ± std, min, max over
5 runs), per-class sensitivity/specificity, bar chart and per-run box plot.

Writes into Results/M1_to_M3/:
    comparison.csv, comparison.md, per_class.csv,
    metrics_bar.png, accuracy_per_run.png

Usage:
    python -m src.compare_models
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

MODELS = [
    ("M1 CNN+SVM", config.M1_RESULTS_DIR),
    ("M2 CNN+LSTM", config.M2_RESULTS_DIR),
    ("M3 CNN+Transformer", config.M3_RESULTS_DIR),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
METRICS = [("acc", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]


def load_summaries() -> list[tuple[str, dict]]:
    return [(name, json.loads((d / "summary.json").read_text())) for name, d in MODELS if (d / "summary.json").exists()]


def comparison_table(summaries) -> pd.DataFrame:
    rows = []
    for name, s in summaries:
        row = {"model": name}
        for key, label in METRICS:
            row[label] = f"{100 * s[f'{key}_mean']:.2f} ± {100 * s[f'{key}_std']:.2f}"
        row["Acc min"] = f"{100 * s['acc_min']:.2f}"
        row["Acc max"] = f"{100 * s['acc_max']:.2f}"
        row["best run"] = s["best_run"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(summaries) -> pd.DataFrame:
    rows = []
    for name, s in summaries:
        for cls, v in s["per_class_mean"].items():
            rows.append({"model": name, "class": cls,
                         "sensitivity": round(100 * v["sensitivity"], 2),
                         "specificity": round(100 * v["specificity"], 2)})
    return pd.DataFrame(rows)


def plot_metrics_bar(summaries, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(summaries)
    x = np.arange(len(METRICS))
    for i, (name, s) in enumerate(summaries):
        means = [100 * s[f"{k}_mean"] for k, _ in METRICS]
        stds = [100 * s[f"{k}_std"] for k, _ in METRICS]
        bars = ax.bar(x + (i - (len(summaries) - 1) / 2) * width, means, width, yerr=stds, capsize=4,
                      label=name, color=COLORS[i % len(COLORS)])
        ax.bar_label(bars, fmt="%.1f", padding=6, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylabel("Test score [%] (mean ± std, 5 runs)")
    lo = min(100 * s[f"{k}_mean"] - 100 * s[f"{k}_std"] for _, s in summaries for k, _ in METRICS)
    ax.set_ylim(max(0, lo - 10), 100)
    ax.legend(loc="upper center", ncol=len(summaries))
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("HMDB51 (clap, drink, eat, wave) - test set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_per_run(summaries, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [[100 * a for a in s["per_run_acc"]] for _, s in summaries]
    ax.boxplot(data, widths=0.5)
    for i, values in enumerate(data, start=1):
        ax.scatter(np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values)), values,
                   color=COLORS[(i - 1) % len(COLORS)], zorder=3)
    ax.set_xticks(range(1, len(summaries) + 1))
    ax.set_xticklabels([name for name, _ in summaries])
    ax.set_ylabel("Test accuracy [%]")
    ax.set_title("Test accuracy of the 5 runs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    summaries = load_summaries()
    out_dir = config.COMPARISON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    table = comparison_table(summaries)
    table.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(table.to_markdown(index=False) + "\n")
    per_class = per_class_table(summaries)
    per_class.to_csv(out_dir / "per_class.csv", index=False)

    plot_metrics_bar(summaries, out_dir / "metrics_bar.png")
    plot_accuracy_per_run(summaries, out_dir / "accuracy_per_run.png")
    print(table.to_markdown(index=False))
    print()
    print(per_class.to_markdown(index=False))


if __name__ == "__main__":
    main()
