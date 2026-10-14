"""Compares each model's baseline against its two ablations: augmentation
on/off and gradual unfreezing vs a frozen backbone (for M1-M3; pairs whose
runs are missing, e.g. M3 without augmentation, are skipped).

Produces bar charts (test acc/precision/recall/f1, mean +/- std over the
3 runs), overlaid best-run validation curves and a CSV table for each pair,
saved to Results/<MODEL_GROUP>/comparisons/.

Usage:
    python -m src.compare_ablations
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

GROUPS = [
    (config.M1_RESULTS_DIR, "M1_cnn"),
    (config.M2_RESULTS_DIR, "M2_cnn_cbam"),
    (config.M3_RESULTS_DIR, "M3_convnext"),
    (config.M4_RESULTS_DIR, "M4_swin"),
    (config.M5_RESULTS_DIR, "M5_coatnet"),
]

ABLATIONS = [
    ("augmentation", "_no_augmentation", "s augmentáciou", "bez augmentácie"),
    ("gradual_unfreeze", "_gradual_unfreeze", "zamrznutý backbone", "postupné rozmrazovanie"),
]

METRICS = ["acc", "precision", "recall", "f1"]


def load_summary(model_dir: Path) -> dict:
    return json.loads((model_dir / "summary.json").read_text())


def best_run_history(model_dir: Path) -> pd.DataFrame:
    summary = load_summary(model_dir)
    best_idx = int(np.argmax(summary["per_run_acc"])) + 1
    return pd.read_csv(model_dir / f"run{best_idx}" / "history.csv")


def plot_metric_bars(out_path: Path, dir_a: Path, dir_b: Path, label_a: str, label_b: str) -> None:
    sum_a, sum_b = load_summary(dir_a), load_summary(dir_b)

    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, [sum_a[f"{m}_mean"] for m in METRICS], width,
           yerr=[sum_a[f"{m}_std"] for m in METRICS], capsize=4, label=f"{dir_a.name} ({label_a})")
    ax.bar(x + width / 2, [sum_b[f"{m}_mean"] for m in METRICS], width,
           yerr=[sum_b[f"{m}_std"] for m in METRICS], capsize=4, label=f"{dir_b.name} ({label_b})")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylabel("Testovacia hodnota metriky")
    ax.set_ylim(0, 1)
    ax.set_title(f"{dir_a.name} vs {dir_b.name} - test metriky (priemer 3 behov +/- std)", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_curve_overlay(out_path: Path, dir_a: Path, dir_b: Path, label_a: str, label_b: str) -> None:
    hist_a, hist_b = best_run_history(dir_a), best_run_history(dir_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title, ylabel in [
        (axes[0], "val_loss", "Validačný loss", "Loss"),
        (axes[1], "val_acc", "Validačná accuracy", "Accuracy"),
    ]:
        ax.plot(hist_a["epoch"], hist_a[key], label=f"{label_a} (val)", linewidth=2)
        ax.plot(hist_b["epoch"], hist_b[key], label=f"{label_b} (val)", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"{dir_a.name} vs {dir_b.name} - priebeh trénovania (najlepší beh)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_table(out_path: Path, dir_a: Path, dir_b: Path) -> None:
    rows = []
    for model_dir in (dir_a, dir_b):
        s = load_summary(model_dir)
        rows.append({
            "model": model_dir.name,
            **{f"{m}_mean": round(s[f"{m}_mean"], 4) for m in METRICS},
            **{f"{m}_std": round(s[f"{m}_std"], 4) for m in METRICS},
            "total_duration": s.get("total_duration", "n/a"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out_path}")


def main() -> None:
    for group_dir, baseline in GROUPS:
        out_dir = group_dir / "comparisons"
        for name, suffix, label_a, label_b in ABLATIONS:
            dir_a = group_dir / baseline
            dir_b = group_dir / f"{baseline}{suffix}"
            if not (dir_a / "summary.json").exists() or not (dir_b / "summary.json").exists():
                print(f"Skipping {dir_b.name}: not trained yet.")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            write_table(out_dir / f"compare_{name}_table.csv", dir_a, dir_b)
            plot_metric_bars(out_dir / f"compare_{name}_metrics.png", dir_a, dir_b, label_a, label_b)
            plot_curve_overlay(out_dir / f"compare_{name}_curves.png", dir_a, dir_b, label_a, label_b)


if __name__ == "__main__":
    main()
