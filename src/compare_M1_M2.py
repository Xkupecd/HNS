"""Compares M1 (ResNet50) and M2 (ResNet50 + CBAM) across all three
configurations (baseline, no augmentation, gradual unfreezing).

Per-run test metrics are recomputed from each run's confusion_matrix.csv,
so the statistics (mean/std, Welch's t-test over the 3 runs) use exactly the
saved test predictions. Outputs go to Results/M1_vs_M2/:
  - comparison_table.csv / .md    metrics, t-test p-values, params, duration
  - compare_accuracy.png          test accuracy per configuration (mean +/- std)
  - compare_metrics_baseline.png  acc/precision/recall/f1 of the two baselines
  - compare_curves_baseline.png   best-run train/val curves of the baselines
  - compare_class_recall.png      per-class recall (3 runs pooled) of the baselines

Usage:
    python -m src.compare_M1_M2
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.models import M1_CNN_ResNet50, M2_CNN_ResNet50_CBAM

CONFIGS = [
    ("baseline", "", "základná (s augmentáciou)"),
    ("no_augmentation", "_no_augmentation", "bez augmentácie"),
    ("gradual_unfreeze", "_gradual_unfreeze", "postupné rozmrazovanie"),
]
MODELS = [
    ("M1", config.M1_RESULTS_DIR, "M1_cnn", M1_CNN_ResNet50),
    ("M2", config.M2_RESULTS_DIR, "M2_cnn_cbam", M2_CNN_ResNet50_CBAM),
]
METRICS = ["acc", "precision", "recall", "f1"]
OUT_DIR = config.COMPARISON_DIR


def run_dirs(model_dir: Path) -> list[Path]:
    return sorted(p for p in model_dir.glob("run*") if (p / "confusion_matrix.csv").exists())


def metrics_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"acc": tp.sum() / cm.sum(), "precision": precision.mean(),
            "recall": recall.mean(), "f1": f1.mean()}


def load_runs(model_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-run metrics and the confusion matrix summed over all runs."""
    rows, cm_sum = [], None
    for rd in run_dirs(model_dir):
        cm = pd.read_csv(rd / "confusion_matrix.csv", index_col=0).to_numpy()
        rows.append(metrics_from_cm(cm))
        cm_sum = cm if cm_sum is None else cm_sum + cm
    return pd.DataFrame(rows), cm_sum


def count_params(module) -> tuple[int, int]:
    model = module.build_model(num_classes=len(config.CLASSES), freeze_backbone=True, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for cfg, _, cfg_label in CONFIGS:
        for model, _, _, module in MODELS:
            key = (model, cfg)
            if key not in results:
                continue
            runs = results[key]["runs"]
            total, trainable = count_params(module)
            row = {"konfigurácia": cfg_label, "model": model}
            for m in METRICS:
                row[f"{m}_mean"] = runs[m].mean()
                row[f"{m}_std"] = runs[m].std(ddof=0)
            row["params_total"] = total
            row["params_trainable_start"] = trainable
            row["total_duration"] = results[key]["duration"]
            rows.append(row)
        if ("M1", cfg) in results and ("M2", cfg) in results:
            a = results[("M1", cfg)]["runs"]["acc"]
            b = results[("M2", cfg)]["runs"]["acc"]
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows[-1]["acc_ttest_p_vs_M1"] = p
    return pd.DataFrame(rows)


def plot_accuracy(results: dict) -> None:
    cfgs = [c for c in CONFIGS if all((m, c[0]) in results for m, *_ in MODELS)]
    x = np.arange(len(cfgs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, *_rest) in enumerate(MODELS):
        means = [results[(model, c[0])]["runs"]["acc"].mean() for c in cfgs]
        stds = [results[(model, c[0])]["runs"]["acc"].std(ddof=0) for c in cfgs]
        bars = ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
                      label="M1 (ResNet50)" if model == "M1" else "M2 (ResNet50 + CBAM)")
        ax.bar_label(bars, labels=[f"{v * 100:.2f}" for v in means], padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in cfgs])
    ax.set_ylabel("Testovacia accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("M1 vs M2 - testovacia accuracy (priemer 3 behov +/- std)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_accuracy.png", dpi=150)
    plt.close(fig)


def plot_baseline_metrics(results: dict) -> None:
    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model, *_rest) in enumerate(MODELS):
        runs = results[(model, "baseline")]["runs"]
        ax.bar(x + (i - 0.5) * width, [runs[m].mean() for m in METRICS], width,
               yerr=[runs[m].std(ddof=0) for m in METRICS], capsize=4,
               label="M1 (ResNet50)" if model == "M1" else "M2 (ResNet50 + CBAM)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylabel("Testovacia hodnota metriky")
    ax.set_ylim(0, 1)
    ax.set_title("M1 vs M2 (základná konfigurácia) - test metriky", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_metrics_baseline.png", dpi=150)
    plt.close(fig)


def plot_baseline_curves(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, *_rest in MODELS:
        hist = results[(model, "baseline")]["best_history"]
        for split, style in [("train", "--"), ("val", "-")]:
            axes[0].plot(hist["epoch"], hist[f"{split}_loss"], style, linewidth=2, label=f"{model} {split}")
            axes[1].plot(hist["epoch"], hist[f"{split}_acc"], style, linewidth=2, label=f"{model} {split}")
    for ax, title, ylabel in [(axes[0], "Loss", "Loss"), (axes[1], "Accuracy", "Accuracy")]:
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("M1 vs M2 (základná konfigurácia) - priebeh trénovania najlepšieho behu", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "compare_curves_baseline.png", dpi=150)
    plt.close(fig)


def plot_class_recall(results: dict) -> None:
    classes = config.CLASSES
    x = np.arange(len(classes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (model, *_rest) in enumerate(MODELS):
        cm = results[(model, "baseline")]["cm_sum"]
        recall = np.diag(cm) / cm.sum(axis=1)
        ax.bar(x + (i - 0.5) * width, recall, width,
               label="M1 (ResNet50)" if model == "M1" else "M2 (ResNet50 + CBAM)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_ylabel("Recall (3 behy spolu)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("M1 vs M2 (základná konfigurácia) - recall po triedach, testovacie dáta")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_class_recall.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = {}
    for model, group_dir, base_name, _module in MODELS:
        for cfg, suffix, _label in CONFIGS:
            model_dir = group_dir / f"{base_name}{suffix}"
            if not (model_dir / "summary.json").exists():
                print(f"Skipping {model_dir.name}: not trained yet.")
                continue
            summary = json.loads((model_dir / "summary.json").read_text())
            runs, cm_sum = load_runs(model_dir)
            best_idx = int(np.argmax(summary["per_run_acc"])) + 1
            results[(model, cfg)] = {
                "runs": runs,
                "cm_sum": cm_sum,
                "duration": summary.get("total_duration", "n/a"),
                "best_history": pd.read_csv(model_dir / f"run{best_idx}" / "history.csv"),
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table(results)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    (OUT_DIR / "comparison_table.md").write_text(table.to_markdown(index=False, floatfmt=".4f"))
    print(table.to_string(index=False))

    plot_accuracy(results)
    if ("M1", "baseline") in results and ("M2", "baseline") in results:
        plot_baseline_metrics(results)
        plot_baseline_curves(results)
        plot_class_recall(results)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
