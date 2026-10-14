"""Compares all five models (M1-M5) in the two configurations trained for
every model: frozen backbone (baseline, with augmentation) and gradual
unfreezing.

Per-run test metrics are recomputed from each run's confusion_matrix.csv
(same as compare_M1_M2.py), so the statistics use exactly the saved test
predictions. For each configuration a one-way ANOVA tests whether the five
models differ at all, and pairwise Welch t-tests (3 runs per model) show
which pairs differ. Outputs go to Results/M1_to_M5/:
  - comparison_table.csv / .md       metrics mean/std, params, duration, p-values vs M1
  - stat_tests.md                    ANOVA + pairwise Welch t-test p-value matrices
  - compare_accuracy.png             test accuracy per model and configuration (mean +/- std)
  - compare_metrics.png              acc/precision/recall/f1 per model, both configurations
  - compare_val_curves.png           best-run validation loss/accuracy of all models
  - compare_class_recall.png         per-class recall heatmap (3 runs pooled), both configurations
  - compare_acc_vs_time.png          test accuracy vs. training time of 3 runs

Usage:
    python -m src.compare_models
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.compare_M1_M2 import load_runs
from src.models import (
    M1_CNN_ResNet50, M2_CNN_ResNet50_CBAM, M3_CNN_ConvNeXt_Tiny, M4_ViT_Swin_Tiny, M5_Hybrid_CoAtNet_0,
)

CONFIGS = [
    ("baseline", "", "zamrznutý backbone"),
    ("gradual_unfreeze", "_gradual_unfreeze", "postupné rozmrazovanie"),
]
MODELS = [
    ("M1", "M1 ResNet50", config.M1_RESULTS_DIR, "M1_cnn", M1_CNN_ResNet50),
    ("M2", "M2 ResNet50+CBAM", config.M2_RESULTS_DIR, "M2_cnn_cbam", M2_CNN_ResNet50_CBAM),
    ("M3", "M3 ConvNeXt-T", config.M3_RESULTS_DIR, "M3_convnext", M3_CNN_ConvNeXt_Tiny),
    ("M4", "M4 Swin-T", config.M4_RESULTS_DIR, "M4_swin", M4_ViT_Swin_Tiny),
    ("M5", "M5 CoAtNet-0", config.M5_RESULTS_DIR, "M5_coatnet", M5_Hybrid_CoAtNet_0),
]
METRICS = ["acc", "precision", "recall", "f1"]
OUT_DIR = config.ALL_COMPARISON_DIR


def duration_minutes(text: str) -> float:
    h, m, s = (int(x) for x in text.split(":"))
    return h * 60 + m + s / 60


def count_params(module) -> tuple[int, int]:
    model = module.build_model(num_classes=len(config.CLASSES), freeze_backbone=True, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_results() -> dict:
    results = {}
    for model, _label, group_dir, base_name, _module in MODELS:
        for cfg, suffix, _cfg_label in CONFIGS:
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
    return results


def models_in(results: dict, cfg: str) -> list[tuple]:
    return [m for m in MODELS if (m[0], cfg) in results]


def welch_p(a: pd.Series, b: pd.Series) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def build_table(results: dict) -> pd.DataFrame:
    params = {m[0]: count_params(m[4]) for m in MODELS if any((m[0], c[0]) in results for c in CONFIGS)}
    rows = []
    for cfg, _, cfg_label in CONFIGS:
        for model, label, *_ in models_in(results, cfg):
            r = results[(model, cfg)]
            row = {"konfigurácia": cfg_label, "model": label}
            for m in METRICS:
                row[f"{m}_mean"] = r["runs"][m].mean()
                row[f"{m}_std"] = r["runs"][m].std(ddof=0)
            row["acc_best_run"] = r["runs"]["acc"].max()
            row["params_total"], row["params_trainable_start"] = params[model]
            row["total_duration"] = r["duration"]
            if model != "M1" and ("M1", cfg) in results:
                row["acc_welch_p_vs_M1"] = welch_p(results[("M1", cfg)]["runs"]["acc"], r["runs"]["acc"])
            rows.append(row)
    return pd.DataFrame(rows)


def write_stat_tests(results: dict) -> None:
    lines = ["# Štatistické testy testovacej accuracy (3 behy na model)", ""]
    for cfg, _, cfg_label in CONFIGS:
        present = models_in(results, cfg)
        if len(present) < 2:
            continue
        groups = [results[(m[0], cfg)]["runs"]["acc"] for m in present]
        f_stat, p_anova = stats.f_oneway(*groups)
        lines += [f"## {cfg_label}", "",
                  f"Jednocestná ANOVA (počet modelov: {len(present)}): F = {f_stat:.2f}, p = {p_anova:.2e}", "",
                  "Párové Welchove t-testy (p-hodnoty):", ""]
        names = [m[0] for m in present]
        matrix = pd.DataFrame(np.nan, index=names, columns=names)
        for (a, *_), (b, *_) in combinations(present, 2):
            p = welch_p(results[(a, cfg)]["runs"]["acc"], results[(b, cfg)]["runs"]["acc"])
            matrix.loc[a, b] = matrix.loc[b, a] = p
        lines += [matrix.to_markdown(floatfmt=".4f"), ""]
    lines += ["Poznámka: pri 3 behoch na model majú testy nízku silu; p < 0,05 berieme ako",
              "štatisticky významný rozdiel. Ak majú oba modely nulový rozptyl, t-test nie je definovaný (nan)."]
    (OUT_DIR / "stat_tests.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def plot_accuracy(results: dict) -> None:
    x = np.arange(len(CONFIGS))
    width = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (model, label, *_) in enumerate(MODELS):
        means = [results[(model, c[0])]["runs"]["acc"].mean() if (model, c[0]) in results else np.nan for c in CONFIGS]
        stds = [results[(model, c[0])]["runs"]["acc"].std(ddof=0) if (model, c[0]) in results else 0 for c in CONFIGS]
        bars = ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=label)
        ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v * 100:.1f}" for v in means], padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in CONFIGS])
    ax.set_ylabel("Testovacia accuracy")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("M1-M5 - testovacia accuracy (priemer 3 behov +/- std)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_accuracy.png", dpi=150)
    plt.close(fig)


def plot_metrics(results: dict) -> None:
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(14, 5), sharey=True)
    x = np.arange(len(METRICS))
    width = 0.8 / len(MODELS)
    for ax, (cfg, _, cfg_label) in zip(axes, CONFIGS):
        for i, (model, label, *_) in enumerate(MODELS):
            if (model, cfg) not in results:
                continue
            runs = results[(model, cfg)]["runs"]
            ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, [runs[m].mean() for m in METRICS], width,
                   yerr=[runs[m].std(ddof=0) for m in METRICS], capsize=2, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
        ax.set_ylim(0.8, 1.0)
        ax.set_title(cfg_label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Testovacia hodnota metriky")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("M1-M5 - test metriky (makro priemer, 3 behy +/- std)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "compare_metrics.png", dpi=150)
    plt.close(fig)


def plot_val_curves(results: dict) -> None:
    fig, axes = plt.subplots(len(CONFIGS), 2, figsize=(13, 9), sharex=True)
    for row, (cfg, _, cfg_label) in enumerate(CONFIGS):
        for model, label, *_ in models_in(results, cfg):
            hist = results[(model, cfg)]["best_history"]
            axes[row][0].plot(hist["epoch"], hist["val_loss"], linewidth=2, label=label)
            axes[row][1].plot(hist["epoch"], hist["val_acc"], linewidth=2, label=label)
        axes[row][0].set_ylabel(f"{cfg_label}\nvalidačný loss")
        axes[row][1].set_ylabel("validačná accuracy")
        for ax in axes[row]:
            ax.grid(alpha=0.3)
        axes[row][1].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("M1-M5 - priebeh validácie najlepšieho behu", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "compare_val_curves.png", dpi=150)
    plt.close(fig)


def plot_class_recall(results: dict) -> None:
    classes = config.CLASSES
    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(14, 7.5))
    for ax, (cfg, _, cfg_label) in zip(axes, CONFIGS):
        present = models_in(results, cfg)
        recall = np.array([np.diag(results[(m[0], cfg)]["cm_sum"]) / results[(m[0], cfg)]["cm_sum"].sum(axis=1)
                           for m in present])
        im = ax.imshow(recall, cmap="RdYlGn", vmin=0.75, vmax=1.0, aspect="auto")
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([m[1] for m in present])
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{cfg_label} - recall po triedach (3 behy spolu, testovacie dáta)")
        for i in range(recall.shape[0]):
            for j in range(recall.shape[1]):
                ax.text(j, i, f"{recall[i, j] * 100:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_class_recall.png", dpi=150)
    plt.close(fig)


def plot_acc_vs_time(results: dict) -> None:
    markers = {"baseline": "o", "gradual_unfreeze": "s"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (model, label, *_) in enumerate(MODELS):
        for cfg, _, cfg_label in CONFIGS:
            if (model, cfg) not in results:
                continue
            r = results[(model, cfg)]
            ax.errorbar(duration_minutes(r["duration"]), r["runs"]["acc"].mean(), yerr=r["runs"]["acc"].std(ddof=0),
                        fmt=markers[cfg], color=colors[i], markersize=9, capsize=3,
                        label=f"{label} ({cfg_label})")
    ax.set_xlabel("Čas trénovania 3 behov [min]")
    ax.set_ylabel("Testovacia accuracy")
    ax.set_title("M1-M5 - accuracy vs. čas trénovania (o = zamrznutý, □ = rozmrazovanie)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "compare_acc_vs_time.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = build_table(results)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    (OUT_DIR / "comparison_table.md").write_text(table.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(table.to_string(index=False))

    write_stat_tests(results)
    plot_accuracy(results)
    plot_metrics(results)
    plot_val_curves(results)
    plot_class_recall(results)
    plot_acc_vs_time(results)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
