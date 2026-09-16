"""
Spoločné vyhodnocovacie funkcie pre všetky modely (SVM, MLP, LSTM, CNN).

Zadanie vyžaduje: úspešnosť (accuracy), presnosť (precision), senzitivita (recall)
na validačných a testovacích dátach, a kontingenčnú maticu pre najlepšie modely.

Aby bolo porovnanie modelov korektné, všetky používajú tie isté funkcie odtiaľto.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from config import RESULTS_DIR, TERRAIN_NAMES

CLASS_LABELS = list(TERRAIN_NAMES.values())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Vypočíta accuracy, precision a recall.

    Precision/recall sú 'macro' priemery - každá trieda má rovnakú váhu bez
    ohľadu na počet vzoriek. Pri vyvážených triedach (u nás 16 okien na triedu)
    je to prirodzená voľba a zodpovedá 'presnosť/senzitivita' zo zadania.

    zero_division=0: ak model nejakú triedu nikdy nepredpovie, precision pre ňu
    je 0 namiesto varovania/NaN.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Jednotný formát výpisu metrík v %."""
    return (
        f"acc={metrics['accuracy'] * 100:6.2f} %  "
        f"prec={metrics['precision'] * 100:6.2f} %  "
        f"recall={metrics['recall'] * 100:6.2f} %"
    )


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Zo zoznamu metrík z viacerých behov (zadanie: 5 opakovaní) spočíta
    min / max / priemer / smerodajnú odchýlku pre každú metriku.
    """
    summary = {}
    for key in ("accuracy", "precision", "recall"):
        values = np.array([run[key] for run in runs])
        summary[key] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def print_summary(name: str, summary: dict[str, dict[str, float]]) -> None:
    """Prehľadný výpis súhrnu z viacerých behov."""
    print(f"\n{name}")
    print(f"  {'metrika':<12} {'min':>8} {'max':>8} {'priemer':>9} {'std':>8}")
    for key, stats in summary.items():
        print(
            f"  {key:<12} {stats['min'] * 100:7.2f}% {stats['max'] * 100:7.2f}% "
            f"{stats['mean'] * 100:8.2f}% {stats['std'] * 100:7.2f}%"
        )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str | None = None,
    normalize: bool = False,
    out_dir: Path = RESULTS_DIR / "figures",
):
    """
    Zobrazí (a voliteľne uloží) kontingenčnú maticu.

    normalize=True zobrazí podiely v rámci skutočnej triedy (riadky), čo sa
    ľahšie číta pri porovnávaní modelov medzi sebou.
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predikovaná trieda")
    ax.set_ylabel("Skutočná trieda")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if filename:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=150)
        print(f"  Kontingenčná matica uložená: {out_dir / filename}")

    return fig, cm
