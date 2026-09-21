"""
Spoločné vyhodnocovacie funkcie pre všetky modely (SVM, MLP, LSTM, CNN).

Zadanie vyžaduje: úspešnosť (accuracy), presnosť (precision), senzitivita (recall)
na validačných a testovacích dátach, a kontingenčnú maticu pre najlepšie modely.

Aby bolo porovnanie modelov korektné, všetky používajú tie isté funkcie odtiaľto.

Oproti pôvodnej verzii pribudlo:
- väčšie, čitateľné písmo vo všetkých grafoch (aj po otočení na šírku v PDF),
- ukladanie kompletných výsledkov KAŽDÉHO behu (predikcie, história, čas trénovania)
  do results/<model>/runs/, nielen posledného splitu,
- kontingenčné matice: jedna pre KAŽDÝ beh, jedna SÚČTOVÁ (spojenie všetkých
  5 behov) a jedna pre NAJLEPŠÍ beh - všetky ako dvojica počty + percentá.
"""
import contextlib
import io
import json
import sys
import time
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

# ---------------------------------------------------------------------------
# Jednotné veľkosti písma pre všetky grafy.
#
# Grafy s priebehom trénovania (2x2 panely) sa v dokumentácii vkladajú otočené
# na šírku takmer celej strany (sidewaysfigure) - aj tak však vychádzajú z
# obrázka vygenerovaného pri fixnom rozlíšení (dpi=150), takže čitateľnosť
# textu v PDF závisí výlučne od toho, aké veľké písmo je v ňom "zapečené".
# Pôvodné (matplotlib default, cca 8-10 bodov) bolo pri zložitejších grafoch
# (veľa kriviek, dlhé popisky konfigurácií v legende) prakticky nečitateľné.
# ---------------------------------------------------------------------------
FONT_SUPTITLE = 19
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK = 13
FONT_LEGEND = 13
FONT_ANNOT = 13


def _style_axes(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Jednotne nastaví veľkosti písma na jednej osi (title/label/ticks/grid)."""
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(alpha=0.3)


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


def select_best_run(runs: list[dict[str, float]]) -> int:
    """
    Vyberie index 'najlepšieho' behu spomedzi viacerých opakovaní - podľa accuracy,
    pri zhode (bežné pri 96 oknách, kde accuracy má hrubé rozlíšenie ~1 %) podľa
    súčtu precision+recall. Rovnaká logika ako pri výbere konfigurácie v SVM.

    Používa sa AŽ PO tom, čo sú všetky behy hotové - nie na výber hyperparametrov,
    len na to, ktorý beh sa ukáže ako reprezentatívny "najlepší prípad".
    """
    scores = [(run["accuracy"], run["precision"] + run["recall"]) for run in runs]
    return max(range(len(runs)), key=lambda i: scores[i])


# ---------------------------------------------------------------------------
# Kontingenčné matice
# ---------------------------------------------------------------------------
def _draw_confusion_matrix(ax, cm: np.ndarray, normalize: bool, title: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot(
        ax=ax, cmap="Blues", values_format=".1f" if normalize else "d", colorbar=False,
        text_kw={"fontsize": FONT_ANNOT},
    )
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.set_xlabel("Predikovaná trieda", fontsize=FONT_LABEL)
    ax.set_ylabel("Skutočná trieda", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_dir: Path,
    filename: str | None = None,
    normalize: bool = False,
):
    """
    Zobrazí (a voliteľne uloží) jednu kontingenčnú maticu.

    normalize=True zobrazí podiely v rámci skutočnej triedy (riadky), čo sa
    ľahšie číta pri porovnávaní modelov medzi sebou.

    Ponechané pre spätnú kompatibilitu (staršie volania) - pre nové výstupy
    je vhodnejšia plot_confusion_matrix_pair (počty aj percentá naraz).
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_confusion_matrix(ax, cm, normalize, title)
    plt.tight_layout()

    if filename:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=150)
        print(f"  Kontingenčná matica uložená: {out_dir / filename}")

    return fig, cm


def plot_confusion_matrix_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_dir: Path,
    filename: str,
    n_okien_info: str | None = None,
) -> np.ndarray:
    """
    Kontingenčná matica v absolútnych počtoch AJ v percentách (podiel v rámci
    skutočnej triedy) vedľa seba v jednom obrázku - najviac informácie na jeden
    súbor. Používa sa pre matice jednotlivých behov, súčtovú aj pre najlepší beh.

    n_okien_info: voliteľný podtitulok (napr. "5 behov spolu, 480 okien"),
    aby bolo z obrázka samého jasné, z čoho matica pochádza.
    """
    import matplotlib.pyplot as plt

    cm_counts = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    cm_pct = cm_counts.astype(float) / np.maximum(cm_counts.sum(axis=1, keepdims=True), 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _draw_confusion_matrix(axes[0], cm_counts, normalize=False, title="Absolútne počty")
    _draw_confusion_matrix(axes[1], cm_pct, normalize=True, title="Percentá v rámci skutočnej triedy")

    suptitle = title if not n_okien_info else f"{title}\n{n_okien_info}"
    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if n_okien_info else 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kontingenčná matica uložená: {out_dir / filename}")
    return cm_counts


def generate_post_run_confusion_matrices(
    model_label: str,
    out_dir: Path,
    y_vals: list[np.ndarray],
    y_val_preds: list[np.ndarray],
    y_tests: list[np.ndarray],
    y_test_preds: list[np.ndarray],
    test_runs: list[dict[str, float]],
    best_filename: str,
    combined_test_filename: str,
    combined_val_filename: str,
) -> int:
    """
    Po dobehnutí všetkých 5 behov vygeneruje:
    - súčtovú kontingenčnú maticu na TESTOVACÍCH dátach (spojenie predikcií
      ku všetkým behom naraz - napr. 5x96 = 480 okien),
    - súčtovú kontingenčnú maticu na VALIDAČNÝCH dátach (rovnako),
    - kontingenčnú maticu NAJLEPŠIEHO behu (podľa select_best_run) na testovacích
      dátach - nahrádza pôvodné ľubovoľné použitie posledného splitu.

    Vracia index najlepšieho behu, aby ho vedel model ešte využiť (napr. do logu).
    """
    n_total_test = sum(len(y) for y in y_tests)
    plot_confusion_matrix_pair(
        np.concatenate(y_tests), np.concatenate(y_test_preds),
        title=f"{model_label} - kontingenčná matica (súčet z {len(y_tests)} behov, testovacie dáta)",
        out_dir=out_dir,
        filename=combined_test_filename,
        n_okien_info=f"{len(y_tests)} behov spolu, {n_total_test} okien",
    )

    n_total_val = sum(len(y) for y in y_vals)
    plot_confusion_matrix_pair(
        np.concatenate(y_vals), np.concatenate(y_val_preds),
        title=f"{model_label} - kontingenčná matica (súčet z {len(y_vals)} behov, validačné dáta)",
        out_dir=out_dir,
        filename=combined_val_filename,
        n_okien_info=f"{len(y_vals)} behov spolu, {n_total_val} okien",
    )

    best_idx = select_best_run(test_runs)
    plot_confusion_matrix_pair(
        y_tests[best_idx], y_test_preds[best_idx],
        title=f"{model_label} - kontingenčná matica (najlepší beh, testovacie dáta)",
        out_dir=out_dir,
        filename=best_filename,
        n_okien_info=f"najlepší beh: split_{best_idx:02d} "
                      f"(TEST acc = {test_runs[best_idx]['accuracy'] * 100:.2f} %)",
    )
    return best_idx


# ---------------------------------------------------------------------------
# Priebehy trénovania
# ---------------------------------------------------------------------------
def plot_training_curves_comparison(
    histories: dict[str, dict[str, list[float]]],
    title: str,
    filename: str,
    out_dir: Path,
) -> None:
    """
    Porovnanie priebehov trénovania (train/val loss a accuracy podľa epoch) naprieč
    viacerými konfiguráciami/behmi na jednom obrázku (zadanie, body 1, 2 a 6).

    histories: {popis_konfigurácie_alebo_behu: history}, kde history má kľúče
    'train_loss', 'val_loss', 'train_acc', 'val_acc' (presne formát z
    mlp_model.train_model / cnn_model.train_model). Popis (kľúč) by mal obsahovať
    hyperparametre (optimizer/lr/batch/epochs), aby graf sám dokumentoval nastavenie.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm as colormaps

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panels = [
        ("train_loss", axes[0, 0], "Train loss"),
        ("val_loss", axes[0, 1], "Val loss"),
        ("train_acc", axes[1, 0], "Train accuracy"),
        ("val_acc", axes[1, 1], "Val accuracy"),
    ]

    n_curves = max(len(histories), 1)
    colors = colormaps.get_cmap("tab10" if n_curves <= 10 else "tab20")(np.linspace(0, 1, n_curves))

    for key, ax, panel_title in panels:
        for color, (label, history) in zip(colors, histories.items()):
            values = history[key]
            epochs = range(1, len(values) + 1)
            if "acc" in key:
                ax.plot(epochs, [v * 100 for v in values], color=color, label=label, linewidth=1.8)
            else:
                ax.plot(epochs, values, color=color, label=label, linewidth=1.8)
        _style_axes(ax, panel_title, "Epocha", "Accuracy [%]" if "acc" in key else "Loss")

    # Legenda pod grafmi, do viacerých riadkov podľa počtu položiek - pri veľa
    # konfiguráciách (napr. 9 pri porovnaní hyperparametrov) je jeden riadok
    # nečitateľný, preto ncol škáluje s počtom kriviek namiesto pevnej hodnoty.
    #
    # Miesto pre legendu sa rezervuje LEN RAZ - cez rect v tight_layout - a
    # legenda sa potom ukotví presne na spodný okraj tejto rezervovanej oblasti
    # (loc="upper center", bbox_to_anchor na tú istú hodnotu ako spodná hranica
    # rect). Predošlá verzia rezervovala miesto cez rect AJ naviac posúvala
    # legendu pod hranicu obrázka (záporný bbox_to_anchor), čo sa pri ukladaní
    # s bbox_inches="tight" sčítalo do zbytočne veľkej prázdnej plochy dole.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 1 if n_curves <= 2 else (2 if n_curves <= 6 else 3)
    n_rows_legend = -(-len(labels) // ncol)  # zaokrúhlenie nahor
    legend_frac = 0.04 + 0.045 * n_rows_legend

    fig.suptitle(title, fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=(0, legend_frac, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=FONT_LEGEND,
               bbox_to_anchor=(0.5, legend_frac - 0.005), bbox_transform=fig.transFigure, frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Priebeh trénovania uložený: {out_dir / filename}")


def plot_single_run_training_curve(
    history: dict[str, list[float]],
    title: str,
    filename: str,
    out_dir: Path,
) -> None:
    """
    Priebeh trénovania JEDNÉHO behu (jeden split) - train aj val krivka v tom
    istom paneli (dve farby), aby bolo v jednom obrázku vidno aj to, či sa model
    na danom splite preučil. Ukladá sa pre každý z 5 finálnych behov zvlášť.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"],
                 color="#4C72B0", linewidth=2, label="Trénovacie dáta")
    axes[0].plot(range(1, len(history["val_loss"]) + 1), history["val_loss"],
                 color="#DD8452", linewidth=2, label="Validačné dáta")
    _style_axes(axes[0], "Loss", "Epocha", "Loss")
    axes[0].legend(fontsize=FONT_LEGEND)

    axes[1].plot(range(1, len(history["train_acc"]) + 1), [v * 100 for v in history["train_acc"]],
                 color="#4C72B0", linewidth=2, label="Trénovacie dáta")
    axes[1].plot(range(1, len(history["val_acc"]) + 1), [v * 100 for v in history["val_acc"]],
                 color="#DD8452", linewidth=2, label="Validačné dáta")
    _style_axes(axes[1], "Accuracy", "Epocha", "Accuracy [%]")
    axes[1].legend(fontsize=FONT_LEGEND)

    fig.suptitle(title, fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_runs_bar_chart(
    labels: list[str],
    val_values: list[float],
    test_values: list[float],
    title: str,
    ylabel: str,
    filename: str,
    out_dir: Path,
    val_summary: dict[str, dict[str, float]] | None = None,
    test_summary: dict[str, dict[str, float]] | None = None,
) -> None:
    """
    Stĺpcový graf úspešnosti (accuracy) naprieč konfiguráciami alebo behmi (splitmi),
    val vs test vedľa seba. Ak sú zadané val_summary/test_summary (výstup summarize_runs),
    dokreslia sa vodorovné čiary priemeru a pásmo min-max - vizuálny náprotivok print_summary.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 6.5))
    ax.bar(x - width / 2, [v * 100 for v in val_values], width, label="Validačné dáta", color="#4C72B0")
    ax.bar(x + width / 2, [v * 100 for v in test_values], width, label="Testovacie dáta", color="#DD8452")

    if val_summary is not None:
        acc = val_summary["accuracy"]
        ax.axhline(acc["mean"] * 100, color="#4C72B0", linestyle="--", linewidth=1.3,
                    label=f"Priemer val = {acc['mean'] * 100:.1f} %")
        ax.axhspan(acc["min"] * 100, acc["max"] * 100, color="#4C72B0", alpha=0.08)
    if test_summary is not None:
        acc = test_summary["accuracy"]
        ax.axhline(acc["mean"] * 100, color="#DD8452", linestyle="--", linewidth=1.3,
                    label=f"Priemer test = {acc['mean'] * 100:.1f} %")
        ax.axhspan(acc["min"] * 100, acc["max"] * 100, color="#DD8452", alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_TICK)
    _style_axes(ax, title, None, ylabel)
    ax.legend(fontsize=FONT_LEGEND)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    print(f"  Graf úspešnosti uložený: {out_dir / filename}")


# ---------------------------------------------------------------------------
# Ukladanie surových výsledkov KAŽDÉHO behu
# ---------------------------------------------------------------------------
def save_run_artifacts(
    out_dir: Path,
    split_idx: int,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    train_seconds: float,
    history: dict[str, list[float]] | None = None,
) -> None:
    """
    Uloží kompletný výsledok JEDNÉHO behu (split) do results/<model>/runs/:
    - split_XX_predictions.npz  - y_val, y_val_pred, y_test, y_test_pred
    - split_XX_history.json     - priebeh trénovania po epochách (ak existuje - SVM nemá)

    Vďaka tomu sa dá kedykoľvek dodatočne spočítať čokoľvek nad predikciami
    (súčtová kontingenčná matica, vyhodnotenie na úrovni merania, ...) bez
    opätovného trénovania.
    """
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        runs_dir / f"split_{split_idx:02d}_predictions.npz",
        y_val=y_val, y_val_pred=y_val_pred,
        y_test=y_test, y_test_pred=y_test_pred,
    )

    if history is not None:
        with open(runs_dir / f"split_{split_idx:02d}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    meta = {
        "split": split_idx,
        "train_seconds": round(train_seconds, 3),
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(runs_dir / f"split_{split_idx:02d}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_runs_csv(
    out_dir: Path,
    val_runs: list[dict[str, float]],
    test_runs: list[dict[str, float]],
    train_seconds: list[float],
) -> Path:
    """
    Zapíše results/<model>/runs/runs.csv - jeden riadok na split (val/test
    accuracy/precision/recall + čas trénovania) a na konci riadky min/max/priemer/std.
    """
    import csv

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "runs.csv"

    header = ["split", "val_acc", "val_prec", "val_rec", "test_acc", "test_prec", "test_rec", "train_seconds"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (val_m, test_m, secs) in enumerate(zip(val_runs, test_runs, train_seconds)):
            writer.writerow([
                f"split_{i:02d}",
                round(val_m["accuracy"] * 100, 2), round(val_m["precision"] * 100, 2), round(val_m["recall"] * 100, 2),
                round(test_m["accuracy"] * 100, 2), round(test_m["precision"] * 100, 2), round(test_m["recall"] * 100, 2),
                round(secs, 2),
            ])

        val_summary = summarize_runs(val_runs)
        test_summary = summarize_runs(test_runs)
        # Pri čase trénovania má zmysel len priemer (na riadku "priemer") a súčet
        # (extra riadok "spolu") - min/max/std času nie sú pre dokumentáciu podstatné.
        extra_time = {"priemer": np.mean(train_seconds), "spolu": np.sum(train_seconds)}
        for stat in ("min", "max", "mean", "std"):
            label = {"min": "min", "max": "max", "mean": "priemer", "std": "std"}[stat]
            writer.writerow([
                label,
                round(val_summary["accuracy"][stat] * 100, 2), round(val_summary["precision"][stat] * 100, 2),
                round(val_summary["recall"][stat] * 100, 2),
                round(test_summary["accuracy"][stat] * 100, 2), round(test_summary["precision"][stat] * 100, 2),
                round(test_summary["recall"][stat] * 100, 2),
                round(extra_time[label], 2) if label in extra_time else "",
            ])
        writer.writerow(["spolu (čas)", "", "", "", "", "", "", round(extra_time["spolu"], 2)])

    print(f"  Tabuľka behov uložená: {out_path}")
    return out_path


class _Tee:
    """Zapíše dáta naraz do viacerých streamov (napr. reálna konzola + pamäťový buffer)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def capture_console_to_markdown(filename: str, out_dir: Path, title: str):
    """
    Zachytí všetko, čo sa vnútri bloku `with` vypíše cez print() (celý priebeh
    porovnávania hyperparametrov/architektúr aj 5 behov), a na konci to uloží
    ako .md súbor - nech sa dá presne zdokumentovať, čo sa pri tréningu dialo.

    Výpis sa naďalej zobrazuje aj v termináli (Tee) - toto ho iba navyše ukladá.
    """
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, buffer)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        out_dir.mkdir(parents=True, exist_ok=True)
        content = f"# {title}\n\n```text\n{buffer.getvalue()}```\n"
        (out_dir / filename).write_text(content, encoding="utf-8")
        print(f"  Priebeh trénovania (log) uložený: {out_dir / filename}")


@contextlib.contextmanager
def timed():
    """Meranie času behu (`with timed() as t: ...; t.seconds`), na zápis do runs.csv."""
    class _Timer:
        seconds = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start
