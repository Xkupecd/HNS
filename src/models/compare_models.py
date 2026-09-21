"""
Zhrnutie a porovnanie všetkých modelov (SVM, MLP, LSTM, CNN) do jednej tabuľky.

Každý model si pri behu uloží do results/<model>/<model>_training_log.md kompletný
výpis z konzoly vrátane finálnej sekcie s 5 rozdeleniami:

    split_00:  VAL  acc= 96.88 %  prec= 97.17 %  recall= 96.88 %   |   TEST  acc= 88.54 % ...

Tento skript tieto riadky z logov vyčíta, prepočíta z nich rovnaké súhrny ako
evaluate.summarize_runs (min/max/priemer/std) a poskladá z nich:

    results/comparison/model_comparison.md   - tabuľky pre dokumentáciu
    results/comparison/model_comparison.csv  - to isté do Excelu
    results/comparison/model_comparison.png  - stĺpcový graf val vs test

Výhoda oproti počítaniu nanovo: netreba nič pretrénovať, porovnanie sa poskladá
z už uložených výsledkov za pár sekúnd. Nevýhoda: závisí od formátu výpisu -
ak sa zmení formát v evaluate.format_metrics, treba upraviť SPLIT_RE nižšie.

Spustenie (z priečinka src/):
    python -m models.compare_models
"""
import csv
import re
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import RESULTS_DIR
from evaluate import round_half_up, summarize_runs

RESULTS_COMPARISON_DIR = RESULTS_DIR / "comparison"

# Poradie modelov v tabuľke + čím sa kŕmia (kvôli porovnaniu "ručné príznaky
# vs. automatická extrakcia", ktoré zadanie vyžaduje)
MODELS = {
    "svm": "54 ručných príznakov",
    "mlp": "54 ručných príznakov",
    "lstm": "surový signál (400 x 6)",
    "cnn": "surový signál (400 x 6)",
}

# "  split_00:  VAL  acc= 96.88 %  prec= 97.17 %  recall= 96.88 %   |   TEST  acc= 88.54 % ..."
SPLIT_RE = re.compile(
    r"split_(?P<split>\d+):\s+VAL\s+"
    r"acc=\s*(?P<val_acc>[\d.]+)\s*%\s+prec=\s*(?P<val_prec>[\d.]+)\s*%\s+recall=\s*(?P<val_rec>[\d.]+)\s*%"
    r"\s*\|\s*TEST\s+"
    r"acc=\s*(?P<test_acc>[\d.]+)\s*%\s+prec=\s*(?P<test_prec>[\d.]+)\s*%\s+recall=\s*(?P<test_rec>[\d.]+)\s*%"
)

# Riadky, z ktorých sa dá vyčítať, ktorá konfigurácia nakoniec vyhrala.
# Každý model si ju pomenúva trochu inak, tak berieme všetky varianty.
BEST_PREFIXES = (
    "Najlepšia architektúra:",
    "Najlepšia konfigurácia podľa validačnej úspešnosti:",
    "Najlepšie parametre trénovania:",
)


def parse_training_log(log_path: Path) -> dict:
    """
    Z logu jedného modelu vytiahne metriky z finálnych 5 behov a názov
    víťaznej konfigurácie.

    Vracia {"val_runs": [...], "test_runs": [...], "best": str}, kde run je
    slovník {"accuracy", "precision", "recall"} v podiele 0-1 - teda presne
    vo formáte, aký očakáva evaluate.summarize_runs.
    """
    text = log_path.read_text(encoding="utf-8")

    val_runs, test_runs = [], []
    for m in SPLIT_RE.finditer(text):
        pct = lambda key: float(m.group(key)) / 100.0
        val_runs.append({"accuracy": pct("val_acc"), "precision": pct("val_prec"), "recall": pct("val_rec")})
        test_runs.append({"accuracy": pct("test_acc"), "precision": pct("test_prec"), "recall": pct("test_rec")})

    if not val_runs:
        raise ValueError(f"V logu {log_path} sa nenašli žiadne riadky 'split_NN: VAL ...'.")

    # Víťaznú konfiguráciu berieme z POSLEDNÉHO takého riadku - je to ten,
    # ktorý sa reálne použil vo finálnych 5 behoch.
    best = ""
    for line in text.splitlines():
        line = line.strip()
        for prefix in BEST_PREFIXES:
            if line.startswith(prefix):
                best = line[len(prefix):].strip()

    return {"val_runs": val_runs, "test_runs": test_runs, "best": best}


def collect_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """
    Prejde results/<model>/<model>_training_log.md pre všetky známe modely.
    Model, ktorý ešte nebol spustený (chýba log), sa preskočí s upozornením.
    """
    collected = {}
    for model in MODELS:
        log_path = results_dir / model / f"{model}_training_log.md"
        if not log_path.exists():
            print(f"  [preskočené] {model.upper()}: chýba {log_path} - spusti najprv ten model.")
            continue

        parsed = parse_training_log(log_path)
        collected[model] = {
            "best": parsed["best"],
            "n_runs": len(parsed["val_runs"]),
            "val": summarize_runs(parsed["val_runs"]),
            "test": summarize_runs(parsed["test_runs"]),
            "val_accs": [run["accuracy"] for run in parsed["val_runs"]],
            "test_accs": [run["accuracy"] for run in parsed["test_runs"]],
        }
    return collected


# ---------------------------------------------------------------------------
# Výstupy
# ---------------------------------------------------------------------------
def _rank(collected: dict[str, dict]) -> list[str]:
    """Modely zoradené podľa priemernej testovacej úspešnosti (najlepší prvý)."""
    return sorted(collected, key=lambda m: collected[m]["test"]["accuracy"]["mean"], reverse=True)


def print_comparison(collected: dict[str, dict]) -> None:
    """Hlavná tabuľka do konzoly - priemer +- std pre val aj test."""
    print(f"\n{'=' * 104}")
    print("Porovnanie modelov (priemer z 5 nezávislých rozdelení)")
    print(f"{'=' * 104}\n")
    print(f"  {'model':<6} {'vstup':<26} {'VAL acc':>16} {'TEST acc':>16} "
          f"{'TEST prec':>16} {'TEST recall':>16}")

    for model in _rank(collected):
        r = collected[model]
        cell = lambda part, key: (
            f"{round_half_up(r[part][key]['mean'] * 100):6.2f}±{round_half_up(r[part][key]['std'] * 100):4.2f}%"
        )
        print(f"  {model.upper():<6} {MODELS[model]:<26} {cell('val', 'accuracy'):>16} "
              f"{cell('test', 'accuracy'):>16} {cell('test', 'precision'):>16} {cell('test', 'recall'):>16}")

    print(f"\n  {'model':<6} {'TEST min':>10} {'TEST max':>10}   víťazná konfigurácia")
    for model in _rank(collected):
        acc = collected[model]["test"]["accuracy"]
        print(f"  {model.upper():<6} {round_half_up(acc['min'] * 100):9.2f}% {round_half_up(acc['max'] * 100):9.2f}%   "
              f"{collected[model]['best']}")


def write_markdown(collected: dict[str, dict], out_dir: Path = RESULTS_COMPARISON_DIR) -> Path:
    """Markdown tabuľky - priamo na skopírovanie do dokumentácie/odovzdania."""
    lines = ["# Porovnanie modelov (SVM / MLP / LSTM / CNN)", ""]
    lines += [f"Zdroj: `results/<model>/<model>_training_log.md`, "
              f"priemer z {max(r['n_runs'] for r in collected.values())} nezávislých rozdelení.", ""]

    lines += ["## Súhrn", "",
              "| Model | Vstup | VAL accuracy | TEST accuracy | TEST precision | TEST recall |",
              "|---|---|---|---|---|---|"]
    for model in _rank(collected):
        r = collected[model]
        cell = lambda part, key: (
            f"{round_half_up(r[part][key]['mean'] * 100):.2f} % ± {round_half_up(r[part][key]['std'] * 100):.2f}"
        )
        lines.append(f"| **{model.upper()}** | {MODELS[model]} | {cell('val', 'accuracy')} | "
                     f"{cell('test', 'accuracy')} | {cell('test', 'precision')} | {cell('test', 'recall')} |")

    lines += ["", "## Rozptyl naprieč behmi (testovacie dáta)", "",
              "| Model | min | max | priemer | std | Víťazná konfigurácia |", "|---|---|---|---|---|---|"]
    for model in _rank(collected):
        acc = collected[model]["test"]["accuracy"]
        lines.append(f"| **{model.upper()}** | {round_half_up(acc['min'] * 100):.2f} % | "
                     f"{round_half_up(acc['max'] * 100):.2f} % | {round_half_up(acc['mean'] * 100):.2f} % | "
                     f"{round_half_up(acc['std'] * 100):.2f} | `{collected[model]['best']}` |")

    lines += ["", "## Accuracy po jednotlivých rozdeleniach (testovacie dáta)", "",
              "| Model | " + " | ".join(f"split_{i:02d}" for i in range(max(r["n_runs"] for r in collected.values()))) + " |",
              "|---" * (1 + max(r["n_runs"] for r in collected.values())) + "|"]
    for model in _rank(collected):
        accs = " | ".join(f"{round_half_up(a * 100):.2f} %" for a in collected[model]["test_accs"])
        lines.append(f"| **{model.upper()}** | {accs} |")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Tabuľka uložená: {out_path}")
    return out_path


def write_csv(collected: dict[str, dict], out_dir: Path = RESULTS_COMPARISON_DIR) -> Path:
    """To isté do CSV (percentá ako čísla, nech sa dá v Exceli ďalej počítať)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.csv"

    header = ["model", "vstup", "pocet_behov"]
    for part in ("val", "test"):
        for key in ("accuracy", "precision", "recall"):
            header += [f"{part}_{key}_{stat}" for stat in ("mean", "std", "min", "max")]
    header.append("najlepsia_konfiguracia")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model in _rank(collected):
            r = collected[model]
            row = [model.upper(), MODELS[model], r["n_runs"]]
            for part in ("val", "test"):
                for key in ("accuracy", "precision", "recall"):
                    row += [round_half_up(r[part][key][stat] * 100) for stat in ("mean", "std", "min", "max")]
            row.append(r["best"])
            writer.writerow(row)

    print(f"  CSV uložené: {out_path}")
    return out_path


def plot_comparison(collected: dict[str, dict], out_dir: Path = RESULTS_COMPARISON_DIR) -> Path:
    """
    Stĺpcový graf val vs test s chybovými úsečkami po min/max.
    Farby a štýl sú rovnaké ako v evaluate.plot_runs_bar_chart, nech grafy
    v odovzdaní vyzerajú jednotne.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = _rank(collected)
    x = np.arange(len(models))
    width = 0.35

    def bars(part, offset, color, label):
        means = np.array([collected[m][part]["accuracy"]["mean"] * 100 for m in models])
        lows = means - np.array([collected[m][part]["accuracy"]["min"] * 100 for m in models])
        highs = np.array([collected[m][part]["accuracy"]["max"] * 100 for m in models]) - means
        ax.bar(x + offset, means, width, label=label, color=color,
               yerr=[lows, highs], capsize=4, ecolor="#444444")
        # popisok nad hornú úsečku (max), nech sa neprekrýva s chybovou úsečkou
        for xi, mean, high in zip(x + offset, means, highs):
            ax.text(xi, mean + high + 1.5, f"{mean:.1f}", ha="center", fontsize=9)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.8), 6))
    bars("val", -width / 2, "#4C72B0", "Validačné dáta")
    bars("test", width / 2, "#DD8452", "Testovacie dáta")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.upper()}\n{MODELS[m]}" for m in models], fontsize=9)
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Porovnanie modelov - priemerná úspešnosť z 5 rozdelení (úsečky = min-max)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Graf uložený: {out_path}")
    return out_path


if __name__ == "__main__":
    collected = collect_results()
    if not collected:
        raise SystemExit("Nenašli sa žiadne výsledky - spusti najprv aspoň jeden model.")

    print_comparison(collected)
    write_markdown(collected)
    write_csv(collected)
    plot_comparison(collected)
