"""
Vygeneruje LaTeX tabuľky (booktabs) s výsledkami VŠETKÝCH 5 behov (val aj test,
accuracy/precision/recall) pre každý model, z už uložených results/<model>/runs/runs.csv.

Netrénuje nič - len prevedie CSV na .tex, aby ich dokumentácia mohla vložiť
cez \input{} a boli vždy presne zosynchronizované s výsledkami v results/.

Spustenie (z priečinka src/):
    python -m models.generate_runs_tables
"""
import csv
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import RESULTS_DIR

LATEX_TABLES_DIR = RESULTS_DIR.parent / "Latex" / "includes" / "tables"

MODELS = {
    "svm": "SVM",
    "mlp": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
}

# Poradie a slovenské popisky riadkov so súhrnnou štatistikou (koniec runs.csv)
SUMMARY_LABELS = {
    "min": "min",
    "max": "max",
    "priemer": "priemer",
    "std": "std",
}


def fmt(value: str) -> str:
    """Desatinná bodka -> čiarka, prázdne bunky ostanú prázdne."""
    if value == "":
        return ""
    return value.replace(".", ",")


def build_table(model_key: str, model_label: str) -> str:
    csv_path = RESULTS_DIR / model_key / "runs" / "runs.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    split_rows = [r for r in rows if r["split"].startswith("split_")]
    summary_rows = [r for r in rows if r["split"] in SUMMARY_LABELS]

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        rf"    \caption{{{model_label} -- výsledky jednotlivých behov (validačné a testovacie dáta)}}",
        rf"    \label{{tab:runs-{model_key}}}",
        r"    \footnotesize",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"         & \multicolumn{3}{c}{Validačné dáta} & \multicolumn{3}{c}{Testovacie dáta} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"        Beh & acc [\%] & prec [\%] & rec [\%] & acc [\%] & prec [\%] & rec [\%] \\",
        r"        \midrule",
    ]

    for r in split_rows:
        label = r["split"].replace("_", "\\_")
        lines.append(
            f"        {label} & {fmt(r['val_acc'])} & {fmt(r['val_prec'])} & {fmt(r['val_rec'])} & "
            f"{fmt(r['test_acc'])} & {fmt(r['test_prec'])} & {fmt(r['test_rec'])} \\\\"
        )

    lines.append(r"        \midrule")
    for r in summary_rows:
        label = SUMMARY_LABELS[r["split"]]
        lines.append(
            f"        {label} & {fmt(r['val_acc'])} & {fmt(r['val_prec'])} & {fmt(r['val_rec'])} & "
            f"{fmt(r['test_acc'])} & {fmt(r['test_prec'])} & {fmt(r['test_rec'])} \\\\"
        )

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    LATEX_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for key, label in MODELS.items():
        content = build_table(key, label)
        out_path = LATEX_TABLES_DIR / f"{key}_runs_table.tex"
        out_path.write_text(content, encoding="utf-8")
        print(f"  {out_path}")
