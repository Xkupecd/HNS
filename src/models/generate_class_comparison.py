"""
Vygeneruje tabuľku úspešnosti po triedach (recall, %) na testovacích dátach
(súčet 5 behov) pre všetky 4 modely, plus riadok s najčastejšou zámenou
(skutočná -> predikovaná, počet okien) pre každý model.

Netrénuje nič - počíta zo starých results/<model>/runs/split_XX_predictions.npz.

Spustenie (z priečinka src/):
    python -m models.generate_class_comparison
"""
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import csv

import numpy as np
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR, TERRAIN_NAMES
from evaluate import round_half_up

RESULTS_COMPARISON_DIR = RESULTS_DIR / "comparison"

CLASS_LABELS = list(TERRAIN_NAMES.values())
CLASS_LABELS_SK = ["betón", "tráva", "kamienky", "piesok", "dlažobný kameň", "bežecká dráha"]

MODELS = ["svm", "mlp", "cnn", "lstm"]
MODEL_LABELS = {"svm": "SVM", "mlp": "MLP", "cnn": "CNN", "lstm": "LSTM"}


def combined_test_predictions(model: str) -> tuple[np.ndarray, np.ndarray]:
    runs_dir = RESULTS_DIR / model / "runs"
    y_tests, y_preds = [], []
    for i in range(5):
        pred = np.load(runs_dir / f"split_{i:02d}_predictions.npz")
        y_tests.append(pred["y_test"])
        y_preds.append(pred["y_test_pred"])
    return np.concatenate(y_tests), np.concatenate(y_preds)


def class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    return [round_half_up(cm[i, i] / cm[i].sum() * 100) for i in range(len(CLASS_LABELS))]


def most_common_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, str, int]:
    """Najčastejšia MIMODIAGONÁLNA zámena (skutočná -> predikovaná) a jej počet."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    return CLASS_LABELS_SK[i], CLASS_LABELS_SK[j], int(off_diag[i, j])


def build_data() -> dict:
    data = {}
    for model in MODELS:
        y_true, y_pred = combined_test_predictions(model)
        data[model] = {
            "recall": class_recall(y_true, y_pred),
            "confusion": most_common_confusion(y_true, y_pred),
            "n": len(y_true),
        }
    return data


def write_csv(data: dict, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trieda"] + [MODEL_LABELS[m] for m in MODELS])
        for ci, cls in enumerate(CLASS_LABELS_SK):
            writer.writerow([cls] + [f"{data[m]['recall'][ci]:.2f}" for m in MODELS])
        writer.writerow(
            ["najčastejšia zámena (skutočná -> predikovaná, počet z {})".format(data[MODELS[0]]["n"])]
            + [f"{data[m]['confusion'][0]} -> {data[m]['confusion'][1]} ({data[m]['confusion'][2]})" for m in MODELS]
        )
    print(f"  CSV uložené: {out_path}")


def write_latex(data: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Recall po jednotlivých triedach na testovacích dátach "
        r"(súčet z 5 behov, 80 okien na triedu a model)}",
        r"    \label{tab:class-recall}",
        r"    \footnotesize",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        Trieda & SVM & MLP & CNN & LSTM \\",
        r"        \midrule",
    ]
    for ci, cls in enumerate(CLASS_LABELS_SK):
        cells = " & ".join(f"{data[m]['recall'][ci]:.2f}" for m in MODELS)
        lines.append(f"        {cls} & {cells} \\\\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Najčastejšia mimodiagonálna zámena na testovacích dátach "
        r"(skutočná trieda $\rightarrow$ predikovaná, počet okien z 80)}",
        r"    \label{tab:class-confusion}",
        r"    \footnotesize",
        r"    \begin{tabular}{ll}",
        r"        \toprule",
        r"        Model & Najčastejšia zámena \\",
        r"        \midrule",
    ]
    for m in MODELS:
        real, pred, count = data[m]["confusion"]
        lines.append(rf"        {MODEL_LABELS[m]} & {real} $\rightarrow$ {pred} ({count}) \\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX tabuľka uložená: {out_path}")


if __name__ == "__main__":
    RESULTS_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    for m in MODELS:
        real, pred, count = data[m]["confusion"]
        print(f"  {MODEL_LABELS[m]}: recall po triedach = {data[m]['recall']}  "
              f"najčastejšia zámena: {real} -> {pred} ({count}/{data[m]['n']})")
    write_csv(data, RESULTS_COMPARISON_DIR / "class_recall_comparison.csv")
    write_latex(
        data,
        Path(__file__).resolve().parent.parent.parent / "Latex" / "includes" / "tables" / "class_recall_table.tex",
    )
