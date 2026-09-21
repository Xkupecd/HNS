"""
SVM klasifikátor terénu nad ručne počítanými časovými príznakmi (54 príznakov).

Slúži ako baseline pre porovnanie s MLP, LSTM a CNN (zadanie, bod 3).

Beží nad príznakmi z features.py. Kľúčový krok je štandardizácia:
príznaky majú extrémne rôzne rozsahy (napr. WL rádovo stovky, NZC rádovo
desiatky, MAV rádovo desatiny), a SVM s RBF jadrom je na škálu veľmi citlivá -
bez štandardizácie by príznaky s veľkým rozsahom úplne prevládli.

StandardScaler sa fituje VÝHRADNE na trénovacích dátach a potom sa aplikuje
na validačné a testovacie - inak by došlo k úniku informácie z testu do tréningu.

Pre finálnu konfiguráciu (5 behov) sa navyše pre KAŽDÝ beh ukladajú surové
predikcie a čas trénovania (results/svm/runs/), a kontingenčné matice sa robia
tri: pre každý beh zvlášť, súčtová (spojenie všetkých 5 behov) a pre najlepší
beh (predtým sa ukladala len z posledného splitu, čo bolo ľubovoľné).
"""
import sys
import time
from pathlib import Path

# Umožní spustiť tento súbor akokoľvek (python svm_model.py z models/,
# python models/svm_model.py z src/, python -m models.svm_model z src/),
# tak, že si vždy sám pridá vlastný priečinok AJ nadradený priečinok src/
# do sys.path - tam ležia zdieľané moduly config.py, features.py (a prípadne
# aj evaluate.py, nech je hocikde z týchto dvoch priečinkov).
_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import N_SPLITS, RANDOM_SEED, RESULTS_DIR
from evaluate import (
    capture_console_to_markdown,
    compute_metrics,
    format_metrics,
    generate_post_run_confusion_matrices,
    plot_confusion_matrix_pair,
    plot_runs_bar_chart,
    print_summary,
    save_run_artifacts,
    summarize_runs,
    write_runs_csv,
)
from features import load_features

RESULTS_SVM_DIR = RESULTS_DIR / "svm"


def build_svm(kernel: str = "rbf", C: float = 10.0, gamma="scale") -> Pipeline:
    """
    SVM v pipeline so štandardizáciou.

    Parametre (zdôvodnenie pre odovzdanie):
    - kernel='rbf': triedy terénu nie sú lineárne separovateľné v priestore
      časových príznakov; RBF zvláda nelineárne hranice
    - C: kompromis medzi šírkou okraja a chybou na tréningu. Vyššie C = menšia
      tolerancia chýb, riziko preučenia; ladí sa na validačných dátach
    - gamma='scale': automaticky 1/(n_features * X.var()), rozumný default
      pre štandardizované dáta
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma, random_state=RANDOM_SEED)),
    ])


def train_and_evaluate(model: Pipeline, data: dict) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Natrénuje model na train a vyhodnotí na val a test. Vracia aj čas trénovania (s)."""
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return (
        compute_metrics(y_val, y_val_pred),
        compute_metrics(y_test, y_test_pred),
        y_val, y_val_pred,
        y_test, y_test_pred,
        train_seconds,
    )


def compare_hyperparameters(split_idx: int = 0) -> dict:
    """
    Porovnanie nastavení SVM na validačných dátach (zadanie: zdôvodnenie parametrov).
    Vráti najlepšiu konfiguráciu podľa validačnej úspešnosti.
    """
    data = load_features(split_idx)

    configs = [
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 10.0},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 100.0, "gamma": "scale"},
        {"kernel": "poly", "C": 10.0},
    ]

    print(f"Porovnanie nastavení SVM (split_{split_idx:02d}):\n")
    print(f"  {'konfigurácia':<32} {'VAL':<42} {'TEST'}")

    best = None
    cfg_labels, val_accs, test_accs = [], [], []
    for cfg in configs:
        model = build_svm(**cfg)
        val_metrics, test_metrics, *_ = train_and_evaluate(model, data)

        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {cfg_str:<32} {format_metrics(val_metrics):<42} {format_metrics(test_metrics)}")

        cfg_labels.append(cfg_str)
        val_accs.append(val_metrics["accuracy"])
        test_accs.append(test_metrics["accuracy"])

        # Pri zhode vo validačnej úspešnosti rozhoduje priemer precision+recall
        # (samotná accuracy má pri 96 oknách hrubé rozlíšenie ~1 %, takže zhody sú časté)
        score = (val_metrics["accuracy"], val_metrics["precision"] + val_metrics["recall"])
        if best is None or score > best["score"]:
            best = {"config": cfg, "score": score, "val_metrics": val_metrics}

    cfg_str = ", ".join(f"{k}={v}" for k, v in best["config"].items())
    print(f"\nNajlepšia konfigurácia podľa validačnej úspešnosti: {cfg_str}")

    plot_runs_bar_chart(
        cfg_labels, val_accs, test_accs,
        title=f"SVM - porovnanie nastavení (split_{split_idx:02d})",
        ylabel="Accuracy [%]",
        filename="hyperparam_comparison.png",
        out_dir=RESULTS_SVM_DIR,
    )

    return best["config"]


def run_multiple_splits(config: dict, n_splits: int = N_SPLITS):
    """
    Spustí najlepšiu konfiguráciu na všetkých dostupných splitoch
    (zadanie: priemer z 5 opakovaní), vypíše min/max/priemer metrík a pre
    KAŽDÝ beh uloží predikcie + kontingenčnú maticu (results/svm/runs/).
    """
    val_runs, test_runs, train_seconds = [], [], []
    y_vals, y_val_preds, y_tests, y_test_preds = [], [], [], []

    print(f"\n{'=' * 70}")
    print(f"Spustenie najlepšej konfigurácie na {n_splits} rozdeleniach")
    print(f"{'=' * 70}")

    for split_idx in range(n_splits):
        try:
            data = load_features(split_idx)
        except FileNotFoundError:
            print(f"  split_{split_idx:02d}: preskočené (okná ešte nie sú vytvorené)")
            continue

        model = build_svm(**config)
        val_metrics, test_metrics, y_val, y_val_pred, y_test, y_test_pred, secs = train_and_evaluate(model, data)

        val_runs.append(val_metrics)
        test_runs.append(test_metrics)
        train_seconds.append(secs)
        y_vals.append(y_val); y_val_preds.append(y_val_pred)
        y_tests.append(y_test); y_test_preds.append(y_test_pred)

        save_run_artifacts(
            RESULTS_SVM_DIR, split_idx, val_metrics, test_metrics,
            y_val, y_val_pred, y_test, y_test_pred, secs, history=None,
        )
        plot_confusion_matrix_pair(
            y_test, y_test_pred,
            title=f"SVM - kontingenčná matica (split_{split_idx:02d}, testovacie dáta)",
            out_dir=RESULTS_SVM_DIR,
            filename=f"runs/split_{split_idx:02d}_confusion.png",
            n_okien_info=f"beh split_{split_idx:02d}, {len(y_test)} okien, čas trénovania {secs:.2f} s",
        )

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}   |   čas={secs:.2f} s")

    if not val_runs:
        print("Žiadny split sa nepodarilo načítať.")
        return None

    write_runs_csv(RESULTS_SVM_DIR, val_runs, test_runs, train_seconds)

    val_summary = summarize_runs(val_runs)
    test_summary = summarize_runs(test_runs)
    print_summary(f"SVM - validačné dáta (z {len(val_runs)} behov)", val_summary)
    print_summary(f"SVM - testovacie dáta (z {len(test_runs)} behov)", test_summary)

    plot_runs_bar_chart(
        [f"split_{i:02d}" for i in range(len(val_runs))],
        [run["accuracy"] for run in val_runs],
        [run["accuracy"] for run in test_runs],
        title="SVM - úspešnosť naprieč 5 behmi (splitmi)",
        ylabel="Accuracy [%]",
        filename="final_splits_accuracy.png",
        out_dir=RESULTS_SVM_DIR,
        val_summary=val_summary,
        test_summary=test_summary,
    )

    # Súčtové matice (5 behov spolu) + matica najlepšieho behu - predtým sa tu
    # ukladal vždy posledný beh (split_04), čo bolo ľubovoľné.
    best_idx = generate_post_run_confusion_matrices(
        "SVM", RESULTS_SVM_DIR, y_vals, y_val_preds, y_tests, y_test_preds, test_runs,
        best_filename="svm_confusion_matrix.png",
        combined_test_filename="svm_confusion_matrix_combined.png",
        combined_val_filename="svm_confusion_matrix_combined_val.png",
    )

    return best_idx


if __name__ == "__main__":
    with capture_console_to_markdown(
        "svm_training_log.md", RESULTS_SVM_DIR, title="SVM - priebeh trénovania a vyhodnotenia"
    ):
        best_config = compare_hyperparameters(split_idx=0)
        run_multiple_splits(best_config)
