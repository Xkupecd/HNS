"""
SVM klasifikátor terénu nad ručne počítanými časovými príznakmi (54 príznakov).

Slúži ako baseline pre porovnanie s MLP, LSTM a CNN (zadanie, bod 3).

Beží nad príznakmi z features.py. Kľúčový krok je štandardizácia:
príznaky majú extrémne rôzne rozsahy (napr. WL rádovo stovky, NZC rádovo
desiatky, MAV rádovo desatiny), a SVM s RBF jadrom je na škálu veľmi citlivá -
bez štandardizácie by príznaky s veľkým rozsahom úplne prevládli.

StandardScaler sa fituje VÝHRADNE na trénovacích dátach a potom sa aplikuje
na validačné a testovacie - inak by došlo k úniku informácie z testu do tréningu.
"""
import sys
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

from config import N_SPLITS, RANDOM_SEED
from evaluate import compute_metrics, format_metrics, print_summary, summarize_runs
from features import load_features


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


def train_and_evaluate(model: Pipeline, data: dict) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Natrénuje model na train a vyhodnotí na val a test."""
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return (
        compute_metrics(y_val, y_val_pred),
        compute_metrics(y_test, y_test_pred),
        y_test,
        y_test_pred,
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
    for cfg in configs:
        model = build_svm(**cfg)
        val_metrics, test_metrics, _, _ = train_and_evaluate(model, data)

        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {cfg_str:<32} {format_metrics(val_metrics):<42} {format_metrics(test_metrics)}")

        # Pri zhode vo validačnej úspešnosti rozhoduje priemer precision+recall
        # (samotná accuracy má pri 96 oknách hrubé rozlíšenie ~1 %, takže zhody sú časté)
        score = (val_metrics["accuracy"], val_metrics["precision"] + val_metrics["recall"])
        if best is None or score > best["score"]:
            best = {"config": cfg, "score": score, "val_metrics": val_metrics}

    cfg_str = ", ".join(f"{k}={v}" for k, v in best["config"].items())
    print(f"\nNajlepšia konfigurácia podľa validačnej úspešnosti: {cfg_str}")
    return best["config"]


def run_multiple_splits(config: dict, n_splits: int = N_SPLITS) -> None:
    """
    Spustí najlepšiu konfiguráciu na všetkých dostupných splitoch
    (zadanie: priemer z 5 opakovaní) a vypíše min/max/priemer metrík.
    """
    val_runs, test_runs = [], []
    last_y_test, last_y_pred = None, None

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
        val_metrics, test_metrics, y_test, y_test_pred = train_and_evaluate(model, data)

        val_runs.append(val_metrics)
        test_runs.append(test_metrics)
        last_y_test, last_y_pred = y_test, y_test_pred

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}")

    if not val_runs:
        print("Žiadny split sa nepodarilo načítať.")
        return

    print_summary(f"SVM - validačné dáta (z {len(val_runs)} behov)", summarize_runs(val_runs))
    print_summary(f"SVM - testovacie dáta (z {len(test_runs)} behov)", summarize_runs(test_runs))

    return last_y_test, last_y_pred


if __name__ == "__main__":
    best_config = compare_hyperparameters(split_idx=0)
    result = run_multiple_splits(best_config)

    if result is not None:
        y_test, y_test_pred = result
        from evaluate import plot_confusion_matrix
        plot_confusion_matrix(
            y_test, y_test_pred,
            title="SVM - kontingenčná matica (testovacie dáta)",
            filename="svm_confusion_matrix.png",
        )
