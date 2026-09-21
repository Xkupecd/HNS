"""
MLP klasifikátor terénu nad rovnakými 54 časovými príznakmi ako SVM.

Keďže MLP dostáva rovnaký vstup ako SVM, priame porovnanie oboch modelov
(zadanie, bod 3) je fér - rozdiel vo výsledkoch pochádza výlučne z modelu,
nie z rôznych vstupných dát.

Postup podľa zadania:
1. Pre PRVÚ štruktúru porovnať priebehy trénovania pri rôznych parametroch
   trénovania (optimalizátor, learning rate, batch size, počet epoch)
2. Porovnať priebehy pre viacero štruktúr siete (odporúčané 3), vrátane
   ukázania vplyvu Dropout vrstvy
3. Najlepšiu štruktúru spustiť 5x (na 5 nezávislých rozdeleniach) a spočítať
   min/max/priemer metrík
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import time

from config import N_SPLITS, RANDOM_SEED, RESULTS_DIR
from evaluate import (
    capture_console_to_markdown,
    compute_metrics,
    format_metrics,
    generate_post_run_confusion_matrices,
    plot_confusion_matrix_pair,
    plot_runs_bar_chart,
    plot_single_run_training_curve,
    plot_training_curves_comparison,
    print_summary,
    save_run_artifacts,
    summarize_runs,
    write_runs_csv,
)
from features import load_features

RESULTS_MLP_DIR = RESULTS_DIR / "mlp"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Architektúra
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    """Popis jednej architektúry - aby sa dala pomenovať a vypísať v tabuľke porovnania."""
    hidden_dims: tuple = (64, 32)
    dropout: float = 0.3
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            drop_str = f"drop={self.dropout}" if self.dropout > 0 else "no-dropout"
            self.name = f"MLP{self.hidden_dims}-{drop_str}"


class MLP(nn.Module):
    """
    Jednoduchý viacvrstvový perceptrón.

    Vstup: 54 príznakov (9 čas. príznakov x 6 kanálov ACC+GYR).
    Výstup: 6 logitov (jeden na terén) - softmax sa aplikuje až v CrossEntropyLoss.
    """

    def __init__(self, input_dim: int, n_classes: int, cfg: MLPConfig):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Dáta
# ---------------------------------------------------------------------------
def make_dataloaders(split_idx: int, batch_size: int):
    """
    Načíta príznaky, naškáluje ich (fit len na train) a poskladá DataLoadery.

    Vracia aj (X_val, y_val, X_test, y_test) ako tenzory na priame vyhodnotenie
    (netreba loader na jedno predpovedanie celej množiny naraz).
    """
    data = load_features(split_idx)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    to_tensor_x = lambda a: torch.tensor(a, dtype=torch.float32)
    to_tensor_y = lambda a: torch.tensor(a, dtype=torch.long)

    train_ds = TensorDataset(to_tensor_x(X_train), to_tensor_y(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_tensors = (to_tensor_x(X_val).to(DEVICE), to_tensor_y(y_val).to(DEVICE))
    test_tensors = (to_tensor_x(X_test).to(DEVICE), to_tensor_y(y_test).to(DEVICE))

    return train_loader, val_tensors, test_tensors


# ---------------------------------------------------------------------------
# Tréning a vyhodnotenie
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, name: str, lr: float):
    """
    name: 'sgdm' (SGD s momentom) alebo 'adamw'.

    Zdôvodnenie voľby (pre odovzdanie): SGDM je klasika s dobrou generalizáciou,
    ale citlivejšia na learning rate. AdamW obsahuje adaptívny learning rate
    a oddelený weight decay - zvyčajne konverguje rýchlejšie a je robustnejší
    na počiatočnú voľbu lr, čo je výhoda pri tak malom datasete ako máme tu.
    """
    if name == "sgdm":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Neznámy optimalizátor: {name}")


def train_one_epoch(model, loader, optimizer, criterion) -> tuple[float, float]:
    """Jedna tréningová epocha. Vráti priemernú loss a accuracy na trénovacích dátach."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * X_batch.size(0)
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_tensors(model, X, y, criterion) -> tuple[float, np.ndarray]:
    """Vyhodnotí model na už pripravených tenzoroch (val/test). Vráti loss a predikcie (numpy)."""
    model.eval()
    outputs = model(X)
    loss = criterion(outputs, y).item()
    preds = outputs.argmax(dim=1).cpu().numpy()
    return loss, preds


def train_model(
    cfg: MLPConfig,
    split_idx: int,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    batch_size: int = 16,
    epochs: int = 20,
    input_dim: int = 54,
    n_classes: int = 6,
    verbose: bool = False,
):
    """
    Natrénuje jednu MLP na danom splite. Vráti
    (model, history, val_metrics, test_metrics, y_val, val_pred, y_test, test_pred, train_seconds).

    history obsahuje priebeh train/val loss a accuracy po epochách - to je presne
    to, čo zadanie žiada porovnať v bode 1 (priebehy trénovania).
    """
    torch.manual_seed(RANDOM_SEED)

    train_loader, (X_val, y_val), (X_test, y_test) = make_dataloaders(split_idx, batch_size)

    model = MLP(input_dim, n_classes, cfg).to(DEVICE)
    optimizer = build_optimizer(model, optimizer_name, lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    t0 = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds = evaluate_tensors(model, X_val, y_val, criterion)
        val_acc = (val_preds == y_val.cpu().numpy()).mean()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"    epoch {epoch + 1:>2}/{epochs}  "
                  f"train_loss={train_loss:.3f} train_acc={train_acc * 100:5.1f}%  "
                  f"val_loss={val_loss:.3f} val_acc={val_acc * 100:5.1f}%")
    train_seconds = time.perf_counter() - t0

    _, val_preds = evaluate_tensors(model, X_val, y_val, criterion)
    _, test_preds = evaluate_tensors(model, X_test, y_test, criterion)

    y_val_np, y_test_np = y_val.cpu().numpy(), y_test.cpu().numpy()
    val_metrics = compute_metrics(y_val_np, val_preds)
    test_metrics = compute_metrics(y_test_np, test_preds)

    return model, history, val_metrics, test_metrics, y_val_np, val_preds, y_test_np, test_preds, train_seconds


# ---------------------------------------------------------------------------
# Porovnania podľa zadania (body 1 a 2)
# ---------------------------------------------------------------------------
def compare_training_hyperparameters(cfg: MLPConfig, split_idx: int = 0) -> dict:
    """
    Bod 1 zo zadania: pre PRVÚ štruktúru porovnať priebeh trénovania pri rôznych
    optimalizátoroch, learning rate, batch size a počte epoch.

    Vráti najlepšiu kombináciu hyperparametrov podľa validačnej accuracy.
    """
    configs = [
        {"optimizer_name": "sgdm", "lr": 0.01, "batch_size": 16, "epochs": 20},
        {"optimizer_name": "sgdm", "lr": 0.001, "batch_size": 16, "epochs": 20},
        {"optimizer_name": "adamw", "lr": 0.01, "batch_size": 16, "epochs": 20},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 20},
        {"optimizer_name": "adamw", "lr": 0.0001, "batch_size": 16, "epochs": 20},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 32, "epochs": 20},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 10},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 15},
    ]

    print(f"Porovnanie parametrov trénovania pre {cfg.name} (split_{split_idx:02d}):\n")
    print(f"  {'konfigurácia':<45} {'VAL':<42} {'TEST'}")

    best = None
    histories = {}
    for train_cfg in configs:
        _, history, val_metrics, test_metrics, *_ = train_model(cfg, split_idx, **train_cfg)

        cfg_str = ", ".join(f"{k}={v}" for k, v in train_cfg.items())
        print(f"  {cfg_str:<45} {format_metrics(val_metrics):<42} {format_metrics(test_metrics)}")
        histories[cfg_str] = history

        score = (val_metrics["accuracy"], val_metrics["precision"] + val_metrics["recall"])
        if best is None or score > best["score"]:
            best = {"config": train_cfg, "score": score}

    cfg_str = ", ".join(f"{k}={v}" for k, v in best["config"].items())
    print(f"\nNajlepšie parametre trénovania: {cfg_str}")

    plot_training_curves_comparison(
        histories,
        title=f"MLP ({cfg.name}) - porovnanie parametrov trénovania (split_{split_idx:02d})",
        filename="hyperparam_training_curves.png",
        out_dir=RESULTS_MLP_DIR,
    )

    return best["config"]


def compare_architectures(training_cfg: dict, split_idx: int = 0) -> MLPConfig:
    """
    Bod 2 zo zadania: porovnanie viacerých štruktúr siete (odporúčané 3),
    vrátane ukázania vplyvu Dropout vrstvy - preto je posledná konfigurácia
    zámerne rovnaká architektúra ako druhá, len bez Dropoutu.
    """
    architectures = [
        MLPConfig(hidden_dims=(32,), dropout=0.3),
        MLPConfig(hidden_dims=(64, 32), dropout=0.3),
        MLPConfig(hidden_dims=(64, 32), dropout=0.0),  # rovnaká štruktúra, BEZ dropoutu -> efekt dropoutu
        MLPConfig(hidden_dims=(128, 64, 32), dropout=0.4),
    ]

    print(f"\nPorovnanie architektúr (split_{split_idx:02d}), parametre trénovania: {training_cfg}\n")
    print(f"  {'architektúra':<28} {'VAL':<42} {'TEST'}")

    best = None
    histories = {}
    for arch in architectures:
        _, history, val_metrics, test_metrics, *_ = train_model(arch, split_idx, **training_cfg)

        print(f"  {arch.name:<28} {format_metrics(val_metrics):<42} {format_metrics(test_metrics)}")
        histories[arch.name] = history

        score = (val_metrics["accuracy"], val_metrics["precision"] + val_metrics["recall"])
        if best is None or score > best["score"]:
            best = {"arch": arch, "score": score}

    print(f"\nNajlepšia architektúra: {best['arch'].name}")

    plot_training_curves_comparison(
        histories,
        title=f"MLP - porovnanie architektúr (split_{split_idx:02d})",
        filename="architecture_training_curves.png",
        out_dir=RESULTS_MLP_DIR,
    )

    return best["arch"]


def run_multiple_splits(cfg: MLPConfig, training_cfg: dict, n_splits: int = N_SPLITS):
    """
    Zadanie bod 6: najlepšiu štruktúru spustiť 5x a spočítať min/max/priemer metrík.

    Navyše pre KAŽDÝ z 5 behov uloží: surové predikcie + históriu trénovania
    (results/mlp/runs/), graf priebehu daného behu a jeho kontingenčnú maticu.
    Po skončení vytvorí súčtové kontingenčné matice (5 behov spolu) a maticu
    najlepšieho behu.
    """
    val_runs, test_runs, train_seconds = [], [], []
    histories = {}
    y_vals, y_val_preds, y_tests, y_test_preds = [], [], [], []

    print(f"\n{'=' * 70}")
    print(f"Spustenie {cfg.name} na {n_splits} rozdeleniach (parametre: {training_cfg})")
    print(f"{'=' * 70}")

    for split_idx in range(n_splits):
        _, history, val_metrics, test_metrics, y_val, val_pred, y_test, test_pred, secs = \
            train_model(cfg, split_idx, **training_cfg)

        val_runs.append(val_metrics)
        test_runs.append(test_metrics)
        train_seconds.append(secs)
        histories[f"split_{split_idx:02d}"] = history
        y_vals.append(y_val); y_val_preds.append(val_pred)
        y_tests.append(y_test); y_test_preds.append(test_pred)

        save_run_artifacts(
            RESULTS_MLP_DIR, split_idx, val_metrics, test_metrics,
            y_val, val_pred, y_test, test_pred, secs, history=history,
        )
        plot_single_run_training_curve(
            history,
            title=f"MLP ({cfg.name}) - priebeh trénovania (split_{split_idx:02d})",
            filename=f"runs/split_{split_idx:02d}_training_curve.png",
            out_dir=RESULTS_MLP_DIR,
        )
        plot_confusion_matrix_pair(
            y_test, test_pred,
            title=f"MLP ({cfg.name}) - kontingenčná matica (split_{split_idx:02d}, testovacie dáta)",
            out_dir=RESULTS_MLP_DIR,
            filename=f"runs/split_{split_idx:02d}_confusion.png",
            n_okien_info=f"beh split_{split_idx:02d}, {len(y_test)} okien, čas trénovania {secs:.2f} s",
        )

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}   |   čas={secs:.2f} s")

    write_runs_csv(RESULTS_MLP_DIR, val_runs, test_runs, train_seconds)

    val_summary = summarize_runs(val_runs)
    test_summary = summarize_runs(test_runs)
    print_summary(f"MLP - validačné dáta (z {len(val_runs)} behov)", val_summary)
    print_summary(f"MLP - testovacie dáta (z {len(test_runs)} behov)", test_summary)

    plot_training_curves_comparison(
        histories,
        title=f"MLP ({cfg.name}) - priebeh trénovania naprieč {n_splits} behmi (splitmi)",
        filename="final_splits_training_curves.png",
        out_dir=RESULTS_MLP_DIR,
    )
    plot_runs_bar_chart(
        list(histories.keys()),
        [run["accuracy"] for run in val_runs],
        [run["accuracy"] for run in test_runs],
        title=f"MLP ({cfg.name}) - úspešnosť naprieč {n_splits} behmi (splitmi)",
        ylabel="Accuracy [%]",
        filename="final_splits_accuracy.png",
        out_dir=RESULTS_MLP_DIR,
        val_summary=val_summary,
        test_summary=test_summary,
    )

    generate_post_run_confusion_matrices(
        "MLP", RESULTS_MLP_DIR, y_vals, y_val_preds, y_tests, y_test_preds, test_runs,
        best_filename="mlp_confusion_matrix.png",
        combined_test_filename="mlp_confusion_matrix_combined.png",
        combined_val_filename="mlp_confusion_matrix_combined_val.png",
    )


if __name__ == "__main__":
    with capture_console_to_markdown(
        "mlp_training_log.md", RESULTS_MLP_DIR, title="MLP - priebeh trénovania a vyhodnotenia"
    ):
        print(f"Device: {DEVICE}\n")

        first_structure = MLPConfig(hidden_dims=(64, 32), dropout=0.3)
        best_training_cfg = compare_training_hyperparameters(first_structure, split_idx=0)

        best_arch = compare_architectures(best_training_cfg, split_idx=0)

        run_multiple_splits(best_arch, best_training_cfg)
