"""
CNN klasifikátor terénu nad surovým signálom (400 vzoriek x 6 kanálov: ACC+GYR).

Na rozdiel od SVM/MLP tu NEPOČÍTAME žiadne ručné príznaky - konvolučné vrstvy
(1D konvolúcia, ako zadanie vyžaduje) si extrakciu čŕt riešia samé, priamo
zo signálu (zadanie, Obr. 3 - "automatická extrakcia príznakov").

Vstup pre Conv1d musí mať tvar (batch, kanály, čas) - teda opak toho, ako
sú okná uložené (n, čas, kanály) - preto transpozícia v make_dataloaders_raw.

Tréningová slučka (train_one_epoch, evaluate_tensors, build_optimizer, DEVICE)
sa nededikuje znova - je prevzatá z mlp_model.py, aby mali všetky NN modely
identickú tréningovú logiku a bolo porovnanie SVM/MLP/LSTM/CNN naozaj fér.
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
for _p in (_this_dir, _this_dir.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
from mlp_model import DEVICE, build_optimizer, evaluate_tensors, train_one_epoch
from windowing import load_windowed_dataset

RESULTS_CNN_DIR = RESULTS_DIR / "cnn"


# ---------------------------------------------------------------------------
# Architektúra
# ---------------------------------------------------------------------------
@dataclass
class CNNConfig:
    """
    conv_channels / kernel_sizes: jeden konvolučný blok na dvojicu (počet filtrov, veľkosť jadra).
    Za každým blokom nasleduje BatchNorm1d (ak use_batchnorm) -> ReLU -> MaxPool1d(2).

    global_pool=True: po poslednom konv. bloku nasleduje Global Average Pooling
    (priemer cez celú časovú os) namiesto Flatten - vstup do FC vrstvy je potom
    len (počet filtrov) namiesto (počet filtrov x zvyšná dĺžka), čo je pri 480
    trénovacích vzorkách oveľa menej náchylné na preučenie.
    """
    conv_channels: tuple = (16, 32)
    kernel_sizes: tuple = (9, 5)
    fc_dim: int = 64
    dropout: float = 0.3
    use_batchnorm: bool = True
    global_pool: bool = True
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            drop_str = f"drop={self.dropout}" if self.dropout > 0 else "no-dropout"
            pool_str = "gap" if self.global_pool else "flatten"
            bn_str = "bn" if self.use_batchnorm else "no-bn"
            self.name = f"CNN{self.conv_channels}-{pool_str}-{bn_str}-{drop_str}"


class CNN(nn.Module):
    """
    Niekoľko blokov [Conv1d -> (BatchNorm1d) -> ReLU -> MaxPool1d(2)],
    potom buď Global Average Pooling, alebo Flatten, a napokon FC -> Dropout -> FC.

    Veľkosť vstupu do prvej FC vrstvy sa nepočíta ručne - zisťuje sa jedným
    "naprázdno" prechodom cez konvolučnú (+ pooling) časť (dummy tenzor),
    aby fungovala ľubovoľná kombinácia počtu blokov, jadier a typu poolingu
    bez ručného prepočítavania rozmerov.
    """

    def __init__(self, input_channels: int, seq_len: int, n_classes: int, cfg: CNNConfig):
        super().__init__()

        conv_layers = []
        in_ch = input_channels
        for out_ch, kernel_size in zip(cfg.conv_channels, cfg.kernel_sizes):
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if cfg.use_batchnorm:
                conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1) if cfg.global_pool else None

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            conv_out = self.conv(dummy)
            if self.global_pool is not None:
                conv_out = self.global_pool(conv_out)
            flat_dim = conv_out.numel()

        fc_layers = [nn.Linear(flat_dim, cfg.fc_dim), nn.ReLU()]
        if cfg.dropout > 0:
            fc_layers.append(nn.Dropout(cfg.dropout))
        fc_layers.append(nn.Linear(cfg.fc_dim, n_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        if self.global_pool is not None:
            x = self.global_pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dáta
# ---------------------------------------------------------------------------
def normalize_channels(X_train, X_val, X_test):
    """
    Štandardizácia po kanáloch (nie po celom okne naraz) - ACC a GYR majú úplne
    inú škálu, takže priemer/smerodajná odchýlka sa počíta zvlášť pre každý
    zo 6 kanálov, cez všetky okná a všetky časové kroky trénovacej množiny.
    """
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8

    normalize = lambda X: (X - mean) / std
    return normalize(X_train), normalize(X_val), normalize(X_test)


def make_dataloaders_raw(split_idx: int, batch_size: int):
    """
    Načíta surové okná, znormalizuje ich po kanáloch, prevedie do tvaru
    (batch, kanály, čas) pre Conv1d a poskladá DataLoadery.
    """
    data = load_windowed_dataset(split_idx)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    X_train, X_val, X_test = normalize_channels(X_train, X_val, X_test)

    # (n, čas, kanály) -> (n, kanály, čas)
    to_conv_shape = lambda X: np.transpose(X, (0, 2, 1))
    to_tensor_x = lambda a: torch.tensor(to_conv_shape(a), dtype=torch.float32)
    to_tensor_y = lambda a: torch.tensor(a, dtype=torch.long)

    train_ds = TensorDataset(to_tensor_x(X_train), to_tensor_y(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_tensors = (to_tensor_x(X_val).to(DEVICE), to_tensor_y(y_val).to(DEVICE))
    test_tensors = (to_tensor_x(X_test).to(DEVICE), to_tensor_y(y_test).to(DEVICE))

    return train_loader, val_tensors, test_tensors


# ---------------------------------------------------------------------------
# Tréning
# ---------------------------------------------------------------------------
def train_model(
    cfg: CNNConfig,
    split_idx: int,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    batch_size: int = 16,
    epochs: int = 20,
    input_channels: int = 6,
    seq_len: int = 400,
    n_classes: int = 6,
    verbose: bool = False,
):
    """
    Natrénuje jednu CNN na danom splite. Vráti
    (model, history, val_metrics, test_metrics, y_val, val_pred, y_test, test_pred, train_seconds).
    """
    torch.manual_seed(RANDOM_SEED)

    train_loader, (X_val, y_val), (X_test, y_test) = make_dataloaders_raw(split_idx, batch_size)

    model = CNN(input_channels, seq_len, n_classes, cfg).to(DEVICE)
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
# Porovnania podľa zadania (body 1 a 2) - rovnaká logika ako v mlp_model.py
# ---------------------------------------------------------------------------
def score_config_across_splits(
    cfg: CNNConfig,
    training_cfg: dict,
    n_splits: int = N_SPLITS,
) -> tuple[float, float, float, dict]:
    """
    Vyhodnotí jednu konfiguráciu na VŠETKÝCH splitoch a vráti
    (priemerná val accuracy, jej smerodajná odchýlka, priemerná val precision+recall).

    PREČO: validačná množina má len 96 okien z 12 meraní, takže jedno zle
    klasifikované meranie posunie accuracy o ~8 %. Výber hyperparametrov podľa
    jedného splitu je preto veľmi hlučný - konfigurácia môže vyzerať najlepšie
    len náhodou. Priemer cez 5 splitov je oveľa stabilnejší základ pre výber
    a zároveň umožňuje preferovať konfigurácie s malým rozptylom.

    Testovacie dáta sa pri výbere NEPOUŽÍVAJÚ - zostávajú nedotknuté.

    Popri tom vracia aj history zo split_00 (reprezentatívny beh, bez ďalšieho
    trénovania navyše) - použije sa na porovnávací graf priebehu trénovania.
    """
    val_accs, val_pr = [], []
    history_split0 = None
    for split_idx in range(n_splits):
        _, history, val_metrics, *_ = train_model(cfg, split_idx, **training_cfg)
        val_accs.append(val_metrics["accuracy"])
        val_pr.append(val_metrics["precision"] + val_metrics["recall"])
        if split_idx == 0:
            history_split0 = history

    return float(np.mean(val_accs)), float(np.std(val_accs)), float(np.mean(val_pr)), history_split0


def compare_training_hyperparameters(cfg: CNNConfig, n_splits: int = N_SPLITS) -> dict:
    """
    Bod 1 zo zadania, ale výber sa robí podľa PRIEMERU validačnej úspešnosti
    cez všetky splity (nie podľa jedného), aby nebol výsledok len náhoda.
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
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 30},
    ]

    print(f"Porovnanie parametrov trénovania pre {cfg.name} "
          f"(priemer validačnej úspešnosti z {n_splits} splitov):\n")
    print(f"  {'konfigurácia':<52} {'VAL acc (priemer +- std)'}")

    best = None
    histories = {}
    for train_cfg in configs:
        mean_acc, std_acc, mean_pr, history = score_config_across_splits(cfg, train_cfg, n_splits)

        cfg_str = ", ".join(f"{k}={v}" for k, v in train_cfg.items())
        print(f"  {cfg_str:<52} {mean_acc * 100:6.2f} % +- {std_acc * 100:5.2f} %")
        histories[cfg_str] = history

        score = (mean_acc, mean_pr)
        if best is None or score > best["score"]:
            best = {"config": train_cfg, "score": score}

    cfg_str = ", ".join(f"{k}={v}" for k, v in best["config"].items())
    print(f"\nNajlepšie parametre trénovania: {cfg_str}")

    plot_training_curves_comparison(
        histories,
        title=f"CNN ({cfg.name}) - porovnanie parametrov trénovania (split_00)",
        filename="hyperparam_training_curves.png",
        out_dir=RESULTS_CNN_DIR,
    )

    return best["config"]


def compare_architectures(training_cfg: dict, n_splits: int = N_SPLITS) -> CNNConfig:
    """
    3 odporúčané štruktúry + ablácie (Dropout, BatchNorm, pooling stratégia),
    aby bolo vidno vplyv každej z týchto volieb zvlášť.

    Výber opäť podľa priemeru validačnej úspešnosti cez všetky splity.
    """
    architectures = [
        CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.3),
        CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.0),        # bez dropoutu
        CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.3, use_batchnorm=False),  # bez BatchNorm
        CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.3, global_pool=False),    # Flatten namiesto GAP
        CNNConfig(conv_channels=(32, 64, 64), kernel_sizes=(15, 9, 5), fc_dim=128, dropout=0.3),
    ]

    print(f"\nPorovnanie architektúr (priemer z {n_splits} splitov), "
          f"parametre trénovania: {training_cfg}\n")
    print(f"  {'architektúra':<34} {'VAL acc (priemer +- std)'}")

    best = None
    histories = {}
    for arch in architectures:
        mean_acc, std_acc, mean_pr, history = score_config_across_splits(arch, training_cfg, n_splits)

        print(f"  {arch.name:<34} {mean_acc * 100:6.2f} % +- {std_acc * 100:5.2f} %")
        histories[arch.name] = history

        score = (mean_acc, mean_pr)
        if best is None or score > best["score"]:
            best = {"arch": arch, "score": score}

    print(f"\nNajlepšia architektúra: {best['arch'].name}")

    plot_training_curves_comparison(
        histories,
        title="CNN - porovnanie architektúr (split_00)",
        filename="architecture_training_curves.png",
        out_dir=RESULTS_CNN_DIR,
    )

    return best["arch"]


def run_multiple_splits(cfg: CNNConfig, training_cfg: dict, n_splits: int = N_SPLITS):
    """
    Zadanie bod 6: najlepšiu štruktúru spustiť 5x a spočítať min/max/priemer metrík.

    Navyše pre KAŽDÝ z 5 behov uloží: surové predikcie + históriu trénovania
    (results/cnn/runs/), graf priebehu daného behu a jeho kontingenčnú maticu.
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
            RESULTS_CNN_DIR, split_idx, val_metrics, test_metrics,
            y_val, val_pred, y_test, test_pred, secs, history=history,
        )
        plot_single_run_training_curve(
            history,
            title=f"CNN ({cfg.name}) - priebeh trénovania (split_{split_idx:02d})",
            filename=f"runs/split_{split_idx:02d}_training_curve.png",
            out_dir=RESULTS_CNN_DIR,
        )
        plot_confusion_matrix_pair(
            y_test, test_pred,
            title=f"CNN ({cfg.name}) - kontingenčná matica (split_{split_idx:02d}, testovacie dáta)",
            out_dir=RESULTS_CNN_DIR,
            filename=f"runs/split_{split_idx:02d}_confusion.png",
            n_okien_info=f"beh split_{split_idx:02d}, {len(y_test)} okien, čas trénovania {secs:.2f} s",
        )

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}   |   čas={secs:.2f} s")

    write_runs_csv(RESULTS_CNN_DIR, val_runs, test_runs, train_seconds)

    val_summary = summarize_runs(val_runs)
    test_summary = summarize_runs(test_runs)
    print_summary(f"CNN - validačné dáta (z {len(val_runs)} behov)", val_summary)
    print_summary(f"CNN - testovacie dáta (z {len(test_runs)} behov)", test_summary)

    plot_training_curves_comparison(
        histories,
        title=f"CNN ({cfg.name}) - priebeh trénovania naprieč {n_splits} behmi (splitmi)",
        filename="final_splits_training_curves.png",
        out_dir=RESULTS_CNN_DIR,
    )
    plot_runs_bar_chart(
        list(histories.keys()),
        [run["accuracy"] for run in val_runs],
        [run["accuracy"] for run in test_runs],
        title=f"CNN ({cfg.name}) - úspešnosť naprieč {n_splits} behmi (splitmi)",
        ylabel="Accuracy [%]",
        filename="final_splits_accuracy.png",
        out_dir=RESULTS_CNN_DIR,
        val_summary=val_summary,
        test_summary=test_summary,
    )

    generate_post_run_confusion_matrices(
        "CNN", RESULTS_CNN_DIR, y_vals, y_val_preds, y_tests, y_test_preds, test_runs,
        best_filename="cnn_confusion_matrix.png",
        combined_test_filename="cnn_confusion_matrix_combined.png",
        combined_val_filename="cnn_confusion_matrix_combined_val.png",
    )


if __name__ == "__main__":
    with capture_console_to_markdown(
        "cnn_training_log.md", RESULTS_CNN_DIR, title="CNN - priebeh trénovania a vyhodnotenia"
    ):
        print(f"Device: {DEVICE}\n")

        first_structure = CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.3)
        best_training_cfg = compare_training_hyperparameters(first_structure)

        best_arch = compare_architectures(best_training_cfg)

        run_multiple_splits(best_arch, best_training_cfg)