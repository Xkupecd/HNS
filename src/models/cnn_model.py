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

from config import N_SPLITS, RANDOM_SEED
from evaluate import compute_metrics, format_metrics, plot_confusion_matrix, print_summary, summarize_runs
from mlp_model import DEVICE, build_optimizer, evaluate_tensors, train_one_epoch
from windowing import load_windowed_dataset


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
    """Natrénuje jednu CNN na danom splite. Vracia rovnakú štruktúru ako mlp_model.train_model."""
    torch.manual_seed(RANDOM_SEED)

    train_loader, (X_val, y_val), (X_test, y_test) = make_dataloaders_raw(split_idx, batch_size)

    model = CNN(input_channels, seq_len, n_classes, cfg).to(DEVICE)
    optimizer = build_optimizer(model, optimizer_name, lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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

    _, val_preds = evaluate_tensors(model, X_val, y_val, criterion)
    _, test_preds = evaluate_tensors(model, X_test, y_test, criterion)

    val_metrics = compute_metrics(y_val.cpu().numpy(), val_preds)
    test_metrics = compute_metrics(y_test.cpu().numpy(), test_preds)

    return model, history, val_metrics, test_metrics, y_test.cpu().numpy(), test_preds


# ---------------------------------------------------------------------------
# Porovnania podľa zadania (body 1 a 2) - rovnaká logika ako v mlp_model.py
# ---------------------------------------------------------------------------
def score_config_across_splits(
    cfg: CNNConfig,
    training_cfg: dict,
    n_splits: int = N_SPLITS,
) -> tuple[float, float, float]:
    """
    Vyhodnotí jednu konfiguráciu na VŠETKÝCH splitoch a vráti
    (priemerná val accuracy, jej smerodajná odchýlka, priemerná val precision+recall).

    PREČO: validačná množina má len 96 okien z 12 meraní, takže jedno zle
    klasifikované meranie posunie accuracy o ~8 %. Výber hyperparametrov podľa
    jedného splitu je preto veľmi hlučný - konfigurácia môže vyzerať najlepšie
    len náhodou. Priemer cez 5 splitov je oveľa stabilnejší základ pre výber
    a zároveň umožňuje preferovať konfigurácie s malým rozptylom.

    Testovacie dáta sa pri výbere NEPOUŽÍVAJÚ - zostávajú nedotknuté.
    """
    val_accs, val_pr = [], []
    for split_idx in range(n_splits):
        _, _, val_metrics, _, _, _ = train_model(cfg, split_idx, **training_cfg)
        val_accs.append(val_metrics["accuracy"])
        val_pr.append(val_metrics["precision"] + val_metrics["recall"])

    return float(np.mean(val_accs)), float(np.std(val_accs)), float(np.mean(val_pr))


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
    for train_cfg in configs:
        mean_acc, std_acc, mean_pr = score_config_across_splits(cfg, train_cfg, n_splits)

        cfg_str = ", ".join(f"{k}={v}" for k, v in train_cfg.items())
        print(f"  {cfg_str:<52} {mean_acc * 100:6.2f} % +- {std_acc * 100:5.2f} %")

        score = (mean_acc, mean_pr)
        if best is None or score > best["score"]:
            best = {"config": train_cfg, "score": score}

    cfg_str = ", ".join(f"{k}={v}" for k, v in best["config"].items())
    print(f"\nNajlepšie parametre trénovania: {cfg_str}")
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
    for arch in architectures:
        mean_acc, std_acc, mean_pr = score_config_across_splits(arch, training_cfg, n_splits)

        print(f"  {arch.name:<34} {mean_acc * 100:6.2f} % +- {std_acc * 100:5.2f} %")

        score = (mean_acc, mean_pr)
        if best is None or score > best["score"]:
            best = {"arch": arch, "score": score}

    print(f"\nNajlepšia architektúra: {best['arch'].name}")
    return best["arch"]


def run_multiple_splits(cfg: CNNConfig, training_cfg: dict, n_splits: int = N_SPLITS):
    val_runs, test_runs = [], []
    last_y_test, last_y_pred = None, None

    print(f"\n{'=' * 70}")
    print(f"Spustenie {cfg.name} na {n_splits} rozdeleniach (parametre: {training_cfg})")
    print(f"{'=' * 70}")

    for split_idx in range(n_splits):
        _, _, val_metrics, test_metrics, y_test, y_test_pred = train_model(cfg, split_idx, **training_cfg)

        val_runs.append(val_metrics)
        test_runs.append(test_metrics)
        last_y_test, last_y_pred = y_test, y_test_pred

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}")

    print_summary(f"CNN - validačné dáta (z {len(val_runs)} behov)", summarize_runs(val_runs))
    print_summary(f"CNN - testovacie dáta (z {len(test_runs)} behov)", summarize_runs(test_runs))

    return last_y_test, last_y_pred


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    first_structure = CNNConfig(conv_channels=(16, 32), kernel_sizes=(9, 5), fc_dim=64, dropout=0.3)
    best_training_cfg = compare_training_hyperparameters(first_structure)

    best_arch = compare_architectures(best_training_cfg)

    y_test, y_test_pred = run_multiple_splits(best_arch, best_training_cfg)

    plot_confusion_matrix(
        y_test, y_test_pred,
        title=f"CNN ({best_arch.name}) - kontingenčná matica (testovacie dáta)",
        filename="cnn_confusion_matrix.png",
    )