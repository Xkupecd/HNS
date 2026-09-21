"""
LSTM klasifikátor terénu nad surovým signálom (400 vzoriek x 6 kanálov: ACC+GYR).

Rovnako ako CNN, aj LSTM si extrakciu čŕt zo signálu robí sama (žiadne ručné
príznaky) - zadanie vyžaduje porovnať SVM/MLP (ručné príznaky) proti LSTM/CNN
(automatická extrakcia).

Prečo naivné "400 časových krokov -> posledný skrytý stav" nefunguje
--------------------------------------------------------------------
Prvá verzia tohto modelu posielala do LSTM celé okno ako 400 časových krokov
po 6 kanáloch a klasifikovala z posledného skrytého stavu h_n. Dávalo to len
~58 % na testovacích dátach (SVM/MLP/CNN pritom ~95 %). Dôvody:

1. 400 krokov je pre LSTM veľmi dlhá sekvencia - gradient sa cez toľko krokov
   rozptýli a sieť sa pri 480 trénovacích oknách nestihne nič naučiť.
2. Terén sa v signáli prejavuje ako *textúra vibrácií* (energia a frekvenčný
   obsah), nie ako jedna udalosť na konci okna. Posledný skrytý stav h_n je
   však ovplyvnený hlavne poslednými vzorkami - zvyšok okna sa zahodí.
3. Okno je stacionárne: absolútna fáza signálu (kde presne okno začína) nenesie
   žiadnu informáciu o teréne, takže sieť sa učí aj náhodný fázový posun.

Tri zmeny, ktoré model dostali z ~58 % na ~96 %
-----------------------------------------------
1. RÁMCOVANIE VSTUPU (frame_size): okno (400, 6) sa preskupí na (80, 30) -
   jeden časový krok LSTM = 5 po sebe idúcich vzoriek zo všetkých 6 kanálov.
   Je to čistý reshape, žiadne ručne počítané príznaky - len sa skráti sekvencia
   (400 -> 80 krokov) a LSTM v jednom kroku vidí krátky úsek priebehu.
2. POOLING CEZ ČAS (pool="meanmax"): namiesto posledného skrytého stavu sa
   klasifikuje z priemeru a maxima výstupov LSTM cez všetky časové kroky.
   Priemer zachytí celkovú textúru okna, maximum výrazné nárazy (kamienky,
   dlažba). Pôvodné "last" ostáva ako voliteľná možnosť do porovnania.
3. AUGMENTÁCIA (iba trénovacie dáta): cyklický časový posun (fáza okna je
   nepodstatná), náhodné preškálovanie amplitúdy po kanáloch (+-10 %, iná
   priľnavosť/uchytenie snímača) a malý gaussovský šum. Dataset má len 480
   trénovacích okien, takže práve toto je najväčší jednotlivý prínos.

Tréningová slučka (build_optimizer, evaluate_tensors, DEVICE) je prevzatá
z mlp_model.py, normalizácia po kanáloch (normalize_channels) z cnn_model.py,
aby mali všetky NN modely identickú tréningovú logiku a bolo porovnanie
SVM/MLP/LSTM/CNN naozaj fér. Navyše tu pribudol kosínusový rozvrh learning rate
(pri dlhšom tréningu s augmentáciou stabilizuje záver tréningu).
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

from cnn_model import normalize_channels
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
from mlp_model import DEVICE, build_optimizer, evaluate_tensors
from windowing import load_windowed_dataset

RESULTS_LSTM_DIR = RESULTS_DIR / "lstm"

N_CHANNELS = 6  # ACC_X/Y/Z + GYR_X/Y/Z


# ---------------------------------------------------------------------------
# Architektúra
# ---------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    """
    frame_size: koľko po sebe idúcich vzoriek signálu tvorí JEDEN časový krok
        LSTM. frame_size=1 je pôvodná (slabá) verzia so 400 krokmi, frame_size=5
        dáva 80 krokov po 30 hodnotách. Ide len o reshape vstupu, nie o príznaky.
    pool: z čoho sa klasifikuje.
        "last"    - posledný skrytý stav h_n (klasický, ale pre textúru signálu slabý)
        "mean"    - priemer výstupov LSTM cez čas
        "meanmax" - priemer + maximum cez čas (spojené do jedného vektora)
    hidden_size/num_layers/bidirectional: veľkosť a počet vrstiev LSTM,
        obojsmerná verzia prechádza sekvenciu v oboch smeroch.
    """
    hidden_size: int = 128
    num_layers: int = 1
    fc_dim: int = 64
    dropout: float = 0.3
    bidirectional: bool = True
    frame_size: int = 5
    pool: str = "meanmax"
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            drop_str = f"drop={self.dropout}" if self.dropout > 0 else "no-dropout"
            dir_str = "bidir" if self.bidirectional else "unidir"
            self.name = (f"LSTM(h={self.hidden_size},L={self.num_layers})-{dir_str}"
                         f"-frame={self.frame_size}-{self.pool}-{drop_str}")


class LSTMClassifier(nn.Module):
    """LSTM -> pooling cez čas -> FC -> Dropout -> FC (logity pre 6 tried)."""

    def __init__(self, n_classes: int, cfg: LSTMConfig, n_channels: int = N_CHANNELS):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=n_channels * cfg.frame_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        if cfg.pool == "meanmax":
            out_dim *= 2

        fc_layers = [nn.Linear(out_dim, cfg.fc_dim), nn.ReLU()]
        if cfg.dropout > 0:
            fc_layers.append(nn.Dropout(cfg.dropout))
        fc_layers.append(nn.Linear(cfg.fc_dim, n_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x: (batch, čas, kanály) -> rámcovanie -> (batch, čas/frame, frame*kanály)
        x = frame_windows(x, self.cfg.frame_size)
        out, (h_n, _) = self.lstm(x)

        if self.cfg.pool == "last":
            z = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.cfg.bidirectional else h_n[-1]
        elif self.cfg.pool == "mean":
            z = out.mean(dim=1)
        elif self.cfg.pool == "meanmax":
            z = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        else:
            raise ValueError(f"Neznámy pooling: {self.cfg.pool}")

        return self.fc(z)


def frame_windows(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    """
    (batch, čas, kanály) -> (batch, čas/frame_size, frame_size*kanály).

    Nie je to extrakcia príznakov, len preskupenie vzoriek: LSTM dostane kratšiu
    sekvenciu, v ktorej jeden krok obsahuje krátky úsek priebehu všetkých kanálov.
    """
    if frame_size == 1:
        return x
    batch, n_time, n_ch = x.shape
    n_frames = n_time // frame_size
    return x[:, :n_frames * frame_size].reshape(batch, n_frames, frame_size * n_ch)


# ---------------------------------------------------------------------------
# Augmentácia (len trénovacie dáta)
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """
    time_shift: náhodný CYKLICKÝ posun okna v čase. Okno je stacionárny úsek
        vibrácií, takže posunuté okno je stále to isté meranie toho istého terénu -
        model sa tak prestane učiť konkrétnu fázu signálu.
    amp_scale: náhodné preškálovanie amplitúdy zvlášť pre každý kanál (napr. 0.1
        = +-10 %) - simuluje mierne odlišné uchytenie/citlivosť snímača.
    noise_std: gaussovský šum (dáta sú už znormalizované, takže 0.05 = 5 % sigma).
    """
    time_shift: bool = True
    amp_scale: float = 0.1
    noise_std: float = 0.05
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            parts = []
            if self.time_shift:
                parts.append("posun")
            if self.amp_scale > 0:
                parts.append(f"amp={self.amp_scale}")
            if self.noise_std > 0:
                parts.append(f"šum={self.noise_std}")
            self.name = "+".join(parts) if parts else "bez augmentácie"


def augment_batch(x: torch.Tensor, aug: AugmentConfig | None) -> torch.Tensor:
    """Aplikuje augmentáciu na jeden batch surových okien (batch, čas, kanály)."""
    if aug is None:
        return x

    batch, n_time, n_ch = x.shape

    if aug.time_shift:
        # nezávislý cyklický posun pre každé okno v batchi
        offsets = torch.randint(0, n_time, (batch, 1), device=x.device)
        idx = (torch.arange(n_time, device=x.device).unsqueeze(0) + offsets) % n_time
        x = torch.gather(x, 1, idx.unsqueeze(-1).expand(batch, n_time, n_ch))

    if aug.amp_scale > 0:
        x = x * (1.0 + aug.amp_scale * torch.randn(batch, 1, n_ch, device=x.device))

    if aug.noise_std > 0:
        x = x + aug.noise_std * torch.randn_like(x)

    return x


# ---------------------------------------------------------------------------
# Dáta
# ---------------------------------------------------------------------------
def make_dataloaders_lstm(split_idx: int, batch_size: int):
    """
    Načíta surové okná, znormalizuje ich po kanáloch (rovnako ako CNN) a poskladá
    DataLoadery. Rámcovanie sa robí až v modeli (forward), aby sa augmentácia
    dala robiť nad pôvodným časovým priebehom.
    """
    data = load_windowed_dataset(split_idx)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    X_train, X_val, X_test = normalize_channels(X_train, X_val, X_test)

    to_tensor_x = lambda a: torch.tensor(a, dtype=torch.float32)
    to_tensor_y = lambda a: torch.tensor(a, dtype=torch.long)

    train_ds = TensorDataset(to_tensor_x(X_train), to_tensor_y(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_tensors = (to_tensor_x(X_val).to(DEVICE), to_tensor_y(y_val).to(DEVICE))
    test_tensors = (to_tensor_x(X_test).to(DEVICE), to_tensor_y(y_test).to(DEVICE))

    return train_loader, val_tensors, test_tensors


# ---------------------------------------------------------------------------
# Tréning
# ---------------------------------------------------------------------------
def train_one_epoch_aug(model, loader, optimizer, criterion, aug: AugmentConfig | None):
    """
    Ako mlp_model.train_one_epoch, len s augmentáciou batchu pred prechodom sieťou.
    (V mlp_model sa nemení, aby MLP/CNN bežali presne tak ako doteraz.)
    """
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        X_batch = augment_batch(X_batch, aug)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * X_batch.size(0)
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)

    return loss_sum / total, correct / total


def train_model(
    cfg: LSTMConfig,
    split_idx: int,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    batch_size: int = 16,
    epochs: int = 120,
    cosine_schedule: bool = True,
    aug: AugmentConfig | None = None,
    n_classes: int = 6,
    verbose: bool = False,
):
    """
    Natrénuje jednu LSTM na danom splite. Vráti
    (model, history, val_metrics, test_metrics, y_val, val_pred, y_test, test_pred, train_seconds).
    """
    torch.manual_seed(RANDOM_SEED)

    train_loader, (X_val, y_val), (X_test, y_test) = make_dataloaders_lstm(split_idx, batch_size)

    model = LSTMClassifier(n_classes, cfg).to(DEVICE)
    optimizer = build_optimizer(model, optimizer_name, lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cosine_schedule else None)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    t0 = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_aug(model, train_loader, optimizer, criterion, aug)
        val_loss, val_preds = evaluate_tensors(model, X_val, y_val, criterion)
        val_acc = (val_preds == y_val.cpu().numpy()).mean()

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"    epoch {epoch + 1:>3}/{epochs}  "
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
# Porovnania podľa zadania - rovnaká logika ako v cnn_model.py
# ---------------------------------------------------------------------------
def score_config_across_splits(
    cfg: LSTMConfig,
    training_cfg: dict,
    n_splits: int = N_SPLITS,
) -> tuple[float, float, float, dict]:
    """
    Vyhodnotí jednu konfiguráciu na VŠETKÝCH splitoch (rovnaký dôvod ako pri CNN -
    validačná množina má len 96 okien, jeden beh je hlučný). Testovacie dáta sa
    pri výbere nepoužívajú.

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


def _compare(variants, training_cfg_for, label_for, header, title, filename, n_splits):
    """
    Spoločné telo všetkých porovnávacích tabuliek: pre každý variant spustí
    5 splitov, vypíše riadok tabuľky, zapamätá priebeh trénovania a vráti
    najlepší variant podľa (val accuracy, val precision+recall).
    """
    print(f"  {header:<58} {'VAL acc (priemer +- std)'}")

    best, histories = None, {}
    for variant in variants:
        cfg, training_cfg = training_cfg_for(variant)
        mean_acc, std_acc, mean_pr, history = score_config_across_splits(cfg, training_cfg, n_splits)

        label = label_for(variant)
        print(f"  {label:<58} {mean_acc * 100:6.2f} % +- {std_acc * 100:5.2f} %")
        histories[label] = history

        score = (mean_acc, mean_pr)
        if best is None or score > best["score"]:
            best = {"variant": variant, "score": score}

    plot_training_curves_comparison(histories, title=title, filename=filename, out_dir=RESULTS_LSTM_DIR)
    return best["variant"]


def compare_input_representation(base_cfg: LSTMConfig, training_cfg: dict, n_splits: int = N_SPLITS) -> LSTMConfig:
    """
    Vstupná reprezentácia: dĺžka sekvencie (frame_size) a spôsob poolingu.

    Toto je hlavná príčina, prečo prvá verzia modelu zlyhávala - porovnanie sa
    preto robí ešte pred ladením parametrov trénovania, nad rovnakými dátami.
    Kvôli názornosti tu ide augmentácia bokom (porovnáva sa čistý vplyv vstupu).
    """
    variants = [
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=1, pool="last"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=1, pool="meanmax"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=5, pool="last"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=5, pool="mean"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=5, pool="meanmax"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=10, pool="meanmax"),
        LSTMConfig(hidden_size=base_cfg.hidden_size, num_layers=1, fc_dim=base_cfg.fc_dim,
                   dropout=0.3, bidirectional=base_cfg.bidirectional, frame_size=20, pool="meanmax"),
    ]

    print(f"\nPorovnanie vstupnej reprezentácie - dĺžka sekvencie a pooling "
          f"(priemer z {n_splits} splitov, bez augmentácie)\n")

    best = _compare(
        variants,
        training_cfg_for=lambda v: (v, training_cfg),
        label_for=lambda v: f"frame={v.frame_size:>2} ({400 // v.frame_size:>3} krokov), pool={v.pool}",
        header="vstupná reprezentácia",
        title="LSTM - porovnanie vstupnej reprezentácie (split_00)",
        filename="input_representation_training_curves.png",
        n_splits=n_splits,
    )
    print(f"\nNajlepšia vstupná reprezentácia: frame_size={best.frame_size}, pool={best.pool}")
    return best


def compare_augmentation(cfg: LSTMConfig, training_cfg: dict, n_splits: int = N_SPLITS) -> AugmentConfig | None:
    """
    Ablácia augmentácie. Dataset má len 480 trénovacích okien (60 meraní),
    takže regularizácia dátami je tu účinnejšia než akékoľvek zväčšovanie siete.
    """
    variants = [
        None,
        AugmentConfig(time_shift=True, amp_scale=0.0, noise_std=0.0),
        AugmentConfig(time_shift=False, amp_scale=0.1, noise_std=0.05),
        AugmentConfig(time_shift=True, amp_scale=0.1, noise_std=0.05),
    ]

    print(f"\nVplyv augmentácie trénovacích dát ({cfg.name}, priemer z {n_splits} splitov)\n")

    best = _compare(
        variants,
        training_cfg_for=lambda v: (cfg, {**training_cfg, "aug": v}),
        label_for=lambda v: v.name if v is not None else "bez augmentácie",
        header="augmentácia",
        title="LSTM - vplyv augmentácie trénovacích dát (split_00)",
        filename="augmentation_training_curves.png",
        n_splits=n_splits,
    )
    print(f"\nNajlepšia augmentácia: {best.name if best is not None else 'bez augmentácie'}")
    return best


def compare_training_hyperparameters(cfg: LSTMConfig, aug: AugmentConfig | None,
                                     n_splits: int = N_SPLITS) -> dict:
    """
    Bod 1 zo zadania - optimalizátor, learning rate, batch size a počet epoch.
    Rozsah epoch je oproti CNN posunutý vyššie: s augmentáciou je každá epocha
    "iná" a model sa dá trénovať dlhšie bez preučenia.
    """
    configs = [
        {"optimizer_name": "sgdm", "lr": 0.01, "batch_size": 16, "epochs": 60},
        {"optimizer_name": "sgdm", "lr": 0.001, "batch_size": 16, "epochs": 60},
        {"optimizer_name": "adamw", "lr": 0.01, "batch_size": 16, "epochs": 60},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 60},
        {"optimizer_name": "adamw", "lr": 0.0001, "batch_size": 16, "epochs": 60},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 32, "epochs": 60},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 30},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 120},
        {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16, "epochs": 200},
    ]

    print(f"\nPorovnanie parametrov trénovania pre {cfg.name} "
          f"(priemer validačnej úspešnosti z {n_splits} splitov):\n")

    best = _compare(
        configs,
        training_cfg_for=lambda c: (cfg, {**c, "aug": aug}),
        label_for=lambda c: ", ".join(f"{k}={v}" for k, v in c.items()),
        header="konfigurácia",
        title=f"LSTM ({cfg.name}) - porovnanie parametrov trénovania (split_00)",
        filename="hyperparam_training_curves.png",
        n_splits=n_splits,
    )
    print("\nNajlepšie parametre trénovania: " + ", ".join(f"{k}={v}" for k, v in best.items()))
    return best


def compare_architectures(base_cfg: LSTMConfig, training_cfg: dict, n_splits: int = N_SPLITS) -> LSTMConfig:
    """
    Bod 2 zo zadania: viacero štruktúr siete (veľkosť a počet vrstiev,
    jedno- vs. obojsmerná LSTM) + ablácia Dropoutu, aby bol vidno jeho vplyv zvlášť.
    """
    def variant(**kwargs):
        params = dict(hidden_size=128, num_layers=1, fc_dim=64, dropout=0.3,
                      bidirectional=True, frame_size=base_cfg.frame_size, pool=base_cfg.pool)
        params.update(kwargs)
        return LSTMConfig(**params)

    architectures = [
        variant(),                                   # referenčná
        variant(dropout=0.0),                        # ablácia dropoutu
        variant(dropout=0.5),                        # silnejší dropout
        variant(hidden_size=64, fc_dim=32),          # menšia sieť
        variant(num_layers=2),                       # hlbšia sieť
        variant(bidirectional=False),                # jednosmerná
    ]

    print(f"\nPorovnanie architektúr (priemer z {n_splits} splitov), "
          f"parametre trénovania: { {k: v for k, v in training_cfg.items() if k != 'aug'} }\n")

    best = _compare(
        architectures,
        training_cfg_for=lambda a: (a, training_cfg),
        label_for=lambda a: a.name,
        header="architektúra",
        title="LSTM - porovnanie architektúr (split_00)",
        filename="architecture_training_curves.png",
        n_splits=n_splits,
    )
    print(f"\nNajlepšia architektúra: {best.name}")
    return best


def run_multiple_splits(cfg: LSTMConfig, training_cfg: dict, n_splits: int = N_SPLITS):
    """
    Zadanie bod 6: najlepšiu štruktúru spustiť 5x a spočítať min/max/priemer metrík.

    Navyše pre KAŽDÝ z 5 behov uloží: surové predikcie + históriu trénovania
    (results/lstm/runs/), graf priebehu daného behu a jeho kontingenčnú maticu.
    Po skončení vytvorí súčtové kontingenčné matice (5 behov spolu) a maticu
    najlepšieho behu.
    """
    val_runs, test_runs, train_seconds = [], [], []
    histories = {}
    y_vals, y_val_preds, y_tests, y_test_preds = [], [], [], []

    printable = {k: (v.name if isinstance(v, AugmentConfig) else v) for k, v in training_cfg.items()}
    print(f"\n{'=' * 70}")
    print(f"Spustenie {cfg.name} na {n_splits} rozdeleniach (parametre: {printable})")
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
            RESULTS_LSTM_DIR, split_idx, val_metrics, test_metrics,
            y_val, val_pred, y_test, test_pred, secs, history=history,
        )
        plot_single_run_training_curve(
            history,
            title=f"LSTM ({cfg.name}) - priebeh trénovania (split_{split_idx:02d})",
            filename=f"runs/split_{split_idx:02d}_training_curve.png",
            out_dir=RESULTS_LSTM_DIR,
        )
        plot_confusion_matrix_pair(
            y_test, test_pred,
            title=f"LSTM ({cfg.name}) - kontingenčná matica (split_{split_idx:02d}, testovacie dáta)",
            out_dir=RESULTS_LSTM_DIR,
            filename=f"runs/split_{split_idx:02d}_confusion.png",
            n_okien_info=f"beh split_{split_idx:02d}, {len(y_test)} okien, čas trénovania {secs:.2f} s",
        )

        print(f"  split_{split_idx:02d}:  VAL  {format_metrics(val_metrics)}   |   "
              f"TEST  {format_metrics(test_metrics)}   |   čas={secs:.2f} s")

    write_runs_csv(RESULTS_LSTM_DIR, val_runs, test_runs, train_seconds)

    val_summary = summarize_runs(val_runs)
    test_summary = summarize_runs(test_runs)
    print_summary(f"LSTM - validačné dáta (z {len(val_runs)} behov)", val_summary)
    print_summary(f"LSTM - testovacie dáta (z {len(test_runs)} behov)", test_summary)

    plot_training_curves_comparison(
        histories,
        title=f"LSTM ({cfg.name}) - priebeh trénovania naprieč {n_splits} behmi (splitmi)",
        filename="final_splits_training_curves.png",
        out_dir=RESULTS_LSTM_DIR,
    )
    plot_runs_bar_chart(
        list(histories.keys()),
        [run["accuracy"] for run in val_runs],
        [run["accuracy"] for run in test_runs],
        title=f"LSTM ({cfg.name}) - úspešnosť naprieč {n_splits} behmi (splitmi)",
        ylabel="Accuracy [%]",
        filename="final_splits_accuracy.png",
        out_dir=RESULTS_LSTM_DIR,
        val_summary=val_summary,
        test_summary=test_summary,
    )

    generate_post_run_confusion_matrices(
        "LSTM", RESULTS_LSTM_DIR, y_vals, y_val_preds, y_tests, y_test_preds, test_runs,
        best_filename="lstm_confusion_matrix.png",
        combined_test_filename="lstm_confusion_matrix_combined.png",
        combined_val_filename="lstm_confusion_matrix_combined_val.png",
    )


if __name__ == "__main__":
    with capture_console_to_markdown(
        "lstm_training_log.md", RESULTS_LSTM_DIR, title="LSTM - priebeh trénovania a vyhodnotenia"
    ):
        print(f"Device: {DEVICE}\n")

        # Východiskové nastavenie trénovania pre prvé dve porovnania
        start_training_cfg = {"optimizer_name": "adamw", "lr": 0.001, "batch_size": 16,
                              "epochs": 60, "aug": None}

        # Krok A: vstupná reprezentácia (dĺžka sekvencie + pooling)
        base_cfg = LSTMConfig(hidden_size=128, num_layers=1, fc_dim=64, dropout=0.3,
                              bidirectional=True, frame_size=5, pool="meanmax")
        best_input = compare_input_representation(base_cfg, start_training_cfg)

        # Krok B: augmentácia trénovacích dát
        best_aug = compare_augmentation(best_input, start_training_cfg)

        # Bod 1 zo zadania: parametre trénovania
        best_training_cfg = compare_training_hyperparameters(best_input, best_aug)
        best_training_cfg = {**best_training_cfg, "aug": best_aug}

        # Bod 2 zo zadania: štruktúry siete (vrátane vplyvu Dropoutu)
        best_arch = compare_architectures(best_input, best_training_cfg)

        # Bod 6 zo zadania: 5 nezávislých rozdelení
        run_multiple_splits(best_arch, best_training_cfg)
