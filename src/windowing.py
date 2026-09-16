"""
Sliding window - narezanie meraní na 1 s okná s posunom 0.5 s.

Podľa zadania: pre ACC a GYR (400 Hz) má okno 400 vzoriek a posun 200 vzoriek.
Meranie má 1800 vzoriek (4.5 s) -> (1800 - 400) / 200 + 1 = 8 okien na meranie.

Vstupom sú priečinky vytvorené v splitting.py:
    data/processed/splits/split_XX/{train,val,test}/<terrain_name>/SEN_x_y_z.csv

Výstupom sú numpy polia:
    X tvaru (n_okien, 400, n_kanálov)   - kanály = ACC_X, ACC_Y, ACC_Z, GYR_X, GYR_Y, GYR_Z
    y tvaru (n_okien,)                  - index triedy terénu 0..5
    groups tvaru (n_okien,)             - ID merania, z ktorého okno pochádza
                                          (na kontrolu, že nedochádza k leaku)

Tento formát ide priamo do LSTM a CNN. Pre SVM a MLP sa z týchto okien
ešte počítajú príznaky (features.py).
"""
from pathlib import Path

import numpy as np

from config import (
    ACTIVE_SENSORS,
    N_SPLITS,
    PROCESSED_DATA_DIR,
    SAMPLING_RATE_HZ,
    STEP_S,
    TERRAIN_NAMES,
    WINDOW_S,
)
from data_loading import load_signal

# Mapovanie názvu terénu -> index triedy 0..5 (pre PyTorch / sklearn)
TERRAIN_TO_LABEL = {name: idx for idx, name in enumerate(TERRAIN_NAMES.values())}
LABEL_TO_TERRAIN = {idx: name for name, idx in TERRAIN_TO_LABEL.items()}


def create_sliding_windows(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Nareže signál (N, n_osi) na prekrývajúce sa okná.

    Vráti pole tvaru (n_okien, window_size, n_osi).
    Neúplný zvyšok na konci sa zahodí.
    """
    n_samples = signal.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, signal.shape[1]))

    starts = range(0, n_samples - window_size + 1, step_size)
    return np.stack([signal[s:s + window_size] for s in starts])


def load_measurement_windows(
    terrain_dir: Path,
    measurement_key: str,
    sensors=ACTIVE_SENSORS,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
) -> np.ndarray:
    """
    Pre JEDNO meranie (napr. "1_2_3") načíta všetky aktívne snímače,
    nareže ich na okná a spojí ich po kanáloch.

    Predpoklad: všetky aktívne snímače majú rovnakú vzorkovaciu frekvenciu,
    takže ich okná sú časovo zarovnané 1:1. (Preto ACTIVE_SENSORS = ACC + GYR,
    obe 400 Hz; MAG má 100 Hz a musel by sa prevzorkovať.)

    Vráti pole tvaru (n_okien, window_size, 3 * len(sensors)).
    """
    fs_values = {SAMPLING_RATE_HZ[s] for s in sensors}
    if len(fs_values) != 1:
        raise ValueError(
            f"Aktívne snímače {sensors} majú rôzne vzorkovacie frekvencie {fs_values}. "
            "Okná by neboli časovo zarovnané - bolo by potrebné prevzorkovanie."
        )
    fs = fs_values.pop()
    window_size = round(window_s * fs)
    step_size = round(step_s * fs)

    per_sensor_windows = []
    for sensor in sensors:
        path = terrain_dir / f"{sensor}_{measurement_key}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Chýba súbor snímača: {path}")
        signal = load_signal(path)  # (N, 3)
        windows = create_sliding_windows(signal, window_size, step_size)
        per_sensor_windows.append(windows)

    # Zarovnanie počtu okien (ak by mali merania mierne odlišnú dĺžku)
    n_windows = min(w.shape[0] for w in per_sensor_windows)
    per_sensor_windows = [w[:n_windows] for w in per_sensor_windows]

    # Spojenie po kanáloch: (n_okien, window_size, 3*n_snímačov)
    return np.concatenate(per_sensor_windows, axis=2)


def build_subset(
    subset_dir: Path,
    sensors=ACTIVE_SENSORS,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prejde jeden podpriečinok (train / val / test) a poskladá z neho dataset.

    Vráti (X, y, groups):
        X      - (n_okien, window_size, n_kanálov)
        y      - (n_okien,) index triedy 0..5
        groups - (n_okien,) ID merania "terrain_speed_session" (na kontrolu leaku)
    """
    X_parts, y_parts, group_parts = [], [], []

    for terrain_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
        label = TERRAIN_TO_LABEL[terrain_dir.name]

        # Kľúče meraní odvodíme z prvého snímača, aby sme každé meranie vzali raz
        first_sensor = sensors[0]
        measurement_keys = sorted(
            p.stem.split("_", 1)[1] for p in terrain_dir.glob(f"{first_sensor}_*.csv")
        )

        for key in measurement_keys:
            windows = load_measurement_windows(terrain_dir, key, sensors, window_s, step_s)
            if windows.shape[0] == 0:
                continue
            X_parts.append(windows)
            y_parts.append(np.full(windows.shape[0], label, dtype=np.int64))
            group_parts.append(np.full(windows.shape[0], key, dtype=object))

    if not X_parts:
        raise RuntimeError(f"V priečinku {subset_dir} sa nenašli žiadne merania.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    return X, y, groups


def build_split_dataset(
    split_idx: int = 0,
    splits_root: Path = PROCESSED_DATA_DIR / "splits",
    sensors=ACTIVE_SENSORS,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Poskladá kompletný dataset pre jedno rozdelenie (split_XX).

    Vráti {"train": (X, y, groups), "val": (...), "test": (...)}.
    """
    split_dir = splits_root / f"split_{split_idx:02d}"
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Priečinok {split_dir} neexistuje. Spusti najprv 'python splitting.py'."
        )

    return {subset: build_subset(split_dir / subset, sensors) for subset in ("train", "val", "test")}


def save_split_dataset(split_idx: int = 0, out_root: Path = PROCESSED_DATA_DIR / "windows") -> None:
    """Uloží narezaný dataset ako .npy, aby sa windowing nemusel opakovať pri každom tréningu."""
    data = build_split_dataset(split_idx)
    out_dir = out_root / f"split_{split_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subset, (X, y, groups) in data.items():
        np.save(out_dir / f"X_{subset}.npy", X)
        np.save(out_dir / f"y_{subset}.npy", y)
        np.save(out_dir / f"groups_{subset}.npy", groups)

    print(f"Uložené do: {out_dir}")


def load_windowed_dataset(
    split_idx: int,
    windows_root: Path = PROCESSED_DATA_DIR / "windows",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Načíta uložené okná (surový signál, nie príznaky) pre daný split.
    Ak ešte nie sú vypočítané, dopočíta a uloží ich.

    Používajú to modely, ktoré pracujú priamo so signálom (LSTM, CNN) -
    analogicky k features.load_features() pre SVM/MLP.
    """
    windows_dir = windows_root / f"split_{split_idx:02d}"

    if not windows_dir.exists():
        save_split_dataset(split_idx, out_root=windows_root)

    data = {}
    for subset in ("train", "val", "test"):
        X = np.load(windows_dir / f"X_{subset}.npy")
        y = np.load(windows_dir / f"y_{subset}.npy")
        data[subset] = (X, y)
    return data


if __name__ == "__main__":
    import sys

    # Predvolene sa spracujú všetky splity; 'python windowing.py 0' spracuje len split_00
    split_indices = [int(a) for a in sys.argv[1:]] or list(range(N_SPLITS))

    for split_idx in split_indices:
        print(f"\n{'=' * 60}\nsplit_{split_idx:02d}\n{'=' * 60}")
        data = build_split_dataset(split_idx=split_idx)

        print(f"Aktívne snímače: {ACTIVE_SENSORS}")
        for subset, (X, y, groups) in data.items():
            counts = np.bincount(y, minlength=len(TERRAIN_TO_LABEL))
            print(f"  {subset:<6} X: {str(X.shape):<20} okien na triedu: {counts.tolist()}, "
                  f"meraní: {len(set(groups))}")

        # Kontrola leaku: žiadne meranie nesmie byť v dvoch podmnožinách naraz
        g_train, g_val, g_test = (set(data[s][2]) for s in ("train", "val", "test"))
        assert not (g_train & g_val), f"LEAK v split_{split_idx:02d}: meranie v train aj val!"
        assert not (g_train & g_test), f"LEAK v split_{split_idx:02d}: meranie v train aj test!"
        assert not (g_val & g_test), f"LEAK v split_{split_idx:02d}: meranie v val aj test!"
        print("  OK: žiadny leak medzi podmnožinami.")

        save_split_dataset(split_idx=split_idx)