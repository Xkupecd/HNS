"""
Extrakcia časových príznakov (features) z okien signálu, pre SVM a MLP
(manuálna extrakcia príznakov - pozri zadanie, Obr. 2).

9 príznakov na kanál: MAV, NZC, NSSC, RMS, WL, WAMP, MAX, MIN, PTP.
Pri ACC+GYR (6 kanálov: X,Y,Z pre každý) je výsledný vektor 9*6 = 54 príznakov.

POZNÁMKA k demo kódu z utils.zip (feature_extraction.py):
- jeho MAV nedelí počtom vzoriek N, takže je to v skutočnosti IAV
  (Integrated Absolute Value), nie MAV podľa vzorca v zadaní - tu je to opravené
- jeho WAMP počíta iba jednosmerné rozdiely (data[i]-data[i+1] >= th), tu je
  použitá štandardná obojsmerná definícia s abs() - ak chceš zachovať presne
  pôvodnú (jednosmernú) verziu pre konzistenciu s ukážkami z prednášky, uprav wamp()
"""
from pathlib import Path

import numpy as np

from config import (
    ACTIVE_SENSORS,
    AXIS_NAMES,
    NSSC_THRESHOLD,
    NZC_THRESHOLD,
    PROCESSED_DATA_DIR,
    WAMP_THRESHOLD,
)

FEATURE_NAMES = ("MAV", "NZC", "NSSC", "RMS", "WL", "WAMP", "MAX", "MIN", "PTP")


def mav(x: np.ndarray) -> float:
    """Priemerná absolútna hodnota: MAV = (1/N) * sum(|x_i|)."""
    return float(np.mean(np.abs(x)))


def nzc(x: np.ndarray, th: float = NZC_THRESHOLD) -> int:
    """Počet prechodov nulou: susedné vzorky majú opačné znamienko A rozdiel >= th."""
    sign_product = x[:-1] * x[1:]
    diffs = np.abs(x[1:] - x[:-1])
    crossed = (sign_product < 0) & (diffs >= th)
    return int(np.sum(crossed))


def nssc(x: np.ndarray, th: float = NSSC_THRESHOLD) -> int:
    """Počet zmien sklonových značiek (lokálne extrémy s dostatočnou 'krivosťou')."""
    temp = (x[1:-1] - x[:-2]) * (x[1:-1] - x[2:])
    return int(np.sum(temp >= th))


def rms(x: np.ndarray) -> float:
    """Odmocnina kvadrátu priemeru: RMS = sqrt((1/N) * sum(x_i^2))."""
    return float(np.sqrt(np.mean(x ** 2)))


def wl(x: np.ndarray) -> float:
    """Dĺžka priebehu: WL = sum(|x_{i+1} - x_i|)."""
    return float(np.sum(np.abs(np.diff(x))))


def wamp(x: np.ndarray, th: float = WAMP_THRESHOLD) -> int:
    """Willisonova amplitúda: počet susedných rozdielov s |x_i - x_{i+1}| >= th."""
    return int(np.sum(np.abs(x[:-1] - x[1:]) >= th))


def ptp(x: np.ndarray) -> float:
    """Dĺžka vlny (peak-to-peak): PTP = MAX - MIN."""
    return float(np.max(x) - np.min(x))


def extract_features_1d(x: np.ndarray) -> np.ndarray:
    """Vypočíta všetkých 9 príznakov pre jeden 1D signál (jeden kanál, jedno okno)."""
    return np.array([
        mav(x), nzc(x), nssc(x), rms(x), wl(x), wamp(x), np.max(x), np.min(x), ptp(x),
    ])


def get_channel_names(sensors=ACTIVE_SENSORS, axes=AXIS_NAMES) -> list[str]:
    """Poradie kanálov musí zodpovedať poradiu, v akom windowing.py skladá okná."""
    return [f"{sensor}_{axis}" for sensor in sensors for axis in axes]


def get_feature_names(sensors=ACTIVE_SENSORS, axes=AXIS_NAMES) -> list[str]:
    """Názvy všetkých 9*n_kanálov príznakov, v poradí zodpovedajúcom extract_features_window."""
    return [f"{ch}_{feat}" for ch in get_channel_names(sensors, axes) for feat in FEATURE_NAMES]


def extract_features_window(window: np.ndarray) -> np.ndarray:
    """
    Vypočíta príznaky pre jedno okno tvaru (window_size, n_channels).
    Vráti vektor dĺžky 9 * n_channels (poradie: kanál 0 [9 príznakov], kanál 1 [9 príznakov], ...).
    """
    n_channels = window.shape[1]
    return np.concatenate([extract_features_1d(window[:, ch]) for ch in range(n_channels)])


def extract_features_dataset(X: np.ndarray) -> np.ndarray:
    """
    Vypočíta príznaky pre celý dataset okien.

    X tvaru (n_okien, window_size, n_channels) -> vráti (n_okien, 9 * n_channels).
    """
    return np.stack([extract_features_window(w) for w in X])


def build_and_save_features(
    split_idx: int = 0,
    windows_root: Path = PROCESSED_DATA_DIR / "windows",
    out_root: Path = PROCESSED_DATA_DIR / "features",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Načíta uložené okná (z windowing.py) pre daný split, vypočíta príznaky
    pre train/val/test a uloží ich ako .npy, aby sa nemuseli počítať opakovane.
    """
    windows_dir = windows_root / f"split_{split_idx:02d}"
    if not windows_dir.exists():
        raise FileNotFoundError(f"Priečinok {windows_dir} neexistuje. Spusti najprv 'python windowing.py'.")

    out_dir = out_root / f"split_{split_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for subset in ("train", "val", "test"):
        X = np.load(windows_dir / f"X_{subset}.npy")
        y = np.load(windows_dir / f"y_{subset}.npy")

        features = extract_features_dataset(X)

        np.save(out_dir / f"features_{subset}.npy", features)
        np.save(out_dir / f"y_{subset}.npy", y)

        result[subset] = (features, y)

    print(f"Príznaky uložené do: {out_dir}")
    return result


def load_features(split_idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Načíta uložené príznaky pre daný split. Ak ešte nie sú vypočítané,
    vypočíta ich (vyžaduje, aby windowing.py už vytvoril okná pre tento split).

    Používa sa spoločne zo všetkých modelov nad príznakmi (SVM aj MLP),
    aby mali garantovane rovnaké dáta.
    """
    feat_dir = PROCESSED_DATA_DIR / "features" / f"split_{split_idx:02d}"

    if not feat_dir.exists():
        return build_and_save_features(split_idx=split_idx)

    data = {}
    for subset in ("train", "val", "test"):
        X = np.load(feat_dir / f"features_{subset}.npy")
        y = np.load(feat_dir / f"y_{subset}.npy")
        data[subset] = (X, y)
    return data


if __name__ == "__main__":
    data = build_and_save_features(split_idx=0)

    feature_names = get_feature_names()
    print(f"Počet príznakov na okno: {len(feature_names)}  (očakávaných: {9 * len(ACTIVE_SENSORS) * 3})")
    print(f"Prvých pár názvov: {feature_names[:6]} ...")

    for subset, (features, y) in data.items():
        print(f"\n{subset}: features {features.shape}, y {y.shape}")
        # Sanity check: NaN/Inf by znamenali chybu vo výpočte (napr. prázdne okno)
        assert np.isfinite(features).all(), f"Neplatné hodnoty (NaN/Inf) v príznakoch pre {subset}!"

    print("\nOK: žiadne NaN/Inf hodnoty v príznakoch.")
