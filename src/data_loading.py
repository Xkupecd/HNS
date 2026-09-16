"""
Knižnica na načítanie surových CSV meraní a zostavenie katalógu (metadát) nad
celou databázou. Dalej ju použijú iné moduly.
Načítanie surových CSV meraní a zostavenie katalógu (metadát) nad
celou databázou.

Formát názvu súboru (podľa README databázy): SEN_x_y_z.csv
    SEN - typ snímača: ACC, GYR, MAG
    x   - kód triedy terénu (1-6)
    y   - kód rýchlosti (2 = 86 % PWM, 3 = 100 % PWM)
    z   - kód session/merania (1-7)
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DATA_DIR, SPEED_NAMES, TERRAIN_NAMES

FILENAME_PATTERN = re.compile(r"^(?P<sensor>ACC|GYR|MAG)_(?P<terrain>\d+)_(?P<speed>\d+)_(?P<session>\d+)\.csv$")


@dataclass(frozen=True)
class MeasurementInfo:
    """Metadáta k jednému CSV súboru (jedno meranie, jeden snímač)."""
    path: Path
    sensor: str
    terrain_code: int
    terrain_name: str
    speed_code: int
    speed_name: str
    session: int

    @property
    def measurement_id(self) -> str:
        """ID merania spoločné pre všetky snímače (bez predpony SEN_)."""
        return f"{self.terrain_code}_{self.speed_code}_{self.session}"


def parse_filename(path: Path) -> MeasurementInfo:
    """Rozparsuje meno súboru SEN_x_y_z.csv na metadáta."""
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Názov súboru nezodpovedá očakávanému formátu SEN_x_y_z.csv: {path.name}")

    terrain_code = int(match.group("terrain"))
    speed_code = int(match.group("speed"))

    return MeasurementInfo(
        path=path,
        sensor=match.group("sensor"),
        terrain_code=terrain_code,
        terrain_name=TERRAIN_NAMES.get(terrain_code, f"unknown_{terrain_code}"),
        speed_code=speed_code,
        speed_name=SPEED_NAMES.get(speed_code, f"unknown_{speed_code}"),
        session=int(match.group("session")),
    )


def build_catalog(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Prejde všetky .csv súbory v priečinku s meraniami a vráti DataFrame
    s jedným riadkom na súbor (metadáta + cesta), zoradený podľa
    terénu / rýchlosti / session / snímača.
    """
    csv_paths = sorted(raw_data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"V priečinku {raw_data_dir} sa nenašli žiadne .csv súbory.")

    records = [parse_filename(p) for p in csv_paths]

    df = pd.DataFrame([r.__dict__ for r in records])
    df = df.sort_values(["terrain_code", "speed_code", "session", "sensor"]).reset_index(drop=True)
    return df


def load_signal(path: Path) -> np.ndarray:
    """
    Načíta jedno CSV meranie ako numpy pole tvaru (N, 3) - stĺpce X, Y, Z.
    Súbory neobsahujú hlavičku.
    """
    return pd.read_csv(path, header=None, names=["X", "Y", "Z"]).to_numpy(dtype=float)


if __name__ == "__main__":
    catalog = build_catalog()
    print(f"Nájdených {len(catalog)} súborov meraní.")
    print(catalog.groupby(["sensor", "terrain_name", "speed_name"]).size())
