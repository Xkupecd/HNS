"""
Rozdelenie datasetu na trénovacie / validačné / testovacie časti.

KĽÚČOVÉ: rozdeľuje sa na úrovni MERANÍ (terrain, speed, session), nie okien.
Okná z toho istého 4.5 s merania sa navzájom prekrývajú a sú silno korelované -
ak by sa rozdeľovalo až po windowingu, okná z jedného merania by mohli skončiť
aj v tréningu aj v teste, čo by umelo nafúklo presnosť a znehodnotilo výsledky.

Pre každý terén je 14 meraní (7 session x 2 rýchlosti) -> podľa zadania sa
náhodne vyberie 10 na trénovanie, 2 na validáciu, 2 na testovanie.

Okrem rozdelenia "v pamäti" (make_all_splits) táto časť vie rozdelenie aj
reálne materializovať na disk - t.j. skopírovať zodpovedajúce surové CSV
súbory do priečinkov data/processed/splits/split_XX/{train,val,test}/<terén>/,
aby bolo vidno konkrétne súbory a dalo sa nad nimi robiť windowing priamo
prechádzaním priečinkov.
"""
import shutil

import numpy as np
import pandas as pd

from config import (
    N_SPLITS,
    N_TEST,
    N_TRAIN,
    N_VAL,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SENSORS,
    SPEED_NAMES,
)
from data_loading import build_catalog


def get_unique_measurements(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Vráti jeden riadok na skutočné meranie (terrain_code, speed_code, session),
    bez duplicity naprieč snímačmi - ACC/GYR/MAG toho istého merania patria
    do toho istého splitu.

    Zahodí merania s nedokumentovaným speed_code (napr. MAG_2_1_1.csv, ktorý
    má speed_code=1 - README databázy pozná iba kódy 2 a 3).
    """
    known_speed = catalog["speed_code"].isin(SPEED_NAMES.keys())
    n_dropped = (~known_speed).sum()
    if n_dropped:
        dropped_files = catalog.loc[~known_speed, "path"].tolist()
        print(f"[splitting] Vynechávam {n_dropped} súbor(ov) s nedokumentovaným speed_code: "
              f"{[p.name for p in dropped_files]}")

    unique = (
        catalog.loc[known_speed, ["terrain_code", "terrain_name", "speed_code", "speed_name", "session"]]
        .drop_duplicates(subset=["terrain_code", "speed_code", "session"])
        .reset_index(drop=True)
    )
    return unique


def make_split(unique_measurements: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """
    Pre daný seed náhodne rozdelí meranie KAŽDÉHO terénu zvlášť na
    train (N_TRAIN) / val (N_VAL) / test (N_TEST).

    Vráti {"train": df, "val": df, "test": df}, každý df so stĺpcami
    terrain_code, terrain_name, speed_code, speed_name, session.
    """
    expected_total = N_TRAIN + N_VAL + N_TEST
    rng = np.random.default_rng(seed)

    parts = {"train": [], "val": [], "test": []}

    for terrain_code, group in unique_measurements.groupby("terrain_code"):
        if len(group) != expected_total:
            raise ValueError(
                f"Terén {terrain_code} má {len(group)} meraní, očakávaných bolo {expected_total} "
                f"({N_TRAIN}+{N_VAL}+{N_TEST}). Skontroluj dataset / anomálie."
            )
        shuffled_idx = rng.permutation(len(group))
        group_shuffled = group.iloc[shuffled_idx].reset_index(drop=True)

        parts["train"].append(group_shuffled.iloc[:N_TRAIN])
        parts["val"].append(group_shuffled.iloc[N_TRAIN:N_TRAIN + N_VAL])
        parts["test"].append(group_shuffled.iloc[N_TRAIN + N_VAL:])

    return {name: pd.concat(dfs).reset_index(drop=True) for name, dfs in parts.items()}


def make_all_splits(catalog: pd.DataFrame | None = None, n_splits: int = N_SPLITS, base_seed: int = RANDOM_SEED) -> list[dict[str, pd.DataFrame]]:
    """
    Vytvorí n_splits nezávislých rozdelení (rôzne seedy = base_seed + i),
    rovnaké pre všetky modely, aby bolo porovnanie SVM/MLP/LSTM/CNN fér.
    """
    if catalog is None:
        catalog = build_catalog()

    unique_measurements = get_unique_measurements(catalog)
    return [make_split(unique_measurements, seed=base_seed + i) for i in range(n_splits)]


def materialize_splits(
    splits: list[dict[str, pd.DataFrame]],
    raw_data_dir=RAW_DATA_DIR,
    out_dir=PROCESSED_DATA_DIR / "splits",
    sensors=SENSORS,
) -> None:
    """
    Reálne skopíruje surové CSV súbory podľa vypočítaných rozdelení do:

        data/processed/splits/split_00/train/<terrain_name>/SEN_x_y_z.csv
        data/processed/splits/split_00/val/<terrain_name>/SEN_x_y_z.csv
        data/processed/splits/split_00/test/<terrain_name>/SEN_x_y_z.csv
        data/processed/splits/split_01/...
        ...

    Priečinok terénu (napr. "concrete") slúži zároveň ako label - žiadny
    ďalší súbor s metadátami netreba, dá sa nad tým rovno iterovať.

    Volá sa opakovane pre každý z N_SPLITS nezávislých rozdelení, takže
    vznikne split_00 .. split_0{N_SPLITS-1}.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)  # čisté prehratie pri opätovnom spustení

    n_missing = 0
    for split_idx, split in enumerate(splits):
        split_dir = out_dir / f"split_{split_idx:02d}"

        for subset_name, measurements in split.items():
            for _, row in measurements.iterrows():
                terrain_name = row["terrain_name"]
                terrain_code, speed_code, session = row["terrain_code"], row["speed_code"], row["session"]

                dest_dir = split_dir / subset_name / terrain_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for sensor in sensors:
                    src_path = raw_data_dir / f"{sensor}_{terrain_code}_{speed_code}_{session}.csv"
                    if not src_path.exists():
                        n_missing += 1
                        continue
                    shutil.copy2(src_path, dest_dir / src_path.name)

    if n_missing:
        print(f"[splitting] Pozor: {n_missing} očakávaných súborov sa nenašlo (chýbajúci snímač pre dané meranie).")

    print(f"Rozdelenia skopírované do: {out_dir}")


if __name__ == "__main__":
    catalog = build_catalog()
    splits = make_all_splits(catalog)

    print(f"Vytvorených {len(splits)} nezávislých rozdelení.")
    split0 = splits[0]
    for name, df in split0.items():
        print(f"  {name}: {len(df)} meraní, terénov: {df['terrain_code'].nunique()}, "
              f"po {df.groupby('terrain_code').size().unique()} na terén")

    # Kontrola, že sa medzi splitmi neprekrývajú žiadne merania (žiadny leak)
    train_ids = set(map(tuple, split0["train"][["terrain_code", "speed_code", "session"]].values))
    val_ids = set(map(tuple, split0["val"][["terrain_code", "speed_code", "session"]].values))
    test_ids = set(map(tuple, split0["test"][["terrain_code", "speed_code", "session"]].values))
    assert not (train_ids & val_ids), "Prekryv train/val!"
    assert not (train_ids & test_ids), "Prekryv train/test!"
    assert not (val_ids & test_ids), "Prekryv val/test!"
    print("OK: train/val/test sa navzájom neprekrývajú (žiadny leak na úrovni meraní).")

    materialize_splits(splits)