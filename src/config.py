"""
Centrálna konfigurácia projektu HNS - Zadanie 1 (klasifikácia terénu).

Všetky ostatné moduly importujú konštanty odtiaľto, aby existovalo
jedno miesto pravdy pre cesty, vzorkovacie frekvencie a parametre okna.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Cesty
# ---------------------------------------------------------------------------
# PROJECT_ROOT = .../Zadanie1  (o dva levely vyššie ako tento súbor: src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "outdoor_terrain_classification_database" / "Measurements"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Snímače a vzorkovacie frekvencie (podľa zadania)
# ---------------------------------------------------------------------------
SENSORS = ("ACC", "GYR", "MAG")

SAMPLING_RATE_HZ = {
    "ACC": 400,
    "GYR": 400,
    "MAG": 100,
}

AXIS_NAMES = ("X", "Y", "Z")

# ---------------------------------------------------------------------------
# Triedy terénu (kód "x" v názve súboru SEN_x_y_z.csv) - podľa README databázy
# ---------------------------------------------------------------------------
TERRAIN_NAMES = {
    1: "concrete",       # Betón
    2: "grass",          # Tráva
    3: "pebbles",        # Kamienky
    4: "sand",           # Piesok
    5: "paving_stone",   # Dlažobný kameň
    6: "running_track",  # Syntetická bežecká dráha
}

# ---------------------------------------------------------------------------
# Rýchlosti (kód "y" v názve súboru) - podľa README databázy
# ---------------------------------------------------------------------------
SPEED_NAMES = {
    2: "pwm_86",   # 86 % PWM (pomalšia)
    3: "pwm_100",  # 100 % PWM (plná rýchlosť)
}

SESSIONS = tuple(range(1, 8))  # kód "z": session 1-7

# ---------------------------------------------------------------------------
# Orezanie / okno (podľa aktuálnej úlohy: 1 s úsek od 2 s do 3 s)
# Neskôr, pri sliding-window kroku podľa zadania, sa toto rozšíri na
# viacero okien s posunom 0.5 s cez celé meranie.
# ---------------------------------------------------------------------------
CROP_START_S = 2.0
CROP_END_S = 3.0

# Náhodnosť pre reprodukovateľnosť (rozdelenie na train/val/test, 5 opakovaní, ...)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Sliding window (podľa zadania: 1 s okno, posun 0.5 s, cez celé meranie)
# ---------------------------------------------------------------------------
WINDOW_S = 1.0
STEP_S = 0.5

# ---------------------------------------------------------------------------
# Rozdelenie na train/val/test (na úrovni meraní, nie okien!)
# Pre každý terén je 14 meraní (7 session x 2 rýchlosti) -> 10/2/2
# ---------------------------------------------------------------------------
N_TRAIN = 10
N_VAL = 2
N_TEST = 2

N_SPLITS = 5  # počet nezávislých náhodných rozdelení (zadanie: priemer z 5 opakovaní)

# Snímače použité na klasifikáciu (rovnaká fs => okná sa dajú priamo skombinovať)
ACTIVE_SENSORS = ("ACC", "GYR")

# ---------------------------------------------------------------------------
# Prahové hodnoty pre časové príznaky (features.py) - NZC, NSSC, WAMP
# Prevzaté ako štartovacia hodnota z demo kódu k predmetu (utils.zip).
# POZOR: ACC (rádovo jednotky m/s^2) a GYR (rádovo desiatky °/s) majú úplne
# inú škálu signálu, takže rovnaký prah pre oba snímače je len zjednodušenie -
# pri zdôvodňovaní parametrov v odovzdaní stojí za zváženie prah nastaviť
# zvlášť pre každý snímač (napr. ako % rozsahu signálu).
# ---------------------------------------------------------------------------
NZC_THRESHOLD = 1e-5
NSSC_THRESHOLD = 1e-4
WAMP_THRESHOLD = 1e-2