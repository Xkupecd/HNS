# Zadanie 1 – Klasifikácia outdoor terénu (HNS)

Klasifikácia terénu (betón, tráva, kamienky, piesok, dlažobný kameň, bežecká
dráha) zo signálov akcelerometra a gyroskopu, pomocou SVM, MLP, LSTM a CNN
(porovnanie ručne extrahovaných príznakov vs. automatickej extrakcie cez
rekurentné/konvolučné vrstvy).

## Rýchly štart

```bash
# 1. Vytvor a aktivuj virtuálne prostredie
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Nainštaluj závislosti
pip install numpy pandas scikit-learn torch matplotlib

# 3. Priprav dáta (pozri nižšie "Dáta")

# 4. Spusti niektorý model (z priečinka src/)
cd src
python -m models.svm_model
python -m models.mlp_model
python -m models.lstm_model
python -m models.cnn_model
```

## Dáta

`data/raw/` a `data/processed/splits/` **nie sú v gite** (sú to len surové/
medzikrokové súbory, dajú sa kedykoľvek vygenerovať znova). V repozitári
naopak **sú** `data/processed/features/` a `data/processed/windows/` –
teda hotové vstupy pre tréning, takže po naklonovaní repa vieš rovno
spúšťať modely bez ďalšej prípravy.

Ak by si si chcel/a dáta vygenerovať odznova (napr. po zmene `config.py`):

1. Stiahni a rozbaľ `outdoor_terrain_classification_database.zip` do
   `data/raw/outdoor_terrain_classification_database/` (surové CSV merania
   sú v podpriečinku `Measurements/`).
2. Z priečinka `src/` postupne spusti:
   ```bash
   python splitting.py   # rozdelí merania 10/2/2 na train/val/test (5x, rôzne seedy)
   python windowing.py   # z rozdelení vyrobí sliding-window okná (.npy)
   python features.py    # z okien vypočíta časové príznaky pre SVM/MLP (.npy)
   ```
   CNN a LSTM si okná (`windowing.py`) načítajú priamo, príznaky (`features.py`)
   potrebujú iba SVM a MLP.

## Ako to celé funguje (pipeline)

```
surové CSV (data/raw/.../Measurements/SEN_x_y_z.csv)
        │  data_loading.py  – naparsuje meno súboru -> (snímač, terén, rýchlosť, session)
        ▼
splitting.py – rozdelí MERANIA (nie okná!) na train/val/test, 10/2/2 na terén,
               5x nezávisle (aby sa dal spraviť priemer z 5 behov)
        ▼
data/processed/splits/split_XX/{train,val,test}/<terén>/SEN_x_y_z.csv
        │  windowing.py – sliding window: 1 s okno, posun 0.5 s (400 vzoriek, krok 200)
        ▼
data/processed/windows/split_XX/{X,y,groups}_{train,val,test}.npy   ──► CNN, LSTM
        │  features.py – 9 príznakov na kanál (MAV, NZC, NSSC, RMS, WL, WAMP, MAX, MIN, PTP)
        ▼
data/processed/features/split_XX/{features,y}_{train,val,test}.npy ──► SVM, MLP
```

Kľúčové: rozdeľuje sa na úrovni **meraní**, nie okien – okná z jedného merania
sa navzájom prekrývajú, takže keby sa rozdeľovalo až po windowingu, dochádzalo
by k úniku informácie (leak) medzi train/val/test.

## Štruktúra repozitára

```
Zadanie1/
├── data/
│   ├── raw/                     # (negitované) surové CSV z databázy
│   └── processed/
│       ├── splits/               # (negitované) medzikrok - CSV rozdelené na train/val/test
│       ├── windows/               # sliding-window okná (.npy) - vstup pre CNN, LSTM
│       └── features/              # časové príznaky (.npy) - vstup pre SVM/MLP
├── src/
│   ├── config.py                 # konštanty: cesty, fs, terény, okno, seedy, prahy
│   ├── data_loading.py           # parsovanie názvov súborov, načítanie signálu
│   ├── splitting.py               # rozdelenie 10/2/2 na úrovni meraní (5x)
│   ├── windowing.py               # sliding window (1 s, posun 0.5 s)
│   ├── features.py                # MAV, NZC, NSSC, RMS, WL, WAMP, MAX, MIN, PTP
│   └── models/
│       ├── svm_model.py           # SVM nad príznakmi (baseline)
│       ├── mlp_model.py           # MLP nad rovnakými príznakmi ako SVM
│       ├── lstm_model.py          # LSTM priamo nad signálom (okná)
│       ├── cnn_model.py           # 1D CNN priamo nad signálom (okná)
│       └── evaluate.py            # accuracy/precision/recall, kontingenčná matica, grafy, log
├── results/
│   ├── svm/    # <model>_confusion_matrix.png, hyperparam_comparison.png,
│   ├── mlp/    # final_splits_accuracy.png, (pri MLP/LSTM/CNN aj) *_training_curves.png,
│   ├── lstm/   # a <model>_training_log.md - pozri "Spúšťanie modelov" nižšie
│   └── cnn/
├── HNS_Zadanie1.pdf                # zadanie
└── utils.zip                       # demo kód z predmetu (referencia k features.py)
```

**Poznámka:** generický `train.py`/`experiments/` runner a `requirements.txt`
zatiaľ nie sú hotové – ak na tom niekto pracuje, dajte vedieť nech sa nebijeme
v tom istom súbore. Zatiaľ každý model (`svm_model.py`, `mlp_model.py`,
`lstm_model.py`, `cnn_model.py`) sa spúšťa priamo ako skript a má vlastný
tréningový loop + `if __name__ == "__main__"` blok, ktorý spustí všetkých
5 splitov a vypíše priemer/min/max metrík.

## Spúšťanie modelov

Spúšťaj vždy z priečinka `src/` (kvôli importom medzi modulmi):

```bash
cd src
python -m models.svm_model     # alebo: python models/svm_model.py
python -m models.mlp_model
python -m models.lstm_model
python -m models.cnn_model
```

Keď sú všetky modely dobehnuté, súhrnné porovnanie sa poskladá z ich logov:

```bash
python -m models.compare_models
```

Nič sa pritom netrénuje – skript číta `results/<model>/<model>_training_log.md`
a uloží do `results/comparison/` tabuľku (`model_comparison.md`), to isté v CSV
a stĺpcový graf `model_comparison.png` (val vs test, úsečky = min-max).

Výsledky sa ukladajú priamo do `results/<model>/` (bez ďalších podpriečinkov):
- `<model>_confusion_matrix.png` – kontingenčná matica najlepšieho modelu na testovacích dátach,
- `hyperparam_training_curves.png` a `architecture_training_curves.png` (MLP/LSTM/CNN) –
  porovnanie priebehov trénovania (train/val loss a accuracy podľa epoch) pri rôznych
  tréningových hyperparametroch, resp. architektúrach (SVM nemá epochy, tak namiesto toho
  má `hyperparam_comparison.png` – stĺpcový graf úspešnosti podľa nastavenia),
- `final_splits_training_curves.png` (MLP/LSTM/CNN) a `final_splits_accuracy.png` (všetky) –
  priebeh/úspešnosť naprieč finálnymi 5 behmi (splitmi) najlepšej konfigurácie,
- `<model>_training_log.md` – kompletný textový výpis z konzoly (všetky porovnávacie
  tabuľky, priebeh 5 behov, súhrn min/max/priemer) uložený ako .md, presne pre potreby
  dokumentácie.
