# Porovnanie modelov (SVM / MLP / LSTM / CNN)

Zdroj: `results/<model>/<model>_training_log.md`, priemer z 5 nezávislých rozdelení.

## Súhrn

| Model | Vstup | VAL accuracy | TEST accuracy | TEST precision | TEST recall |
|---|---|---|---|---|---|
| **SVM** | 54 ručných príznakov | 96.46 % ± 2.52 | 96.25 % ± 3.88 | 96.99 % ± 2.63 | 96.25 % ± 3.88 |
| **LSTM** | surový signál (400 x 6) | 95.21 % ± 3.40 | 95.63 % ± 2.59 | 95.91 % ± 2.35 | 95.63 % ± 2.59 |
| **MLP** | 54 ručných príznakov | 96.04 % ± 2.02 | 95.42 % ± 3.27 | 96.34 % ± 2.44 | 95.42 % ± 3.27 |
| **CNN** | surový signál (400 x 6) | 96.46 % ± 3.46 | 95.21 % ± 5.34 | 96.30 % ± 3.91 | 95.21 % ± 5.34 |

## Rozptyl naprieč behmi (testovacie dáta)

| Model | min | max | priemer | std | Víťazná konfigurácia |
|---|---|---|---|---|---|
| **SVM** | 88.54 % | 98.96 % | 96.25 % | 3.88 | `kernel=linear, C=1.0` |
| **LSTM** | 90.62 % | 97.92 % | 95.63 % | 2.59 | `LSTM(h=128,L=1)-bidir-frame=1-meanmax-no-dropout` |
| **MLP** | 90.62 % | 100.00 % | 95.42 % | 3.27 | `MLP(64, 32)-no-dropout` |
| **CNN** | 85.42 % | 100.00 % | 95.21 % | 5.34 | `CNN(16, 32)-gap-bn-drop=0.3` |

## Accuracy po jednotlivých rozdeleniach (testovacie dáta)

| Model | split_00 | split_01 | split_02 | split_03 | split_04 |
|---|---|---|---|---|---|
| **SVM** | 88.54 % | 97.92 % | 97.92 % | 97.92 % | 98.96 % |
| **LSTM** | 96.88 % | 96.88 % | 97.92 % | 95.83 % | 90.62 % |
| **MLP** | 90.62 % | 93.75 % | 97.92 % | 94.79 % | 100.00 % |
| **CNN** | 93.75 % | 97.92 % | 98.96 % | 85.42 % | 100.00 % |
