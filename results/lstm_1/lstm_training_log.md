# LSTM - priebeh trénovania a vyhodnotenia

```text
Device: cuda

Porovnanie parametrov trénovania pre LSTM(h=64,L=1)-unidir-drop=0.3 (priemer validačnej úspešnosti z 5 splitov):

  konfigurácia                                         VAL acc (priemer +- std)
  optimizer_name=sgdm, lr=0.01, batch_size=16, epochs=20  30.21 % +-  4.32 %
  optimizer_name=sgdm, lr=0.001, batch_size=16, epochs=20  19.38 % +-  1.41 %
  optimizer_name=adamw, lr=0.01, batch_size=16, epochs=20  54.79 % +-  5.25 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=20  41.04 % +- 13.58 %
  optimizer_name=adamw, lr=0.0001, batch_size=16, epochs=20  27.29 % +-  3.45 %
  optimizer_name=adamw, lr=0.001, batch_size=32, epochs=20  36.88 % +-  3.52 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=10  32.92 % +-  6.64 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=15  38.75 % +-  4.54 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=30  46.46 % +-  8.88 %

Najlepšie parametre trénovania: optimizer_name=adamw, lr=0.01, batch_size=16, epochs=20
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/hyperparam_training_curves.png

Porovnanie architektúr (priemer z 5 splitov), parametre trénovania: {'optimizer_name': 'adamw', 'lr': 0.01, 'batch_size': 16, 'epochs': 20}

  architektúra                       VAL acc (priemer +- std)
  LSTM(h=64,L=1)-unidir-drop=0.3      54.79 % +-  5.25 %
  LSTM(h=64,L=1)-unidir-no-dropout    62.50 % +-  5.47 %
  LSTM(h=64,L=2)-unidir-drop=0.3      20.00 % +-  6.16 %
  LSTM(h=128,L=1)-unidir-drop=0.3     61.67 % +- 17.26 %
  LSTM(h=64,L=1)-bidir-drop=0.3       50.83 % +-  9.42 %

Najlepšia architektúra: LSTM(h=64,L=1)-unidir-no-dropout
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/architecture_training_curves.png

======================================================================
Spustenie LSTM(h=64,L=1)-unidir-no-dropout na 5 rozdeleniach (parametre: {'optimizer_name': 'adamw', 'lr': 0.01, 'batch_size': 16, 'epochs': 20})
======================================================================
  split_00:  VAL  acc= 66.67 %  prec= 72.77 %  recall= 66.67 %   |   TEST  acc= 62.50 %  prec= 68.64 %  recall= 62.50 %
  split_01:  VAL  acc= 69.79 %  prec= 69.66 %  recall= 69.79 %   |   TEST  acc= 61.46 %  prec= 61.12 %  recall= 61.46 %
  split_02:  VAL  acc= 59.38 %  prec= 60.85 %  recall= 59.38 %   |   TEST  acc= 53.12 %  prec= 56.78 %  recall= 53.12 %
  split_03:  VAL  acc= 54.17 %  prec= 56.59 %  recall= 54.17 %   |   TEST  acc= 57.29 %  prec= 62.71 %  recall= 57.29 %
  split_04:  VAL  acc= 62.50 %  prec= 68.27 %  recall= 62.50 %   |   TEST  acc= 56.25 %  prec= 51.50 %  recall= 56.25 %

LSTM - validačné dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       54.17%   69.79%    62.50%    5.47%
  precision      56.59%   72.77%    65.63%    5.98%
  recall         54.17%   69.79%    62.50%    5.47%

LSTM - testovacie dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       53.12%   62.50%    58.13%    3.45%
  precision      51.50%   68.64%    60.15%    5.76%
  recall         53.12%   62.50%    58.13%    3.45%
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/final_splits_training_curves.png
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/final_splits_accuracy.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/lstm_confusion_matrix.png
```
