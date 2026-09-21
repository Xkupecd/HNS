# LSTM - priebeh trénovania a vyhodnotenia

```text
Device: cuda


Porovnanie vstupnej reprezentácie - dĺžka sekvencie a pooling (priemer z 5 splitov, bez augmentácie)

  vstupná reprezentácia                                      VAL acc (priemer +- std)
  frame= 1 (400 krokov), pool=last                            57.50 % +- 11.70 %
  frame= 1 (400 krokov), pool=meanmax                         89.79 % +-  2.90 %
  frame= 5 ( 80 krokov), pool=last                            80.83 % +-  2.52 %
  frame= 5 ( 80 krokov), pool=mean                            83.54 % +-  3.81 %
  frame= 5 ( 80 krokov), pool=meanmax                         88.54 % +-  4.32 %
  frame=10 ( 40 krokov), pool=meanmax                         84.17 % +-  4.34 %
  frame=20 ( 20 krokov), pool=meanmax                         76.67 % +-  2.52 %
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/input_representation_training_curves.png

Najlepšia vstupná reprezentácia: frame_size=1, pool=meanmax

Vplyv augmentácie trénovacích dát (LSTM(h=128,L=1)-bidir-frame=1-meanmax-drop=0.3, priemer z 5 splitov)

  augmentácia                                                VAL acc (priemer +- std)
  bez augmentácie                                             89.79 % +-  2.90 %
  posun                                                       91.88 % +-  4.81 %
  amp=0.1+šum=0.05                                            90.83 % +-  3.32 %
  posun+amp=0.1+šum=0.05                                      92.29 % +-  5.42 %
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/augmentation_training_curves.png

Najlepšia augmentácia: posun+amp=0.1+šum=0.05

Porovnanie parametrov trénovania pre LSTM(h=128,L=1)-bidir-frame=1-meanmax-drop=0.3 (priemer validačnej úspešnosti z 5 splitov):

  konfigurácia                                               VAL acc (priemer +- std)
  optimizer_name=sgdm, lr=0.01, batch_size=16, epochs=60      85.42 % +-  3.67 %
  optimizer_name=sgdm, lr=0.001, batch_size=16, epochs=60     47.29 % +-  4.45 %
  optimizer_name=adamw, lr=0.01, batch_size=16, epochs=60     89.17 % +-  4.55 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=60    92.29 % +-  5.42 %
  optimizer_name=adamw, lr=0.0001, batch_size=16, epochs=60   73.75 % +-  1.21 %
  optimizer_name=adamw, lr=0.001, batch_size=32, epochs=60    90.62 % +-  4.52 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=30    87.50 % +-  2.87 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=120   92.71 % +-  3.95 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=200   92.71 % +-  2.64 %
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/hyperparam_training_curves.png

Najlepšie parametre trénovania: optimizer_name=adamw, lr=0.001, batch_size=16, epochs=200

Porovnanie architektúr (priemer z 5 splitov), parametre trénovania: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 200}

  architektúra                                               VAL acc (priemer +- std)
  LSTM(h=128,L=1)-bidir-frame=1-meanmax-drop=0.3              92.71 % +-  2.64 %
  LSTM(h=128,L=1)-bidir-frame=1-meanmax-no-dropout            95.21 % +-  3.40 %
  LSTM(h=128,L=1)-bidir-frame=1-meanmax-drop=0.5              92.08 % +-  3.88 %
  LSTM(h=64,L=1)-bidir-frame=1-meanmax-drop=0.3               94.58 % +-  3.39 %
  LSTM(h=128,L=2)-bidir-frame=1-meanmax-drop=0.3              94.38 % +-  3.27 %
  LSTM(h=128,L=1)-unidir-frame=1-meanmax-drop=0.3             93.54 % +-  3.57 %
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/architecture_training_curves.png

Najlepšia architektúra: LSTM(h=128,L=1)-bidir-frame=1-meanmax-no-dropout

======================================================================
Spustenie LSTM(h=128,L=1)-bidir-frame=1-meanmax-no-dropout na 5 rozdeleniach (parametre: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 200, 'aug': 'posun+amp=0.1+šum=0.05'})
======================================================================
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/split_00_confusion.png
  split_00:  VAL  acc= 92.71 %  prec= 93.29 %  recall= 92.71 %   |   TEST  acc= 96.88 %  prec= 97.06 %  recall= 96.88 %   |   čas=28.53 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/split_01_confusion.png
  split_01:  VAL  acc= 90.62 %  prec= 92.42 %  recall= 90.62 %   |   TEST  acc= 96.88 %  prec= 97.17 %  recall= 96.88 %   |   čas=28.85 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/split_02_confusion.png
  split_02:  VAL  acc= 97.92 %  prec= 98.15 %  recall= 97.92 %   |   TEST  acc= 97.92 %  prec= 97.98 %  recall= 97.92 %   |   čas=30.47 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/split_03_confusion.png
  split_03:  VAL  acc= 94.79 %  prec= 95.05 %  recall= 94.79 %   |   TEST  acc= 95.83 %  prec= 95.96 %  recall= 95.83 %   |   čas=42.95 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/split_04_confusion.png
  split_04:  VAL  acc=100.00 %  prec=100.00 %  recall=100.00 %   |   TEST  acc= 90.62 %  prec= 91.38 %  recall= 90.62 %   |   čas=28.56 s
  Tabuľka behov uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/runs/runs.csv

LSTM - validačné dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       90.62%  100.00%    95.21%    3.40%
  precision      92.42%  100.00%    95.78%    2.88%
  recall         90.62%  100.00%    95.21%    3.40%

LSTM - testovacie dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       90.62%   97.92%    95.62%    2.59%
  precision      91.38%   97.98%    95.91%    2.35%
  recall         90.62%   97.92%    95.62%    2.59%
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/final_splits_training_curves.png
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/final_splits_accuracy.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/lstm_confusion_matrix_combined.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/lstm_confusion_matrix_combined_val.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/lstm/lstm_confusion_matrix.png
```
