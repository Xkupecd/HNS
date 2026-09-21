# MLP - priebeh trénovania a vyhodnotenia

```text
Device: cuda

Porovnanie parametrov trénovania pre MLP(64, 32)-drop=0.3 (split_00):

  konfigurácia                                  VAL                                        TEST
  optimizer_name=sgdm, lr=0.01, batch_size=16, epochs=20 acc= 93.75 %  prec= 94.58 %  recall= 93.75 % acc= 91.67 %  prec= 93.95 %  recall= 91.67 %
  optimizer_name=sgdm, lr=0.001, batch_size=16, epochs=20 acc= 81.25 %  prec= 85.22 %  recall= 81.25 % acc= 82.29 %  prec= 88.41 %  recall= 82.29 %
  optimizer_name=adamw, lr=0.01, batch_size=16, epochs=20 acc= 94.79 %  prec= 95.28 %  recall= 94.79 % acc= 94.79 %  prec= 96.03 %  recall= 94.79 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=20 acc= 93.75 %  prec= 94.30 %  recall= 93.75 % acc= 91.67 %  prec= 94.44 %  recall= 91.67 %
  optimizer_name=adamw, lr=0.0001, batch_size=16, epochs=20 acc= 70.83 %  prec= 80.81 %  recall= 70.83 % acc= 81.25 %  prec= 87.55 %  recall= 81.25 %
  optimizer_name=adamw, lr=0.001, batch_size=32, epochs=20 acc= 94.79 %  prec= 95.21 %  recall= 94.79 % acc= 91.67 %  prec= 93.36 %  recall= 91.67 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=10 acc= 92.71 %  prec= 93.39 %  recall= 92.71 % acc= 90.62 %  prec= 93.46 %  recall= 90.62 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=15 acc= 94.79 %  prec= 95.41 %  recall= 94.79 % acc= 91.67 %  prec= 94.44 %  recall= 91.67 %

Najlepšie parametre trénovania: optimizer_name=adamw, lr=0.001, batch_size=16, epochs=15
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/hyperparam_training_curves.png

Porovnanie architektúr (split_00), parametre trénovania: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 15}

  architektúra                 VAL                                        TEST
  MLP(32,)-drop=0.3            acc= 93.75 %  prec= 94.30 %  recall= 93.75 % acc= 92.71 %  prec= 94.93 %  recall= 92.71 %
  MLP(64, 32)-drop=0.3         acc= 94.79 %  prec= 95.41 %  recall= 94.79 % acc= 91.67 %  prec= 94.44 %  recall= 91.67 %
  MLP(64, 32)-no-dropout       acc= 96.88 %  prec= 97.17 %  recall= 96.88 % acc= 90.62 %  prec= 93.16 %  recall= 90.62 %
  MLP(128, 64, 32)-drop=0.4    acc= 94.79 %  prec= 95.21 %  recall= 94.79 % acc= 91.67 %  prec= 94.44 %  recall= 91.67 %

Najlepšia architektúra: MLP(64, 32)-no-dropout
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/architecture_training_curves.png

======================================================================
Spustenie MLP(64, 32)-no-dropout na 5 rozdeleniach (parametre: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 15})
======================================================================
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/split_00_confusion.png
  split_00:  VAL  acc= 96.88 %  prec= 97.17 %  recall= 96.88 %   |   TEST  acc= 90.62 %  prec= 93.16 %  recall= 90.62 %   |   čas=0.29 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/split_01_confusion.png
  split_01:  VAL  acc= 94.79 %  prec= 95.52 %  recall= 94.79 %   |   TEST  acc= 93.75 %  prec= 94.71 %  recall= 93.75 %   |   čas=0.26 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/split_02_confusion.png
  split_02:  VAL  acc= 97.92 %  prec= 98.15 %  recall= 97.92 %   |   TEST  acc= 97.92 %  prec= 98.15 %  recall= 97.92 %   |   čas=0.31 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/split_03_confusion.png
  split_03:  VAL  acc= 92.71 %  prec= 93.40 %  recall= 92.71 %   |   TEST  acc= 94.79 %  prec= 95.69 %  recall= 94.79 %   |   čas=0.30 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/split_04_confusion.png
  split_04:  VAL  acc= 97.92 %  prec= 98.04 %  recall= 97.92 %   |   TEST  acc=100.00 %  prec=100.00 %  recall=100.00 %   |   čas=0.32 s
  Tabuľka behov uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/runs/runs.csv

MLP - validačné dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       92.71%   97.92%    96.04%    2.02%
  precision      93.40%   98.15%    96.45%    1.79%
  recall         92.71%   97.92%    96.04%    2.02%

MLP - testovacie dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       90.62%  100.00%    95.42%    3.27%
  precision      93.16%  100.00%    96.34%    2.44%
  recall         90.62%  100.00%    95.42%    3.27%
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/final_splits_training_curves.png
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/final_splits_accuracy.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/mlp_confusion_matrix_combined.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/mlp_confusion_matrix_combined_val.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/mlp/mlp_confusion_matrix.png
```
