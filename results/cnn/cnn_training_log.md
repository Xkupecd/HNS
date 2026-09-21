# CNN - priebeh trénovania a vyhodnotenia

```text
Device: cuda

Porovnanie parametrov trénovania pre CNN(16, 32)-gap-bn-drop=0.3 (priemer validačnej úspešnosti z 5 splitov):

  konfigurácia                                         VAL acc (priemer +- std)
  optimizer_name=sgdm, lr=0.01, batch_size=16, epochs=20  81.46 % +-  3.63 %
  optimizer_name=sgdm, lr=0.001, batch_size=16, epochs=20  69.58 % +-  7.94 %
  optimizer_name=adamw, lr=0.01, batch_size=16, epochs=20  86.04 % +-  7.29 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=20  91.88 % +-  1.67 %
  optimizer_name=adamw, lr=0.0001, batch_size=16, epochs=20  72.08 % +-  6.54 %
  optimizer_name=adamw, lr=0.001, batch_size=32, epochs=20  92.50 % +-  4.03 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=10  88.33 % +-  6.60 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=15  89.38 % +-  4.99 %
  optimizer_name=adamw, lr=0.001, batch_size=16, epochs=30  96.46 % +-  3.46 %

Najlepšie parametre trénovania: optimizer_name=adamw, lr=0.001, batch_size=16, epochs=30
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/hyperparam_training_curves.png

Porovnanie architektúr (priemer z 5 splitov), parametre trénovania: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 30}

  architektúra                       VAL acc (priemer +- std)
  CNN(16, 32)-gap-bn-drop=0.3         96.46 % +-  3.46 %
  CNN(16, 32)-gap-bn-no-dropout       95.62 % +-  3.75 %
  CNN(16, 32)-gap-no-bn-drop=0.3      94.38 % +-  1.56 %
  CNN(16, 32)-flatten-bn-drop=0.3     88.54 % +-  2.72 %
  CNN(32, 64, 64)-gap-bn-drop=0.3     95.62 % +-  5.29 %

Najlepšia architektúra: CNN(16, 32)-gap-bn-drop=0.3
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/architecture_training_curves.png

======================================================================
Spustenie CNN(16, 32)-gap-bn-drop=0.3 na 5 rozdeleniach (parametre: {'optimizer_name': 'adamw', 'lr': 0.001, 'batch_size': 16, 'epochs': 30})
======================================================================
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/split_00_confusion.png
  split_00:  VAL  acc=100.00 %  prec=100.00 %  recall=100.00 %   |   TEST  acc= 93.75 %  prec= 95.45 %  recall= 93.75 %   |   čas=9.75 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/split_01_confusion.png
  split_01:  VAL  acc= 90.62 %  prec= 94.00 %  recall= 90.62 %   |   TEST  acc= 97.92 %  prec= 97.92 %  recall= 97.92 %   |   čas=9.73 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/split_02_confusion.png
  split_02:  VAL  acc= 95.83 %  prec= 96.67 %  recall= 95.83 %   |   TEST  acc= 98.96 %  prec= 99.02 %  recall= 98.96 %   |   čas=9.54 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/split_03_confusion.png
  split_03:  VAL  acc= 95.83 %  prec= 96.39 %  recall= 95.83 %   |   TEST  acc= 85.42 %  prec= 89.09 %  recall= 85.42 %   |   čas=9.50 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/split_04_confusion.png
  split_04:  VAL  acc=100.00 %  prec=100.00 %  recall=100.00 %   |   TEST  acc=100.00 %  prec=100.00 %  recall=100.00 %   |   čas=9.55 s
  Tabuľka behov uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/runs/runs.csv

CNN - validačné dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       90.62%  100.00%    96.46%    3.46%
  precision      94.00%  100.00%    97.41%    2.31%
  recall         90.62%  100.00%    96.46%    3.46%

CNN - testovacie dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       85.42%  100.00%    95.21%    5.34%
  precision      89.09%  100.00%    96.30%    3.91%
  recall         85.42%  100.00%    95.21%    5.34%
  Priebeh trénovania uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/final_splits_training_curves.png
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/final_splits_accuracy.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/cnn_confusion_matrix_combined.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/cnn_confusion_matrix_combined_val.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/cnn/cnn_confusion_matrix.png
```
