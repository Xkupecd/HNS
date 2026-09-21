# SVM - priebeh trénovania a vyhodnotenia

```text
Porovnanie nastavení SVM (split_00):

  konfigurácia                     VAL                                        TEST
  kernel=linear, C=1.0             acc= 96.88 %  prec= 97.17 %  recall= 96.88 % acc= 88.54 %  prec= 91.78 %  recall= 88.54 %
  kernel=linear, C=10.0            acc= 95.83 %  prec= 96.39 %  recall= 95.83 % acc= 87.50 %  prec= 91.37 %  recall= 87.50 %
  kernel=rbf, C=1.0, gamma=scale   acc= 96.88 %  prec= 97.06 %  recall= 96.88 % acc= 93.75 %  prec= 95.45 %  recall= 93.75 %
  kernel=rbf, C=10.0, gamma=scale  acc= 96.88 %  prec= 97.06 %  recall= 96.88 % acc= 95.83 %  prec= 96.26 %  recall= 95.83 %
  kernel=rbf, C=100.0, gamma=scale acc= 95.83 %  prec= 96.19 %  recall= 95.83 % acc= 93.75 %  prec= 94.92 %  recall= 93.75 %
  kernel=poly, C=10.0              acc= 94.79 %  prec= 95.32 %  recall= 94.79 % acc= 88.54 %  prec= 91.61 %  recall= 88.54 %

Najlepšia konfigurácia podľa validačnej úspešnosti: kernel=linear, C=1.0
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/hyperparam_comparison.png

======================================================================
Spustenie najlepšej konfigurácie na 5 rozdeleniach
======================================================================
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/split_00_confusion.png
  split_00:  VAL  acc= 96.88 %  prec= 97.17 %  recall= 96.88 %   |   TEST  acc= 88.54 %  prec= 91.78 %  recall= 88.54 %   |   čas=0.00 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/split_01_confusion.png
  split_01:  VAL  acc= 91.67 %  prec= 92.14 %  recall= 91.67 %   |   TEST  acc= 97.92 %  prec= 98.04 %  recall= 97.92 %   |   čas=0.00 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/split_02_confusion.png
  split_02:  VAL  acc= 97.92 %  prec= 98.04 %  recall= 97.92 %   |   TEST  acc= 97.92 %  prec= 97.98 %  recall= 97.92 %   |   čas=0.00 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/split_03_confusion.png
  split_03:  VAL  acc= 96.88 %  prec= 96.94 %  recall= 96.88 %   |   TEST  acc= 97.92 %  prec= 98.15 %  recall= 97.92 %   |   čas=0.00 s
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/split_04_confusion.png
  split_04:  VAL  acc= 98.96 %  prec= 99.02 %  recall= 98.96 %   |   TEST  acc= 98.96 %  prec= 99.02 %  recall= 98.96 %   |   čas=0.00 s
  Tabuľka behov uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/runs/runs.csv

SVM - validačné dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       91.67%   98.96%    96.46%    2.52%
  precision      92.14%   99.02%    96.66%    2.37%
  recall         91.67%   98.96%    96.46%    2.52%

SVM - testovacie dáta (z 5 behov)
  metrika           min      max   priemer      std
  accuracy       88.54%   98.96%    96.25%    3.88%
  precision      91.78%   99.02%    96.99%    2.63%
  recall         88.54%   98.96%    96.25%    3.88%
  Graf úspešnosti uložený: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/final_splits_accuracy.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/svm_confusion_matrix_combined.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/svm_confusion_matrix_combined_val.png
  Kontingenčná matica uložená: /home/dominik/5.SEMESTER/HNS/Zadanie1/results/svm/svm_confusion_matrix.png
```
