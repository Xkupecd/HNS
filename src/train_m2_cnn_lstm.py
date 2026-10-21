"""Train and evaluate M2 (CNN + LSTM) per the assignment: 5 training runs,
training-curve plots, test-set ACC/Precision/Recall/F1 + confusion matrix,
and run statistics (mean/std over the 5 runs).

Usage:
    python -m src.train_m2_cnn_lstm
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.experiment import run_experiment
from src.models.M2_CNN_LSTM import build_head


def main() -> None:
    run_experiment("M2_cnn_lstm", results_root=config.M2_RESULTS_DIR, build_head_fn=build_head)


if __name__ == "__main__":
    main()
