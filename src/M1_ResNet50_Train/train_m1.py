"""Train and evaluate M1 (ResNet50, transfer learning, frozen backbone) per
the assignment: 3 training runs, training-curve plots, test-set ACC/Precision/
Recall/F1 + confusion matrix, and run statistics (mean/std over the 3 runs).

This is the baseline M1 configuration used for the M1-M5 model comparison.
See train_m1_no_augmentation.py and train_m1_gradual_unfreeze.py for the
assignment's separate augmentation / gradual-unfreezing ablations.

Usage:
    python -m src.M1_ResNet50_Train.train_m1
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.experiment import run_experiment
from src.models.M1_CNN_ResNet50 import build_model


def main() -> None:
    run_experiment(
        "M1_cnn",
        results_root=config.M1_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
    )


if __name__ == "__main__":
    main()
