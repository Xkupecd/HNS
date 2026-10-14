"""Train and evaluate M5 (CoAtNet-0 hybrid, transfer learning, frozen
backbone) with the same procedure and hyperparameters as M1-M4: 3 training
runs, training-curve plots, test-set ACC/Precision/Recall/F1 + confusion
matrix and run statistics (mean/std over the 3 runs).

This is the baseline M5 configuration (directly comparable to the frozen
M1-M4). See train_m5_gradual_unfreeze.py for the fine-tuned variant.

Usage:
    python -m src.M5_CoAtNet.train_m5
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.experiment import run_experiment
from src.models.M5_Hybrid_CoAtNet_0 import NORM_MEAN, NORM_STD, build_model


def main() -> None:
    run_experiment(
        "M5_coatnet",
        results_root=config.M5_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
        mean=NORM_MEAN, std=NORM_STD,
    )


if __name__ == "__main__":
    main()
