"""Train and evaluate M2 (ResNet50 + CBAM, transfer learning, frozen backbone)
per the assignment: 3 training runs, training-curve plots, test-set
ACC/Precision/Recall/F1 + confusion matrix, and run statistics (mean/std
over the 3 runs).

Same setup as M1's train_m1.py (frozen pretrained backbone, augmentation,
identical hyperparameters) - only the CBAM blocks are added - so the two are
directly comparable.

Usage:
    python -m src.M2_ResNet_CBAM.train_m2
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.experiment import run_experiment
from src.models.M2_CNN_ResNet50_CBAM import build_model


def main() -> None:
    run_experiment(
        "M2_cnn_cbam",
        results_root=config.M2_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
    )


if __name__ == "__main__":
    main()
