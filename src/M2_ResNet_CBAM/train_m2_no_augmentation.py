"""Augmentation ablation for M2: same setup as train_m2.py with augmentation
switched off, mirroring M1's train_m1_no_augmentation.py.

Usage:
    python -m src.M2_ResNet_CBAM.train_m2_no_augmentation
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
        "M2_cnn_cbam_no_augmentation",
        results_root=config.M2_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=False,
    )


if __name__ == "__main__":
    main()
