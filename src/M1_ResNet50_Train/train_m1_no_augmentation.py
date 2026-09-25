"""Augmentation ablation (assignment: "aspoň pre jeden model ukážte vplyv
augmentácie"): trains M1 with the exact same setup as train_m1.py but with
augmentation switched off, so it's directly comparable to the M1_cnn
baseline (which trains with augmentation) in results_summary.md.

Usage:
    python -m src.M1_ResNet50_Train.train_m1_no_augmentation
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
        "M1_cnn_no_augmentation",
        results_root=config.M1_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=False,
    )


if __name__ == "__main__":
    main()
