"""Gradual-unfreezing ablation for M2: starts from the frozen-backbone M2
(only CBAM blocks + head trainable) and unfreezes layer4 -> layer3 ->
layer2 -> layer1 with the same schedule as M1's train_m1_gradual_unfreeze.py.

Usage:
    python -m src.M2_ResNet_CBAM.train_m2_gradual_unfreeze
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT.name != "src":
    _ROOT = _ROOT.parent
sys.path.append(str(_ROOT.parent))
from src import config
from src.engine import build_unfreeze_schedule
from src.experiment import run_experiment
from src.models.M2_CNN_ResNet50_CBAM import UNFREEZE_STAGES, build_model, unfreeze_stage


def main() -> None:
    schedule = build_unfreeze_schedule(UNFREEZE_STAGES, config.EPOCHS)
    run_experiment(
        "M2_cnn_cbam_gradual_unfreeze",
        results_root=config.M2_RESULTS_DIR,
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
        unfreeze_schedule=schedule,
        unfreeze_fn=unfreeze_stage,
    )


if __name__ == "__main__":
    main()
