"""Gradual-unfreezing configuration for M3: starts from the frozen-backbone
ConvNeXt-T (only the new classifier trainable) and unfreezes stages.3 ->
stages.2 -> stages.1 -> stages.0 with the same schedule as M1/M2 (one stage
every 6 epochs from epoch 7, see engine.build_unfreeze_schedule). The stem
stays frozen, like ResNet50's conv1.

Usage:
    python -m src.M3_ConvNeXt.train_m3_gradual_unfreeze
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
from src.models.M3_CNN_ConvNeXt_Tiny import (
    NORM_MEAN, NORM_STD, UNFREEZE_STAGES, build_model, unfreeze_stage,
)


def main() -> None:
    schedule = build_unfreeze_schedule(UNFREEZE_STAGES, config.EPOCHS)
    run_experiment(
        "M3_convnext_gradual_unfreeze",
        results_root=config.M3_RESULTS_DIR,
        # fully unfrozen ConvNeXt-T peaks at ~7.7 GiB at batch 128 (8 GB GPU);
        # gradient checkpointing brings it to ~2.3 GiB without changing the math
        build_model_fn=lambda num_classes: build_model(
            num_classes, freeze_backbone=True, grad_checkpointing=True),
        augment_train=True,
        unfreeze_schedule=schedule,
        unfreeze_fn=unfreeze_stage,
        mean=NORM_MEAN, std=NORM_STD,
    )


if __name__ == "__main__":
    main()
