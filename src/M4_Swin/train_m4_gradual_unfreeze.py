"""Gradual-unfreezing configuration for M4: starts from the frozen-backbone
Swin-T (only the new classifier trainable) and unfreezes layers.3 ->
layers.2 -> layers.1 -> layers.0 with the same schedule as M1-M3 (one stage
every 6 epochs from epoch 7, see engine.build_unfreeze_schedule). The patch
embedding stays frozen, like ResNet50's conv1.

Usage:
    python -m src.M4_Swin.train_m4_gradual_unfreeze
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
from src.models.M4_ViT_Swin_Tiny import (
    NORM_MEAN, NORM_STD, UNFREEZE_STAGES, build_model, unfreeze_stage,
)


def main() -> None:
    schedule = build_unfreeze_schedule(UNFREEZE_STAGES, config.EPOCHS)
    run_experiment(
        "M4_swin_gradual_unfreeze",
        results_root=config.M4_RESULTS_DIR,
        # fully unfrozen Swin-T needs ~8.7 GiB at batch 128 (more than the 8 GB
        # GPU); gradient checkpointing brings it to ~2.8 GiB without changing the math
        build_model_fn=lambda num_classes: build_model(
            num_classes, freeze_backbone=True, grad_checkpointing=True),
        augment_train=True,
        unfreeze_schedule=schedule,
        unfreeze_fn=unfreeze_stage,
        mean=NORM_MEAN, std=NORM_STD,
    )


if __name__ == "__main__":
    main()
