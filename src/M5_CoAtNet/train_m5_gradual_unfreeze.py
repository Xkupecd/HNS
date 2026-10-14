"""Gradual-unfreezing configuration for M5: starts from the frozen-backbone
CoAtNet-0 (only the new classifier trainable) and unfreezes stages.3 ->
stages.2 (transformer) -> stages.1 -> stages.0 (MBConv) with the same
schedule as M1-M4 (one stage every 6 epochs from epoch 7, see
engine.build_unfreeze_schedule). The stem stays frozen, like ResNet50's conv1.

Usage:
    python -m src.M5_CoAtNet.train_m5_gradual_unfreeze
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
from src.models.M5_Hybrid_CoAtNet_0 import (
    NORM_MEAN, NORM_STD, UNFREEZE_STAGES, build_model, unfreeze_stage,
)


def main() -> None:
    schedule = build_unfreeze_schedule(UNFREEZE_STAGES, config.EPOCHS)
    run_experiment(
        "M5_coatnet_gradual_unfreeze",
        results_root=config.M5_RESULTS_DIR,
        # fully unfrozen CoAtNet-0 needs ~10.6 GiB at batch 128 (more than the 8 GB
        # GPU); gradient checkpointing brings it to ~3.3 GiB without changing the math
        build_model_fn=lambda num_classes: build_model(
            num_classes, freeze_backbone=True, grad_checkpointing=True),
        augment_train=True,
        unfreeze_schedule=schedule,
        unfreeze_fn=unfreeze_stage,
        mean=NORM_MEAN, std=NORM_STD,
    )


if __name__ == "__main__":
    main()
