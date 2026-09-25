"""Gradual-unfreezing ablation (assignment: "aspoň pre jeden model realizujte
postupné rozmrazovanie vrstiev"): starts from the same frozen-backbone M1 as
the baseline, then progressively unfreezes layer4 -> layer3 -> layer2 ->
layer1 over the course of training (see engine.build_unfreeze_schedule),
instead of keeping the backbone frozen for all `epochs`. Same epoch budget
and other hyperparameters as train_m1.py, so it's directly comparable to the
M1_cnn baseline in results_summary.md.

Usage:
    python -m src.M1_ResNet50_Train.train_m1_gradual_unfreeze
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
from src.models.M1_CNN_ResNet50 import UNFREEZE_STAGES, build_model, unfreeze_stage


def main() -> None:
    schedule = build_unfreeze_schedule(UNFREEZE_STAGES, config.EPOCHS)
    run_experiment(
        "M1_cnn_gradual_unfreeze",
        build_model_fn=lambda num_classes: build_model(num_classes, freeze_backbone=True),
        augment_train=True,
        unfreeze_schedule=schedule,
        unfreeze_fn=unfreeze_stage,
    )


if __name__ == "__main__":
    main()
