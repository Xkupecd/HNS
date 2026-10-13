"""Shared training-experiment loop: N runs, per-run + best-run plots, test
metrics, run statistics. Used by all train_*.py scripts (baseline models and
the augmentation / gradual-unfreezing ablations) so the procedure is
identical everywhere except for the model builder and the a few knobs.
"""
import json
import random
import time
from datetime import timedelta

import numpy as np
import torch

from src import config
from src.data_preprocessing.dataset import build_dataloaders
from src.engine import fit, predict
from src.evaluation import compute_metrics, save_confusion_matrix, summarize_runs
from src.logging_utils import tee_stdout_to_file
from src.plots import plot_confusion_matrix, plot_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(
    model_name: str,
    results_root,
    build_model_fn,
    augment_train: bool = True,
    unfreeze_schedule: dict[int, list[str]] | None = None,
    unfreeze_fn=None,
    num_runs: int = config.NUM_RUNS,
    epochs: int = config.EPOCHS,
    mean: tuple = config.IMAGENET_MEAN,
    std: tuple = config.IMAGENET_STD,
) -> dict:
    """Trains `build_model_fn` `num_runs` times and writes everything to
    <results_root>/<model_name>/: per-run history/checkpoint/confusion matrix/plots,
    the best run's plots, training_log.txt, and summary.json.

    `mean`/`std` set the input normalization; they must match the statistics
    the backbone was pretrained with (ImageNet for most models)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = results_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout_to_file(model_dir / "training_log.txt")
    print(f"device={device} normalization mean={mean} std={std}")

    loaders = build_dataloaders(augment_train=augment_train, mean=mean, std=std)
    class_names = config.CLASSES

    total_start = time.perf_counter()
    run_metrics = []
    for run_idx in range(1, num_runs + 1):
        run_dir = model_dir / f"run{run_idx}"
        set_seed(config.SEED + run_idx)
        run_start = time.perf_counter()

        model = build_model_fn(num_classes=len(class_names)).to(device)
        fit(
            model, loaders,
            epochs=epochs, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
            device=device, run_dir=run_dir,
            unfreeze_schedule=unfreeze_schedule, unfreeze_fn=unfreeze_fn,
        )
        plot_training_curves(
            run_dir / "history.csv",
            run_dir / "training_curves.png",
            title=f"{model_name} - run{run_idx}",
        )

        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        y_true, y_pred = predict(model, loaders["test"], device)
        metrics = compute_metrics(y_true, y_pred, class_names)
        save_confusion_matrix(metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.csv")
        plot_confusion_matrix(
            metrics["confusion_matrix"], class_names,
            run_dir / "confusion_matrix.png",
            title=f"{model_name} - run{run_idx} - confusion matrix",
        )

        run_elapsed = timedelta(seconds=round(time.perf_counter() - run_start))
        print(f"[{model_name} run{run_idx}] test_acc={metrics['acc']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
              f"f1={metrics['f1']:.4f} duration={run_elapsed}")
        run_metrics.append(metrics)

    total_elapsed = timedelta(seconds=round(time.perf_counter() - total_start))
    summary = summarize_runs(run_metrics)
    summary["per_run_acc"] = [m["acc"] for m in run_metrics]
    summary["total_duration"] = str(total_elapsed)
    print(f"[{model_name}] summary: {summary}")
    print(f"[{model_name}] total training duration: {total_elapsed}")

    with (model_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    best_run_idx = int(np.argmax(summary["per_run_acc"])) + 1
    plot_training_curves(
        model_dir / f"run{best_run_idx}" / "history.csv",
        model_dir / "best_run_training_curves.png",
        title=f"{model_name} - best run (run{best_run_idx})",
    )
    plot_confusion_matrix(
        run_metrics[best_run_idx - 1]["confusion_matrix"], class_names,
        model_dir / "best_run_confusion_matrix.png",
        title=f"{model_name} - best run (run{best_run_idx}) - confusion matrix",
    )

    return summary
