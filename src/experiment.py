"""Shared training-experiment loop for the temporal models (M2, M3):
N runs, per-run + best-run plots, test metrics, run statistics.

The models are trained on the cached frozen-CNN features (src/features.py);
best.pt therefore holds only the temporal head, which is loaded into
VideoClassifier.head for camera inference and GradCAM.
"""
import json
import time
from datetime import timedelta

import numpy as np
import torch

from src import config
from src.engine import fit, predict, set_seed
from src.evaluation import compute_metrics, save_confusion_matrix, summarize_runs
from src.features import build_feature_loaders
from src.logging_utils import tee_stdout_to_file
from src.plots import plot_confusion_matrix, plot_training_curves


def save_run_results(metrics: dict, class_names: list[str], run_dir, title: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.png",
                          title=f"{title} - confusion matrix")
    with (run_dir / "test_metrics.json").open("w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)


def finish_experiment(model_name: str, model_dir, run_metrics: list[dict], total_start: float) -> dict:
    """Writes summary.json and the best run's confusion matrix (shared with M1)."""
    total_elapsed = timedelta(seconds=round(time.perf_counter() - total_start))
    summary = summarize_runs(run_metrics)
    summary["per_run_acc"] = [m["acc"] for m in run_metrics]
    summary["best_run"] = int(np.argmax(summary["per_run_acc"])) + 1
    summary["total_duration"] = str(total_elapsed)
    print(f"[{model_name}] summary: {json.dumps(summary, indent=2)}")
    print(f"[{model_name}] total training duration: {total_elapsed}")

    with (model_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    best = summary["best_run"]
    plot_confusion_matrix(
        run_metrics[best - 1]["confusion_matrix"], config.CLASSES,
        model_dir / "best_run_confusion_matrix.png",
        title=f"{model_name} - best run (run{best}) - confusion matrix",
    )
    return summary


def run_experiment(
    model_name: str,
    results_root,
    build_head_fn,
    num_runs: int = config.NUM_RUNS,
    epochs: int = config.EPOCHS,
) -> dict:
    """Trains `build_head_fn` `num_runs` times and writes everything to
    <results_root>/: per-run history/checkpoint/confusion matrix/plots,
    the best run's plots, training_log.txt, and summary.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = results_root
    model_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout_to_file(model_dir / "training_log.txt")
    print(f"device={device} backbone={config.BACKBONE} (frozen, cached features) clip_len={config.CLIP_LEN} "
          f"batch={config.BATCH_SIZE} epochs={epochs} lr={config.LR} weight_decay={config.WEIGHT_DECAY}")

    class_names = config.CLASSES

    total_start = time.perf_counter()
    run_metrics = []
    for run_idx in range(1, num_runs + 1):
        run_dir = model_dir / f"run{run_idx}"
        set_seed(config.SEED + run_idx)
        run_start = time.perf_counter()
        loaders = build_feature_loaders()

        model = build_head_fn(num_classes=len(class_names)).to(device)
        if run_idx == 1:
            print(f"[{model_name}] trainable params (temporal head): {sum(p.numel() for p in model.parameters()):,}")
        fit(
            model, loaders,
            epochs=epochs, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
            device=device, run_dir=run_dir,
        )
        plot_training_curves(run_dir / "history.csv", run_dir / "training_curves.png",
                             title=f"{model_name} - run{run_idx}")

        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        y_true, y_pred = predict(model, loaders["test"], device)
        metrics = compute_metrics(y_true, y_pred, class_names)
        save_run_results(metrics, class_names, run_dir, title=f"{model_name} - run{run_idx}")

        run_elapsed = timedelta(seconds=round(time.perf_counter() - run_start))
        print(f"[{model_name} run{run_idx}] test_acc={metrics['acc']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
              f"f1={metrics['f1']:.4f} duration={run_elapsed}")
        run_metrics.append(metrics)
        del model, loaders
        torch.cuda.empty_cache()

    summary = finish_experiment(model_name, model_dir, run_metrics, total_start)
    best = summary["best_run"]
    plot_training_curves(model_dir / f"run{best}" / "history.csv", model_dir / "best_run_training_curves.png",
                         title=f"{model_name} - best run (run{best})")
    return summary
