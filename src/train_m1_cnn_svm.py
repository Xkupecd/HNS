"""Train and evaluate M1 (CNN + SVM): frozen ConvNeXt-B frame features,
pooled over time (mean and std of the 16 frame vectors -> 2048-d clip
descriptor), classified by an RBF-kernel SVM.

The 5 runs differ by seed: each run draws N_TRAIN_CLIPS of the N_AUG_CLIPS
cached augmented clips of every training video, so the SVM's training set
changes between runs while val/test features are fixed. C and gamma are
chosen on the validation set.

Usage:
    python -m src.train_m1_cnn_svm
"""
import csv
import sys
import time
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.engine import set_seed
from src.evaluation import compute_metrics
from src.experiment import finish_experiment, save_run_results
from src.features import load_cache
from src.logging_utils import tee_stdout_to_file

N_TRAIN_CLIPS = 5
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]
GAMMA_GRID = ["scale", 3e-5, 1e-4, 3e-4]


def pool_clip_features(frame_feats) -> np.ndarray:
    """[B, T, D] -> [B, 2D]: temporal mean (appearance) + std (change over
    time). Always computed in torch (unbiased std), for cached numpy features
    as well as for live camera features, so both match exactly."""
    f = torch.as_tensor(frame_feats).float()
    return torch.cat([f.mean(1), f.std(1)], dim=1).cpu().numpy()


def sample_train_clips(X_aug: np.ndarray, y: np.ndarray, n_clips: int, rng: np.random.Generator):
    """[N, A, T, D] -> n_clips randomly chosen augmented clips per video."""
    picks = np.stack([rng.choice(X_aug.shape[1], n_clips, replace=False) for _ in range(len(X_aug))])
    clips = X_aug[np.arange(len(X_aug))[:, None], picks].reshape(-1, *X_aug.shape[2:])
    return pool_clip_features(clips), np.repeat(y, n_clips)


def build_svm(C: float, gamma) -> object:
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                                               random_state=config.SEED))


def grid_search(X_train, y_train, X_val, y_val, seed: int) -> tuple[dict, list[dict]]:
    """Picks (C, gamma) with the best val accuracy; ties go to the smaller C
    (wider margin, less overfitting)."""
    rows, best = [], None
    for gamma in GAMMA_GRID:
        for C in C_GRID:
            svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed))
            svm.fit(X_train, y_train)
            val_acc = float((svm.predict(X_val) == y_val).mean())
            rows.append({"C": C, "gamma": gamma, "val_acc": val_acc})
            if best is None or val_acc > best["val_acc"] or (val_acc == best["val_acc"] and C < best["C"]):
                best = rows[-1]
    return best, rows


def main() -> None:
    model_name = "M1_cnn_svm"
    model_dir = config.M1_RESULTS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout_to_file(model_dir / "training_log.txt")
    print(f"backbone={config.BACKBONE} (frozen, cached features) clip_len={config.CLIP_LEN} "
          f"train_clips_per_video={N_TRAIN_CLIPS}/{config.N_AUG_CLIPS}")

    X_aug, y_aug = load_cache("train")
    X_val, y_val = load_cache("val")
    X_test, y_test = load_cache("test")
    X_val, X_test = pool_clip_features(X_val), pool_clip_features(X_test)
    class_names = config.CLASSES

    total_start = time.perf_counter()
    run_metrics = []
    for run_idx in range(1, config.NUM_RUNS + 1):
        run_dir = model_dir / f"run{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = config.SEED + run_idx
        set_seed(seed)
        run_start = time.perf_counter()

        X_train, y_train = sample_train_clips(X_aug, y_aug, N_TRAIN_CLIPS, np.random.default_rng(seed))
        best, grid_rows = grid_search(X_train, y_train, X_val, y_val, seed)
        with (run_dir / "grid_search.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["C", "gamma", "val_acc"])
            writer.writeheader()
            writer.writerows(grid_rows)

        svm = build_svm(best["C"], best["gamma"])
        svm.fit(X_train, y_train)
        joblib.dump(svm, run_dir / "svm.joblib")
        train_acc = float((svm.predict(X_train) == y_train).mean())

        metrics = compute_metrics(y_test.tolist(), svm.predict(X_test).tolist(), class_names)
        metrics.update({"C": best["C"], "gamma": best["gamma"], "train_acc": train_acc, "val_acc": best["val_acc"]})
        save_run_results(metrics, class_names, run_dir, title=f"{model_name} - run{run_idx}")

        run_elapsed = timedelta(seconds=round(time.perf_counter() - run_start))
        print(f"[{model_name} run{run_idx}] C={best['C']} gamma={best['gamma']} "
              f"train_acc={train_acc:.4f} val_acc={best['val_acc']:.4f} test_acc={metrics['acc']:.4f} "
              f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
              f"f1={metrics['f1']:.4f} duration={run_elapsed}")
        run_metrics.append(metrics)

    finish_experiment(model_name, model_dir, run_metrics, total_start)


if __name__ == "__main__":
    main()
