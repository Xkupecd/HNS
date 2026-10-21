"""Test the best model on own webcam clips and fine-tune it to improve the
camera results, per the assignment ("kvalitu najlepšieho modelu reálne
otestujte na kamere a dotrénujte model pre zlepšenie výsledkov na kamere").

Steps:
  1. extract frames of the clips recorded with record_clips.py
     (data/camera/<class>/*.avi -> data/camera_frames/...) and cache their
     frozen-CNN features (camera-train with N_AUG_CLIPS augmented versions);
  2. split the camera clips per class into camera-train / camera-test;
  3. evaluate the trained model on camera-test and on the HMDB51 test set;
  4. fine-tune on HMDB51-train + camera-train (camera clips repeated so they
     make up about 1/3 of the training data, otherwise the 361 HMDB videos
     dominate):
       - M2/M3: the temporal head is trained further from the best
         checkpoint for a fixed number of epochs with a 3x lower learning
         rate; the last epoch is kept, since camera-test must not be used
         for model selection;
       - M1: the SVM is refit with the same C/gamma on HMDB + camera clips;
  5. evaluate again on camera-test and the HMDB51 test set (to check that the
     model did not forget the original data).

Results go to Results/camera_finetune/<model>/; the fine-tuned model can be
run live with demo.py --checkpoint.

Usage:
    python -m src.camera.finetune_camera              # best model
    python -m src.camera.finetune_camera --model m3 --epochs 10
"""
import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.camera.inference import MODELS, best_model_key, best_run
from src.data_preprocessing.extract_frames import extract_video
from src.engine import predict, set_seed, train_one_epoch, write_history
from src.evaluation import compute_metrics
from src.experiment import save_run_results
from src.features import FeatureDataset, build_cache, load_cache
from src.logging_utils import tee_stdout_to_file
from src.train_m1_cnn_svm import N_TRAIN_CLIPS, build_svm, pool_clip_features, sample_train_clips

LR_FACTOR = 1 / 3
CAMERA_SHARE = 1 / 3


def extract_camera_frames() -> list[tuple[str, int]]:
    samples = []
    for label, class_name in enumerate(config.CLASSES):
        for video_path in sorted((config.CAMERA_VIDEO_DIR / class_name).glob("*.avi")):
            if extract_video(video_path, config.CAMERA_FRAMES_DIR / class_name / video_path.stem):
                samples.append((f"{class_name}/{video_path.stem}", label))
    return samples


def split_camera(samples: list[tuple[str, int]], test_ratio: float, seed: int):
    """Per-class random split, so every class appears in camera-test."""
    rng = random.Random(seed)
    train, test = [], []
    for label in range(len(config.CLASSES)):
        cls = [s for s in samples if s[1] == label]
        rng.shuffle(cls)
        n_test = max(1, round(len(cls) * test_ratio)) if len(cls) > 1 else 0
        test += cls[:n_test]
        train += cls[n_test:]
    return train, test


def eval_sets(model_key: str, model, device, test_sets: dict) -> dict:
    out = {}
    for name, (X, y) in test_sets.items():
        if model_key == "m1":
            y_pred = model.predict(pool_clip_features(X)).tolist()
        else:
            _, y_pred = predict(model, DataLoader(FeatureDataset(X, y), batch_size=config.BATCH_SIZE), device)
        out[name] = compute_metrics(y.tolist(), y_pred, config.CLASSES)
    return out


def finetune_head(head, train_ds, epochs: int, device, out_dir: Path) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.LR * LR_FACTOR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(head, loader, optimizer, criterion, device, scaler=scaler)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        print(f"[finetune] epoch {epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
    write_history(history, out_dir / "history.csv")
    torch.save(head.state_dict(), out_dir / "finetuned.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default=None, help="default: best mean test accuracy")
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test-ratio", type=float, default=0.4, help="share of camera clips per class held out")
    args = parser.parse_args()

    model_key = args.model or best_model_key()
    model_label, results_dir, build_fn = MODELS[model_key]
    run = args.run or best_run(results_dir)
    out_dir = config.CAMERA_RESULTS_DIR / results_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tee_stdout_to_file(out_dir / "finetune_log.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cam_samples = extract_camera_frames()
    if not cam_samples:
        sys.exit(f"no clips in {config.CAMERA_VIDEO_DIR} - record some with: python -m src.camera.record_clips --person <name>")
    cam_train, cam_test = split_camera(cam_samples, args.test_ratio, config.SEED)
    print(f"model: {model_label} run{run} | camera clips: {len(cam_samples)} "
          f"(train {len(cam_train)}, test {len(cam_test)})")
    with (out_dir / "camera_split.json").open("w") as f:
        json.dump({"train": cam_train, "test": cam_test}, f, indent=2)

    # camera caches are rebuilt every time, since new clips may have been recorded
    build_cache("camera_train", cam_train, config.CAMERA_FRAMES_DIR, augmented=True, device=device, force=True)
    build_cache("camera_test", cam_test, config.CAMERA_FRAMES_DIR, augmented=False, device=device, force=True)
    test_sets = {"camera_test": load_cache("camera_test"), "hmdb_test": load_cache("test")}
    X_hmdb, y_hmdb = load_cache("train")
    X_cam, y_cam = load_cache("camera_train")
    repeat = max(1, round(CAMERA_SHARE / (1 - CAMERA_SHARE) * len(y_hmdb) / max(1, len(y_cam))))
    print(f"fine-tuning on {len(y_hmdb)} HMDB + {len(y_cam)}x{repeat} camera clips")

    set_seed(config.SEED)
    if model_key == "m1":
        model = joblib.load(results_dir / f"run{run}" / "svm.joblib")
    else:
        model = build_fn(num_classes=len(config.CLASSES)).head.to(device)
        model.load_state_dict(torch.load(results_dir / f"run{run}" / "best.pt", map_location=device, weights_only=True))

    before = eval_sets(model_key, model, device, test_sets)

    if model_key == "m1":
        run_params = json.loads((results_dir / f"run{run}" / "test_metrics.json").read_text())
        rng = np.random.default_rng(config.SEED)
        parts = [sample_train_clips(X_hmdb, y_hmdb, N_TRAIN_CLIPS, rng)]
        parts += [sample_train_clips(X_cam, y_cam, N_TRAIN_CLIPS, rng) for _ in range(repeat)]
        model = build_svm(run_params["C"], run_params["gamma"])
        model.fit(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        joblib.dump(model, out_dir / "finetuned_svm.joblib")
    else:
        train_ds = ConcatDataset([FeatureDataset(X_hmdb, y_hmdb)] + [FeatureDataset(X_cam, y_cam)] * repeat)
        finetune_head(model, train_ds, args.epochs, device, out_dir)

    after = eval_sets(model_key, model, device, test_sets)

    table = []
    for stage, results in (("before", before), ("after", after)):
        for set_name, metrics in results.items():
            save_run_results(metrics, config.CLASSES, out_dir / f"{stage}_{set_name}",
                             title=f"{model_label} {stage} fine-tuning - {set_name}")
            table.append({"stage": stage, "test set": set_name, "videos": int(metrics["confusion_matrix"].sum()),
                          **{k: round(100 * metrics[k], 2) for k in ("acc", "precision", "recall", "f1")}})
    df = pd.DataFrame(table)
    df.to_csv(out_dir / "before_after.csv", index=False)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
