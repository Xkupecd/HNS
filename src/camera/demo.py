"""Real-time action recognition from a webcam (or a video file).

The last WINDOW_SEC seconds of frames are kept in a buffer; every STRIDE_SEC
CLIP_LEN frames spread uniformly over that window are classified, so the
model sees about as much of the action as in an HMDB51 clip. Probabilities are
smoothed with an exponential moving average; below CONF_THRESHOLD the demo
shows "?" instead of a class (the model only knows 4 actions and always
outputs one of them).

Usage:
    python -m src.camera.demo                       # best model, webcam 0
    python -m src.camera.demo --model m2 --camera 1
    python -m src.camera.demo --checkpoint Results/camera_finetune/M2_cnn_lstm/finetuned.pt --model m2
    python -m src.camera.demo --source video.avi --no-display   # offline test, prints predictions
Keys: q = quit.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.camera.inference import ActionPredictor, best_model_key

STRIDE_SEC = 0.25
EMA_ALPHA = 0.5
COLORS_BGR = [(176, 114, 76), (82, 132, 221), (104, 168, 85), (170, 80, 170)]


def select_window(buffer: deque, now: float, window: float, n: int) -> list[np.ndarray] | None:
    """Picks the buffered frame closest to each of `n` uniform time points in
    [now - window, now]; None until the buffer covers most of the window."""
    times = np.array([t for t, _ in buffer])
    if len(times) < n or now - times[0] < 0.8 * window:
        return None
    targets = now - window + (np.arange(n) + 0.5) * window / n
    idx = np.abs(times[None, :] - targets[:, None]).argmin(1)
    return [buffer[i][1] for i in idx]


def draw_overlay(frame: np.ndarray, probs: np.ndarray | None, model_name: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (260, 40 + 30 * len(config.CLASSES)), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    if probs is None:
        label = "collecting frames..."
    else:
        top = int(probs.argmax())
        label = config.CLASSES[top] if probs[top] >= config.CONF_THRESHOLD else "?"
        for i, (name, p) in enumerate(zip(config.CLASSES, probs)):
            y = 50 + 30 * i
            cv2.rectangle(frame, (90, y - 14), (90 + int(150 * p), y + 4), COLORS_BGR[i], -1)
            cv2.putText(frame, f"{name}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{100 * p:.0f}%", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{model_name} | {fps:.0f} fps", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["m1", "m2", "m3"], default=None, help="default: best mean test accuracy")
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="e.g. a camera-finetuned model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", type=Path, default=None, help="video file instead of the webcam")
    parser.add_argument("--no-display", action="store_true", help="print predictions instead of showing a window")
    parser.add_argument("--save", type=Path, default=None, help="write the annotated stream to this video file")
    args = parser.parse_args()

    predictor = ActionPredictor(args.model or best_model_key(), run=args.run, checkpoint=args.checkpoint)
    print(f"model: {predictor.name} ({predictor.source}) on {predictor.device}")

    cap = cv2.VideoCapture(str(args.source) if args.source else args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open {'file ' + str(args.source) if args.source else 'camera ' + str(args.camera)}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = config.WINDOW_SEC
    if args.source:
        # HMDB51 clips can be shorter than the window (some are ~1.5 s)
        window = min(window, cap.get(cv2.CAP_PROP_FRAME_COUNT) / file_fps)
    writer = None

    buffer: deque = deque(maxlen=int(config.WINDOW_SEC * 120))
    probs, last_pred, frame_idx = None, -np.inf, 0
    fps, t_prev = 0.0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # a file is replayed on its own time axis, the webcam on wall-clock time
        now = frame_idx / file_fps if args.source else time.monotonic()
        frame_idx += 1
        if not args.source:
            frame = cv2.flip(frame, 1)  # mirror view; flips are part of the training augmentation
        buffer.append((now, frame))

        if now - last_pred >= STRIDE_SEC:
            clip = select_window(buffer, now, window, config.CLIP_LEN)
            if clip is not None:
                p = predictor.predict_proba(clip)
                probs = p if probs is None else EMA_ALPHA * p + (1 - EMA_ALPHA) * probs
                last_pred = now
                if args.no_display:
                    top = int(probs.argmax())
                    print(f"t={now:6.2f}s  {config.CLASSES[top]:6s} "
                          + "  ".join(f"{c}={100 * v:4.1f}%" for c, v in zip(config.CLASSES, probs)))

        t_now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        annotated = draw_overlay(frame, probs, predictor.name, fps)
        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), file_fps if args.source else 30, (w, h))
            writer.write(annotated)
        if not args.no_display:
            cv2.imshow("HNS Projekt2 - action recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
