"""Record own webcam clips for testing / fine-tuning the best model on the camera.

For every class, `--n` clips of `--duration` seconds are recorded: press SPACE,
a 3-2-1 countdown follows, then the clip is recorded and saved as

    data/camera/<class>/<person>_<class>_<NN>.avi

Recording more people / backgrounds / lighting makes the camera evaluation
more meaningful; use a different --person for each one (the person name is
used to keep each person's clips together when splitting, see finetune_camera.py).

Usage:
    python -m src.camera.record_clips --person dominik --n 10
    python -m src.camera.record_clips --person dominik --classes drink eat --n 5
Keys: SPACE = record next clip, s = skip to the next class, q = quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

COUNTDOWN_SEC = 3


def put_text(frame, text: str, y: int = 30, color=(0, 255, 255), scale: float = 0.8):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def next_index(class_dir: Path, prefix: str) -> int:
    existing = [int(p.stem.rsplit("_", 1)[1]) for p in class_dir.glob(f"{prefix}_*.avi")]
    return max(existing, default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--classes", nargs="*", default=config.CLASSES, choices=config.CLASSES)
    parser.add_argument("--n", type=int, default=10, help="clips per class")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds per clip")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for class_name in args.classes:
        class_dir = config.CAMERA_VIDEO_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.person}_{class_name}"
        recorded = 0
        while recorded < args.n:
            ok, frame = cap.read()
            if not ok:
                sys.exit("camera read failed")
            view = cv2.flip(frame, 1)
            put_text(view, f"{class_name.upper()}  clip {recorded + 1}/{args.n}")
            put_text(view, "SPACE = record, s = next class, q = quit", y=60, color=(255, 255, 255), scale=0.55)
            cv2.imshow("record clips", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                return
            if key == ord("s"):
                break
            if key != ord(" "):
                continue

            start = time.monotonic()
            while (left := COUNTDOWN_SEC - (time.monotonic() - start)) > 0:
                ok, frame = cap.read()
                view = cv2.flip(frame, 1)
                put_text(view, f"{class_name.upper()} in {int(left) + 1}", scale=1.5)
                cv2.imshow("record clips", view)
                cv2.waitKey(1)

            out_path = class_dir / f"{prefix}_{next_index(class_dir, prefix):02d}.avi"
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)  # stored unmirrored, like the dataset videos
                view = cv2.flip(frame, 1)
                put_text(view, f"REC {class_name}", color=(0, 0, 255), scale=1.0)
                cv2.imshow("record clips", view)
                cv2.waitKey(1)
            writer.release()
            recorded += 1
            print(f"saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
