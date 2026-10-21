"""Extract FRAMES_PER_VIDEO uniformly spaced frames from every video of the
selected classes and store them as 224x224 JPGs:

    data/frames/<class>/<video_stem>/000.jpg ... 031.jpg

Frames are resized to 224x224 without cropping (HMDB51 widths vary from 320
to 560 px at 240 px height, so a center crop could cut off the hands). The
webcam demo resizes its frames exactly the same way.

Usage:
    python -m src.data_preprocessing.extract_frames
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


def read_all_frames(video_path: Path) -> list[np.ndarray]:
    """Decodes the whole clip; CAP_PROP_FRAME_COUNT is unreliable for some
    HMDB51 .avi files, so frames are counted by actually reading them."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def uniform_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Centers of `n_samples` equal segments; repeats frames for very short clips."""
    return ((np.arange(n_samples) + 0.5) * n_frames / n_samples).astype(int).clip(0, n_frames - 1)


def resize_frame(frame_bgr: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)


def extract_video(video_path: Path, out_dir: Path, n_samples: int = config.FRAMES_PER_VIDEO) -> int:
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == n_samples:
        return n_samples
    frames = read_all_frames(video_path)
    if not frames:
        print(f"WARNING: no frames decoded from {video_path}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(uniform_indices(len(frames), n_samples)):
        cv2.imwrite(str(out_dir / f"{i:03d}.jpg"), resize_frame(frames[idx]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return n_samples


def extract_dataset(video_dir: Path, frames_dir: Path, classes: list[str]) -> None:
    jobs = []
    for class_name in classes:
        for video_path in sorted((video_dir / class_name).glob("*.avi")):
            jobs.append((video_path, frames_dir / class_name / video_path.stem))
    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as pool:
        results = list(pool.map(extract_video, *zip(*jobs)))
    n_ok = sum(1 for r in results if r)
    print(f"Extracted {n_ok}/{len(jobs)} videos into {frames_dir}")


def main() -> None:
    extract_dataset(config.VIDEO_DIR, config.FRAMES_DIR, config.CLASSES)


if __name__ == "__main__":
    main()
