"""Video-clip dataset over the extracted frames.

Each sample is a clip tensor [CLIP_LEN, 3, H, W] built from the
FRAMES_PER_VIDEO stored frames of one video:
  - train: temporal jitter, i.e. one random frame from each of CLIP_LEN equal
    segments (TSN-style sampling), plus clip-consistent augmentation;
  - val/test: every second stored frame (fixed, deterministic).
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import config
from src.data_preprocessing.transforms import ClipTransform


def read_manifest(path: Path = config.SPLIT_DIR / "manifest.csv") -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def sample_frame_indices(n_stored: int, clip_len: int, train: bool, rng: random.Random | None = None) -> list[int]:
    seg = n_stored / clip_len
    if train:
        rng = rng or random
        return [min(int(seg * i + rng.random() * seg), n_stored - 1) for i in range(clip_len)]
    return [min(int(seg * i + seg / 2), n_stored - 1) for i in range(clip_len)]


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        frames_dir: Path,
        train: bool,
        augment: bool = True,
        clip_len: int = config.CLIP_LEN,
        mean: tuple = config.IMAGENET_MEAN,
        std: tuple = config.IMAGENET_STD,
    ):
        self.samples = samples
        self.frames_dir = frames_dir
        self.train = train
        self.clip_len = clip_len
        self.transform = ClipTransform(train=train and augment, mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.samples)

    def frame_paths(self, idx: int) -> list[Path]:
        video, _ = self.samples[idx]
        return sorted((self.frames_dir / video).glob("*.jpg"))

    def load_frames(self, idx: int) -> list[Image.Image]:
        paths = self.frame_paths(idx)
        indices = sample_frame_indices(len(paths), self.clip_len, self.train)
        return [Image.open(paths[i]).convert("RGB") for i in indices]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.load_frames(idx)), self.samples[idx][1]


def split_samples(rows: list[dict], split: str) -> list[tuple[str, int]]:
    return [(r["video"], config.CLASSES.index(r["label"])) for r in rows if r["split"] == split]


def worker_init(worker_id: int) -> None:
    # makes temporal jitter / augmentation reproducible for a given torch seed
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
