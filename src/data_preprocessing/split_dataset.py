"""Split the extracted videos into train/val/test (70/15/15).

HMDB51 often cuts several clips from the same movie / YouTube video (e.g.
`#20_Rhythm_clap_..._0.avi` ... `_5.avi`), with the same person, scene and
lighting. A plain random split would put such near-duplicates into both train
and test and overestimate accuracy, so the split is done by source video
(group): all clips of one source end up in the same subset, and the groups are
chosen so that every class is split close to 70/15/15.

Writes data_split/manifest.csv with columns: video,label,group,split
(`video` is the frame folder relative to data/frames).

Usage:
    python -m src.data_preprocessing.split_dataset
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


def source_group(video_stem: str) -> str:
    """HMDB51 names are <source>_<action>_<6 meta tags>, e.g.
    `310ToYuma_drink_h_cm_np1_le_goo_3` -> `310ToYuma` (the meta tags are
    body part, camera motion, #people, viewpoint, quality and clip index)."""
    return video_stem.rsplit("_", 7)[0]


def assign_splits(labels: np.ndarray, groups: np.ndarray, seed: int, n_trials: int = 20000) -> np.ndarray:
    """Assigns whole groups to train/val/test so that every class is split as
    close to 70/15/15 as possible.

    Group sizes are very uneven (one clap source has 31 clips, most have 1-3),
    so StratifiedGroupKFold left only 2 clap videos in the test set. Instead,
    `n_trials` random group assignments are drawn and the one with the
    smallest relative deviation from the target per-class counts is kept."""
    group_names, group_idx = np.unique(groups, return_inverse=True)
    n_classes = labels.max() + 1
    group_class = np.zeros((len(group_names), n_classes))
    np.add.at(group_class, (group_idx, labels), 1)

    ratios = np.array([config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO])
    target = ratios[:, None] * group_class.sum(0)[None, :]  # [split, class]

    rng = np.random.default_rng(seed)
    best_cost, best_assign = np.inf, None
    for _ in range(n_trials):
        assign = rng.choice(3, size=len(group_names), p=ratios)
        counts = np.stack([group_class[assign == s].sum(0) for s in range(3)])
        cost = (((counts - target) / target) ** 2).sum()
        if cost < best_cost:
            best_cost, best_assign = cost, assign

    split_names = np.array(["train", "val", "test"], dtype=object)
    return split_names[best_assign[group_idx]]


def build_manifest(frames_dir: Path = config.FRAMES_DIR, seed: int = config.SEED) -> list[dict]:
    videos, labels = [], []
    for label, class_name in enumerate(config.CLASSES):
        for video_dir in sorted((frames_dir / class_name).iterdir()):
            videos.append(f"{class_name}/{video_dir.name}")
            labels.append(label)
    groups = np.array([source_group(Path(v).name) for v in videos])
    split = assign_splits(np.array(labels), groups, seed)
    return [
        {"video": v, "label": config.CLASSES[l], "group": g, "split": s}
        for v, l, g, s in zip(videos, labels, groups, split)
    ]


def write_manifest(rows: list[dict], out_path: Path = config.SPLIT_DIR / "manifest.csv") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "group", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_manifest()
    write_manifest(rows)
    print(f"Wrote {len(rows)} entries to {config.SPLIT_DIR / 'manifest.csv'}")
    for split_name in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split_name]
        per_class = {c: sum(r["label"] == c for r in subset) for c in config.CLASSES}
        print(f"{split_name:5s} {len(subset):4d} ({len(subset) / len(rows):.1%}) {per_class}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = {r["group"] for r in rows if r["split"] == a} & {r["group"] for r in rows if r["split"] == b}
        assert not shared, f"source videos shared between {a} and {b}: {shared}"


if __name__ == "__main__":
    main()
