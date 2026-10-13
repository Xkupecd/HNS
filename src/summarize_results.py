"""Collects every trained configuration's summary.json into one comparison
table (CSV + Markdown) per model group, e.g. Results/M1_CNN/results_summary.md.

Usage:
    python -m src.summarize_results
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def summarize_group(group_dir: Path) -> None:
    rows = []
    for model_dir in sorted(group_dir.iterdir()):
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append({
            "model": model_dir.name,
            "acc": fmt(summary["acc_mean"], summary["acc_std"]),
            "precision": fmt(summary["precision_mean"], summary["precision_std"]),
            "recall": fmt(summary["recall_mean"], summary["recall_std"]),
            "f1": fmt(summary["f1_mean"], summary["f1_std"]),
            "total_duration": summary.get("total_duration", "n/a"),
        })

    if not rows:
        print(f"No {group_dir.name}/<config>/summary.json found yet - train a model first.")
        return

    df = pd.DataFrame(rows)
    csv_path = group_dir / "results_summary.csv"
    md_path = group_dir / "results_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))

    print(df.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}\n")


def main() -> None:
    for group_dir in (config.M1_RESULTS_DIR, config.M2_RESULTS_DIR, config.M3_RESULTS_DIR,
                      config.M4_RESULTS_DIR, config.M5_RESULTS_DIR):
        if group_dir.exists():
            summarize_group(group_dir)


if __name__ == "__main__":
    main()
