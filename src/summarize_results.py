"""Collects every trained model's summary.json into one comparison table
(CSV + Markdown) for the documentation.

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


def main() -> None:
    rows = []
    for model_dir in sorted(config.RESULTS_DIR.iterdir()):
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
        print("No Results/<model>/summary.json found yet - train a model first.")
        return

    df = pd.DataFrame(rows)
    csv_path = config.RESULTS_DIR / "results_summary.csv"
    md_path = config.RESULTS_DIR / "results_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))

    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
