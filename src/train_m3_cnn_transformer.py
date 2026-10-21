"""Train and evaluate M3 (hybrid CNN + temporal Transformer), with the same
procedure and parameters as M2 so only the temporal model differs.

Usage:
    python -m src.train_m3_cnn_transformer
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.experiment import run_experiment
from src.models.M3_CNN_Transformer import build_head


def main() -> None:
    run_experiment("M3_cnn_transformer", results_root=config.M3_RESULTS_DIR, build_head_fn=build_head)


if __name__ == "__main__":
    main()
