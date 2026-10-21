"""Test-set metrics: accuracy, precision, recall, F1, confusion matrix, and
per-class sensitivity/specificity (one class vs. the rest)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def per_class_sensitivity_specificity(cm: np.ndarray, class_names: list[str]) -> dict:
    """The assignment asks for sensitivity/specificity for binary tasks; with
    4 classes they are reported per class, treating that class as positive."""
    out = {}
    total = cm.sum()
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[name] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        }
    return out


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm,
        "per_class": per_class_sensitivity_specificity(cm, class_names),
    }


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(path)


def summarize_runs(run_metrics: list[dict]) -> dict:
    """Mean/std across the assignment's required 5 training runs per model."""
    out = {}
    for key in ("acc", "precision", "recall", "f1"):
        values = np.array([m[key] for m in run_metrics])
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
        out[f"{key}_min"] = float(values.min())
        out[f"{key}_max"] = float(values.max())
    class_names = list(run_metrics[0]["per_class"])
    out["per_class_mean"] = {
        c: {k: float(np.mean([m["per_class"][c][k] for m in run_metrics])) for k in ("sensitivity", "specificity")}
        for c in class_names
    }
    return out
