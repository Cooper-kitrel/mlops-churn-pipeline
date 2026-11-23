"""Evaluation metrics and threshold checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Return classification metrics. roc_auc is the primary metric."""
    y_true = np.asarray(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def check_thresholds(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Return a list of failure messages (empty list means all thresholds pass)."""
    failures = []
    for name, minimum in thresholds.items():
        if name not in metrics:
            failures.append(f"{name}: metric not computed")
        elif metrics[name] < minimum:
            failures.append(
                f"{name}: {metrics[name]:.4f} < required {minimum}")
    return failures
