"""Binary classification metrics that match what the 2023 paper reported
(accuracy / sensitivity / specificity) plus AUC and F1 for a modern write-up."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0     # recall for malignant
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:                                     # only one class present
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def format_metrics(m: dict) -> str:
    return (
        f"acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  "
        f"spec={m['specificity']:.3f}  prec={m['precision']:.3f}  "
        f"f1={m['f1']:.3f}  auc={m['auc']:.3f}  "
        f"(tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']})"
    )
