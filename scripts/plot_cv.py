"""Figures + comparison table from a cross-validation run.

    python scripts/plot_cv.py --cv-dir artifacts/resnet3d_lidc_cv

Writes into --cv-dir:
    roc_cv.png                 pooled out-of-fold ROC, 3D CNN vs Haralick+ANN
    confusion_matrix_cv.png    3D CNN, pooled OOF, threshold 0.5
    fold_auc.png               per-fold AUC (both methods)
and prints a Markdown results table for the README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", default="artifacts/resnet3d_lidc_cv")
    args = ap.parse_args()
    d = Path(args.cv_dir)

    cnn = np.load(d / "oof_predictions.npz")
    yt, yp = cnn["y_true"], cnn["y_prob"]
    base = np.load(d / "baseline_oof.npz") if (d / "baseline_oof.npz").exists() else None
    folds = pd.read_csv(d / "fold_metrics.csv") if (d / "fold_metrics.csv").exists() else None
    summary = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
    except ImportError:
        print("[warn] matplotlib/sklearn missing - skipping figures")
        return

    # --- ROC ---
    plt.figure(figsize=(5.2, 5))
    fpr, tpr, _ = roc_curve(yt, yp)
    plt.plot(fpr, tpr, lw=2, label=f"3D CNN (AUC {auc(fpr, tpr):.3f})")
    if base is not None:
        fb, tb, _ = roc_curve(base["y_true"], base["y_prob"])
        plt.plot(fb, tb, lw=2, ls="--", label=f"Haralick+ANN (AUC {auc(fb, tb):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
    plt.xlabel("1 - specificity"); plt.ylabel("sensitivity")
    plt.title("Pooled out-of-fold ROC (5-fold patient-grouped CV)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(d / "roc_cv.png", dpi=150); plt.close()

    # --- confusion matrix ---
    ConfusionMatrixDisplay.from_predictions(
        yt, (yp >= 0.5).astype(int), display_labels=["benign", "malignant"],
        colorbar=False)
    plt.title("3D CNN - pooled OOF predictions (thr 0.5)")
    plt.tight_layout(); plt.savefig(d / "confusion_matrix_cv.png", dpi=150); plt.close()

    # --- per-fold AUC ---
    if folds is not None:
        plt.figure(figsize=(5.5, 3.5))
        x = folds["fold"]
        plt.bar(x - 0.18, folds["auc"], width=0.36, label="3D CNN")
        plt.ylim(0.5, 1.0); plt.xlabel("fold"); plt.ylabel("AUC")
        plt.xticks(list(x)); plt.title("Per-fold test AUC")
        plt.legend(); plt.tight_layout()
        plt.savefig(d / "fold_auc.png", dpi=150); plt.close()

    print(f"[plots] wrote roc_cv.png, confusion_matrix_cv.png, fold_auc.png in {d}\n")

    # --- markdown table ---
    def line(name, m):
        return (f"| {name} | {m['auc']:.3f} | {m['sensitivity']:.3f} "
                f"| {m['specificity']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} |")

    print("| Method (5-fold patient-grouped CV) | Pooled AUC | Sens | Spec | Acc | F1 |")
    print("|---|---|---|---|---|---|")
    print("| Tadvi et al. 2023 (paper, private data, single split) | - | 0.887 | 0.971 | 0.920 | - |")
    if base is not None:
        from src.utils.metrics import binary_metrics
        print(line("Haralick + ANN (2023 method, reimplemented)",
                   binary_metrics(base["y_true"], base["y_prob"])))
    from src.utils.metrics import binary_metrics
    print(line("**3D CNN (ResNet-10)**", binary_metrics(yt, yp)))
    if summary:
        a = summary.get("auc", {})
        print(f"\n3D CNN fold-wise AUC: {a.get('mean', float('nan')):.3f} "
              f"+/- {a.get('std', float('nan')):.3f}")


if __name__ == "__main__":
    main()
