"""Make the training-curve figure and a method-comparison table from an artifacts run.

    python scripts/plot_history.py --run artifacts/resnet3d_lidc

Writes  <run>/training_curves.png  and prints a markdown table you can paste
into the README results section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline-metrics", default=None,
                    help="optional path to a JSON dict of the Haralick+ANN baseline test metrics")
    args = ap.parse_args()
    run = Path(args.run)

    hist = pd.read_csv(run / "history.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
        ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_auc"], label="val AUC")
        ax[1].plot(hist["epoch"], hist["val_acc"], label="val acc")
        ax[1].set_title("validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(run / "training_curves.png", dpi=140)
        print(f"[plot] wrote {run/'training_curves.png'}")
    except ImportError:
        print("[warn] matplotlib missing, skipping figure")

    test = json.load(open(run / "test_metrics.json"))
    rows = [("3D ResNet (this repo)", test)]
    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        rows.insert(0, ("Haralick + ANN (2023 re-impl)", json.load(open(args.baseline_metrics))))

    print("\n| Method | Test AUC | Sensitivity | Specificity | Accuracy | F1 |")
    print("|---|---|---|---|---|---|")
    print("| Tadvi et al. 2023 (paper, private data) | - | 0.887 | 0.971 | 0.920 | - |")
    for name, m in rows:
        print(f"| {name} | {m.get('auc', float('nan')):.3f} | {m['sensitivity']:.3f} "
              f"| {m['specificity']:.3f} | {m['accuracy']:.3f} | {m['f1']:.3f} |")


if __name__ == "__main__":
    main()
