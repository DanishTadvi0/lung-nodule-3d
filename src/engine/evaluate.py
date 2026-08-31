"""Stand-alone evaluation + ROC / confusion-matrix plots for a trained run.

    python -m src.engine.evaluate --run artifacts/resnet3d_lidc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import NodulePatchDataset, make_splits
from ..models.resnet3d import build_model
from ..utils.metrics import binary_metrics, format_metrics
from ..utils.seed import seed_everything


def _plot(y_true, y_prob, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    except ImportError:
        print("[warn] matplotlib/sklearn plotting unavailable, skipping figures")
        return
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC - 3D ResNet nodule malignancy")
    plt.savefig(out_dir / "roc.png", dpi=140, bbox_inches="tight")
    plt.close()
    ConfusionMatrixDisplay.from_predictions(y_true, (y_prob >= 0.5).astype(int),
                                            display_labels=["benign", "malignant"])
    plt.savefig(out_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[plots] wrote {out_dir/'roc.png'} and {out_dir/'confusion_matrix.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="artifacts/<run_name> directory")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]
    seed_everything(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest = make_splits(pd.read_csv(cfg["data"]["manifest"]), cfg, int(cfg["seed"]))
    ds = NodulePatchDataset(manifest, cfg, args.split)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            targets.append(y.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets)

    m = binary_metrics(y_true, y_prob)
    print(f"[{args.split}]", format_metrics(m))
    with open(run_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in m.items()}, f, indent=2)
    _plot(y_true, y_prob, run_dir)


if __name__ == "__main__":
    main()
