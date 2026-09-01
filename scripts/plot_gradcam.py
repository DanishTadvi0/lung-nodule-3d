"""Render Grad-CAM .npz files (from src.utils.gradcam) as a montage PNG.

    python -m src.utils.gradcam --run artifacts/resnet3d_lidc_cv --checkpoint fold1.pt --patch-id <id>
    ... repeat for a few patch-ids ...
    python scripts/plot_gradcam.py --run artifacts/resnet3d_lidc_cv --out artifacts/resnet3d_lidc_cv/gradcam.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/resnet3d_lidc_cv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    files = sorted(run.glob("gradcam_*.npz"))
    if not files:
        raise SystemExit(f"no gradcam_*.npz in {run} - run src.utils.gradcam first")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for j, f in enumerate(files):
        z = np.load(f)
        patch, cam, prob = z["patch"], z["cam"], z["prob"]
        mid = patch.shape[2] // 2
        axes[0][j].imshow(patch[:, :, mid], cmap="gray")
        axes[0][j].set_title(f"{f.stem.replace('gradcam_','')}\np(malignant)={prob[1]:.2f}", fontsize=8)
        axes[1][j].imshow(patch[:, :, mid], cmap="gray")
        axes[1][j].imshow(cam[:, :, mid], cmap="jet", alpha=0.45)
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel("CT (mid-slice)", fontsize=9)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle("3D CNN Grad-CAM - where the model looks")
    fig.tight_layout()
    out = Path(args.out) if args.out else run / "gradcam.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
