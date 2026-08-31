"""Reimplementation of the ORIGINAL 2023 method (Tadvi et al., IJNRD) as a baseline:
GLCM / Haralick texture features + a small back-prop neural network.

Original pipeline: median filter -> morphological lung segmentation -> resize to
3 resolutions -> Haar wavelet -> GLCM in 4 directions -> 7 Haralick features
(252-D vector) -> feed-forward ANN (252-20-2).

Here the lung-segmentation step is unnecessary because we already work on
nodule-centred patches, so we reproduce the *feature + classifier* half faithfully
and report it next to the 3D CNN. Same patient-safe split, same metrics.

    python -m src.baseline.haralick_ann
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..data.dataset import make_splits
from ..data.transforms import hu_window
from ..utils.metrics import binary_metrics, format_metrics
from ..utils.seed import seed_everything

_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_PROPS = ["energy", "correlation", "homogeneity", "contrast", "dissimilarity", "ASM"]
_LEVELS = 32


def _haralick_vector(patch3d: np.ndarray) -> np.ndarray:
    """Take a few representative axial slices, build multi-resolution GLCM features."""
    d = patch3d.shape[2]
    slices = [patch3d[:, :, z] for z in (d // 4, d // 2, 3 * d // 4)]
    feats = []
    for sl in slices:
        for scale in (1.0, 0.5, 0.25):
            img = sl if scale == 1.0 else resize(sl, (max(int(sl.shape[0] * scale), 8),
                                                      max(int(sl.shape[1] * scale), 8)),
                                                 anti_aliasing=True)
            q = np.clip((img * (_LEVELS - 1)).round().astype(np.uint8), 0, _LEVELS - 1)
            glcm = graycomatrix(q, distances=[1], angles=_ANGLES,
                                levels=_LEVELS, symmetric=True, normed=True)
            for prop in _PROPS:
                feats.extend(graycoprops(glcm, prop).ravel().tolist())
    return np.asarray(feats, dtype=np.float32)


def _build_xy(df, cfg):
    from pathlib import Path
    pdir = Path(cfg["data"]["patch_dir"])
    lo, hi = cfg["data"]["hu_clip"]
    X, y = [], []
    for _, row in df.iterrows():
        if row["label"] not in (0, 1):
            continue
        vol = np.load(pdir / f"{row['patch_id']}.npy").astype(np.float32)
        vol = hu_window(vol, lo, hi)
        X.append(_haralick_vector(vol))
        y.append(int(row["label"]))
    return np.vstack(X), np.asarray(y)


def main(argv=None):
    cfg = load_config(argv)
    seed = int(cfg["seed"])
    seed_everything(seed)

    manifest = make_splits(pd.read_csv(cfg["data"]["manifest"]), cfg, seed)
    tr = manifest[manifest["split"].isin(["train", "val"])]
    te = manifest[manifest["split"] == "test"]

    print("[baseline] extracting Haralick features ...")
    Xtr, ytr = _build_xy(tr, cfg)
    Xte, yte = _build_xy(te, cfg)
    print(f"[baseline] feature dim = {Xtr.shape[1]}  train={len(ytr)}  test={len(yte)}")

    scaler = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation="logistic",
                        solver="lbfgs", max_iter=800, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)

    prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
    m = binary_metrics(yte, prob)
    print("[baseline: Haralick + ANN]", format_metrics(m))

    import json
    from pathlib import Path
    out = Path(cfg["output"]["dir"]) / cfg["output"]["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "baseline_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in m.items()}, f, indent=2)
    print(f"[baseline] saved {out/'baseline_metrics.json'}")
    return m


if __name__ == "__main__":
    main()
