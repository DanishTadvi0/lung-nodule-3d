"""Generate a tiny SYNTHETIC nodule-patch dataset so you can validate the whole
pipeline (train -> evaluate -> gradcam -> baseline) on a laptop CPU in minutes,
before touching real LIDC-IDRI data.

Malignant patches get a brighter, more irregular, spiculated blob; benign patches
get a small round smooth blob. Not medically meaningful - just enough signal for
a sanity check.

    python scripts/make_synthetic_data.py --n 200 --patch-vox 64
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _blob(size, rng, malignant):
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    c = size / 2 + rng.normal(0, 2, 3)
    r = (size * (0.18 if malignant else 0.13)) + rng.normal(0, 1)
    d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    vol = np.full((size, size, size), -900.0, np.float32)            # "air"
    core = d < r
    vol[core] = 30 + rng.normal(0, 15, core.sum())
    if malignant:
        for _ in range(rng.integers(4, 9)):                          # spicules
            dirn = rng.normal(size=3); dirn /= np.linalg.norm(dirn)
            for t in np.linspace(r, r * 2.2, 20):
                p = np.round(c + t * dirn).astype(int)
                if (p >= 0).all() and (p < size).all():
                    vol[p[0], p[1], p[2]] = 20
    vol += rng.normal(0, 25, vol.shape)                              # scanner noise
    return vol.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--patch-vox", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out)
    pdir = out / "patches"
    pdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for i in range(args.n):
        malignant = bool(i % 2)                                      # balanced
        patient = f"SYN-{i // 3:04d}"                                # ~3 nodules/patient
        pid = f"{patient}_n{i}"
        np.save(pdir / f"{pid}.npy", _blob(args.patch_vox, rng, malignant))
        rows.append({
            "patch_id": pid, "patient_id": patient,
            "malignancy": 5 if malignant else 1,
            "label": int(malignant), "diameter_mm": 12.0 if malignant else 7.0,
            "n_annotations": 3, "source": "SYNTHETIC",
        })

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.n} synthetic patches + {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
