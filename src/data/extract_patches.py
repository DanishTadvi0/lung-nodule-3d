"""Turn raw LIDC-IDRI DICOM into a tidy set of 3D nodule patches + manifest.

Prerequisites
-------------
1. Download LIDC-IDRI DICOM from TCIA (or a subset) with the NBIA Data Retriever.
2. Configure pylidc so it can find the DICOM. Create ~/.pylidcrc (or ~/pylidc.conf
   on Windows: C:\\Users\\<you>\\pylidc.conf) with:

       [dicom]
       path = D:\\datasets\\LIDC-IDRI
       warn = True

Labelling rule (matches the common LIDC malignancy convention)
-------------------------------------------------------------
Each nodule is scored 1..5 for malignancy by up to 4 radiologists.
    median < 3  -> benign      (label 0)
    median > 3  -> malignant   (label 1)
    median == 3 -> ambiguous   (label -1, kept in manifest, dropped at train time)

Usage
-----
    python -m src.data.extract_patches --out data/processed --patch-mm 40 --patch-vox 64
    python -m src.data.extract_patches --limit 25          # quick partial run
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def _label_from_median(med: float) -> int:
    if med < 3:
        return 0
    if med > 3:
        return 1
    return -1


def extract(out_dir: Path, patch_mm: float, patch_vox: int, limit: int | None):
    import pylidc as pl
    from pylidc.utils import consensus

    patch_dir = out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    scans = pl.query(pl.Scan)
    total = scans.count()
    print(f"[pylidc] {total} scans visible")

    rows = []
    n_done = 0
    for scan in scans:
        if limit and n_done >= limit:
            break
        try:
            vol = scan.to_volume()                      # (H, W, Z) in HU
        except Exception as e:                           # missing/incomplete series
            print(f"  skip {scan.patient_id}: {e}")
            continue

        spacing = np.array([scan.pixel_spacing, scan.pixel_spacing, scan.slice_spacing],
                           dtype=float)
        nodules = scan.cluster_annotations()
        for nidx, anns in enumerate(nodules):
            mals = [a.malignancy for a in anns]
            med = float(np.median(mals))
            label = _label_from_median(med)

            # consensus centroid in voxel coords
            _, cbbox, _ = consensus(anns, clevel=0.5,
                                    pad=[(0, 0), (0, 0), (0, 0)])
            center = np.array([(s.start + s.stop) / 2 for s in cbbox])

            half_vox = (patch_mm / spacing) / 2.0
            lo = np.round(center - half_vox).astype(int)
            hi = np.round(center + half_vox).astype(int)
            lo = np.clip(lo, 0, np.array(vol.shape) - 1)
            hi = np.clip(hi, 1, np.array(vol.shape))
            crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
            if min(crop.shape) < 4:
                continue

            # resample the physical box to an isotropic (patch_vox ** 3) cube
            factors = [patch_vox / s for s in crop.shape]
            cube = zoom(crop, factors, order=1).astype(np.float32)
            cube = cube[:patch_vox, :patch_vox, :patch_vox]
            cube = np.pad(cube, [(0, patch_vox - s) for s in cube.shape], mode="edge")

            diam = float(np.mean([a.diameter for a in anns]))
            pid = f"{scan.patient_id}_n{nidx}"
            np.save(patch_dir / f"{pid}.npy", cube)
            rows.append({
                "patch_id": pid,
                "patient_id": scan.patient_id,
                "malignancy": med,
                "label": label,
                "diameter_mm": round(diam, 2),
                "n_annotations": len(anns),
                "source": "LIDC-IDRI",
            })
        n_done += 1
        if n_done % 10 == 0:
            print(f"  processed {n_done} scans, {len(rows)} nodules so far")

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pos = sum(r["label"] == 1 for r in rows)
    neg = sum(r["label"] == 0 for r in rows)
    amb = sum(r["label"] == -1 for r in rows)
    print(f"\n[done] {len(rows)} nodules -> {manifest_path}")
    print(f"       benign={neg}  malignant={pos}  ambiguous(median==3)={amb}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--patch-mm", type=float, default=40.0,
                    help="physical side length of the cube around each nodule")
    ap.add_argument("--patch-vox", type=int, default=64,
                    help="voxel side length after isotropic resampling")
    ap.add_argument("--limit", type=int, default=None, help="max scans to process")
    args = ap.parse_args()
    extract(Path(args.out), args.patch_mm, args.patch_vox, args.limit)
