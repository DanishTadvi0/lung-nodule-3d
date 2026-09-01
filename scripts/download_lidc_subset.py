"""Download a SUBSET of LIDC-IDRI CT scans from TCIA and lay them out the way
pylidc expects:  <out>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

LIDC-IDRI is fully public - no TCIA login/token needed.

    # normal use: download 50 CT scans and organise them
    python scripts/download_lidc_subset.py --n-patients 50 --out data/raw/LIDC-IDRI

    # resume / add more later
    python scripts/download_lidc_subset.py --n-patients 50 --start 50 --out data/raw/LIDC-IDRI

    # only re-organise an already-downloaded staging dir (no new download)
    python scripts/download_lidc_subset.py --organise-only --out data/raw/LIDC-IDRI

Rough size: ~110-130 MB per CT scan  ->  50 scans ~= 6 GB.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pydicom


def _as_dicts(obj):
    """getSeries may return a list[dict] or a pandas DataFrame depending on version."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict("records")
    return list(obj)


def download(uids, stage: Path):
    from tcia_utils import nbia
    stage.mkdir(parents=True, exist_ok=True)
    # one series at a time -> resilient to a single bad series / timeout
    for i, uid in enumerate(uids, 1):
        print(f"[tcia] ({i}/{len(uids)}) {uid[:45]}...")
        try:
            nbia.downloadSeries([uid], input_type="list", path=str(stage))
        except Exception as e:
            print(f"       ! failed: {e}")


def organise(stage: Path, out: Path) -> dict:
    """Move every CT .dcm under `stage` into <out>/<PatientID>/<StudyUID>/<SeriesUID>/."""
    out.mkdir(parents=True, exist_ok=True)
    tags = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality"]
    patients, moved, skipped = set(), 0, 0
    for dcm in stage.rglob("*.dcm"):
        try:
            h = pydicom.dcmread(dcm, stop_before_pixels=True, specific_tags=tags)
        except Exception:
            continue
        if getattr(h, "Modality", None) != "CT":
            skipped += 1
            continue
        dest = out / h.PatientID / h.StudyInstanceUID / h.SeriesInstanceUID
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dcm), str(dest / dcm.name))
        patients.add(h.PatientID)
        moved += 1
    return {"ct_files": moved, "non_ct_skipped": skipped, "patients": len(patients)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/LIDC-IDRI", help="final pylidc-ready tree")
    ap.add_argument("--n-patients", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="skip the first N CT scans (to resume)")
    ap.add_argument("--organise-only", action="store_true",
                    help="skip download, just re-organise the staging dir")
    args = ap.parse_args()

    out = Path(args.out)
    stage = out.parent / "_staging"

    if not args.organise_only:
        from tcia_utils import nbia
        print("[tcia] listing LIDC-IDRI CT series ...")
        series = _as_dicts(nbia.getSeries(collection="LIDC-IDRI", modality="CT"))
        series = [s for s in series if s.get("Modality", "CT") == "CT"]
        series.sort(key=lambda s: s.get("PatientID", ""))
        subset = series[args.start:args.start + args.n_patients]
        uids = [s["SeriesInstanceUID"] for s in subset]
        print(f"[tcia] {len(series)} CT series available; downloading {len(uids)}")
        download(uids, stage)

    print("[organise] sorting DICOM into <PatientID>/<Study>/<Series>/ ...")
    stats = organise(stage, out)
    print(f"[organise] {stats}")
    shutil.rmtree(stage, ignore_errors=True)

    n_pat = len([p for p in out.iterdir() if p.is_dir()]) if out.exists() else 0
    print(f"\n[done] {n_pat} patient folders under {out.resolve()}")
    print("Next:")
    print(f"  1. point ~/.pylidcrc  [dicom] path = {out.resolve()}")
    print(f"  2. python -m src.data.extract_patches --out data/processed --limit {n_pat}")


if __name__ == "__main__":
    main()
