"""Download a SUBSET of LIDC-IDRI DICOM straight from TCIA's public API (no GUI).

Uses `tcia_utils` (pip install tcia-utils). LIDC-IDRI is a fully public collection,
so no login/token is needed.

    python scripts/download_lidc_subset.py --n-patients 120 --out data/raw/LIDC-IDRI

Then point pylidc at --out (see README section 4, Path B) and run extract_patches.

Rough size: ~110-130 MB per patient  ->  120 patients ~= 15 GB.
Start small (--n-patients 60) to prove the pipeline, then scale up.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/LIDC-IDRI")
    ap.add_argument("--n-patients", type=int, default=120)
    ap.add_argument("--start", type=int, default=0, help="skip the first N patients (for resuming)")
    args = ap.parse_args()

    try:
        from tcia_utils import nbia
    except ImportError:
        raise SystemExit("pip install tcia-utils  first")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[tcia] listing LIDC-IDRI series ...")
    series = nbia.getSeries(collection="LIDC-IDRI")            # list of dicts
    # one CT series per patient in LIDC-IDRI; keep deterministic order
    series = sorted(series, key=lambda s: s["PatientID"])
    subset = series[args.start:args.start + args.n_patients]
    uids = [s["SeriesInstanceUID"] for s in subset]
    print(f"[tcia] downloading {len(uids)} series -> {out}")

    nbia.downloadSeries(
        uids,
        input_type="list",
        path=str(out),
        format="dict",
    )
    print("[done] now set your pylidc config to point at:", out.resolve())
    print("       then: python -m src.data.extract_patches --out data/processed --limit", args.n_patients)


if __name__ == "__main__":
    main()
