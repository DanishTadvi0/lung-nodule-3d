"""One resilient command for the whole data step on Colab.

Colab wipes /content when the runtime recycles, so this script:
  1. if patches already exist locally  -> done
  2. elif a backup zip exists on Drive -> restore it (fast, ~1 min)
  3. else                              -> download DICOM, extract patches,
                                          then back the patches up to Drive

After it finishes once, every later session is just step 2.

    python scripts/prepare_data.py --n-patients 50 \
        --drive-dir /content/drive/MyDrive/lidc-project

    # to add more patients later, bump --n-patients and pass --force-extract
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def sh(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=50)
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/lidc-project")
    ap.add_argument("--patch-mm", type=float, default=40.0)
    ap.add_argument("--patch-vox", type=int, default=64)
    ap.add_argument("--raw-dir", default="data/raw/LIDC-IDRI")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    proc = Path("data/processed")
    manifest = proc / "manifest.csv"
    drive = Path(args.drive_dir)
    backup = drive / f"processed_{args.n_patients}pt.zip"

    # 1. already have patches?
    if manifest.exists() and not args.force_extract:
        n = sum(1 for _ in (proc / "patches").glob("*.npy"))
        print(f"[prepare] patches already present ({n} .npy). Nothing to do.")
        return

    # 2. restore from Drive backup?
    if backup.exists() and not args.force_extract:
        print(f"[prepare] restoring {backup} ...")
        proc.mkdir(parents=True, exist_ok=True)
        sh("unzip", "-q", "-o", backup, "-d", "data")
        n = sum(1 for _ in (proc / "patches").glob("*.npy"))
        print(f"[prepare] restored {n} patches from Drive.")
        return

    # 3. full build
    raw = Path(args.raw_dir)
    have_raw = raw.exists() and any(raw.rglob("*.dcm"))
    if not have_raw:
        print(f"[prepare] downloading {args.n_patients} CT scans from TCIA ...")
        sh(sys.executable, "scripts/download_lidc_subset.py",
           "--n-patients", args.n_patients, "--out", raw)
    else:
        print(f"[prepare] raw DICOM already on disk at {raw}")

    # pylidc config
    cfgtxt = f"[dicom]\npath = {raw.resolve()}\nwarn = True\n"
    Path.home().joinpath(".pylidcrc").write_text(cfgtxt)
    Path("/root/.pylidcrc").write_text(cfgtxt)   # Colab runs as root
    print("[prepare] wrote ~/.pylidcrc")

    print("[prepare] extracting 3D patches ...")
    sh(sys.executable, "-m", "src.data.extract_patches",
       "--out", "data/processed", "--patch-mm", args.patch_mm,
       "--patch-vox", args.patch_vox, "--limit", args.n_patients)

    if not manifest.exists():
        raise SystemExit("[prepare] extraction produced no manifest - check errors above")

    # back up to Drive
    drive.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        backup.unlink()
    print(f"[prepare] backing up patches -> {backup}")
    shutil.make_archive(str(backup.with_suffix("")), "zip", root_dir="data", base_dir="processed")
    print(f"[prepare] done. Backup: {backup}  ({backup.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
