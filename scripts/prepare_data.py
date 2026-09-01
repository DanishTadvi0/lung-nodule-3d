"""One resilient command for the whole data step on Colab.

Colab wipes /content when the runtime recycles, so this works in CHUNKS and
backs the patch set up to Drive after every chunk:

  - restores data/processed/ from Drive if a backup exists
  - downloads the next chunk of CT scans, extracts 3D patches
  - re-backs-up data/processed/ to Drive
  - repeats until --target patients are done

If the session dies, just run the exact same command again - it picks up where
it left off from the Drive backup.

    python scripts/prepare_data.py --target 200 --chunk 50 \
        --drive-dir /content/drive/MyDrive/lidc-project

Training later only needs the Drive backup:
    python scripts/prepare_data.py --restore-only --drive-dir .../lidc-project
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROC = Path("data/processed")
STATE = PROC / "_state.json"
RAW = Path("data/raw/LIDC-IDRI")


def sh(*cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"downloaded_through": 0}


def save_state(s: dict):
    PROC.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def restore(backup: Path) -> bool:
    if not backup.exists():
        return False
    print(f"[prepare] restoring {backup} ({backup.stat().st_size/1e6:.0f} MB) ...")
    sh("unzip", "-q", "-o", backup, "-d", "data")
    return True


def backup_to_drive(backup: Path):
    backup.parent.mkdir(parents=True, exist_ok=True)
    tmp = backup.with_suffix(".tmp.zip")
    if tmp.exists():
        tmp.unlink()
    shutil.make_archive(str(tmp.with_suffix("")), "zip", root_dir="data", base_dir="processed")
    tmp.replace(backup)                       # atomic-ish: never leave a half-written backup
    n = sum(1 for _ in (PROC / "patches").glob("*.npy"))
    print(f"[prepare] backed up {n} patches -> {backup}  ({backup.stat().st_size/1e6:.0f} MB)")


def do_chunk(start: int, n: int, patch_mm: float, patch_vox: int):
    sh(sys.executable, "scripts/download_lidc_subset.py",
       "--n-patients", n, "--start", start, "--out", RAW)
    cfgtxt = f"[dicom]\npath = {RAW.resolve()}\nwarn = True\n"
    Path.home().joinpath(".pylidcrc").write_text(cfgtxt)
    Path("/root/.pylidcrc").write_text(cfgtxt)
    # re-extract everything currently on disk (idempotent - rebuilds the manifest)
    sh(sys.executable, "-m", "src.data.extract_patches",
       "--out", "data/processed", "--patch-mm", patch_mm,
       "--patch-vox", patch_vox, "--limit", start + n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200, help="total patients to end up with")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/lidc-project")
    ap.add_argument("--patch-mm", type=float, default=40.0)
    ap.add_argument("--patch-vox", type=int, default=64)
    ap.add_argument("--restore-only", action="store_true")
    args = ap.parse_args()

    backup = Path(args.drive_dir) / "processed.zip"
    restore(backup)
    state = load_state()

    if args.restore_only:
        n = sum(1 for _ in (PROC / "patches").glob("*.npy")) if (PROC / "patches").exists() else 0
        print(f"[prepare] restore-only: {n} patches, downloaded_through={state['downloaded_through']}")
        return

    while state["downloaded_through"] < args.target:
        start = state["downloaded_through"]
        n = min(args.chunk, args.target - start)
        print(f"\n[prepare] ===== chunk: patients {start}..{start+n} =====")
        do_chunk(start, n, args.patch_mm, args.patch_vox)
        if not (PROC / "manifest.csv").exists():
            raise SystemExit("[prepare] no manifest produced - check errors above")
        state["downloaded_through"] = start + n
        save_state(state)
        backup_to_drive(backup)

    import pandas as pd
    m = pd.read_csv(PROC / "manifest.csv")
    print("\n[prepare] DONE")
    print(m["label"].value_counts())
    print(m["patient_id"].nunique(), "patients with >=1 nodule")


if __name__ == "__main__":
    main()
