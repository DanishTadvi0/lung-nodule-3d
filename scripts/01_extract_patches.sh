#!/usr/bin/env bash
# Extract 3D nodule patches from raw LIDC-IDRI DICOM (needs ~/pylidc.conf configured).
set -euo pipefail
python -m src.data.extract_patches --out data/processed --patch-mm 40 --patch-vox 64 "$@"
