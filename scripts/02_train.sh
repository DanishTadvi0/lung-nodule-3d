#!/usr/bin/env bash
set -euo pipefail
python -m src.engine.train --output.run_name resnet3d_lidc "$@"
python -m src.baseline.haralick_ann
python -m src.engine.evaluate --run artifacts/resnet3d_lidc
