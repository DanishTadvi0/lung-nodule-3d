# Lung Nodule Malignancy — 3D CNN

An end-to-end 3D convolutional neural network for benign-vs-malignant classification of
pulmonary nodules on chest CT, trained and evaluated on the public **LIDC-IDRI** dataset.

This is a modern reimplementation of the methodology in:

> **Lung cancer detection system using lung CT image processing** — D. Tadvi, A. Kumbhar,
> R. Powar, A. Koli. *IJNRD*, Vol. 8, Issue 1, Jan 2023 (IJNRD2301211).

The 2023 paper classified CT slices with morphological lung segmentation → Haar wavelet →
GLCM / Haralick texture features (252-D) → a 252-20-2 back-propagation ANN, on ~216 private
hospital slices with a single train/test split. This repository keeps the task
(binary benign vs malignant) and rebuilds the pipeline on LIDC-IDRI with a **3D CNN over
volumetric nodule patches**, evaluated with **5-fold patient-grouped cross-validation**.
The original Haralick + ANN method is reimplemented in
[`src/baseline/haralick_ann.py`](src/baseline/haralick_ann.py) and run through the same
cross-validation for a head-to-head comparison.

## Results

**Data:** 189 LIDC-IDRI patients, **405 nodules** (238 benign / 167 malignant) after
dropping the 198 with a median radiologist malignancy score of exactly 3. Evaluation is
5-fold `GroupKFold` on `patient_id` — every nodule is held out exactly once and no patient
spans folds.

| Method (5-fold patient-grouped CV) | Pooled AUC | Sens | Spec | Acc | F1 |
|---|---|---|---|---|---|
| Tadvi et al. 2023 (paper, private data, single split) | – | 0.887 | 0.971 | 0.920 | – |
| Haralick + ANN — 2023 method, reimplemented on LIDC | 0.711 | 0.581 | 0.744 | 0.677 | 0.597 |
| **3D CNN (ResNet-10, ~0.9 M params)** | **0.905** | **0.753** | **0.891** | **0.835** | **0.785** |

Fold-wise AUC: 3D CNN **0.912 ± 0.035** vs Haralick + ANN **0.725 ± 0.035**; all five CNN
folds between 0.87 and 0.96.

| Pooled out-of-fold ROC | Confusion matrix (thr 0.5) | Per-fold AUC |
|---|---|---|
| ![ROC](artifacts/resnet3d_lidc_cv/roc_cv.png) | ![confusion](artifacts/resnet3d_lidc_cv/confusion_matrix_cv.png) | ![folds](artifacts/resnet3d_lidc_cv/fold_auc.png) |

Grad-CAM on held-out nodules — the network localises the nodule core and margin; malignant
cases score p ≥ 0.99, benign p ≈ 0.1:

![Grad-CAM](artifacts/resnet3d_lidc_cv/gradcam.png)

**Limitations:** 189 / 1018 LIDC patients used (compute); single architecture, no
ensembling or test-time augmentation; consensus-radiologist labels, not biopsy-confirmed;
no external-dataset validation. See [`REPORT.md`](REPORT.md) for the full write-up.

## Method

| | 2023 paper | This repository |
|---|---|---|
| Unit of analysis | one 2D axial slice | 3D patch — 40 mm cube, resampled to isotropic 64³ voxels, 48³ crop into the network |
| Lung segmentation | morphological opening/closing, fixed disk SE (size 15) | not required — patches are centred on the consensus nodule annotation |
| Features | Haar wavelet + GLCM + 7 Haralick features (4 directions, 3 scales), 252-D | learned end-to-end by 3D convolutions |
| Classifier | feed-forward ANN 252-20-2, back-propagation | 3D ResNet-10 (~0.9 M params); AdamW, cosine LR with warm-up, mixed precision, weighted sampling for class balance, 3D flips / 90° rotations / intensity jitter, early stop on validation AUC |
| Label | "cancerous / non-cancerous" | median radiologist malignancy: `< 3` benign, `> 3` malignant, `== 3` excluded (standard LIDC convention) |
| Evaluation | single train/test split, accuracy + sens/spec | 5-fold patient-grouped CV; AUC / sensitivity / specificity / F1; pooled out-of-fold ROC; Grad-CAM |

## Repository layout

```
lung-nodule-3d/
├── config/default.yaml           every hyper-parameter; override on the CLI
├── src/
│   ├── config.py                 YAML config + "--a.b.c value" overrides
│   ├── data/
│   │   ├── extract_patches.py     raw LIDC DICOM ──pylidc──▶ 3D .npy patches + manifest.csv
│   │   ├── dataset.py             torch Dataset + patient-safe train/val/test split
│   │   └── transforms.py          HU windowing, 3D flip/rotate/crop augmentation
│   ├── models/resnet3d.py         3D ResNet-10 / -18
│   ├── engine/
│   │   ├── train.py               single-split training loop
│   │   ├── cross_validate.py      5-fold patient-grouped CV
│   │   └── evaluate.py            metrics + ROC + confusion-matrix figures
│   ├── baseline/haralick_ann.py   the 2023 Haralick + ANN method, same CV
│   └── utils/{metrics,seed,gradcam}.py
├── scripts/
│   ├── download_lidc_subset.py    download a CT subset from TCIA, lay it out for pylidc
│   ├── prepare_data.py            chunked, resumable: download ▶ extract ▶ back up
│   ├── make_synthetic_data.py     synthetic patches for a no-download pipeline test
│   ├── plot_cv.py / plot_gradcam.py / plot_history.py
├── notebooks/colab_train.ipynb
└── tests/test_smoke.py           end-to-end run on synthetic data
```

## Reproducing the results

### 0. Environment

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On a fresh Windows machine, PyTorch also needs the Microsoft Visual C++ Redistributable
(x64) — <https://aka.ms/vs/17/release/vc_redist.x64.exe> — if `import torch` raises
`OSError: [WinError 126]`.

### 1. Pipeline sanity check (no data, no GPU)

```bash
python scripts/make_synthetic_data.py --n 200 --patch-vox 64
python -m src.engine.train --train.epochs 3 --train.num_workers 0 --output.run_name smoke
python -m src.engine.evaluate --run artifacts/smoke
pytest -q
```

### 2. Build the dataset from LIDC-IDRI

LIDC-IDRI is public (no TCIA account needed). `prepare_data.py` downloads CT scans in
chunks, extracts nodule patches, and writes `data/processed/{patches/*.npy, manifest.csv}`.
On Colab it also backs the patch set up to Google Drive so it survives a runtime restart.

```bash
python scripts/prepare_data.py --target 200 --chunk 50 \
    --drive-dir /content/drive/MyDrive/lidc-project
```

The included `data/processed/manifest.csv` documents the exact 405 nodules used here
(patch files themselves are not committed — ~0.5 GB).

### 3. Train and evaluate

```bash
python -m src.baseline.haralick_ann --folds 5 --output.run_name resnet3d_lidc
python -m src.engine.cross_validate --folds 5 \
    --train.epochs 60 --train.batch_size 24 --model.depth 10 --data.patch_size 48 \
    --output.run_name resnet3d_lidc
python scripts/plot_cv.py --cv-dir artifacts/resnet3d_lidc_cv
```

### Common overrides

| Goal | Flag |
|---|---|
| Smaller GPU / OOM | `--train.batch_size 16 --data.patch_size 40 --model.depth 10` |
| Larger GPU | `--train.batch_size 64 --model.depth 18 --data.patch_size 56` |
| No class balancing | `--train.balance_classes false` |
| CPU-only | `--train.amp false --train.num_workers 0` |

## Notes on the implementation

`pylidc` (2020) predates Python 3.12 and NumPy 2.x; `src/data/extract_patches.py` restores
the removed `configparser.SafeConfigParser` and `np.int` / `np.float` aliases at import so
it runs on current Colab (Python 3.13). Patch extraction is CPU-only and the model is small
(~0.9 M params), so the full pipeline runs on a free Colab T4 in well under an hour.

## Roadmap

- [x] LIDC subset, `manifest.csv`, patch extraction
- [x] Haralick + ANN baseline under 5-fold patient-grouped CV
- [x] 3D CNN beats the baseline on pooled AUC
- [x] Grad-CAM and failure-case inspection
- [ ] Scale to 500+ patients
- [ ] Ablations: 2D vs 2.5D vs 3D; patch size; augmentation
- [ ] ResNet-18 and deeper backbones
- [ ] Self-supervised pre-training (MAE / DINOv2 on unlabeled LIDC) then fine-tune
- [ ] CNN–Transformer hybrid head

## References

- LUNA16 challenge (LIDC-IDRI subset) — <https://luna16.grand-challenge.org/Data/>
- `pylidc` — <https://pylidc.github.io/>
- *A deep 3D residual CNN for false-positive reduction in pulmonary nodule detection* — <https://pubmed.ncbi.nlm.nih.gov/29500816/>
- *3D multi-view CNNs for lung nodule classification* — <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5690636/>
- *Lung Nodule Classification Using Biomarkers, Volumetric Radiomics, and 3D CNNs* — <https://pmc.ncbi.nlm.nih.gov/articles/PMC8329152/>
- *Lung Nodule-SSM: Self-Supervised Lung Nodule Detection & Classification* — <https://arxiv.org/abs/2505.15120>
- *LMLCC-Net: semi-supervised malignancy prediction* — <https://arxiv.org/pdf/2505.06370>
- *MAEMC-NET: hybrid self-supervised SPN malignancy* — <https://pmc.ncbi.nlm.nih.gov/articles/PMC11861088/>
- Project-MONAI tutorials — <https://github.com/Project-MONAI/tutorials>

## Data use and license

LIDC-IDRI is distributed by The Cancer Imaging Archive (TCIA) under a Creative Commons
Attribution license for research use; cite the collection and the LUNA16 challenge if you
use the data. This repository's code is released under the MIT License (see `LICENSE`).
