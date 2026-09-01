# Modernising a 2023 Lung-Cancer CT Pipeline with a 3D CNN

**Danish Tadvi** — reimplementation and extension of *"Lung cancer detection system using
lung CT image processing"* (IJNRD2301211, Jan 2023).

---

## Abstract

The 2023 paper classified lung CT slices as cancerous / non-cancerous using morphological
lung segmentation, Haar-wavelet + GLCM Haralick texture features (252-D), and a 252-20-2
back-propagation neural network, reaching 92 % accuracy on ~216 private hospital slices
with a single train/test split. This project rebuilds the same task on the public
**LIDC-IDRI** dataset with an **end-to-end 3D convolutional neural network** operating on
volumetric nodule patches, evaluated with **5-fold patient-grouped cross-validation**. On
405 nodules from 189 patients, the 3D CNN reaches **pooled AUC 0.905** (fold-wise
0.912 ± 0.035), versus **0.711** for a faithful reimplementation of the 2023 Haralick+ANN
method on the identical splits — a ~0.19 AUC improvement. Grad-CAM confirms the network
attends to the nodule and its margin.

## 1. Background

Lung cancer has the highest cancer mortality; five-year survival rises from ~15 % to
50–70 % with early detection. CT screening produces pulmonary nodules whose malignancy is
hard to judge visually. The 2023 work showed a classical computer-vision pipeline could
separate the two classes, but (a) on a small private dataset that cannot be re-obtained,
(b) with hand-crafted 2D features, and (c) with a fixed-size morphological structuring
element that the authors themselves identified as the cause of their misclassifications.

## 2. Data

- **Source:** LIDC-IDRI (public, TCIA), 189 patients downloaded programmatically.
- **Annotations:** `pylidc`; nodules clustered across up to 4 radiologists.
- **Label:** median malignancy score — `<3` benign, `>3` malignant, `==3` excluded.
- **Patches:** 40 mm cube around each nodule's consensus centroid, resampled to an
  isotropic 64³ voxel grid, Hounsfield units windowed to [-1000, 400] → [0, 1].
- **Result:** 603 nodules → **405 usable** (238 benign, 167 malignant) after dropping 198
  ambiguous.

## 3. Method

| | 2023 paper | This work |
|---|---|---|
| Input | 1 axial slice | 3D patch, 48³ (random/centre crop of 64³) |
| Lung segmentation | morphological, fixed SE | not required (nodule-centred patches) |
| Features | Haar + GLCM + 7 Haralick, 252-D | learned by 3D conv layers |
| Model | ANN 252-20-2, back-prop | 3D ResNet-10, ~0.9 M params |
| Training | — | AdamW, cosine LR + warmup, mixed precision, weighted sampling for class balance, 3D flips / 90° rotations / intensity jitter, early stop on validation AUC |
| Evaluation | 1 split, accuracy | 5-fold `GroupKFold` on `patient_id`; pooled out-of-fold ROC; AUC / sensitivity / specificity / F1 |

**Baseline for comparison:** the 2023 feature+classifier half (GLCM Haralick at 3 slices ×
3 scales × 4 directions → 216-D, `StandardScaler`, MLP `(20,)` logistic) run through the
*same* 5-fold patient-grouped CV.

## 4. Results

| Method (5-fold patient-grouped CV) | Pooled AUC | Sens | Spec | Acc | F1 |
|---|---|---|---|---|---|
| Tadvi et al. 2023 (paper, private data, single split) | – | 0.887 | 0.971 | 0.920 | – |
| Haralick + ANN — 2023 method, reimplemented on LIDC | 0.711 | 0.581 | 0.744 | 0.677 | 0.597 |
| **3D CNN (ResNet-10)** | **0.905** | **0.753** | **0.891** | **0.835** | **0.785** |

Fold-wise AUC: **0.912 ± 0.035** (CNN) vs **0.725 ± 0.035** (baseline); all five CNN folds
between 0.87 and 0.96. See `artifacts/resnet3d_lidc_cv/` for `roc_cv.png`,
`confusion_matrix_cv.png`, `fold_auc.png`, `gradcam.png`.

**Grad-CAM:** on held-out nodules the network's activation concentrates on the nodule core
and margin; malignant examples score p ≥ 0.99, benign p ≈ 0.1.

## 5. Discussion

- **Volumetric context matters.** The single biggest change — 2D slice → 3D patch with
  learned features — accounts for most of the +0.19 AUC. Spiculation and lobulation, the
  strongest malignancy cues, are 3D shape properties invisible to a single-slice GLCM.
- **The baseline is not a straw man.** 0.71 pooled AUC is a reasonable score for Haralick
  features and matches the classical-CV literature; the gap is the method, not a crippled
  reimplementation.
- **Failure modes** (cf. the 2023 paper's "reasons for misclassification"): the CNN's
  false negatives cluster on small (<6 mm) nodules and part-solid nodules near vessels,
  where the 40 mm patch contains distracting anatomy. False positives are mostly benign
  nodules with irregular but non-malignant margins.

## 6. Limitations

189 / 1018 LIDC patients (compute, not method, limited); single architecture, no
ensembling or test-time augmentation; consensus-radiologist labels rather than
biopsy-confirmed truth; no external-dataset validation.

## 7. Future work

5-fold CV at 500+ patients; ResNet-18 / ablation on patch size and 2D-vs-2.5D-vs-3D;
self-supervised pre-training (masked auto-encoder or DINOv2 on unlabeled LIDC) then
fine-tune; a CNN–Transformer hybrid head; calibration analysis.

## 8. Reproducibility

All code, config, the 405-nodule manifest, per-fold metrics, and figures are in this
repository. `python scripts/prepare_data.py` builds the dataset from TCIA;
`python -m src.engine.cross_validate` reproduces the CNN result;
`python -m src.baseline.haralick_ann` the baseline. Fixed seed, patient-grouped splits,
saved config per run.
