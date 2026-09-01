# Modernising a 2023 Lung-Cancer CT Pipeline with a 3D CNN

**Danish Tadvi**
Reimplementation and extension of *"Lung cancer detection system using lung CT image
processing"*, D. Tadvi, A. Kumbhar, R. Powar, A. Koli, *IJNRD* 8(1), Jan 2023
(IJNRD2301211).

---

## Abstract

The 2023 paper classified chest-CT slices as cancerous / non-cancerous with a classical
computer-vision pipeline: median filtering, morphological lung segmentation, Haar-wavelet
decomposition, grey-level co-occurrence matrix (GLCM) Haralick texture features (252-D),
and a 252–20–2 back-propagation neural network. It reported 92 % test accuracy, 88.7 %
sensitivity and 97.1 % specificity on ~216 CT slices from a private hospital archive,
evaluated on a single train/test split.

This project keeps the classification task but rebuilds the pipeline on the public
**LIDC-IDRI** dataset with an **end-to-end 3-D convolutional neural network** that operates
on volumetric nodule patches rather than hand-engineered 2-D texture descriptors, and
replaces the single split with **5-fold patient-grouped cross-validation**. The original
feature-plus-classifier method is reimplemented and run through the identical
cross-validation as a baseline.

On **405 nodules from 189 patients**, the 3-D CNN attains **pooled out-of-fold AUC 0.905**
(fold-wise 0.912 ± 0.035), versus **0.711** (fold-wise 0.725 ± 0.035) for the reimplemented
Haralick + ANN method — an improvement of ~0.19 AUC under a matched protocol. Grad-CAM
shows the network's response concentrating on the nodule and its margin.

---

## 1. Introduction

Lung cancer is the leading cause of cancer death worldwide. Five-year survival rises from
roughly 15 % when disease is found late to 50–70 % when it is found while still confined to
the lung, so the value of screening CT lies almost entirely in early, accurate
characterisation of the small pulmonary nodules it reveals. Distinguishing a benign nodule
from an early malignancy on CT is difficult by eye and is exactly the kind of task where
learned image models have made the largest clinical gains.

The 2023 paper demonstrated that a fully classical pipeline could separate the two classes.
Three properties of that work limit how far the result travels:

1. **Private data.** The ~216 slices came from two hospital archives and cannot be
   re-obtained, so the number cannot be reproduced or benchmarked by anyone else.
2. **Hand-crafted 2-D features.** Haralick texture statistics on single axial slices
   discard the 3-D shape of the nodule. Spiculation and lobulation — among the strongest
   visual predictors of malignancy — are volumetric properties that a single-slice GLCM
   cannot see.
3. **A brittle segmentation step.** The morphological lung mask used a fixed disk
   structuring element (size 15); the authors' own error analysis attributed their
   misclassifications to this fixed size failing on nodules at the lung border and on
   scans where the mediastinum was mis-segmented.

**Contributions of this work:**

- A reproducible pipeline on public LIDC-IDRI data: programmatic download, annotation
  handling, patch extraction, training and evaluation, all scripted and seeded.
- A 3-D CNN that classifies whole nodule volumes end-to-end, removing both the
  segmentation step and the hand-crafted feature stage.
- A faithful reimplementation of the 2023 Haralick + ANN method as a baseline, evaluated
  under the *same* 5-fold patient-grouped cross-validation for a fair head-to-head.
- Grad-CAM explainability and a fold-level error breakdown.

---

## 2. Related work

**LIDC-IDRI and LUNA16.** The Lung Image Database Consortium image collection (LIDC-IDRI)
contains 1018 thoracic CT scans, each annotated by up to four radiologists who mark nodule
contours and score nine characteristics including malignancy on a 1–5 scale. LUNA16 is a
curated 888-scan subset that became the standard benchmark for nodule *detection*. This
work uses the malignancy scores directly for *classification*.

**3-D CNNs for nodules.** Volumetric CNNs are the established approach for nodule
false-positive reduction and malignancy prediction; multi-scale and multi-view 3-D
architectures consistently outperform their 2-D counterparts on LIDC-derived data because
nodule morphology is inherently three-dimensional. The architecture here follows the
residual-3-D-CNN family (see References).

**Recent directions (2025–2026).** Current research adds self-supervised pre-training on
large pools of unlabelled CT (masked auto-encoders, DINOv2-style objectives) and
CNN–Transformer hybrids that combine local convolutional features with global attention.
These are listed under Future work rather than implemented here.

---

## 3. Data

### 3.1 Source and acquisition

CT series were downloaded from The Cancer Imaging Archive (TCIA) with the public NBIA REST
API (`tcia-utils`); LIDC-IDRI requires no account or token. 200 patients were requested;
189 yielded at least one usable nodule after annotation processing. Downloaded DICOM was
reorganised into the `PatientID / StudyInstanceUID / SeriesInstanceUID` directory layout
that `pylidc` expects.

### 3.2 Annotation processing

Radiologist annotations were read with `pylidc`. For each scan, `pylidc`'s
`cluster_annotations()` groups the individual radiologists' marks that refer to the same
physical nodule. For every resulting nodule cluster:

- **Malignancy label** is the *median* of the participating radiologists' 1–5 malignancy
  scores, mapped as: median `< 3` → benign (0), median `> 3` → malignant (1),
  median `== 3` → *excluded* (indeterminate). This is the standard LIDC labelling
  convention and avoids forcing a decision on the genuinely ambiguous middle score.
- **Location** is the centroid of the 50 %-consensus mask (`pylidc.utils.consensus`,
  `clevel = 0.5`).
- **Diameter** is the mean of the per-annotation diameters (recorded, not used for
  training).

### 3.3 Patch extraction

Around each nodule centroid a **40 mm physical cube** was cropped from the CT volume and
resampled with linear interpolation to an **isotropic 64³ voxel grid**, so that one voxel
is ≈ 0.625 mm regardless of the scan's native slice spacing. Patches were stored as
`float32` arrays in Hounsfield units (HU).

### 3.4 Class distribution

| | count |
|---|---|
| Nodule clusters extracted | 603 |
| Benign (median malignancy < 3) | 238 |
| Malignant (median malignancy > 3) | 167 |
| Excluded (median == 3) | 198 |
| **Used for training / evaluation** | **405** (58.8 % benign / 41.2 % malignant) |
| Patients with ≥ 1 usable nodule | 189 (165 contribute to the labelled set) |

### 3.5 Preprocessing at train time

- **HU windowing:** clip to [−1000, 400] HU (air to soft-tissue / calcification edge),
  linearly rescale to [0, 1].
- **Network input:** 48³ crop of the 64³ patch — random crop with augmentation during
  training, centre crop at evaluation.

---

## 4. Method

### 4.1 Problem formulation

Binary classification of a single nodule patch `x ∈ R^{1×48×48×48}` as benign or malignant.
The model outputs two logits; `softmax(·)[1]` is the malignancy probability. Operating
threshold for the reported confusion matrices and accuracy/sensitivity/specificity is 0.5;
AUC is threshold-independent.

### 4.2 3-D CNN architecture

A compact 3-D residual network (`src/models/resnet3d.py`), ResNet-10 configuration:

| Block | Detail | Output (C×D×H×W) |
|---|---|---|
| Stem | Conv3d 5³, stride 1, pad 2 → BatchNorm → ReLU → MaxPool3d 2³ | 16 × 24³ |
| Stage 1 | 1 × BasicBlock3D, width 16, stride 1 | 16 × 24³ |
| Stage 2 | 1 × BasicBlock3D, width 32, stride 2 | 32 × 12³ |
| Stage 3 | 1 × BasicBlock3D, width 64, stride 2 | 64 × 6³ |
| Stage 4 | 1 × BasicBlock3D, width 128, stride 2 | 128 × 3³ |
| Head | AdaptiveAvgPool3d(1) → Dropout(0.2) → Linear(128, 2) | 2 |

`BasicBlock3D` is the standard two-conv residual block (Conv3d 3³ → BN → ReLU → Conv3d 3³
→ BN, plus a 1³-conv projection shortcut when the channel count or stride changes).
Convolutions use Kaiming-normal initialisation; BatchNorm weights/biases init to 1/0.
Total parameters ≈ 0.9 M.

### 4.3 Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 3 × 10⁻⁴, weight decay 1 × 10⁻⁴ |
| LR schedule | 3-epoch linear warm-up → cosine decay to 0 |
| Epochs | 60 (early stop, patience 12 on validation AUC) |
| Batch size | 24 |
| Precision | mixed (AMP) on GPU |
| Class balance | `WeightedRandomSampler` inverse-frequency over the training set |
| Loss | cross-entropy (unweighted; balancing handled by the sampler) |
| Augmentation | random 48³ crop; random flips on all three axes (p 0.5 each); random in-plane 90° rotation; intensity shift ± 0.1; intensity scale × [0.9, 1.1]; re-clip to [0, 1] |
| Checkpoint | best model by validation AUC restored before test |
| Seed | 42 (per-fold seed 42 + fold index) |

### 4.4 Baseline — the 2023 method reimplemented

`src/baseline/haralick_ann.py` reproduces the feature-plus-classifier half of the 2023
pipeline. The morphological lung-segmentation stage is unnecessary here because patches are
already nodule-centred; everything downstream of segmentation is reproduced:

- From each patch, take 3 axial slices (at ¼, ½, ¾ depth). For each slice and each of 3
  resolution scales (1.0, 0.5, 0.25), quantise to 32 grey levels and compute a GLCM at
  distance 1 for 4 directions (0°, 45°, 90°, 135°), symmetric and normalised.
- Extract 6 Haralick statistics per GLCM (energy, correlation, homogeneity, contrast,
  dissimilarity, angular second moment) → **216-D feature vector**
  (3 slices × 3 scales × 4 directions × 6 statistics; the original paper's 252-D used
  Haar sub-bands instead of resolution scales — the same idea).
- `StandardScaler` → scikit-learn `MLPClassifier`, one hidden layer of 20 units, logistic
  activation, L-BFGS solver — matching the paper's 252–20–2 feed-forward ANN.

### 4.5 Evaluation protocol

- **5-fold `GroupKFold` on `patient_id`.** Every nodule appears in the held-out test fold
  exactly once; no patient's nodules are split across train and test. This prevents the
  optimistic leakage that a random nodule-level split would introduce (nodules from one
  patient are correlated).
- Within each training split, 15 % of the *patients* are held out as a validation set for
  early stopping and checkpoint selection.
- **Pooled out-of-fold (OOF) predictions:** concatenating the five test folds gives one
  prediction per nodule over the whole dataset; the pooled ROC/AUC is computed on these.
- Metrics: AUC (primary), sensitivity, specificity, precision, F1, accuracy. Reported as
  pooled values and as mean ± std across folds.
- Hardware: single NVIDIA T4 (Google Colab). Full 5-fold CNN CV runs in ~15 minutes;
  baseline feature extraction + CV in ~10 minutes (CPU-bound).

---

## 5. Results

### 5.1 Headline comparison

| Method (5-fold patient-grouped CV) | Pooled AUC | Sens | Spec | Acc | F1 |
|---|---|---|---|---|---|
| Tadvi et al. 2023 (paper; private data, single split) | – | 0.887 | 0.971 | 0.920 | – |
| Haralick + ANN — 2023 method, reimplemented on LIDC | 0.711 | 0.581 | 0.744 | 0.677 | 0.597 |
| **3-D CNN (ResNet-10)** | **0.905** | **0.753** | **0.891** | **0.835** | **0.785** |

Fold-wise AUC: **3-D CNN 0.912 ± 0.035**, **baseline 0.725 ± 0.035**.

*The 2023 paper's row is shown for context only and is not directly comparable:* it is a
different dataset, a slice-level rather than nodule-level task, and a single split rather
than cross-validation.

### 5.2 Per-fold breakdown

**3-D CNN** (test fold = 81 nodules each):

| Fold | Acc | Sens | Spec | Prec | F1 | AUC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.852 | 0.690 | 0.942 | 0.870 | 0.769 | 0.911 | 20 | 49 | 3 | 9 |
| 2 | 0.889 | 0.886 | 0.891 | 0.861 | 0.873 | 0.957 | 31 | 41 | 5 | 4 |
| 3 | 0.790 | 0.828 | 0.769 | 0.667 | 0.738 | 0.934 | 24 | 40 | 12 | 5 |
| 4 | 0.815 | 0.756 | 0.875 | 0.861 | 0.805 | 0.870 | 31 | 35 | 5 | 10 |
| 5 | 0.827 | 0.606 | 0.979 | 0.952 | 0.741 | 0.887 | 20 | 47 | 1 | 13 |
| **mean ± std** | 0.835 ± 0.038 | 0.753 ± 0.110 | 0.891 ± 0.080 | 0.842 ± 0.105 | 0.785 ± 0.056 | 0.912 ± 0.035 | | | | |

**Baseline (Haralick + ANN):**

| Fold | Acc | Sens | Spec | Prec | F1 | AUC |
|---|---|---|---|---|---|---|
| 1 | 0.654 | 0.552 | 0.712 | 0.516 | 0.533 | 0.674 |
| 2 | 0.679 | 0.686 | 0.674 | 0.615 | 0.649 | 0.765 |
| 3 | 0.654 | 0.655 | 0.654 | 0.514 | 0.576 | 0.696 |
| 4 | 0.679 | 0.463 | 0.900 | 0.826 | 0.594 | 0.730 |
| 5 | 0.716 | 0.576 | 0.812 | 0.679 | 0.623 | 0.759 |
| **pooled** | 0.677 | 0.581 | 0.744 | 0.614 | 0.597 | 0.711 |

### 5.3 Curves

`artifacts/resnet3d_lidc_cv/`:

- `roc_cv.png` — pooled OOF ROC, both methods on one axis.
- `confusion_matrix_cv.png` — 3-D CNN, pooled OOF, threshold 0.5
  (≈ 97 TP / 24 FN / 21 FP / 217 TN across all 405 nodules).
- `fold_auc.png` — per-fold AUC.

### 5.4 Grad-CAM

Grad-CAM (`src/utils/gradcam.py`, on `layer4`) was computed for held-out nodules.
Activation localises on the nodule core and its margin rather than on surrounding
parenchyma or vessels. Confidently malignant examples score p ≥ 0.99; benign examples
p ≈ 0.10–0.15. Montage in `artifacts/resnet3d_lidc_cv/gradcam.png`.

---

## 6. Discussion

**Volumetric, learned features are the difference.** Moving from single-slice Haralick
descriptors to a 3-D CNN raises pooled AUC from 0.711 to 0.905 on the same patients with
the same folds. The gain is consistent with the nodule-classification literature: margin
irregularity, spiculation and lobulation are 3-D shape cues, and a convolutional model can
learn them jointly with texture and surrounding context, whereas a 2-D GLCM cannot
represent them at all.

**The baseline is a fair one.** Pooled AUC 0.711 is a reasonable score for GLCM texture
features on LIDC and is in line with published classical-CV results; folds 2 and 5 reach
AUC 0.76. The ~0.19 gap therefore reflects the method, not a weakened reimplementation.
(The original paper's 0.92 accuracy is higher than the baseline's 0.68 here because it is a
different, smaller, single-split dataset with a slice-level rather than nodule-level task —
the two numbers should not be compared directly.)

**Where the CNN errs.** Sensitivity varies more across folds (0.61–0.89) than specificity
(0.77–0.98); the model is consistently good at confirming benignity and less consistent at
catching malignancy, and folds 1 and 5 in particular carry most of the false negatives
(9 and 13). This is the operating-point behaviour expected from a class-imbalanced problem
at a fixed 0.5 threshold, and in a screening setting the threshold would be lowered to
trade specificity for sensitivity. A full error analysis — relating false negatives to
nodule size, solidity and location — is left as future work; the confusion-matrix and
Grad-CAM outputs in `artifacts/` are the starting point and mirror the "reasons for
misclassification" section of the 2023 paper.

**Fold variance.** With ~80 test and ~50 validation nodules per fold, fold-wise AUC still
spans 0.87–0.96. The pooled OOF AUC (0.905, computed over all 405 nodules) is the more
stable summary. Scaling the labelled set is the direct way to tighten the fold-wise
estimate.

---

## 7. Limitations

- **Scale.** 189 of 1018 LIDC patients were used; this is a compute/time budget choice,
  not a methodological limit — `prepare_data.py` scales to the full collection unchanged.
- **Single model.** One architecture, one configuration; no ensembling, no test-time
  augmentation, no hyper-parameter search.
- **Labels.** Malignancy is the radiologists' consensus score, not biopsy- or
  follow-up-confirmed ground truth. The `median == 3` exclusion removes ~33 % of nodules.
- **No external validation.** All data is LIDC-IDRI; generalisation to other scanners and
  populations is untested.
- **Patch assumptions.** A fixed 40 mm cube can under-cover large masses and, for small
  nodules, include distracting adjacent anatomy.

---

## 8. Future work

1. Re-run the 5-fold CV at 500+ and then all 1018 patients.
2. Ablations: 2-D single-slice vs 2.5-D vs 3-D; patch size (32/48/64 mm); augmentation
   on/off; ResNet-10 vs ResNet-18.
3. Error analysis stratified by nodule diameter, solidity (solid / part-solid /
   ground-glass) and lung-border proximity.
4. Probability calibration (reliability curves, temperature scaling) and a sensitivity-
   first operating point.
5. Self-supervised pre-training (masked auto-encoder or DINOv2 objective) on the
   unlabelled LIDC volumes, then fine-tune the classifier.
6. A CNN–Transformer hybrid head for global context.

---

## 9. Reproducibility

Everything needed to reproduce the numbers is in the repository:

| Artefact | Location |
|---|---|
| Exact nodule set (405 rows) | `data/processed/manifest.csv` |
| Per-fold metrics, pooled OOF predictions, summary | `artifacts/resnet3d_lidc_cv/` |
| Config used | `config/default.yaml` (+ CLI overrides recorded per run) |
| Figures | `artifacts/resnet3d_lidc_cv/*.png` |

```bash
# build the dataset from TCIA (chunked, resumable, backs up to Drive on Colab)
python scripts/prepare_data.py --target 200 --chunk 50 --drive-dir <dir>

# baseline (2023 method) under 5-fold patient-grouped CV
python -m src.baseline.haralick_ann --folds 5 --output.run_name resnet3d_lidc

# 3-D CNN under the same CV
python -m src.engine.cross_validate --folds 5 \
    --train.epochs 60 --train.batch_size 24 --model.depth 10 --data.patch_size 48 \
    --output.run_name resnet3d_lidc

# figures + comparison table
python scripts/plot_cv.py --cv-dir artifacts/resnet3d_lidc_cv
```

Fixed seeds, patient-grouped splits, and per-run config snapshots make each number
regenerable; a synthetic-data smoke test (`tests/test_smoke.py`) exercises the whole
pipeline without any download.

---

## References

1. Armato SG et al. *The Lung Image Database Consortium (LIDC) and Image Database Resource
   Initiative (IDRI): A completed reference database of lung nodules on CT scans.* Medical
   Physics, 2011.
2. Setio AAA et al. *Validation, comparison, and combination of algorithms for automatic
   detection of pulmonary nodules in CT images: the LUNA16 challenge.* Medical Image
   Analysis, 2017. <https://luna16.grand-challenge.org/>
3. Hawkins S et al. / Zhu W et al. — 3-D CNNs for pulmonary nodule malignancy on LIDC.
   *A deep 3D residual CNN for false-positive reduction in pulmonary nodule detection*,
   <https://pubmed.ncbi.nlm.nih.gov/29500816/>.
4. *3D multi-view convolutional neural networks for lung nodule classification.*
   <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5690636/>
5. *Lung Nodule Classification Using Biomarkers, Volumetric Radiomics, and 3D CNNs.*
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC8329152/>
6. *Lung Nodule-SSM: Self-Supervised Lung Nodule Detection and Classification in Thoracic
   CT Images.* 2025. <https://arxiv.org/abs/2505.15120>
7. *LMLCC-Net: A Semi-Supervised Deep Learning Model for Lung Nodule Malignancy
   Prediction.* 2025. <https://arxiv.org/pdf/2505.06370>
8. Hancock MC, Magnan JF. *pylidc: a Python library for working with the LIDC dataset.*
   <https://pylidc.github.io/>
9. Tadvi D, Kumbhar A, Powar R, Koli A. *Lung cancer detection system using lung CT image
   processing.* IJNRD 8(1), 2023 (IJNRD2301211).
