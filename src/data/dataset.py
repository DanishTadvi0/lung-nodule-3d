"""Dataset + patient-safe splitting for pre-extracted 3D nodule patches.

Expected on disk (produced by src/data/extract_patches.py):

    data/processed/patches/<patch_id>.npy      # float32 array (D, H, W) in HU
    data/processed/manifest.csv                # columns below

manifest.csv columns:
    patch_id, patient_id, malignancy, label, diameter_mm, source
    - malignancy : median radiologist score 1..5 (0 if unknown)
    - label      : 0 = benign, 1 = malignant  (see extract_patches for the rule)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import EvalTransform, TrainAugment, hu_window


def make_splits(manifest: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Add a 'split' column (train/val/test) without leaking a patient across splits."""
    key = cfg["data"].get("split_by", "patient_id")
    groups = manifest[key].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(round(n * cfg["data"]["test_fraction"]))
    n_val = int(round(n * cfg["data"]["val_fraction"]))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])

    def assign(g):
        if g in test_g:
            return "test"
        if g in val_g:
            return "val"
        return "train"

    manifest = manifest.copy()
    manifest["split"] = manifest[key].map(assign)
    return manifest


class NodulePatchDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, split: str, seed: int = 0):
        self.cfg = cfg
        self.split = split
        self.patch_dir = Path(cfg["data"]["patch_dir"])
        self.hu_lo, self.hu_hi = cfg["data"]["hu_clip"]

        df = manifest[manifest["split"] == split].reset_index(drop=True)
        if cfg["data"].get("exclude_ambiguous", True):
            df = df[df["label"].isin([0, 1])].reset_index(drop=True)
        self.df = df

        if split == "train":
            self.tf = TrainAugment(cfg["augment"], cfg["data"]["patch_size"], seed=seed)
        else:
            self.tf = EvalTransform(cfg["data"]["patch_size"])

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy().astype(int)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (2.0 * counts)
        return torch.tensor(w, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        counts = np.bincount(self.labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        per_class = 1.0 / counts
        return per_class[self.labels]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vol = np.load(self.patch_dir / f"{row['patch_id']}.npy").astype(np.float32)
        vol = hu_window(vol, self.hu_lo, self.hu_hi)
        vol = self.tf(vol)
        x = torch.from_numpy(vol)[None]           # (1, D, H, W)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y
