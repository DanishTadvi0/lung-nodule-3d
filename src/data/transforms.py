"""Lightweight 3D volume transforms (numpy in, numpy out).

Kept dependency-free on purpose so the pipeline runs even if MONAI is missing.
If you have MONAI you can swap these for monai.transforms for a richer set.
"""
from __future__ import annotations

import numpy as np


def hu_window(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip Hounsfield units to a window and scale to [0, 1]."""
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo)


def center_crop(vol: np.ndarray, size: int) -> np.ndarray:
    slices = []
    for dim in vol.shape:
        start = max((dim - size) // 2, 0)
        slices.append(slice(start, start + size))
    return _pad_to(vol[tuple(slices)], size)


def random_crop(vol: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    slices = []
    for dim in vol.shape:
        if dim <= size:
            slices.append(slice(0, dim))
        else:
            start = int(rng.integers(0, dim - size + 1))
            slices.append(slice(start, start + size))
    return _pad_to(vol[tuple(slices)], size)


def _pad_to(vol: np.ndarray, size: int) -> np.ndarray:
    pad = [(0, max(size - d, 0)) for d in vol.shape]
    if any(p != (0, 0) for p in pad):
        vol = np.pad(vol, pad, mode="edge")
    return vol


class TrainAugment:
    """Random flips, in-plane 90-deg rotations, intensity shift, random crop."""

    def __init__(self, cfg: dict, patch_size: int, seed: int = 0):
        self.a = cfg
        self.size = patch_size
        self.rng = np.random.default_rng(seed)

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        vol = random_crop(vol, self.size, self.rng)

        if self.a.get("flip", True):
            for axis in range(3):
                if self.rng.random() < 0.5:
                    vol = np.flip(vol, axis=axis)

        if self.a.get("rot90", True):
            k = int(self.rng.integers(0, 4))
            vol = np.rot90(vol, k=k, axes=(1, 2))

        shift = float(self.a.get("intensity_shift", 0.0))
        if shift:
            vol = vol + self.rng.uniform(-shift, shift)

        lo, hi = self.a.get("scale", [1.0, 1.0])
        if lo != 1.0 or hi != 1.0:
            vol = vol * self.rng.uniform(lo, hi)

        return np.ascontiguousarray(np.clip(vol, 0.0, 1.0), dtype=np.float32)


class EvalTransform:
    def __init__(self, patch_size: int):
        self.size = patch_size

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(center_crop(vol, self.size), dtype=np.float32)
