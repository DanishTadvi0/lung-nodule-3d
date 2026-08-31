"""Grad-CAM for the 3D ResNet - produces a saliency volume you can overlay on
the nodule patch. Good for a 'model explainability' figure in the report / CV.

    python -m src.utils.gradcam --run artifacts/resnet3d_lidc --patch-id LIDC-IDRI-0003_n0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.transforms import center_crop, hu_window
from ..models.resnet3d import build_model


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=1):
        logits = self.model(x)
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.softmax(logits, 1)[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--patch-id", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.run) / args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])

    cam = GradCAM3D(model, model.layer4)

    vol = np.load(Path(cfg["data"]["patch_dir"]) / f"{args.patch_id}.npy").astype(np.float32)
    vol = hu_window(vol, *cfg["data"]["hu_clip"])
    vol = center_crop(vol, cfg["data"]["patch_size"])
    x = torch.from_numpy(vol)[None, None]

    heat, prob = cam(x)
    print(f"p(malignant) = {prob[1]:.3f}")
    out = Path(args.run) / f"gradcam_{args.patch_id}.npz"
    np.savez(out, patch=vol, cam=heat, prob=prob)
    print(f"saved {out}  (load in a notebook and overlay mid-slices)")


if __name__ == "__main__":
    main()
