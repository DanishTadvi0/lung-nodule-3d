"""K-fold patient-grouped cross-validation for the 3D CNN.

The right evaluation when the dataset is small: every patient is in the held-out
test fold exactly once, and metrics are reported as mean +/- std across folds,
plus pooled out-of-fold predictions for a single ROC curve.

    python -m src.engine.cross_validate --folds 5 --train.epochs 60 --model.depth 10

Writes artifacts/<run_name>_cv/:
    fold_metrics.csv     one row per fold
    summary.json         mean/std of each metric
    oof_predictions.npz  out-of-fold y_true / y_prob for every nodule
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import load_config
from ..data.dataset import NodulePatchDataset
from ..models.resnet3d import build_model
from ..utils.metrics import binary_metrics, format_metrics
from ..utils.seed import seed_everything
from .train import evaluate


def _fold_frame(manifest, train_pat, val_pat, test_pat):
    df = manifest.copy()
    split = np.where(df["patient_id"].isin(test_pat), "test",
             np.where(df["patient_id"].isin(val_pat), "val", "train"))
    df["split"] = split
    return df


def train_fold(manifest, cfg, seed, device):
    ds = {s: NodulePatchDataset(manifest, cfg, s, seed=seed) for s in ("train", "val", "test")}
    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    pin = torch.cuda.is_available()

    if cfg["train"].get("balance_classes", True):
        w = ds["train"].sample_weights()
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        tl = DataLoader(ds["train"], batch_size=bs, sampler=sampler,
                        num_workers=nw, pin_memory=pin, drop_last=True)
    else:
        tl = DataLoader(ds["train"], batch_size=bs, shuffle=True,
                        num_workers=nw, pin_memory=pin, drop_last=True)
    vl = DataLoader(ds["val"], batch_size=bs, num_workers=nw)
    tel = DataLoader(ds["test"], batch_size=bs, num_workers=nw)

    model = build_model(cfg).to(device)
    ce_w = None if cfg["train"].get("balance_classes", True) else ds["train"].class_weights().to(device)
    crit = nn.CrossEntropyLoss(weight=ce_w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / max(warmup, 1) if e < warmup
        else 0.5 * (1 + np.cos(np.pi * (e - warmup) / max(epochs - warmup, 1))))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_state, since = -np.inf, None, 0
    patience = cfg["train"].get("early_stop_patience", 999)
    for epoch in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        vm, _, _ = evaluate(model, vl, device, crit)
        auc = vm["auc"] if np.isfinite(vm["auc"]) else vm["accuracy"]
        if auc > best_auc:
            best_auc, since = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tm, y_true, y_prob = evaluate(model, tel, device, crit)
    return tm, y_true, y_prob, {k: len(v) for k, v in ds.items()}, model.state_dict()


def main(argv=None):
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--folds", type=int, default=5)
    args, rest = pre.parse_known_args(argv)
    cfg = load_config(rest)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(cfg["data"]["manifest"])
    manifest = manifest[manifest["label"].isin([0, 1])].reset_index(drop=True)
    groups = manifest["patient_id"].to_numpy()
    y = manifest["label"].to_numpy()
    print(f"[cv] {len(manifest)} nodules / {len(set(groups))} patients / {args.folds} folds")
    print(f"[cv] class balance: {np.bincount(y).tolist()}  device={device}")

    out = Path(cfg["output"]["dir"]) / f"{cfg['output']['run_name']}_cv"
    out.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=args.folds)
    rows, oof_true, oof_prob = [], [], []
    for k, (tr_idx, te_idx) in enumerate(gkf.split(manifest, y, groups), 1):
        test_pat = set(groups[te_idx])
        train_pat = list(set(groups[tr_idx]))
        rng = np.random.default_rng(seed + k)
        rng.shuffle(train_pat)
        n_val = max(1, int(round(len(train_pat) * 0.15)))
        val_pat, fit_pat = set(train_pat[:n_val]), set(train_pat[n_val:])
        fold_df = _fold_frame(manifest, fit_pat, val_pat, test_pat)

        tm, yt, yp, sizes, state = train_fold(fold_df, cfg, seed + k, device)
        print(f"[fold {k}/{args.folds}] sizes={sizes}  {format_metrics(tm)}")
        rows.append({"fold": k, **{kk: tm[kk] for kk in
                     ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc")}})
        oof_true.append(yt)
        oof_prob.append(yp)
        torch.save({"model": state, "cfg": cfg, "fold": k},
                   out / f"fold{k}.pt")

    with open(out / "fold_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows).drop(columns="fold")
    summary = {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in df.columns}
    oof_true = np.concatenate(oof_true)
    oof_prob = np.concatenate(oof_prob)
    summary["pooled_auc"] = float(binary_metrics(oof_true, oof_prob)["auc"])
    np.savez(out / "oof_predictions.npz", y_true=oof_true, y_prob=oof_prob)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[CV SUMMARY]")
    for c in df.columns:
        print(f"  {c:12s} {summary[c]['mean']:.3f} +/- {summary[c]['std']:.3f}")
    print(f"  pooled AUC   {summary['pooled_auc']:.3f}")
    print(f"\nsaved -> {out}")
    return summary


if __name__ == "__main__":
    main()
