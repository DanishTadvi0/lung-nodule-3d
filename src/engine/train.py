"""Train the 3D nodule-malignancy classifier.

    python -m src.engine.train                         # uses config/default.yaml
    python -m src.engine.train --train.epochs 30 --model.depth 18

Writes to artifacts/<run_name>/:
    config.yaml        resolved config
    best.pt            best checkpoint (by val AUC)
    last.pt            final checkpoint
    history.csv        per-epoch metrics
    splits.csv         which patch went to train/val/test
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import load_config
from ..data.dataset import NodulePatchDataset, make_splits
from ..models.resnet3d import build_model
from ..utils.metrics import binary_metrics, format_metrics
from ..utils.seed import seed_everything


def _loaders(cfg, seed):
    manifest = pd.read_csv(cfg["data"]["manifest"])
    manifest = make_splits(manifest, cfg, seed)

    ds = {s: NodulePatchDataset(manifest, cfg, s, seed=seed) for s in ("train", "val", "test")}
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]

    if cfg["train"].get("balance_classes", True):
        w = ds["train"].sample_weights()
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        train_loader = DataLoader(ds["train"], batch_size=bs, sampler=sampler,
                                  num_workers=nw, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(ds["train"], batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)

    val_loader = DataLoader(ds["val"], batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(ds["test"], batch_size=bs, shuffle=False, num_workers=nw)
    return manifest, ds, train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    probs, targets, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.append(y.cpu().numpy())
    if not probs:
        return {"loss": float("nan")}, np.array([]), np.array([])
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    m = binary_metrics(targets, probs)
    m["loss"] = float(np.mean(losses))
    return m, targets, probs


def main(argv=None):
    cfg = load_config(argv)
    seed = int(cfg["seed"])
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output"]["dir"]) / cfg["output"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=False)

    manifest, ds, train_loader, val_loader, test_loader = _loaders(cfg, seed)
    manifest[["patch_id", "patient_id", "label", "split"]].to_csv(out_dir / "splits.csv", index=False)
    print(f"[data] train={len(ds['train'])}  val={len(ds['val'])}  test={len(ds['test'])}")
    print(f"[data] train class balance: {np.bincount(ds['train'].labels, minlength=2).tolist()}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['model']['name']} depth={cfg['model']['depth']}  params={n_params/1e6:.2f}M  device={device}")

    # If we already oversample the minority class, don't also reweight the loss.
    ce_weight = None if cfg["train"].get("balance_classes", True) \
        else ds["train"].class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 0)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = -np.inf
    patience = cfg["train"].get("early_stop_patience", 999)
    since_best = 0
    hist_path = out_dir / "history.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_sens", "val_spec", "val_auc"])

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        tr_losses = []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_losses.append(loss.item())
        sched.step()

        val_m, _, _ = evaluate(model, val_loader, device, criterion)
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch+1:3d}/{epochs}  lr={lr_now:.2e}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_m['loss']:.4f}  "
              f"{format_metrics(val_m)}  ({time.time()-t0:.1f}s)")

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, lr_now, tr_loss, val_m["loss"],
                                    val_m["accuracy"], val_m["sensitivity"],
                                    val_m["specificity"], val_m["auc"]])

        monitor = val_m.get(cfg["train"]["monitor"].replace("val_", ""), val_m["auc"])
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "last.pt")
        if np.isfinite(monitor) and monitor > best_metric:
            best_metric = monitor
            since_best = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "best.pt")
            print(f"          ^ new best {cfg['train']['monitor']}={best_metric:.4f}")
        else:
            since_best += 1
            if since_best >= patience:
                print(f"[early-stop] no improvement in {patience} epochs")
                break

    # Final test with the best checkpoint.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m, y_true, y_prob = evaluate(model, test_loader, device, criterion)
    print("\n[TEST]", format_metrics(test_m))
    np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_prob=y_prob)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in test_m.items()}, f, indent=2)
    return test_m


if __name__ == "__main__":
    main()
