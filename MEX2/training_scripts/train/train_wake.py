"""Train the "Hey Rapi" wake-word detector on a single GPU.

    source ~/rapi_vcm/venv/bin/activate
    python train_wake.py --feat $FEAT --out $OUT --gpu 1

The selection metric is *true-positive rate at a false-accept rate <= 1 %*,
measured on the validation split, because a wake word must stay silent on
everything else.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# --------------------------------------------------------------------------- #
# GPU selection MUST happen before torch is imported.
# --------------------------------------------------------------------------- #
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=1)
_known, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn as nn                                          # noqa: E402
from torch.utils.data import DataLoader                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm.datasets import AverageMeter, WakeDataset        # noqa: E402
from rapi_vcm.models import build_wake, count_parameters       # noqa: E402

TARGET_FAR = 0.001          # 0.1 % false-accept rate on validation negatives


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feat", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--patience", type=int, default=12)
    return p.parse_args()


def collect_scores(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits.append(model(xb).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics_at_threshold(logits, labels, thr):
    scores = 1.0 / (1.0 + np.exp(-logits))
    pos = labels == 1
    neg = ~pos
    tpr = float(((scores >= thr) & pos).sum() / max(pos.sum(), 1))
    far = float(((scores >= thr) & neg).sum() / max(neg.sum(), 1))
    return tpr, far


def pick_threshold(logits, labels, target_far=TARGET_FAR):
    """Threshold whose false-accept rate on the negatives is `target_far`.

    Taken as the (1 - target_far) quantile of the *negative* scores, which is
    stable epoch to epoch (unlike a threshold derived from the interleaved
    sorted list) and gives the FAR by construction.
    """
    scores = 1.0 / (1.0 + np.exp(-logits))
    neg = scores[labels == 0]
    if neg.size == 0:
        return 0.5
    return float(np.quantile(neg, 1.0 - target_far))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wake] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={device} n_gpu={torch.cuda.device_count()}", flush=True)
    if device.type == "cuda":
        print(f"[wake] using GPU 0 -> {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    train_ds = WakeDataset(args.feat, split=0, augment=True)
    val_ds = WakeDataset(args.feat, split=1, augment=False)
    print(f"[wake] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=len(train_ds) > args.batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_wake().to(device)
    print(f"[wake] parameters={count_parameters(model)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.999))
    total_steps = max(len(train_loader) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)

    n_pos = int((train_ds.y[train_ds.idx] == 1).sum())
    n_neg = int((train_ds.y[train_ds.idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = device.type == "cuda"

    best = -1.0
    best_epoch = -1
    history = []
    log_path = os.path.join(args.out, "wake_history.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["epoch", "train_loss", "val_loss", "val_tpr",
                                 "val_far", "threshold", "lr", "seconds"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meter = AverageMeter()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = bce(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            meter.update(loss.item(), xb.size(0))

        scores, labels = collect_scores(model, val_loader, device)
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(scores), torch.from_numpy(labels).float()).item())
        thr = pick_threshold(scores, labels)
        tpr, far = metrics_at_threshold(scores, labels, thr)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": meter.avg,
                        "val_loss": val_loss, "tpr": tpr, "far": far,
                        "threshold": thr, "lr": lr_now, "seconds": dt})
        print(f"[{epoch:03d}/{args.epochs}] loss={meter.avg:.4f} "
              f"val_loss={val_loss:.4f} TPR@FAR<=0.1%={tpr:.4f} far={far:.4f} "
              f"thr={thr:.3f} ({dt:.1f}s)", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [epoch, round(meter.avg, 6), round(val_loss, 6), round(tpr, 6),
                 round(far, 6), round(thr, 6), f"{lr_now:.3e}", round(dt, 2)])

        if tpr > best + 1e-4:
            best, best_epoch = tpr, epoch
            torch.save({
                "model": model.state_dict(),
                "threshold": thr,
                "val_tpr": tpr,
                "val_far": far,
                "epoch": epoch,
                "feat_dir": args.feat,
            }, os.path.join(args.out, "wake_best.pt"))
        elif epoch - best_epoch > args.patience:
            print(f"[wake] early stop at epoch {epoch}", flush=True)
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_tpr_at_far": best,
        "epochs_run": len(history),
        "parameters": count_parameters(model),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "final_val": history[-1] if history else None,
    }
    with open(os.path.join(args.out, "wake_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

