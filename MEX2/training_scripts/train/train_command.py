"""Train the 19-class command classifier (+ slot head) on a single GPU.

    source ~/rapi_vcm/venv/bin/activate
    python train_command.py --feat $FEAT --out $OUT --gpu 1

`--gpu` selects exactly ONE of the 8 GPUs on the DGX node: the script pins
CUDA_VISIBLE_DEVICES to that index before torch is imported, so no other GPU
is ever touched.
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

from rapi_vcm.datasets import AverageMeter, CommandDataset     # noqa: E402
from rapi_vcm.labels import INTENTS, SLOT_CLASSES              # noqa: E402
from rapi_vcm.models import build_command, count_parameters    # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feat", required=True, help="feature directory")
    p.add_argument("--out", required=True, help="checkpoint directory")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--slot-weight", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--patience", type=int, default=15)
    return p.parse_args()


def evaluate(model, loader, device, slot_weight):
    model.eval()
    intent_ok = slot_ok = slot_ok_when_intent_ok = slot_relevant = 0
    n = 0
    loss_m = AverageMeter()
    ce = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for xb, yi, ys in loader:
            xb = xb.to(device, non_blocking=True)
            yi = yi.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            li, ls = model(xb)
            loss = ce(li, yi).mean() + slot_weight * ce(ls, ys).mean()
            loss_m.update(loss.item(), xb.size(0))
            pi = li.argmax(1)
            ps = ls.argmax(1)
            intent_ok += (pi == yi).sum().item()
            slot_ok += (ps == ys).sum().item()
            mask = ys != (len(SLOT_CLASSES) - 1)
            slot_relevant += mask.sum().item()
            slot_ok_when_intent_ok += (((ps == ys) & (pi == yi) & mask).sum().item())
            n += xb.size(0)
    return {
        "loss": loss_m.avg,
        "intent_acc": intent_ok / max(n, 1),
        "slot_acc_all": slot_ok / max(n, 1),
        "slot_acc_relevant": slot_ok_when_intent_ok / max(slot_relevant, 1),
        "n": n,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={device} n_gpu={torch.cuda.device_count()}", flush=True)
    if device.type == "cuda":
        print(f"[train] using GPU 0 -> {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    train_ds = CommandDataset(args.feat, split=0, augment=True)
    val_ds = CommandDataset(args.feat, split=1, augment=False)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=len(train_ds) > args.batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_command().to(device)
    print(f"[train] parameters={count_parameters(model)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.999))
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup = max(int(0.05 * total_steps), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)

    ce = nn.CrossEntropyLoss()
    slot_w = args.slot_weight
    use_amp = device.type == "cuda"          # bfloat16 autocast, no loss scaling

    best = -1.0
    best_epoch = -1
    history = []
    log_path = os.path.join(args.out, "command_history.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            ["epoch", "train_loss", "val_loss", "intent_acc",
             "slot_acc_all", "slot_acc_relevant", "lr", "seconds"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meter = AverageMeter()
        for xb, yi, ys in train_loader:
            xb = xb.to(device, non_blocking=True)
            yi = yi.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                li, ls = model(xb)
                loss = ce(li, yi) + slot_w * ce(ls, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            meter.update(loss.item(), xb.size(0))

        val = evaluate(model, val_loader, device, slot_w)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": meter.avg, **val,
                        "lr": lr_now, "seconds": dt})
        print(f"[{epoch:03d}/{args.epochs}] loss={meter.avg:.4f} "
              f"val_loss={val['loss']:.4f} intent={val['intent_acc']:.4f} "
              f"slot={val['slot_acc_relevant']:.4f} lr={lr_now:.2e} "
              f"({dt:.1f}s)", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                [epoch, round(meter.avg, 6), round(val["loss"], 6),
                 round(val["intent_acc"], 6), round(val["slot_acc_all"], 6),
                 round(val["slot_acc_relevant"], 6), f"{lr_now:.3e}",
                 round(dt, 2)])

        if val["intent_acc"] > best + 1e-4:
            best = val["intent_acc"]
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "intents": INTENTS,
                "slot_classes": SLOT_CLASSES,
                "epoch": epoch,
                "val": val,
                "feat_dir": args.feat,
            }, os.path.join(args.out, "command_best.pt"))
        elif epoch - best_epoch > args.patience:
            print(f"[train] early stop at epoch {epoch} "
                  f"(best {best:.4f} @ {best_epoch})", flush=True)
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_intent_acc": best,
        "epochs_run": len(history),
        "parameters": count_parameters(model),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "final_val": history[-1] if history else None,
    }
    with open(os.path.join(args.out, "command_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
