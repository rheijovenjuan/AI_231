"""Evaluate the trained command + wake models on the held-out test split.

    python evaluate.py --feat $FEAT --out $OUT

Writes into $OUT/metrics/:
    metrics.json        headline numbers
    command_per_class.csv
    command_confusion.png
    wake_roc.csv
    wake_thresholds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm.datasets import CommandDataset, WakeDataset   # noqa: E402
from rapi_vcm.labels import INTENTS, SLOT_CLASSES           # noqa: E402
from rapi_vcm.models import build_command, build_wake       # noqa: E402


def load_command(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_command().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_wake(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_wake().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def command_logits(model, loader, device):
    li, ls, yi, ys = [], [], [], []
    for xb, a, b in loader:
        xb = xb.to(device)
        i, s = model(xb)
        li.append(i.cpu().numpy()); ls.append(s.cpu().numpy())
        yi.append(a.numpy()); ys.append(b.numpy())
    return (np.concatenate(li), np.concatenate(ls),
            np.concatenate(yi), np.concatenate(ys))


@torch.no_grad()
def wake_logits(model, loader, device):
    lo, y = [], []
    for xb, yb in loader:
        lo.append(model(xb.to(device)).cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(lo), np.concatenate(y)


def prf(y_true, y_pred, n):
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n); sup = np.zeros(n)
    for t, p in zip(y_true, y_pred):
        sup[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return prec, rec, f1, sup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=2, help="0 train 1 val 2 test")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.out, "metrics")
    os.makedirs(mdir, exist_ok=True)

    # ------------------------------------------------------------- command ----
    cmd_model, cmd_ckpt = load_command(
        os.path.join(args.out, "command_best.pt"), device)
    ds = CommandDataset(args.feat, split=args.split, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    li, ls, yi, ys = command_logits(cmd_model, dl, device)
    pi, ps = li.argmax(1), ls.argmax(1)

    intent_acc = float((pi == yi).mean())
    top2 = np.argsort(-li, axis=1)[:, :2]
    top2_acc = float((top2 == yi[:, None]).any(1).mean())
    prec, rec, f1, sup = prf(yi, pi, len(INTENTS))
    macro_f1 = float(np.mean(f1))
    none_idx = len(SLOT_CLASSES) - 1
    slot_mask = ys != none_idx                       # utterances that carry a value
    slot_acc = float((ps[slot_mask] == ys[slot_mask]).mean()) if slot_mask.any() else 0.0
    slot_acc_when_intent_ok = float(
        ((ps == ys) & (pi == yi) & slot_mask).sum() / max((slot_mask & (pi == yi)).sum(), 1))
    slot_acc_all = float((ps == ys).mean())

    print(f"[command] split={args.split} n={len(ds)} "
          f"intent_acc={intent_acc:.4f} macro_f1={macro_f1:.4f} "
          f"slot_acc={slot_acc:.4f}", flush=True)

    with open(os.path.join(mdir, "command_per_class.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["intent", "precision", "recall", "f1", "support"])
        for i, name in enumerate(INTENTS):
            w.writerow([name, round(float(prec[i]), 4), round(float(rec[i]), 4),
                        round(float(f1[i]), 4), int(sup[i])])

    # confusion matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yi, pi, labels=list(range(len(INTENTS))))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(INTENTS)), INTENTS, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Command intent confusion (test, acc={intent_acc:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(mdir, "command_confusion.png"), dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------- wake -----
    wake_model, wake_ckpt = load_wake(
        os.path.join(args.out, "wake_best.pt"), device)
    wds = WakeDataset(args.feat, split=args.split, augment=False)
    wdl = DataLoader(wds, batch_size=args.batch, shuffle=False, num_workers=4)
    wl, wy = wake_logits(wake_model, wdl, device)
    wprob = 1.0 / (1.0 + np.exp(-wl))
    neg = wy == 0
    pos = wy == 1

    def tpr_far(thr):
        return (float(((wprob >= thr) & pos).sum() / max(pos.sum(), 1)),
                float(((wprob >= thr) & neg).sum() / max(neg.sum(), 1)))

    thr_best = float(wake_ckpt.get("threshold", 0.5))
    tpr_at_thr, far_at_thr = tpr_far(thr_best)

    # sweep
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197),
                                     [thr_best]]))
    rows = []
    for t in grid:
        tpr, far = tpr_far(float(t))
        rows.append((float(t), tpr, far))
    with open(os.path.join(mdir, "wake_thresholds.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "tpr", "far"])
        for t, tpr, far in rows:
            w.writerow([round(t, 4), round(tpr, 5), round(far, 6)])

    def best_tpr_at_far(target):
        ok = [(tpr, far, t) for t, tpr, far in rows if far <= target]
        if not ok:
            return {"tpr": 0.0, "far": None, "threshold": None}
        tpr, far, t = max(ok)
        return {"tpr": tpr, "far": far, "threshold": t}

    # EER: the point where FNR (= 1 - TPR) equals FPR
    seq = sorted(rows, key=lambda r: r[0])
    tprs = np.array([r[1] for r in seq]); fars = np.array([r[2] for r in seq])
    diff = tprs - (1.0 - fars)                 # TPR - (1 - FAR) = TPR + FAR - 1
    eer_idx = int(np.argmin(np.abs(diff)))
    eer = float((1.0 - tprs[eer_idx] + fars[eer_idx]) / 2.0)

    print(f"[wake] split={args.split} n={len(wds)} "
          f"TPR@thr={tpr_at_thr:.4f} FAR@thr={far_at_thr:.5f} "
          f"thr={thr_best:.3f} EER={eer:.4f}", flush=True)

    metrics = {
        "command": {
            "split": args.split,
            "n_samples": int(len(ds)),
            "intent_accuracy": intent_acc,
            "top2_accuracy": top2_acc,
            "macro_f1": macro_f1,
            "slot_accuracy": slot_acc,
            "slot_accuracy_all_including_none": slot_acc_all,
            "slot_accuracy_when_intent_correct": slot_acc_when_intent_ok,
            "n_slot_samples": int(slot_mask.sum()),
            "best_epoch": cmd_ckpt.get("epoch"),
            "per_class_f1": {n: round(float(f1[i]), 4)
                             for i, n in enumerate(INTENTS)},
            "per_class_support": {n: int(sup[i]) for i, n in enumerate(INTENTS)},
        },
        "wake": {
            "split": args.split,
            "n_samples": int(len(wds)),
            "n_positive": int(pos.sum()),
            "n_negative": int(neg.sum()),
            "threshold": thr_best,
            "tpr_at_threshold": tpr_at_thr,
            "far_at_threshold": far_at_thr,
            "tpr_at_far_1pct": best_tpr_at_far(0.01),
            "tpr_at_far_0p1pct": best_tpr_at_far(0.001),
            "tpr_at_far_0p01pct": best_tpr_at_far(0.0001),
            "equal_error_rate": eer,
            "best_epoch": wake_ckpt.get("epoch"),
        },
    }
    with open(os.path.join(mdir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
