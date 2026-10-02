"""Evaluate the keyword spotter on the held-out test split.

    python evaluate_keyword.py --feat $FEAT --out $OUT

Writes into $OUT/metrics/:
    keyword_metrics.json     headline numbers (keyword level + downstream)
    keyword_per_keyword.csv  precision / recall / F1 per keyword
    keyword_confusion.png    downstream 19-class confusion

The keyword set is turned back into the 19-class command with
`rapi_vcm.keywords.keywords_to_intent`, which is what the system outputs, so
both levels are reported. The sigmoid threshold is tuned on the *validation*
split (never on the test split).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm.datasets import KeywordDataset             # noqa: E402
from rapi_vcm.keywords import KEYWORDS, keywords_to_intent  # noqa: E402
from rapi_vcm.labels import INTENTS, SLOT_CLASSES        # noqa: E402
from rapi_vcm.models import build_keyword                # noqa: E402

NONE_IDX = len(SLOT_CLASSES) - 1
GRID = [round(x, 3) for x in np.arange(0.20, 0.81, 0.05)]


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_keyword().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device):
    prob, gold_kw, gold_i, gold_s = [], [], [], []
    for xb, kw, yi, ys in loader:
        prob.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        gold_kw.append(kw.numpy())
        gold_i.append(yi.numpy())
        gold_s.append(ys.numpy())
    return (np.concatenate(prob), np.concatenate(gold_kw),
            np.concatenate(gold_i), np.concatenate(gold_s))


def downstream(prob, thr, gold_i, gold_s):
    """keyword probabilities -> (intent accuracy, slot accuracies, preds).

    `thr` may be a scalar or a per-keyword array.
    """
    pred_i = np.empty(len(prob), dtype=np.int64)
    pred_s = np.full(len(prob), NONE_IDX, dtype=np.int64)
    active = prob >= np.asarray(thr, dtype=np.float64)
    for r, row in enumerate(active):
        intent, slot = keywords_to_intent(KEYWORDS[i] for i in np.nonzero(row)[0])
        pred_i[r] = INTENTS.index(intent)
        if slot is not None:
            pred_s[r] = SLOT_CLASSES.index(slot)
    intent_acc = float((pred_i == gold_i).mean())
    mask = gold_s != NONE_IDX
    slot_acc = float((pred_s[mask] == gold_s[mask]).mean()) if mask.any() else 0.0
    return intent_acc, slot_acc, float((pred_s == gold_s).mean()), pred_i, pred_s


def tune_global(vprob, vkw, vi, vs):
    """scalar threshold with the best validation intent accuracy."""
    best = (-1.0, 0.5)
    for t in GRID:
        acc = downstream(vprob, t, vi, vs)[0]
        if acc > best[0]:
            best = (acc, t)
    return best[1], best[0]


def tune_greedy(vprob, vkw, vi, vs, start):
    """Per-keyword thresholds by coordinate ascent on *downstream intent
    accuracy* (the metric that is actually reported).

    Starts from the uniform `start`, so it can never do worse than the
    global threshold on the validation split. Two passes over the 40
    keywords x the threshold grid.
    """
    n_kw = vkw.shape[1]
    thr = np.full(n_kw, float(start), dtype=np.float64)
    best = downstream(vprob, thr, vi, vs)[0]
    for _ in range(2):
        for i in range(n_kw):
            keep_t, keep_acc = thr[i], best
            for t in GRID:
                if t == thr[i]:
                    continue
                thr[i] = t
                acc = downstream(vprob, thr, vi, vs)[0]
                if acc > keep_acc + 1e-9:
                    keep_acc, keep_t = acc, t
            thr[i], best = keep_t, keep_acc
    return thr, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=2, help="0 train 1 val 2 test")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--thr", type=float, default=None,
                    help="fixed threshold (default: tune on the val split)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = os.path.join(args.out, "metrics")
    os.makedirs(mdir, exist_ok=True)
    model, ckpt = load(os.path.join(args.out, "keyword_best.pt"), device)

    # ------------------------------------------------- threshold selection --
    thr = args.thr
    g_thr = 0.5
    per_kw = None
    thr_source = "fixed (--thr)"
    if thr is None:
        val_ds = KeywordDataset(args.feat, split=1, augment=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4)
        vprob, vgkw, vi, vs = predict(model, val_dl, device)
        g_thr, g_acc = tune_global(vprob, vgkw, vi, vs)
        p_thr, p_acc = tune_greedy(vprob, vgkw, vi, vs, g_thr)
        if p_acc > g_acc + 1e-9:
            thr, per_kw = p_thr, p_thr
            thr_source = f"per-keyword (greedy, intent acc), val " \
                         f"{p_acc:.4f} (global {g_acc:.4f} @ {g_thr:.2f})"
        else:
            thr = g_thr
            thr_source = f"global, val intent acc {g_acc:.4f} " \
                         f"(greedy {p_acc:.4f})"

    # ------------------------------------------------------------ test set --
    ds = KeywordDataset(args.feat, split=args.split, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    prob, gkw, gi, gs = predict(model, dl, device)
    pred = prob >= thr

    tp = (pred & (gkw >= 0.5)).sum(0)
    fp = (pred & (gkw < 0.5)).sum(0)
    fn = (~pred & (gkw >= 0.5)).sum(0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    seen = (tp + fn) > 0
    exact = float((pred == (gkw >= 0.5)).all(1).mean())

    intent_acc, slot_acc, slot_acc_all, pred_i, pred_s = downstream(
        prob, thr, gi, gs)
    mask = gs != NONE_IDX

    print(f"[keyword] split={args.split} n={len(ds)} "
          f"thr={'per-keyword' if per_kw is not None else round(float(thr), 2)}"
          f" ({thr_source})", flush=True)
    print(f"[keyword] micro P/R={tp.sum() / max(tp.sum() + fp.sum(), 1):.4f}/"
          f"{tp.sum() / max(tp.sum() + fn.sum(), 1):.4f} "
          f"macro_f1={f1[seen].mean():.4f} exact={exact:.4f} "
          f"intent_acc={intent_acc:.4f} slot_acc={slot_acc:.4f}", flush=True)

    with open(os.path.join(mdir, "keyword_per_keyword.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["keyword", "precision", "recall", "f1", "support"])
        for i, name in enumerate(KEYWORDS):
            w.writerow([name, round(float(prec[i]), 4), round(float(rec[i]), 4),
                        round(float(f1[i]), 4), int(tp[i] + fn[i])])

    # downstream per-class table + confusion matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    tp_c = np.zeros(len(INTENTS)); fp_c = np.zeros(len(INTENTS))
    fn_c = np.zeros(len(INTENTS))
    for t, p in zip(gi, pred_i):
        if t == p:
            tp_c[t] += 1
        else:
            fp_c[p] += 1
            fn_c[t] += 1
    p_c = tp_c / np.maximum(tp_c + fp_c, 1)
    r_c = tp_c / np.maximum(tp_c + fn_c, 1)
    f_c = 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9)

    with open(os.path.join(mdir, "keyword_per_class.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["intent", "precision", "recall", "f1", "support"])
        for i, name in enumerate(INTENTS):
            w.writerow([name, round(float(p_c[i]), 4), round(float(r_c[i]), 4),
                        round(float(f_c[i]), 4), int(tp_c[i] + fn_c[i])])

    cm = confusion_matrix(gi, pred_i, labels=list(range(len(INTENTS))))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(INTENTS)), INTENTS, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Keyword spotter -> intent (split={args.split}, "
                 f"acc={intent_acc:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(mdir, "keyword_confusion.png"), dpi=130)
    plt.close(fig)

    metrics = {
        "split": args.split,
        "n_samples": int(len(ds)),
        "threshold": float(g_thr),
        "threshold_source": thr_source,
        "keyword_thresholds": ([round(float(t), 3) for t in per_kw]
                               if per_kw is not None else None),
        "keyword_micro_precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
        "keyword_micro_recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
        "keyword_macro_f1": float(f1[seen].mean()),
        "keyword_exact_match": exact,
        "intent_accuracy": intent_acc,
        "slot_accuracy": slot_acc,
        "slot_accuracy_all_including_none": slot_acc_all,
        "n_slot_samples": int(mask.sum()),
        "macro_f1_downstream": float(np.mean(f_c)),
        "best_epoch": ckpt.get("epoch"),
        "per_keyword_f1": {n: round(float(f1[i]), 4)
                           for i, n in enumerate(KEYWORDS)},
        "per_class_f1": {n: round(float(f_c[i]), 4)
                         for i, n in enumerate(INTENTS)},
        "per_class_support": {n: int(tp_c[i] + fn_c[i])
                              for i, n in enumerate(INTENTS)},
    }
    with open(os.path.join(mdir, "keyword_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    baseline_path = os.path.join(mdir, "metrics.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as fh:
            base = json.load(fh).get("command", {})
        if base:
            metrics["baseline_softmax"] = {
                "intent_accuracy": base.get("intent_accuracy"),
                "macro_f1": base.get("macro_f1"),
                "slot_accuracy": base.get("slot_accuracy"),
            }
            print(f"[compare] softmax baseline intent="
                  f"{base.get('intent_accuracy')} vs keyword "
                  f"{intent_acc:.4f}", flush=True)
            with open(os.path.join(mdir, "keyword_metrics.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)

    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
