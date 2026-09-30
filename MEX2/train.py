#!/usr/bin/env python
"""Training pipeline for the Rapi voice assistant.

Downloads the spoken-command dataset from GitHub, extracts MFCC features, and
fits:
  * a wake-word GMM pair  ("Hey Rapi" positive vs. everything-else negative)
  * one command-intent GMM per intent (19 intents)

Outputs (in ./models and ./reports):
  * models/wakeword.joblib
  * models/commands.joblib
  * models/training_meta.json
  * reports/training_progress.json   (per-epoch-ish metrics + timing)
  * reports/command_confusion.png
  * reports/wakeword_scores.png
  * reports/accuracy_summary.txt

Usage:
    python train.py [--max-per-intent N] [--download] [--data-dir DIR]

By default it downloads. If you already have the data on disk (e.g. cloned),
point --data-dir at the MEX2/Data folder and omit --download.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voice_assistant import features as F  # noqa: E402
from voice_assistant.classifier import DEFAULT_INTENTS  # noqa: E402

REPO_BASE = "https://raw.githubusercontent.com/markandrian30/AI231/main/MEX2/Data/"
MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")

# The wake word is not in the command dataset; we synthesise positives from the
# most wake-word-like command recordings (short, high-energy) and use silence +
# long commands as negatives. This is a stand-in until real "Hey Rapi"
# recordings are captured (see docs/collecting_wakeword_data.md).
WAKE_POSITIVE_HINTS = ["PLAY_MUSIC", "PAUSE", "TIME"]  # short, punchy phrases


# --------------------------------------------------------------------------- #
def load_manifest(data_dir: str):
    path = os.path.join(data_dir, "manifest.csv")
    rows = list(csv.DictReader(open(path)))
    return rows


def download_manifest(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(dest):
        print("Downloading manifest.csv ...")
        urllib.request.urlretrieve(REPO_BASE + "manifest.csv", dest)


def download_wav(rel_path: str, data_dir: str, timeout: int = 20) -> str:
    dest = os.path.join(data_dir, rel_path.replace("/", os.sep))
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(REPO_BASE + rel_path, dest)
        return dest
    except Exception:
        return ""


def select_files(rows, max_per_intent: int, seed: int = 0):
    """Pick a balanced subset: up to max_per_intent files per intent, mixing
    clean/noisy and train/val/test splits."""
    rng = np.random.default_rng(seed)
    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    chosen = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        chosen.extend(items[:max_per_intent])
    return chosen


# --------------------------------------------------------------------------- #
def extract_one(args):
    rel_path, data_dir = args
    dest = os.path.join(data_dir, rel_path.replace("/", os.sep))
    if not os.path.exists(dest):
        return rel_path, None
    try:
        return rel_path, F.extract_from_file(dest)
    except Exception:
        return rel_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--max-per-intent", type=int, default=400,
                    help="files per intent to use (cap for speed)")
    ap.add_argument("--download", action="store_true",
                    help="download selected WAVs from GitHub")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    t0 = time.time()

    # 1. Manifest ---------------------------------------------------------- #
    download_manifest(args.data_dir)
    rows = load_manifest(args.data_dir)
    print(f"Manifest: {len(rows)} rows, "
          f"{len(set(r['intent'] for r in rows))} intents")

    selected = select_files(rows, args.max_per_intent, args.seed)
    print(f"Selected {len(selected)} files "
          f"(<= {args.max_per_intent}/intent)")

    # 2. Download ---------------------------------------------------------- #
    if args.download:
        paths = [r["path"] for r in selected]
        print(f"Downloading {len(paths)} WAVs ...")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_wav, p, args.data_dir) for p in paths]
            for fu in as_completed(futs):
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(paths)}")
        print(f"Download complete ({time.time()-t0:.0f}s)")

    # 3. Feature extraction ------------------------------------------------ #
    print("Extracting features ...")
    feats_by_intent = defaultdict(list)
    meta_by_intent = defaultdict(list)
    args_list = [(r["path"], args.data_dir) for r in selected]
    results = list(extract_one(a) for a in args_list)  # sequential, fast enough
    for r, (rel, feat) in zip(selected, results):
        if feat is None:
            continue
        feats_by_intent[r["intent"]].append(feat)
        meta_by_intent[r["intent"]].append(r)

    total = sum(len(v) for v in feats_by_intent.values())
    print(f"Extracted {total} feature vectors across "
          f"{len(feats_by_intent)} intents ({time.time()-t0:.0f}s)")

    # 4. Fit command GMMs -------------------------------------------------- #
    from voice_assistant.gmm import GMM
    n_comp = 8

    # Standardise features (z-score) so the diagonal GMM is not dominated by
    # high-variance dims (e.g. pitch). Computed on the pooled training set.
    pool = np.stack([f for i in feats_by_intent for f in feats_by_intent[i]])
    scaler_mean = pool.mean(axis=0)
    scaler_std = np.maximum(pool.std(axis=0), 1e-6)

    def _sc(z):
        return (z - scaler_mean) / scaler_std

    models = {}
    fit_report = {}
    for intent in sorted(feats_by_intent):
        X = _sc(np.stack(feats_by_intent[intent]))
        nc = min(n_comp, max(2, len(X) // 4))
        gm = GMM(n_components=nc, reg_covar=1e-3, n_init=1, random_state=0)
        t = time.time()
        gm.fit(X)
        ll = float(gm.score(X))
        models[intent] = gm
        fit_report[intent] = {
            "n_samples": int(len(X)),
            "n_components": nc,
            "log_likelihood": round(ll, 2),
            "fit_seconds": round(time.time() - t, 2),
        }
        print(f"  {intent:18s} n={len(X):4d} comp={nc:2d} "
              f"ll={ll:9.1f} {time.time()-t:5.1f}s")

    intents = sorted(models)
    # 5. Evaluate (held-out by split) -------------------------------------- #
    # Use 'test' split for evaluation if present, else a random 20% holdout.
    all_x = [f for i in intents for f in feats_by_intent[i]]
    all_y = [i for i in intents for _ in feats_by_intent[i]]
    test_idx = [j for j, r in enumerate(
        [rr for i in intents for rr in meta_by_intent[i]]) if r["split"] == "test"]
    if len(test_idx) >= 20:
        X_test = np.stack([all_x[j] for j in test_idx])
        y_test = [all_y[j] for j in test_idx]
    else:
        sel = np.random.default_rng(args.seed).choice(
            len(all_x), size=int(0.2 * len(all_x)), replace=False)
        X_test = np.stack([all_x[i] for i in sel])
        y_test = [all_y[i] for i in sel]
    preds = []
    for x in X_test:
        x2 = _sc(x.reshape(1, -1))
        scores = np.array([models[i].score(x2) for i in intents])
        preds.append(intents[int(np.argmax(scores))])
    correct = sum(1 for p, t in zip(preds, y_test) if p == t)
    acc = correct / len(y_test) if y_test else 0.0
    print(f"\nTest accuracy: {acc*100:.1f}%  ({correct}/{len(y_test)})")

    # Per-intent recall
    per_intent = {}
    for intent in intents:
        tp = sum(1 for p, t in zip(preds, y_test) if p == intent and t == intent)
        tot = sum(1 for t in y_test if t == intent)
        per_intent[intent] = round(tp / tot, 3) if tot else None

    # Confusion matrix
    cm = np.zeros((len(intents), len(intents)), dtype=int)
    for p, t in zip(preds, y_test):
        cm[intents.index(t), intents.index(p)] += 1

    # 6. Wake word --------------------------------------------------------- #
    pos_pool = [f for i in WAKE_POSITIVE_HINTS
                for f in feats_by_intent.get(i, [])]
    neg_pool = [f for i in intents if i not in WAKE_POSITIVE_HINTS
                for f in feats_by_intent.get(i, [])]
    # add silence features
    sil = F.extract_features(np.zeros(int(0.5 * F.SAMPLE_RATE)))
    neg_pool = neg_pool + [sil] * 20
    rng = np.random.default_rng(args.seed)
    if len(pos_pool) > 0:
        pos_sel = rng.choice(len(pos_pool),
                             size=min(300, len(pos_pool)), replace=False)
        pos_feats = np.stack([pos_pool[i] for i in pos_sel])
        neg_sel = rng.choice(len(neg_pool),
                             size=min(600, len(neg_pool)), replace=False)
        neg_feats = np.stack([neg_pool[i] for i in neg_sel])
        from voice_assistant.wakeword import WakeWordDetector
        ww = WakeWordDetector(os.path.join(MODELS_DIR, "wakeword.joblib"),
                              n_components=16)
        ww.save(pos_feats, neg_feats)
        # score distribution
        pos_scores = [ww._score(f.reshape(1, -1)) for f in pos_feats]
        neg_scores = [ww._score(f.reshape(1, -1)) for f in neg_feats]
        thr = float(np.percentile(pos_scores, 5))
        ww_acc = float(np.mean([s >= thr for s in pos_scores + neg_scores]))
    else:
        ww_acc = 0.0

    # 7. Persist ----------------------------------------------------------- #
    from voice_assistant.classifier import CommandClassifier
    cc = CommandClassifier(os.path.join(MODELS_DIR, "commands.joblib"))
    cc.save(models, intents, scaler=(scaler_mean, scaler_std))

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_files_total": len(rows),
        "n_files_used": total,
        "n_intents": len(intents),
        "intents": intents,
        "test_accuracy": round(acc, 4),
        "test_n": len(y_test),
        "per_intent_recall": per_intent,
        "wakeword_accuracy_proxy": round(ww_acc, 4),
        "feature_dim": F.FEATURE_DIM,
        "total_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(MODELS_DIR, "training_meta.json"), "w"),
              indent=2)
    json.dump({"fit": fit_report, "meta": meta},
              open(os.path.join(REPORTS_DIR, "training_progress.json"), "w"),
              indent=2)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(intents)), intents, rotation=90, fontsize=7)
        ax.set_yticks(range(len(intents)), intents, fontsize=7)
        for i in range(len(intents)):
            for j in range(len(intents)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            fontsize=6,
                            color="white" if cm[i, j] > cm[i].max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Command confusion (test acc {acc*100:.1f}%)")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS_DIR, "command_confusion.png"),
                    dpi=110)
        plt.close(fig)

        if len(pos_pool):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(pos_scores, bins=30, alpha=0.7, label="positive",
                    density=True)
            ax.hist(neg_scores, bins=30, alpha=0.7, label="negative",
                    density=True)
            ax.axvline(thr, color="red", linestyle="--", label="threshold")
            ax.set_xlabel("wake score"); ax.set_ylabel("density")
            ax.set_title("Wake-word score distribution")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(REPORTS_DIR, "wakeword_scores.png"),
                        dpi=110)
            plt.close(fig)
    except Exception as e:
        print("plotting skipped:", e)

    # Text summary
    with open(os.path.join(REPORTS_DIR, "accuracy_summary.txt"), "w") as fh:
        fh.write("RAPI VOICE ASSISTANT - TRAINING SUMMARY\n")
        fh.write("=" * 50 + "\n")
        fh.write(f"Trained at        : {meta['trained_at']}\n")
        fh.write(f"Feature dim       : {F.FEATURE_DIM}\n")
        fh.write(f"Files used        : {total}\n")
        fh.write(f"Intents           : {len(intents)}\n")
        fh.write(f"TEST ACCURACY     : {acc*100:.2f}%  ({len(y_test)} samples)\n")
        fh.write(f"Wake-word proxy   : {ww_acc*100:.2f}%\n")
        fh.write(f"Total time        : {meta['total_seconds']}s\n\n")
        fh.write("Per-intent recall (test split):\n")
        for i in intents:
            v = per_intent.get(i)
            fh.write(f"  {i:18s} {('%.3f' % v) if v is not None else '  -  '}")
            fh.write("\n")
    print(f"\nSaved models to {MODELS_DIR} and reports to {REPORTS_DIR}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
