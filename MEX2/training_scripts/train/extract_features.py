"""Extract fixed-length log-mel features for every command utterance.

Reads `manifest.csv` of the OptionB dataset and writes float16 arrays that
`train_command.py` / `evaluate.py` consume.

    python extract_features.py --dataset $DATA --out $FEAT --jobs 24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F          # noqa: E402
from rapi_vcm.keywords import (             # noqa: E402
    KEYWORDS, keywords_to_multihot, transcript_to_keywords,
)
from rapi_vcm.labels import (               # noqa: E402
    INTENTS, SLOT_CLASSES, folder_to_intent,
)

SPLIT_MAP = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}


def read_manifest(dataset: str):
    rows = []
    with open(os.path.join(dataset, "manifest.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            path = os.path.join(dataset, r["path"])
            if not os.path.exists(path):
                continue
            rows.append(r)
    return rows


def _one(path_frames):
    path, seconds = path_frames
    try:
        x = F.load_audio(path)
        spec = F.logmel(x)
        true_len = spec.shape[1]
        feats = F.pad_crop(F.cmvn(spec),
                           int(seconds * F.SAMPLE_RATE / F.HOP_LENGTH))
        return feats.astype(np.float16), true_len
    except Exception as exc:                                   # pragma: no cover
        print(f"FAILED {path}: {exc}", file=sys.stderr)
        return None, -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=F.CMD_SECONDS)
    args = ap.parse_args()

    rows = read_manifest(args.dataset)
    if not rows:
        raise SystemExit("manifest.csv not found or empty")

    os.makedirs(args.out, exist_ok=True)
    n = len(rows)
    frames = int(args.seconds * F.SAMPLE_RATE / F.HOP_LENGTH)
    X = np.zeros((n, F.N_MELS, frames), dtype=np.float16)
    intent = np.zeros(n, dtype=np.int64)
    slot = np.zeros(n, dtype=np.int64)
    kw = np.zeros((n, len(KEYWORDS)), dtype=np.uint8)
    label = np.zeros(n, dtype=np.int64)
    split = np.zeros(n, dtype=np.int64)
    true_len = np.zeros(n, dtype=np.int32)

    folders = sorted({r["label"] for r in rows})
    folder_to_id = {f: i for i, f in enumerate(folders)}

    jobs = [(os.path.join(args.dataset, r["path"]), args.seconds) for r in rows]
    print(f"[extract] {n} files -> {frames} frames, workers={args.jobs}", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (feats, tl) in enumerate(ex.map(_one, jobs, chunksize=32)):
            if feats is None:
                continue
            r = rows[i]
            X[i] = feats
            true_len[i] = tl
            intent[i] = INTENTS.index(folder_to_intent(r["label"]))
            label[i] = folder_to_id[r["label"]]
            sv = (r.get("slot_value") or "").strip()
            slot[i] = SLOT_CLASSES.index(sv) if sv in SLOT_CLASSES else len(SLOT_CLASSES) - 1
            kw[i] = keywords_to_multihot(
                transcript_to_keywords(r.get("transcript") or ""))
            split[i] = SPLIT_MAP.get((r.get("split") or "train").lower(), 0)
            done += 1
            if done % 2000 == 0:
                print(f"[extract] {done}/{n}", flush=True)

    keep = true_len > 0
    X, intent, slot, kw, label, split, true_len = (
        a[keep] for a in (X, intent, slot, kw, label, split, true_len))

    np.save(os.path.join(args.out, "cmd_X.npy"), X)
    np.save(os.path.join(args.out, "cmd_intent.npy"), intent)
    np.save(os.path.join(args.out, "cmd_slot.npy"), slot)
    np.save(os.path.join(args.out, "cmd_kw.npy"), kw)
    np.save(os.path.join(args.out, "cmd_label.npy"), label)
    np.save(os.path.join(args.out, "cmd_split.npy"), split)
    np.save(os.path.join(args.out, "cmd_len.npy"), true_len)

    durations = [float(r["duration_sec"]) for r, k in zip(rows, keep) if k]
    meta = {
        "n_files": int(X.shape[0]),
        "shape": list(X.shape),
        "seconds": args.seconds,
        "frames": frames,
        "folders": folders,
        "intents": INTENTS,
        "keywords": KEYWORDS,
        "slot_classes": SLOT_CLASSES,
        "splits": {str(int(k)): int(v)
                   for k, v in zip(*np.unique(split, return_counts=True))},
        "duration_sec": {
            "mean": float(np.mean(durations)),
            "p50": float(np.percentile(durations, 50)),
            "p95": float(np.percentile(durations, 95)),
            "max": float(np.max(durations)),
        },
        "frames_true": {
            "mean": float(true_len.mean()),
            "p95": float(np.percentile(true_len, 95)),
            "max": int(true_len.max()),
        },
    }
    with open(os.path.join(args.out, "cmd_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta["splits"], indent=2))
    print(f"[extract] wrote {X.shape} to {args.out}")


if __name__ == "__main__":
    main()
