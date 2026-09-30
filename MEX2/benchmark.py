#!/usr/bin/env python
"""Benchmark suite for the Rapi voice assistant.

Produces a reproducible report a tester can run on BOTH a desktop and the
Raspberry Pi to compare performance. Measures:

  1. Classification accuracy  - on the held-out test split (clean + noisy)
  2. Per-intent recall / confusion - which intents get confused
  3. Inference latency         - feature extraction + classification time
                                 (mean / p50 / p95 / p99) in ms
  4. Wake-word latency         - gate + GMM scoring time per window
  5. Model size + memory       - bytes on disk, RSS while classifying
  6. Throughput                - utterances/second on this machine

Output:
  reports/benchmark.json   (machine-readable)
  reports/benchmark.txt    (human-readable)

Usage:
    python benchmark.py [--n N] [--data-dir DIR]

Defaults to N=300 files per intent (use the same data you trained on).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voice_assistant import features as F  # noqa: E402
from voice_assistant.classifier import CommandClassifier  # noqa: E402
from voice_assistant.wakeword import WakeWordDetector  # noqa: E402

MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")


def load_manifest(data_dir):
    return list(csv.DictReader(open(os.path.join(data_dir, "manifest.csv"))))


def select_test_files(rows, max_per_intent, seed=0):
    rng = np.random.default_rng(seed)
    by_intent = defaultdict(list)
    for r in rows:
        if r["split"] == "test":
            by_intent[r["intent"]].append(r)
    chosen = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        chosen.extend(items[:max_per_intent])
    return chosen


def rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        try:
            import psutil, os
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--n", type=int, default=300, help="test files per intent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load models
    cc = CommandClassifier(os.path.join(MODELS_DIR, "commands.joblib"))
    assert cc.load(), "commands.joblib not found - run train.py first"
    ww = WakeWordDetector(os.path.join(MODEL_DIR := MODELS_DIR, "wakeword_synthetic.joblib"))
    assert ww.load(), "wakeword_synthetic.joblib not found - run make_wake_data.py first"

    rows = load_manifest(args.data_dir)
    test_rows = select_test_files(rows, args.n, args.seed)
    print(f"Evaluating {len(test_rows)} held-out test files "
          f"({len(set(r['intent'] for r in test_rows))} intents)")

    # ---- Accuracy + latency -------------------------------------------- #
    feats_cache = {}
    lat_extract = []
    lat_classify = []
    preds, truths = [], []
    for r in test_rows:
        path = os.path.join(args.data_dir, r["path"].replace("/", os.sep))
        if not os.path.exists(path):
            continue
        te = time.perf_counter()
        x = F.load_audio(path)
        feats = F.extract_features(x)
        lat_extract.append((time.perf_counter() - te) * 1000)
        tc = time.perf_counter()
        intent, conf, _ = cc.predict(feats)
        lat_classify.append((time.perf_counter() - tc) * 1000)
        preds.append(intent)
        truths.append(r["intent"])

    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    acc = correct / len(truths) if truths else 0.0

    # Per-intent recall
    intents = sorted(set(truths))
    recall = {}
    for i in intents:
        tp = sum(1 for p, t in zip(preds, truths) if p == i and t == i)
        tot = sum(1 for t in truths if t == i)
        recall[i] = round(tp / tot, 3) if tot else None

    # Confusion
    cm = np.zeros((len(intents), len(intents)), dtype=int)
    for p, t in zip(preds, truths):
        cm[intents.index(t), intents.index(p)] += 1

    # ---- Wake-word latency --------------------------------------------- #
    # Simulate a stream: score a 1.0 s window repeatedly.
    dummy = np.random.randn(int(1.0 * F.SAMPLE_RATE)) * 0.01
    ww_lat = []
    for _ in range(50):
        tw = time.perf_counter()
        ww.detect_window(dummy)
        ww_lat.append((time.perf_counter() - tw) * 1000)

    # ---- Model size + memory ------------------------------------------- #
    size_cmd = os.path.getsize(os.path.join(MODELS_DIR, "commands.joblib"))
    size_ww = os.path.getsize(os.path.join(MODELS_DIR, "wakeword_synthetic.joblib"))
    mem = rss_mb()

    def stats(arr):
        a = np.array(arr)
        return {"mean": round(float(a.mean()), 2),
                "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2),
                "p99": round(float(np.percentile(a, 99)), 2),
                "max": round(float(a.max()), 2)}

    throughput = len(truths) / (sum(lat_extract) / 1000 + sum(lat_classify) / 1000) \
        if truths else 0.0

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": _platform(),
        "n_test_files": len(truths),
        "n_intents": len(intents),
        "overall_accuracy": round(acc, 4),
        "per_intent_recall": recall,
        "latency_ms": {
            "feature_extraction": stats(lat_extract),
            "classification": stats(lat_classify),
            "wake_word_window": stats(ww_lat),
        },
        "throughput_utts_per_sec": round(throughput, 2),
        "model_size_bytes": {"commands": size_cmd, "wakeword": size_ww},
        "peak_memory_mb": round(mem, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
    json.dump(report, open(os.path.join(REPORTS_DIR, "benchmark.json"), "w"),
              indent=2)

    # Human readable
    with open(os.path.join(REPORTS_DIR, "benchmark.txt"), "w") as fh:
        fh.write("RAPI VOICE ASSISTANT - BENCHMARK REPORT\n")
        fh.write("=" * 56 + "\n")
        fh.write(f"Platform            : {report['platform']}\n")
        fh.write(f"Generated           : {report['generated_at']}\n")
        fh.write(f"Test files          : {len(truths)}  ({len(intents)} intents)\n")
        fh.write(f"OVERALL ACCURACY    : {acc*100:.2f}%\n\n")
        fh.write("Latency (ms):\n")
        for k, v in report["latency_ms"].items():
            fh.write(f"  {k:22s} mean={v['mean']:7.2f}  p50={v['p50']:7.2f}  "
                     f"p95={v['p95']:7.2f}  p99={v['p99']:7.2f}  max={v['max']:7.2f}\n")
        fh.write(f"\nThroughput          : {throughput:.1f} utterances/sec\n")
        fh.write(f"Model size          : commands={size_cmd/1024:.0f} KB, "
                 f"wakeword={size_ww/1024:.0f} KB\n")
        fh.write(f"Peak memory         : {mem:.0f} MB\n\n")
        fh.write("Per-intent recall:\n")
        for i in intents:
            v = recall[i]
            bar = "#" * int((v or 0) * 30)
            fh.write(f"  {i:18s} {(v if v is not None else 0):.3f}  {bar}\n")
        fh.write(f"\nTotal time          : {report['total_seconds']}s\n")

    print("\n" + "=" * 56)
    print(f"OVERALL ACCURACY: {acc*100:.2f}%  ({len(truths)} test files)")
    print(f"Feature extraction : {stats(lat_extract)['mean']:.1f} ms mean")
    print(f"Classification     : {stats(lat_classify)['mean']:.1f} ms mean")
    print(f"Wake-word window   : {stats(ww_lat)['mean']:.1f} ms mean")
    print(f"Throughput         : {throughput:.1f} utts/s")
    print(f"Peak memory        : {mem:.0f} MB")
    print(f"Reports written to {REPORTS_DIR}/benchmark.{{json,txt}}")


def _platform():
    import platform
    return f"{platform.system()} {platform.release()} / Python {platform.python_version()} / " \
           f"{os.cpu_count()} cores"


if __name__ == "__main__":
    main()
