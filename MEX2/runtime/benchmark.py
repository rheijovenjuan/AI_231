"""Benchmarks for the Rapi voice-command models (run on PC *and* on the Pi).

    python runtime/benchmark.py --model output/onnx --label pc
    python runtime/benchmark.py --model output/onnx --label rpi4 --threads 2

Writes `benchmark_<label>.json` next to the model and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

try:                                   # Unix (DGX, Raspberry Pi)
    import resource

    def peak_rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:                    # Windows
    def peak_rss_mb() -> float:
        try:
            import ctypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PrivateUsage", ctypes.c_size_t)]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            for dll, fn in (("kernel32", "K32GetProcessMemoryInfo"),
                            ("psapi", "GetProcessMemoryInfo")):
                try:
                    f = getattr(ctypes.windll[dll], fn)
                except AttributeError:
                    continue
                f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
                f.restype = ctypes.c_int
                if f(handle, ctypes.byref(c), c.cb):
                    return c.PeakWorkingSetSize / (1024.0 * 1024.0)
            return -1.0
        except Exception:
            return -1.0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F                     # noqa: E402
from runtime.pipeline import VoicePipeline             # noqa: E402


def _stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p90_ms": round(float(np.percentile(a, 90)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(a.min()), 3),
            "max_ms": round(float(a.max()), 3),
            "iters": int(a.size)}


def bench_fn(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return _stats(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="output/onnx")
    ap.add_argument("--label", default="run")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--metrics", default=None,
                    help="metrics.json produced by train/evaluate.py")
    args = ap.parse_args()

    t_load = time.perf_counter()
    pipe = VoicePipeline(args.model, threads=args.threads)
    load_s = time.perf_counter() - t_load

    rng = np.random.default_rng(0)
    cmd_wave = rng.standard_normal(int(F.CMD_SECONDS * F.SAMPLE_RATE)).astype(np.float32) * 0.1
    wake_wave = rng.standard_normal(int(F.WAKE_SECONDS * F.SAMPLE_RATE)).astype(np.float32) * 0.1
    long_wave = rng.standard_normal(int(F.SAMPLE_RATE * 4)).astype(np.float32) * 0.1

    # ------------------------------------------------------------- latency ----
    t_feat = bench_fn(lambda: pipe._feats(cmd_wave, pipe.cmd_frames), args.iters)
    t_wake = bench_fn(lambda: pipe.wake_score(wake_wave), args.iters)
    t_cmd = bench_fn(lambda: pipe.classify(cmd_wave), args.iters)

    def _scan():
        pipe.scan(long_wave, hop=0.25)
    t_scan = bench_fn(_scan, max(args.iters // 10, 10), warmup=3)

    # end-to-end = feature extraction for both stages + two inferences
    t_e2e_mean = t_feat["mean_ms"] + t_wake["mean_ms"] + t_cmd["mean_ms"]

    # --------------------------------------------------------------- files ----
    onnx_dir = pipe.onnx_dir
    sizes = {}
    for name in ("wake.onnx", "keyword.onnx", "command.onnx"):
        p = os.path.join(onnx_dir, name)
        if os.path.exists(p):
            sizes[name] = round(os.path.getsize(p) / 1024.0, 1)

    params = pipe.card.get("parameters", {})
    peak_rss_mb_value = peak_rss_mb()

    # accuracy: the shipped head determines which metrics file to quote
    mroot = os.path.join(os.path.dirname(onnx_dir), "metrics")
    head = pipe.card.get("head", "command")
    if args.metrics:
        mpath = args.metrics
    else:
        name = "keyword_metrics.json" if head == "keyword" else "metrics.json"
        mpath = os.path.join(mroot, name)
    accuracy = None
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            accuracy = json.load(fh)
    wake_metrics = None                       # wake numbers always live in metrics.json
    wpath = os.path.join(mroot, "metrics.json")
    if os.path.exists(wpath):
        with open(wpath, encoding="utf-8") as fh:
            wake_metrics = json.load(fh).get("wake")

    results = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
            "onnxruntime": _ort_version(),
            "threads": args.threads,
            "cpu_count": os.cpu_count(),
        },
        "artifacts": {
            "files_kb": sizes,
            "parameters": params,
            "model_dir": os.path.abspath(onnx_dir),
            "load_seconds": round(load_s, 3),
        },
        "latency": {
            "feature_extract_cmd": t_feat,
            "wake_inference": t_wake,
            "command_inference": t_cmd,
            "scan_4s_audio": t_scan,
            "wake_plus_command_est_ms": round(t_e2e_mean, 3),
        },
        "throughput": {
            "wake_inferences_per_s": round(1000.0 / max(t_wake["mean_ms"], 1e-6), 1),
            "commands_per_s": round(1000.0 / max(t_cmd["mean_ms"], 1e-6), 1),
        },
        "memory": {"peak_rss_mb": round(peak_rss_mb_value, 1)},
        "accuracy": accuracy,
        "wake_accuracy": wake_metrics,
        "real_time_factor": round((t_wake["mean_ms"] + t_cmd["mean_ms"]) /
                                  (1000.0 * (F.WAKE_SECONDS + F.CMD_SECONDS)), 5),
    }

    out = os.path.join(onnx_dir, f"benchmark_{args.label}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------------------------------------------- report -----
    print(f"\n### Benchmark - {args.label}\n")
    print(f"| metric | value |")
    print(f"|---|---:|")
    print(f"| platform | {results['environment']['platform']} |")
    print(f"| onnxruntime | {results['environment']['onnxruntime']} |")
    print(f"| threads | {args.threads or 'default'} |")
    print(f"| model load | {load_s * 1000:.1f} ms |")
    for name, kb in sizes.items():
        print(f"| {name} size | {kb} KB |")
    print(f"| parameters (wake/kw/cmd) | {params.get('wake', '-')} / "
          f"{params.get('keyword', '-')} / {params.get('command', '-')} |")
    print(f"| wake inference p50 / p99 | {t_wake['p50_ms']} / {t_wake['p99_ms']} ms |")
    print(f"| command inference p50 / p99 | {t_cmd['p50_ms']} / {t_cmd['p99_ms']} ms |")
    print(f"| feature extraction (2.5 s) p50 | {t_feat['p50_ms']} ms |")
    print(f"| full scan of 4 s audio p50 | {t_scan['p50_ms']} ms |")
    print(f"| wake + command est. end-to-end | {t_e2e_mean:.2f} ms |")
    print(f"| real-time factor | {results['real_time_factor']} |")
    print(f"| peak RSS | {results['memory']['peak_rss_mb']} MB |")
    if accuracy:
        if head == "keyword":
            print(f"| test intent accuracy (keyword head) | "
                  f"{accuracy.get('intent_accuracy')} |")
            print(f"| test slot accuracy | {accuracy.get('slot_accuracy')} |")
            print(f"| keyword macro F1 | {accuracy.get('keyword_macro_f1')} |")
        else:
            c = accuracy.get("command", {})
            print(f"| test intent accuracy | {c.get('intent_accuracy')} |")
            print(f"| test slot accuracy | {c.get('slot_accuracy')} |")
        w = wake_metrics or {}
        print(f"| wake TPR @ thr | {w.get('tpr_at_threshold')} |")
        print(f"| wake FAR @ thr | {w.get('far_at_threshold')} |")
    print(f"\nwritten: {out}\n")


def _ort_version():
    try:
        import onnxruntime as ort
        return ort.__version__
    except Exception:
        return "n/a"


if __name__ == "__main__":
    main()
