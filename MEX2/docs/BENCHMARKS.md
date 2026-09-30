# Benchmarks

Reproduce on any machine:

```bash
python  runtime/benchmark.py --model output/onnx --label pc
python3 runtime/benchmark.py --model output/onnx --label rpi4 --threads 2
```

Each run writes `output/onnx/benchmark_<label>.json` (full detail: environment,
latency percentiles, throughput, memory, and the accuracy metrics for the
*shipped* head — `output/metrics/keyword_metrics.json`, or `metrics.json`
when the model card says `head: command`).

---

## 1. Measured — PC (Windows 11, CPU only)

Run 2026-09-29, `output/onnx/benchmark_pc.json`:

| metric | value |
|---|---:|
| platform | Windows 11 (10.0.26200) |
| CPU | Intel Family 6 Model 165, 12 logical cores |
| Python / onnxruntime | 3.13.7 / 1.30.0 |
| threads | ORT default (all cores) |
| **model load** | **209 ms** |
| `wake.onnx` size | **113.0 KB** |
| `keyword.onnx` size | **507.4 KB** |
| `command.onnx` size | 506.9 KB |
| parameters (wake / keyword / command) | 23 665 / 123 496 / 123 303 |
| wake inference p50 / p99 | **2.54 / 3.40 ms** |
| keyword inference p50 / p99 | **7.69 / 10.06 ms** |
| feature extraction, 2.5 s audio (p50) | 4.17 ms |
| full wake scan of 4 s audio (p50) | 32.76 ms |
| wake + command, end-to-end (est.) | **14.5 ms** |
| real-time factor (audio sec / wall sec) | 0.0028 |
| throughput | 393 wake inferences/s, 130 commands/s |
| **peak RSS** | **140.2 MB** (whole Python process) |

Notes:

* "command/keyword inference" times `pipe.classify()` — with the shipped
  model card that is one `keyword.onnx` forward pass plus the rule (rule cost
  is negligible, < 0.01 ms).
* A single wake scan = feature extraction (1.2 s window) + `wake.onnx`
  (~3 ms) → ~6 ms. Scanning every 100 ms costs well under 10 % of one core.
* "wake + command est." = one wake inference + one command inference +
  their feature extraction — the per-event cost once the wake word fires.
* Accuracy attached to this run (keyword head): intent 99.24 %, slot 99.29 %,
  keyword macro F1 0.9912, wake TPR 99.75 % @ FAR 0.11 %
  (see [`ACCURACY.md`](ACCURACY.md)).

---

## 2. Raspberry Pi 4 (4 GB) — how to measure

No Pi is attached to this workstation, so **no Pi numbers are claimed here**.
Run the same script on the Pi and paste the table:

```bash
# on the Raspberry Pi 4, 64-bit Raspberry Pi OS
scp -r output/ rapi@<pi>:~/rapi_vcm/output/
scp -r rapi_vcm/ runtime/ requirements_pi.txt rapi@<pi>:~/rapi_vcm/
ssh rapi@<pi>
sudo apt update && sudo apt install -y python3-pip libportaudio2 libsndfile1
pip3 install -r ~/rapi_vcm/requirements_pi.txt
cd ~/rapi_vcm
python3 runtime/benchmark.py --model output/onnx --label rpi4 --threads 2
```

Then append the generated `output/onnx/benchmark_rpi4.json` summary here.

What to expect (engineering estimate, **not measured**): the Pi 4's Cortex-A72
is roughly 4–8× slower per core than the desktop part above at this workload,
so order-of-magnitude expectations are:

| quantity | estimate (unverified) |
|---|---:|
| wake inference | 10 – 25 ms |
| keyword inference | 40 – 90 ms |
| feature extraction (2.5 s) | 15 – 30 ms |
| peak RSS | well under 200 MB |

Even at the pessimistic end the pipeline stays far below real time: the
always-listening loop only needs one wake inference per 100 ms hop, and a
command inference runs once per event. Use `--threads 2` so the Pi stays
responsive.

---

## 3. Sizes and artifacts

| artifact | size | parameters |
|---|---:|---:|
| `wake.onnx` | 113.0 KB | 23 665 |
| `keyword.onnx` (shipped) | 507.4 KB | 123 496 |
| `command.onnx` (baseline) | 506.9 KB | 123 303 |
| all three + `model_card.json` | ~1 140 KB | 270 464 |
| PyTorch checkpoints (`*_best.pt`) | ~508 KB each | training only |

The full deployment payload (`output/onnx/`) is ~1.1 MB — no torch, librosa,
numba or model-server required at inference time. Only `wake.onnx` +
`keyword.onnx` are needed to run the shipped pipeline; `command.onnx` can be
deleted for the baseline comparison.

---

## 4. Reproducing a full run

| step | command | host |
|---|---|---|
| train + evaluate + export | `bash ~/rapi_vcm/run_keyword.sh && bash ~/rapi_vcm/run_eval_keyword.sh` | DGX |
| copy artifacts | `python ssh_helper.py get .../out/onnx output/onnx` | PC |
| local benchmarks | `python runtime/benchmark.py --model output/onnx --label pc` | PC |
| Pi benchmarks | `python3 runtime/benchmark.py --model output/onnx --label rpi4 --threads 2` | Pi 4 |
