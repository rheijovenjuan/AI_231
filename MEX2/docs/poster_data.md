# Poster / infographic fill sheet — ME2 Voice Controlled Smart Device

Every `{TBD}` / `{fill}` / `<todo>` from the infographic template, filled with
**measured** values from this repository (run-5 models, benchmarks re-run
2026-10-02/03 on PC and Raspberry Pi 4). Sources are linked next to each block.

---

## Header

| field | value |
|---|---|
| Title | **ME2 — Voice Controlled Smart Device** |
| Subtitle | Always-on keyword + intent recognition at the edge · no cloud round-trip |
| Date | **2 October 2026** (benchmarks measured 2026-10-02/03) |
| Author / team | **Rhei Juan** (GitHub `rheijovenjuan`, Pi user `rgjuan`) |
| Badges | Model · Dataset · Training on A100 · Validation on RPi 4 + PC |

---

## Panel 1 — Model

| row (template) | fill |
|---|---|
| Mic · 16 kHz → log-mel {TBD}×{TBD} | **40 × 250** (2.5 s window; HTK mel 20–7 600 Hz, per-utterance CMVN; wake uses 40 × 120 = 1.2 s) |
| Causal encoder × {TBD} · {TBD}-dim *(template wording)* | **DS-CNN backbone, MobileNetV2-style** (depthwise-separable convs, widths 64→96→128→160→192, 192-d pooled embedding). Not causal/streaming: scores a full 2.5 s window. |
| Intent head | **40-keyword multi-label sigmoids → deterministic rule → 19 intents** (shipped `keyword.onnx`); baseline: 19-way softmax (`command.onnx`) |
| Slot head | **Rule over the keyword set** — 6 slot intents, 19 spoken values + `NONE`; baseline has a learned slot head (20 classes) |
| Actuator | Wake score ≥ **0.900** → ack beep → intent/slot → `app.py` action (music, lights, alarm, …) |
| Raspberry Pi 4 | onnxruntime 1.30.0 · `--threads 2` · payload **~1.1 MB**, no torch/librosa at inference |
| **Parameters / weights** | **0.27 M total · 1.1 MB** — wake 23 665 (113 KB) / keyword 123 496 (507 KB, shipped) / command 123 303 (507 KB) |

Sources: [`BENCHMARKS.md` §3](BENCHMARKS.md), `rapi_vcm/models.py`,
`output/onnx/model_card.json`.

---

## Panel 2 — Dataset

| row | fill |
|---|---|
| Source | **`markandrian30/AI231`** (GitHub, `MEX2/OptionB`), access 2026-09, no DOI — cite repo + date; **+ 94 real-voice clips** recorded by this project, committed under [`command_data/`](../command_data/) |
| Hours / utts | **8.8 h / 18 469 utts** (18 375 corpus + 94 real; 2 dead-air takes dropped from 96) |
| Speakers | **100 TTS + 1 real** (`real1`); test split = unseen `s91`–`s100` |
| Labels | **19 intents · 6 slots** (19 values + `NONE`) **· 40 keywords** |
| Splits | speaker-disjoint **train 14 752 / val 1 875 / test 1 842** (test untouched by real data) |

Sources: [`ACCURACY.md` §1](ACCURACY.md), [`TRAINING.md` §3+§11](TRAINING.md).

---

## Panel 3 — Training on the A100 cluster

| row | fill |
|---|---|
| Cluster | **1 × A100-SXM4-40GB** (`n003.ai.internal`), `n_gpu=1`, DGX `~/rapi_vcm` |
| Objective | keyword: **BCEWithLogitsLoss** (40 labels, `pos_weight = raw neg/pos`) → rule; baseline: **CrossEntropy** (19 intents + slots); wake: **BCEWithLogits** |
| Optimiser | **AdamW** (β = 0.9/0.999, weight decay) + **OneCycleLR** (max lr ≈ 1.6e-3 → ~0; history `lr` column) |
| Steps / loss | **60 epochs, best epoch 50** · final val loss **0.0345** · val intent **98.29 %** · seed **1234** (data gen 20260929) · wall ≈ **15 min** per full run (stage table in [`TRAINING.md` §10](TRAINING.md)) |
| Real-voice retrain (run-5) | 94 clips merged → vocab check 18 469/18 469 → `run_keyword.sh --reextract` + train/eval (`output/training/logs/real1_retrain.log`) |

---

## Panel 4 — Validation on the Raspberry Pi 4

*(measured 2026-10-02, run-5 ONNX — [`output/onnx/benchmark_rpi4.json`](../output/onnx/benchmark_rpi4.json))*

| row (template) | fill |
|---|---|
| Keyword / intent acc | **99.29 % / 99.46 %** (test intent / slot, n = 1 842; identical ONNX on PC and Pi) |
| False-accept rate | **0.000 %** wake FAR @ 0.900 (0 / 1 904 negative windows) · streaming false accepts **0 / 19** |
| Latency p95 / RTF | **69.3 ms** wake+command end-to-end (est.) — keyword p50 40.0 / p99 70.7 ms · **RTF 0.016** |
| Runtime | **onnxruntime 1.30.0 · 2 threads** (Pi OS 6.18 aarch64, Python 3.13.5) |

### Pi 4 detail (all measured)

| quantity | Pi 4 p50 / p99 |
|---|---:|
| model load | 649 ms |
| feature extraction (2.5 s) | 9.99 / 10.14 ms |
| wake inference | 17.18 / 32.08 ms |
| keyword inference | 39.96 / 70.73 ms |
| scan of 4 s audio | 224.9 / 286.8 ms |
| wake + command end-to-end (est.) | 69.3 ms |
| throughput | 59.8 wake/s · 23.5 cmd/s |
| peak RSS | 147.0 MB |
| real-time factor | 0.0160 |

## Panel 5 — PC (Windows 11, for comparison)

*(measured 2026-10-03, run-5 ONNX — [`output/onnx/benchmark_pc.json`](../output/onnx/benchmark_pc.json))*

| quantity | PC |
|---|---:|
| model load | 221.6 ms |
| wake inference p50 / p99 | 3.36 / 6.47 ms |
| keyword inference p50 / p99 | 9.71 / 11.86 ms |
| scan of 4 s audio p50 | 37.30 ms |
| wake + command end-to-end (est.) | 16.9 ms |
| throughput | 284.9 wake/s · 102.5 cmd/s |
| peak RSS | 139.6 MB |
| real-time factor | 0.00359 |

Headline accuracy (both hosts): intent **99.29 %**, slot **99.46 %**,
keyword macro F1 **0.9924**; baseline softmax **99.84 %** — [`ACCURACY.md` §2-3](ACCURACY.md).

---

## Panel 6 — "To Be Submitted"

| field | fill |
|---|---|
| GitHub repository | **`rheijovenjuan/AI_231`** — public, MIT |
| Dataset location | **`markandrian30/AI231`** on GitHub (`MEX2/OptionB`) — no DOI deposited; cite repo + access date (2026-09). Real-voice + wake recordings committed in this repo (`command_data/`, `wakeword_data/`) |
| A100 cluster | **`n003.ai.internal` · 1 × A100-SXM4-40GB · `n_gpu=1`** · wall ≈ 15 min/run · seeds `1234` (data `20260929`) |
| Model weights | [`output/onnx/`](../output/onnx/) (`wake/keyword/command.onnx` + `model_card.json`) and [`output/training/checkpoints/`](../output/training/checkpoints/) — MIT, committed in repo |

### Reviewer checklist

| # | item | status |
|---:|---|---|
| 1 | Repo public, one-command reproduction | **done** — `./setup_pi.sh`, `python app.py`; training `bash ~/rapi_vcm/run_keyword.sh && bash ~/rapi_vcm/run_eval_keyword.sh` |
| 2 | Dataset licensed and citable (DOI) | **done** — source repo cited + access date, no-DOI note (course corpus), [`SUBMISSION.md` §1](SUBMISSION.md) |
| 3 | Training logs + final checkpoint committed | **done** — `output/training/{logs,checkpoints,history}/`, `output/metrics/` |
| 4 | Pi 4 latency reproduced by the posted script | **done** — `python3 runtime/benchmark.py --model output/onnx --label rpi4 --threads 2` → `benchmark_rpi4.json` in repo |
| 5 | Held-out test set, unseen speakers | **done** — speaker-disjoint 70/15/15; test `s91`–`s100`; wake corpus voice-disjoint + recording-disjoint |
| 6 | Baseline of comparable size compared | **done** — `command.onnx` 123 303 vs keyword 123 496 params; test intent **99.84 % vs 99.29 %** |
