# Inference

Two models, one pipeline, no cloud:

```
microphone ──► 16 kHz mono float32
      │
      ├─► sliding 1.2 s window ─► wake.onnx  ── score ─┐
      │                                                 │ score < threshold → keep listening
      │      wake word accepted                         ▼
      └─► next 2.5 s of audio ──► keyword.onnx ──► 40 keyword probabilities
                                                  │  p ≥ threshold per keyword
                                                  ▼
                                   rule (rapi_vcm/keywords.py) ──► INTENT (+ slot)
```

* **Primary output = the class of the recognised speech** (one of 19 labels).
* The class is *derived*, not classified: the model only reports which of 40
  keywords crossed their threshold, and `keywords_to_intent()` maps that set
  to the intent (with priority rules for overlaps) and slot value.
* The slot value that follows `TIMER` / `ALARM` / `TEMPERATURE` /
  `BRIGHTNESS` / `COLOR` / `CREATE_REMINDER` is produced as an *extra* field
  (`--json` / `--show-slot`); it is never required to emit the class.
* `command.onnx` (softmax baseline) is still exported; if `model_card.json`
  has no `head: keyword`, the pipeline falls back to it unchanged.

---

## 1. Model files

```
output/onnx/
├── wake.onnx          binary wake-word detector
├── keyword.onnx       40-keyword multi-label head  (shipped)
├── command.onnx       19-way intent head + 20-way slot head (baseline)
├── model_card.json    input contract, labels, thresholds, feature parameters
└── benchmark_<label>.json
```

| | `wake.onnx` | `keyword.onnx` | `command.onnx` |
|---|---|---|---|
| input name | `features` | `features` | `features` |
| input shape | `(batch, 1, 40, 120)` | `(batch, 1, 40, 250)` | `(batch, 1, 40, 250)` |
| input dtype | float32 | float32 | float32 |
| outputs | `wake_logit` `(batch,)` | `keyword_logits` `(batch,40)` | `intent_logits (batch,19)`, `slot_logits (batch,20)` |
| probability | `sigmoid(logit)` | `sigmoid(logits)` ≥ threshold | `softmax(logits)` |
| parameters | 23 665 | 123 496 | 123 303 |
| opset | 17 | 17 | 17 |

`model_card.json` carries the authoritative values (`feature_config`,
`head`, `keywords`, `keyword_thresholds`, `intents`, `slot_classes`,
`wake_threshold`) — the runtime reads it instead of hard-coding anything.

---

## 2. Feature contract

`rapi_vcm/features.py` is the single source of truth. Reproduce it by hand if
you port to another language:

| step | value |
|---|---|
| 1 | mono-mix, resample to 16 000 Hz |
| 2 | STFT: 512-sample Hann window, hop 160 (32 ms / 10 ms) |
| 3 | power spectrum → 40 triangular HTK-mel filters, 20–7 600 Hz |
| 4 | `log(power_mel + 1e-6)` |
| 5 | per-utterance CMVN over the time axis: `(x-mean)/(std+1e-5)` |
| 6 | right-pad / centre-crop to 250 frames (command) or 120 frames (wake) |
| 7 | `float32`, shaped `(1, 1, 40, T)` |

Because training and inference share this module, features are bit-identical
on the DGX node, on a laptop, and on the Pi.

---

## 3. Python API

```python
from runtime.pipeline import VoicePipeline
from rapi_vcm import features as F

pipe = VoicePipeline("output/onnx")           # or "output" — both work
print(pipe.threshold)                          # wake threshold from model_card

# full recording: wake scan + keyword spot
wave = F.load_audio("recording.wav")           # 16 kHz mono float32
print(pipe.scan(wave))
# {'detected': True, 'wake_score': 0.97, 'threshold': 0.9,
#  'wake_offset_sec': 0.4, 'intent': 'TIMER', 'confidence': 0.96,
#  'keywords': [{'keyword': 'timer', 'p': 1.0},
#               {'keyword': '10 seconds', 'p': 0.98}],
#  'slot': '10 seconds', 'slot_confidence': 0.94}
print(pipe.head)                              # 'keyword' or 'intent_softmax'

# individual stages
score = pipe.wake_score(wave[:int(1.2 * 16000)])   # 0..1
result = pipe.classify(wave[:int(2.5 * 16000)])    # {'intent': ..., 'keywords': [...]}
```

`VoicePipeline(model_dir, wake_threshold=None, providers=None, threads=0)`

| argument | meaning |
|---|---|
| `model_dir` | folder with `keyword.onnx` (or its parent — `onnx/` is auto-detected) |
| `wake_threshold` | override the threshold stored in `model_card.json` |
| `providers` | ONNX Runtime providers, default `["CPUExecutionProvider"]` |
| `threads` | `intra_op_num_threads`; `0` = ORT default, use `2` on a Pi |

---

## 4. Command-line

### 4.1 PC — `runtime/pc_test.py`

```bash
python runtime/pc_test.py --model output/onnx --file clip.wav      # wake + class
python runtime/pc_test.py --model output/onnx --dir clips --no-wake # class only
python runtime/pc_test.py --model output/onnx --mic                 # live mic
python runtime/pc_test.py --model output/onnx --file clip.wav --json # JSON
python runtime/pc_test.py --model output/onnx --file clip.wav --show-slot
python runtime/pc_test.py --model output/onnx --file clip.wav --playback  # demo
```

| flag | effect |
|---|---|
| `--no-wake` | skip wake detection, classify the whole clip (batch evaluation) |
| `--json` | one JSON object per input |
| `--show-slot` | print the recognised slot value next to the class |
| `--playback` | play the clip (or the captured command with `--mic`) out loud *before* classifying |
| `--threshold 0.6` | override the wake threshold |
| `--threads 2` | ONNX Runtime threads |
| `--seconds 3` | truncate each file to the first N seconds |

Example output:

```
[clip1.wav                    ] [wake 0.93] PLAY_MUSIC         conf=0.98
[clip2.wav                    ] [wake 0.91] TIMER             conf=0.96  slot=10 seconds
[rec.wav                      ] [wake 0.31] -- no wake word --

6 files, 5 woken, 0.42s (0.070s/file)
```

### 4.2 Raspberry Pi — `runtime/pi_run.py`

```bash
python3 runtime/pi_run.py --model output/onnx              # prints ">>> CLASS"
python3 runtime/pi_run.py --model output/onnx --json       # JSON per event
python3 runtime/pi_run.py --model output/onnx --show-slot
python3 runtime/pi_run.py --model output/onnx --playback   # play the captured
                                                            # command back first
python3 runtime/pi_run.py --model output/onnx --threads 2 --vad-rms 6e-3
python3 runtime/pi_run.py --model output/onnx --led-pin 17 # activity LED
python3 runtime/pi_run.py --model output/onnx --duration 60 # self-terminating
```

| flag | default | meaning |
|---|---|---|
| `--threads` | 2 | ONNX Runtime threads (keeps the Pi responsive) |
| `--hop-ms` | 100 | wake scan every 100 ms |
| `--vad-rms` | 6e-3 | below this energy no inference runs (CPU saver) |
| `--backend` | auto | `sounddevice` (PortAudio) or `pyaudio` |
| `--device` | – | input device index/name |
| `--duration` | ∞ | stop after N seconds |
| `--playback` | off | play the recorded command out loud *before* classifying it |
| `--playback-device` | system default | output device for `--playback` |
| `--json` / `--show-slot` / `--threshold` | – | as above |

Runtime behaviour:

1. Ring buffer of 2 s is filled with 30 ms frames.
2. Every 100 ms, *if* the energy gate passes, the last 1.2 s is classified by
   `wake.onnx`.
3. On a hit (`score ≥ threshold`) it prints `WAKE <score>` (or the JSON
   `{"event":"wake"}`), then captures the following audio until 0.45 s of
   silence or 2.5 s of speech.
4. With `--playback` the input stream is paused, the captured segment is
   played through the speaker, and the input stream is restarted — so the
   audience hears exactly what the model is about to classify (no echo, no
   buffer overflow).
5. The captured segment goes to `keyword.onnx`; the keyword set is mapped to
   the class by `keywords_to_intent()`, which is printed as `>>> <CLASS>`.
6. Return to step 2.

Exit with `Ctrl-C`; a summary (`frames`, `wake_scans`, `wake_detected`,
`commands`, `uptime_sec`) is printed.

### 4.3 Benchmarks — `runtime/benchmark.py`

```bash
python  runtime/benchmark.py --model output/onnx --label pc
python3 runtime/benchmark.py --model output/onnx --label rpi4 --threads 2
```

See [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 5. Thresholds

| knob | where | effect |
|---|---|---|
| wake threshold | `model_card.json` → `wake_threshold`, or `--threshold` | higher = fewer false wakes, more missed wakes |
| keyword thresholds | `model_card.json` → `keyword_thresholds` (40 values) | higher = fewer keywords claimed; the rule only sees keywords that pass |
| confidence | `result["confidence"]` | rule-derived confidence = max keyword probability of the winning keywords |
| `--vad-rms` | `pi_run.py` | energy gate; lowering it scans more (more CPU) |

Selection procedures (documented in `ACCURACY.md`), both on the
**validation** split, then frozen:

* **wake threshold** — the `(1 − 0.001)`-quantile of the negative scores,
  i.e. a false-accept rate of **0.1 %**. Tune per room with `--threshold` if
  your environment is noisier.
* **keyword thresholds** — greedy coordinate ascent over a fixed grid,
  objective = downstream *intent* accuracy (global 0.55 baseline; shipped
  array keeps 0.55 with `stop` 0.80 and `up`/`down` 0.20).

---

## 6. Raspberry Pi deployment

```bash
# 1. copy the project + model
scp -r output/ rapi@<pi>:~/rapi_vcm/output/
scp -r rapi_vcm/ runtime/ requirements_pi.txt rapi@<pi>:~/rapi_vcm/

# 2. dependencies (64-bit Raspberry Pi OS recommended)
sudo apt update
sudo apt install -y python3-pip libportaudio2 libsndfile1
pip3 install -r ~/rapi_vcm/requirements_pi.txt

# 3. run
cd ~/rapi_vcm
python3 runtime/pi_run.py --model output/onnx
```

Memory footprint: the three ONNX graphs plus runtime buffers stay well below
150 MB RSS — comfortable on a 4 GB Pi 4. Nothing else needs to be installed
(torch, librosa and numba are *not* required at inference time).
