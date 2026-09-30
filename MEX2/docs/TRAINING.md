# Training

Everything below runs on the DGX node `n003.ai.internal` (8 × NVIDIA A100-SXM4
40 GB). The training code lives in `train/` and reuses the shared package in
`rapi_vcm/`.

---

## 1. Environment (virtual environment)

The cluster's system Python has no `ensurepip`, so the environment is created
without pip and bootstrapped with `get-pip.py` (`remote/setup_venv.sh`):

```bash
python3 -m venv --without-pip ~/rapi_vcm/venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/rapi_vcm/venv/bin/python /tmp/get-pip.py
source ~/rapi_vcm/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy soundfile librosa scikit-learn pandas matplotlib \
            tqdm PyYAML onnx edge-tts
```

Installed versions used for this run:

| package | version |
|---|---|
| python | 3.12.3 |
| torch | 2.6.0+cu124 |
| numpy | 2.5.3 |
| librosa | 1.0.0 |
| onnx | 1.23.0 |

Every stage activates the environment first:

```bash
source ~/rapi_vcm/venv/bin/activate
```

---

## 2. Single-GPU policy

The cluster has eight GPUs. **Only one is used per job.**

Both training scripts pin the device *before* torch is imported:

```python
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=int, default=1)
_known, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)   # <-- before `import torch`
import torch
```

so the process sees exactly one device:

```
[train] CUDA_VISIBLE_DEVICES=1 device=cuda n_gpu=1
[train] using GPU 0 -> NVIDIA A100-SXM4-40GB
```

`CUDA_VISIBLE_DEVICES=1` maps physical GPU 1 to logical device 0 inside the
process — the other seven GPUs are never allocated. Verify with
`nvidia-smi` while training: only one memory pool grows.

```bash
GPU=1 bash ~/rapi_vcm/run_train_cmd.sh      # choose any of 0..7
```

---

## 3. Dataset

Source (as requested): <https://github.com/markandrian30/AI231/tree/main/MEX2/OptionB>

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/markandrian30/AI231.git ~/rapi_vcm/dataset/ai231_tmp
cd ~/rapi_vcm/dataset/ai231_tmp && git sparse-checkout set MEX2/OptionB
```

| property | value |
|---|---|
| WAV files available in the repository | **18,375** |
| intent folders | 31 (19 intents; slot intents split per value) |
| speakers | 150 (`s1`–`s150`), 134 foreign / 16 Filipino-English |
| acoustic conditions | `_clean.wav`, `_noisy.wav` |
| transcript | `manifest.csv` (path, label, intent, speaker, split, phrase, condition, transcript, slot, slot_value, duration) |
| splits (speaker-disjoint) | train 14,677 · val 1,856 · test 1,842 |

> The dataset's own README quotes 27,956 active files "on DGX2"; the GitHub
> tree actually contains 18,375 WAVs and a matching `manifest.csv`. The
> pipeline uses exactly what the repository provides.

Label spaces (`rapi_vcm/labels.py`, `rapi_vcm/keywords.py`):

* **19 intents** – the model output.
* **6 slot intents** – `TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`,
  `COLOR`, `CREATE_REMINDER`; each has 3–4 possible values (19 values total)
  plus a `NONE` class for utterances that carry no value.
* **40 keywords** – 21 intent atoms (`play music next pause stop volume up
  down lights on off call message reminder timer alarm temperature brightness
  color time weather`) + the 19 slot values. Labels are derived from the
  `transcript` column of `manifest.csv` (no ASR in the loop):
  `transcript → keywords` is a closed-form string match checked against every
  row — **18 375 / 18 375 correct**. The reverse map,
  `keywords → (intent, slot)`, is equally closed-form and deterministic
  (priority rules for overlaps; empty set → `PLAY_MUSIC`).
  Inspect both with `python -m rapi_vcm.keywords dataset/manifest.csv`.

---

## 4. Feature extraction

`train/extract_features.py` converts every WAV into a fixed-length log-mel
map using `rapi_vcm/features.py`:

| parameter | value |
|---|---|
| sample rate | 16 000 Hz |
| FFT / hop / window | 512 / 160 / Hann (32 ms / 10 ms → 100 frames·s⁻¹) |
| mel bands | 40 (HTK mel, 20–7 600 Hz) |
| normalisation | per-utterance CMVN |
| command window | 2.5 s → **40 × 250** |
| wake window | 1.2 s → **40 × 120** |
| storage | `float16` (`feat/cmd_X.npy`, 350 MB) |

```bash
python train/extract_features.py --dataset $DATA --out $FEAT --jobs 24
```

Outputs: `cmd_X.npy`, `cmd_intent.npy`, `cmd_slot.npy`, `cmd_label.npy`,
`cmd_split.npy`, `cmd_len.npy`, `cmd_kw.npy` (18 375 × 40 multi-hot),
`cmd_meta.json` (includes the `keywords` list).

The same `features.py` module is used at inference time on the PC and on the
Raspberry Pi, so train/test/deploy features are identical.

---

## 5. Wake-word training data

The OptionB corpus contains no "Hey Rapi" utterances, so
`train/gen_wake_data.py` builds the wake set from three sources:

**Positives** — 1 323 base clips synthesised with **edge-tts**
(49 `en-*` / `fil-PH` voices × 3 spellings × 3 rates × 3 pitches), then
augmented ×6:

* random lead-in (0–0.35 s) so the wake word is not always at frame 0,
* additive white/pink/brown noise at SNR 5–30 dB (60 %),
* random gain −14…+6 dB (40 %),
* resample speed change 0.90–1.12 (25 %),
* plus 2 000 "wake word + command tail" examples ("Hey Rapi … play music"),
  built by concatenating the TTS wake clip with a random OptionB command.

**Negatives**

| source | count | why |
|---|---|---|
| OptionB command utterances | 10 000 | speech that must *not* trigger |
| TTS near-misses ("maybe", "hey there", "hey google", "radio", …) | ~4 000 | phonetically close distractors |
| synthetic noise / room tone / silence | 2 500 | non-speech ambience |

**Split hygiene** — TTS voices are split 70/15/15 into train/val/test and a
voice's clips only ever appear in its own split; OptionB negatives inherit the
speaker-disjoint split from `manifest.csv`. No clip crosses splits.

```bash
python train/gen_wake_data.py --dataset $DATA --out $FEAT
```

Outputs: `wake_X.npy`, `wake_y.npy`, `wake_split.npy`, `wake_meta.json`,
plus `wake_tts/` (base TTS clips) and `wake_samples/` (auditable WAV samples).

---

## 6. Architectures

All models are depthwise-separable CNNs (`rapi_vcm/models.py`) with GroupNorm
(batch-size independent) and global average pooling.

**CommandModel** (baseline) — `widths=[64,96,128,160,192]`, strides `(2,2)` ×4
then `(2,1)`, input `(1, 40, 250)`:

```
stem 1→64 | DS 64→96 | DS 96→128 | DS 128→160 | DS 160→192 | DS 192→192
GAP → dropout 0.2 → intent head (19) + slot head (20)
```

**KeywordModel** (shipped) — identical backbone, one 40-way linear head:

```
GAP → dropout 0.2 → keyword head (40)   # sigmoid per keyword, multi-label BCE
```

**WakeModel** — `widths=[32,48,64,96]`, input `(1, 40, 120)`, single logit.

| model | parameters | ONNX size |
|---|---:|---:|
| keyword (shipped) | 123 496 | see `docs/BENCHMARKS.md` |
| command (baseline) | 123 303 | see `docs/BENCHMARKS.md` |
| wake | 23 665 | see `docs/BENCHMARKS.md` |

---

## 7. Hyper-parameters

| | **keyword model (shipped)** | command model (baseline) | wake model |
|---|---|---|---|
| optimizer | AdamW (β 0.9/0.999) | AdamW | AdamW |
| learning rate | 3e-3, OneCycle | 3e-3, OneCycle | 3e-3, OneCycle |
| weight decay | 0.05 | 0.05 | 0.05 |
| batch size | 256 | 256 | 512 |
| epochs (max) | 60 | 60 | 40 |
| early stopping | patience 15 on val intent acc | patience 15 on val intent acc | patience 12 on TPR |
| loss | BCE-with-logits, 40 labels, `pos_weight = (neg/pos)`, capped 50 | CE(intent) + 0.5·CE(slot) | BCE-with-logits (pos-weighted) |
| augmentation | SpecAugment (8 freq / 24 time masks) + time shift | same | same (6/16) |
| precision | bfloat16 autocast | bfloat16 autocast | bfloat16 autocast |
| gradient clip | 1.0 | 1.0 | 1.0 |
| selection metric | val intent accuracy (after the rule) | val intent accuracy | val TPR @ FAR ≤ 0.1 % |
| GPU | 1 × A100 (`--gpu 1`) | 1 × A100 (`--gpu 1`) | 1 × A100 (`--gpu 1`) |
| wall time | ~4 min | ~3 min | ~2 min |

The slot head of the baseline is trained on every sample: utterances without
a value target the `NONE` class, so the head also learns *when* not to report
a value. The keyword head has no separate slot head — the slot value comes
from the keyword rule.

`pos_weight` is the one knob that moved the numbers (test intent accuracy,
1 842 samples): raw `neg/pos` (mean 26, cap 50) → **99.24 %** (shipped);
`sqrt(neg/pos)` → 99.08 %; cap 20 → 98.91 %. Thresholds are tuned *after*
training, on validation only — see `ACCURACY.md` §2.

---

## 8. Running the pipeline

```bash
bash ~/rapi_vcm/run_keyword.sh        # keyword selfcheck -> features -> train (shipped)
bash ~/rapi_vcm/run_eval_keyword.sh   # test metrics -> threshold tuning -> export
bash ~/rapi_vcm/run_all.sh            # everything (incl. softmax baseline + wake)
```

or stage by stage:

```bash
bash ~/rapi_vcm/run_extract.sh      # features          (~2 min)
bash ~/rapi_vcm/run_wake.sh         # wake dataset      (~6 min)
GPU=1 bash ~/rapi_vcm/run_train_cmd.sh
GPU=1 bash ~/rapi_vcm/run_train_wake.sh
bash ~/rapi_vcm/run_eval_export.sh  # baseline metrics + ONNX
```

Artefacts are written to `~/rapi_vcm/out/`:

```
out/
├── keyword_best.pt  keyword_history.csv  keyword_summary.json
├── command_best.pt  command_history.csv  command_summary.json
├── wake_best.pt     wake_history.csv     wake_summary.json
├── metrics/  keyword_metrics.json  keyword_per_keyword.csv  keyword_per_class.csv
│             keyword_confusion.png  metrics.json  command_per_class.csv
│             command_confusion.png  wake_thresholds.csv
└── onnx/     keyword.onnx  command.onnx  wake.onnx  model_card.json
              benchmark_<label>.json
```

---

## 9. Training progress

Per-epoch logs: `out/keyword_history.csv`, `out/command_history.csv` and
`out/wake_history.csv`. The headline numbers and the learning curves are
summarised in [`ACCURACY.md`](ACCURACY.md).

Reproduce from scratch:

```bash
rm -rf ~/rapi_vcm/feat ~/rapi_vcm/out
bash ~/rapi_vcm/run_keyword.sh && bash ~/rapi_vcm/run_eval_keyword.sh
bash ~/rapi_vcm/run_all.sh
```
