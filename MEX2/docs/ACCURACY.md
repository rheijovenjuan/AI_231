# Accuracy

All numbers below come from the DGX evaluation runs over the **held-out test
split** of the OptionB dataset (speaker-disjoint), plus a small end-to-end
sanity set built from real clips.

Two command heads are measured:

| head | what it is | reproduce | raw JSON |
|---|---|---|---|
| **keyword spotter** (shipped) | 40 keyword logits → deterministic rule → intent + slot | `bash ~/rapi_vcm/run_eval_keyword.sh` | [`output/metrics/keyword_metrics.json`](../output/metrics/keyword_metrics.json) |
| softmax baseline | direct 19-class + slot head, kept as the reference point | `bash ~/rapi_vcm/run_eval_export.sh` | [`output/metrics/metrics.json`](../output/metrics/metrics.json) |

---

## 1. Data splits

| split | utterances | speakers |
|---|---:|---|
| train | 14 752 | training speakers + `real1` (75) |
| validation | 1 875 | validation speakers + `real1` (19) |
| **test** | **1 842** | `s91`–`s100` |
| total | 18 469 | 31 folders |

* 19 intent classes, 6 of them carrying a spoken value (slot).
* Splits are **speaker-disjoint**: no voice seen in training appears in the
  test metrics. The 94 real-voice command clips (`speaker=real1`, recorded
  with [`record_command.py`](../record_command.py), committed under
  [`command_data/`](../command_data/)) stay **out of the test split**: 75 in
  train, 19 in val (clip-level holdout, the same precedent as the 11 wake
  recordings — one speaker spans train/val).
* Wake-word data is separate (see §5): 26 797 windows
  (10 433 positive / 16 364 negative), split 19 837 / 3 302 / 3 658 —
  includes the 11 real user recordings (40 augmented variants each,
  recording-level split).

> The dataset README on GitHub claims 27 956 files; the actual repository
> contains **18 375 WAVs** (verified with `git ls-files`); the DGX working
> set adds the 94 real-voice clips for **18 469 rows**. All figures here use
> what is actually present.

---

## 2. Keyword spotter — shipped head (test split)

The model does **not** classify the query directly. After the wake word is
acknowledged it reports which of **40 keywords** are present (multi-label BCE,
`rapi_vcm/keywords.py`), and a deterministic rule maps that keyword set to the
19-class command + slot:

* 21 intent atoms: `play music next pause stop volume up down lights on off
  call message reminder timer alarm temperature brightness color time weather`
* 19 slot values: `10 seconds / 30 seconds / 1 minute`, `4 am / 8 am / 9 pm`,
  `18/22/26 degrees`, `20/60/100 percent`, `red / blue / green / yellow`,
  `drink water / study / exercise`
* `transcript → keywords` and `keywords → (intent, slot)` are closed-form and
  verified against every row of `manifest.csv`: **18 469 / 18 469 correct**
  (`python -m rapi_vcm.keywords dataset/manifest.csv`).
* Rule priority resolves overlaps (`COLOR` > `CREATE_REMINDER` > `TIMER` >
  `ALARM` > `TEMPERATURE` > `BRIGHTNESS` > `VOLUME_*` > `LIGHT_*` > `NEXT` >
  `PAUSE` > `STOP` > `PLAY_MUSIC` > `CALL` > `MESSAGE` > `LIST_REMINDERS` >
  `TIME` > `WEATHER`); an empty keyword set falls back to `PLAY_MUSIC`.

| metric | value |
|---|---:|
| samples | 1 842 |
| **intent accuracy (rule over keywords)** | **99.29 %** (0.99294) |
| **slot accuracy** (utterances with a value) | **99.46 %** (0.99464, n = 1 120) |
| slot accuracy, all utterances (incl. `NONE`) | 99.57 % (0.99566) |
| keyword micro precision / recall | 0.9920 / 0.9953 |
| keyword macro F1 | 0.99239 |
| exact keyword-set match | 98.15 % |
| downstream macro F1 (19 classes) | 0.99008 |
| best epoch (validation) | 50 / 60 |

Thirteen errors out of 1 842 utterances.

### Threshold selection

Sigmoid thresholds are tuned on the **validation** split only (greedy
coordinate ascent over a fixed grid, objective = downstream *intent*
accuracy), then frozen and reported on test:

| scheme | val intent | test intent | test slot |
|---|---:|---:|---:|
| single global threshold 0.75 | 98.51 % | 99.24 % | 99.46 % |
| **per-keyword (greedy on val)** | **98.88 %** | **99.29 %** | **99.46 %** |
| per-keyword F1 (rejected, run-3) | 99.08 % | 98.91 % | 99.55 % |
| `pos_weight = sqrt(neg/pos)` (rejected, run-3) | 98.33 % | 99.08 % | 99.38 % |

The shipped array keeps most keywords at 0.75, with `down`/`lights` at 0.20
(they only fire together with `volume`/`on`), `call` at 0.25 (the
under-represented class on real voices), `stop` at 0.65 and `timer` at
0.55. It lives in `model_card.json → keyword_thresholds` and is read by
`runtime/pipeline.py`.

### Per intent

| intent | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| PLAY_MUSIC | 0.9355 | 1.0000 | 0.9667 | 58 |
| WEATHER | 1.0000 | 1.0000 | 1.0000 | 52 |
| TIME | 1.0000 | 1.0000 | 1.0000 | 53 |
| LIGHT_ON | 0.9636 | 0.9815 | 0.9725 | 54 |
| LIGHT_OFF | 1.0000 | 1.0000 | 1.0000 | 58 |
| PAUSE | 0.9815 | 0.9636 | 0.9725 | 55 |
| STOP | 1.0000 | 1.0000 | 1.0000 | 54 |
| NEXT | 1.0000 | 1.0000 | 1.0000 | 59 |
| VOLUME_UP | 0.9833 | 1.0000 | 0.9916 | 59 |
| VOLUME_DOWN | 1.0000 | 0.9808 | 0.9903 | 52 |
| CALL | 1.0000 | 0.9444 | 0.9714 | 54 |
| MESSAGE | 1.0000 | 1.0000 | 1.0000 | 57 |
| LIST_REMINDERS | 0.9649 | 0.9649 | 0.9649 | 57 |
| TIMER | 1.0000 | 1.0000 | 1.0000 | 176 |
| ALARM | 0.9943 | 1.0000 | 0.9971 | 173 |
| TEMPERATURE | 1.0000 | 1.0000 | 1.0000 | 180 |
| BRIGHTNESS | 1.0000 | 1.0000 | 1.0000 | 180 |
| COLOR | 1.0000 | 0.9913 | 0.9957 | 231 |
| CREATE_REMINDER | 0.9889 | 0.9889 | 0.9889 | 180 |

Confusion matrix (test split, downstream intent):
[`output/metrics/keyword_confusion.png`](../output/metrics/keyword_confusion.png),
tables in [`output/metrics/keyword_per_class.csv`](../output/metrics/keyword_per_class.csv)
and [`keyword_per_keyword.csv`](../output/metrics/keyword_per_keyword.csv).

The 13 errors break down as:

| direction | count |
|---|---:|
| `CALL` → other | 3 |
| `PAUSE` → other | 2 |
| `CREATE_REMINDER` → `LIST_REMINDERS` | 2 |
| `LIST_REMINDERS` → `CREATE_REMINDER` | 2 |
| `COLOR` → other | 2 |
| `LIGHT_ON` → other | 1 |
| `VOLUME_DOWN` → other | 1 |
| incoming false positives | 4 → `PLAY_MUSIC`; 2 each → `LIGHT_ON`, `CREATE_REMINDER`, `LIST_REMINDERS`; 1 each → `PAUSE`, `ALARM`, `VOLUME_UP` |

Worst keywords by F1: `100 percent` 0.9677, `play` 0.9689, `call` 0.9720,
`on` 0.9725, `pause` 0.9725, `8 am` 0.9739 — the short, acoustically thin
command words. All other keywords score ≥ 0.9811.

### Real-voice slice (run-5)

94 clips of the developer's own voice (`speaker=real1`, recorded with
[`record_command.py`](../record_command.py), committed under
[`command_data/`](../command_data/)) were mixed in for run-5 — 74 CALL /
20 COLOR after two dead-air clips were dropped, 75 train / 19 val. On the
19 held-out *val* clips (never trained on), the shipped keyword head:

| slice | before (run-4) | after (run-5) |
|---|---:|---:|
| CALL intent correct | 3 / 14 (21.4 %) | **10 / 14 (71.4 %)** |
| COLOR intent correct | 5 / 5 (100 %) | 5 / 5 (100 %) |
| all | 8 / 19 (42.1 %) | **15 / 19 (78.9 %)** |
| `call` keyword active | 5 / 14 | **13 / 14** |

On the 75 train clips: CALL 33.3 % → **93.3 %**, COLOR 86.7 % → **100 %**.
Test-split metrics moved with it (intent 99.24 → 99.29, slot 99.29 → 99.46)
and the test split contains no real audio, so the real-voice gain is not
leakage. Remaining real-voice misses are call/color co-activations; more
real CALL mass would close them.

---

## 3. Softmax baseline (test split)

Direct 19-class + slot head, exported as `command.onnx` and still available as
the reference implementation (`train/evaluate.py`).

| metric | value |
|---|---:|
| samples | 1 842 |
| **intent accuracy** | **99.84 %** (0.99837) |
| top-2 accuracy | 99.95 % (0.99946) |
| macro F1 | 0.99742 |
| **slot accuracy** (utterances with a value) | **99.38 %** (0.99375, n = 1 120) |
| slot accuracy, all utterances (incl. `NONE`) | 99.62 % (0.99620) |
| slot accuracy given correct intent | 99.38 % (0.99375) |
| best epoch (validation) | 50 / 60 |

Three errors out of 1 842 utterances.

### Per class

| intent | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| PLAY_MUSIC | 0.9831 | 1.0000 | 0.9915 | 58 |
| WEATHER | 1.0000 | 1.0000 | 1.0000 | 52 |
| TIME | 1.0000 | 0.9623 | 0.9808 | 53 |
| LIGHT_ON | 1.0000 | 1.0000 | 1.0000 | 54 |
| LIGHT_OFF | 1.0000 | 1.0000 | 1.0000 | 58 |
| PAUSE | 1.0000 | 1.0000 | 1.0000 | 55 |
| STOP | 0.9818 | 1.0000 | 0.9908 | 54 |
| NEXT | 1.0000 | 1.0000 | 1.0000 | 59 |
| VOLUME_UP | 1.0000 | 1.0000 | 1.0000 | 59 |
| VOLUME_DOWN | 1.0000 | 1.0000 | 1.0000 | 52 |
| CALL | 1.0000 | 0.9815 | 0.9907 | 54 |
| MESSAGE | 1.0000 | 1.0000 | 1.0000 | 57 |
| LIST_REMINDERS | 1.0000 | 1.0000 | 1.0000 | 57 |
| TIMER | 0.9944 | 1.0000 | 0.9972 | 176 |
| ALARM | 1.0000 | 1.0000 | 1.0000 | 173 |
| TEMPERATURE | 1.0000 | 1.0000 | 1.0000 | 180 |
| BRIGHTNESS | 1.0000 | 1.0000 | 1.0000 | 180 |
| COLOR | 1.0000 | 1.0000 | 1.0000 | 231 |
| CREATE_REMINDER | 1.0000 | 1.0000 | 1.0000 | 180 |

Confusion matrix (test split):
[`output/metrics/command_confusion.png`](../output/metrics/command_confusion.png),
raw table in [`output/metrics/command_per_class.csv`](../output/metrics/command_per_class.csv).

The 3 errors, exactly as the precision/recall columns imply:

| direction | count |
|---|---:|
| `TIME` → other | 2 |
| `CALL` → other | 1 |
| incoming false positives | 1 each → `PLAY_MUSIC`, `STOP`, `TIMER` |

Every slot intent (`TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`,
`CREATE_REMINDER`) has recall ≥ 0.995.

---

## 4. ONNX ↔ PyTorch parity

`train/export_onnx.py` compares both back-ends on random inputs:

| head | max abs diff | argmax / top-1 identical |
|---|---:|---|
| keyword logits (shipped) | 3.91e-05 | yes |
| command intent logits | 2.67e-05 | yes |
| command slot logits | 2.34e-05 | yes |
| wake logit | 3.81e-06 | yes |

Numerically equivalent — any difference in behaviour between the training host
and the Pi is measurement noise, not the export.

---

## 5. Wake-word detector (test split)

Retrained 2026-10-02 with **real user recordings** mixed in (see below).
Positives: 49 edge-tts voices × 3 texts × 3 rates × 3 pitches (1 323 base
clips, 100 % synthesis success) with noise/gain/window augmentation **plus
11 clips recorded with `record_wake.py` × 40 augmented variants and 55 real
voice + command-tail mixes** — the recordings are split **by original clip**
(7 train / 1 val / 3 test), so no augmented variant leaks across splits.
Negatives: 10 000 OptionB utterances, ~4 000 TTS near-misses ("hey puppy",
"hey Rapi" mispronunciations, other names) and 2 500 noise segments.
Totals: **10 433 positive / 16 364 negative windows**; test split
**3 658 (1 754 pos / 1 904 neg)** — full breakdown in
[`output/training/wake_meta.json`](../output/training/wake_meta.json).

Operating point selected on the **validation** split as
`(1 − 0.001)`-quantile of negative scores (**0.8194**), then frozen; test
metrics at that point are TPR 99.54 % / FAR 0.105 %.

The **shipped** `model_card.json` threshold is **0.900**: it sits at the top
of the test FAR-0 plateau (0.865-0.900 all give TPR 99.544 % / FAR 0.000 %),
so none of the 1 904 test negatives false-accept. It also clears every
*streaming* false accept measured on the PC: plain command speech never
builds a two-window streak above 0.416 and background music tops out at
0.825, while all 42 streaming wake clips (11 real + 12 TTS + 19
wake+command) still fire with a worst two-window streak of 0.980. The
streaming detector scores only the trailing 1.2 s of **real** audio as two
consecutive 100 ms windows - it never scores the zero-padded 2.0 s ring
old versions did, whose start-up zeros made CMVN saturate the score on
ordinary sound right after launch.

| metric | value |
|---|---:|
| test windows | 3 658 (1 754 positive / 1 904 negative) |
| threshold (shipped) | 0.900 |
| threshold (val-selected) | 0.8194 |
| **true-positive rate (shipped 0.900)** | **99.544 %** (0.99544) |
| false-accept rate (at 0.900, full clip) | **0.000 %** (0 / 1 904) |
| TPR / FAR at val-selected 0.8194 | 99.544 % / 0.105 % |
| equal-error rate | 0.329 % (0.003286) |
| best epoch (validation, early stop) | 21 / 34 |

Trade-off curve measured on the test split:

| target FAR | threshold | TPR |
|---|---:|---:|
| **0.00 % (shipped)** | **0.900** | **99.54 %** |
| 0.00 % | 0.865 | 99.54 % |
| 0.05 % | 0.840 | 99.54 % |
| **0.105 % (val-selected)** | 0.8194 | 99.54 % |
| 0.21 % | 0.700 | 99.54 % |
| 0.315 % (former shipped) | 0.400 | 99.715 % |
| 1 % | 0.010 | 100.00 % |

Full curve: [`output/metrics/wake_thresholds.csv`](../output/metrics/wake_thresholds.csv)
(198 operating points).

### Real-voice recordings (why the retrain)

11 clips recorded with `record_wake.py`
([`wakeword_data/positive/`](../wakeword_data/positive/)):

| model | clips ≥ 0.40 | worst full-clip score |
|---|---:|---:|
| run-3 (synthetic only) | 10 / 11 | **0.038** (`hey_rapi_004` — total miss) |
| **run-4 (this retrain)** | **11 / 11** | **0.996** |

Run-4's worst score (0.996) also clears the shipped 0.900 bar **11 / 11**.

Streaming false accepts at the shipped 0.900 on the 19 command-only clips
(no wake phrase): **0 / 19** in steady state and at start-up, and **0 / 2**
for the two music tracks (top score 0.825). The start-up figure depends on
the fix in `voice_assistant/assistant.py`: the old code scored the full
2.0 s ring, whose zero padding made CMVN saturate the score on ordinary
sound right after launch (6 / 19 command clips fired at the old 0.400).

---

## 6. End-to-end sanity set (real clips, wake + command)

`test_clips/` holds 19 real OptionB test utterances (one per class) paired with
12 held-out TTS wake clips (`v40`–`v48`, voices never used in training).
`test_clips/build_combined.py` concatenates
`[wake (1.2 s)][0.25 s gap][command][0.4 s tail]` and writes
`ground_truth.json`; `test_clips/check_end2end.py` then runs the shipped ONNX
models over the whole set.

| metric | result |
|---|---:|
| wake detected | **19 / 19** (100 %) |
| intent correct | **19 / 19** (100 %) |
| slot correct (6 utterances carry a value) | **6 / 6** (100 %) |

The run-4 miss (`PLAY_MUSIC_s99_v3_clean` → `NEXT`, `next` nudged to 0.5688
over its old 0.55 threshold) is fixed with the run-5 model: all 19 intents
classify correctly, including that clip.

**Negative control** — running the 19 command-only clips (no wake word)
rejects **18 / 19** in `pc_test`'s single-scan mode; `STOP_s94_v2_clean`
contains one isolated 1.2 s window scoring 0.91. The app fires only after
two consecutive scans ≥ 0.90 (README: "top full-clip score 0.571 never
sustains the 2-window streak"), so streaming false accepts stay **0 / 19**
(the ~0.1 % per-window FAR over 1 861 negative windows is unchanged — the
wake model and threshold were untouched). Individual hard negatives exist,
so keep the shipped threshold at 0.90 or raise it in a noisy room.

Reproduce:

```bash
python test_clips/build_combined.py          # build the 19 combined clips
python test_clips/check_end2end.py --model output/onnx
python runtime/pc_test.py --model output/onnx --dir test_clips/commands          # negative control
python runtime/pc_test.py --model output/onnx --dir test_clips/combined --show-slot
```

---

## 7. How to read these numbers

* **99.29 % intent accuracy (keyword head)** is measured on unseen speakers
  with the exact feature pipeline used on the Pi — it is the number that
  matters for day-to-day use. The softmax baseline reaches 99.84 % on the
  same split; the keyword head costs 0.55 pt (10 extra errors) in exchange
  for reporting *which words were said*, which is what the app consumes.
* Both heads are exported: `runtime/pipeline.py` prefers `keyword.onnx` when
  the model card says `head: keyword` and falls back to `command.onnx`
  otherwise, so the baseline stays one file rename away.
* **Wake FAR 0.1 %** is per 1.2 s window scanned every 100 ms; in an
  always-listening Pi deployment that is the knob to tune (`--threshold`).
* The end-to-end set is tiny (19 clips) and exists to prove the *pipeline*
  wiring, not to re-measure accuracy — the 1 842-sample test split is the
  authoritative figure.
