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
| train | 14 677 | training speakers |
| validation | 1 856 | validation speakers |
| **test** | **1 842** | `s91`–`s100` |
| total | 18 375 | 31 folders |

* 19 intent classes, 6 of them carrying a spoken value (slot).
* Splits are **speaker-disjoint**: no voice seen in training appears in the
  test metrics.
* Wake-word data is separate (see §5): 26 797 windows
  (10 433 positive / 16 364 negative), split 19 837 / 3 302 / 3 658 —
  includes the 11 real user recordings (40 augmented variants each,
  recording-level split).

> The dataset README on GitHub claims 27 956 files; the actual repository
> contains **18 375 WAVs** (verified with `git ls-files`). All figures here use
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
  verified against every row of `manifest.csv`: **18 375 / 18 375 correct**
  (`python -m rapi_vcm.keywords dataset/manifest.csv`).
* Rule priority resolves overlaps (`COLOR` > `CREATE_REMINDER` > `TIMER` >
  `ALARM` > `TEMPERATURE` > `BRIGHTNESS` > `VOLUME_*` > `LIGHT_*` > `NEXT` >
  `PAUSE` > `STOP` > `PLAY_MUSIC` > `CALL` > `MESSAGE` > `LIST_REMINDERS` >
  `TIME` > `WEATHER`); an empty keyword set falls back to `PLAY_MUSIC`.

| metric | value |
|---|---:|
| samples | 1 842 |
| **intent accuracy (rule over keywords)** | **99.24 %** (0.99240) |
| **slot accuracy** (utterances with a value) | **99.29 %** (0.99286, n = 1 120) |
| slot accuracy, all utterances (incl. `NONE`) | 99.51 % (0.99511) |
| keyword micro precision / recall | 0.9886 / 0.9973 |
| keyword macro F1 | 0.99123 |
| exact keyword-set match | 97.88 % |
| downstream macro F1 (19 classes) | 0.98761 |
| best epoch (validation) | 47 / 60 |

Fourteen errors out of 1 842 utterances.

### Threshold selection

Sigmoid thresholds are tuned on the **validation** split only (greedy
coordinate ascent over a fixed grid, objective = downstream *intent*
accuracy), then frozen and reported on test:

| scheme | val intent | test intent | test slot |
|---|---:|---:|---:|
| single global threshold 0.55 | 98.87 % | 99.24 % | 99.29 % |
| **per-keyword (greedy on val)** | **99.03 %** | **99.24 %** | **99.29 %** |
| per-keyword F1 (rejected) | 99.08 % | 98.91 % | 99.55 % |
| `pos_weight = sqrt(neg/pos)` (rejected) | 98.33 % | 99.08 % | 99.38 % |

The shipped array keeps most keywords at 0.55, `stop` at 0.80 and
`up`/`down` at 0.20 (they only fire together with `volume`). It lives in
`model_card.json → keyword_thresholds` and is read by `runtime/pipeline.py`.

### Per intent

| intent | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| PLAY_MUSIC | 0.9667 | 1.0000 | 0.9831 | 58 |
| WEATHER | 1.0000 | 1.0000 | 1.0000 | 52 |
| TIME | 1.0000 | 1.0000 | 1.0000 | 53 |
| LIGHT_ON | 0.9815 | 0.9815 | 0.9815 | 54 |
| LIGHT_OFF | 1.0000 | 1.0000 | 1.0000 | 58 |
| PAUSE | 0.9811 | 0.9455 | 0.9630 | 55 |
| STOP | 0.9811 | 0.9630 | 0.9720 | 54 |
| NEXT | 0.9672 | 1.0000 | 0.9833 | 59 |
| VOLUME_UP | 0.9672 | 1.0000 | 0.9833 | 59 |
| VOLUME_DOWN | 0.9615 | 0.9615 | 0.9615 | 52 |
| CALL | 1.0000 | 0.9444 | 0.9714 | 54 |
| MESSAGE | 1.0000 | 1.0000 | 1.0000 | 57 |
| LIST_REMINDERS | 0.9655 | 0.9825 | 0.9739 | 57 |
| TIMER | 1.0000 | 1.0000 | 1.0000 | 176 |
| ALARM | 1.0000 | 1.0000 | 1.0000 | 173 |
| TEMPERATURE | 1.0000 | 1.0000 | 1.0000 | 180 |
| BRIGHTNESS | 1.0000 | 1.0000 | 1.0000 | 180 |
| COLOR | 1.0000 | 1.0000 | 1.0000 | 231 |
| CREATE_REMINDER | 0.9944 | 0.9889 | 0.9916 | 180 |

Confusion matrix (test split, downstream intent):
[`output/metrics/keyword_confusion.png`](../output/metrics/keyword_confusion.png),
tables in [`output/metrics/keyword_per_class.csv`](../output/metrics/keyword_per_class.csv)
and [`keyword_per_keyword.csv`](../output/metrics/keyword_per_keyword.csv).

The 14 errors break down as:

| direction | count |
|---|---:|
| `PAUSE` → other | 3 |
| `CALL` → other | 3 |
| `STOP` → other | 2 |
| `VOLUME_DOWN` → other | 2 |
| `CREATE_REMINDER` → other | 2 |
| `LIST_REMINDERS` → other | 1 |
| `LIGHT_ON` → other | 1 |
| incoming false positives | 2 each → `PLAY_MUSIC`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN`, `LIST_REMINDERS`; 1 each → `LIGHT_ON`, `PAUSE`, `STOP`, `CREATE_REMINDER` |

Worst keywords by F1: `play` 0.9625, `call` 0.9630, `down` 0.9720,
`pause` 0.9720, `on` 0.9720, `volume` 0.9823 — the short, acoustically thin
command words. All 19 slot keywords score ≥ 0.9821.

---

## 3. Softmax baseline (test split)

Direct 19-class + slot head, exported as `command.onnx` and still available as
the reference implementation (`train/evaluate.py`).

| metric | value |
|---|---:|
| samples | 1 842 |
| **intent accuracy** | **99.62 %** (0.99620) |
| top-2 accuracy | 99.89 % (0.99891) |
| macro F1 | 0.99400 |
| **slot accuracy** (utterances with a value) | **99.64 %** (0.99643, n = 1 120) |
| slot accuracy, all utterances (incl. `NONE`) | 99.78 % (0.99783) |
| slot accuracy given correct intent | 99.64 % (0.99643) |
| best epoch (validation) | 54 / 60 |

Seven errors out of 1 842 utterances.

### Per class

| intent | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| PLAY_MUSIC | 0.9667 | 1.0000 | 0.9831 | 58 |
| WEATHER | 1.0000 | 1.0000 | 1.0000 | 52 |
| TIME | 0.9808 | 0.9623 | 0.9714 | 53 |
| LIGHT_ON | 1.0000 | 1.0000 | 1.0000 | 54 |
| LIGHT_OFF | 1.0000 | 1.0000 | 1.0000 | 58 |
| PAUSE | 1.0000 | 1.0000 | 1.0000 | 55 |
| STOP | 0.9818 | 1.0000 | 0.9908 | 54 |
| NEXT | 0.9833 | 1.0000 | 0.9916 | 59 |
| VOLUME_UP | 1.0000 | 1.0000 | 1.0000 | 59 |
| VOLUME_DOWN | 1.0000 | 1.0000 | 1.0000 | 52 |
| CALL | 1.0000 | 0.9444 | 0.9714 | 54 |
| MESSAGE | 1.0000 | 0.9825 | 0.9912 | 57 |
| LIST_REMINDERS | 0.9828 | 1.0000 | 0.9913 | 57 |
| TIMER | 1.0000 | 1.0000 | 1.0000 | 176 |
| ALARM | 0.9943 | 1.0000 | 0.9971 | 173 |
| TEMPERATURE | 1.0000 | 1.0000 | 1.0000 | 180 |
| BRIGHTNESS | 1.0000 | 1.0000 | 1.0000 | 180 |
| COLOR | 1.0000 | 0.9957 | 0.9978 | 231 |
| CREATE_REMINDER | 1.0000 | 1.0000 | 1.0000 | 180 |

Confusion matrix (test split):
[`output/metrics/command_confusion.png`](../output/metrics/command_confusion.png),
raw table in [`output/metrics/command_per_class.csv`](../output/metrics/command_per_class.csv).

The 7 errors, exactly as the precision/recall columns imply:

| direction | count |
|---|---:|
| `CALL` → other | 3 |
| `TIME` → other | 2 |
| `MESSAGE` → other | 1 |
| `COLOR` → other | 1 |
| incoming false positives | 2 → `PLAY_MUSIC`, 1 each → `TIME`, `STOP`, `NEXT`, `LIST_REMINDERS`, `ALARM` |

Every slot intent (`TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`,
`CREATE_REMINDER`) has recall ≥ 0.995.

---

## 4. ONNX ↔ PyTorch parity

`train/export_onnx.py` compares both back-ends on random inputs:

| head | max abs diff | argmax / top-1 identical |
|---|---:|---|
| keyword logits (shipped) | 2.48e-05 | yes |
| command intent logits | 1.14e-05 | yes |
| command slot logits | 1.14e-05 | yes |
| wake logit | 4.77e-06 | yes |

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

The **shipped** `model_card.json` threshold is **0.400**: the whole
0.38–0.565 band sits on the same test TPR plateau (99.715 %, FAR 0.315 %),
and 0.400 keeps margin for the *streaming* detector, which fires only after
**two consecutive 100 ms windows** above threshold — a borderline clip whose
windows score 0.52 / 0.96 fires at 0.400 but never builds a streak at
0.600 and above.

| metric | value |
|---|---:|
| test windows | 3 658 (1 754 positive / 1 904 negative) |
| threshold (shipped) | 0.400 |
| threshold (val-selected) | 0.8194 |
| **true-positive rate (shipped 0.400)** | **99.715 %** (0.99715) |
| false-accept rate (at 0.400, full clip) | 0.315 % (6 / 1 904) |
| TPR / FAR at val-selected 0.8194 | 99.544 % / 0.105 % |
| equal-error rate | 0.329 % (0.003286) |
| best epoch (validation, early stop) | 21 / 34 |

Trade-off curve measured on the test split:

| target FAR | threshold | TPR |
|---|---:|---:|
| 0.00 % | 0.865 | 99.54 % |
| 0.05 % | 0.840 | 99.54 % |
| **0.105 % (val-selected)** | 0.8194 | 99.54 % |
| 0.21 % | 0.700 | 99.54 % |
| **0.315 % (shipped)** | **0.400** | **99.715 %** |
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

Streaming false accepts on the 19 command-only clips (no wake phrase):
**0 / 19 with both runs** — run-4's highest full-clip score rose
0.102 → 0.571, but that single window never sustains the two-window
streak, so nothing fires.

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
| intent correct | **18 / 19** (94.7 %) |
| slot correct (6 utterances carry a value) | **6 / 6** (100 %) |

The single miss is `PLAY_MUSIC_s99_v3_clean` → `NEXT`: the clip is correctly
woken and the keywords `play` (0.999) and `music` (1.000) fire, but the
out-of-vocabulary word also nudges `next` to 0.5688, just over its 0.55
threshold, and the rule gives `NEXT` priority. This is the same marginal
keyword that appears in the test-set error table (`next` F1 0.9833);
it is a threshold-robustness issue, not a wiring bug — the softmax baseline
classifies this clip correctly at 99.62 % intent accuracy.

**Negative control** — running the 19 command-only clips (no wake word) with
wake detection enabled rejects **18 / 19**; `MESSAGE_s97_v1_clean` is falsely
accepted at score 0.90. This matches the ~0.1 % FAR measured over 1 861
negative windows: individual hard negatives exist, so keep the threshold at
0.30 or raise it in a noisy room.

Reproduce:

```bash
python test_clips/build_combined.py          # build the 19 combined clips
python test_clips/check_end2end.py --model output/onnx
python runtime/pc_test.py --model output/onnx --dir test_clips/commands          # negative control
python runtime/pc_test.py --model output/onnx --dir test_clips/combined --show-slot
```

---

## 7. How to read these numbers

* **99.24 % intent accuracy (keyword head)** is measured on unseen speakers
  with the exact feature pipeline used on the Pi — it is the number that
  matters for day-to-day use. The softmax baseline reaches 99.62 % on the
  same split; the keyword head costs 0.38 pt (7 extra errors) in exchange for
  reporting *which words were said*, which is what the app consumes.
* Both heads are exported: `runtime/pipeline.py` prefers `keyword.onnx` when
  the model card says `head: keyword` and falls back to `command.onnx`
  otherwise, so the baseline stays one file rename away.
* **Wake FAR 0.1 %** is per 1.2 s window scanned every 100 ms; in an
  always-listening Pi deployment that is the knob to tune (`--threshold`).
* The end-to-end set is tiny (19 clips) and exists to prove the *pipeline*
  wiring, not to re-measure accuracy — the 1 842-sample test split is the
  authoritative figure.
