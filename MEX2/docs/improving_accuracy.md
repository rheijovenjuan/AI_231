# Improving accuracy

Status: the big-ticket items from the original version of this document are
**done and measured**. The first table is where things stand; the last
section lists what is still open.

## Where it stands (measured 2026-09)

| check | result |
|---|---|
| held-out test split (1842 clips, `data/manifest.csv`) | intent **99.29 %** (clean 99.35 / noisy 99.24) |
| acoustic keyword head alone (1510 eval clips) | **99.54 %** |
| hybrid transcript + intent (same 1510) | **99.93 %** (6 rescued by the transcript, 0 regressions) |
| text-slot rules on ground-truth transcripts (18375) | **100 %** correct intent |
| state-machine sim (19 wake+command clips, transcribe order) | **19/19**, 0 ordering violations |
| wake detector, test split, shipped threshold 0.40 | TPR **99.75 %**, FAR **0.000 %** |
| wake false accepts on 19 command-only clips | **0/19** (top score 0.102) |

## What was done

### 1. More data (was the biggest win)
The original ≤300 clips/intent (~88 %) became ~900/intent, clean + noisy,
speaker-disjoint test split (18375 clips):

```bash
python train.py --download --max-per-intent 900
```

→ held-out intent **99.29 %**.

### 2. Better features / stronger classifier (replaced, not tuned)
The GMM / logistic-regression options were superseded by small CNN heads
trained on DGX and exported to ONNX (`command.onnx`, `keyword.onnx`,
`wake.onnx`): MFCC + delta input, hard-negative wake training (run-3: 10k
command negatives + TTS near-misses "hey rappy" etc.), export parity
≤ 1.5e-5 against PyTorch (see `docs/ACCURACY.md`).

### 3. Disambiguate the confusable pairs (hybrid policy)
Instead of a two-stage domain classifier, intent is now decided by
`voice_assistant.assistant._decide()` from **two evidence sources**:

- the acoustic keyword head (40-keyword ONNX spotter + confidence gate), and
- `text_intent()` - regex rules over the Whisper transcript.

Strong rules (dim/brightness, color, lights, volume, STOP/PAUSE, ...)
override a gate-passed acoustic result; weak rules (CALL, MESSAGE, NEXT,
PLAY_MUSIC, alarms/timers/info) only rescue a refused gate, corroborate via
the acoustic keyword head (`_TEXT_KW_HINT`), or open the utterance with
their cue word ("message Anna" - but not "... next episode", which lands in
a later clause). CALL is exempt from the opening-cue rule: ASR often turns
"Color red" into "call ..." and the acoustic colour evidence must win.
This fixed transcript-correct/intent-wrong cases without letting narrative
transcripts steal commands.

### 4. Slots via ASR (Whisper instead of Vosk)
`--transcribe` runs faster-whisper `base.en` (int8 CPU, ~1.1 s/clip) on the
captured command; the transcript is printed/logged **before** the intent and
feeds `slots_from_text()` (numbers, colors incl. free-form "purple",
brightness percent). Keyword slots and text slots are merged under
`--slot-policy keyword|text|hybrid`. Color parsing accepts ~28 palette
words + "lights in X" phrasings (`classifier.color_from_text`).

### 5. Evaluation hygiene (done)
- Held-out **test** split only; clean vs noisy reported separately.
- Wake reported as FAR/FRR curves, not accuracy: `output/metrics/wake_thresholds.csv`
  (199 operating points) + cross-corpus report `reports/wake_cross_corpus.json`.
- Regression suites: `reports/hybrid_eval.json` (1510), `reports/slot_policy_eval.json`
  (1120), transcript-order sim (19), OOV/TTS probes (dim/step-down/color).

## Still open

- **Per-frame attention / mean-max pooling and spectral-contrast features** -
  the CNN mostly fixed this, but short confusables (TIME/CALL/NEXT) could
  still benefit.
- **Pi 4 benchmark** for the full pipeline (whisper base.en is the bottleneck;
  commands in `docs/BENCHMARKS.md`). If it is too slow, fall back to a tiny
  streaming ASR (vosk ~40 MB) for slots only.
- **Real "Hey Rapi" negatives** from live mic recordings - see
  `collecting_wakeword_data.md`; the current FAR numbers are TTS/split based.
- **Clean up Whisper mishears as they appear** - add the phrase as a rule or
  alias in `text_intent()`/`slots_from_text()` and re-run the hybrid eval.
