# "Hey Rapi" Wake Word — Synthetic Data + Augmentation

The spoken-command dataset does **not** contain the wake word "Hey Rapi", so the
wake-word detector is trained on a corpus that is manufactured synthetically.
By default the speech is **neural TTS (edge-tts)** — real voices that genuinely
say *"Hey Rapi"* — with an offline formant/PSOLA fallback for machines without
network access. This is the standard bootstrap for a custom wake word (same
idea behind Picovoice Porcupine / Mozilla wake-word projects).

## How the data is made

`make_wake_data.py` runs four stages, all deterministic (seeded) and CPU-only:

1. **Synthesis** — the speech source is chosen with `--tts`:
   - **`edge` (default via `auto`)** — `edge-tts` renders each phrase in 10
     neural voices × 3 rate variants × 3 pitch variants. The audio **actually
     sounds like the phrase**. First run needs network; everything is cached
     in `data/wake_tts/` (see `plan.json` for which file says what).
   - **`offline`** — a small formant / PSOLA-style synthesiser renders spoken
     phrases from a phoneme sequence (vowels, sonorants, stop bursts) with a
     per-voice F0 contour, formant scale and breathiness. This needs no
     network and runs on a Pi, **but the output is a rough buzz-like
     approximation — it does NOT sound like "Hey Rapi"**. It is only a
     pipeline bootstrap.
   - **Positives:** the phrase *"Hey Rapi"* (also `"Hey Rapi!"`, `"Hey, Rapi"`).
   - **Negatives:** look-alike / near-collision phrases (*"Hey Ray"*,
     *"Hey Rapid"*, *"Hey Rapi please"*, *"Hey Siri"*, *"Okay Google"*, …)
     plus clearly-unrelated everyday speech (*"What time is it"*,
     *"Play some music"*, …) and **non-speech** audio (silence + room tone).
2. **Augmentation** — every clip is distorted to simulate real conditions:
   pitch shift, speed change, gain, simple reverb, white-noise (SNR 12–30 dB)
   and low-frequency rumble. This multiplies the effective dataset size.
3. **Features** — each (augmented) clip is turned into the same 82-dim MFCC
   + delta + scalar vector used everywhere else (`voice_assistant.features`).
4. **Model** — two GMMs (positive vs negative) are fit with a **from-scratch
   NumPy GMM** (`voice_assistant/gmm.py`, diagonal covariances, EM). The
   decision is a log-likelihood ratio: `score = LLR(P(x|pos) − P(x|neg))`;
   fire when `score >= threshold`.

> The GMM is implemented from scratch (not `sklearn`) so the project has no
> heavy ML dependency and the model is a tiny, fast, Pi-friendly artifact.

## Training & evaluation

```bash
python make_wake_data.py            # defaults (edge-tts speech, cached)
python make_wake_data.py --tts edge # force neural TTS
python make_wake_data.py --tts offline  # force the formant fallback
python make_wake_data.py --pos-per-voice 30 --neg-per-voice 30 --voices 6 --aug 4
python make_wake_data.py --dry-run  # print the plan, write nothing
```

Evaluation is on a **held-out split by clip** (a test clip's augmentations are
never seen in training), so the numbers are honest.

Latest run (defaults: 6 voices, 240 TTS positives + 420 negatives, ×6
augmentation → 3,960 vectors, 16 components; 660 clean clips):

| metric (held-out test) | value |
|------------------------|-------|
| Accuracy               | 71.1% |
| Recall (catch "Hey Rapi") | 82.5% |
| Precision              | 57.1% |
| F1                     | 67.5% |
| ROC AUC                | **81.6%** |
| Decision threshold (LLR) | ≥ −1.08 |

Confusion (test): TP=297 FP=223 FN=63 TN=407. Negatives split by type:
silence/room-tone never fires, everyday speech mostly rejected, and the
*phonetically-near* collisions (*"Hey Ray"*, *"Hey Rapid"*) remain the hard
case.

> **Why these numbers are lower than the old formant-synth report (~92%)?**
> The old corpus was synthetic buzz in, synthetic buzz out - the GMM was
> separating *synthesiser artifacts*, not words. With real neural speech,
> distinguishing `"Hey Ray"` from `"Hey Rapi"` from `"What time is it"` is a
> genuine speech-recognition problem that a bag-of-MFCC GMM cannot fully
> solve. The numbers above are the honest capability of this legacy path.

## Which wake model does the app actually use?

The **combined app (`rapi_combined`) does not use this GMM at all** - it runs
`wake.onnx`, a small CNN trained in the sibling voice-command-model project
(run-3: 8k edge-tts positives + 10k command negatives + hard negatives + noise;
test split TPR 99.75 % at the shipped 0.40 threshold, FAR 0.000 %, see its
`docs/ACCURACY.md`). This script stays as the dependency-light legacy/demo
path.

## Files produced

- `models/wakeword_synthetic.joblib` — the wake-word model (pos + neg GMMs).
- `reports/wake_synthetic_report.txt` / `.json` — corpus, metrics, threshold.
- `data/wake_tts/` — the raw neural-TTS clips **that actually say the
  phrases** (`plan.json` maps each file to its text/voice). **Listen here to
  verify the audio.**
- `data/synth_wake/` — the trimmed WAVs used for training (positives,
  negatives, silence).

## Wiring into the app

The combined app no longer resolves this synthetic model at runtime - the
wiring now goes through the ONNX path:

- `app.py` → `runtime.pipeline.VoicePipeline` → `wake.onnx`, threshold read
  from `output/onnx/model_card.json` (`wake_threshold`, shipped 0.40;
  override with `--wake-trigger`).
- This script's calibrated GMM threshold lives on only as an offline artifact:
  `threshold_llr` in `reports/wake_synthetic_report.json`.

The runtime detector (`voice_assistant/wakeword.py`) scores the **whole**
candidate utterance (leading/trailing silence trimmed, capped at ~1.5 s) — this
matches how the model was trained and is what makes the runtime agree with the
offline metrics.

## Honest limitations & next step

Neural-TTS positives say the phrase correctly, but TTS still differs from how
*you* say "Hey Rapi" — the numbers validate the full pipeline and give a
working baseline, they are not a guarantee on real human speech, and the
precision above shows this GMM path is the weak link. For production use the
ONNX `wake.onnx` path (see the previous section); for this GMM path, record
~200 real "Hey Rapi" clips (see `docs/collecting_wakeword_data.md`) and
retrain with the **same** pipeline (identical features + GMM + threshold) —
the model is a drop-in.
