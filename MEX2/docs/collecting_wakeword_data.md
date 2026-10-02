# Collecting real "Hey Rapi" wake-word data

The command dataset (GitHub `MEX2/Data`) contains **commands**, not the wake
word. The shipped `models/wakeword.joblib` is therefore a **proxy**: positives
are short command clips, negatives are other commands + silence. It works for a
demo but is not a real "Hey Rapi" detector.

To make it real, collect your own data and retrain.

## What to record

**Positives (~200–400 clips of "Hey Rapi"):**
- Multiple speakers (ideally 5–10 people), multiple accents.
- Varying distance, pace, and background (quiet room, light TV, fan).
- 0.5–1.5 s each, 16 kHz mono WAV.
- Include near-misses you *want* to trigger ("hey rapi", "hey rapi?").

**Negatives (~500+ clips):**
- Silence / room tone (several seconds each).
- Other words that rhyme or share sounds: "hey baby", "hey rap", "hey rapid",
  "heir app", "hair up", general conversation.
- Background noise without the phrase.

## How to record (PC, easiest)

A helper script ships at the repo root — 16 kHz mono WAVs, numbered
`hey_rapi_001.wav` ... , with peak-level warnings and optional playback:

```bash
python record_wake.py --count 10      # 10 clips back-to-back -> wakeword_data/positive/
python record_wake.py                 # interactive: Enter = one clip, q = quit
python record_wake.py --play          # listen back after every clip
python record_wake.py --device 2      # pick a mic (index or name substring)
python record_wake.py --out wakeword_data/negative --prefix room_tone --seconds 5
```

Score a clip against the shipped `wake.onnx` (target: > 0.40):

```bash
python runtime/pc_test.py --model output/onnx --file wakeword_data/positive/hey_rapi_001.wav
```

CMD-only alternative (needs ffmpeg): record with

```bash
mkdir wakeword_data\positive
ffmpeg -y -f dshow -i audio="Headset (EarPods)" -t 2 -ar 16000 -ac 1 wakeword_data\positive\hey_rapi_001.wav
```

(listenable device names: `ffmpeg -list_devices true -f dshow -i dummy 2>&1 | findstr /i audio`)

## How to record (Pi / phone)

On the Pi or a phone, record 16 kHz mono WAVs. A tiny recorder:

```python
import sounddevice as sd, soundfile as sf, numpy as np
sr = 16000
print("Recording 3 s...")
x = sd.rec(int(3*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
sf.write("hey_rapi_001.wav", x, sr)
```

Organise as:
```
wakeword_data/
├── positive/   hey_rapi_001.wav ...
└── negative/   silence_001.wav, other_001.wav ...
```

## Retrain

Add a small script (or extend `train.py`) that:
1. Loads all positives → `pos_feats`, all negatives → `neg_feats`
   (using `voice_assistant.features.extract_from_file`).
2. Calls `WakeWordDetector(model_path).save(pos_feats, neg_feats)`.
3. Evaluates: true-positive rate at a chosen false-alarm rate, and the
   score-separation (the `wakeword_scores.png` histogram).

Aim for: **>95% detection** of real "Hey Rapi" with **<1 false alarm/hour**
of ambient audio.

## Calibration

After retraining, set `assistant.WAKE_THRESHOLD` (or the `--wake-threshold`
you pass to `Assistant`) to a value on the ROC curve that balances missed
wakes vs. false triggers. Log both rates in `reports/`.
