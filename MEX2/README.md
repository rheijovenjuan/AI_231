# Rapi - combined voice assistant (assistant shell + ONNX models)

The smart-home assistant UI/actions from the `voice_assistant` project, driven
by the trained ONNX models from the voice-command-model project: **"Hey Rapi"
wake word -> 40-keyword spotter -> 19 intents + slot -> action dispatcher**.
CPU-only, runs on a desktop or a Raspberry Pi 4 (4 GB).

```
 mic -> VAD/gate -> wake.onnx ("Hey Rapi", 1.2 s window)
                        |
                        v  settle 0.3 s, onset, 0.5 s trailing-silence endpoint
                   keyword.onnx (40 multi-label keywords, 2.5 s window)
                        |
                        v  deterministic rule (rapi_vcm.keywords)
              19-class intent + slot value
                        |
                        v
              confidence gate -> ActionDispatcher -> Tkinter UI / console
```

Intent and slot are decided directly from the keyword probabilities (two
small CNN forward passes) - speech-to-text is **display-only**: with
`faster-whisper` installed the app prints what it heard *before* the intent,
so you can verify the microphone input. The wake word is acknowledged with a
spoken **"Yes?"** (edge-tts, cached in `audio/confirm_yes.wav`) instead of a
beep.

## What you need

`output/onnx/` must contain the trained artifacts (copied from the
voice-command-model project):

| file | purpose |
|------|---------|
| `wake.onnx` | "Hey Rapi" detector (threshold from `model_card.json`, 0.90) |
| `keyword.onnx` | 40-keyword multi-label spotter + per-keyword thresholds |
| `command.onnx` | softmax baseline head (same features, not used by the app) |
| `model_card.json` | classes, thresholds, keyword thresholds, metrics |

All training / performance / inference documentation lives in [`docs/`](docs/):

| document | what it covers |
|----------|----------------|
| [docs/TRAINING.md](docs/TRAINING.md) | dataset, features, how the ONNX heads were trained (DGX), export + parity |
| [docs/ACCURACY.md](docs/ACCURACY.md) | measured accuracy: intent, slots, wake FAR/FRR curves |
| [docs/INFERENCE.md](docs/INFERENCE.md) | runtime API (`VoicePipeline`), thresholds, flags |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | latency benchmarks + Raspberry Pi measurement commands |
| [docs/raspberry_pi.md](docs/raspberry_pi.md) | **install guide for Raspberry Pi 4 (4 GB)** |
| [docs/testing_on_pi.md](docs/testing_on_pi.md) | **on-device acceptance runbook for the Pi (benchmark + live tests)** |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | submission table, reviewer checklist, poster fields (dataset/cluster/weights) |
| [docs/improving_accuracy.md](docs/improving_accuracy.md) | what was tried, measured results, what's left |
| [docs/testing_checklist.md](docs/testing_checklist.md) | manual test checklist |
| [docs/collecting_wakeword_data.md](docs/collecting_wakeword_data.md) | recording real "Hey Rapi" data |
| [docs/synthetic_wakeword.md](docs/synthetic_wakeword.md) | legacy GMM wake-word path (not used by app) |

## Quick start

```bash
pip install -r requirements.txt

# 1. file mode - run recorded commands through the whole intent pipeline
#    (prints the transcription first, then the intent)
python app.py --mode file --file test_clips/commands/PLAY_MUSIC_s99_v3_clean.wav

# 2. live mic: say "Hey Rapi", hear "Yes?", then say a command
python app.py --mode mic

# 3. Tkinter GUI (status panels + log; Exit button closes the app)
python app.py --mode gui
python app.py --mode mic --gui      # GUI + mic
python app.py --mode mic --gui --small-screen  # compact GUI for a
                                    # 3.5in 480x320 Pi touchscreen
```

Useful flags:

```bash
--onnx DIR             # model directory (default: output/onnx)
--min-confidence 0.45  # refuse to dispatch below this keyword confidence
--wake-trigger 0.0     # raise the wake score needed to open the mic
                       # (0 = calibrated card threshold; try 0.8 to cut
                       #  false wakes if your environment allows it)
--vad-rms 2e-4         # speech/silence energy gate (mean square / 30 ms)
--transcribe           # show the STT transcription first (default on;
                       # --no-transcribe disables; first run downloads
                       # whisper base.en ~145 MB)
--stt-model base.en    # faster-whisper model size (tiny.en is faster)
--confirm voice        # wake acknowledgement: spoken "Yes?" (default) /
                       # beep / off
--small-screen         # compact GUI layout for a 3.5in 480x320 display
                       # (Raspberry Pi touchscreen; use with --gui)
--device ID            # mic input: sounddevice index or name substring
                       # (default = system input; see python -m sounddevice)
```

Every decision is printed as `>>> INTENT` with `keywords=`/`confidence=`,
after the `you said: "..."` transcript; GUI events show both.

## Modes and what was fixed in this merge

- `voice_assistant/assistant.py` is now an **ONNX-backed** state machine
  (`runtime.VoicePipeline`): idle ring-buffer wake scan every 100 ms (scored
  only above the energy gate, two consecutive scans required), then command
  capture with wake-tail settle, voiced onset, and an endpoint on 0.5 s of
  trailing silence.
- Endpointing runs on **audio-sample time**, not wall clock, so file
  playback and live mic behave identically.
- Energy gate lowered to `2e-4` (rms 0.014): the previous 6e-3 treated quiet
  command speech as silence and the endpointer starved.
- Confidence gate: at least one active keyword and `confidence >=
  0.45`, and the `PLAY_MUSIC` rule fallback is never dispatched without
  play/music evidence - wrong guesses print "Not sure" instead of firing an
  action.
- `flush()` finishes a capture when the input ends (file mode / stream end).
- `--playback`-style confirmation is not here; use `runtime/mic.py` from the
  source project for that flow.

## Measured on this machine (test_clips)

| check | result |
|-------|--------|
| state machine (19 `test_clips/combined/*.wav`: wake + command) | **19/19** correct intent + slot |
| ... with transcription enabled | **19/19** transcript printed **before** the intent, 0 ordering violations |
| file mode (19 `test_clips/commands/*.wav`) | **19/19** correct |
| held-out test split (1842 clips, `data/manifest.csv`) | intent **99.29 %** (clean 99.35 / noisy 99.24) |
| hybrid transcript + intent (1510 clipped intents) | **99.93 %** (acoustic-only 99.54 %; 6 rescued by the transcript, 0 regressions) |
| text-slot rules on ground-truth transcripts (18375 clips) | **100 %** correct intent |
| false wake on command-only clips (no wake phrase) | **0/19** streaming (run-4 wake model; top full-clip score 0.571 never sustains the 2-window streak) |
| refused instead of guessed | 0 (confidence gate + fallback guard) |

## Layout

```
app.py                     # entry point: --mode file|mic|gui + flags
voice_assistant/
  assistant.py             # ONNX state machine (wake -> capture -> classify)
  stt.py                   # faster-whisper transcription (display + slots)
  confirm.py               # spoken "Yes?" wake acknowledgement (edge-tts)
  actions.py               # ActionDispatcher (music, lights, weather, ...)
  ui.py                    # Tkinter GUI (shows heard keywords)
  audio_input.py           # mic capture (30 ms chunks) / file source
  classifier.py            # text -> intent/slot helpers (contacts, colors)
  features.py              # MFCC features for the LEGACY GMM path only
rapi_vcm/                  # feature contract + keyword vocabulary/rules
runtime/pipeline.py        # VoicePipeline: wake.onnx + keyword.onnx inference
output/onnx/               # trained artifacts (see table above)
music/                     # your mp3/wav files for PLAY_MUSIC (picked at random)
test_clips/                # wake / command / combined wav clips
data/                      # OptionB commands (for file-mode experiments)
make_wake_data.py          # legacy GMM wake corpus builder (now uses edge-tts;
                           # listen to data/wake_tts/*.wav - not used by app.py)
models/, reports/          # legacy GMM artifacts (not used by app.py)
```

## Wake-word audio

`make_wake_data.py` (legacy GMM path) now synthesises with **edge-tts** so the
positives really say "Hey Rapi" - raw clips land in `data/wake_tts/` with a
`plan.json` describing each file. The old offline formant fallback is still
available via `--tts offline`, but its output does not sound like the phrase.
The ONNX `wake.onnx` used by this app was trained on edge-tts audio **plus 11
real user recordings** (`wakeword_data/positive/`, recorded with
`record_wake.py`, augmented 40x with clip-level split hygiene on the DGX -
[docs/TRAINING.md](docs/TRAINING.md) § 10); all 11 recordings now score
>= 0.90 (worst 0.996 - before the retrain one clip scored 0.038).

## Raspberry Pi

Full step-by-step install for a **Pi 4 (4 GB)**: see
[docs/raspberry_pi.md](docs/raspberry_pi.md) (or just run `./setup_pi.sh`).
What to run and check on the device (benchmark, smoke tests, live mic,
`--small-screen`, `--no-transcribe`, `--no-asr`): see
[docs/testing_on_pi.md](docs/testing_on_pi.md).
Same code, same flags - the trained ONNX models ship in `output/onnx/`, so
nothing needs training on the Pi.
