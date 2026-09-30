# Installing on a Raspberry Pi 4 (4 GB)

Target: **Raspberry Pi OS 64-bit** (Bookworm), Python 3.11, a USB mic and a
speaker (3.5 mm jack or USB). Everything is CPU-only - no GPU, no training on
the Pi: the trained ONNX models ship inside `output/onnx/` of this repo.

Memory budget on 4 GB: OS ~250 MB + onnxruntime ~100 MB + faster-whisper
`base.en` int8 ~400-500 MB -> comfortably under 1 GB total. If you are tight
(forget-me-not desktop use), run with `--stt-model tiny.en` or
`--no-transcribe`.

## 1. System packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip \
                    portaudio19-dev libsndfile1 ffmpeg
```

- `portaudio19-dev` - required by `sounddevice` (mic capture)
- `libsndfile1` - required by `soundfile` (wav/mp3 decoding)
- `ffmpeg` - optional, extra codec fallback for pygame

## 2. Get the code

```bash
git clone https://github.com/<your-user>/MEX2.git ~/mex2
cd ~/mex2
```

No model download step: `output/onnx/` (wake.onnx, keyword.onnx,
command.onnx, model_card.json, ~1.2 MB) is part of the repository.

## 3. Python environment + dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs onnxruntime, sounddevice, pygame, edge-tts, soundfile and
faster-whisper (CPU int8). Expect ~300 MB of downloads; on a slow SD card
give it 5-10 minutes.

## 4. First run (no mic needed)

```bash
python app.py --mode file --file test_clips/commands/PLAY_MUSIC_s99_v3_clean.wav
```

Expected output order:

```
you said: "play some music"
>>> PLAY_MUSIC   keywords=[...] confidence=...
Playing <track>
```

The **first** transcribing run downloads the whisper model (`base.en`,
~145 MB) into `~/.cache/huggingface` - do this while online.

## 5. Live demo

```bash
python app.py --mode mic --gui     # GUI with panels + log + Exit button
python app.py --mode mic           # headless console
```

Say **"Hey Rapi"**, hear **"Yes?"**, then speak a command (e.g.
"dim the lights to 40 percent", "message Anna", "volume 50",
"set an alarm for 7 30 am").

## 6. Audio devices

```bash
# list devices and note the input index
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Make the USB mic the default input (`raspi-config` -> Advanced -> Audio),
and test with `arecord -d 3 test.wav && aplay test.wav`.

## 7. Tuning knobs for the Pi

```bash
--wake-trigger 0.0     # raise (e.g. 0.40 -> 0.55) if false wakes bother you
--min-confidence 0.45  # refuse to act on low-confidence keywords
--vad-rms 2e-4         # speech energy gate; raise in noisy rooms
--stt-model tiny.en    # faster transcription, less RAM
--no-transcribe        # acoustic-only (fastest, no whisper at all)
--confirm beep         # instead of spoken "Yes?"
```

The shipped wake threshold lives in `output/onnx/model_card.json`
(`wake_threshold`, 0.40) - edit it or override with `--wake-trigger`.

## 8. Run at boot (optional)

```bash
cat > ~/mex2/run_rapi.sh <<'EOF'
#!/usr/bin/env bash
cd ~/mex2
source venv/bin/activate
export DISPLAY=:0
exec python app.py --mode mic --gui
EOF
chmod +x ~/mex2/run_rapi.sh

# autostart on the desktop session:
mkdir -p ~/.config/lxsession/LXDE-pi
echo "@~/mex2/run_rapi.sh" >> ~/.config/lxsession/LXDE-pi/autostart
```

## 9. Benchmarks

Targets for a Pi 4 (4 GB) and the exact commands to measure them are in
[`BENCHMARKS.md`](BENCHMARKS.md). Quick sanity:

```bash
python benchmark.py --n 50
```

- wake scan ~5-15 ms per 100 ms window
- command classify <100 ms
- transcription dominates end-to-end latency (~1-3 s with `base.en`,
  ~0.5-1 s with `tiny.en`)

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `PortAudio library not found` | `sudo apt install portaudio19-dev`, reinstall sounddevice in the venv |
| `sndfile library not found` | `sudo apt install libsndfile1` |
| No mic detected | `sd.query_devices()`, set default input in `raspi-config` |
| `invalid sample rate [PaErrorCode -9997]` | mic has no 16 kHz mode - the app auto-falls-back to the device's native rate and resamples to 16 kHz (`[mic] device has no 16 kHz mode ... capturing at 44100 Hz`). If it still fails: `python -m sounddevice` to list devices / fix the default in `raspi-config` |
| GUI won't start | need a display (`DISPLAY=:0`) or VNC; headless -> `--mode mic` without `--gui` |
| Whisper download fails | run once while online, or pre-copy `~/.cache/huggingface` |
| High latency / stutter | use `--stt-model tiny.en`, close the desktop browser, check `vcgencmd measure_temp` for throttling |
| False wakes | raise `--wake-trigger` / `wake_threshold` in the model card |
| No sound on "play music" | pygame needed for mp3: verify `python -c "import pygame; pygame.mixer.init(frequency=16000)"` |
