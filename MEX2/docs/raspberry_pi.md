# Installing on a Raspberry Pi 4 (4 GB)

Target: **Raspberry Pi OS 64-bit** (Bookworm), Python 3.11, a USB mic and a
speaker (3.5 mm jack or USB). Everything is CPU-only - no GPU, no training on
the Pi: the trained ONNX models ship inside `output/onnx/` of this repo.

Memory budget on 4 GB: OS ~250 MB + onnxruntime ~100 MB + faster-whisper
`base.en` int8 ~400-500 MB -> comfortably under 1 GB total. If you are tight
(forget-me-not desktop use), run with `--stt-model tiny.en` or
`--no-asr`.

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
python app.py --mode mic --gui --small-screen   # same, compact layout for a
                                    # 3.5in 480x320 touchscreen HAT
python app.py --mode mic           # headless console
```

Say **"Hey Rapi"**, hear **"Yes?"**, then speak a command (e.g.
"dim the lights to 40 percent", "message Anna", "volume 50",
"set an alarm for 7 30 am").

## 6. Audio devices (USB mic + Bluetooth speaker)

List what is plugged in:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"  # inputs
aplay -l                                                        # outputs (ALSA)
wpctl status                       # Pi OS Bookworm (PipeWire): sinks/sources + ids
```

**Bluetooth speaker** (everything the app plays - music, "Yes?", beeps -
follows the system default output):

```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# scan on            # find the speaker's MAC, then:
[bluetooth]# pair AA:BB:CC:DD:EE:FF
[bluetooth]# connect AA:BB:CC:DD:EE:FF
[bluetooth]# trust AA:BB:CC:DD:EE:FF
[bluetooth]# quit

# make it the default output - Bookworm (PipeWire):
wpctl status                    # Sinks under Audio -> note the id
wpctl set-default <id>
# older releases (PulseAudio):
pactl set-default-sink bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink
```

**USB mic** (input):

```bash
# Bookworm (PipeWire): Sources under Audio -> note the id
wpctl set-default <id>
# PulseAudio: pactl list sources short; pactl set-default-source <NAME>
# legacy:     raspi-config -> Advanced Options -> Audio -> pick the USB mic
```

If the BT headset also exposes a *microphone*, keep the **USB mic** as the
default source - do not let Bluetooth grab the input.

Verify:

```bash
arecord -d 3 -f S16_LE -r 16000 t.wav && aplay t.wav   # mic -> speaker loop
aplay audio/confirm_yes.wav                            # "Yes?" through the BT speaker
```

**In the app**: output always follows the default sink; input uses the
default source. To force a specific input anyway:

```bash
python -m sounddevice                      # note the input index (or name)
python app.py --mode mic --gui --device 2  # or a name: --device "USB"
```

## 7. Tuning knobs for the Pi

```bash
--wake-trigger 0.0     # raise (e.g. 0.40 -> 0.55) if false wakes bother you
--min-confidence 0.45  # refuse to act on low-confidence keywords
--vad-rms 2e-4         # speech energy gate; raise in noisy rooms
--stt-model tiny.en    # faster transcription, less RAM
--no-asr               # acoustic-only (fastest, no whisper at all)
--no-transcribe        # keep whisper for intent/slots, hide its transcript
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
exec python app.py --mode mic --gui        # add --small-screen for a 3.5in display
EOF
chmod +x ~/mex2/run_rapi.sh

# autostart on the desktop session:
mkdir -p ~/.config/lxsession/LXDE-pi
echo "@~/mex2/run_rapi.sh" >> ~/.config/lxsession/LXDE-pi/autostart
```

## 9. Benchmarks

Measured Pi 4 numbers (wake p50 17.8 ms · keyword p50 39.4 ms · peak RSS
147 MB · RTF 0.016) and the reproduction command are in
[`BENCHMARKS.md`](BENCHMARKS.md) §2. Quick sanity:

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
| `[mic] input overflow` (occasional) | PortAudio dropped a chunk while the CPU was busy (Whisper/ONNX spikes) - harmless, one 30 ms gap. If constant: `--stt-model tiny.en`, close the desktop browser, check throttling (`vcgencmd measure_temp`) |
| GUI won't start | need a display (`DISPLAY=:0`) or VNC; headless -> `--mode mic` without `--gui` |
| Whisper download fails | run once while online, or pre-copy `~/.cache/huggingface` |
| High latency / stutter | use `--stt-model tiny.en`, close the desktop browser, check `vcgencmd measure_temp` for throttling |
| False wakes | raise `--wake-trigger` / `wake_threshold` in the model card |
| No sound on "play music" | pygame needed for mp3: verify `python -c "import pygame; pygame.mixer.init(frequency=16000)"` |
| BT speaker connected, no sound from the app | it follows the **default sink**: `wpctl status` + `wpctl set-default <id>` (or `pactl set-default-sink ...`), restart the app |
| Wrong mic used | `python -m sounddevice` to list inputs, then `--device <index>` (or `--device "USB"`); or set the default source (`wpctl set-default <id>`) |
