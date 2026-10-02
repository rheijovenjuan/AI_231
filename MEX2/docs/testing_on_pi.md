# Testing on the Raspberry Pi 4

On-device acceptance runbook. Install first with the
[install guide](raspberry_pi.md) (sections 1–4); this document is what to
**run and check** once the environment is up. Every expected value below is
measured — sources are linked next to each number.

Run everything from the repo root (`MEX2/`), inside the venv:

```bash
cd ~/AI_231/MEX2 && source .venv/bin/activate
```

---

## 1. No-mic smoke tests (run these first)

```bash
# fast full regression, no Whisper / beeps / TTS (~10 s)
python test_clips/sim_quick.py                 # expect: intent 18/19, exit 0
                                                # (v45 endpoint-window miss,
                                                #  see ACCURACY.md § 6)

# wake + command over the 19 combined clips (ground truth file)
python test_clips/check_end2end.py             # expect: wake 19/19,
                                               # intent 19/19 (run-5 fixed the
                                               # old NEXT miss), exit 0

# wake scores on the committed real recordings (target >= 0.90)
python runtime/pc_test.py --dir wakeword_data/positive   # expect: 11/11

# wake scores on the 12 held-out TTS wake clips
python runtime/pc_test.py --dir test_clips/wake          # expect: 12/12
```

## 2. Benchmark

```bash
python runtime/benchmark.py --model output/onnx --label rpi4 --threads 2
```

Writes `output/onnx/benchmark_rpi4.json` and prints a markdown table.
**Always use `--threads 2`** (leaves cores free for audio/system).

Expected on a Pi 4 4 GB (measured 2026-10-02, [`BENCHMARKS.md` § 2]):

| quantity | expected |
|---|---:|
| model load | ~650 ms |
| wake inference p50 / p99 | ~17.2 / ~32.1 ms |
| keyword inference p50 / p99 | ~40.0 / ~70.7 ms |
| scan of 4 s audio p50 | ~225 ms |
| wake + command end-to-end (est.) | ~69 ms |
| peak RSS | ~147 MB |
| real-time factor | ~0.016 |

If your numbers are >2× worse: check `onnxruntime` is the aarch64 wheel,
`--threads 2` was used, and no other build is compiling in the background.
Full PC↔Pi comparison: [`BENCHMARKS.md`](BENCHMARKS.md).

## 3. Live microphone test

```bash
python app.py --mode mic              # voice ack ("Yes?")
python app.py --mode mic --gui        # same + status/log window
```

Check, in order:

1. **Wake** — say "Hey Rapi": the acknowledgement beep plays, log shows
   the wake score ≥ 0.90. Try it from 1–2 m away and with the
   recordings played through a phone speaker.
2. **Command** — after the beep, say e.g. "set alarm for 8 am" →
   `ALARM conf=…`, action runs.
3. **Endpointing** — pause mid-phrase (~0.5 s): it keeps listening
   (pause tolerance 0.90 s). A short clip of rambling speech stops at
   **3.5 s** max; silence gives up at **8 s** (`TIMEOUT` event).
   Nothing slow is left in the loop — these are covered by
   `test_endpoint.py` (run on PC).
4. **Acknowledgement beeps** are clearly audible (volume 0.8, 2 beeps).
5. **Music** — say "play music": playback starts essentially instantly
   (the warm-up preloads a track in the background; first cold play on
   Windows was 777 ms → warm 0.7 ms, same mechanism on the Pi).
6. **False wakes** — talk normally / play music for a couple of minutes
   without the wake phrase: no trigger (measured 0/19 on command-only
   clips, `ACCURACY.md` § 5).

## 4. Flags to verify (all shipped, all visible here)

```bash
python app.py --mode mic --gui --small-screen   # compact layout for the
                                                # 3.5in 480x320 touchscreen
python app.py --mode mic --no-transcribe        # Whisper still runs for
                                                # intent/slots, transcript
                                                # hidden from log/UI
python app.py --mode mic --no-asr               # no Whisper at all —
                                                # acoustic keyword path only
                                                # (fastest; fixed vocab slots)
python app.py --mode mic --stt-model tiny.en    # smaller/faster Whisper
python app.py --mode mic --wake-trigger 0.95    # raise wake threshold if
                                                # false wakes bother you
                                                # (0 = card value 0.900)
python app.py --mode mic --confirm beep         # ack style: voice | beep | off
python app.py --mode mic --device 2             # pick the USB mic
```

Also quick: `python app.py --mode file --file test_clips/commands/...wav`
runs the whole pipeline off a file (transcript first, then the intent).

## 5. Record more wake data on the Pi (optional)

```bash
python record_wake.py --count 10                # -> wakeword_data/positive/
python runtime/pc_test.py --file wakeword_data/positive/hey_rapi_001.wav
```

Target: wake score ≥ 0.90 (the committed 11 clips all score ≥ 0.99).
Score-guide and retrain pointer: [`collecting_wakeword_data.md`](collecting_wakeword_data.md).

## 6. Acceptance table

| # | check | expected | where |
|---:|---|---|---|
| 1 | `sim_quick.py` | 18/19, exit 0 | § 1 |
| 2 | `check_end2end.py` | wake 19/19, intent 19/19 | § 1 |
| 3 | `pc_test --dir wakeword_data/positive` | 11/11 ≥ 0.90 | § 1 |
| 4 | benchmark (`--threads 2`) | wake p50 ≤ 25 ms, keyword p50 ≤ 60 ms, RSS ≤ 200 MB, RTF ≤ 0.05 | § 2 |
| 5 | live wake + command | fires reliably, action runs | § 3 |
| 6 | endpointing behaviours | pause kept, 3.5 s cap, 8 s timeout | § 3 |
| 7 | `--small-screen` GUI | fits 480×320, no clipped panels | § 4 |
| 8 | `--no-transcribe` / `--no-asr` | as documented, both start fine | § 4 |
| 9 | false wakes (no wake phrase) | none in a few minutes of audio | § 3 |

Save `output/onnx/benchmark_rpi4.json` (it is committed) so runs stay
comparable; note the Pi model/OS/threads in the PR if numbers move.
