"""Record "Hey Rapi" wake-word clips (16 kHz mono WAV) from a PC microphone.

Interactive (default):

    python record_wake.py                # Enter = record one clip, q = quit
    python record_wake.py --play         # also listen back after each clip

Batch:

    python record_wake.py --count 10     # 10 clips back-to-back, then exit

Clips land in wakeword_data/positive/ (negatives: --out wakeword_data/negative
while recording room tone, "hey baby", "hey rap", ...). Score them against the
shipped wake model:

    python runtime/pc_test.py --model output/onnx --file wakeword_data/positive/hey_rapi_001.wav

Use them end-to-end by copying into test_clips/wake/ and running
test_clips/build_combined.py + test_clips/check_end2end.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from voice_assistant import features as F  # noqa: E402


def record(seconds: float, device) -> np.ndarray:
    """Capture `seconds` at 16 kHz mono; fall back to native rate + resample
    if the device refuses to open at 16 kHz (same approach as audio_input)."""
    n = int(seconds * F.SAMPLE_RATE)
    try:
        x = sd.rec(n, samplerate=F.SAMPLE_RATE, channels=1, dtype="float32",
                   device=device)
        sd.wait()
    except Exception:
        dev = device if isinstance(device, int) else sd.default.device[0]
        native = int(sd.query_devices(dev)["default_samplerate"])
        x = sd.rec(n, samplerate=native, channels=1, dtype="float32",
                   device=device)
        sd.wait()
        x = F.resample_linear(x, native, F.SAMPLE_RATE)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record Hey Rapi wake-word clips (16 kHz mono WAV)")
    ap.add_argument("--out", default=os.path.join(HERE, "wakeword_data",
                                                  "positive"),
                    help="output folder (default wakeword_data/positive)")
    ap.add_argument("--prefix", default="hey_rapi",
                    help="filename prefix (hey_rapi_001.wav, ...)")
    ap.add_argument("--seconds", type=float, default=1.5,
                    help="clip length in seconds (default 1.5)")
    ap.add_argument("--count", type=int, default=0,
                    help="record N clips back-to-back, then exit "
                         "(default: interactive, one per Enter)")
    ap.add_argument("--device", default=None,
                    help="sounddevice index or name substring, "
                         "e.g. --device 1 or --device 'USB' "
                         "(see python -m sounddevice)")
    ap.add_argument("--play", action="store_true",
                    help="play each clip back after recording it")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    info = (sd.query_devices(args.device, "input") if args.device is not None
            else sd.query_devices(kind="input"))
    print(f"Recording {args.seconds:.1f}s clips at {F.SAMPLE_RATE} Hz mono")
    print(f"  input : [{info['index']}] {info['name']}")
    print(f"  output: {args.out}")

    idx = 1
    while os.path.exists(os.path.join(args.out, f"{args.prefix}_{idx:03d}.wav")):
        idx += 1

    done = 0
    while True:
        if args.count:
            if done >= args.count:
                break
            print(f"[{done + 1}/{args.count}] speak in 3... 2... 1...")
            time.sleep(1.0)
        else:
            try:
                ans = input("Enter = record (q + Enter = quit) > ").strip().lower()
            except EOFError:
                break
            if ans in ("q", "quit", "exit"):
                break
        x = record(args.seconds, args.device)
        path = os.path.join(args.out, f"{args.prefix}_{idx:03d}.wav")
        sf.write(path, x, F.SAMPLE_RATE)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        warn = ("  <-- CLIPPING, move back?" if peak > 0.99 else
                "  <-- very quiet, move closer?" if peak < 0.01 else "")
        print(f"  saved {os.path.basename(path)}  "
              f"{len(x) / F.SAMPLE_RATE:.2f}s  peak={peak:.2f}{warn}")
        if args.play:
            try:
                sd.play(x, F.SAMPLE_RATE)
                sd.wait()
            except Exception as exc:  # noqa: BLE001
                print(f"  (playback unavailable: {exc})")
        idx += 1
        done += 1

    print(f"done - {done} clip(s) in {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
