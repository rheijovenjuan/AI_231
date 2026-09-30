"""Raspberry Pi 4 runner: always-on wake word + voice command classifier.

    sudo apt install libportaudio2           # once, for the sounddevice backend
    pip install -r requirements_pi.txt

    python runtime/pi_run.py --model output/onnx
    python runtime/pi_run.py --model output/onnx --json --led-pin 17
    python runtime/pi_run.py --model output/onnx --playback   # demo: play the
                                      # captured command back before classifying

The script prints ONE line per recognised command (the class name), or a JSON
object when --json is given.  Ctrl-C stops it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "..", "output", "onnx"))
    ap.add_argument("--json", action="store_true", help="JSON output per event")
    ap.add_argument("--show-slot", action="store_true",
                    help="print the recognised slot value as well")
    ap.add_argument("--playback", action="store_true",
                    help="play the recorded command out loud before classifying it")
    ap.add_argument("--playback-device", default=None,
                    help="output device index/name for --playback "
                         "(default: system output)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2,
                    help="onnxruntime threads (2 keeps the Pi responsive)")
    ap.add_argument("--vad-rms", type=float, default=6e-3,
                    help="energy gate; below this no inference runs")
    ap.add_argument("--hop-ms", type=int, default=100)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "sounddevice", "pyaudio"])
    ap.add_argument("--device", default=None, help="input device index/name")
    ap.add_argument("--led-pin", type=int, default=None,
                    help="BCM pin for an activity LED (optional)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run forever)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # ---- keep the Pi cool / responsive -------------------------------------
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    from runtime.pipeline import VoicePipeline
    from runtime.mic import listen

    t0 = time.perf_counter()
    pipe = VoicePipeline(args.model, wake_threshold=args.threshold,
                         threads=args.threads)
    print(f"[pi] model loaded in {time.perf_counter() - t0:.2f}s "
          f"from {os.path.abspath(pipe.onnx_dir)}", flush=True)
    print(f"[pi] intents={len(pipe.intents)} "
          f"wake_threshold={pipe.threshold:.3f}", flush=True)

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    listen(pipe,
           backend=args.backend,
           device=device,
           hop_ms=args.hop_ms,
           vad_rms=args.vad_rms,
           max_seconds=args.duration,
           show_slot=args.show_slot,
           as_json=args.json,
           led_pin=args.led_pin,
           verbose=args.verbose,
           playback=args.playback,
           playback_device=args.playback_device)


if __name__ == "__main__":
    main()
