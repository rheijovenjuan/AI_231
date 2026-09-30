"""Local (PC) test harness for the Rapi voice-command model.

Examples
--------
Classify one file, wake word included:

    python runtime/pc_test.py --model output/onnx --file "Hey Rapi play music.wav"

Classify a directory of command clips (wake word bypassed):

    python runtime/pc_test.py --model output/onnx --dir data/test_clips --no-wake

Live microphone:

    python runtime/pc_test.py --model output/onnx --mic

Machine-readable output:

    python runtime/pc_test.py --model output/onnx --file a.wav --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F                     # noqa: E402
from runtime.pipeline import VoicePipeline             # noqa: E402


def iter_files(args):
    files, seen = [], set()
    for p in list(args.file or []):
        if os.path.normcase(p) not in seen:
            seen.add(os.path.normcase(p))
            files.append(p)
    for d in args.dir or []:
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for p in sorted(glob.glob(os.path.join(d, "**", ext),
                                      recursive=True)):
                if os.path.normcase(p) not in seen:
                    seen.add(os.path.normcase(p))
                    files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="output/onnx",
                    help="directory with wake.onnx / keyword.onnx / command.onnx")
    ap.add_argument("--file", action="append", help="audio file (repeatable)")
    ap.add_argument("--dir", action="append", help="directory of audio files")
    ap.add_argument("--mic", action="store_true", help="use the microphone")
    ap.add_argument("--no-wake", action="store_true",
                    help="skip wake detection, classify the whole clip")
    ap.add_argument("--json", action="store_true", help="print JSON per clip")
    ap.add_argument("--show-slot", action="store_true",
                    help="also print the recognised slot value")
    ap.add_argument("--playback", action="store_true",
                    help="play each clip (or the captured command with --mic) "
                         "out loud before classifying it")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0,
                    help="onnxruntime intra-op threads (0 = default)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="only use the first N seconds of each file")
    args = ap.parse_args()

    pipe = VoicePipeline(args.model, wake_threshold=args.threshold,
                         threads=args.threads)
    print(f"[model] {os.path.abspath(pipe.onnx_dir)}  "
          f"threshold={pipe.threshold:.3f} intents={len(pipe.intents)}",
          flush=True)

    if args.mic:
        from runtime.mic import listen
        listen(pipe, show_slot=args.show_slot, as_json=args.json,
               playback=args.playback)
        return

    files = iter_files(args)
    if not files:
        ap.error("give --file/--dir or use --mic")

    counts, detected, t0 = {}, 0, time.perf_counter()
    for path in files:
        wave = F.load_audio(path)
        if args.seconds:
            wave = wave[:int(args.seconds * F.SAMPLE_RATE)]
        if args.playback:
            from runtime.mic import play
            print(f"[play] {os.path.basename(path)}", flush=True)
            play(wave, sr=F.SAMPLE_RATE)
        if args.no_wake:
            res = pipe.classify(wave)
            res["detected"] = True
        else:
            res = pipe.scan(wave)
        detected += bool(res.get("detected"))
        if res.get("detected"):
            counts[res["intent"]] = counts.get(res["intent"], 0) + 1

        if args.json:
            out = dict(file=path, **{k: v for k, v in res.items()
                                     if k != "scores"})
            print(json.dumps(out), flush=True)
        elif res.get("detected"):
            ws = res.get("wake_score")
            wtag = f"[wake {ws:.2f}]" if ws is not None else "[wake ----]"
            line = f"{wtag} {res['intent']:16s} conf={res['confidence']:.2f}"
            if args.show_slot and "slot" in res:
                line += f"  slot={res['slot']}"
            print(f"{os.path.basename(path):52s} {line}", flush=True)
        else:
            print(f"{os.path.basename(path):52s} "
                  f"[wake {res.get('wake_score', 0.0):.2f}] "
                  f"-- no wake word --", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{len(files)} files, {detected} woken, {dt:.1f}s "
          f"({dt / max(len(files), 1):.3f}s/file)", flush=True)
    if counts:
        print("class histogram:", json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
