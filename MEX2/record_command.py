"""Record real CALL / COLOR command clips (16 kHz mono WAV) from a PC mic.

The keyword spotter confuses "call" with "color" on real voices (TTS train
data is 426 CALL vs 1859 COLOR clips, none from a real speaker). Record
real-voice clips for BOTH intents - the script interleaves them 2 CALL : 1
COLOR so CALL gets the extra mass.

Interactive (default): shows a phrase, Enter = record, r = redo, s = skip,
q = quit.

    python record_command.py                 # both intents, cycles phrases
    python record_command.py --intent call   # only CALL phrases
    python record_command.py --play          # listen back after each clip

Batch:

    python record_command.py --count 30      # 30 clips back-to-back, then exit

Clips land in command_data/<LABEL>/ and every clip is appended to
command_data/manifest.csv with the exact columns of data/manifest.csv
(path, label, intent, speaker, split, phrase_id, variant_id, transcript,
slot, slot_value, duration_sec), split=train, speaker=real1.

Then on the DGX: upload the folder, merge the manifest rows into
dataset/.../OptionB/manifest.csv, re-extract features and retrain the
keyword + command heads (see docs/TRAINING.md).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from voice_assistant import features as F  # noqa: E402

MANIFEST_COLS = ["path", "label", "intent", "speaker", "split", "phrase_id",
                 "variant_id", "transcript", "slot", "slot_value",
                 "duration_sec"]

# (phrase, slot_value) - CALL mirrors the three TTS templates plus natural
# contact phrases; COLOR is exactly the 12 TTS templates (hard negatives).
CALL_PHRASES = [
    ("Call", ""),
    ("Place a call", ""),
    ("Make a phone call", ""),
    ("Call mom", ""),
    ("Call dad", ""),
    ("Call Sarah", ""),
    ("Call Daniel", ""),
    ("Call John", ""),
    ("Call Emily", ""),
    ("Call the office", ""),
    ("Call me back", ""),
    ("Can you call Sarah", ""),
]
COLOR_PHRASES = [
    ("Color red", "red"),
    ("Color blue", "blue"),
    ("Color green", "green"),
    ("Color yellow", "yellow"),
    ("Change the lights to red", "red"),
    ("Change the lights to blue", "blue"),
    ("Change the lights to green", "green"),
    ("Change the lights to yellow", "yellow"),
    ("Set the lights to red", "red"),
    ("Set the lights to blue", "blue"),
    ("Set the lights to green", "green"),
    ("Set the lights to yellow", "yellow"),
]


def beep(freq: float = 880.0, dur: float = 0.15, vol: float = 0.3) -> None:
    """Short 'speak now' tone right before capture (best-effort)."""
    try:
        t = np.linspace(0.0, dur, int(F.SAMPLE_RATE * dur), False)
        tone = vol * np.sin(2 * np.pi * freq * t)
        fade = int(0.01 * F.SAMPLE_RATE)
        tone[:fade] *= np.linspace(0.0, 1.0, fade)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade)
        sd.play(tone.astype(np.float32), F.SAMPLE_RATE)
        sd.wait()
    except Exception as exc:  # noqa: BLE001 - audio is optional
        print(f"  (beep unavailable: {exc})")


def record(seconds: float, device) -> np.ndarray:
    """Capture `seconds` at 16 kHz mono; fall back to native rate + resample
    if the device refuses to open at 16 kHz (same approach as record_wake)."""
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


def trim_speech(x: np.ndarray, frame_ms: int = 30,
                margin: float = 0.15) -> np.ndarray:
    """Drop leading/trailing quiet frames, keeping `margin` seconds of pad."""
    n = int(F.SAMPLE_RATE * frame_ms / 1000)
    if x.size < 2 * n:
        return x
    fr = x[:len(x) // n * n].reshape(-1, n)
    voiced = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    hot = np.where(voiced > 0.01)[0]
    if hot.size == 0:
        return x
    keep = int(margin * F.SAMPLE_RATE / n)
    a = max(int(hot[0]) - keep, 0) * n
    b = min(int(hot[-1]) + 1 + keep, len(fr)) * n
    return x[a:b]


def phrase_pool(intent: str):
    # label = OptionB folder name: CALL / COLOR_RED / ... (folder_to_intent)
    if intent == "call":
        return [(t, v, "CALL") for t, v in CALL_PHRASES]
    if intent == "color":
        return [(t, v, f"COLOR_{v.upper()}") for t, v in COLOR_PHRASES]
    return ([(t, v, "CALL") for t, v in CALL_PHRASES]
            + [(t, v, f"COLOR_{v.upper()}") for t, v in COLOR_PHRASES])


def next_phrases(intent: str):
    """Yield (transcript, slot_value, label, phrase_id) forever.

    both: 2 CALL prompts per COLOR prompt (CALL is the under-represented
    class); phrase ids restart per label (r1..rN, matching v1..vN style)."""
    if intent != "both":
        pool = phrase_pool(intent)
        i = 0
        while True:
            t, v, label = pool[i % len(pool)]
            yield t, v, label, f"r{i % len(pool) + 1}"
            i += 1
    ci = ki = 0
    call = [(t, v, "CALL") for t, v in CALL_PHRASES]
    color = [(t, v, f"COLOR_{v.upper()}") for t, v in COLOR_PHRASES]
    step = 0
    while True:
        if step % 3 == 1:                       # every 3rd prompt is COLOR
            t, v, label = color[ki % len(color)]
            yield t, v, label, f"r{ki % len(color) + 1}"
            ki += 1
        else:
            t, v, label = call[ci % len(call)]
            yield t, v, label, f"r{ci % len(call) + 1}"
            ci += 1
        step += 1


def load_manifest(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_manifest(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record real CALL / COLOR command clips for retraining")
    ap.add_argument("--out", default=os.path.join(HERE, "command_data"),
                    help="output folder (default command_data)")
    ap.add_argument("--intent", choices=("both", "call", "color"),
                    default="both",
                    help="phrase pool (default both = 2 CALL : 1 COLOR)")
    ap.add_argument("--seconds", type=float, default=2.5,
                    help="clip length in seconds (default 2.5 = cmd window)")
    ap.add_argument("--count", type=int, default=0,
                    help="record N clips back-to-back, then exit "
                         "(default: interactive)")
    ap.add_argument("--device", default=None,
                    help="sounddevice index or name substring, "
                         "e.g. --device 1 or --device 'USB'")
    ap.add_argument("--speaker", default="real1",
                    help="speaker tag (default real1; keep it out of "
                         "val/test so the split stays speaker-disjoint)")
    ap.add_argument("--split", default="train", choices=("train", "val"),
                    help="manifest split (default train; test is refused)")
    ap.add_argument("--play", action="store_true",
                    help="play each clip back after recording it")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep leading/trailing silence instead of trimming")
    args = ap.parse_args()

    info = (sd.query_devices(args.device, "input") if args.device is not None
            else sd.query_devices(kind="input"))
    man_path = os.path.join(args.out, "manifest.csv")
    existing = load_manifest(man_path)
    taken = {r["path"] for r in existing}

    print(f"Recording {args.seconds:.1f}s clips at {F.SAMPLE_RATE} Hz mono")
    print(f"  input : [{info['index']}] {info['name']}")
    print(f"  output: {args.out}  (manifest: {len(existing)} rows)")
    print(f"  pool  : {args.intent}  "
          f"({len(CALL_PHRASES)} CALL / {len(COLOR_PHRASES)} COLOR phrases)")

    prompts = next_phrases(args.intent)
    done = {"CALL": 0, "COLOR": 0}
    session = 0
    cur = next(prompts)
    prev = None
    while True:
        if args.count and session >= args.count:
            break
        transcript, slot_value, label, phrase_id = cur
        tag = f"[{session + 1}]" + (f"/{args.count}" if args.count else "")
        print(f"\n{tag} {label}: {transcript!r}")
        if args.count:
            time.sleep(1.2)               # reading time for the phrase
            print("  *beep* - speak now")
            beep()
        else:
            try:
                ans = input("  Enter = record, r = redo last, s = skip, "
                            "q = quit > ").strip().lower()
            except EOFError:
                break
            if ans in ("q", "quit", "exit"):
                break
            if ans in ("s", "skip"):
                cur = next(prompts)
                continue
            if ans in ("r", "redo"):
                if prev is None:
                    print("  (nothing to redo yet)")
                    continue
                cur = prev
                transcript, slot_value, label, phrase_id = cur
                print(f"  {label}: {transcript!r}")
            time.sleep(0.5)               # settle after the keypress
            print("  *beep* - speak now")
            beep()

        x = record(args.seconds, args.device)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.99:
            print("  CLIPPING - move back? saved anyway")
        elif peak < 0.01:
            print("  very quiet - move closer? (r = redo this phrase)")
        if not args.no_trim:
            x = trim_speech(x)

        take = 1
        while (f"{label}/{label}_{args.speaker}_{phrase_id}"
               f"_take{take}.wav") in taken:
            take += 1
        rel = (f"{label}/{label}_{args.speaker}_{phrase_id}"
               f"_take{take}.wav")
        path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, x, F.SAMPLE_RATE)
        row = {"path": rel, "label": label,
               "intent": "COLOR" if label.startswith("COLOR") else label,
               "speaker": args.speaker, "split": args.split,
               "phrase_id": phrase_id, "variant_id": "clean",
               "transcript": transcript,
               "slot": "color" if label.startswith("COLOR") else "",
               "slot_value": slot_value,
               "duration_sec": f"{len(x) / F.SAMPLE_RATE:.3f}"}
        append_manifest(man_path, row)
        taken.add(rel)
        done["CALL" if label == "CALL" else "COLOR"] += 1
        print(f"  saved {rel}  {len(x) / F.SAMPLE_RATE:.2f}s  "
              f"peak={peak:.2f}  slot={slot_value or '-'}")
        if args.play:
            try:
                sd.play(x, F.SAMPLE_RATE)
                sd.wait()
            except Exception as exc:  # noqa: BLE001
                print(f"  (playback unavailable: {exc})")
        session += 1
        prev, cur = cur, next(prompts)

    print(f"\ndone - {session} clip(s): CALL {done['CALL']}, "
          f"COLOR {done['COLOR']}")
    print(f"manifest: {os.path.abspath(man_path)}")
    if session:
        print("next: upload to the DGX and merge the manifest rows into the "
              "OptionB dataset (docs/TRAINING.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
