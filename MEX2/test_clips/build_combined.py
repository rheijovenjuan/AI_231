"""Concatenate "Hey Rapi" wake clips with command clips for end-to-end tests.

    python test_clips/build_combined.py

Writes test_clips/combined/*.wav plus a ground_truth.json
( {"<file>": {"intent": ..., "slot": ...}}, ... ).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WAKE_DIR = os.path.join(HERE, "wake")
CMD_DIR = os.path.join(HERE, "commands")
OUT_DIR = os.path.join(HERE, "combined")
MANIFEST = os.path.join(os.path.dirname(HERE), "dataset", "manifest.csv")


def load_manifest() -> dict:
    """{<folder>/<file>: (intent, slot_value)} from the OptionB manifest."""
    out = {}
    if not os.path.exists(MANIFEST):
        return out
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[os.path.basename(row["path"])] = (
                row["intent"], row.get("slot_value") or None)
    return out


def parse_command(path: str, manifest: dict):
    name = os.path.basename(path)
    if name in manifest:
        return manifest[name]
    intent = name.split("_s")[0]
    return intent, None


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    wakes = sorted(glob.glob(os.path.join(WAKE_DIR, "*.wav")))
    cmds = sorted(glob.glob(os.path.join(CMD_DIR, "*.wav")))
    if not wakes or not cmds:
        raise SystemExit(f"need wake/ and command clips under {HERE}")

    truth = {}
    manifest = load_manifest()
    gap = np.zeros(int(0.25 * F.SAMPLE_RATE), dtype=np.float32)
    tail = np.zeros(int(0.40 * F.SAMPLE_RATE), dtype=np.float32)

    for i, cp in enumerate(cmds):
        intent, slot = parse_command(cp, manifest)
        wp = wakes[i % len(wakes)]
        w = F.load_audio(wp)
        c = F.load_audio(cp)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if c.ndim > 1:
            c = c.mean(axis=1)
        w = w.astype(np.float32)
        # pad/truncate the wake clip to the detector window so that
        # scan() starts the command segment exactly at the gap
        wl = int(F.WAKE_SECONDS * F.SAMPLE_RATE)
        if w.shape[0] < wl:
            w = np.pad(w, (0, wl - w.shape[0]))
        else:
            w = w[:wl]
        audio = np.concatenate([w, gap, c.astype(np.float32), tail])
        name = f"{os.path.splitext(os.path.basename(wp))[0]}__" \
               f"{os.path.basename(cp)}"
        sf.write(os.path.join(OUT_DIR, name), audio, F.SAMPLE_RATE,
                 subtype="PCM_16")
        truth[name] = {"intent": intent, "slot": slot,
                       "wake_clip": os.path.basename(wp),
                       "command_clip": os.path.basename(cp)}

    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w",
              encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
    print(f"{len(truth)} combined clips -> {OUT_DIR}")


if __name__ == "__main__":
    main()
