"""Build the "Hey Rapi" wake-word training set.

Positives are synthesised with several TTS voices (edge-tts) and then
augmented; negatives come from three sources:

  1. ordinary command utterances from the OptionB dataset (they never contain
     the wake word and make excellent hard negatives),
  2. TTS "near miss" phrases ("maybe", "hey there", "hey google", ...),
  3. synthetic noise / room-tone / silence windows.

    python gen_wake_data.py --dataset $DATA --out $FEAT

Outputs (inside --out):  wake_X.npy, wake_y.npy, wake_split.npy, wake_meta.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F                       # noqa: E402
from rapi_vcm.augment import (mix_at_snr, place_clip, random_gain,   # noqa: E402
                              random_noise, random_window)
import soundfile as sf                                   # noqa: E402

WAKE_TEXTS = ["Hey Rapi", "Hey Rapi!", "Hey, Rapi"]

HARD_NEGATIVES = [
    "maybe", "hey there", "hi", "hello", "hey google", "ok google",
    "hey siri", "hey buddy", "her rapid", "a rapid", "rapido", "gravel",
    "radio", "harry", "happy", "rainy", "paper", "library", "pirate",
    "api", "rpm", "hurray", "harpy", "car pie", "they wrapped it",
    "pay attention", "play music", "what time is it",
    # near-wake names + observed cross-corpus false accepts (improving_accuracy #1)
    "hey ravi", "hey rappy", "hey robi", "hey robby", "hey ruby",
    "hey rap", "hey rapy", "hey rabi", "hey ray", "hey alexa",
    "hey cortana", "okay google", "hey ricky", "hey ralph",
    "volume up", "call anna", "remind me to call mom",
    "where is the nearest coffee",
]

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


# --------------------------------------------------------------------- TTS ---
def _ffmpeg_to_wav(src: str, dst: str) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src,
                    "-ar", str(F.SAMPLE_RATE), "-ac", "1",
                    "-acodec", "pcm_s16le", dst],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


async def _edge_synth(text, voice, rate, pitch, out_wav, sem) -> bool:
    import edge_tts
    tmp = out_wav + ".mp3"
    async with sem:
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await comm.save(tmp)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                    _ffmpeg_to_wav(tmp, out_wav)
                    os.remove(tmp)
                    return True
            except Exception as exc:
                if attempt == 2:
                    print(f"[tts] {voice} {text!r}: {exc}", file=sys.stderr)
                await asyncio.sleep(1.0 + attempt)
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


async def synth_batch(items, concurrency=6):
    """items = [(key, text, voice, rate, pitch, path), ...] -> {key: bool}"""
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        key, text, voice, rate, pitch, path = it
        if os.path.exists(path) and os.path.getsize(path) > 500:
            return key, True
        return key, await _edge_synth(text, voice, rate, pitch, path, sem)

    return dict(await asyncio.gather(*[one(it) for it in items]))


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def list_voices(limit: int, seed: int):
    import edge_tts

    async def _l():
        return await edge_tts.list_voices()

    voices = _run_async(_l())
    pool = sorted({v["ShortName"] for v in voices
                   if v["ShortName"].startswith(("en-", "fil-"))})
    random.Random(seed).shuffle(pool)
    return pool[:limit]


# ------------------------------------------------------------------ audio ----
def trim_silence(x: np.ndarray, sr: int = F.SAMPLE_RATE,
                 thresh_db: float = -42.0, pad: float = 0.06) -> np.ndarray:
    if x.size == 0:
        return x
    win = max(int(0.02 * sr), 1)
    n = x.shape[0] // win
    if n == 0:
        return x
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    active = np.where(db > thresh_db)[0]
    if active.size == 0:
        return x.astype(np.float32)
    lo = max(int(active[0] * win - pad * sr), 0)
    hi = min(int((active[-1] + 1) * win + pad * sr), x.shape[0])
    return x[lo:hi].astype(np.float32)


def speed_change(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample-based speed change (also shifts pitch: intentional here)."""
    from scipy.signal import resample_poly
    if abs(factor - 1.0) < 1e-3 or x.size < F.SAMPLE_RATE // 4:
        return x
    g = int(round(factor * 1000))
    return resample_poly(x, g, 1000).astype(np.float32)


def window_to_feature(win: np.ndarray) -> np.ndarray:
    return F.pad_crop(F.cmvn(F.logmel(win)), F.WAKE_FRAMES)


# ------------------------------------------------------------------ helpers --
def read_manifest(dataset: str):
    rows = []
    with open(os.path.join(dataset, "manifest.csv"), newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = os.path.join(dataset, r["path"])
            if os.path.exists(p):
                rows.append((p, r))
    return rows


def build_tts_cache(base_dir, voices, texts, rates, pitches, seed):
    os.makedirs(base_dir, exist_ok=True)
    plan, items = [], []
    for vi, voice in enumerate(voices):
        for ti, text in enumerate(texts):
            for ri, rate in enumerate(rates):
                for pi, pitch in enumerate(pitches):
                    key = f"v{vi}_{ti}_{ri}_{pi}"
                    path = os.path.join(base_dir, key + ".wav")
                    items.append((key, text, voice, rate, pitch, path))
                    plan.append({"key": key, "voice": voice, "text": text,
                                 "path": path})
    print(f"[tts] synthesising {len(items)} base clips "
          f"({len(voices)} voices x {len(texts)} texts x {len(rates)} rates "
          f"x {len(pitches)} pitches)", flush=True)
    res = _run_async(synth_batch(items))
    ok = [p for p in plan if res.get(p["key"])]
    print(f"[tts] {len(ok)}/{len(items)} base clips ok", flush=True)
    return ok


def voice_split(voices, seed, ratios=(0.70, 0.15, 0.15)):
    shuffled = sorted(set(voices))
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n1, n2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return {v: (0 if i < n1 else 1 if i < n2 else 2)
            for i, v in enumerate(shuffled)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="OptionB directory")
    ap.add_argument("--out", required=True, help="feature output directory")
    ap.add_argument("--voices", type=int, default=80)
    ap.add_argument("--pos-target", type=int, default=8000)
    ap.add_argument("--neg-optionb", type=int, default=10000)
    ap.add_argument("--neg-noise", type=int, default=2500)
    ap.add_argument("--neg-tts", type=int, default=4000)
    ap.add_argument("--real", default=None,
                    help="directory of REAL 'Hey Rapi' recordings "
                         "(record_wake.py output; augmented per clip)")
    ap.add_argument("--real-per-clip", type=int, default=40,
                    help="augmented variants per real recording "
                         "(variant 0 is the clean placement)")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    samples = os.path.join(args.out, "wake_samples")
    os.makedirs(samples, exist_ok=True)
    WLEN = int(F.WAKE_SECONDS * F.SAMPLE_RATE)

    X, y, s = [], [], []

    def add(feats, label, split, raw=None, name=None):
        X.append(feats.astype(np.float16))
        y.append(label)
        s.append(split)
        if raw is not None and name:
            sf.write(os.path.join(samples, name), raw, F.SAMPLE_RATE,
                     subtype="PCM_16")

    # ---------------------------------------------------------------- TTS ----
    voices = list_voices(args.voices, args.seed)
    print(f"[tts] {len(voices)} candidate voices", flush=True)
    base = build_tts_cache(os.path.join(args.out, "wake_tts"), voices,
                           WAKE_TEXTS, ["-15%", "+0%", "+15%"],
                           ["-25Hz", "+0Hz", "+25Hz"], args.seed)
    if len(base) < 100:
        raise SystemExit("too few TTS clips produced - aborting")

    neg_dir = os.path.join(args.out, "wake_tts_neg")
    os.makedirs(neg_dir, exist_ok=True)
    neg_voices = voices[:max(len(voices) // 2, 8)]
    # each phrase across several voices/rates/pitches: one voice per phrase
    # let the model memorise the timbre instead of the /p/-/v/ distinction
    neg_items = []
    for i, text in enumerate(HARD_NEGATIVES):
        for j, v in enumerate(neg_voices[:6]):
            rate = ["-15%", "+0%", "+15%"][j % 3]
            pitch = ["-25Hz", "+0Hz", "+25Hz"][j % 3]
            neg_items.append((f"hn_{i}_{j}", text, v, rate, pitch,
                              os.path.join(neg_dir, f"hn_{i}_{j}.wav")))
    neg_res = _run_async(synth_batch(neg_items))
    neg_clips = [it for it in neg_items if neg_res.get(it[0])]
    print(f"[tts] {len(neg_clips)}/{len(neg_items)} hard negatives ok",
          flush=True)

    # -------------------------------------------------------------- splits ----
    vsplit = voice_split([p["voice"] for p in base], args.seed)
    manifest = read_manifest(args.dataset)

    # ---------------------------------------------------- positive samples ----
    per_clip = max(args.pos_target // max(len(base), 1), 1)
    print(f"[build] {len(base)} base clips x {per_clip} variants", flush=True)
    kept = 0
    for b in base:
        try:
            x = trim_silence(F.load_audio(b["path"]))
        except Exception:
            continue
        if x.shape[0] < int(0.15 * F.SAMPLE_RATE):
            continue
        split = vsplit.get(b["voice"], 0)
        for k in range(per_clip):
            lead = int(float(rng.uniform(0.0, 0.35)) * F.SAMPLE_RATE)
            win = place_clip(x, WLEN, rng, max_lead=lead)
            if rng.random() < 0.6:
                win = mix_at_snr(win, random_noise(WLEN, rng),
                                 float(rng.uniform(5.0, 30.0)))
            if rng.random() < 0.4:
                win = random_gain(win, rng, -14.0, 6.0)
            if rng.random() < 0.25:
                win = place_clip(speed_change(x, float(rng.uniform(0.9, 1.12))),
                                 WLEN, rng, max_lead=lead)
            kept += 1
            add(window_to_feature(win), 1, split,
                raw=(win if kept <= 16 else None),
                name=(f"pos_{kept:03d}.wav" if kept <= 16 else None))

    # ------------------------------------------- positives with command tail --
    tail_n = max(args.pos_target // 4, 1)
    for i in range(tail_n):
        b = pyrng.choice(base)
        try:
            wake = trim_silence(F.load_audio(b["path"]))
            tail = F.load_audio(pyrng.choice(manifest)[0])
        except Exception:
            continue
        seg = np.zeros(WLEN, dtype=np.float32)
        lead = int(rng.uniform(0.0, 0.25) * F.SAMPLE_RATE)
        k = min(wake.shape[0], WLEN - lead)
        seg[lead:lead + k] = wake[:k]
        if tail.shape[0]:
            start = lead + k + int(rng.uniform(0.0, 0.15) * F.SAMPLE_RATE)
            if start < WLEN:
                t = min(tail.shape[0], WLEN - start)
                seg[start:start + t] = tail[:t] * 0.9
        if rng.random() < 0.6:
            seg = mix_at_snr(seg, random_noise(WLEN, rng),
                             float(rng.uniform(8, 30)))
        add(window_to_feature(seg), 1, vsplit.get(b["voice"], 0),
            raw=(seg if i < 8 else None),
            name=(f"pos_tail_{i:03d}.wav" if i < 8 else None))

    # --------------------------------------- positives: REAL recordings ------
    # From record_wake.py (real human "Hey Rapi"). Split hygiene: the 70/15/15
    # assignment happens on the ORIGINAL clip, so every augmented variant of a
    # recording stays in its own split (no leakage into val/test).
    real_clips, real_variants = [], 0
    rsplit = {}
    if args.real and os.path.isdir(args.real):
        real_clips = sorted(f for f in os.listdir(args.real)
                            if f.lower().endswith((".wav", ".flac")))
        random.Random(args.seed + 7).shuffle(real_clips)   # deterministic
        n = len(real_clips)
        n1 = max(int(n * 0.70), 1) if n else 0
        n2 = max(n1 + max(int(n * 0.15), 1), n1 + 1) if n else 0
        rsplit = {c: (0 if i < n1 else 1 if i < min(n2, n) else 2)
                  for i, c in enumerate(real_clips)}
        print(f"[real] {len(real_clips)} recordings "
              f"-> train={sum(v == 0 for v in rsplit.values())} "
              f"val={sum(v == 1 for v in rsplit.values())} "
              f"test={sum(v == 2 for v in rsplit.values())} "
              f"x {args.real_per_clip} variants", flush=True)
        for c in real_clips:
            try:
                x = trim_silence(F.load_audio(os.path.join(args.real, c)))
            except Exception:
                continue
            if x.shape[0] < int(0.10 * F.SAMPLE_RATE):
                continue
            split = rsplit[c]
            for k in range(args.real_per_clip):
                lead = int(float(rng.uniform(0.0, 0.35)) * F.SAMPLE_RATE)
                win = place_clip(x, WLEN, rng, max_lead=lead)
                if k > 0:      # variant 0 = clean placement of the real take
                    if rng.random() < 0.7:
                        win = mix_at_snr(win, random_noise(WLEN, rng),
                                         float(rng.uniform(5.0, 30.0)))
                    if rng.random() < 0.5:
                        win = random_gain(win, rng, -14.0, 6.0)
                    if rng.random() < 0.3:
                        win = place_clip(
                            speed_change(x, float(rng.uniform(0.9, 1.12))),
                            WLEN, rng, max_lead=lead)
                real_variants += 1
                add(window_to_feature(win), 1, split,
                    raw=(win if real_variants <= 12 else None),
                    name=(f"real_pos_{real_variants:03d}.wav"
                          if real_variants <= 12 else None))
        # a few "Hey Rapi ... command tail" examples from the REAL voice too
        for i in range(max(len(real_clips) * 5, 1)):
            c = pyrng.choice(real_clips)
            try:
                wake = trim_silence(F.load_audio(os.path.join(args.real, c)))
                tail = F.load_audio(pyrng.choice(manifest)[0])
            except Exception:
                continue
            seg = np.zeros(WLEN, dtype=np.float32)
            lead = int(rng.uniform(0.0, 0.25) * F.SAMPLE_RATE)
            k = min(wake.shape[0], WLEN - lead)
            seg[lead:lead + k] = wake[:k]
            if tail.shape[0]:
                st = lead + k + int(rng.uniform(0.0, 0.15) * F.SAMPLE_RATE)
                if st < WLEN:
                    t = min(tail.shape[0], WLEN - st)
                    seg[st:st + t] = tail[:t] * 0.9
            if rng.random() < 0.6:
                seg = mix_at_snr(seg, random_noise(WLEN, rng),
                                 float(rng.uniform(8, 30)))
            real_variants += 1
            add(window_to_feature(seg), 1, rsplit[c],
                raw=(seg if i < 6 else None),
                name=(f"real_tail_{i:03d}.wav" if i < 6 else None))

    # --------------------------------------------------- negative: OptionB ----
    order = list(range(len(manifest)))
    pyrng.shuffle(order)
    n_opt = 0
    for idx in order[:args.neg_optionb]:
        path, row = manifest[idx]
        try:
            wave = F.load_audio(path)
        except Exception:
            continue
        split = {"train": 0, "val": 1, "valid": 1, "test": 2}.get(
            (row.get("split") or "train").lower(), 0)
        if split == 0:
            win = random_window(wave, WLEN, rng)
        elif wave.shape[0] > WLEN:
            st = (wave.shape[0] - WLEN) // 2
            win = wave[st:st + WLEN].astype(np.float32)
        else:
            win = random_window(wave, WLEN, np.random.default_rng(idx))
        if rng.random() < 0.35:
            win = mix_at_snr(win, random_noise(WLEN, rng),
                             float(rng.uniform(0, 20)))
        n_opt += 1
        add(window_to_feature(win), 0, split,
            raw=(win if n_opt <= 8 else None),
            name=(f"neg_cmd_{n_opt:03d}.wav" if n_opt <= 8 else None))

    # ------------------------------------------------- negative: TTS near miss -
    per_neg = max(args.neg_tts // max(len(neg_clips), 1), 1)
    for i, (key, text, voice, rate, pitch, path) in enumerate(neg_clips):
        if not os.path.exists(path):
            continue
        try:
            x = trim_silence(F.load_audio(path))
        except Exception:
            continue
        split = vsplit.get(voice, 0)
        for k in range(per_neg):
            win = place_clip(x, WLEN, rng, max_lead=int(0.4 * F.SAMPLE_RATE))
            if rng.random() < 0.5:
                win = mix_at_snr(win, random_noise(WLEN, rng),
                                 float(rng.uniform(5, 30)))
            add(window_to_feature(win), 0, split,
                raw=(win if (i < 8 and k == 0) else None),
                name=(f"neg_tts_{i:03d}.wav" if (i < 8 and k == 0) else None))

    # -------------------------------------------------- negative: noise -------
    for i in range(args.neg_noise):
        win = random_noise(WLEN, rng)
        win = (win / (np.abs(win).max() + 1e-9)) * float(rng.uniform(0.02, 0.5))
        if i % 3 == 0:
            win = win * float(rng.uniform(0.001, 0.02))     # near-silence
        win = win.astype(np.float32)
        add(window_to_feature(win), 0, int(rng.integers(0, 3)),
            raw=(win if i < 5 else None),
            name=(f"neg_noise_{i:03d}.wav" if i < 5 else None))

    # ------------------------------------------------------------- save -------
    Xa = np.stack(X)
    ya = np.array(y, dtype=np.int64)
    sa = np.array(s, dtype=np.int64)
    np.save(os.path.join(args.out, "wake_X.npy"), Xa)
    np.save(os.path.join(args.out, "wake_y.npy"), ya)
    np.save(os.path.join(args.out, "wake_split.npy"), sa)

    meta = {
        "n": int(Xa.shape[0]),
        "shape": list(Xa.shape),
        "seconds": F.WAKE_SECONDS,
        "frames": F.WAKE_FRAMES,
        "wake_texts": WAKE_TEXTS,
        "n_voices": len(voices),
        "base_tts_clips": len(base),
        "hard_negative_clips": len(neg_clips),
        "real_recordings": len(real_clips),
        "real_variants": int(real_variants),
        "real_per_split": {str(k): int(v) for k, v in
                           zip(*np.unique(
                               [rsplit[c] for c in real_clips
                                if c in rsplit] or [0],
                               return_counts=True))},
        "counts": {"positive": int((ya == 1).sum()),
                   "negative": int((ya == 0).sum())},
        "splits": {str(k): int(v) for k, v in zip(*np.unique(sa, return_counts=True))},
        "pos_per_split": {str(k): int(v) for k, v in
                          zip(*np.unique(sa[ya == 1], return_counts=True))},
        "neg_per_split": {str(k): int(v) for k, v in
                          zip(*np.unique(sa[ya == 0], return_counts=True))},
        "voice_split_size": {str(k): int(v) for k, v in
                             zip(*np.unique(list(vsplit.values()), return_counts=True))},
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "wake_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps({k: meta[k] for k in
                      ("n", "counts", "splits", "pos_per_split",
                       "neg_per_split")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
