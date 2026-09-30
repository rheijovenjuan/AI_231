#!/usr/bin/env python
"""Synthetic "Hey Rapi" wake-word data generation + training + evaluation.

The spoken-command dataset does NOT contain the wake word "Hey Rapi", so this
script manufactures a dedicated wake-word corpus SYNTHETICALLY:

  POSITIVES  : neural TTS (edge-tts) says "Hey Rapi" in many voices, rates and
               pitches - this is the default and the only mode whose audio
               actually SOUNDS like "Hey Rapi".  Cached under data/wake_tts/.
  FALLBACK   : a deterministic formant / PSOLA-style synthesiser (offline, no
               network, runs on a Pi).  Its output is a rough buzz-like
               approximation of the phrase, NOT intelligible speech - use it
               only when edge-tts is unavailable (--tts offline).
  NEGATIVES  : the same TTS (or the offline synth) renders look-alike /
               unrelated phrases (near-collisions such as "Hey Ray",
               "Hey Rapid", "Hey Siri", plus everyday commands) AND
               non-speech audio (silence + room tone + noise).

Every clip is then heavily AUGMENTED (pitch, speed, gain, reverb, white noise,
rumble, random pauses) so a small GMM generalises across distortions.

Pipeline (CPU only, Pi-friendly):
    TTS/synth WAVs -> MFCC features (voice_assistant.features) ->
    GMM(pos) vs GMM(neg)  ->  models/wakeword_synthetic.joblib
    evaluation on a HELD-OUT split (by clip) -> reports/wake_synthetic_report.*

Usage:
    python make_wake_data.py            # edge-tts corpus + augment + train + eval
    python make_wake_data.py --tts edge # force neural TTS (network needed)
    python make_wake_data.py --tts offline  # force the formant fallback
    python make_wake_data.py --dry-run  # print the plan, write nothing
    python make_wake_data.py --pos-per-voice 50 --aug 8

Listen to the synthesised audio:
    data/wake_tts/*.wav          raw neural TTS clips ("Hey Rapi" + negatives)
    data/synth_wake/pos/*.wav    trimmed positives used for training
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from voice_assistant import features as F  # noqa: E402

SR = F.SAMPLE_RATE          # 16000
SYNTH_DIR = os.path.join(HERE, "data", "synth_wake")
REPORTS_DIR = os.path.join(HERE, "reports")
MODEL_OUT = os.path.join(HERE, "models", "wakeword_synthetic.joblib")
CACHE_MANIFEST = os.path.join(SYNTH_DIR, "synth_manifest.json")
TTS_DIR = os.path.join(HERE, "data", "wake_tts")

WAKE_TEXT = "Hey Rapi"
# Positives: the wake phrase alone AND with a continuation ("Hey Rapi, stop")
# - a wake word must fire on those too.  Keeping them out of the NEGATIVE set
# matters: a bag-of-MFCC model cannot reject an utterance that literally
# contains the wake phrase, so listing them as negatives just tanks precision.
WAKE_TEXTS = (WAKE_TEXT, "Hey Rapi!", "Hey, Rapi", "Hey Rapi please",
              "Hey Rapi now", "Hey Rapi stop", "Hey Rapi okay",
              "Hey Rapi's friend", "Hey Rapi Rapi", "Hey Ra Pi",
              "Hey Rap I")

# Neural voices for edge-tts (network synthesis).  Rate + pitch variants add
# diversity across positives so the GMM does not overfit a handful of timbres.
EDGE_VOICES = (
    "en-US-ChristopherNeural", "en-US-AriaNeural", "en-US-GuyNeural",
    "en-US-JennyNeural", "en-US-MichelleNeural", "en-US-RogerNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural",
    "en-IN-PrabhatNeural",
)
EDGE_RATES = ("+0%", "-12%", "+12%")
EDGE_PITCHES = ("+0Hz", "-20Hz", "+20Hz")

# Negative phrases: chosen to share phonemes with "Hey Rapi" (h-, r-, -ay,
# -ee, -ah) so the model learns fine-grained contrasts, plus clearly-unrelated
# everyday speech that must be ignored.  Utterances that CONTAIN the wake
# word are positives (see WAKE_TEXTS), never negatives.
NEGATIVE_PHRASES = [
    # near-collisions (same letters / sounds, wrong word)
    "Hey Ray", "Hey Rapid", "Hey Rapids", "Hey Rappy", "Hey Rapy",
    "Hey Rap", "Hey Ray please", "Hey Ray stop",
    # other wake-style / assistant phrases (should be ignored)
    "Hey Siri", "Hey Alexa", "Okay Google", "Hey Cortana", "Hey Ravi",
    "Hey Robi", "Hey Rabi",
    # everyday speech that is clearly NOT the wake word
    "What time is it", "Play some music", "Turn the lights on",
    "What is the weather today", "Remind me to call mom",
    "Set a timer for ten minutes", "Call Anna", "Dim the lights",
    "Change the color to blue", "Adjust the temperature",
    "Pause the song", "Stop the music", "Next song", "Volume up",
    "Thank you very much", "I love you", "How are you doing",
    "Where is the nearest coffee shop", "Tell me a joke",
    "Read me the news please", "What day is it today",
    "Turn off the fan", "Open the blinds", "Lock the front door",
]


# --------------------------------------------------------------------------- #
# Deterministic offline "TTS" (formant / PSOLA-style)
# --------------------------------------------------------------------------- #
def _phoneme(text: str):
    """Very small English phonemeiser -> list of (kind, freqs, dur).

    kind: 'v' voiced vowel, 's' sonorant consonant (nasal/liquid/fricative),
          'c' stop consonant (burst), 'p' pause.
    freqs: for vowels -> formants [F1,F2]; for sonorants -> [F0-ish, F2].
    """
    t = " " + text.lower().replace("'", "") + " "
    t = t.replace("-", " ").replace(",", " ")
    words = [w for w in t.split() if w]
    out = []
    for w in words:
        out.extend(_word_phonemes(w))
        out.append(("p", None, 0.06))  # inter-word gap
    return out


_VOWEL_FORMANTS = {
    "a": (800, 1200), "e": (500, 1800), "i": (300, 2200),
    "o": (450, 900), "u": (320, 700), "ay": (800, 1700),
    "ee": (300, 2200), "oh": (450, 1000), "uh": (500, 1300),
}


def _word_phonemes(w: str):
    """Greedy phoneme decomposition for a lowercase word."""
    seq = []
    i = 0
    n = len(w)

    def vowel_after(j):
        return j < n and w[j] in "aeiou"

    while i < n:
        ch = w[i]
        if ch == " ":
            i += 1
            continue
        # two-letter diphthongs
        if w[i:i + 2] in ("ay", "ea", "ee", "oo", "ou", "ow", "ai", "ei", "oa"):
            dig = w[i:i + 2]
            fm = {"ay": "ay", "ea": "ee", "ee": "ee", "oo": "u",
                  "ou": "u", "ow": "oh", "ai": "ay", "ei": "ay",
                  "oa": "oh"}[dig]
            seq.append(("v", _VOWEL_FORMANTS[fm], 0.16))
            i += 2
            continue
        if ch in "aeiou":
            fm = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u"}[ch]
            # lengthen if doubled or before a consonant cluster
            dur = 0.14
            if i + 1 < n and w[i + 1] == ch:
                dur = 0.22
            seq.append(("v", _VOWEL_FORMANTS[fm], dur))
            i += 1
            continue
        # consonants
        if ch in "hlrn":
            seq.append(("s", (1500, 2500), 0.09))
            i += 1
        elif ch in "mw":
            seq.append(("s", (900, 1400), 0.08))
            i += 1
        elif ch in "sf":
            seq.append(("s", (4000, 6000), 0.10))
            i += 1
        elif ch in "tksbdgp":
            seq.append(("c", (2500, 4000), 0.05))
            i += 1
        elif ch in "y":
            seq.append(("s", (1800, 2600), 0.07))
            i += 1
        elif ch in "zh":
            seq.append(("s", (3000, 4500), 0.10))
            i += 1
        else:
            # unknown letter -> short sonorant so we never drop a sound
            seq.append(("s", (1600, 2400), 0.06))
            i += 1
    return seq


def _voice_profile(rng: random.Random, idx: int):
    """A distinct synthetic voice: F0 range, formant scale, breathiness."""
    base_f0 = 110 + 40 * (idx % 5) + rng.uniform(-12, 12)   # 98..~210 Hz
    f0_var = rng.uniform(0.06, 0.14)                        # declination depth
    form_scale = 1.0 + rng.uniform(-0.12, 0.12)             # vocal-tract size
    breath = rng.uniform(0.02, 0.12)                        # noise mix
    rate = rng.uniform(0.92, 1.08)                          # speaking rate
    return dict(f0=base_f0, f0_var=f0_var, form=form_scale,
                breath=breath, rate=rate)


def synth_phrase(text: str, voice: dict, rng: random.Random,
                 stress_first: bool = True) -> np.ndarray:
    """Render `text` to a mono float32 @ SR waveform using the voice profile."""
    phon = _phoneme(text)
    # total duration to normalise prosody
    total = sum(p[2] for p in phon) + 0.08
    # build F0 contour over voiced frames (declination + slight rise)
    n_total = int(total * SR)
    t = np.arange(n_total) / SR
    f0_contour = voice["f0"] * (1.0 - voice["f0_var"] * (t / max(1e-6, total)))
    f0_contour += 8 * np.sin(2 * np.pi * 0.5 * t)  # gentle intonation

    # cumulative time cursor
    cursor = 0.0
    out = np.zeros(n_total, dtype=np.float32)
    for (kind, freqs, dur) in phon:
        dur = dur / voice["rate"]
        start = int(cursor * SR)
        length = int(dur * SR)
        if start >= n_total:
            break
        length = min(length, n_total - start)
        if length <= 0:
            cursor += dur
            continue
        if kind == "p":
            cursor += dur  # inter-word gap: no acoustic energy
            continue
        seg_t = np.arange(length) / SR
        seg_f0 = f0_contour[start:start + length]
        if kind == "v":
            f1, f2 = freqs[0] * voice["form"], freqs[1] * voice["form"]
            seg = _voiced_vowel(seg_t, seg_f0, f1, f2, voice["breath"], rng)
        elif kind == "s":
            f1, f2 = freqs[0], freqs[1]
            seg = _sonorant(seg_t, seg_f0, f1, f2, voice["breath"], rng)
        else:  # stop burst
            seg = _burst(seg_t, freqs, rng)
        # amplitude envelope (attack/release)
        env = _amphann(length, 0.02, 0.03)
        out[start:start + length] += seg * env
        cursor += dur

    # overall normalisation
    peak = float(np.max(np.abs(out))) + 1e-9
    out = out / peak * 0.85
    return out


def _voiced_vowel(t, f0, f1, f2, breath, rng):
    """Sum of harmonics + two formant bands + a little breath noise."""
    n = len(t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros(n, dtype=np.float32)
    for h in range(1, 9):
        sig += (1.0 / h) * np.sin(phase * h)
    # formant emphasis via simple resonators (two-pole approx)
    sig = _resonate(sig, f1, 1.0) * 1.0 + _resonate(sig, f2, 0.6)
    # breath
    nz = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float32)
    sig = sig + nz * breath
    return sig


def _sonorant(t, f0, f1, f2, breath, rng):
    n = len(t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros(n, dtype=np.float32)
    for h in range(1, 5):
        sig += (1.0 / h) * np.sin(phase * h)
    sig = _resonate(sig, f1, 0.8) + _resonate(sig, f2, 0.4)
    nz = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float32)
    sig = sig + nz * (breath + 0.05)
    return sig


def _burst(t, freqs, rng):
    n = len(t)
    fc = freqs[0]
    nz = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float32)
    # band-pass around fc
    sig = _resonate(nz, fc, 1.0)
    env = np.exp(-np.arange(n) / max(1, n * 0.4))
    return sig * env


def _resonate(x, f, gain):
    """Cheap single-pole resonator emphasising frequency f (Hz)."""
    if f <= 0:
        return x * gain
    rc = 1.0 / (2 * np.pi * f)
    dt = 1.0 / SR
    a = dt / (rc + dt)
    y = np.zeros_like(x, dtype=np.float32)
    yp = 0.0
    for i in range(len(x)):
        yp = yp + a * (x[i] - yp)
        y[i] = yp
    # subtract low-frequency component so it acts as a band emphasis
    y = y - np.convolve(y, np.ones(8) / 8, mode="same")
    return y * gain


def _amphann(n, attack, release):
    env = np.ones(n, dtype=np.float32)
    a = int(attack * SR); r = int(release * SR)
    if a > 0:
        env[:a] *= np.linspace(0, 1, a)
    if r > 0:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def _trim_edges(x: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    if len(x) == 0:
        return x
    idx = np.where(np.abs(x) > thresh)[0]
    if idx.size == 0:
        return x
    pad = int(0.03 * SR)
    a = max(0, idx[0] - pad)
    b = min(len(x), idx[-1] + pad)
    return x[a:b]


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return x[:0]
    xp = np.linspace(0.0, 1.0, num=len(x))
    xn = np.linspace(0.0, 1.0, num=n_out)
    return np.interp(xn, xp, x).astype(np.float32)


def augment(x: np.ndarray, rng: random.Random) -> np.ndarray:
    y = x.copy()
    orig_len = len(x)

    # 1) pitch shift
    if rng.random() < 0.85:
        f = rng.uniform(0.90, 1.10)
        y = _resample_linear(y, SR, int(SR / f))
        y = _refit_len(y, orig_len)
    # 2) time stretch
    if rng.random() < 0.7:
        f = rng.uniform(0.92, 1.08)
        y = _resample_linear(y, SR, int(SR / f))
        y = _refit_len(y, orig_len)
    # 3) gain
    y = y * rng.uniform(0.6, 1.6)
    # 4) simple reverb
    if rng.random() < 0.6:
        decay = rng.uniform(0.2, 0.5)
        rt = int(SR * rng.uniform(0.02, 0.06))
        rt = min(rt, max(3, len(y) // 2))  # keep taps within the signal
        if 2 < rt:
            rev = np.zeros_like(y)
            for tap in range(1, rt):  # tap=0 would make y[:-0] empty
                rev[tap:] += y[:-tap] * (decay ** (tap / rt))
            y = y + rev * 0.5
    # 5) white noise
    if rng.random() < 0.7:
        snr_db = rng.uniform(12, 30)
        sig_p = float(np.mean(y ** 2)) + 1e-9
        noise_p = sig_p / (10 ** (snr_db / 10))
        wn = np.array([rng.gauss(0, 1) for _ in range(len(y))], dtype=np.float32)
        wn *= (np.sqrt(noise_p) / (np.std(wn) + 1e-9))
        y = y + wn
    # 6) rumble (brown noise)
    if rng.random() < 0.4:
        n = len(y)
        brown = np.cumsum(np.array([rng.gauss(0, 1) for _ in range(n)],
                                   dtype=np.float32))
        brown -= brown.mean()
        brown /= (np.std(brown) + 1e-9)
        y = y + brown * rng.uniform(0.02, 0.08)
    # 7) random leading/trailing silence
    if rng.random() < 0.5:
        lead = int(rng.uniform(0, 0.12) * SR)
        y = np.concatenate([np.zeros(lead, dtype=np.float32), y])
    if rng.random() < 0.5:
        trail = int(rng.uniform(0, 0.10) * SR)
        y = np.concatenate([y, np.zeros(trail, dtype=np.float32)])

    peak = float(np.max(np.abs(y))) + 1e-9
    y = y / peak * rng.uniform(0.5, 0.9)
    return y.astype(np.float32)


def _refit_len(y, target):
    if len(y) > target:
        return y[:target]
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y


def make_silence(rng):
    n = int(rng.uniform(0.25, 0.9) * SR)
    x = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float32)
    return x * rng.uniform(0.0005, 0.004)


def make_room_tone(rng):
    n = int(rng.uniform(0.3, 1.0) * SR)
    f = rng.uniform(50, 180)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * t) * 0.3 +
         np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float32) * 0.2)
    return (x * rng.uniform(0.002, 0.01)).astype(np.float32)


def _save_wav(path, x, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm16 = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


# --------------------------------------------------------------------------- #
# Neural TTS (edge-tts) - the audio source that actually SOUNDS like speech
# --------------------------------------------------------------------------- #
def resolve_tts_mode(choice: str) -> str:
    """auto -> 'edge' when edge-tts is importable, else 'offline'."""
    if choice != "auto":
        return choice
    try:
        import edge_tts  # noqa: F401
        return "edge"
    except Exception:
        return "offline"


def _tts_key(text: str, voice: str, rate: str, pitch: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_"
                   for ch in text.lower()).strip("_")[:24] or "clip"
    vslug = voice.replace("Neural", "").replace("-", "")[:16]
    h = hashlib.sha1(f"{text}|{voice}|{rate}|{pitch}".encode()).hexdigest()[:8]
    return f"{slug}_{vslug}_{h}"


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def _decode_audio(src: str):
    """Any audio file -> (float32 mono, sr).  soundfile first, then ffmpeg."""
    try:
        import soundfile as sf
        d, sr = sf.read(src, dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        return d, int(sr)
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    if not ff:
        return None, None
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "decoded.wav")
        rc = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", src, "-ac", "1",
             "-acodec", "pcm_s16le", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc != 0 or not os.path.exists(out):
            return None, None
        import soundfile as sf
        d, sr = sf.read(out, dtype="float32")
        return (d.mean(axis=1) if d.ndim > 1 else d), int(sr)


async def _edge_one(sem, key, text, voice, rate, pitch, path, results):
    if os.path.exists(path) and os.path.getsize(path) > 400:
        results[key] = True
        return
    import edge_tts
    tmp = path + ".mp3"
    async with sem:
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, voice, rate=rate,
                                           pitch=pitch).save(tmp)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 400:
                    d, sr = _decode_audio(tmp)
                    if d is not None and len(d) > 1600:
                        _save_wav(path, _resample_to_16k(d, sr))
                        results[key] = True
                break
            except Exception:
                await asyncio.sleep(1.0 + attempt)
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass


def tts_batch(items) -> dict:
    """Synthesise [(key, text, voice, rate, pitch, path), ...] -> {key: ok}.

    Cached files count as ok, so repeats are free.
    """
    if not items:
        return {}
    results: dict = {}
    sem = asyncio.Semaphore(4)

    async def run():
        await asyncio.gather(*[_edge_one(sem, *it, results) for it in items])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
    return results


def tts_plan(n: int, texts, prefix: str):
    """Deterministic (text, voice, rate, pitch) plan of length n."""
    items = []
    nt = len(EDGE_VOICES) * len(texts)
    for k in range(n):
        text = texts[k % len(texts)]
        voice = EDGE_VOICES[(k // len(texts)) % len(EDGE_VOICES)]
        rate = EDGE_RATES[(k // nt) % len(EDGE_RATES)]
        pitch = EDGE_PITCHES[(k // (nt * len(EDGE_RATES))) % len(EDGE_PITCHES)]
        path = os.path.join(TTS_DIR, _tts_key(text, voice, rate, pitch) + ".wav")
        items.append((f"{prefix}{k:04d}", text, voice, rate, pitch, path))
    return items


def _load_tts(path: str):
    try:
        x = F.load_audio(path)
        return x.astype(np.float32) if x.size else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    label: str
    text: str
    kind: str
    path: str
    base: np.ndarray


def build_corpus(args):
    rng = random.Random(args.seed)
    clips = []
    n_pos = args.pos_per_voice * args.voices
    n_neg = args.neg_per_voice * args.voices

    voices = [_voice_profile(rng, i) for i in range(args.voices)]

    mode = resolve_tts_mode(args.tts)
    print(f"[corpus] speech source: "
          f"{'edge-tts neural TTS (real speech)' if mode == 'edge' else 'offline formant synth (NOT intelligible speech)'}")

    pos_items: list = []
    neg_items: list = []
    pos_ok: dict = {}
    neg_ok: dict = {}
    if mode == "edge":
        os.makedirs(TTS_DIR, exist_ok=True)
        print(f"[tts] building corpus with edge-tts "
              f"({n_pos} pos + {n_neg} neg, cache {TTS_DIR}) ...", flush=True)
        pos_items = tts_plan(n_pos, WAKE_TEXTS, "p")
        neg_items = tts_plan(n_neg, NEGATIVE_PHRASES, "n")
        # plan.json says which clip says what - handy for listening to them
        with open(os.path.join(TTS_DIR, "plan.json"), "w") as fh:
            json.dump([{"file": os.path.basename(path), "text": text,
                        "voice": voice, "rate": rate, "pitch": pitch}
                       for (_k, text, voice, rate, pitch, path)
                       in pos_items + neg_items], fh, indent=2)
        pos_ok = tts_batch(pos_items)
        neg_ok = tts_batch(neg_items)
        print(f"[tts] edge-tts ok: positives {sum(pos_ok.values())}/{n_pos}, "
              f"negatives {sum(neg_ok.values())}/{n_neg} "
              f"(failures fall back to the offline synth)", flush=True)

    # POSITIVES
    made = 0
    vi = 0
    while made < n_pos:
        it = pos_items[made] if mode == "edge" else None
        text = it[1] if it else WAKE_TEXT
        wav = _load_tts(it[5]) if (it and pos_ok.get(it[0])) else None
        if wav is None:
            v = voices[vi % len(voices)]
            wav = _trim_edges(synth_phrase(text, v, rng))
        else:
            wav = _trim_edges(wav)
        if len(wav) >= int(0.2 * SR):
            path = os.path.join(SYNTH_DIR, "pos", f"pos_v{vi % len(voices)}_{made}.wav")
            _save_wav(path, wav)
            clips.append(Clip("pos", text, "tts", path, wav))
            made += 1
        vi += 1
    print(f"[corpus] positives (clean): {made}")

    # NEGATIVES (speech)
    phrases = NEGATIVE_PHRASES[:]
    made = 0
    pi = 0
    vi = 0
    while made < n_neg:
        it = neg_items[made] if mode == "edge" else None
        text = it[1] if it else phrases[pi % len(phrases)]
        wav = _load_tts(it[5]) if (it and neg_ok.get(it[0])) else None
        if wav is None:
            v = voices[vi % len(voices)]
            wav = _trim_edges(synth_phrase(text, v, rng))
        else:
            wav = _trim_edges(wav)
        if len(wav) >= int(0.15 * SR):
            slug = "".join(ch for ch in text[:12] if ch.isalnum()) or "x"
            path = os.path.join(SYNTH_DIR, "neg", f"neg_{slug}_{made}.wav")
            _save_wav(path, wav)
            clips.append(Clip("neg", text, "tts", path, wav))
            made += 1
        pi += 1
        vi += 1
    print(f"[corpus] negatives (speech): {made}")

    # NEGATIVES (non-speech)
    for i in range(args.silence_clips):
        x = make_silence(rng)
        path = os.path.join(SYNTH_DIR, "neg", f"silence_{i}.wav")
        _save_wav(path, x)
        clips.append(Clip("neg", "<silence>", "silence", path, x))
    for i in range(args.silence_clips // 2):
        x = make_room_tone(rng)
        path = os.path.join(SYNTH_DIR, "neg", f"roomtone_{i}.wav")
        _save_wav(path, x)
        clips.append(Clip("neg", "<roomtone>", "roomtone", path, x))
    print(f"[corpus] silence/roomtone negatives: {args.silence_clips + args.silence_clips // 2}")
    return clips


# --------------------------------------------------------------------------- #
# Features + split
# --------------------------------------------------------------------------- #
def extract_for_clip(x):
    if len(x) < int(0.2 * SR):
        return None
    try:
        f = F.extract_features(x, SR)
    except Exception:
        return None
    return f if np.all(np.isfinite(f)) else None


def stratified_split(clips, aug, test_frac, rng):
    pos = [c for c in clips if c.label == "pos"]
    neg = [c for c in clips if c.label == "neg"]
    n_pos_test = max(1, int(len(pos) * test_frac))
    n_neg_test = max(1, int(len(neg) * test_frac))
    pos_test = set(id(c) for c in rng.sample(pos, n_pos_test))
    neg_test = set(id(c) for c in rng.sample(neg, n_neg_test))

    Xtr, ytr, Xte, yte = [], [], [], []
    for c in clips:
        in_test = (id(c) in pos_test) if c.label == "pos" else (id(c) in neg_test)
        X, y = (Xte, yte) if in_test else (Xtr, ytr)
        for v in [c.base] + [augment(c.base, rng) for _ in range(aug - 1)]:
            f = extract_for_clip(v)
            if f is not None:
                X.append(f); y.append(c.label)
    n_aug = len(Xtr) + len(Xte)
    return np.stack(Xtr), np.array(ytr), np.stack(Xte), np.array(yte), n_aug


from voice_assistant.gmm import GMM  # stable, importable GMM (picklable)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(scores, y, t):
    pred = (scores >= t).astype(int)
    tgt = (np.array(y) == "pos").astype(int)
    tp = int(((pred == 1) & (tgt == 1)).sum())
    fp = int(((pred == 1) & (tgt == 0)).sum())
    fn = int(((pred == 0) & (tgt == 1)).sum())
    tn = int(((pred == 0) & (tgt == 0)).sum())
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _best_threshold(scores, y):
    tgt = (np.array(y) == "pos").astype(int)
    order = np.argsort(-scores)
    best_t, best_f1 = 0.0, -1.0
    for i in range(1, len(order) + 1):
        t = scores[order[i - 1]]
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (tgt == 1)).sum())
        fp = int(((pred == 1) & (tgt == 0)).sum())
        fn = int(((pred == 0) & (tgt == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _roc_points(scores, y):
    tgt = (np.array(y) == "pos").astype(int)
    order = np.argsort(-np.asarray(scores))
    scores = np.asarray(scores)[order]; tgt = tgt[order]
    npos = max(1, tgt.sum()); nneg = max(1, len(tgt) - tgt.sum())
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for s, t in zip(scores, tgt):
        if t == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / npos); fpr.append(fp / nneg)
    return np.array(tpr), np.array(fpr)


def _auc(roc):
    tpr, fpr = roc
    return float(np.trapz(tpr, fpr))


def _plot_scores(scores, y, thr, roc, auc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    pos = scores[y == "pos"]; neg = scores[y == "neg"]
    ax[0].hist(neg, bins=40, alpha=0.6, label="negative", color="#d9534f")
    ax[0].hist(pos, bins=40, alpha=0.6, label="positive", color="#5cb85c")
    ax[0].axvline(thr, color="black", ls="--", lw=1.5, label=f"threshold={thr:.2f}")
    ax[0].set_title("Wake-word LLR score distribution (test)")
    ax[0].set_xlabel("LLR score"); ax[0].set_ylabel("count"); ax[0].legend()
    tpr, fpr = roc
    ax[1].plot(fpr, tpr, color="#337ab7", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].set_title("Receiver Operating Characteristic")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voices", type=int, default=6)
    ap.add_argument("--pos-per-voice", type=int, default=40)
    ap.add_argument("--neg-per-voice", type=int, default=40)
    ap.add_argument("--silence-clips", type=int, default=120)
    ap.add_argument("--aug", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--tts", choices=("auto", "edge", "offline"),
                    default="auto",
                    help="speech source: edge = neural TTS that really says "
                         "'Hey Rapi' (needs network once, then cached); "
                         "offline = formant synth fallback (not "
                         "intelligible); auto = edge when available")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tts_mode = resolve_tts_mode(args.tts)

    if args.dry_run:
        print(f"PLAN (synthetic, {tts_mode}, deterministic):")
        print(f"  voices={args.voices} pos/voice={args.pos_per_voice} "
              f"neg/voice={args.neg_per_voice} silence={args.silence_clips}")
        print(f"  clean positives={args.voices*args.pos_per_voice} "
              f"clean negatives={args.voices*args.neg_per_voice + args.silence_clips}")
        tot = (args.voices * (args.pos_per_voice + args.neg_per_voice)) * args.aug
        print(f"  augmented x{args.aug} -> ~{tot} feature vectors")
        print("  (no files written)")
        return

    t0 = time.time()
    rng = random.Random(args.seed)

    clips = build_corpus(args)
    n_pos = sum(1 for c in clips if c.label == "pos")
    n_neg = sum(1 for c in clips if c.label == "neg")
    print(f"[corpus] TOTAL clean clips: {len(clips)} (pos={n_pos}, neg={n_neg})")

    os.makedirs(SYNTH_DIR, exist_ok=True)
    man = {
        "wake_text": WAKE_TEXT, "sr": SR,
        "generator": ("edge-tts neural TTS + augmentation"
                      if tts_mode == "edge" else
                      "offline formant/PSOLA synthesiser + augmentation"),
        "negative_phrases": sorted(set(c.text for c in clips
                                       if c.label == "neg" and c.kind == "tts")),
        "n_clean": len(clips), "n_pos": n_pos, "n_neg": n_neg,
        "voices": args.voices, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(man, f, indent=2)

    print("[feats] extracting + augmenting ...")
    Xtr, ytr, Xte, yte, n_aug = stratified_split(clips, args.aug, args.test_frac, rng)
    print(f"[feats] train={Xtr.shape} test={Xte.shape} (augmented vectors={n_aug})")

    ncomp = min(args.components, max(2, int(0.05 * len(Xtr))))
    Xp = Xtr[ytr == "pos"]; Xn = Xtr[ytr == "neg"]
    print(f"[fit] pos n={len(Xp)} neg n={len(Xn)} components={ncomp}")
    pos_gm = GMM(n_components=ncomp, reg_covar=1e-4, n_init=2,
                 random_state=args.seed).fit(Xp)
    neg_gm = GMM(n_components=ncomp, reg_covar=1e-4, n_init=2,
                 random_state=args.seed).fit(Xn)

    def llr(x):
        x = np.atleast_2d(x)
        return float(pos_gm.score(x) - neg_gm.score(x))

    tr_scores = np.array([llr(x) for x in Xtr])
    te_scores = np.array([llr(x) for x in Xte])
    thr = _best_threshold(tr_scores, ytr) if args.threshold == 0.0 else args.threshold

    m_tr = _metrics(tr_scores, ytr, thr)
    m_te = _metrics(te_scores, yte, thr)
    roc = _roc_points(te_scores, yte)
    auc = _auc(roc)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    import joblib
    joblib.dump({
        "pos": pos_gm, "neg": neg_gm, "threshold": float(thr),
        "meta": {"source": ("synthetic-edge-tts" if tts_mode == "edge"
                            else "synthetic-offline"),
                 "wake_text": WAKE_TEXT,
                 "components": ncomp, "n_train": len(Xtr), "n_test": len(Xte),
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    }, MODEL_OUT)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    rep_txt = os.path.join(REPORTS_DIR, "wake_synthetic_report.txt")
    rep_json = os.path.join(REPORTS_DIR, "wake_synthetic_report.json")
    rep_png = os.path.join(REPORTS_DIR, "wake_synthetic_scores.png")

    report = {
        "wake_text": WAKE_TEXT,
        "source": ("edge-tts neural TTS + augmentation" if tts_mode == "edge"
                   else "offline formant/PSOLA TTS + augmentation"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "corpus": {"clean_total": len(clips), "pos": n_pos, "neg": n_neg,
                   "augmented_vectors": n_aug, "aug_factor": args.aug,
                   "voices": args.voices,
                   "negative_phrases": man["negative_phrases"]},
        "model": {"type": "2x GMM", "components": ncomp,
                  "feature_dim": int(F.FEATURE_DIM), "path": MODEL_OUT},
        "threshold_llr": round(float(thr), 4),
        "train": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_tr.items()},
        "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_te.items()},
        "roc_auc": round(auc, 4),
        "score_stats": {
            "pos_mean": round(float(np.mean(te_scores[yte == 'pos'])), 3),
            "pos_std": round(float(np.std(te_scores[yte == 'pos'])), 3),
            "neg_mean": round(float(np.mean(te_scores[yte == 'neg'])), 3),
            "neg_std": round(float(np.std(te_scores[yte == 'neg'])), 3),
        },
    }
    with open(rep_json, "w") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append("=" * 64)
    lines.append("  'HEY RAPI' WAKE WORD - SYNTHETIC DATA TRAINING REPORT")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"Wake word        : {WAKE_TEXT}")
    lines.append(f"Data source      : {report['source']}")
    lines.append(f"Created          : {report['created']}")
    lines.append(f"Elapsed          : {report['elapsed_sec']} s")
    lines.append("")
    lines.append("--- CORPUS ---")
    lines.append(f"Clean clips      : {len(clips)}  (pos={n_pos}, neg={n_neg})")
    lines.append(f"Aug factor       : x{args.aug}  ->  {n_aug} feature vectors")
    lines.append(f"Voices           : {args.voices} synthetic timbres")
    lines.append(f"Negative phrases : {len(man['negative_phrases'])} unique "
                 f"(near-collisions + everyday speech + silence + room tone)")
    lines.append("")
    lines.append("--- MODEL ---")
    lines.append(f"Type             : 2x GMM (positive vs negative)")
    lines.append(f"Components       : {ncomp}")
    lines.append(f"Feature dim      : {F.FEATURE_DIM} (MFCC+delta+scalar)")
    lines.append(f"Saved to         : {MODEL_OUT}")
    lines.append("")
    lines.append(f"Decision rule    : score = LLR(P(x|pos)-P(x|neg)); fire if score >= {thr:.3f}")
    lines.append("")
    lines.append("--- METRICS (held-out test, split by CLIP) ---")
    lines.append(f"{'metric':<12}{'train':>10}{'test':>10}")
    for k in ("acc", "precision", "recall", "f1"):
        lines.append(f"{k:<12}{m_tr[k]*100:>9.1f}%{m_te[k]*100:>9.1f}%")
    lines.append("")
    lines.append(f"ROC AUC (test)   : {auc*100:.1f}%")
    lines.append("")
    lines.append("--- SCORE DISTRIBUTION (test) ---")
    lines.append(f"positive  mean={report['score_stats']['pos_mean']:>8.3f}  std={report['score_stats']['pos_std']:.3f}")
    lines.append(f"negative  mean={report['score_stats']['neg_mean']:>8.3f}  std={report['score_stats']['neg_std']:.3f}")
    lines.append(f"threshold (LLR)  : {thr:.3f}")
    lines.append("")
    lines.append("--- CONFUSION (test) ---")
    lines.append(f"TP={m_te['tp']}  FP={m_te['fp']}  FN={m_te['fn']}  TN={m_te['tn']}")
    lines.append("")
    if tts_mode == "edge":
        lines.append("NOTE: positives are neural TTS 'Hey Rapi' (edge-tts) +")
        lines.append("augmentation - the WAVs genuinely say the wake phrase.")
        lines.append("Raw audio: data/wake_tts/   corpus: data/synth_wake/pos/")
    else:
        lines.append("NOTE: positives are OFFLINE FORMANT synth 'Hey Rapi'.")
        lines.append("This output does NOT sound like the phrase - install")
        lines.append("edge-tts (pip install edge-tts) and rerun for real")
        lines.append("speech audio, or record human clips instead")
        lines.append("(docs/collecting_wakeword_data.md).")
    lines.append("For production robustness, record ~200 real human 'Hey Rapi'")
    lines.append("clips (docs/collecting_wakeword_data.md) and retrain with the")
    lines.append("SAME pipeline (drop-in: identical features + GMM + threshold).")
    lines.append("")
    with open(rep_txt, "w") as f:
        f.write("\n".join(lines))

    try:
        _plot_scores(te_scores, yte, thr, roc, auc, rep_png)
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print("\n" + "\n".join(lines))
    print(f"[saved] model  -> {MODEL_OUT}")
    print(f"[saved] report -> {rep_txt}")
    print(f"[saved] json   -> {rep_json}")


if __name__ == "__main__":
    main()
