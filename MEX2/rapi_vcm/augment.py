"""Waveform + feature-space augmentation helpers.

Waveform helpers are used while synthesising the wake-word training set;
`spec_augment` is applied on-the-fly while training both models.
"""

from __future__ import annotations

import numpy as np

from .features import SAMPLE_RATE


# ------------------------------------------------------------------ noise ----
def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via the Voss-McCartney-ish IIR approximation."""
    w = rng.standard_normal(n).astype(np.float64)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    from scipy.signal import lfilter
    y = lfilter(b, a, w)
    y -= y.mean()
    y /= (y.std() + 1e-9)
    return y.astype(np.float32)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float64)
    y = np.cumsum(w)
    y -= y.mean()
    y /= (y.std() + 1e-9)
    return y.astype(np.float32)


NOISE_KINDS = (white_noise, pink_noise, brown_noise)


def random_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return NOISE_KINDS[int(rng.integers(len(NOISE_KINDS)))](n, rng)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix `noise` into `signal` at the requested SNR (both float32 mono)."""
    if noise.shape[0] < signal.shape[0]:
        reps = int(np.ceil(signal.shape[0] / max(noise.shape[0], 1)))
        noise = np.tile(noise, reps)[:signal.shape[0]]
    sig_p = float(np.mean(signal ** 2)) + 1e-12
    noi_p = float(np.mean(noise ** 2)) + 1e-12
    k = np.sqrt(sig_p / (noi_p * 10.0 ** (snr_db / 10.0)))
    out = signal + k * noise
    m = float(np.max(np.abs(out)) or 1.0)
    if m > 1.0:
        out = out / m
    return out.astype(np.float32)


def random_gain(x: np.ndarray, rng: np.random.Generator,
                lo_db: float = -12.0, hi_db: float = 6.0) -> np.ndarray:
    g = 10.0 ** (float(rng.uniform(lo_db, hi_db)) / 20.0)
    out = x * g
    m = float(np.max(np.abs(out)) or 1.0)
    if m > 1.0:
        out = out / m
    return out.astype(np.float32)


# ----------------------------------------------------------------- windows ---
def random_window(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Cut a `length` sample window; pad with zeros when the clip is short."""
    n = x.shape[0]
    if n <= length:
        out = np.zeros(length, dtype=np.float32)
        start = int(rng.integers(0, max(length - n, 1) + 1)) if n < length else 0
        out[start:start + n] = x
        return out
    start = int(rng.integers(0, n - length + 1))
    return x[start:start + length].astype(np.float32)


def place_clip(clip: np.ndarray, length: int, rng: np.random.Generator,
               max_lead: int = 0) -> np.ndarray:
    """Put `clip` at a random position inside a zero-padded window.

    `max_lead` extra samples of lead-in silence lets the detector learn that
    the wake word does not start exactly at frame 0.
    """
    out = np.zeros(length, dtype=np.float32)
    lead = int(rng.integers(0, max_lead + 1)) if max_lead > 0 else 0
    room = length - lead
    if clip.shape[0] >= room:
        seg = clip[:room]
    else:
        seg = clip
    out[lead:lead + seg.shape[0]] = seg
    return out


# -------------------------------------------------------------- SpecAugment --
def spec_augment(feats: np.ndarray, rng: np.random.Generator,
                 freq_mask: int = 8, time_mask: int = 24,
                 n_freq: int = 2, n_time: int = 2) -> np.ndarray:
    """Frequency / time masking on a (mel, frames) log-spectrogram."""
    out = feats.copy()
    f, t = out.shape
    for _ in range(n_freq):
        w = int(rng.integers(0, freq_mask + 1))
        if w == 0 or w >= f:
            continue
        s = int(rng.integers(0, f - w + 1))
        m = float(out[:, s:s + w].mean())
        out[:, s:s + w] = m
    for _ in range(n_time):
        w = int(rng.integers(0, time_mask + 1))
        if w == 0 or w >= t:
            continue
        s = int(rng.integers(0, t - w + 1))
        m = float(out[:, s:s + w].mean())
        out[:, s:s + w] = m
    return out.astype(np.float32)


def time_shift(feats: np.ndarray, rng: np.random.Generator,
               max_shift: int = 15) -> np.ndarray:
    """Shift a (mel, frames) map with zero fill (no wrap-around)."""
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return feats
    out = np.zeros_like(feats)
    if s > 0:
        out[:, s:] = feats[:, :-s]
    else:
        out[:, :s] = feats[:, -s:]
    return out
