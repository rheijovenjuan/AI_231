"""Audio loading and feature extraction.

Everything here is deliberately dependency-light so it runs on a Raspberry Pi:
  * Loading uses the stdlib ``wave`` module (the dataset is 16 kHz / mono /
    16-bit PCM). ``soundfile`` is used as a fallback when available.
  * Features are computed with NumPy + SciPy only (no librosa/torch needed at
    inference time). MFCCs are implemented directly on top of a DCT so the
    runtime footprint stays tiny.

Feature vector per utterance (fixed length, see FEATURE_DIM):
    [ 26 MFCC deltas (mean over time)
    , 26 MFCC delta-of-deltas (mean over time)
    , 1 log-energy (mean)
    , 1 zero-crossing-rate (mean)
    , 1 pitch (mean of voiced frames, Hz)
    , 1 duration (seconds) ]
"""

from __future__ import annotations

import math
import os
import wave
from typing import List, Tuple

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 256
N_MFCC = 26
N_FILTERS = 40
FEATURE_DIM = 26 * 3 + 1 + 1 + 1 + 1  # 82 (MFCC + delta + delta-delta + 4 scalar)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Return (float64 mono samples in [-1, 1], sample_rate)."""
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int32)
        if nch > 1:  # downmix
            arr = arr.reshape(-1, nch).mean(axis=1)
        if sw == 2:
            x = arr.astype(np.float64) / 32768.0
        else:
            x = arr.astype(np.float64) / 2147483648.0
        return x, sr
    except (wave.Error, EOFError):
        # Fallback for non-PCM containers.
        import soundfile as sf

        x, sr = sf.read(path, dtype="float64", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x, sr


def resample_linear(x: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return x
    n_out = int(round(len(x) * target_sr / sr))
    if n_out <= 1:
        return np.zeros(1)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x)


def load_audio(path: str) -> np.ndarray:
    x, sr = load_wav(path)
    return resample_linear(x, sr, SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Feature primitives
# --------------------------------------------------------------------------- #
def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Mel filter bank of shape (n_mels, n_fft//2 + 1) matching rfft output."""
    n_bins = n_fft // 2 + 1  # number of columns the power spectrum has

    def hz2mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = mel2hz(hz2mel(0)), mel2hz(hz2mel(sr / 2))
    mels = np.linspace(low, high, n_mels + 2)
    hz_pts = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    f = np.zeros((n_mels, n_bins))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for j in range(l, c):
            if c != l:
                f[i - 1, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r != c:
                f[i - 1, j] = (r - j) / (r - c)
    return f


def _dct_ii(x: np.ndarray, K: int) -> np.ndarray:
    """Type-II DCT along axis=1 (the mel band axis), keeping K coefficients.

    Input  (n_frames, n_bands) -> Output (n_frames, K).
    """
    n = x.shape[1]
    k = np.arange(K)[:, None]
    m = np.arange(n)[None, :]
    basis = np.cos(np.pi * k * (2 * m + 1) / (2 * n))  # (K, n_bands)
    return x @ basis.T                                  # (n_frames, K)


def mfcc_frame(x: np.ndarray, sr: int, n_fft: int, hop: int,
               n_mfcc: int, n_filters: int) -> np.ndarray:
    """Return (n_frames, n_mfcc) MFCC matrix."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
    window = np.hamming(n_fft)
    frames = np.empty((n_frames, n_fft))
    for i in range(n_frames):
        frames[i] = x[i * hop : i * hop + n_fft] * window
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    mel = _mel_filters(sr, n_fft, n_filters)
    logmel = np.log(power @ mel.T + 1e-10)
    return _dct_ii(logmel, n_mfcc)


def _delta(feats: np.ndarray, n: int = 2) -> np.ndarray:
    """Delta features (first derivative) with a window of +/- n frames."""
    T, D = feats.shape
    denom = 2.0 * sum(i * i for i in range(1, n + 1))
    out = np.zeros_like(feats)
    for t in range(T):
        acc = np.zeros(D)
        for k in range(1, n + 1):
            if t + k < T:
                acc += k * feats[t + k]
            if t - k >= 0:
                acc -= k * feats[t - k]
        out[t] = acc / denom
    return out


def _log_energy(x: np.ndarray, n_fft: int, hop: int) -> float:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    energies = []
    for i in range(max(1, n_frames)):
        seg = x[i * hop : i * hop + n_fft]
        energies.append(math.log(1e-10 + float(np.mean(seg ** 2))))
    return float(np.mean(energies)) if energies else 0.0


def _zcr(x: np.ndarray, n_fft: int, hop: int) -> float:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    vals = []
    for i in range(max(1, n_frames)):
        seg = x[i * hop : i * hop + n_fft]
        vals.append(float(np.mean(np.abs(np.diff(np.sign(seg))))))
    return float(np.mean(vals)) if vals else 0.0


def _pitch(x: np.ndarray, sr: int, n_fft: int, hop: int) -> float:
    """Simple autocorrelation pitch estimate; 0.0 if unvoiced."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    pitches = []
    min_lag = int(sr / 400)
    max_lag = int(sr / 70)
    for i in range(max(1, n_frames)):
        seg = x[i * hop : i * hop + n_fft]
        if np.max(np.abs(seg)) < 1e-4:
            continue
        ac = np.correlate(seg, seg, mode="full")[n_fft - 1:]
        ac = ac[: max_lag + 1]
        if ac[0] <= 0:
            continue
        peak = int(np.argmax(ac[min_lag:max_lag + 1])) + min_lag
        if ac[peak] > 0.3 * ac[0]:
            pitches.append(sr / peak)
    return float(np.median(pitches)) if pitches else 0.0


def extract_features(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a fixed-length feature vector of shape (FEATURE_DIM,)."""
    mf = mfcc_frame(x, sr, N_FFT, HOP, N_MFCC, N_FILTERS)
    if mf.shape[0] < 2:
        mf = np.vstack([mf, mf])
    d1 = _delta(mf)
    d2 = _delta(d1)
    vec = [
        *mf.mean(axis=0).tolist(),
        *d1.mean(axis=0).tolist(),
        *d2.mean(axis=0).tolist(),
        _log_energy(x, N_FFT, HOP),
        _zcr(x, N_FFT, HOP),
        _pitch(x, sr, N_FFT, HOP),
        len(x) / float(sr),
    ]
    v = np.asarray(vec, dtype=np.float64)
    assert v.shape[0] == FEATURE_DIM, v.shape
    return v


def extract_from_file(path: str) -> np.ndarray:
    return extract_features(load_audio(path))


def list_wavs(root: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)
