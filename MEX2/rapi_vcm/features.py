"""Audio loading and log-mel feature extraction (pure numpy/scipy).

The same code runs during training on the DGX node and during inference on a
laptop or on the Raspberry Pi, which guarantees that train/test/inference
features are bit-for-bit consistent.  No librosa/numba dependency so that the
Raspberry Pi install stays light.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ----------------------------------------------------------------- acoustic --
SAMPLE_RATE = 16_000
N_FFT = 512          # 32 ms window
HOP_LENGTH = 160     # 10 ms hop  -> 100 frames / second
WINDOW = "hann"
N_MELS = 40
FMIN = 20.0
FMAX = 7600.0
EPS = 1e-6

# ------------------------------------------------------------ utterance size --
CMD_SECONDS = 2.5    # command window  (250 frames)
WAKE_SECONDS = 1.2   # wake-word window (120 frames)

CMD_FRAMES = int(CMD_SECONDS * SAMPLE_RATE / HOP_LENGTH)   # 250
WAKE_FRAMES = int(WAKE_SECONDS * SAMPLE_RATE / HOP_LENGTH) # 120

FEAT_DIM = N_MELS


# ------------------------------------------------------------------- audio ---
def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV/anything-soundfile-reads as mono float32 at `sr`."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        mono = resample_poly(mono, sr // g, file_sr // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_raw(path: str):
    """Return (waveform, original sample rate) without resampling."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), file_sr


# -------------------------------------------------------------- mel bank -----
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT,
                   n_mels: int = N_MELS, fmin: float = FMIN,
                   fmax: float = FMAX) -> np.ndarray:
    """Triangular HTK-mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    # energy normalisation so loud/soft clips stay comparable
    enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
    fb *= enorm[:, None].astype(np.float32)
    return fb


_MEL_FB = mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def _frame(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]))
    n_frames = 1 + (x.shape[0] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def logmel(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Waveform (float32) -> log-mel filterbank, shape (N_MELS, T)."""
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.shape[0] < N_FFT:
        x = np.pad(x, (0, N_FFT - x.shape[0]))
    frames = _frame(x, N_FFT, HOP_LENGTH)
    spec = np.fft.rfft(frames * _WINDOW[None, :], axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = _MEL_FB @ power.T                       # (n_mels, n_frames)
    return np.log(mel + EPS).astype(np.float32)


def cmvn(feats: np.ndarray) -> np.ndarray:
    """Per-utterance cepstral mean / variance normalisation over time."""
    mean = feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True)
    return ((feats - mean) / (std + 1e-5)).astype(np.float32)


def pad_crop(feats: np.ndarray, frames: int) -> np.ndarray:
    """Right-pad with zeros or centre-crop to exactly `frames` columns."""
    t = feats.shape[1]
    if t == frames:
        return feats
    if t > frames:
        start = (t - frames) // 2
        return feats[:, start:start + frames]
    out = np.zeros((feats.shape[0], frames), dtype=feats.dtype)
    out[:, :t] = feats
    return out


def extract(path: str, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """File -> fixed-length (N_MELS, frames) log-mel feature map."""
    x = load_audio(path, sr=sr)
    frames = int(seconds * sr / HOP_LENGTH)
    return pad_crop(cmvn(logmel(x, sr)), frames)


def extract_waveform(x: np.ndarray, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """In-memory waveform -> fixed-length (N_MELS, frames) log-mel feature map."""
    frames = int(seconds * sr / HOP_LENGTH)
    return pad_crop(cmvn(logmel(x, sr)), frames)
