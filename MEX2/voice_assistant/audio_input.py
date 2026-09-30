"""Audio input: microphone capture + endpointing (VAD), with a file fallback.

On the Raspberry Pi this opens the real microphone (PortAudio via ``sounddevice``).
For local/desktop testing it can instead "listen" to a WAV file, so the entire
pipeline is testable without hardware. Both expose the same iterator interface::

    for chunk in source.iter_chunks():   # 16 kHz mono float64, ~30 ms each
        ...
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np

from . import features as F

CHUNK_MS = 30
CHUNK = int(F.SAMPLE_RATE * CHUNK_MS / 1000)


class Microphone:
    """Live 16 kHz mono capture using sounddevice (PortAudio)."""

    def __init__(self, device: Optional[int] = None, chunk_ms: int = CHUNK_MS):
        import sounddevice as sd
        self.sd = sd
        self.chunk = int(F.SAMPLE_RATE * chunk_ms / 1000)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.device = device
        self._stop = False
        self._device_sr = F.SAMPLE_RATE

    def _cb(self, indata, frames, t, status):
        if status:
            print("[mic]", status)
        # PortAudio delivers float32; the pipeline expects float64.
        x = indata[:, 0].astype(np.float64, copy=False)
        if self._device_sr != F.SAMPLE_RATE:
            x = F.resample_linear(x, self._device_sr, F.SAMPLE_RATE)
        self.q.put(x)

    def __enter__(self):
        try:
            self.stream = self.sd.InputStream(
                samplerate=F.SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=self.chunk, device=self.device, callback=self._cb,
                latency="high")
        except Exception as exc:
            # Many USB mics (esp. on Raspberry Pi) have no 16 kHz mode, and
            # ALSA/PortAudio will not resample for us (PaErrorCode -9997).
            # Open at the device's native rate; _cb resamples to 16 kHz.
            info = self.sd.query_devices(self.device, "input")
            rate = int(round(float(info.get("default_samplerate",
                                             F.SAMPLE_RATE))))
            if rate <= 0:
                rate = 48000
            self._device_sr = rate
            blocksize = max(1, int(round(rate * self.chunk / F.SAMPLE_RATE)))
            print(f"[mic] device has no 16 kHz mode ({exc}); capturing at "
                  f"{rate} Hz -> resampling to {F.SAMPLE_RATE} Hz")
            self.stream = self.sd.InputStream(
                samplerate=rate, channels=1, dtype="float32",
                blocksize=blocksize, device=self.device, callback=self._cb,
                latency="high")
        self.stream.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self.stream.stop()
        self.stream.close()

    def iter_chunks(self) -> Iterator[np.ndarray]:
        while not self._stop:
            try:
                yield self.q.get(timeout=0.5)
            except queue.Empty:
                continue

    def read_until_silence(self, pre_roll: int = 3, max_secs: float = 6.0,
                           silence_secs: float = 0.6,
                           energy_min: float = 1e-4) -> np.ndarray:
        """Block until speech starts, then record until trailing silence."""
        start = time.time()
        voiced: list = []
        waiting = True
        last_voice = 0.0
        while time.time() - start < max_secs:
            chunk = self.q.get(timeout=0.5)
            e = float(np.mean(chunk ** 2))
            if e > energy_min:
                waiting = False
                last_voice = time.time()
                voiced.append(chunk)
            elif not waiting:
                voiced.append(chunk)
                if time.time() - last_voice > silence_secs:
                    break
        if not voiced:
            return np.zeros(CHUNK)
        return np.concatenate(voiced)


class FileSource:
    """Pretend-microphone that streams a WAV file in chunks."""

    def __init__(self, path: str):
        self.x, _ = F.load_wav(path)
        self.x = F.resample_linear(self.x, _, F.SAMPLE_RATE)
        self.i = 0

    def iter_chunks(self) -> Iterator[np.ndarray]:
        while self.i < len(self.x):
            yield self.x[self.i:self.i + CHUNK]
            self.i += CHUNK

    def read_until_silence(self, **_) -> np.ndarray:
        return self.x


def make_source(kind: str, path: Optional[str] = None,
                device: Optional[int] = None):
    if kind == "file":
        return FileSource(path)
    return Microphone(device=device)
