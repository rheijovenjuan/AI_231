"""Offline speech-to-text for the captured command (faster-whisper).

The transcription is shown BEFORE the intent result, so you can verify what
the microphone actually heard.  Lazy-loaded: the model downloads on first
use (~145 MB for base.en) and is cached by HuggingFace afterwards.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class Transcriber:
    def __init__(self, model_size: str = "base.en"):
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except Exception:
            return False

    def warmup(self) -> None:
        """Load/download the model in the background (first use is slow)."""
        threading.Thread(target=self._ensure, daemon=True).start()

    def _ensure(self):
        with self._lock:
            if self._model is None and self.load_error is None:
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(self.model_size, device="cpu",
                                                compute_type="int8")
                except Exception as exc:  # noqa: BLE001
                    self.load_error = str(exc)
        return self._model

    def transcribe(self, wave: np.ndarray) -> str:
        """float32 mono 16 kHz -> text ("" when unavailable)."""
        model = self._ensure()
        if model is None:
            return ""
        x = np.asarray(wave, dtype=np.float32).reshape(-1)
        segs, _info = model.transcribe(x, beam_size=1, language="en",
                                       vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()
