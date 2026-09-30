"""Spoken wake confirmation ("Yes?") - cached edge-tts audio, beep fallback.

On the first run the phrase is synthesised once with edge-tts and cached as
`audio/confirm_yes.wav`; after that wake-ups play it instantly.  Playback is
NON-blocking so microphone capture keeps running while the confirmation is
spoken.  If TTS is unavailable (no network / no edge-tts) the caller falls
back to the short beep.
"""
from __future__ import annotations

import asyncio
import os
import threading
from typing import Optional

CONFIRM_TEXT = "Yes?"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIRM_WAV = os.path.join(HERE, "audio", "confirm_yes.wav")


def _synth(path: str, text: str, timeout: float = 8.0) -> bool:
    """edge-tts text -> wav at native sample rate (async, timeout-guarded)."""
    import numpy as np

    async def go():
        import edge_tts
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".mp3"
        await edge_tts.Communicate(text, "en-US-JennyNeural",
                                   rate="-5%").save(tmp)
        import soundfile as sf
        d, sr = sf.read(tmp, dtype="float32")
        os.remove(tmp)
        if d.ndim > 1:
            d = d.mean(axis=1)
        # trim the TTS trailing/leading padding so the ack is snappy
        import numpy as np
        a = np.abs(d)
        lvl = max(float(a.max()) * 0.02, 1e-4)
        on = np.where(a >= lvl)[0]
        if len(on):
            s = max(int(on[0] - 0.05 * sr), 0)
            e = min(int(on[-1] + 0.10 * sr), len(d))
            d = d[s:e]
        sf.write(path, d, sr)

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(go(), timeout=timeout))
        finally:
            loop.close()
        return os.path.exists(path) and os.path.getsize(path) > 400
    except Exception:
        return False


def ensure_confirmation(text: str = CONFIRM_TEXT,
                        path: str = CONFIRM_WAV) -> Optional[str]:
    """Return a playable confirmation wav, generating it in the background.

    Never blocks the caller: if the file already exists it is returned
    immediately; otherwise synthesis runs in a daemon thread and the caller
    keeps using the beep until it is ready.
    """
    if os.path.exists(path) and os.path.getsize(path) > 400:
        return path
    threading.Thread(target=_synth, args=(path, text), daemon=True).start()
    return None


def play_wav(path: str) -> bool:
    """Play a wav file without blocking (returns False if audio failed)."""
    try:
        import sounddevice as sd
        import soundfile as sf
        d, sr = sf.read(path, dtype="float32")
        sd.play(d, sr)
        return True
    except Exception:
        return False
