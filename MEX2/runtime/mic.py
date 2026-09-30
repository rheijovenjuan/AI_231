"""Microphone capture loop shared by the PC and Raspberry Pi runners.

Backends: `sounddevice` (default) or `pyaudio`.  Both are tried in that order
when `backend="auto"`.
"""

from __future__ import annotations

import sys
import time

import numpy as np

DEFAULT_SR = 16_000


def open_stream(sr=DEFAULT_SR, frame_ms=30, backend="auto", device=None):
    """Return (stream, read_fn, backend_name) with read_fn() -> np.float32 mono."""
    frame = int(sr * frame_ms / 1000)

    if backend in ("auto", "sounddevice"):
        try:
            import sounddevice as sd

            sd.check_input_settings(samplerate=sr, channels=1, dtype="float32")
            stream = sd.InputStream(samplerate=sr, channels=1,
                                    dtype="float32", blocksize=frame,
                                    device=device)

            def read():
                data, _ = stream.read(frame)
                return np.asarray(data, dtype=np.float32).reshape(-1)

            return stream, read, "sounddevice"
        except Exception as exc:
            if backend == "sounddevice":
                raise
            sys.stderr.write(f"[mic] sounddevice unavailable ({exc})\n")

    import pyaudio
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr,
                     input=True, frames_per_buffer=frame)

    def read():
        return np.frombuffer(stream.read(frame, exception_on_overflow=False),
                             dtype=np.float32).copy()

    return (pa, stream), read, "pyaudio"


class RingBuffer:
    def __init__(self, seconds: float, sr: int):
        self.n = int(seconds * sr)
        self.buf = np.zeros(self.n, dtype=np.float32)
        self.sr = sr

    def push(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float32).reshape(-1)
        if f.shape[0] >= self.n:
            self.buf[:] = f[-self.n:]
            return
        self.buf = np.roll(self.buf, -f.shape[0])
        self.buf[-f.shape[0]:] = f

    def last(self, seconds: float) -> np.ndarray:
        n = min(int(seconds * self.sr), self.n)
        return self.buf[-n:].copy()

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.buf ** 2) + 1e-12))


def blink(pin, times=1, on=0.08, off=0.08):
    """Optional Raspberry Pi LED feedback (no-op if RPi.GPIO is absent)."""
    if pin is None:
        return
    try:
        import RPi.GPIO as GPIO
    except Exception:
        return
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        for _ in range(times):
            GPIO.output(pin, GPIO.HIGH); time.sleep(on)
            GPIO.output(pin, GPIO.LOW); time.sleep(off)
    except Exception:
        pass


def play(wave, sr=DEFAULT_SR, backend="auto", device=None) -> bool:
    """Play a float32 mono clip through the default (or `device`) output.

    Returns True if the clip was played, False if no output backend worked.
    """
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return False
    if backend in ("auto", "sounddevice"):
        try:
            import sounddevice as sd

            sd.play(x, samplerate=sr, device=device, dtype="float32")
            sd.wait()
            return True
        except Exception as exc:
            if backend == "sounddevice":
                raise
            sys.stderr.write(f"[mic] sounddevice playback unavailable ({exc})\n")

    try:
        import pyaudio
    except Exception as exc:
        sys.stderr.write(f"[mic] no playback backend ({exc})\n")
        return False
    pa = pyaudio.PyAudio()
    try:
        out = pa.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
        out.write(x.tobytes())
        out.stop_stream()
        out.close()
        return True
    except Exception as exc:
        sys.stderr.write(f"[mic] playback failed ({exc})\n")
        return False
    finally:
        pa.terminate()


def _pause_input(stream) -> bool:
    """Stop the mic while audio is being played out (avoids echo/overflow)."""
    try:
        if isinstance(stream, tuple):                 # pyaudio
            stream[1].stop_stream()
        else:                                         # sounddevice
            stream.stop()
        return True
    except Exception:
        return False


def _resume_input(stream) -> None:
    try:
        if isinstance(stream, tuple):
            stream[1].start_stream()
        else:
            stream.start()
    except Exception as exc:
        sys.stderr.write(f"[mic] could not restart the input stream ({exc})\n")


def listen(pipeline, sr=DEFAULT_SR, frame_ms=30, hop_ms=100,
           buffer_sec=2.0, vad_rms=6e-3, max_seconds=None,
           show_slot=False, as_json=False, led_pin=None,
           backend="auto", device=None, verbose=False,
           on_result=None, playback=False, playback_device=None,
           playback_backend=None) -> dict:
    """Blocking mic loop: wake word -> command -> print the class.

    With `playback=True` the captured command is played out loud *before* it
    is classified, so an audience hears exactly what the model was given.

    Returns a summary dict when the loop stops.
    """
    stream, read, name = open_stream(sr=sr, frame_ms=frame_ms,
                                     backend=backend, device=device)
    ring = RingBuffer(buffer_sec, sr)
    wl = int(pipeline.wake_seconds * sr)
    frames_per_hop = max(int(hop_ms / frame_ms), 1)

    print(f"[mic] backend={name} sr={sr} frame={frame_ms}ms "
          f"threshold={pipeline.threshold:.3f}", flush=True)
    print("[mic] listening for wake word ...", flush=True)

    stats = {"frames": 0, "wake_scans": 0, "commands": 0,
             "wake_detected": 0, "started": time.time()}
    capturing = False
    cap = []
    cap_silence = 0.0
    cap_speech = 0.0
    wake_score_at_trigger = 0.0
    t_start = time.time()
    step = 0

    try:
        while True:
            frame = read()
            ring.push(frame)
            stats["frames"] += 1
            step += 1
            if max_seconds is not None and time.time() - t_start > max_seconds:
                break
            if step % frames_per_hop:
                continue

            rms = float(np.sqrt(np.mean(ring.last(buffer_sec) ** 2) + 1e-12))
            frame_rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))

            if not capturing:
                if rms < vad_rms:
                    continue                      # silence: skip inference
                score = pipeline.wake_score(ring.last(pipeline.wake_seconds))
                stats["wake_scans"] += 1
                if verbose:
                    print(f"[scan] wake={score:.3f} rms={rms:.4f}", flush=True)
                if score >= pipeline.threshold:
                    stats["wake_detected"] += 1
                    wake_score_at_trigger = score
                    capturing = True
                    cap = []
                    cap_silence = 0.0
                    cap_speech = 0.0
                    blink(led_pin, 1)
                    if not as_json:
                        print(f"WAKE {score:.2f}", flush=True)
                    else:
                        print('{"event":"wake","score":%.3f}' % score, flush=True)
            else:
                cap.append(frame.copy())
                cap_silence = (cap_silence + frame_ms / 1000.0
                               if frame_rms < vad_rms else 0.0)
                if frame_rms >= vad_rms:
                    cap_speech += frame_ms / 1000.0
                total = len(cap) * frame_ms / 1000.0
                enough = total >= pipeline.cmd_seconds
                done = (cap_speech > 0.25 and cap_silence >= 0.45) or \
                       (cap_silence >= 1.0) or enough
                if done:
                    wave = np.concatenate(cap) if cap else np.zeros(1)
                    if playback:
                        if not as_json:
                            print(f"[mic] playback {total:.2f}s ...", flush=True)
                        paused = _pause_input(stream)
                        try:
                            play(wave, sr=sr,
                                 backend=playback_backend or backend,
                                 device=playback_device)
                        except Exception as exc:
                            sys.stderr.write(f"[mic] playback error ({exc})\n")
                        finally:
                            if paused:
                                _resume_input(stream)
                    res = pipeline.classify(wave)
                    res["wake_score"] = round(wake_score_at_trigger, 4)
                    res["command_seconds"] = round(total, 2)
                    stats["commands"] += 1
                    capturing = False
                    if on_result:
                        on_result(res)
                    if as_json:
                        import json as _json
                        print(_json.dumps(res), flush=True)
                    else:
                        print(f">>> {res['intent']}", flush=True)
                        if show_slot and "slot" in res:
                            print(f"    slot: {res['slot']} "
                                  f"({res.get('slot_confidence', 0):.2f})",
                                  flush=True)
                        print("[mic] listening for wake word ...", flush=True)
                    blink(led_pin, 2)
    except KeyboardInterrupt:
        print("\n[mic] stopped", flush=True)
    finally:
        try:
            if isinstance(stream, tuple):
                stream[1].stop_stream(); stream[1].close(); stream[0].terminate()
            else:
                stream.stop(); stream.close()
        except Exception:
            pass
    stats["uptime_sec"] = round(time.time() - stats["started"], 1)
    return stats
