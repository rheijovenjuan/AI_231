"""Fast wake->command regression WITHOUT Whisper (no ASR, no beeps, no TTS).

    python test_clips/sim_quick.py

Same flow as the assistant sees live (feed_chunk -> wake -> command), but
intent/slots come from the acoustic keyword path only, so a full run takes
~15 s instead of minutes. Use it for quick change-verification; use the
transcribe variant when you need transcript quality checked.
Exit code 0 only if every clip matches ground truth.
"""

from __future__ import annotations

import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from voice_assistant.assistant import Assistant  # noqa: E402
from voice_assistant import features as F        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INTENTS = ["PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE",
           "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE",
           "LIST_REMINDERS", "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS",
           "COLOR", "CREATE_REMINDER"]


def expected(name: str) -> str:
    core = name.split("__", 1)[-1]
    for it in INTENTS:
        if core.startswith(it + "_") or core.startswith(it + "."):
            return it
    return "?"


def main() -> int:
    events = []
    t0 = time.perf_counter()
    a = Assistant(onnx_dir=os.path.join(ROOT, "output", "onnx"),
                  music_dir=os.path.join(ROOT, "music"),
                  on_event=lambda **kw: events.append(kw),
                  confirm="off", transcribe=False, beeps=False)
    print(f"startup (no whisper): {time.perf_counter() - t0:.2f}s")

    files = sorted(glob.glob(os.path.join(HERE, "combined", "*.wav")))
    ok = 0
    for f in files:
        a.state = a.state.__class__.IDLE
        a.ring[:] = 0
        a._ring_fill = 0
        a._wake_streak = 0
        a._scan_acc_ms = 0
        events.clear()

        x = F.load_audio(f)
        for i in range(0, len(x), 480):
            a.feed_chunk(x[i:i + 480])
        a.flush()

        results = [e for e in events if "result" in e]
        exp = expected(os.path.basename(f))
        got = results[0]["result"] if results else "-"
        good = got == exp
        ok += good
        print(f"{os.path.basename(f):45s} -> {got:16s} "
              f"{'OK' if good else 'FAIL exp=' + exp}")

    print(f"\nintent {ok}/{len(files)} correct "
          f"in {time.perf_counter() - t0:.1f}s")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
