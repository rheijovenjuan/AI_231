"""Generate two short demo tracks (sine melodies) into ./music so the
play/pause/stop/next actions are audible in the demo. Stdlib only."""
import math
import os
import struct
import wave

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")
os.makedirs(OUT, exist_ok=True)
SR = 22050


def tone(freq, dur, sr=SR, vol=0.3):
    n = int(sr * dur)
    return [vol * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def melody(name, notes, beat=0.25):
    samples = []
    for f, d in notes:
        samples += tone(f, d * beat)
    path = os.path.join(OUT, name)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32767))
                               for s in samples))
    print("wrote", path, f"{len(samples)/SR:.1f}s")


# C major arpeggio loop
melody("demo_song_1.wav", [(262, 1), (330, 1), (392, 1), (523, 2),
                           (392, 1), (330, 1), (262, 2)])
# G major, different feel
melody("demo_song_2.wav", [(392, 1), (494, 1), (587, 1), (784, 2),
                           (587, 1), (494, 1), (392, 2)])
