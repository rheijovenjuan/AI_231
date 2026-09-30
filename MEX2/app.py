#!/usr/bin/env python
"""Rapi voice assistant - desktop / Raspberry Pi entry point.

Modes
-----
  --mode file   Play a recorded command WAV through the pipeline (no mic needed).
                Great for testing on a PC.  Repeat with --file for many.
  --mode mic    Live microphone: listen for "Hey Rapi", then a command.
                (Needs a mic + sounddevice. This is the real demo mode.)
  --mode gui    Open the Tkinter UI (status panels + log + Exit). Add
                --small-screen for a 3.5in 480x320 Pi touchscreen.

Examples
--------
  python app.py --mode file --file data/PLAY_MUSIC/PLAY_MUSIC_s1_v1_clean.wav
  python app.py --mode gui
  python app.py --mode mic --gui
  python app.py --mode mic --gui --small-screen   # 3.5in display
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from voice_assistant.assistant import Assistant  # noqa: E402
from voice_assistant.audio_input import make_source  # noqa: E402
from voice_assistant import features as F  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
ONNX_DIR = os.path.join(HERE, "output", "onnx")
MUSIC_DIR = os.path.join(HERE, "music")


def build_assistant(on_event, onnx_dir: str = ONNX_DIR,
                    min_confidence: float = 0.45,
                    wake_threshold: float | None = None,
                    wake_trigger: float = 0.0,
                    vad_rms: float | None = None,
                    transcribe: bool = True,
                    stt_model: str = "base.en",
                    confirm: str = "voice",
                    slot_policy: str = "text") -> Assistant:
    """Wake word + command recognition run on the ONNX models (output/onnx).

    transcribe=True prints the STT transcription before the intent result;
    confirm="voice" speaks "Yes?" on wake (beep/off for the alternatives);
    slot_policy="text" lets the transcript override the acoustic slot value.
    """
    kw = dict(onnx_dir=onnx_dir, music_dir=MUSIC_DIR, on_event=on_event,
              wake_threshold=wake_threshold, wake_trigger=wake_trigger,
              min_confidence=min_confidence, transcribe=transcribe,
              stt_model=stt_model, confirm=confirm,
              slot_policy=slot_policy)
    if vad_rms is not None:
        kw["vad_rms"] = vad_rms
    return Assistant(**kw)


def run_file_mode(files, onnx_dir: str = ONNX_DIR,
                  min_confidence: float = 0.45,
                  transcribe: bool = True,
                  stt_model: str = "base.en",
                  slot_policy: str = "text"):
    """Run each WAV through the classifier + dispatcher, print results."""
    assistant = build_assistant(on_event=_print_event, onnx_dir=onnx_dir,
                                min_confidence=min_confidence,
                                transcribe=transcribe, stt_model=stt_model,
                                confirm="off", slot_policy=slot_policy)
    print(f"Models: {os.path.abspath(assistant.pipe.onnx_dir)} "
          f"(head={assistant.pipe.head}, threshold={assistant.pipe.threshold:.3f})")
    print(f"Music dir: {MUSIC_DIR} "
          f"({len(assistant.dispatcher.music.tracks)} tracks)\n")
    for f in files:
        x = F.load_audio(f)
        res = assistant.handle_utterance(x)
        kws = ", ".join(res["keywords"]) or "-"
        if res["text"]:
            print(f"  {os.path.basename(f):40s} you said: \"{res['text']}\"")
            print(f"  {'':40s} -> {res['intent']:16s} "
                  f"conf={res['confidence']:.3f} kw=[{kws}]")
        else:
            print(f"  {os.path.basename(f):40s} -> {res['intent']:16s} "
                  f"conf={res['confidence']:.3f} kw=[{kws}]")
        print(f"      slots={res['slots']} => {res['status']}\n")


def _print_event(**kw):
    parts = []
    for k in ("text", "state", "status", "result", "slots", "confidence",
              "keywords"):
        if k in kw and kw[k]:
            parts.append(f"{k}={kw[k]}")
    if parts:
        print("  [event]", " ".join(parts))


def run_gui(with_mic: bool, music_dir: str = MUSIC_DIR,
            onnx_dir: str = ONNX_DIR, min_confidence: float = 0.45,
            wake_trigger: float = 0.0, vad_rms: float | None = None,
            transcribe: bool = True, stt_model: str = "base.en",
            confirm: str = "voice", slot_policy: str = "text",
            small_screen: bool = False, mic_device=None):
    import tkinter as tk
    from voice_assistant.ui import RapiUI

    root = tk.Tk()
    ev_q: "queue.Queue" = __import__("queue").Queue()

    def on_event(**kw):
        ev_q.put(kw)

    assistant = build_assistant(on_event=on_event, onnx_dir=onnx_dir,
                                min_confidence=min_confidence,
                                wake_trigger=wake_trigger, vad_rms=vad_rms,
                                transcribe=transcribe, stt_model=stt_model,
                                confirm=confirm, slot_policy=slot_policy)

    ui = RapiUI(root, on_command=lambda cmd: _dispatch_text(assistant, ui, cmd),
                small_screen=small_screen)
    # Route BOTH state-machine events and dispatcher actions to the UI.
    assistant.dispatcher.on_event = on_event

    def pump():
        try:
            while True:
                kw = ev_q.get_nowait()
                ui.on_event(**kw)
        except Exception:
            pass
        root.after(50, pump)

    root.after(50, pump)

    if with_mic:
        # Start in the wake-first state and begin listening for "Hey Rapi".
        on_event(state="idle", status='Waiting for wake word: "Hey Rapi"...')
        ui.log_line('Say "Hey Rapi" to wake me up.')
        threading.Thread(target=_mic_loop, args=(assistant, mic_device),
                         daemon=True).start()
    else:
        ui.log_line("Rapi ready (no mic) - display only.")

    root.mainloop()


def _dispatch_text(assistant, ui, text: str):
    """Handle a typed/clicked command (text) directly."""
    from voice_assistant.classifier import parse_slots
    intent = _text_to_intent(text)
    slots = parse_slots(intent, text)
    if intent == "BRIGHTNESS" and not any(c.isdigit() for c in text):
        slots["percent"] = ""        # "dim lights": step down, no number
    msg = assistant.dispatcher.dispatch(intent, slots)
    ui.on_event(result=intent, slots=slots, status=msg)


_TEXT_MAP = {
    "play music": "PLAY_MUSIC", "play": "PLAY_MUSIC",
    "pause": "PAUSE", "stop": "STOP", "next song": "NEXT", "next": "NEXT",
    "volume up": "VOLUME_UP", "volume down": "VOLUME_DOWN",
    "lights on": "LIGHT_ON", "lights off": "LIGHT_OFF",
    "weather": "WEATHER", "what time is it": "TIME", "time": "TIME",
    "call mom": "CALL", "message dad": "MESSAGE",
    "remind me to drink water": "CREATE_REMINDER",
    "timer 30 seconds": "TIMER", "alarm 8 am": "ALARM",
    "brightness 50 percent": "BRIGHTNESS",
    "dim lights": "BRIGHTNESS", "dim the lights": "BRIGHTNESS",
    "color red": "COLOR", "color green": "COLOR", "color blue": "COLOR",
    "temperature 22 degrees": "TEMPERATURE",
}


def _text_to_intent(text: str) -> str:
    t = text.lower().strip()
    if t in _TEXT_MAP:
        return _TEXT_MAP[t]
    # heuristic fallback
    if "bright" in t or "dim" in t:
        return "BRIGHTNESS"
    if "color" in t or "colour" in t:
        return "COLOR"
    if "temperature" in t or "temp" in t:
        return "TEMPERATURE"
    if "remind" in t:
        return "CREATE_REMINDER"
    if "timer" in t:
        return "TIMER"
    if "alarm" in t:
        return "ALARM"
    if "call" in t:
        return "CALL"
    if "message" in t:
        return "MESSAGE"
    if "volume" in t and "up" in t:
        return "VOLUME_UP"
    if "volume" in t and "down" in t:
        return "VOLUME_DOWN"
    if "next" in t:
        return "NEXT"
    if "pause" in t:
        return "PAUSE"
    if "stop" in t:
        return "STOP"
    if "play" in t:
        return "PLAY_MUSIC"
    if "light" in t and "off" in t:
        return "LIGHT_OFF"
    if "light" in t:
        return "LIGHT_ON"
    if "weather" in t:
        return "WEATHER"
    if "time" in t:
        return "TIME"
    return "WEATHER"


def _mic_loop(assistant: Assistant, device=None):
    try:
        src = make_source("mic", device=device)
    except Exception as e:
        print("[mic] unavailable:", e)
        return
    with src:
        print("[mic] listening for 'Hey Rapi' ... (Ctrl+C to stop)")
        for chunk in src.iter_chunks():
            assistant.feed_chunk(chunk)


def main():
    ap = argparse.ArgumentParser(description="Rapi voice assistant")
    ap.add_argument("--mode", choices=["file", "mic", "gui"], default="gui")
    ap.add_argument("--file", action="append", default=[],
                    help="WAV file to run in file mode (repeatable)")
    ap.add_argument("--gui", action="store_true",
                    help="open GUI alongside mic mode")
    ap.add_argument("--small-screen", action="store_true",
                    help="compact GUI layout for a 3.5in 480x320 display "
                         "(Raspberry Pi touchscreen; use with --gui)")
    ap.add_argument("--device", default=None,
                    help="mic input: sounddevice index or name substring "
                         "(see `python -m sounddevice`); default = system "
                         "input, e.g. --device 2 or --device 'USB'")
    ap.add_argument("--music-dir", default=MUSIC_DIR)
    ap.add_argument("--onnx", default=ONNX_DIR,
                    help="directory with wake.onnx / keyword.onnx / "
                         "model_card.json")
    ap.add_argument("--min-confidence", type=float, default=0.45,
                    help="refuse to dispatch below this keyword confidence")
    ap.add_argument("--wake-trigger", type=float, default=0.0,
                    help="raise the wake score needed to open the microphone "
                         "(0 = use the calibrated card threshold)")
    ap.add_argument("--vad-rms", type=float, default=None,
                    help="energy gate for speech onset/silence (mean square "
                         "per 30 ms chunk; default 2e-4)")
    ap.add_argument("--transcribe", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="print the STT transcription before the intent "
                         "(needs faster-whisper; --no-transcribe to disable)")
    ap.add_argument("--stt-model", default="base.en",
                    help="faster-whisper model size (tiny.en/base.en/...)")
    ap.add_argument("--confirm", choices=["voice", "beep", "off"],
                    default="voice",
                    help="wake acknowledgement: spoken 'Yes?' / beep / silent")
    ap.add_argument("--slot-policy", choices=["text", "keyword"],
                    default="text",
                    help="text = ASR transcript overrides the acoustic slot "
                         "value; keyword = acoustic value wins (transcript "
                         "only fills gaps)")
    args = ap.parse_args()

    os.makedirs(MUSIC_DIR, exist_ok=True)

    if args.mode == "file":
        if not args.file:
            ap.error("--mode file requires at least one --file")
        run_file_mode(args.file, onnx_dir=args.onnx,
                      min_confidence=args.min_confidence,
                      transcribe=args.transcribe,
                      stt_model=args.stt_model,
                      slot_policy=args.slot_policy)
    elif args.mode == "mic":
        if args.gui:
            run_gui(with_mic=True, onnx_dir=args.onnx,
                    min_confidence=args.min_confidence,
                    wake_trigger=args.wake_trigger, vad_rms=args.vad_rms,
                    transcribe=args.transcribe, stt_model=args.stt_model,
                    confirm=args.confirm, slot_policy=args.slot_policy,
                    small_screen=args.small_screen, mic_device=args.device)
        else:
            assistant = build_assistant(on_event=_print_event,
                                        onnx_dir=args.onnx,
                                        min_confidence=args.min_confidence,
                                        wake_trigger=args.wake_trigger,
                                        vad_rms=args.vad_rms,
                                        transcribe=args.transcribe,
                                        stt_model=args.stt_model,
                                        confirm=args.confirm,
                                        slot_policy=args.slot_policy)
            print(f"Models: {os.path.abspath(args.onnx)} "
                  f"(head={assistant.pipe.head}, "
                  f"trigger={assistant.trigger:.3f})")
            if args.transcribe and assistant.transcriber is None:
                print("[stt] faster-whisper not installed - run "
                      "`pip install faster-whisper` for transcriptions "
                      "(continuing without)")
            _mic_loop(assistant, args.device)
    else:  # gui
        run_gui(with_mic=False, onnx_dir=args.onnx,
                min_confidence=args.min_confidence,
                wake_trigger=args.wake_trigger, vad_rms=args.vad_rms,
                transcribe=args.transcribe, stt_model=args.stt_model,
                confirm=args.confirm, slot_policy=args.slot_policy,
                small_screen=args.small_screen, mic_device=args.device)


if __name__ == "__main__":
    main()
