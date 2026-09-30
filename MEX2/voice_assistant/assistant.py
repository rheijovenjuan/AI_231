"""High-level orchestrator: wake word -> command -> slots -> action.

This is the brain of the assistant. It is transport-agnostic: feed it audio
chunks (from a mic or a file) and it emits events that the UI renders.

Flow
----
    idle --(hear "Hey Rapi")--> listening
    listening --(command utterance)--> classify -> parse slots -> dispatch
    dispatch --> idle   (music keeps playing across commands)

Models
------
The GMM joblib models of the original ``voice_assistant`` project are replaced
by the ONNX pipeline built in the sibling project:

* ``wake.onnx``      - "Hey Rapi" detector (1.2 s sliding window, sigmoid)
* ``keyword.onnx``   - 40-keyword multi-label spotter; a deterministic rule
                       maps the active keywords to the 19-class intent + slot
                       (``rapi_vcm.keywords.keywords_to_intent``)

Live capture improvements over the plain loop (the reason live labels used to
come out wrong):

* the wake-word tail is skipped (gap-based settle: onset search waits for
  the first quiet chunk after the wake fires, so "…rapi" never lands
  inside the command window; a fixed cap covers gapless speech),
* the segment starts at the first *voiced* frame and ends after
  ``TRAILING_SILENCE`` of quiet, so silence padding no longer dominates the
  log-mel input;
* a confidence gate (``min_confidence`` + at least one active keyword)
  refuses to dispatch a guess - it reports "not sure" instead of a wrong
  class.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .actions import ActionDispatcher
from .audio_input import CHUNK_MS
from .classifier import color_from_text, contact_from_text, parse_slots

SR = 16_000

RING_SEC = 2.0           # wake scan history
SCAN_HOP_MS = 100        # score the wake model every 100 ms of audio
SETTLE_SEC = 0.30        # deprecated: replaced by gap-based settle below
MAX_SETTLE_SEC = 0.75    # wait this long for a post-wake gap at most
TRAILING_SILENCE = 0.50  # end of command = this much quiet
MIN_SPEECH_SEC = 0.15    # shorter than this -> "didn't catch that"
MAX_SPEECH_SEC = 2.50    # matches the model's 2.5 s command window
LISTEN_TIMEOUT = 6.0     # give up waiting for a command
WAKE_THRESHOLD = 0.0     # 0 = take the threshold from model_card.json
WAKE_TRIGGER_FLOOR = 0.0  # >0 raises the calm trigger bar over card threshold
# Energy gate (mean square per 30 ms chunk).  6e-3 (rms 0.077) was far too
# strict: quiet command speech sits around rms 0.02-0.05 (e 4e-4..2.5e-3),
# so it counted as silence and the endpointer starved.  2e-4 = rms 0.014.
DEFAULT_VAD_RMS = 2e-4


def play_beep(freq: float = 880.0, dur: float = 0.15, vol: float = 0.3,
              sr: int = SR) -> None:
    """Play a short acknowledgement tone (best-effort, never raises)."""
    try:
        import sounddevice as sd
        t = np.linspace(0.0, dur, int(sr * dur), False)
        tone = vol * np.sin(2 * np.pi * freq * t)
        fade = int(0.01 * sr)
        if fade > 0:
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        sd.play(tone, sr)
        sd.wait()
    except Exception as exc:  # noqa: BLE001 - audio is optional
        print(f"[beep] audio unavailable ({exc}); continuing silently")


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"


def slots_from_keyword(intent: str, slot: Optional[str]) -> dict:
    """Map a keyword-head slot string onto the dispatcher's slot dict.

    ``{intent: key}`` mirrors what ``classifier.parse_slots`` produces, but
    the value comes from the acoustic keyword spotter instead of a
    transcript, so no ASR is involved.
    """
    slots = parse_slots(intent, "")          # sensible defaults
    if not slot:
        return slots
    s = slot.strip()
    low = intent
    if low == "COLOR":
        slots["color"] = s
    elif low == "BRIGHTNESS":
        slots["percent"] = s.split()[0]
    elif low == "TEMPERATURE":
        slots["degrees"] = s.split()[0]
    elif low == "TIMER":
        num = s.split()[0]
        secs = int(num) * 60 if "minute" in s else int(num)
        slots["duration"] = f"{secs}s"
    elif low == "ALARM":
        parts = s.split()
        h = int(parts[0]) % 12
        ap = parts[1].upper() if len(parts) > 1 else "AM"
        slots["time"] = f"{h:02d}:00 {ap}"
    elif low == "CREATE_REMINDER":
        slots["task"] = s
    return slots


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60}


def slots_from_text(intent: str, text: str) -> dict:
    """Slot values parsed from the ASR transcript (doc: slot filling via ASR).

    Returns ONLY values that are literally evidenced in the transcript -
    never parser defaults - so the caller can distinguish "found" from
    "not found" and keep the acoustic keyword head's value instead.
    """
    if not text:
        return {}
    t = " " + re.sub(r"[^a-z0-9% ]", " ", text.lower()) + " "
    out: dict = {}
    if intent == "COLOR":
        c = color_from_text(text)
        if c:
            out["color"] = c
    elif intent == "BRIGHTNESS":
        m = (re.search(r"(\d{1,3})\s*%", t)
             or re.search(r"(\d{1,3})\s+percent\b", t)
             or re.search(r"brightness\s+(?:to\s+|at\s+)?(\d{1,3})\b", t))
        if m:
            out["percent"] = str(max(0, min(100, int(m.group(1)))))
    elif intent == "TEMPERATURE":
        m = (re.search(r"(-?\d{1,3})\s*degree", t)
             or re.search(r"temperature\s+(?:to\s+)?(-?\d{1,3})\b", t))
        if m:
            out["degrees"] = m.group(1)
    elif intent == "VOLUME_SET":
        m = re.search(r"(\d{1,3})", t)
        if m:
            out["percent"] = str(max(0, min(100, int(m.group(1)))))
    elif intent == "TIMER":
        num = r"(\d+|(" + "|".join(_WORDNUM) + r"))"
        m = re.search(num + r"\s*(second|sec|minute|min)\w*\b", t)
        if m:
            raw = m.group(1)
            n = int(raw) if raw.isdigit() else _WORDNUM[raw]
            secs = n * 60 if m.group(3).startswith("min") else n
            out["duration"] = f"{secs}s"
    elif intent == "ALARM":
        m = (re.search(r"\b(\d{1,2})\s*(?::)?\s*(\d{2})?\s*(am|pm)\b", t)
             or re.search(r"\b(\d{1,2})\s*(?::)?\s*(\d{2})?\s*([ap])\s*m\b", t))
        if m:
            h = int(m.group(1)) % 12 or 12
            mm = m.group(2) or "00"
            if not mm.isdigit() or not 0 <= int(mm) <= 59:
                mm = "00"
            ap = m.group(3) or {"a": "AM", "p": "PM"}.get(m.group(4) or "",
                                                           "AM")
            out["time"] = f"{h:02d}:{int(mm):02d} {ap.upper()}"
    elif intent in ("CALL", "MESSAGE"):
        c = contact_from_text(text)
        if c:
            out["contact"] = c
    elif intent == "CREATE_REMINDER":
        m = (re.search(r"remind\s+me\s+(?:to|about)\s+(.+?)(?:\s+[.!?]|\s*$)",
                       t)
             or re.search(r"remind\s+(?:to|about)\s+(.+?)(?:\s+[.!?]|\s*$)", t)
             or re.search(r"reminder\s+(?:to|for)\s+(.+?)(?:\s+[.!?]|\s*$)", t))
        if m:
            out["task"] = m.group(1).strip()
    return out


def text_intent(text: str) -> Optional[tuple]:
    """Transcript-driven intent: ``(intent, color_value, strong)`` or None.

    The acoustic head only knows a fixed 40-keyword vocabulary, so it
    cannot express phrases like "dim the lights" or "change the lights
    to lavender" even when the ASR transcribed them correctly.  These
    high-precision rules run on the transcript and correct the acoustic
    intent whenever the words are clear; None = no evidence, keep the
    acoustic result.  ``strong=False`` marks evidence too weak to
    override a gate-passed acoustic result (bare colour words seen in
    ASR mishears like "White off please", info intents).
    """
    if not text:
        return None
    t = " " + re.sub(r"[^a-z0-9% ]", " ", text.lower()) + " "

    # high-specificity / slot-bearing intents first
    if re.search(r"\balarm\b|\bwake\s+me\b", t):
        return ("ALARM", None, False)
    if (re.search(r"\btimer\b|\bcountdown\b", t)
            or re.search(r"\b\d{1,4}\s*(?:seconds?|secs?|minutes?|mins?)\b", t)):
        return ("TIMER", None, False)
    if re.search(r"\bremind\s+me\s+(?:to|about)\b|\bcreate\s+(?:a\s+)?reminder"
                 r"|\breminder\s+(?:for\s+)?\w", t):
        return ("CREATE_REMINDER", None, True)
    if re.search(r"\breminders?\b", t):
        return ("LIST_REMINDERS", None, False)
    if re.search(r"\btemp(?:erature)?\b|\bdegrees?\b", t):
        return ("TEMPERATURE", None, False)

    # volume percentage ("volume 50", "set volume to 30 percent") - checked
    # before brightness so "volume 50 percent" doesn't read as a light level
    if re.search(r"\bvolume\b", t) and re.search(r"\d", t):
        return ("VOLUME_SET", None, True)

    # brightness: "dim ..." (with or without a number) or explicit numbers.
    # dim* also catches ASR garbles like "Dimba Lights" / "Dimba likes to"
    if re.search(r"\bdim\w*|\bdarker\b", t):
        return ("BRIGHTNESS", None, True)
    if re.search(r"\bbrightness\b", t) or re.search(r"\d{1,3}\s*%|\d{1,3}\s+percent", t):
        return ("BRIGHTNESS", None, True)

    # colour: palette colour or free-form word after a colour cue.
    # only a real cue ("change the lights to ...") is strong enough to
    # override a confident acoustic result.
    col = color_from_text(t)
    if col:
        strong = bool(re.search(
            r"colou?r|\blights?\s+to\b|\b(?:change|set|turn|make|paint)\b", t))
        return ("COLOR", col, strong)

    # lights on/off
    if (re.search(r"\blights?\s+(?:off|out)\b", t)
            or re.search(r"\boff\s+(?:the\s+)?lights?\b", t)):
        return ("LIGHT_OFF", None, True)
    if (re.search(r"\blights?\s+on\b", t)
            or re.search(r"\bon\s+(?:the\s+)?lights?\b", t)):
        return ("LIGHT_ON", None, True)

    # volume
    if (re.search(r"\blouder\b|\bturn\s+(?:it|that)\s+up\b", t)
            or (re.search(r"\bvolume\b", t)
                and re.search(r"\bup\b|\bincrease\b", t))):
        return ("VOLUME_UP", None, True)
    if (re.search(r"\bquieter\b|\bsilence\s+it\b|\bturn\s+(?:it|that)\s+down\b",
                  t)
            or (re.search(r"\bvolume\b", t)
                and re.search(r"\bdown\b|\blower\b|\bdecrease\b", t))):
        return ("VOLUME_DOWN", None, True)

    # transport (next/pause/stop must beat play, as in the keyword rules;
    # "playback"/"play back" is the stop alias used by the dataset).
    # NEXT alone is weak: narrative transcripts ("...next episode...") must
    # not beat a gate-passed acoustic result - genuine next rescues still
    # work via corroboration (_TEXT_KW_HINT) or a refused gate.
    if re.search(r"\bnext\b|\bskip\b", t):
        return ("NEXT", None, False)
    if re.search(r"\bpause\b|\bhold\s+the\s+music\b", t):
        return ("PAUSE", None, True)
    if re.search(r"\bstop\b|\bend\s+(?:the\s+)?playback\b"
                 r"|\bplayback\b|\bplay\s+back\b", t):
        return ("STOP", None, True)
    if re.search(r"\bcall\b", t):
        return ("CALL", None, False)
    if re.search(r"\bmessage\b|\bsend\s+(?:a\s+)?(?:message|text)\b", t):
        return ("MESSAGE", None, False)
    if re.search(r"\bplay\b|\bmusic\b|\bsong\b", t):
        return ("PLAY_MUSIC", None, False)

    # info intents: only rescue a refused acoustic result
    if re.search(r"\bweather\b", t):
        return ("WEATHER", None, False)
    if re.search(r"\btime\b|\bo'?clock\b", t):
        return ("TIME", None, False)
    return None


# acoustic keyword atoms that corroborate a weak text-rule intent: when the
# transcript says e.g. CALL, the word "call" must also have fired in the
# acoustic keyword head for the transcript to override a passing result.
_TEXT_KW_HINT: dict = {
    "CALL": {"call"},
    "MESSAGE": {"message"},
    "PAUSE": {"pause"},
    "STOP": {"stop", "playback"},
    "NEXT": {"next"},
    "VOLUME_UP": {"volume", "up"},
    "VOLUME_DOWN": {"volume", "down"},
    "VOLUME_SET": {"volume"},
    "LIGHT_ON": {"lights", "on"},
    "LIGHT_OFF": {"lights", "off"},
    "PLAY_MUSIC": {"play", "music"},
    "COLOR": {"color", "colour"},
    "BRIGHTNESS": {"brightness"},
    "TEMPERATURE": {"temperature"},
    "TIMER": {"timer"},
    "ALARM": {"alarm", "wake"},
    "CREATE_REMINDER": {"reminder"},
    "LIST_REMINDERS": {"reminder"},
    "TIME": {"time"},
    "WEATHER": {"weather"},
}


def _first_clause_cue(text: str, intent: str) -> bool:
    """True when the intent's cue word *opens* the utterance.

    Whisper marks pauses with '.', so a real command puts its cue in the
    first clause ("call Anna", "message Anna"), while narrative mentions
    ("...next episode ...") land in later clauses - those stay weak and
    cannot override a gate-passed acoustic result.
    """
    atoms = _TEXT_KW_HINT.get(intent)
    if not atoms or not text:
        return False
    clause = re.split(r"[.!?]+", text, maxsplit=1)[0]
    ct = " " + re.sub(r"[^a-z0-9% ]", " ", clause.lower()) + " "
    return any(re.search(rf"\b{re.escape(a)}\b", ct) for a in atoms)


def merged_slots(intent: str, slot: Optional[str], text: str = "",
                 policy: str = "keyword") -> dict:
    """Combine acoustic keyword slot + ASR transcript slot (doc item #5).

    policy="keyword": transcript fills only what the keyword head missed
    (its ``slot`` string empty) - cannot regress the acoustic pipeline.
    policy="text":    the transcript wins wherever it parses (fixes wrong
    acoustic values too, but trusts ASR).
    """
    slots = slots_from_keyword(intent, slot)
    from_text = slots_from_text(intent, text)
    if not from_text:
        return slots
    if policy == "text" or not slot:
        slots.update(from_text)
    return slots


class Assistant:
    def __init__(self, onnx_dir: str, music_dir: str,
                 on_event: Optional[Callable] = None,
                 wake_threshold: Optional[float] = None,
                 wake_trigger: float = WAKE_TRIGGER_FLOOR,
                 min_confidence: float = 0.45,
                 vad_rms: float = DEFAULT_VAD_RMS,
                 threads: int = 2,
                 beeps: bool = True,
                 confirm: str = "voice",
                 transcribe: bool = False,
                 stt_model: str = "base.en",
                 slot_policy: str = "text"):
        import sys
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from runtime.pipeline import VoicePipeline
        from . import confirm as confirm_mod
        from .stt import Transcriber

        self.pipe = VoicePipeline(onnx_dir, wake_threshold=wake_threshold,
                                  threads=threads)
        self.dispatcher = ActionDispatcher(music_dir, on_event)
        self.on_event = on_event or (lambda **_: None)
        self.min_confidence = float(min_confidence)
        self.vad_rms = float(vad_rms)
        self.beeps = beeps
        # wake acknowledgement: spoken "Yes?" (cached tts) | beep | silent
        self.confirm = confirm if confirm in ("voice", "beep", "off") else "voice"
        self._confirm_wav = (confirm_mod.ensure_confirmation()
                             if self.confirm == "voice" else None)
        # transcription (shown before the intent result)
        self.transcriber = None
        if transcribe and Transcriber.available():
            self.transcriber = Transcriber(stt_model)
            self.transcriber.warmup()
        # "keyword" = transcript fills gaps only; "text" = transcript wins
        self.slot_policy = slot_policy if slot_policy in ("keyword", "text") \
            else "keyword"

        if not os.path.exists(os.path.join(self.pipe.onnx_dir, "wake.onnx")):
            raise FileNotFoundError(f"wake.onnx not found in {onnx_dir}")

        # Trigger bar: never below the calibrated card threshold, never below
        # the (higher) calm floor - genuine wakes score ~1.0, so the floor
        # only removes marginal false triggers.
        self.trigger = max(self.pipe.threshold, float(wake_trigger))

        self.state = State.IDLE
        self.ring = np.zeros(int(RING_SEC * SR), dtype=np.float32)
        self._ring_fill = self.ring.shape[0]   # idle audio currently in ring
        self._wake_streak = 0                  # consecutive scans >= threshold
        self._scan_acc_ms = 0
        self._cmd: list = []
        self._onset = None
        self._listen_samples = 0     # audio time since wake fired
        self._speech_samples = 0     # voiced audio since command onset
        self._silent_samples = 0     # trailing silence since last voiced chunk
        self._gap_seen = False       # post-wake quiet chunk seen (gap settle)
        self._wake_score = 0.0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)
                             + 1e-12))

    def _push_ring(self, x: np.ndarray) -> None:
        n = x.shape[0]
        if n >= self.ring.shape[0]:
            self.ring[:] = x[-self.ring.shape[0]:]
            return
        self.ring = np.roll(self.ring, -n)
        self.ring[-n:] = x

    def _acknowledge_wake(self) -> None:
        """Spoken "Yes?" confirmation; beep fallback; silent when off.

        If a song is playing it keeps going but is ducked to low volume
        while the assistant listens (restored when it goes idle again).
        """
        self.dispatcher.music.duck()
        if not self.beeps or self.confirm == "off":
            return
        if self.confirm == "voice":
            from .confirm import play_wav
            if self._confirm_wav and play_wav(self._confirm_wav):
                return
            if self._confirm_wav is None:
                from . import confirm as confirm_mod
                self._confirm_wav = confirm_mod.ensure_confirmation()
        play_beep(freq=880.0, dur=0.15)

    def flush(self) -> None:
        """End of input (file finished / stream closed): finish what we have."""
        if self.state == State.LISTENING:
            if (self._onset is not None
                    and self._speech_samples >= int(MIN_SPEECH_SEC * SR)):
                seg = np.concatenate(self._cmd[self._onset:])
                self._process_command(seg)
            else:
                self._reset_idle("Didn't catch a command.")

    # ------------------------------------------------------------------ #
    # state machine
    # ------------------------------------------------------------------ #
    def feed_chunk(self, chunk: np.ndarray) -> None:
        """Push one 30 ms audio chunk; advance the state machine."""
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.state == State.IDLE:
            self._feed_wake(x)
        elif self.state == State.LISTENING:
            self._feed_command(x)

    def _feed_wake(self, x: np.ndarray):
        self._push_ring(x)
        self._ring_fill = min(self._ring_fill + x.shape[0],
                              self.ring.shape[0])
        self._scan_acc_ms += CHUNK_MS
        if self._scan_acc_ms < SCAN_HOP_MS:
            return
        if self._rms(self.ring) < self.vad_rms:
            self._wake_streak = 0
            return                      # silence: no inference at all
        if self._ring_fill < self.ring.shape[0]:
            self._scan_acc_ms = 0
            return                      # ring still refilling after a wake
        self._scan_acc_ms = 0
        score = self.pipe.wake_score(self.ring)
        if score >= self.trigger:
            self._wake_streak += 1      # must hold up over two scans
        else:
            self._wake_streak = 0
        if self._wake_streak >= 2:
            self._wake_score = score
            self._wake_streak = 0
            self.state = State.LISTENING
            self.ring[:] = 0.0          # drop stale wake audio immediately
            self._ring_fill = 0
            self._cmd = []
            self._onset = None
            self._listen_samples = 0
            self._speech_samples = 0
            self._silent_samples = 0
            self._gap_seen = False
            self._acknowledge_wake()
            self.on_event(state=State.LISTENING.value, wake_score=round(score, 3),
                          status="Yes? I'm listening.")

    def _feed_command(self, x: np.ndarray):
        n = x.shape[0]
        self._listen_samples += n
        if self._listen_samples > int(LISTEN_TIMEOUT * SR):
            self._reset_idle("Timed out waiting for a command.")
            return

        self._cmd.append(x)
        voiced = float(np.mean(x.astype(np.float64) ** 2)) >= self.vad_rms

        if self._onset is None:
            # gap-based settle: the wake phrase tail must end (first quiet
            # chunk) before onset search starts, whatever the trigger timing
            if not self._gap_seen:
                if (not voiced
                        or self._listen_samples >= int(MAX_SETTLE_SEC * SR)):
                    self._gap_seen = True
                else:
                    return
            if not voiced:
                return
            self._onset = len(self._cmd) - 1
            self._speech_samples = 0
            self._silent_samples = 0
            return

        if voiced:
            self._speech_samples += n
            self._silent_samples = 0
        elif self._speech_samples <= 0:
            # silence before the first word: keep trimming the front
            self._onset = len(self._cmd) - 1
            return
        else:
            self._silent_samples += n

        speech_done = (self._speech_samples >= int(MIN_SPEECH_SEC * SR)
                       and self._silent_samples >= int(TRAILING_SILENCE * SR))
        too_long = self._speech_samples >= int(MAX_SPEECH_SEC * SR)
        if speech_done or too_long:
            seg = np.concatenate(self._cmd[self._onset:])
            self._process_command(seg)

    def _acceptable(self, intent: str, kws: list, conf: float) -> bool:
        """Confidence gate + no blind fallback dispatch.

        ``keywords_to_intent`` falls back to PLAY_MUSIC for keyword sets no
        rule matches - never dispatch that fallback without play/music
        evidence (refuse instead of guessing).
        """
        if conf < self.min_confidence or not kws:
            return False
        if intent == "PLAY_MUSIC" and not (set(kws) & {"play", "music"}):
            return False
        return True

    def _decide(self, text: str, res: dict):
        """Final decision: ``(ok, intent, slots, conf, kws, refuse_msg)``.

        Transcript rules run first (they can express what the fixed
        40-keyword acoustic vocabulary cannot).  Strong text evidence
        corrects the acoustic intent; weak evidence (bare colour words,
        info intents) only fills in when the acoustic gate refuses, so
        an ASR mishear never overrides a confident acoustic result.
        With no usable text (ASR off/empty) the acoustic path decides.
        """
        conf = float(res.get("confidence", 0.0))
        kws = [k["keyword"] for k in res.get("keywords", [])]
        ac_intent = res.get("intent", "PLAY_MUSIC")
        ac_slot = res.get("slot")
        ac_ok = self._acceptable(ac_intent, kws, conf)

        ti = text_intent(text)
        if ti is not None:
            t_intent = ti[0]
            # weak evidence must be corroborated by the acoustic keyword
            # head (e.g. transcript says CALL and 'call' also fired) - or
            # open the utterance with its cue word ("message Anna"), which
            # makes it command evidence instead of narrative - so a lone
            # ASR mishear never overrides a confident acoustic hit.
            # CALL is exempt from the opening-cue rule: ASR often turns
            # "Color red" into "call ..." and then the acoustic colour
            # evidence (palette word + slot) is the only correct source.
            corroborated = bool(set(_TEXT_KW_HINT.get(t_intent, ())) & set(kws))
            strong = (ti[2]
                      or (_first_clause_cue(text, t_intent)
                          and t_intent != "CALL"))
            if (not ac_ok) or strong or corroborated:
                slot_arg = ac_slot if t_intent == ac_intent else None
                slots = merged_slots(t_intent, slot_arg, text, self.slot_policy)
                if t_intent == "BRIGHTNESS" and not re.search(r"\d", text):
                    slots.pop("percent", None)   # "dim": no number -> step down
                return True, t_intent, slots, conf, kws, None

        if not ac_ok:
            heard = ", ".join(kws) if kws else "nothing"
            msg = (f"Not sure (confidence {conf:.2f}, keywords: {heard}) - "
                   f"say the command again.")
            return False, ac_intent, {}, conf, kws, msg
        slots = merged_slots(ac_intent, ac_slot, text, self.slot_policy)
        return True, ac_intent, slots, conf, kws, None

    def _transcribe(self, x: np.ndarray) -> str:
        if self.transcriber is None:
            return ""
        try:
            return self.transcriber.transcribe(x)
        except Exception as exc:      # noqa: BLE001 - ASR must never break flow
            print(f"[stt] failed: {exc}")
            return ""

    def _process_command(self, x: np.ndarray):
        self.state = State.PROCESSING
        self.on_event(state=State.PROCESSING.value, status="Thinking...")
        text = self._transcribe(x)
        if text:
            # transcription comes FIRST so you can verify what was heard
            self.on_event(text=text)
        res = self.pipe.classify(x)
        ok, intent, slots, conf, kws, msg = self._decide(text, res)
        if not ok:
            self.on_event(confidence=round(conf, 3), keywords=kws)
            self._reset_idle(msg)
            return
        msg = self.dispatcher.dispatch(intent, slots)
        self.on_event(result=intent, confidence=round(conf, 3),
                      keywords=kws, slots=slots)
        self._reset_idle(msg)

    def _reset_idle(self, msg: str):
        if msg:
            # every path back to idle must SAY what happened - the timeout
            # and flush paths used to pass a message here that was dropped
            self.on_event(status=msg)
        self._cmd = []
        self._onset = None
        self._speech_samples = 0
        self._silent_samples = 0
        self._listen_samples = 0
        self._gap_seen = False
        self.dispatcher.music.unduck()   # full volume again once idle
        self._scan_acc_ms = SCAN_HOP_MS      # score the next chunk
        self.state = State.IDLE
        self.on_event(state=State.IDLE.value)

    # -- convenience for file-based / scripted testing -------------------- #
    def handle_utterance(self, x: np.ndarray) -> dict:
        """Classify a single command utterance directly (no wake word)."""
        text = self._transcribe(x)
        res = self.pipe.classify(x)
        ok, intent, slots, conf, kws, msg = self._decide(text, res)
        if ok:
            msg = self.dispatcher.dispatch(intent, slots)
        return {"intent": intent, "confidence": round(conf, 3),
                "keywords": kws, "slot": res.get("slot"), "slots": slots,
                "text": text,
                "status": msg, "dispatched": ok}
