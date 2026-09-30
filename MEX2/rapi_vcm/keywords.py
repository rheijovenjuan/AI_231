"""Keyword vocabulary for the Rapi keyword spotter.

Two stages, mirroring the rest of the system:

1. ``transcript_to_keywords(text)`` - the multi-label target: which of the
   40 keywords appear in the utterance. Labels are derived from the OptionB
   transcripts, so they are exact and free.
2. ``keywords_to_intent(kw)`` - a deterministic rule that maps a keyword set
   back to the primary output, the 19-class command, plus its slot value.

The wake word ("hey rapi") is NOT part of this vocabulary: ``wake.onnx``
acknowledges it before the query reaches the spotter.
"""

from __future__ import annotations

import csv
import re

from .labels import SLOT_VALUES

# (keyword, surface forms) - matched as whole words / phrases, lower case.
_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("play", ("play", "playing")),
    ("music", ("music", "song")),
    ("next", ("next", "skip")),
    ("pause", ("pause",)),
    ("stop", ("stop", "end", "playback")),
    ("volume", ("volume",)),
    ("up", ("up", "increase")),
    ("down", ("down", "lower")),
    ("lights", ("lights", "light")),
    ("on", ("on", "power")),
    ("off", ("off", "out")),
    ("call", ("call", "phone")),
    ("message", ("message", "send")),
    ("reminder", ("reminder", "reminders", "remind", "list", "show")),
    ("timer", ("timer", "countdown")),
    ("alarm", ("alarm", "wake")),
    ("temperature", ("temperature",)),
    ("brightness", ("brightness",)),
    ("color", ("color", "colour")),
    ("time", ("time",)),
    ("weather", ("weather",)),
]

_SLOT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("10 seconds", ("10 seconds",)),
    ("30 seconds", ("30 seconds",)),
    ("1 minute", ("1 minute", "1 minutes")),
    ("4 am", ("4 am",)),
    ("8 am", ("8 am",)),
    ("9 pm", ("9 pm",)),
    ("18 degrees", ("18 degrees",)),
    ("22 degrees", ("22 degrees",)),
    ("26 degrees", ("26 degrees",)),
    ("20 percent", ("20 percent",)),
    ("60 percent", ("60 percent",)),
    ("100 percent", ("100 percent",)),
    ("red", ("red",)),
    ("blue", ("blue",)),
    ("green", ("green",)),
    ("yellow", ("yellow",)),
    ("drink water", ("drink water",)),
    ("study", ("study",)),
    ("exercise", ("exercise",)),
]

KEYWORDS: list[str] = [k for k, _ in _INTENT_KEYWORDS + _SLOT_KEYWORDS]
KEYWORD_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(KEYWORDS)}
N_KEYWORDS = len(KEYWORDS)

INTENT_KEYWORDS: list[str] = [k for k, _ in _INTENT_KEYWORDS]
SLOT_KEYWORDS: list[str] = [k for k, _ in _SLOT_KEYWORDS]

_ALIASES: dict[str, str] = {alias: kw
                            for kw, aliases in _INTENT_KEYWORDS + _SLOT_KEYWORDS
                            for alias in aliases}
_PHRASES: tuple[str, ...] = tuple(kw for kw in KEYWORDS if " " in kw)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

COLOR_KEYWORDS: tuple[str, ...] = ("red", "blue", "green", "yellow")
CREATE_SLOT_KEYWORDS = tuple(SLOT_VALUES["CREATE_REMINDER"])


def _norm(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def transcript_to_keywords(text: str) -> list[str]:
    """Keywords present in `text`, in vocabulary order."""
    norm = _norm(text)
    hit = {_ALIASES[t] for t in _TOKEN_RE.findall(norm) if t in _ALIASES}
    hit |= {kw for kw in _PHRASES if kw in norm}
    return [k for k in KEYWORDS if k in hit]


def keywords_to_multihot(keywords) -> list[float]:
    v = [0.0] * N_KEYWORDS
    for k in keywords:
        v[KEYWORD_TO_IDX[k]] = 1.0
    return v


def multihot_to_keywords(v) -> list[str]:
    return [KEYWORDS[i] for i, x in enumerate(v) if x >= 0.5]


def keywords_to_intent(keywords) -> tuple[str, str | None]:
    """Deterministic keyword-set -> (intent, slot value or None).

    Priority resolves the overlaps that exist in the templates:
    "play next song" -> NEXT (not PLAY_MUSIC), "change the lights to blue"
    -> COLOR (not LIGHT_ON), "stop playing" -> STOP (not PLAY_MUSIC).
    """
    k = set(keywords)

    def slot_for(intent: str) -> str | None:
        low = {x.lower() for x in k}
        for cand in SLOT_VALUES[intent]:
            if cand.lower() in low:
                return cand
        return None

    if k & set(COLOR_KEYWORDS):
        return "COLOR", next(c for c in COLOR_KEYWORDS if c in k)
    if k & set(CREATE_SLOT_KEYWORDS):
        return "CREATE_REMINDER", slot_for("CREATE_REMINDER")
    if "timer" in k:
        return "TIMER", slot_for("TIMER")
    if "alarm" in k:
        return "ALARM", slot_for("ALARM")
    if "temperature" in k:
        return "TEMPERATURE", slot_for("TEMPERATURE")
    if "brightness" in k:
        return "BRIGHTNESS", slot_for("BRIGHTNESS")
    if "volume" in k and k & {"up", "increase"}:
        return "VOLUME_UP", None
    if "volume" in k and k & {"down", "lower"}:
        return "VOLUME_DOWN", None
    if "lights" in k and "on" in k:
        return "LIGHT_ON", None
    if "lights" in k and "off" in k:
        return "LIGHT_OFF", None
    if "next" in k:
        return "NEXT", None
    if "pause" in k:
        return "PAUSE", None
    if "stop" in k:
        return "STOP", None
    if "play" in k or "music" in k:
        return "PLAY_MUSIC", None
    if "call" in k:
        return "CALL", None
    if "message" in k:
        return "MESSAGE", None
    if "reminder" in k:
        return "LIST_REMINDERS", None
    if "time" in k:
        return "TIME", None
    if "weather" in k:
        return "WEATHER", None
    return "PLAY_MUSIC", None


def intent_from_transcript(text: str) -> tuple[str, str | None]:
    return keywords_to_intent(transcript_to_keywords(text))


def selfcheck(manifest_path: str) -> int:
    """Check the rule against every manifest row.

    Returns the number of distinct (transcript, want, got) disagreements;
    0 means the keyword vocabulary + rules reproduce the dataset exactly.
    """
    bad: set = set()
    n = 0
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            want_slot = (row.get("slot_value") or "").strip() or None
            got_intent, got_slot = intent_from_transcript(row["transcript"])
            if got_intent != row["intent"] or got_slot != want_slot:
                bad.add((row["transcript"], row["intent"], want_slot,
                         got_intent, got_slot))
    for t, wi, ws, gi, gs in sorted(bad):
        print(f"  MISS {t!r}: want {wi}/{ws} got {gi}/{gs}")
    print(f"{n - len(bad)}/{n} rows correct ({len(KEYWORDS)} keywords)")
    return len(bad)


def list_keywords() -> None:
    """Print the vocabulary: keyword, aliases, and the intent it feeds."""
    print(f"{N_KEYWORDS} keywords "
          f"({len(INTENT_KEYWORDS)} intent atoms + {len(SLOT_KEYWORDS)} slot values)\n")
    print("intent atoms")
    for kw, aliases in _INTENT_KEYWORDS:
        alias = ", ".join(a for a in aliases if a != kw)
        print(f"  {kw:<13s} <- {alias}" if alias else f"  {kw}")
    print("\nslot values")
    for kw, aliases in _SLOT_KEYWORDS:
        alias = ", ".join(a for a in aliases if a != kw)
        print(f"  {kw:<13s} <- {alias}" if alias else f"  {kw}")
    print("\nexamples (keywords -> output)")
    for text in ("play some music", "set a timer for 10 seconds",
                 "change the lights to blue", "turn the brightness to 60 percent"):
        kws = transcript_to_keywords(text)
        intent, slot = keywords_to_intent(kws)
        print(f"  {text!r}\n      -> {kws}\n      -> {intent}"
              + (f" / {slot}" if slot else ""))


if __name__ == "__main__":
    import os
    import sys

    if "--list" in sys.argv or "-l" in sys.argv:
        list_keywords()
        sys.exit(0)

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        cands = [
            os.path.join(root, "dataset", "manifest.csv"),
            os.path.join(root, "dataset", "ai231_tmp", "MEX2", "OptionB",
                         "manifest.csv"),
        ]
        path = next((p for p in cands if os.path.exists(p)), cands[0])
    sys.exit(1 if selfcheck(path) else 0)
