"""Command intent classifier + slot parser.

Architecture (chosen for Raspberry-Pi feasibility):
    * One Gaussian Mixture Model (GMM) per intent, fit on MFCC features.
    * At inference, score a feature vector against every intent GMM and take
      the argmax of the log-likelihoods. This is O(#intents) tiny matrix ops
      and runs in a few milliseconds on a Pi CPU.
    * Slot values (numbers, colours, times, durations, contact names, reminder
      tasks) are parsed from the transcript / recognised text with lightweight
      regex + fuzzy matching, so the acoustic model only has to pick the intent.

The whole model is a single ``joblib`` file (a dict of GMMs + metadata).
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import features as F

DEFAULT_INTENTS = [
    "PLAY_MUSIC", "PAUSE", "STOP", "WEATHER", "TIME",
    "LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR", "TEMPERATURE",
    "CREATE_REMINDER", "TIMER", "ALARM", "CALL", "MESSAGE",
    "NEXT", "VOLUME_UP", "VOLUME_DOWN", "LIST_REMINDERS",
]

COLORS = [
    # dataset palette
    "red", "green", "blue", "yellow",
    # extended palette - colour commands are not limited to RGB
    "orange", "purple", "violet", "indigo", "pink", "magenta", "cyan",
    "teal", "turquoise", "lavender", "lime", "gold", "silver", "bronze",
    "white", "black", "brown", "gray", "grey", "navy", "maroon",
    "crimson", "beige", "cream", "olive", "coral", "salmon",
]
CONTACTS_DEFAULT = ["mom", "dad", "anna", "james", "friend", "office",
                    "school", "brother", "sister", "doctor"]

# words that may follow a colour cue but are never a colour themselves
_COLOR_STOP = {"the", "a", "an", "on", "off", "to", "of", "in", "my",
               "some", "its", "it", "all", "more", "less", "dimmer",
               "darker", "brighter", "brightest"}


def color_from_text(text: str) -> Optional[str]:
    """Colour named in ``text``: palette match first, else the free-form
    word right after a colour cue ("change the lights to lavender")."""
    if not text:
        return None
    t = " " + re.sub(r"[^a-z0-9% ]", " ", text.lower()) + " "
    for c in COLORS:
        if re.search(rf"\b{c}\b", t):
            return c
    m = (re.search(r"\bcolou?r(?:ed)?\s+(?:the\s+)?(?:lights?\s+)?"
                   r"(?:to\s+)?(\w+)", t)
         or re.search(r"\blights?\s+(?:to|in)\s+(?:be\s+|the\s+)?(\w+)", t)
         or re.search(r"\bmake\s+(?:the\s+)?lights?\s+(?:be\s+|more\s+)?(\w+)",
                      t))
    if m:
        w = m.group(1)
        if len(w) >= 3 and not w.isdigit() and w not in _COLOR_STOP:
            return w
    return None


# filler words stripped around a contact name ("call my friend" -> "friend")
_CONTACT_LEAD = {"to", "the", "a", "an", "my", "our", "her", "his", "up",
                 "who", "is"}
_CONTACT_TRAIL = {"please", "now", "thanks", "today", "tonight", "asap",
                  "right", "away", "me", "up"}


def contact_from_text(text: str) -> Optional[str]:
    """Contact named after a call/message cue: 'call Anna' -> 'anna'.

    The word(s) right after the cue ARE the caller id - no contact book
    required.  Known names pass through unchanged; the raw ASR words are
    returned for unknown names.
    """
    if not text:
        return None
    t = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    m = re.search(r"\b(?:call|ring|dial|phone|message|text)\b", t)
    if not m:
        return None
    tail = t[m.end():].split()
    while tail and tail[0] in _CONTACT_LEAD:
        tail.pop(0)
    while tail and tail[-1] in _CONTACT_TRAIL:
        tail.pop()
    if not tail:
        return None
    return " ".join(tail[:3])


class CommandClassifier:
    def __init__(self, model_path: str, n_components: int = 8):
        self.model_path = model_path
        self.n_components = n_components
        self.models: Dict[str, object] = {}
        self.intents: List[str] = []
        self.contacts: List[str] = CONTACTS_DEFAULT[:]
        self.scaler = None  # (mean, std) for feature standardisation

    # -- persistence ------------------------------------------------------- #
    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        import joblib

        blob = joblib.load(self.model_path)
        self.models = blob["models"]
        self.intents = blob["intents"]
        self.contacts = blob.get("contacts", self.contacts)
        sc = blob.get("scaler")
        if sc is not None:
            self.scaler = (np.asarray(sc["mean"], dtype=np.float64),
                           np.asarray(sc["std"], dtype=np.float64))
        return True

    def save(self, models: Dict[str, object], intents: List[str],
             contacts: Optional[List[str]] = None,
             scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> str:
        import joblib

        self.models = models
        self.intents = intents
        if contacts:
            self.contacts = contacts
        if scaler is not None:
            self.scaler = (np.asarray(scaler[0], dtype=np.float64),
                           np.asarray(scaler[1], dtype=np.float64))
        payload = {"models": models, "intents": intents,
                   "contacts": self.contacts}
        if self.scaler is not None:
            payload["scaler"] = {"mean": self.scaler[0], "std": self.scaler[1]}
        joblib.dump(payload, self.model_path)
        return self.model_path

    def _prep(self, feats: np.ndarray) -> np.ndarray:
        """Standardise features (no-op when no scaler was saved)."""
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        if self.scaler is not None:
            mean, std = self.scaler
            return (feats - mean) / std
        return feats

    # -- inference --------------------------------------------------------- #
    def predict(self, feats: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Return (intent, confidence, scores-per-intent)."""
        feats = self._prep(feats)
        scores = np.array([self.models[i].score(feats) for i in self.intents])
        if len(scores) == 0:
            return self.intents[0] if self.intents else "UNKNOWN", 0.0, scores
        best = int(np.argmax(scores))
        # Convert to a soft confidence via softmax over the top-k.
        top = np.sort(scores)[::-1][:3]
        if len(top) < 2:
            conf = 1.0
        else:
            conf = float(np.exp(top[0]) / (np.exp(top[0]) + np.exp(top[1]) + 1e-9))
        return self.intents[best], conf, scores


# --------------------------------------------------------------------------- #
# Slot parsing (text based)
# --------------------------------------------------------------------------- #
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _word_to_num(tok: str) -> Optional[int]:
    tok = tok.lower().strip(".,")
    if tok.isdigit():
        return int(tok)
    if tok in _NUM_WORDS:
        return _NUM_WORDS[tok]
    return None


def _fuzzy(text: str, choices: List[str]) -> Optional[str]:
    """Pick the closest choice to text (word-boundary token match)."""
    text = text.lower()
    best, best_score = None, 0.0
    for c in choices:
        if re.search(rf"\b{re.escape(c)}\b", text):
            return c
        tl, cl = set(re.findall(r"\w+", text)), set(re.findall(r"\w+", c))
        if not cl:
            continue
        score = len(tl & cl) / len(cl)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


def parse_slots(intent: str, text: str) -> Dict[str, str]:
    """Extract slot values from the (transcribed) command text."""
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    slots: Dict[str, str] = {}

    if intent == "COLOR":
        slots["color"] = color_from_text(text) or "white"

    elif intent == "BRIGHTNESS":
        m = re.search(r"(\d{1,3})\s*(?:%|percent)?", t)
        if m:
            pct = max(0, min(100, int(m.group(1))))
        else:
            w = _word_to_num(re.search(r"(twenty|thirty|forty|fifty|sixty|"
                                        r"seventy|eighty|ninety|ten|fifteen)", t).group(1)) \
                if re.search(r"(twenty|thirty|forty|fifty|sixty|seventy|"
                              r"eighty|ninety|ten|fifteen)", t) else None
            pct = 100 if w is None else min(100, w)
        slots["percent"] = str(pct)

    elif intent == "TEMPERATURE":
        m = re.search(r"(-?\d{1,3})", t)
        slots["degrees"] = m.group(1) if m else "22"

    elif intent == "VOLUME_SET":
        m = re.search(r"(\d{1,3})", t)
        if m:
            slots["percent"] = str(max(0, min(100, int(m.group(1)))))

    elif intent == "TIMER":
        m = re.search(r"(\d+)\s*(sec|second|min|minute|m\b|s\b)", t)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            secs = val if unit.startswith("s") else val * 60
        else:
            secs = 60
        slots["duration"] = f"{secs}s"

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
            slots["time"] = f"{h:02d}:{int(mm):02d} {ap.upper()}"
        else:
            slots["time"] = "08:00 AM"

    elif intent == "CALL":
        slots["contact"] = (contact_from_text(text)
                            or _fuzzy(t, CONTACTS_DEFAULT) or "unknown")

    elif intent == "MESSAGE":
        slots["contact"] = (contact_from_text(text)
                            or _fuzzy(t, CONTACTS_DEFAULT) or "unknown")

    elif intent == "CREATE_REMINDER":
        m = re.search(r"remind me to (.+)$", t.strip())
        slots["task"] = m.group(1).strip() if m else "something"

    return slots
