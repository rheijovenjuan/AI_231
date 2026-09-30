"""Label spaces for the Rapi voice-command model.

Two label spaces are defined:
  * INTENTS  - the 19 user-facing command classes (the model output)
  * SLOTS    - the value that follows TIMER / ALARM / TEMPERATURE /
               BRIGHTNESS / COLOR / CREATE_REMINDER, plus NONE
"""

from __future__ import annotations

INTENTS = [
    "PLAY_MUSIC",
    "WEATHER",
    "TIME",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "CALL",
    "MESSAGE",
    "LIST_REMINDERS",
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
]

# Intents whose utterance carries a trailing value that must be recognised.
SLOT_INTENTS = [
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
]

SLOT_VALUES = {
    "TIMER": ["10 seconds", "30 seconds", "1 minute"],
    "ALARM": ["4 AM", "8 AM", "9 PM"],
    "TEMPERATURE": ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS": ["20 percent", "60 percent", "100 percent"],
    "COLOR": ["red", "blue", "green", "yellow"],
    "CREATE_REMINDER": ["drink water", "study", "exercise"],
}

NONE = "NONE"

INTENT_TO_IDX = {n: i for i, n in enumerate(INTENTS)}
IDX_TO_INTENT = {i: n for n, i in INTENT_TO_IDX.items()}

# Slot class list. Order is stable so the ONNX head stays compatible.
SLOT_CLASSES = [v for intent in SLOT_INTENTS for v in SLOT_VALUES[intent]] + [NONE]
SLOT_TO_IDX = {n: i for i, n in enumerate(SLOT_CLASSES)}
IDX_TO_SLOT = {i: n for n, i in SLOT_TO_IDX.items()}


def folder_to_intent(folder: str) -> str:
    """`TIMER_10s` -> `TIMER`, `PLAY_MUSIC` -> `PLAY_MUSIC`."""
    if folder in INTENT_TO_IDX:
        return folder
    for intent in SLOT_INTENTS:
        if folder.startswith(intent + "_"):
            return intent
    raise KeyError(f"cannot map folder {folder!r} to an intent")


def intent_has_slot(intent: str) -> bool:
    return intent in SLOT_INTENTS
