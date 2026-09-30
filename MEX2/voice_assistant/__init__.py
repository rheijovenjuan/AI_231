"""Rapi - a lightweight, Raspberry-Pi-friendly voice command assistant.

Pipeline:
    mic -> VAD -> wake-word detector ("Hey Rapi") -> command classifier
        -> slot parser -> action dispatcher -> Tkinter UI

Designed to run on a Raspberry Pi 4 (4 GB) on CPU only.
"""

__version__ = "1.0.0"
WAKE_WORD = "Hey Rapi"
