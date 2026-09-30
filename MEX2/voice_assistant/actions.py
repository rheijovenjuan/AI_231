"""Action dispatcher -- turns (intent, slots) into real side-effects.

Everything is UI-agnostic: the dispatcher talks to a small ``UIBackend``
interface so the same logic drives both the Tkinter GUI and a headless CLI.
Music playback uses a background thread so a new command never interrupts an
already-playing track (requirement: "when a new query is done the music should
still be playing").
"""

from __future__ import annotations

import datetime as dt
import os
import random
import re
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Music player (thread-safe, non-blocking)
# --------------------------------------------------------------------------- #
class MusicPlayer:
    DUCK_FACTOR = 0.30        # volume multiplier while the assistant listens

    def __init__(self, music_dir: str):
        self.music_dir = music_dir
        self._player = None
        self._lock = threading.RLock()   # reentrant: play() nests in next()
        self.state = "stopped"          # stopped | playing | paused
        self.current: Optional[str] = None
        self._last_track: Optional[str] = None   # for stop -> play restart
        self.volume = 0.7
        self._ducked = False            # listening: music kept at low volume
        self._track_idx = 0
        self.tracks = self._scan()

    def _scan(self) -> List[str]:
        if not os.path.isdir(self.music_dir):
            return []
        exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
        return sorted(
            os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir)
            if f.lower().endswith(exts))

    def _ensure_player(self):
        """Lazy-import a backend so the Pi only pays for what it uses."""
        if self._player is None:
            try:
                import pygame
                # NB: the kwarg is `frequency` - `freq` raises TypeError
                # and used to drop us silently onto the winsound fallback
                # (which cannot play the mp3s in the music folder).
                pygame.mixer.init(frequency=16000, size=-16, channels=1,
                                  buffer=512)
                self._player = pygame
            except Exception as exc:  # noqa: BLE001
                print(f"[music] pygame unavailable ({exc}); "
                      f"falling back to winsound (wav only)")
                self._player = "winsound"
        return self._player

    def _load_current(self):
        if not self.tracks:
            return None
        path = self.tracks[self._track_idx % len(self.tracks)]
        self.current = os.path.basename(path)
        return path

    def play(self):
        with self._lock:
            if self.state == "playing":
                return          # already going: a new query must not interrupt
            if self.state == "paused":
                # resume WHERE IT WAS PAUSED - not from the beginning
                try:
                    if isinstance(self._player, str):
                        self._start_locked()   # winsound cannot unpause
                    else:
                        self._player.mixer.music.unpause()
                    self.state = "playing"
                except Exception as e:      # noqa: BLE001
                    print("[music] resume error:", e)
                return
            # stopped (or first run): re-scan every start so whatever is in
            # the folder right now is the playlist
            self.tracks = self._scan()
            if not self.tracks:
                self.state = "stopped"
                self.current = None
                return
            if self._last_track:
                # stopped earlier: restart THAT track from the beginning
                names = [os.path.basename(p) for p in self.tracks]
                self._track_idx = (names.index(self._last_track)
                                   if self._last_track in names
                                   else random.randrange(len(self.tracks)))
            else:
                # nothing played yet: take any file from the folder
                self._track_idx = random.randrange(len(self.tracks))
            self._start_locked()

    def _start_locked(self):
        """Start ``tracks[_track_idx]``. Caller must hold the lock."""
        path = self._load_current()
        if path is None:
            return
        self._last_track = self.current   # remember for stop -> play
        self._ensure_player()
        try:
            if isinstance(self._player, str):
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME |
                                   winsound.SND_ASYNC)
            else:
                self._player.mixer.music.load(path)
                self._apply_volume()
                self._player.mixer.music.play(loops=-1)
            self.state = "playing"
        except Exception as e:  # pragma: no cover
            self.state = "error"
            print("[music] playback error:", e)

    # -- ducking (wake word fired: keep the song going, just quieter) ------ #
    def duck(self):
        with self._lock:
            if self._ducked or self.state != "playing":
                return
            self._ducked = True
            self._apply_volume()

    def unduck(self):
        with self._lock:
            if not self._ducked:
                return
            self._ducked = False
            self._apply_volume()

    def _apply_volume(self):
        if self._player is None:
            return
        if isinstance(self._player, str):
            if not getattr(self, "_vol_warned", False):
                self._vol_warned = True
                print("[music] volume control unavailable (winsound "
                      "fallback) - install pygame for volume/ducking")
            return          # winsound has no volume API
        v = self.volume * (self.DUCK_FACTOR if self._ducked else 1.0)
        try:
            self._player.mixer.music.set_volume(v)
        except Exception:
            pass

    def pause(self):
        with self._lock:
            if self.state != "playing":
                return
            if isinstance(self._player, str):
                import winsound     # no native pause: stop (restarts later)
                winsound.PlaySound(None, winsound.SND_ASYNC)
            else:
                self._player.mixer.music.pause()
            self.state = "paused"

    def resume(self):
        with self._lock:
            if self.state != "paused":
                return
            if isinstance(self._player, str):
                self._start_locked()
            else:
                self._player.mixer.music.unpause()
            self.state = "playing"

    def stop(self):
        with self._lock:
            try:
                if isinstance(self._player, str):
                    import winsound
                    winsound.PlaySound(None, winsound.SND_ASYNC)
                else:
                    self._player.mixer.music.stop()
            except Exception:
                pass
            self.state = "stopped"
            self.current = None

    def next(self):
        with self._lock:
            self.tracks = self._scan() or self.tracks
            self._track_idx += 1
            if self.tracks:
                self._start_locked()

    def volume_up(self):
        with self._lock:
            self.volume = min(1.0, self.volume + 0.1)
            self._apply_volume()

    def volume_down(self):
        with self._lock:
            self.volume = max(0.0, self.volume - 0.1)
            self._apply_volume()

    def set_volume(self, percent: int):
        with self._lock:
            self.volume = max(0.0, min(1.0, percent / 100.0))
            self._apply_volume()


# --------------------------------------------------------------------------- #
# Weather (offline-friendly, no API key required)
# --------------------------------------------------------------------------- #
def get_weather() -> str:
    """Return a short weather string.

    Tries the free, key-less Open-Meteo API for Manila (the likely locale).
    Falls back to a deterministic placeholder so the demo always works offline.
    """
    import urllib.request
    import json

    try:
        url = ("https://api.open-meteo.com/v1/forecast?latitude=14.5995"
               "&longitude=120.9842&current=temperature_2m,relative_humidity_2m,"
               "apparent_temperature,weather_code,wind_speed_10m")
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.load(r)
        cur = data["current"]
        desc = _wmo_desc(cur["weather_code"])
        return (f"Today: {desc}, {cur['temperature_2m']}°C "
                f"(feels {cur['apparent_temperature']}°C), "
                f"humidity {cur['relative_humidity_2m']}%, "
                f"wind {cur['wind_speed_10m']} km/h")
    except Exception:
        return "Weather service unavailable (offline). Clear skies, 28°C."


def _wmo_desc(code: int) -> str:
    return {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
            3: "Overcast", 45: "Fog", 48: "Rime fog", 51: "Light drizzle",
            53: "Drizzle", 55: "Heavy drizzle", 61: "Light rain",
            63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow",
            80: "Rain showers", 95: "Thunderstorm"}.get(code, "Unknown")


def _secs_until(alarm_time: str) -> Optional[float]:
    """Seconds from now until the next occurrence of ``HH:MM AM/PM``.

    Returns None when the string cannot be parsed (no timer is scheduled
    then - the spoken confirmation still happens).
    """
    m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)", str(alarm_time), re.I)
    if not m:
        return None
    h = int(m.group(1)) % 12
    if m.group(3).lower() == "pm":
        h += 12
    now = dt.datetime.now()
    target = now.replace(hour=h, minute=int(m.group(2)), second=0,
                          microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)     # already past: tomorrow
    return (target - now).total_seconds()


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
class ActionDispatcher:
    def __init__(self, music_dir: str, on_event: Optional[Callable] = None):
        self.music = MusicPlayer(music_dir)
        self.on_event = on_event or (lambda **_: None)
        self.lights = {"on": False, "brightness": 100, "color": "white"}
        self.thermostat = 24
        self.reminders: List[str] = []
        self._timers: Dict[str, threading.Timer] = {}
        self.alarm: Optional[str] = None
        self._alarm_timer: Optional[threading.Timer] = None

    def _emit(self, **kw):
        self.on_event(**kw)

    def dispatch(self, intent: str, slots: Dict[str, str]) -> str:
        handler = getattr(self, f"_do_{intent.lower()}", None)
        if handler is None:
            msg = f"I don't know how to '{intent}' yet."
            self._emit(status=msg)
            return msg
        msg = handler(slots)
        # Echo the command status once; the Assistant's final event carries
        # result/confidence/keywords, so logging both would duplicate lines.
        self._emit(status=msg)
        return msg

    # -- music ------------------------------------------------------------- #
    def _do_play_music(self, s):
        self.music.play()
        if self.music.state == "playing":
            msg = f"Playing {self.music.current}"
        elif not self.music.tracks:
            msg = "No tracks in the music folder"
        else:
            msg = "Playback error"
        self._emit(music={"state": self.music.state,
                          "track": self.music.current})
        return msg

    def _do_pause(self, s):
        self.music.pause()
        self._emit(music={"state": self.music.state,
                          "track": self.music.current})
        return "Paused"

    def _do_stop(self, s):
        self.music.stop()
        self._emit(music={"state": self.music.state, "track": None})
        return "Stopped"

    def _do_next(self, s):
        self.music.next()
        if self.music.state == "playing":
            msg = f"Now playing {self.music.current}"
        elif not self.music.tracks:
            msg = "No tracks in the music folder"
        else:
            msg = "Playback error"
        self._emit(music={"state": self.music.state,
                          "track": self.music.current})
        return msg

    def _do_volume_up(self, s):
        self.music.volume_up()
        self._emit(music={"volume": self.music.volume})
        return f"Volume up ({int(self.music.volume*100)}%)"

    def _do_volume_down(self, s):
        self.music.volume_down()
        self._emit(music={"volume": self.music.volume})
        return f"Volume down ({int(self.music.volume*100)}%)"

    def _do_volume_set(self, s):
        raw = str(s.get("percent", "")).rstrip("%")
        try:
            pct = max(0, min(100, int(raw)))
        except ValueError:
            pct = int(self.music.volume * 100)   # no number: echo current
        self.music.set_volume(pct)
        self._emit(music={"volume": self.music.volume})
        return f"Volume set to {pct}%"

    # -- info -------------------------------------------------------------- #
    def _do_weather(self, s):
        msg = get_weather()
        self._emit(weather=msg)
        return msg

    def _do_time(self, s):
        now = dt.datetime.now().strftime("%I:%M %p on %A, %B %d")
        self._emit(time=now)
        return f"It's {now}"

    # -- lights ------------------------------------------------------------ #
    def _do_light_on(self, s):
        self.lights["on"] = True
        self._emit(light=dict(self.lights))
        return "Lights on"

    def _do_light_off(self, s):
        self.lights["on"] = False
        self._emit(light=dict(self.lights))
        return "Lights off"

    def _do_brightness(self, s):
        raw = s.get("percent")
        if raw in (None, ""):
            # "dim (the) lights" without a number: step the level down
            pct = max(10, int(self.lights.get("brightness", 100)) - 30)
        else:
            pct = max(0, min(100, int(raw)))
        self.lights["on"] = True
        self.lights["brightness"] = pct
        self._emit(light=dict(self.lights))
        return f"Lights dimmed to {pct}%"

    def _do_color(self, s):
        col = s.get("color", "white")
        self.lights["on"] = True
        self.lights["color"] = col
        self._emit(light=dict(self.lights))
        return f"Light color set to {col}"

    # -- thermostat -------------------------------------------------------- #
    def _do_temperature(self, s):
        deg = int(s.get("degrees", 22))
        self.thermostat = deg
        self._emit(thermostat=deg)
        return f"Adjusted temperature to {deg} degrees"

    # -- reminders --------------------------------------------------------- #
    def _do_create_reminder(self, s):
        task = s.get("task", "something")
        self.reminders.append(task)
        self._emit(reminder={"added": task, "all": list(self.reminders)})
        return f"Reminder set: {task}"

    def _do_list_reminders(self, s):
        msg = "Your reminders: " + (", ".join(self.reminders)
                                    if self.reminders else "(none)")
        self._emit(reminder={"all": list(self.reminders)})
        return msg

    # -- timers / alarms --------------------------------------------------- #
    def _do_timer(self, s):
        secs = int(s.get("duration", "60s").rstrip("s"))
        name = f"timer-{time.time()}"
        self._timers[name] = threading.Timer(
            secs, lambda: self._emit(timer_done=f"{secs}s timer finished"))
        self._timers[name].daemon = True    # never block app exit
        self._timers[name].start()
        self._emit(timer={"started": secs})
        return f"Timer set for {secs} seconds"

    def _do_alarm(self, s):
        t = s.get("time", "08:00 AM")
        self.alarm = t
        if self._alarm_timer is not None:
            self._alarm_timer.cancel()      # re-setting replaces the old one
        secs = _secs_until(t)
        if secs is not None:
            self._alarm_timer = threading.Timer(
                secs, lambda: self._emit(alarm_done=f"Alarm! It's {t}"))
            self._alarm_timer.daemon = True
            self._alarm_timer.start()
        self._emit(alarm={"set": t})
        return f"Alarm set for {t}"

    # -- communication ----------------------------------------------------- #
    def _do_call(self, s):
        who = s.get("contact", "unknown")
        if who == "unknown":
            msg = "Who should I call?"
            self._emit(call=msg)
            return msg
        msg = f"Calling {who.title()}"
        self._emit(call=msg)
        return msg

    def _do_message(self, s):
        who = s.get("contact", "unknown")
        if who == "unknown":
            msg = "Who should I message?"
            self._emit(message=msg)
            return msg
        msg = f"Messaging {who.title()}"
        self._emit(message=msg)
        return msg
