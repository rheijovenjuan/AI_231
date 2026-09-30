"""Tkinter GUI for the Rapi voice assistant.

Shows:
  * a status bar (idle / listening / processing)
  * a "smart light" block that turns on/off, dims, and changes colour
  * a thermostat readout
  * a music-player panel (track, state, volume)
  * a scrolling event log (every command + result)
  * a weather / time / reminder / call readout
  * an Exit button that closes the app

The UI is driven entirely by the dispatcher's ``on_event`` callbacks, so it
works identically whether commands arrive by voice (mic) or from the text
path (file mode / tests).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from typing import Callable, Dict, Optional

COLOR_HEX = {
    "white": "#f5f5f5", "red": "#ff4d4d", "green": "#4dff88",
    "blue": "#4d88ff", "yellow": "#ffe14d", "off": "#2b2b2b",
    "orange": "#ff9d4d", "purple": "#a855f7", "violet": "#b388ff",
    "indigo": "#5c6bc0", "pink": "#ff6fb5", "magenta": "#e14dff",
    "cyan": "#4dd0e1", "teal": "#26a69a", "turquoise": "#40e0d0",
    "lavender": "#c4b5fd", "lime": "#c6ff4d", "gold": "#ffd54f",
    "silver": "#c0c0c0", "bronze": "#cd7f32", "black": "#1a1a1a",
    "brown": "#8d6e63", "gray": "#9e9e9e", "grey": "#9e9e9e",
    "navy": "#1a237e", "maroon": "#800000", "crimson": "#dc143c",
    "beige": "#d9c7a7", "cream": "#fffdd0", "olive": "#808000",
    "coral": "#ff7f50", "salmon": "#fa8072",
}


class RapiUI:
    def __init__(self, root: tk.Tk, on_command: Callable[[str], None]):
        self.root = root
        self.on_command = on_command
        root.title("Rapi - Voice Assistant")
        root.geometry("760x560")
        root.configure(bg="#1e1e1e")
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        pad = {"padx": 10, "pady": 6}
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(fill="x", **pad)
        tk.Label(top, text="RAPI", font=("Segoe UI", 22, "bold"),
                 fg="#4dff88", bg="#1e1e1e").pack(side="left")
        tk.Button(top, text="Exit", width=8, relief="flat", bg="#4a1d1d",
                  fg="#ff8a8a", activebackground="#6b2828",
                  activeforeground="#fff",
                  command=self.root.destroy).pack(side="right")
        self.state_lbl = tk.Label(top, text="idle", font=("Segoe UI", 12),
                                  fg="#aaa", bg="#1e1e1e")
        self.state_lbl.pack(side="right", padx=(0, 10))

        # Left column: light + thermostat
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.pack(side="left", fill="both", expand=True, **pad)

        self.light = tk.Canvas(left, width=160, height=160, bg="#1e1e1e",
                               highlightthickness=0)
        self.light.pack(pady=10)
        self._light_rect = self.light.create_rectangle(
            20, 20, 140, 140, fill=COLOR_HEX["off"], outline="#444", width=3)
        tk.Label(left, text="Smart Light", fg="#ccc", bg="#1e1e1e",
                 font=("Segoe UI", 10)).pack()
        self.light_info = tk.Label(left, text="off", fg="#888", bg="#1e1e1e")
        self.light_info.pack()

        self.thermo_lbl = tk.Label(left, text="Thermostat: 24°C",
                                   fg="#ffe14d", bg="#1e1e1e",
                                   font=("Segoe UI", 14, "bold"))
        self.thermo_lbl.pack(pady=12)

        # Right column: music + info
        right = tk.Frame(self.root, bg="#1e1e1e")
        right.pack(side="right", fill="both", expand=True, **pad)

        tk.Label(right, text="Music", fg="#ccc", bg="#1e1e1e",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.music_track = tk.Label(right, text="(nothing playing)",
                                    fg="#fff", bg="#1e1e1e")
        self.music_track.pack(anchor="w")
        self.music_state = tk.Label(right, text="state: stopped",
                                    fg="#888", bg="#1e1e1e")
        self.music_state.pack(anchor="w")
        self.vol_bar = tk.Canvas(right, width=200, height=14, bg="#1e1e1e",
                                 highlightthickness=0)
        self.vol_bar.pack(anchor="w", pady=(2, 2))
        self._vol_rect = self.vol_bar.create_rectangle(
            0, 0, 0, 14, fill="#4dff88", outline="")
        self.vol_lbl = tk.Label(right, text="volume: 70%", fg="#888",
                                bg="#1e1e1e")
        self.vol_lbl.pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="Info", fg="#ccc", bg="#1e1e1e",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        self.info_lbl = tk.Label(right, text="", fg="#eee", bg="#1e1e1e",
                                 font=("Segoe UI", 14), justify="left",
                                 wraplength=300, anchor="w")
        self.info_lbl.pack(anchor="w", fill="x")
        self.timer_lbl = tk.Label(right, text="", fg="#4dff88", bg="#1e1e1e",
                                  font=("Segoe UI", 16, "bold"), anchor="w")
        self.timer_lbl.pack(anchor="w", fill="x")
        self._timer_left: Optional[int] = None

        # Bottom: event log
        bottom = tk.Frame(self.root, bg="#1e1e1e")
        bottom.pack(fill="both", expand=True, **pad)
        self.log = scrolledtext.ScrolledText(bottom, height=8, bg="#121212",
                                             fg="#ddd", insertbackground="#fff",
                                             font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, side="top")

    # ------------------------------------------------------------------ #
    def log_line(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    # -- event handlers (called by the dispatcher) ---------------------- #
    def on_event(self, **kw):
        # dispatchers may fire from worker threads (timer done, TTS) - hop
        # every event onto the Tk main loop to keep tkinter thread-safe
        self.root.after(0, lambda k=kw: self._apply(k))

    def _apply(self, kw: Dict):
        if "state" in kw:
            self.state_lbl.config(text=kw["state"],
                                  fg="#4dff88" if kw["state"] == "listening"
                                  else "#aaa")
        if "status" in kw:
            self.log_line(f"> {kw['status']}")
        if kw.get("text"):
            self.log_line(f'  "{kw["text"]}"')
        if "result" in kw and kw["result"]:
            self.log_line(f"  [{kw['result']}] conf={kw.get('confidence')} "
                          f"slots={kw.get('slots')}")
        if kw.get("keywords"):
            self.log_line(f"  heard: {', '.join(kw['keywords'])}")
        if "light" in kw:
            self._update_light(kw["light"])
        if "thermostat" in kw:
            self.thermo_lbl.config(text=f"Thermostat: {kw['thermostat']}°C")
        if "music" in kw:
            self._update_music(kw["music"])
        if "weather" in kw:
            self.info_lbl.config(text=kw["weather"])
        if "time" in kw:
            self.info_lbl.config(text=kw["time"])
        if "reminder" in kw:
            r = kw["reminder"]
            self.info_lbl.config(
                text="Reminders: " + (", ".join(r["all"]) if r["all"] else "(none)"))
        if "call" in kw:
            self.info_lbl.config(text=kw["call"])
        if "message" in kw:
            self.info_lbl.config(text=kw["message"])
        if "alarm" in kw:
            self.info_lbl.config(text=f"Alarm set for {kw['alarm']['set']}")
        if "alarm_done" in kw:
            self.log_line(f"!! {kw['alarm_done']}")
            self.info_lbl.config(text=kw["alarm_done"])
        if "timer" in kw:
            self.info_lbl.config(text=f"Timer started ({kw['timer']['started']}s)")
            self._start_countdown(int(kw["timer"]["started"]))
        if "timer_done" in kw:
            self.log_line(f"!! {kw['timer_done']}")
            self.info_lbl.config(text=kw["timer_done"])
            self._timer_left = None
            self.timer_lbl.config(text="0s - done!")

    def _start_countdown(self, secs: int):
        self._timer_left = secs
        self._tick()

    def _tick(self):
        if self._timer_left is None:
            return
        if self._timer_left <= 0:
            self.timer_lbl.config(text="0s - done!")
            self._timer_left = None
            return
        self.timer_lbl.config(text=f"Timer: {self._timer_left}s")
        self._timer_left -= 1
        self.root.after(1000, self._tick)

    def _update_light(self, light: Dict):
        if not light.get("on"):
            col = COLOR_HEX["off"]
            info = "off"
        else:
            base = COLOR_HEX.get(light.get("color", "white"), "#f5f5f5")
            info = f"on · {light.get('color','white')} · {light.get('brightness',100)}%"
            # dim by scaling the colour toward black
            col = _dim(base, light.get("brightness", 100))
        self.light.itemconfig(self._light_rect, fill=col)
        self.light_info.config(text=info)

    def _update_music(self, m: Dict):
        if "track" in m:
            self.music_track.config(
                text=m["track"] or "(nothing playing)")
        if "state" in m:
            self.music_state.config(text=f"state: {m['state']}")
        if "volume" in m:
            w = int(200 * m["volume"])
            self.vol_bar.coords(self._vol_rect, 0, 0, w, 14)
            self.vol_lbl.config(text=f"volume: {int(m['volume'] * 100)}%")


def _dim(hexcolor: str, pct: int) -> str:
    pct = max(0, min(100, pct)) / 100.0
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*pct):02x}{int(g*pct):02x}{int(b*pct):02x}"
