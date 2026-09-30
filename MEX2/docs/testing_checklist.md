# Testing checklist

Use this to verify the assistant works end-to-end, on a PC and on the Pi.

## A. Build & train
- [ ] `pip install -r requirements.txt` succeeds (CPU only).
- [ ] `python train.py --download --max-per-intent 300` completes.
- [ ] `models/commands.joblib` and `models/wakeword.joblib` exist.
- [ ] `reports/accuracy_summary.txt` shows overall accuracy ≥ 85%.
- [ ] `reports/command_confusion.png` and `wakeword_scores.png` are generated.

## B. Inference (no mic)
- [ ] `python app.py --mode file --file data/PLAY_MUSIC/...wav` → intent
      `PLAY_MUSIC`, status "Playing …".
- [ ] Repeat for WEATHER, TIME, LIGHT_ON, LIGHT_OFF, STOP, COLOR, TEMPERATURE.
- [ ] Weather returns a real forecast (online) or the offline fallback.
- [ ] Time returns the current time.

## C. Actions / UI
- [ ] `python app.py --mode gui` opens the window.
- [ ] Click **Play** → music panel shows a track + "playing"; **Pause** →
      "paused"; **Stop** → "stopped". A *new* command does not kill playback
      until you explicitly Stop.
- [ ] **Lights On/Off** toggles the light block; **Dim 50%** halves its
      brightness; **Red/Green/Blue** changes its colour.
- [ ] **Temp 22** → "Thermostat: 22°C" and log "adjusted temperature to 22 degrees".
- [ ] **Remind** → log + info panel show the reminder; **list** shows it.
- [ ] **Timer 30s** → starts; after 30 s the log shows "timer finished" **and
      two beeps play**.
- [ ] **Alarm 8am** → "Alarm set for 08:00 AM"; when it fires: log "Alarm!"
      **and two beeps**.
- [ ] **Call mom** / **Msg dad** → info panel "Calling mom" / "Messaging dad".

## D. Slots
- [ ] "brightness 60 percent" → percent=60.
- [ ] "color blue" → color=blue.
- [ ] "temperature 26 degrees" → degrees=26.
- [ ] "timer 1 minute" → duration=60s.
- [ ] "alarm 9 pm" → time=09:00 PM.
- [ ] "call james" → contact=james.

## E. Live mic (Pi / real demo)
- [ ] `python app.py --mode mic --gui` starts; mic is detected.
- [ ] Saying "Hey Rapi" flips state to **listening** ("Yes? I'm listening.").
- [ ] A following command is classified and the UI updates.
- [ ] Silence times out back to **idle** after ~6 s.
- [ ] No false wake on random talk (watch the log for a few minutes).

## F. Benchmarks (PC vs Pi)
- [ ] `python benchmark.py --n 200` on the PC → save `reports/benchmark.txt`.
- [ ] Same on the Pi 4 → compare:
    - [ ] feature extraction < ~250 ms
    - [ ] total latency < ~500 ms
    - [ ] steady RAM < 512 MB
    - [ ] accuracy within a few points of the PC number

## G. Edge cases
- [ ] Very quiet command still detected (lower energy threshold if needed).
- [ ] Long command (>6 s) is cut off gracefully.
- [ ] Unknown/garbled audio → "I don't know how to …" (no crash).
- [ ] Repeated commands don't leak memory (RAM flat over 100 commands).
