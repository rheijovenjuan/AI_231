"""Wake-word detector for "Hey Rapi".

Two-stage design that runs comfortably on a Raspberry Pi 4 (CPU only):

Stage 1 -- energy / spectral gate (very cheap, runs every frame).
    Rejects silence and most non-speech audio before the expensive model.

Stage 2 -- a Gaussian Mixture Model (GMM) over MFCC features, trained on
    positive examples ("Hey Rapi") vs. negative examples (everything else +
    silence). This is the classic, tiny-footprint wake-word architecture
    (same family as the original Google Assistant / Porcupine approach).

The model is a plain ``sklearn`` GMM serialised with ``joblib`` so it loads in
milliseconds and uses a few MB of RAM -- ideal for a 4 GB Pi.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from . import features as F

# Gate thresholds (tuned for 16 kHz, ~30 ms frames).
FRAME_MS = 30
GATE_ENERGY_MIN = 5e-5      # below this RMS^2 -> silence (softened for real mics)
GATE_SPECTRAL_MIN = 0.02    # below this spectral flux -> not speech-like
GATE_REL_FACTOR = 9.0       # a frame must be this many x the noise-floor power
                            # to count as speech (rejects steady room noise)
GATE_MOD_MIN = 0.15         # min coefficient-of-variation of recent energy to
                            # count as speech (steady noise has CV ~0.07, speech
                            # ~0.45+). Requires GATE_MOD_WINDOW frames of history.
GATE_MOD_WINDOW = 12        # ~0.36 s of recent frames used for the modulation

# Automatic gain control. The wake model was trained on synthetic TTS that
# sits near a 0.5 peak (RMS^2 ~ 1e-2). A real microphone is far quieter
# (RMS^2 ~ 1e-4..3e-4), so without normalisation the energy gate drops most
# of the speech and the wake word never fires. We track a slow envelope of
# recent speech energy and scale each incoming chunk up to TARGET_PEAK.
AGC_TARGET_RMS = 0.08       # normalise speech to this RMS (synthetic TTS level)
AGC_SMOOTH = 0.5            # per-chunk smoothing (tracks level both ways, fast)
AGC_NOISE_ATTACK = 0.3      # floor drops quickly toward quiet frames
AGC_NOISE_RELEASE = 0.02    # floor creeps up slowly on loud frames
AGC_SNR_MIN_DB = 12.0       # only amplify when signal is this far above noise
AGC_MIN_REF = 5e-4          # floor so we don't divide by ~zero
AGC_MAX_GAIN_DB = 30.0      # clamp gain so we never amplify pure noise wildly


class WakeWordDetector:
    def __init__(self, model_path: str, n_components: int = 16):
        self.model_path = model_path
        self.n_components = n_components
        self.pos_model = None   # GMM for "Hey Rapi"
        self.neg_model = None   # GMM for everything else
        self.frame = int(F.SAMPLE_RATE * FRAME_MS / 1000)
        self.buf = np.zeros(self.frame)
        self.buf_i = 0
        self.prev_power = None
        # AGC state (see module constants above).
        self._agc_env = 0.0          # smoothed current speech RMS
        self._noise_floor = 0.0      # learned ambient noise RMS
        self._n_frames = 0
        # Rolling energy history for the modulation (speech-likeness) gate.
        self._energy_hist: List[float] = []

    # -- persistence ------------------------------------------------------- #
    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        import joblib

        blob = joblib.load(self.model_path)
        self.pos_model = blob["pos"]
        self.neg_model = blob["neg"]
        return True

    def save(self, pos_feats: np.ndarray, neg_feats: np.ndarray) -> str:
        from voice_assistant.gmm import GMM

        self.pos_model = GMM(n_components=self.n_components,
                             reg_covar=1e-3, n_init=2, random_state=0).fit(pos_feats)
        self.neg_model = GMM(n_components=self.n_components,
                             reg_covar=1e-3, n_init=2, random_state=0).fit(neg_feats)
        import joblib

        joblib.dump({"pos": self.pos_model, "neg": self.neg_model},
                    self.model_path)
        return self.model_path

    # -- streaming helpers ------------------------------------------------- #
    def _update_noise(self, rms: float) -> None:
        """Adaptively track the ambient-noise floor (RMS).

        The floor always drifts toward the QUIETEST recent energy (so it
        represents room noise, not speech) and creeps up slowly when things
        stay loud. Because it chases the minimum, steady room noise settles
        the floor onto itself (so the relative gate rejects it), while a
        speech burst sits well above it (so it passes the gate).
        """
        self._n_frames += 1
        # Quiet frames pull the floor down toward the current level (fast).
        if rms < self._noise_floor:
            self._noise_floor = (AGC_NOISE_ATTACK * rms
                                 + (1 - AGC_NOISE_ATTACK) * self._noise_floor)
        else:
            # Loud frames let the floor creep up only very slowly, so a brief
            # speech burst doesn't inflate the "noise" estimate.
            self._noise_floor = (AGC_NOISE_RELEASE * rms
                                 + (1 - AGC_NOISE_RELEASE) * self._noise_floor)

    def _agc(self, chunk: np.ndarray) -> np.ndarray:
        """Normalise a chunk to the training loudness (see AGC_* constants).

        Tracks a slow ambient-noise floor and a smoothed speech envelope, then
        scales the chunk so its RMS lands on AGC_TARGET_RMS -- but only when
        the signal is clearly above the noise floor (SNR >= AGC_SNR_MIN_DB).
        This lifts a quiet real microphone into the range the synthetic wake
        model was trained on, WITHOUT amplifying steady room noise into
        garbage that would false-trigger the GMM.
        """
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
        self._update_noise(rms)
        # Speech envelope: smooth the CURRENT level (tracks both ways fast).
        self._agc_env = AGC_SMOOTH * rms + (1 - AGC_SMOOTH) * self._agc_env

        ref = max(self._agc_env, AGC_MIN_REF)
        if ref < 1e-4:
            return chunk  # silence: nothing to normalise
        # SNR guard: only amplify when signal clearly beats the noise floor.
        snr_db = 20.0 * float(np.log10(ref / max(self._noise_floor, 1e-6)))
        if snr_db < AGC_SNR_MIN_DB:
            return chunk  # steady noise: pass through unamplified
        gain = AGC_TARGET_RMS / ref
        gain = float(np.clip(gain, 1.0, 10 ** (AGC_MAX_GAIN_DB / 20.0)))
        return chunk * gain

    def push(self, chunk: np.ndarray) -> List[np.ndarray]:
        """Feed a chunk of mono samples; return list of full frames."""
        frames = []
        chunk = self._agc(chunk)
        self.buf = np.concatenate([self.buf[self.buf_i:], chunk])
        self.buf_i = 0
        while len(self.buf) >= self.frame:
            fr = self.buf[: self.frame].copy()
            self.buf = self.buf[self.frame:]
            frames.append(fr)
        return frames

    def _gate(self, frame: np.ndarray) -> bool:
        e = float(np.mean(frame ** 2))
        # Keep a rolling energy history for the modulation check.
        self._energy_hist.append(e)
        if len(self._energy_hist) > GATE_MOD_WINDOW:
            self._energy_hist.pop(0)
        # Absolute silence floor.
        if e < GATE_ENERGY_MIN:
            self.prev_power = None
            return False
        # Relative floor: reject frames not clearly louder than ambient noise.
        nf = self._noise_floor
        if nf > 1e-5 and e < (nf * nf) * GATE_REL_FACTOR:
            self.prev_power = None
            return False
        # Modulation: steady room noise has near-constant energy (low CV);
        # speech has syllabic variation (high CV). Require enough history.
        if len(self._energy_hist) >= 6:
            hist = np.asarray(self._energy_hist, dtype=float)
            mean = float(np.mean(hist))
            if mean > 1e-7:
                cv = float(np.std(hist) / mean)
                if cv < GATE_MOD_MIN:
                    self.prev_power = None
                    return False
        power = np.abs(np.fft.rfft(frame * np.hamming(len(frame)))) ** 2
        if self.prev_power is not None:
            flux = float(np.mean(np.abs(power - self.prev_power)))
        else:
            flux = e
        self.prev_power = power
        return flux > GATE_SPECTRAL_MIN

    def _score(self, feats: np.ndarray) -> float:
        """Log-likelihood ratio  P(x|wake) - P(x|other). Higher = more wake."""
        feats = np.atleast_2d(feats)
        return float(self.pos_model.score(feats) - self.neg_model.score(feats))

    def detect_window(self, x: np.ndarray) -> float:
        """Score a candidate utterance (mono, 16 kHz).

        Returns a wake-score (higher = more likely "Hey Rapi").

        NOTE: the model is trained on WHOLE short utterances (full-clip MFCC
        stats + a duration feature), so we must score the whole candidate --
        not a sub-window. Scoring an arbitrary sub-clip puts the features
        (especially duration) far out of the training distribution. We only
        trim leading/trailing silence and cap the length to a sane wake-word
        bound (~1.5 s).
        """
        if len(x) < int(0.25 * F.SAMPLE_RATE):
            return -np.inf
        # Trim leading/trailing silence (keep the voiced core).
        frame = int(0.02 * F.SAMPLE_RATE)
        if len(x) > frame:
            rms = [float(np.mean(x[i:i + frame] ** 2))
                   for i in range(0, len(x) - frame + 1, frame)]
            thr = max(1e-5, 0.02 * float(np.max(rms)))
            voiced = [i * frame for i, e in enumerate(rms) if e >= thr]
            if voiced:
                lo, hi = voiced[0], voiced[-1] + frame
                x = x[max(0, lo - frame): min(len(x), hi + frame)]
        # Cap to a reasonable wake-word length (take the loudest 1.5 s).
        cap = int(1.5 * F.SAMPLE_RATE)
        if len(x) > cap:
            win = int(0.1 * F.SAMPLE_RATE)
            n_win = 1 + (len(x) - win) // win
            energies = [float(np.mean(x[i * win:(i + 1) * win] ** 2))
                        for i in range(n_win)]
            best = int(np.argmax(energies))
            s = max(0, best * win - cap // 2)
            x = x[s:s + cap]
        if len(x) < int(0.25 * F.SAMPLE_RATE):
            return -np.inf
        feats = F.extract_features(x)
        return self._score(feats)

    def detect_stream(self, frames: List[np.ndarray],
                      threshold: float) -> bool:
        """Given gated speech frames accumulated since the last reset, decide.

        Scores a SLIDING WINDOW of the most recent ~0.8 s of speech (plus a
        little trailing silence) rather than the whole accumulated prefix.
        The model was trained on whole short utterances, so a fixed, complete
        window scores far more reliably than a growing prefix -- especially at
        low microphone levels where the prefix starts out too short/weak.
        """
        # Two guards against the startup transient (a few leaked noise frames
        # in the first ~0.3 s while the noise floor is still learning):
        #   1) don't fire until the detector has been running ~0.4 s, and
        #   2) require a minimum of ~0.3 s of sustained gated speech.
        settle_frames = int(0.6 * F.SAMPLE_RATE / self.frame)
        if self._n_frames < settle_frames:
            return False
        min_frames = int(0.3 * F.SAMPLE_RATE / self.frame)
        if len(frames) < min_frames:
            return False
        # Window: last ~0.8 s of speech (27 frames @ 30 ms) + trailing tail.
        win = frames[-27:]
        x = np.concatenate(win)
        return self.detect_window(x) >= threshold
