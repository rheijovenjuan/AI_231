"""End-to-end runtime pipeline: wake word -> command class (+ slot).

    from runtime.pipeline import VoicePipeline
    p = VoicePipeline("output/onnx")
    print(p.scan(waveform_16k))      # full record: wake detection + class

Only numpy / scipy / onnxruntime are required, so the same code runs on a
laptop and on a Raspberry Pi 4.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from rapi_vcm import features as F
from rapi_vcm.keywords import KEYWORDS as _KEYWORDS
from rapi_vcm.keywords import keywords_to_intent
from rapi_vcm.labels import INTENTS as _INTENTS
from rapi_vcm.labels import SLOT_INTENTS as _SLOT_INTENTS
from rapi_vcm.labels import SLOT_VALUES as _SLOT_VALUES

_SLOT_VALUE_TO_INTENT = {v: k for k, vals in _SLOT_VALUES.items()
                         for v in vals}


class VoicePipeline:
    def __init__(self, model_dir: str, wake_threshold: float | None = None,
                 providers=None, threads: int = 0, cmd_seconds: float | None = None,
                 wake_seconds: float | None = None):
        import onnxruntime as ort

        onnx_dir = model_dir
        card_path = os.path.join(model_dir, "model_card.json")
        if not os.path.exists(os.path.join(model_dir, "command.onnx")) \
                and not os.path.exists(os.path.join(model_dir, "keyword.onnx")):
            onnx_dir = os.path.join(model_dir, "onnx")
            card_path = os.path.join(onnx_dir, "model_card.json")

        self.onnx_dir = onnx_dir
        self.card = {}
        if os.path.exists(card_path):
            with open(card_path, encoding="utf-8") as fh:
                self.card = json.load(fh)

        cfg = self.card.get("feature_config", {})
        self.sr = int(cfg.get("sample_rate", F.SAMPLE_RATE))
        self.cmd_seconds = float(cmd_seconds or cfg.get("cmd_seconds", F.CMD_SECONDS))
        self.wake_seconds = float(wake_seconds or cfg.get("wake_seconds", F.WAKE_SECONDS))
        self.cmd_frames = int(self.cmd_seconds * self.sr / F.HOP_LENGTH)
        self.wake_frames = int(self.wake_seconds * self.sr / F.HOP_LENGTH)
        self.intents = self.card.get("intents") or _default_intents()
        self.slot_classes = self.card.get("slot_classes") or []
        self.threshold = float(wake_threshold if wake_threshold is not None
                               else self.card.get("wake_threshold", 0.5))

        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        prov = providers or ["CPUExecutionProvider"]
        self.wake_sess = ort.InferenceSession(
            os.path.join(onnx_dir, "wake.onnx"), so, providers=prov)

        kw_path = os.path.join(onnx_dir, "keyword.onnx")
        if os.path.exists(kw_path):
            self.head = "keyword"
            self.kw_sess = ort.InferenceSession(kw_path, so, providers=prov)
            self.cmd_sess = None
            self.keywords = self.card.get("keywords") or list(_KEYWORDS)
            self.keyword_threshold = float(self.card.get("keyword_threshold",
                                                          0.5))
            thrs = self.card.get("keyword_thresholds")
            if not isinstance(thrs, list) or len(thrs) != len(self.keywords):
                thrs = None
            self.keyword_thresholds = np.asarray(
                thrs if thrs is not None
                else [self.keyword_threshold] * len(self.keywords),
                dtype=np.float64)
        else:
            self.head = "intent_softmax"
            self.kw_sess = None
            self.cmd_sess = ort.InferenceSession(
                os.path.join(onnx_dir, "command.onnx"), so, providers=prov)

        self.stats = {"wake_runs": 0, "cmd_runs": 0,
                      "wake_ms": [], "cmd_ms": []}

    # ------------------------------------------------------------- features --
    def _feats(self, wave: np.ndarray, frames: int) -> np.ndarray:
        x = np.asarray(wave, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        f = F.cmvn(F.logmel(x, self.sr))
        return F.pad_crop(f, frames)[None, None, ...].astype(np.float32)

    # ----------------------------------------------------------------- runs --
    def wake_score(self, wave: np.ndarray) -> float:
        x = self._feats(wave, self.wake_frames)
        t0 = time.perf_counter()
        out = self.wake_sess.run(None, {"features": x})[0]
        self.stats["wake_runs"] += 1
        self.stats["wake_ms"].append((time.perf_counter() - t0) * 1000.0)
        return float(1.0 / (1.0 + np.exp(-float(out.reshape(-1)[0]))))

    def classify(self, wave: np.ndarray) -> dict:
        x = self._feats(wave, self.cmd_frames)
        t0 = time.perf_counter()
        if self.head == "keyword":
            logits = self.kw_sess.run(None, {"features": x})[0]
            self.stats["cmd_runs"] += 1
            self.stats["cmd_ms"].append((time.perf_counter() - t0) * 1000.0)
            return self._from_keywords(logits[0])

        li, ls = self.cmd_sess.run(None, {"features": x})
        self.stats["cmd_runs"] += 1
        self.stats["cmd_ms"].append((time.perf_counter() - t0) * 1000.0)

        prob_i = _softmax(li[0])
        prob_s = _softmax(ls[0])
        i = int(prob_i.argmax())
        intent = self.intents[i]
        result = {
            "intent": intent,
            "confidence": round(float(prob_i[i]), 4),
            "top3": [{"intent": self.intents[int(k)],
                      "confidence": round(float(prob_i[k]), 4)}
                     for k in np.argsort(-prob_i)[:3]],
        }
        # slot value (only meaningful for TIMER..CREATE_REMINDER)
        if self.slot_classes and intent in _SLOT_INTENTS:
            relevant = [j for j, name in enumerate(self.slot_classes)
                        if _SLOT_VALUE_TO_INTENT.get(name) == intent]
            if relevant:
                best_s = max(relevant, key=lambda j: prob_s[j])
                result["slot"] = self.slot_classes[best_s]
                result["slot_confidence"] = round(float(prob_s[best_s]), 4)
        return result

    def _from_keywords(self, raw) -> dict:
        """keyword logits -> active keyword set -> (intent, slot)."""
        prob = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
        n = min(len(prob), len(self.keyword_thresholds))
        active = sorted(
            ((self.keywords[i], float(prob[i]))
             for i in range(n) if float(prob[i]) >= self.keyword_thresholds[i]),
            key=lambda t: -t[1])
        names = [n for n, _ in active]
        intent, slot = keywords_to_intent(names)
        result = {
            "intent": intent,
            "confidence": round(active[0][1] if active
                                else float(prob.max()), 4),
            "keywords": [{"keyword": n, "p": round(p, 4)}
                         for n, p in active],
        }
        if slot is not None:
            want = slot.lower()
            idx = next((i for i, n in enumerate(self.keywords)
                        if n.lower() == want), None)
            name = next((s for s in self.slot_classes
                         if s.lower() == want), None)
            if name is not None:
                result["slot"] = name
                if idx is not None:
                    result["slot_confidence"] = round(float(prob[idx]), 4)
        return result

    # ------------------------------------------------------ full recording ----
    def scan(self, wave: np.ndarray, hop: float = 0.1,
             return_scores: bool = False) -> dict:
        """Slide the wake detector over `wave`, then classify what follows."""
        x = np.asarray(wave, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        wl = int(self.wake_seconds * self.sr)
        step = max(int(hop * self.sr), 1)
        if x.shape[0] < wl:
            x = np.pad(x, (0, wl - x.shape[0]))

        scores = []
        positions = []
        best = (-1.0, 0)
        for start in range(0, x.shape[0] - wl + 1, step):
            sc = self.wake_score(x[start:start + wl])
            scores.append(sc)
            positions.append(start)
            if sc > best[0]:
                best = (sc, start)
            if sc >= self.threshold:
                break

        hit = {"detected": bool(scores) and max(scores) >= self.threshold,
               "wake_score": round(float(max(scores) if scores else 0.0), 4),
               "threshold": self.threshold,
               "wake_offset_sec": round(best[1] / self.sr, 3) if scores else None}
        if not hit["detected"]:
            if return_scores:
                hit["scores"] = [round(s, 4) for s in scores]
            return hit

        # command segment: from the end of the wake word to the end of speech
        cmd_start = best[1] + wl
        seg = x[cmd_start:cmd_start + int(self.cmd_seconds * self.sr)]
        if seg.shape[0] < int(self.cmd_seconds * self.sr):
            seg = np.pad(seg, (0, int(self.cmd_seconds * self.sr) - seg.shape[0]))
        hit.update(self.classify(seg))
        if return_scores:
            hit["scores"] = [round(s, 4) for s in scores]
        return hit


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _default_intents():
    return list(_INTENTS)
