"""End-to-end accuracy check: wake word + command against ground_truth.json.

    python test_clips/check_end2end.py --model output/onnx

Prints wake detection rate, intent accuracy, slot accuracy and per-clip
failures. Exit code 0 if intent accuracy >= --min-acc (default 0.9).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F              # noqa: E402
from runtime.pipeline import VoicePipeline      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED = os.path.join(HERE, "combined")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/onnx")
    ap.add_argument("--dir", default=COMBINED)
    ap.add_argument("--truth", default=os.path.join(COMBINED, "ground_truth.json"))
    ap.add_argument("--min-acc", type=float, default=0.9)
    args = ap.parse_args()

    with open(args.truth, encoding="utf-8") as fh:
        truth = json.load(fh)

    pipe = VoicePipeline(args.model)
    ok_wake = ok_intent = ok_slot = n = n_slot = 0
    for name in sorted(truth):
        path = os.path.join(args.dir, name)
        if not os.path.exists(path):
            print(f"missing {name}")
            continue
        n += 1
        res = pipe.scan(F.load_audio(path))
        want = truth[name]
        wake_ok = bool(res.get("detected"))
        intent_ok = wake_ok and res.get("intent") == want["intent"]
        slot_ok = None
        if want.get("slot"):
            n_slot += 1
            slot_ok = intent_ok and res.get("slot") == want["slot"]
            ok_slot += bool(slot_ok)
        ok_wake += wake_ok
        ok_intent += intent_ok
        status = "ok  " if intent_ok else "FAIL"
        line = f"{status} {name}"
        if not intent_ok:
            line += (f"\n      want={want['intent']} "
                     f"got={res.get('intent')} "
                     f"wake={res.get('wake_score')}")
        if slot_ok is False:
            line += f"\n      slot want={want['slot']} got={res.get('slot')}"
        print(line, flush=True)

    print(f"\nwake  {ok_wake}/{n}  ({ok_wake / max(n, 1):.3f})")
    print(f"intent {ok_intent}/{n}  ({ok_intent / max(n, 1):.3f})")
    if n_slot:
        print(f"slot   {ok_slot}/{n_slot}  ({ok_slot / max(n_slot, 1):.3f})")
    acc = ok_intent / max(n, 1)
    sys.exit(0 if acc >= args.min_acc else 1)


if __name__ == "__main__":
    main()
