"""Export the trained checkpoints to ONNX and verify numeric parity.

    python export_onnx.py --feat $FEAT --out $OUT

Produces $OUT/onnx/{command.onnx,wake.onnx,model_card.json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rapi_vcm import features as F                    # noqa: E402
from rapi_vcm.keywords import KEYWORDS                # noqa: E402
from rapi_vcm.labels import INTENTS, SLOT_CLASSES      # noqa: E402
from rapi_vcm.models import (                          # noqa: E402
    build_command, build_keyword, build_wake,
)


def export(model, shape, path, output_names, opset=17):
    dummy = torch.zeros(shape, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["features"],
        output_names=output_names,
        dynamic_axes={"features": {0: "batch"},
                      output_names[0]: {0: "batch"},
                      **({output_names[1]: {0: "batch"}}
                         if len(output_names) > 1 else {})},
        opset_version=opset,
        do_constant_folding=True,
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    onnx_dir = os.path.join(args.out, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    device = torch.device("cpu")

    cmd_path = os.path.join(args.out, "command_best.pt")
    wake_path = os.path.join(args.out, "wake_best.pt")
    kw_path = os.path.join(args.out, "keyword_best.pt")
    report = {}

    # ------------------------------------------------------------- command ----
    ckpt = torch.load(cmd_path, map_location=device, weights_only=False)
    cmd = build_command()
    cmd.load_state_dict(ckpt["model"])
    cmd.eval()
    out_cmd = os.path.join(onnx_dir, "command.onnx")
    export(cmd, (1, 1, F.N_MELS, F.CMD_FRAMES), out_cmd,
           ["intent_logits", "slot_logits"], args.opset)

    # --------------------------------------------------------------- wake -----
    wckpt = torch.load(wake_path, map_location=device, weights_only=False)
    wake = build_wake()
    wake.load_state_dict(wckpt["model"])
    wake.eval()
    out_wake = os.path.join(onnx_dir, "wake.onnx")
    export(wake, (1, 1, F.N_MELS, F.WAKE_FRAMES), out_wake,
           ["wake_logit"], args.opset)

    # ------------------------------------------------------------ keyword ----
    kw_model = kw_ckpt = None
    out_kw = None
    if os.path.exists(kw_path):
        kw_ckpt = torch.load(kw_path, map_location=device, weights_only=False)
        kw_model = build_keyword()
        kw_model.load_state_dict(kw_ckpt["model"])
        kw_model.eval()
        out_kw = os.path.join(onnx_dir, "keyword.onnx")
        export(kw_model, (1, 1, F.N_MELS, F.CMD_FRAMES), out_kw,
               ["keyword_logits"], args.opset)

    # ------------------------------------------------------- parity check -----
    with torch.no_grad():
        x_cmd = torch.randn(8, 1, F.N_MELS, F.CMD_FRAMES)
        ti, ts = cmd(x_cmd)
        x_wake = torch.randn(8, 1, F.N_MELS, F.WAKE_FRAMES)
        tw = wake(x_wake)
        tk = kw_model(x_cmd) if kw_model is not None else None

    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        cs = ort.InferenceSession(out_cmd, so, providers=["CPUExecutionProvider"])
        ws = ort.InferenceSession(out_wake, so, providers=["CPUExecutionProvider"])
        oi, os_ = cs.run(None, {"features": x_cmd.numpy()})
        ow = ws.run(None, {"features": x_wake.numpy()})[0]
        report["parity"] = {
            "command_intent_max_abs_diff": float(np.abs(oi - ti.numpy()).max()),
            "command_slot_max_abs_diff": float(np.abs(os_ - ts.numpy()).max()),
            "wake_max_abs_diff": float(np.abs(ow - tw.numpy()).max()),
            "command_argmax_equal": bool((oi.argmax(1) == ti.numpy().argmax(1)).all()),
        }
        if kw_model is not None:
            ks = ort.InferenceSession(out_kw, so, providers=["CPUExecutionProvider"])
            ok_ = ks.run(None, {"features": x_cmd.numpy()})[0]
            report["parity"]["keyword_max_abs_diff"] = float(
                np.abs(ok_ - tk.numpy()).max())
            report["parity"]["keyword_top1_equal"] = bool(
                ((ok_ >= 0.5) == (tk.numpy() >= 0.5)).all())
        print("[export] parity", json.dumps(report["parity"], indent=2),
              flush=True)
    except ImportError:
        report["parity"] = "onnxruntime not installed on the training host"

    # ---------------------------------------------------------- model card ----
    def size_kb(p):
        return round(os.path.getsize(p) / 1024.0, 1)

    kw_thr = 0.5
    kw_thrs = None
    kw_metrics_path = os.path.join(args.out, "metrics", "keyword_metrics.json")
    if os.path.exists(kw_metrics_path):
        with open(kw_metrics_path, encoding="utf-8") as fh:
            m = json.load(fh)
        kw_thr = float(m.get("threshold", 0.5))
        kw_thrs = m.get("keyword_thresholds")

    card = {
        "created_from": {"command": cmd_path, "wake": wake_path,
                         "keyword": kw_path if kw_model else None},
        "files": {"command.onnx": size_kb(out_cmd),
                  "wake.onnx": size_kb(out_wake),
                  **({"keyword.onnx": size_kb(out_kw)} if out_kw else {})},
        "parameters": {
            "command": sum(p.numel() for p in cmd.parameters()),
            "wake": sum(p.numel() for p in wake.parameters()),
            **({"keyword": sum(p.numel() for p in kw_model.parameters())}
               if kw_model else {}),
        },
        "head": "keyword" if kw_model else "intent_softmax",
        "keywords": KEYWORDS,
        "keyword_threshold": kw_thr,
        "keyword_thresholds": kw_thrs,
        "input": {
            "name": "features",
            "dtype": "float32",
            "shape": ["batch", 1, F.N_MELS, F.CMD_FRAMES],
            "wake_shape": ["batch", 1, F.N_MELS, F.WAKE_FRAMES],
        },
        "outputs": {
            "keyword": ["keyword_logits (%d)" % len(KEYWORDS)],
            "command": ["intent_logits (19)", "slot_logits (%d)" % len(SLOT_CLASSES)],
            "wake": ["wake_logit"],
        },
        "intents": INTENTS,
        "slot_classes": SLOT_CLASSES,
        "feature_config": {
            "sample_rate": F.SAMPLE_RATE,
            "n_fft": F.N_FFT,
            "hop_length": F.HOP_LENGTH,
            "n_mels": F.N_MELS,
            "fmin": F.FMIN,
            "fmax": F.FMAX,
            "normalisation": "per-utterance CMVN",
            "cmd_seconds": F.CMD_SECONDS,
            "wake_seconds": F.WAKE_SECONDS,
        },
        "wake_threshold": float(wckpt.get("threshold", 0.5)),
        "opset": args.opset,
        "intents_count": len(INTENTS),
    }
    card.update(report)
    with open(os.path.join(onnx_dir, "model_card.json"), "w",
              encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(json.dumps({k: card[k] for k in ("files", "parameters",
                                           "wake_threshold")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
