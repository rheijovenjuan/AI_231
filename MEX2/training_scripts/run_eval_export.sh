set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
echo "[eval] test-split metrics"
python train/evaluate.py --feat "$ROOT/feat" --out "$ROOT/out"
echo "[export] onnx"
python train/export_onnx.py --out "$ROOT/out"
echo "[done] artifacts in $ROOT/out"
ls -la "$ROOT/out" "$ROOT/out/onnx" "$ROOT/out/metrics"
