set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
echo "[eval] keyword spotter on the test split"
python train/evaluate_keyword.py --feat "$ROOT/feat" --out "$ROOT/out"
echo "[export] onnx"
python train/export_onnx.py --out "$ROOT/out"
ls -la "$ROOT/out/onnx"
