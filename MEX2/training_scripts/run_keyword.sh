set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
DATA="$ROOT/dataset/ai231_tmp/MEX2/OptionB"

echo "[keywords] vocabulary self-check"
python -m rapi_vcm.keywords "$DATA/manifest.csv"

if [ -f "$ROOT/feat/cmd_kw.npy" ] && [ "$1" != "--reextract" ]; then
  echo "[extract] cmd_kw.npy already present - skipping"
else
  echo "[extract] features + keyword labels"
  python train/extract_features.py --dataset "$DATA" --out "$ROOT/feat" --jobs 24
fi

echo "[train] keyword spotter (1 GPU)"
python train/train_keyword.py --feat "$ROOT/feat" --out "$ROOT/out" --gpu 1
