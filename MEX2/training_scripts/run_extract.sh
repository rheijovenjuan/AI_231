set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
DATA="$ROOT/dataset/ai231_tmp/MEX2/OptionB"
FEAT="$ROOT/feat"
mkdir -p "$FEAT"
export OMP_NUM_THREADS=8
echo "[extract] start $(date)"
python train/extract_features.py --dataset "$DATA" --out "$FEAT" --jobs 24
echo "[extract] done $(date)"
