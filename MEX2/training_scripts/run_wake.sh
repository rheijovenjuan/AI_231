set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
DATA="$ROOT/dataset/ai231_tmp/MEX2/OptionB"
FEAT="$ROOT/feat"
mkdir -p "$FEAT"
REAL="$ROOT/real_wake/positive"
EXTRA=""
[ -d "$REAL" ] && EXTRA="--real $REAL --real-per-clip 40"
echo "[wake] start $(date) EXTRA=$EXTRA"
python train/gen_wake_data.py --dataset "$DATA" --out "$FEAT" $EXTRA
echo "[wake] done $(date)"