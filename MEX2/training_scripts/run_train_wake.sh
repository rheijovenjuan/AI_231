set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
GPU=${GPU:-1}
echo "[train-wake] using GPU $GPU"
python train/train_wake.py --feat "$ROOT/feat" --out "$ROOT/out" \
    --gpu "$GPU" --epochs "${EPOCHS:-40}" --batch "${BATCH:-512}"
