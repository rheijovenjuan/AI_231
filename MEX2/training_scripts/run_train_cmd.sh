set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
GPU=${GPU:-1}
echo "[train] using GPU $GPU (CUDA_VISIBLE_DEVICES pinned inside the script)"
python train/train_command.py --feat "$ROOT/feat" --out "$ROOT/out" \
    --gpu "$GPU" --epochs "${EPOCHS:-60}" --batch "${BATCH:-256}"
