#!/usr/bin/env bash
# Full pipeline on the DGX node - run it with:
#     bash run_all.sh
# Every stage runs inside ~/rapi_vcm/venv (a dedicated virtual environment)
# and every training stage is pinned to a single GPU (default: GPU 1).
set -e
ROOT=/home/rhei.joven.juan/rapi_vcm
export GPU=${GPU:-1}
export EPOCHS=${EPOCHS:-60}

echo "=============================================================="
echo " Rapi VCM - full training pipeline"
echo " venv : $ROOT/venv"
echo " GPU  : $GPU  (1 of 8, pinned via CUDA_VISIBLE_DEVICES)"
echo "=============================================================="

bash "$ROOT/run_extract.sh"
bash "$ROOT/run_wake.sh"
bash "$ROOT/run_train_cmd.sh"
bash "$ROOT/run_train_wake.sh"
bash "$ROOT/run_eval_export.sh"
echo "ALL STAGES FINISHED"
