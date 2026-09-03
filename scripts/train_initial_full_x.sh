#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

PYTHON="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [ ! -f "${PYTHON}" ]; then
  PYTHON="$(which python3 || which python)"
fi

echo "========================================================================"
echo "Starting APAL Initial FULL-X Training (60 ep, batch=256, accum=16, BF16)"
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"

mkdir -p logs

${PYTHON} train.py   experiment=initial_worker_pointer_v2_full_x   train.batch_size=256   train.max_episodes=60   seed=42   2>&1 | tee logs/train_initial_full_x.log
