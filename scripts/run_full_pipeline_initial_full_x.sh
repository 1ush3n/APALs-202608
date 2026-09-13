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
echo "Phase 1: Starting APAL Initial FULL-X Training (60 ep, batch=128, BF16)"
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"
mkdir -p logs

${PYTHON} train.py   experiment=initial_worker_pointer_v2_full_x   train.batch_size=128   train.accumulation_steps=8   train.num_envs=4   hardware.num_envs=4   hardware.worker_pointer_v2_fast_default_num_envs=4   train.max_episodes=60   seed=42   > logs/train_initial_full_x.log 2>&1

echo "========================================================================"
echo "Phase 1 Completed! Locating checkpoint..."
echo "========================================================================"

LATEST_RUN=$(ls -td results/01_initial_main/initial_worker_pointer_v2_full_x/initial_worker_pointer_v2_full_x_* | head -n 1)
BEST_CKPT="${LATEST_RUN}/checkpoints/best.ckpt"
if [ ! -f "${BEST_CKPT}" ]; then
  BEST_CKPT="${LATEST_RUN}/checkpoints/last.ckpt"
fi

echo "Selected Checkpoint: ${BEST_CKPT}"

echo "========================================================================"
echo "Phase 2: 20-Scenario Stochastic Evaluation on 4 Instances x 5 Seeds"
echo "========================================================================"
bash "${SCRIPT_DIR}/eval_stochastic_20runs.sh" full_x_initial "${BEST_CKPT}" initial_worker_pointer_v2_full_x

echo "========================================================================"
echo "Phase 3: Parsing All Summaries"
echo "========================================================================"
${PYTHON} scripts/parse_stochastic_eval_summary.py

echo "========================================================================"
echo "All tasks finished successfully! Auto-shutdown in 15 seconds..."
echo "========================================================================"

if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then
  sync
  sleep 15
  shutdown -h now || /usr/bin/shutdown || true
fi
