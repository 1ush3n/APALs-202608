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
echo "Starting APAL Initial Strict Ablation Suite in PARALLEL (3 Concurrent Trainings)"
echo "1. initial_operation_only_strict (60 ep, batch=256, accum=16, BF16)"
echo "2. initial_operation_station_strict (60 ep, batch=256, accum=16, BF16)"
echo "3. initial_homogeneous_graphsage_strict (60 ep, batch=256, accum=16, BF16)"
echo "All strict ablations: BF16 precision, 60 episodes, MinWait completion, no Dynamic EFT"
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"

mkdir -p logs

echo ">>> Launching [1/3] Initial Operation-Only Strict in background..."
${PYTHON} train.py   experiment=initial_operation_only_strict   train.batch_size=256   train.max_episodes=60   seed=42   > logs/train_initial_operation_only_strict.log 2>&1 &
PID_OP_ONLY=$!

echo ">>> Launching [2/3] Initial Operation+Station Strict in background..."
${PYTHON} train.py   experiment=initial_operation_station_strict   train.batch_size=256   train.max_episodes=60   seed=42   > logs/train_initial_operation_station_strict.log 2>&1 &
PID_OP_STATION=$!

echo ">>> Launching [3/3] Initial Homogeneous GraphSAGE Strict in background..."
${PYTHON} train.py   experiment=initial_homogeneous_graphsage_strict   train.batch_size=256   train.max_episodes=60   seed=42   > logs/train_initial_homogeneous_graphsage_strict.log 2>&1 &
PID_GRAPHSAGE=$!

echo "All 3 parallel training jobs launched:"
echo "  PID 1 (Operation-Only):         ${PID_OP_ONLY}"
echo "  PID 2 (Operation+Station):      ${PID_OP_STATION}"
echo "  PID 3 (Homogeneous GraphSAGE):   ${PID_GRAPHSAGE}"
echo "Waiting for all 3 jobs to complete..."

wait ${PID_OP_ONLY} || echo "[WARN] Operation-Only process exited with non-zero status"
wait ${PID_OP_STATION} || echo "[WARN] Operation+Station process exited with non-zero status"
wait ${PID_GRAPHSAGE} || echo "[WARN] Homogeneous GraphSAGE process exited with non-zero status"

echo "========================================================================"
echo "All 3 parallel initial strict ablation experiments trained!"
echo "Starting evaluation on 4 benchmark instances (283, 680, 2338, 3182)..."
echo "========================================================================"

bash "${SCRIPT_DIR}/eval_initial_strict_ablations.sh"

echo "========================================================================"
echo "Initial strict ablation training and evaluation suite completed!"
echo "========================================================================"

if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then
  echo "Auto-shutdown is ENABLED. Shutting down the machine in 10 seconds to stop billing..."
  sync
  sleep 10
  shutdown -h now || /usr/bin/shutdown || true
fi
