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

PYTHON="/root/miniconda3/bin/python"

echo "========================================================================"
echo "Starting APAL Strict Ablation Suite in PARALLEL (3 Concurrent Trainings)"
echo "1. reschedule_task_delay_r5_operation_only_strict (60 ep, batch=32, envs=2)"
echo "2. reschedule_task_delay_r5_operation_station_strict (60 ep, batch=32, envs=2)"
echo "3. reschedule_task_delay_r5_homogeneous_graphsage_strict (60 ep, batch=32, envs=2)"
echo "Note: All strict ablations have reschedule_baseline_identity_conditioning=false"
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"

mkdir -p logs

echo ">>> Launching [1/3] Operation-Only Strict in background..."
${PYTHON} train.py   experiment=reschedule_task_delay_r5_operation_only_strict   train.batch_size=32   train.num_envs=2   hardware.num_envs=2   hardware.worker_pointer_v2_fast_default_num_envs=2   train.max_episodes=60   seed=42   reschedule_task_delay.reschedule_baseline_identity_conditioning=false   > logs/train_operation_only_strict.log 2>&1 &
PID_OP_ONLY=$!

echo ">>> Launching [2/3] Operation+Station Strict in background..."
${PYTHON} train.py   experiment=reschedule_task_delay_r5_operation_station_strict   train.batch_size=32   train.num_envs=2   hardware.num_envs=2   hardware.worker_pointer_v2_fast_default_num_envs=2   train.max_episodes=60   seed=42   reschedule_task_delay.reschedule_baseline_identity_conditioning=false   > logs/train_operation_station_strict.log 2>&1 &
PID_OP_STATION=$!

echo ">>> Launching [3/3] Homogeneous GraphSAGE Strict in background..."
${PYTHON} train.py   experiment=reschedule_task_delay_r5_homogeneous_graphsage_strict   train.batch_size=32   train.num_envs=2   hardware.num_envs=2   hardware.worker_pointer_v2_fast_default_num_envs=2   train.max_episodes=60   seed=42   reschedule_task_delay.reschedule_baseline_identity_conditioning=false   > logs/train_homogeneous_graphsage_strict.log 2>&1 &
PID_GRAPHSAGE=$!

echo "All 3 parallel training jobs launched:"
echo "  PID 1 (Operation-Only):        ${PID_OP_ONLY}"
echo "  PID 2 (Operation+Station):     ${PID_OP_STATION}"
echo "  PID 3 (Homogeneous GraphSAGE):  ${PID_GRAPHSAGE}"
echo "Waiting for all 3 jobs to complete..."

wait ${PID_OP_ONLY}
wait ${PID_OP_STATION}
wait ${PID_GRAPHSAGE}

echo "========================================================================"
echo "All 3 parallel strict ablation experiments trained successfully!"
echo "Starting full 36-scenario evaluation..."
echo "========================================================================"

bash "${SCRIPT_DIR}/eval_strict_ablations.sh"

echo "========================================================================"
echo "Strict ablation training and evaluation suite completed!"
echo "========================================================================"
