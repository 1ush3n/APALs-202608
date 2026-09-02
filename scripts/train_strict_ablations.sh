#!/usr/bin/env bash
set -eo pipefail

cd /root/autodl-tmp/APALs-202608-full-x

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

PYTHON="/root/miniconda3/bin/python"

echo "========================================================================"
echo "Starting APAL Strict Ablation Suite (3 Strict Ablation Experiments)"
echo "1. reschedule_task_delay_r5_operation_only_strict (60 episodes)"
echo "2. reschedule_task_delay_r5_operation_station_strict (60 episodes)"
echo "3. reschedule_task_delay_r5_homogeneous_graphsage_strict (60 episodes)"
echo "Note: All strict ablations have reschedule_baseline_identity_conditioning=false"
echo "========================================================================"

# --- Experiment 1: Operation-Only Strict ---
echo ""
echo ">>> [1/3] Training Operation-Only Strict..."
${PYTHON} train.py \
  experiment=reschedule_task_delay_r5_operation_only_strict \
  train.num_envs=4 \
  hardware.num_envs=4 \
  hardware.worker_pointer_v2_fast_default_num_envs=4 \
  train.max_episodes=60 \
  seed=42 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

# --- Experiment 2: Operation+Station Strict ---
echo ""
echo ">>> [2/3] Training Operation+Station Strict..."
${PYTHON} train.py \
  experiment=reschedule_task_delay_r5_operation_station_strict \
  train.num_envs=4 \
  hardware.num_envs=4 \
  hardware.worker_pointer_v2_fast_default_num_envs=4 \
  train.max_episodes=60 \
  seed=42 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

# --- Experiment 3: Homogeneous GraphSAGE Strict ---
echo ""
echo ">>> [3/3] Training Homogeneous GraphSAGE Strict..."
${PYTHON} train.py \
  experiment=reschedule_task_delay_r5_homogeneous_graphsage_strict \
  train.num_envs=4 \
  hardware.num_envs=4 \
  hardware.worker_pointer_v2_fast_default_num_envs=4 \
  train.max_episodes=60 \
  seed=42 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

echo "========================================================================"
echo "All 3 strict ablation experiments trained successfully!"
echo "========================================================================"
