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

PYTHON="/root/miniconda3/bin/python"

echo "========================================================================"
echo "Evaluating Strict Ablation Checkpoints on 36-scenario R5 Benchmark..."
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"

# Find latest best checkpoints for each of the 3 experiments
CKPT_OP_ONLY=$(ls -t results/02_reschedule_main/reschedule_task_delay_r5_operation_only_strict/*/checkpoints/best.ckpt | head -n 1)
CKPT_OP_STATION=$(ls -t results/02_reschedule_main/reschedule_task_delay_r5_operation_station_strict/*/checkpoints/best.ckpt | head -n 1)
CKPT_GRAPHSAGE=$(ls -t results/02_reschedule_main/reschedule_task_delay_r5_homogeneous_graphsage_strict/*/checkpoints/best.ckpt | head -n 1)

echo "Found Checkpoints:"
echo "1. Operation-Only: ${CKPT_OP_ONLY}"
echo "2. Operation+Station: ${CKPT_OP_STATION}"
echo "3. Homogeneous GraphSAGE: ${CKPT_GRAPHSAGE}"

echo ""
echo ">>> [1/3] Evaluating Operation-Only Strict on 36 scenarios..."
${PYTHON} scripts/evaluate_reschedule_manifest.py \
  experiment=reschedule_task_delay_r5_operation_only_strict \
  model_path="${CKPT_OP_ONLY}" \
  manifest_path=data/r5_task_delay_v1/manifest.json \
  output_dir=results/02_reschedule_main/r5_eval_operation_only_strict \
  temperature=0.0 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

echo ""
echo ">>> [2/3] Evaluating Operation+Station Strict on 36 scenarios..."
${PYTHON} scripts/evaluate_reschedule_manifest.py \
  experiment=reschedule_task_delay_r5_operation_station_strict \
  model_path="${CKPT_OP_STATION}" \
  manifest_path=data/r5_task_delay_v1/manifest.json \
  output_dir=results/02_reschedule_main/r5_eval_operation_station_strict \
  temperature=0.0 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

echo ""
echo ">>> [3/3] Evaluating Homogeneous GraphSAGE Strict on 36 scenarios..."
${PYTHON} scripts/evaluate_reschedule_manifest.py \
  experiment=reschedule_task_delay_r5_homogeneous_graphsage_strict \
  model_path="${CKPT_GRAPHSAGE}" \
  manifest_path=data/r5_task_delay_v1/manifest.json \
  output_dir=results/02_reschedule_main/r5_eval_homogeneous_graphsage_strict \
  temperature=0.0 \
  reschedule_task_delay.reschedule_baseline_identity_conditioning=false

echo "========================================================================"
echo "All 3 strict ablation evaluations completed successfully!"
echo "========================================================================"
