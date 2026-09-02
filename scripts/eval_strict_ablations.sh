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
echo "Evaluating Strict Ablation Checkpoints on 36-scenario R5 Benchmark (PARALLEL)"
echo "Working directory: ${PROJECT_ROOT}"
echo "========================================================================"

mkdir -p logs

find_ckpt() {
  local exp_dir="$1"
  local best
  best=$(ls -t ${exp_dir}/*/checkpoints/best.ckpt 2>/dev/null | head -n 1 || true)
  if [ -n "${best}" ] && [ -f "${best}" ]; then
    echo "${best}"
  else
    local last
    last=$(ls -t ${exp_dir}/*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)
    echo "${last}"
  fi
}

CKPT_OP_ONLY=$(find_ckpt "results/02_reschedule_main/reschedule_task_delay_r5_operation_only_strict")
CKPT_OP_STATION=$(find_ckpt "results/02_reschedule_main/reschedule_task_delay_r5_operation_station_strict")
CKPT_GRAPHSAGE=$(find_ckpt "results/02_reschedule_main/reschedule_task_delay_r5_homogeneous_graphsage_strict")

echo "Found Checkpoints:"
echo "1. Operation-Only:       ${CKPT_OP_ONLY:-<none>}"
echo "2. Operation+Station:    ${CKPT_OP_STATION:-<none>}"
echo "3. Homogeneous GraphSAGE: ${CKPT_GRAPHSAGE:-<none>}"

PID_EVAL_1=""
PID_EVAL_2=""
PID_EVAL_3=""

if [ -n "${CKPT_OP_ONLY}" ] && [ -f "${CKPT_OP_ONLY}" ]; then
  echo ">>> Launching [1/3] Operation-Only Strict Evaluation on 36 scenarios..."
  ${PYTHON} scripts/evaluate_reschedule_manifest.py     experiment=reschedule_task_delay_r5_operation_only_strict     model_path="${CKPT_OP_ONLY}"     manifest_path=data/r5_task_delay_v1/manifest.json     output_dir=results/02_reschedule_main/r5_eval_operation_only_strict     temperature=0.0     reschedule_task_delay.reschedule_baseline_identity_conditioning=false     > logs/eval_operation_only_strict.log 2>&1 &
  PID_EVAL_1=$!
fi

if [ -n "${CKPT_OP_STATION}" ] && [ -f "${CKPT_OP_STATION}" ]; then
  echo ">>> Launching [2/3] Operation+Station Strict Evaluation on 36 scenarios..."
  ${PYTHON} scripts/evaluate_reschedule_manifest.py     experiment=reschedule_task_delay_r5_operation_station_strict     model_path="${CKPT_OP_STATION}"     manifest_path=data/r5_task_delay_v1/manifest.json     output_dir=results/02_reschedule_main/r5_eval_operation_station_strict     temperature=0.0     reschedule_task_delay.reschedule_baseline_identity_conditioning=false     > logs/eval_operation_station_strict.log 2>&1 &
  PID_EVAL_2=$!
fi

if [ -n "${CKPT_GRAPHSAGE}" ] && [ -f "${CKPT_GRAPHSAGE}" ]; then
  echo ">>> Launching [3/3] Homogeneous GraphSAGE Strict Evaluation on 36 scenarios..."
  ${PYTHON} scripts/evaluate_reschedule_manifest.py     experiment=reschedule_task_delay_r5_homogeneous_graphsage_strict     model_path="${CKPT_GRAPHSAGE}"     manifest_path=data/r5_task_delay_v1/manifest.json     output_dir=results/02_reschedule_main/r5_eval_homogeneous_graphsage_strict     temperature=0.0     reschedule_task_delay.reschedule_baseline_identity_conditioning=false     > logs/eval_homogeneous_graphsage_strict.log 2>&1 &
  PID_EVAL_3=$!
fi

[ -n "${PID_EVAL_1}" ] && wait ${PID_EVAL_1}
[ -n "${PID_EVAL_2}" ] && wait ${PID_EVAL_2}
[ -n "${PID_EVAL_3}" ] && wait ${PID_EVAL_3}

echo "========================================================================"
echo "All available strict ablation evaluations completed successfully!"
echo "========================================================================"
