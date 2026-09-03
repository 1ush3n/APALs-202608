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

PYTHON="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [ ! -f "${PYTHON}" ]; then
  PYTHON="$(which python3 || which python)"
fi

echo "========================================================================"
echo "Evaluating Initial Strict Ablation Checkpoints on 4 Benchmark Instances"
echo "Datasets: 283, 680, 2338, 3182"
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

CKPT_OP_ONLY=$(find_ckpt "results/01_initial_main/initial_operation_only_strict")
CKPT_OP_STATION=$(find_ckpt "results/01_initial_main/initial_operation_station_strict")
CKPT_GRAPHSAGE=$(find_ckpt "results/01_initial_main/initial_homogeneous_graphsage_strict")

echo "Found Checkpoints:"
echo "1. Operation-Only:        ${CKPT_OP_ONLY:-<none>}"
echo "2. Operation+Station:     ${CKPT_OP_STATION:-<none>}"
echo "3. Homogeneous GraphSAGE: ${CKPT_GRAPHSAGE:-<none>}"

eval_one_model() {
  local model_name="$1"
  local ckpt="$2"
  local exp="$3"
  if [ -z "${ckpt}" ] || [ ! -f "${ckpt}" ]; then
    echo "[WARN] Checkpoint for ${model_name} not found, skipping."
    return
  fi
  echo ">>> Evaluating ${model_name} on 4 instances: 283, 680, 2338, 3182..."
  for scale in 283 680 2338 3182; do
    echo "  [${model_name}] Instance ${scale}..."
    ${PYTHON} evaluate_model.py       experiment="${exp}"       model_path="${ckpt}"       test_data="data/${scale}.csv"       temperature=0.0       no_gantt=true       output_dir="results/01_initial_main/eval_${model_name}/${scale}"       > "logs/eval_${model_name}_${scale}.log" 2>&1
  done
  echo "  [${model_name}] Evaluation complete!"
}

eval_one_model "operation_only_strict" "${CKPT_OP_ONLY}" "initial_operation_only_strict" &
PID1=$!
eval_one_model "operation_station_strict" "${CKPT_OP_STATION}" "initial_operation_station_strict" &
PID2=$!
eval_one_model "homogeneous_graphsage_strict" "${CKPT_GRAPHSAGE}" "initial_homogeneous_graphsage_strict" &
PID3=$!

wait ${PID1}
wait ${PID2}
wait ${PID3}

echo "========================================================================"
echo "All Initial Strict Ablation evaluations finished!"
echo "Parsing results table..."
echo "========================================================================"

${PYTHON} scripts/parse_initial_eval_summary.py
