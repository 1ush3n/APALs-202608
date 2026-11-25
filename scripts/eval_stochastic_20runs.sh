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

MODEL_NAME="$1"
CKPT_PATH="$2"
EXP_NAME="$3"

if [ -z "${MODEL_NAME}" ] || [ -z "${CKPT_PATH}" ] || [ -z "${EXP_NAME}" ]; then
  echo "Usage: $0 <model_name> <checkpoint_path> <experiment_name>"
  exit 1
fi

echo "========================================================================"
echo "Starting 20-Scenario Stochastic Evaluation for: ${MODEL_NAME}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Datasets: 283, 680, 2338, 3182"
echo "Seeds: 42, 43, 44, 45, 46 (temperature=0.01) + seed 42 (temperature=0.0)"
echo "Concurrency limit: 4 parallel jobs to protect running training"
echo "========================================================================"

mkdir -p logs/eval_${MODEL_NAME}

DATASETS=(283 680 2338 3182)
SEEDS=(42 43 44 45 46)
MAX_JOBS=4

run_eval() {
  local scale="$1"
  local seed="$2"
  local temp="$3"
  local tag="$4"
  local out_dir="results/01_initial_main/eval_${MODEL_NAME}/real_${scale}/${tag}"
  local log_file="logs/eval_${MODEL_NAME}/eval_${scale}_${tag}.log"
  
  ${PYTHON} evaluate_model.py     experiment="${EXP_NAME}"     model_path="${CKPT_PATH}"     test_data="data/${scale}.csv"     seed="${seed}"     temperature="${temp}"     no_gantt=true     output_dir="${out_dir}"     > "${log_file}" 2>&1
  
  echo "  [Done] ${MODEL_NAME} scale=${scale} ${tag}"
}

# Run 4 datasets * 5 seeds (temp=0.01) + 4 datasets * 1 seed (temp=0)
for scale in "${DATASETS[@]}"; do
  # 1. Deterministic baseline
  (run_eval "${scale}" 42 0.0 "temp0_seed42") &
  while [ $(jobs -r | wc -l) -ge ${MAX_JOBS} ]; do sleep 1; done

  # 2. 5 stochastic runs
  for seed in "${SEEDS[@]}"; do
    (run_eval "${scale}" "${seed}" 0.01 "temp001_seed${seed}") &
    while [ $(jobs -r | wc -l) -ge ${MAX_JOBS} ]; do sleep 1; done
  done
done

# Wait for all background eval jobs to complete
wait

echo "========================================================================"
echo "All 20+ evaluations completed for: ${MODEL_NAME}"
echo "========================================================================"
