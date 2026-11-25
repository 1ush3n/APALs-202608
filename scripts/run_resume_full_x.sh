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

RUN_ID="initial_worker_pointer_v2_full_x_260904-123836"
RUN_DIR="${PROJECT_ROOT}/results/01_initial_main/initial_worker_pointer_v2_full_x/${RUN_ID}"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"
BEST_CKPT="${RUN_DIR}/checkpoints/best.ckpt"
FLAG_FILE="${PROJECT_ROOT}/results/01_initial_main/FULL_X_PIPELINE_COMPLETE"

# Ensure any previous flag file is removed
rm -f "${FLAG_FILE}"

echo "========================================================================"
echo "Phase 1: Resuming APAL Initial FULL-X Training (Episodes 46 to 60)"
echo "Run Dir: ${RUN_DIR}"
echo "Resume Checkpoint: ${LAST_CKPT}"
echo "Start Time: $(date)"
echo "========================================================================"
mkdir -p logs

${PYTHON} train.py \
  experiment=initial_worker_pointer_v2_full_x \
  run_id="${RUN_ID}" \
  resume=true \
  resume_checkpoint_path="${LAST_CKPT}" \
  train.batch_size=128 \
  train.accumulation_steps=8 \
  train.num_envs=4 \
  hardware.num_envs=4 \
  hardware.worker_pointer_v2_fast_default_num_envs=4 \
  train.max_episodes=60 \
  seed=42 \
  >> logs/train_initial_full_x.log 2>&1

echo "========================================================================"
echo "Phase 1 Completed at $(date)! Checking checkpoints..."
echo "========================================================================"

if [ ! -f "${BEST_CKPT}" ]; then
  echo "Error: best.ckpt not found at ${BEST_CKPT}"
  exit 1
fi

echo "Selected Best Checkpoint: ${BEST_CKPT}"
ls -lh "${BEST_CKPT}"

echo "========================================================================"
echo "Phase 2: 20-Scenario Stochastic Evaluation on 4 Instances x 5 Seeds"
echo "Start Time: $(date)"
echo "========================================================================"
bash "${SCRIPT_DIR}/eval_stochastic_20runs.sh" full_x_initial "${BEST_CKPT}" initial_worker_pointer_v2_full_x

echo "========================================================================"
echo "Phase 3: Parsing All Summaries"
echo "Start Time: $(date)"
echo "========================================================================"
${PYTHON} scripts/parse_stochastic_eval_summary.py

echo "========================================================================"
echo "APAL FULL-X Pipeline FINISHED SUCCESSFULLY at $(date)!"
echo "========================================================================"
touch "${FLAG_FILE}"
