#!/usr/bin/env bash
set -euo pipefail

# 服务器执行前请确保已进入 rag_env；也可通过 PYTHON_BIN 指定解释器。
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" scripts/run_initial_models_unified_parallel.py "$@"
