#!/usr/bin/env bash
set -eo pipefail

cd /root/autodl-tmp/APALs-202608-full-x

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo '=========================================================='
echo 'Starting FULL-X Reschedule Training with BIC (main-final)'
echo '=========================================================='

/root/miniconda3/bin/python train.py   experiment=reschedule_task_delay_r5_full_x   train.num_envs=4   hardware.num_envs=4   hardware.worker_pointer_v2_fast_default_num_envs=4   train.max_episodes=60   seed=42

echo '=========================================================='
echo 'Training finished successfully!'
echo '=========================================================='
