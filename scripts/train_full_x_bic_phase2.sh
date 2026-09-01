#!/usr/bin/env bash
set -eo pipefail

cd /root/autodl-tmp/APALs-202608-full-x

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo '========================================================================'
echo 'Starting FULL-X Phase 2 (Episodes 61-120) with BIC from Phase 1 best.ckpt'
echo '========================================================================'

/root/miniconda3/bin/python train.py   experiment=reschedule_task_delay_r5_full_x   reschedule_task_delay.reschedule_baseline_model_path=/root/autodl-tmp/APALs-202608-full-x/results/02_reschedule_main/reschedule_task_delay_r5_full_x/reschedule_task_delay_r5_full_x_260901-213714/checkpoints/best.ckpt   train.num_envs=4   hardware.num_envs=4   hardware.worker_pointer_v2_fast_default_num_envs=4   train.max_episodes=60   seed=43

echo '========================================================================'
echo 'Phase 2 (120 Total Episodes) finished successfully!'
echo '========================================================================'
