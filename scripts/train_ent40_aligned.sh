#!/bin/bash
set -e
cd /root/autodl-tmp/APALs-202608-full-x
echo "[Start] Launching FULL-X ent40-aligned training warm-started from yesterday's best.ckpt..."

/root/miniconda3/bin/python train.py \
    experiment=reschedule_task_delay_r5_full_x_ent40_aligned \
    train.num_envs=4 \
    hardware.num_envs=4 \
    hardware.worker_pointer_v2_fast_default_num_envs=4 \
    train.max_episodes=60 \
    seed=42
