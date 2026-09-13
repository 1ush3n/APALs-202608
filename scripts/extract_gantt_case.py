"""
Extract exact schedule for real_283 medium_early scenario using Hydra runtime.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from evaluate_reschedule_model import _load_policy_weights
from runtime.reschedule_eval import evaluate_reschedule_model
from runtime.checkpoints import (
    load_checkpoint,
    apply_checkpoint_model_spec,
)
from runtime.hydra_config import initialize_hydra_runtime
from runtime.reschedule_manifest import load_reschedule_manifest
from runtime.initial_worker_mapping import apply_initial_worker_mapping

raw_args = [
    "experiment=reschedule_task_delay_r5_full_x",
]
args = initialize_hydra_runtime(
    raw_args,
    target=configs,
    project_root=PROJECT_ROOT,
    default_experiment="reschedule_task_delay_r5_full_x",
)

model_path = PROJECT_ROOT / "results/02_reschedule_main/reschedule_task_delay_r5_full_x/reschedule_task_delay_r5_full_x_260901-232349/checkpoints/best.ckpt"
print(f"Loading checkpoint from: {model_path}")
checkpoint = load_checkpoint(model_path)
apply_checkpoint_model_spec(configs, checkpoint.model_spec, explicit_fields=set())

import argparse

cli_parser = argparse.ArgumentParser(description="Extract schedule under scenario")
cli_parser.add_argument("instance_id", nargs="?", default="real_283", help="Target instance ID")
cli_parser.add_argument("--mes", action="store_true", help="Enable Minimum Earliest Station (MES) decoding")
cli_parser.add_argument("--mes_lambda", type=float, default=2.0, help="MES penalty lambda")
cli_parser.add_argument("--mes_jump", type=int, default=1, help="MES max station jump")
cli_parser.add_argument("--mes_mode", type=str, default="soft_penalty", choices=["soft_penalty", "hard_bound", "combined"], help="MES mode")
cli_parser.add_argument("--baseline_csv", type=str, default=None, help="Custom baseline schedule CSV path")
cli_parser.add_argument("--out_csv", type=str, default=None, help="Custom output CSV path")
cli_args = cli_parser.parse_args()

instance_id = cli_args.instance_id
print(f"Target instance: {instance_id}")

# Setup target instance from manifest
manifest_path = PROJECT_ROOT / "data/r5_task_delay_v1/manifest.json"
manifest = load_reschedule_manifest(manifest_path)
entry = manifest.get(instance_id)

configs.enable_reschedule_mode = True
configs.reschedule_manifest_path = str(manifest.path)
configs.reschedule_eval_instance_id = instance_id
configs.data_file_path = str(entry.data_path)
configs.reschedule_baseline_schedule_path = str(cli_args.baseline_csv or entry.baseline_schedule_path)
configs.reschedule_eval_scenario_path = str(entry.scenario_path)

if cli_args.mes:
    configs.enable_mes_decoding = True
    configs.mes_penalty_lambda = cli_args.mes_lambda
    configs.mes_max_station_jump = cli_args.mes_jump
    configs.mes_mode = cli_args.mes_mode
    print(f"[MES Enabled] lambda={cli_args.mes_lambda}, jump={cli_args.mes_jump}, mode={cli_args.mes_mode}")

apply_initial_worker_mapping(configs, entry.data_path, explicit_fields=set())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = AirLineEnv_Graph(data_path_or_dir=str(entry.data_path), seed=int(configs.seed))
model = HBGATPN(configs).to(device)
load_stats = _load_policy_weights(model, model_path, device)
print("Load stats:", load_stats)

agent = PPOAgent(
    model,
    configs.lr,
    configs.gamma,
    configs.k_epochs,
    configs.eps_clip,
    device,
    batch_size=configs.batch_size,
    total_timesteps=1,
    config=configs,
)

print("Evaluating medium_early...")
makespan, balance, reward, best_schedule, duration, worker_util, station_util = evaluate_reschedule_model(
    env,
    agent,
    num_runs=1,
    scenario_ids=["medium_early"],
    temperature=0.0,
    skip_value_estimation=True,
)

print(f"Result Makespan: {makespan:.4f} h (Target: 306.5635 h)")
print(f"Assigned tasks count: {len(best_schedule)}")

rows = []
for item in best_schedule:
    tid, sid, team, start_t, finish_t = item
    rows.append({
        "TaskID": int(tid),
        "StationID": int(sid),
        "Team": str([int(w) for w in team]),
        "Start": float(start_t),
        "End": float(finish_t),
        "Duration": float(finish_t - start_t)
    })

out_df = pd.DataFrame(rows)
if cli_args.out_csv:
    out_csv = Path(cli_args.out_csv)
else:
    suffix = "_mes" if cli_args.mes else ""
    out_csv = PROJECT_ROOT / "data" / "r5_task_delay_v1" / f"{instance_id}_medium_early_repaired_schedule{suffix}.csv"
out_df.to_csv(out_csv, index=False)
print(f"Saved repaired schedule to: {out_csv}")

print(f"\n--- Station Distribution for {instance_id} Repaired Schedule ---")
for s in range(5):
    sdf = out_df[out_df["StationID"] == s]
    if not sdf.empty:
        print(f"  Station {s+1}: count={len(sdf):4d} ({len(sdf)/len(out_df)*100:4.1f}%), Start: [{sdf.Start.min():.1f}h -> {sdf.Start.max():.1f}h], End: [{sdf.End.min():.1f}h -> {sdf.End.max():.1f}h]")

