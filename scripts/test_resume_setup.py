import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import configs
from runtime.hydra_config import initialize_hydra_runtime
from runtime.paths import resolve_checkpoint_paths
from runtime.artifacts import run_context
from train_lightning import _resolve_resume_checkpoint_path, _resume_start_episode
from runtime.checkpoints import load_checkpoint

def main():
    cli_args = [
        "experiment=initial_worker_pointer_v2_full_x",
        "run_id=initial_worker_pointer_v2_full_x_260904-123836",
        "resume=true",
        "resume_checkpoint_path=/root/autodl-tmp/APALs-202608-initial-strict/results/01_initial_main/initial_worker_pointer_v2_full_x/initial_worker_pointer_v2_full_x_260904-123836/checkpoints/last.ckpt",
        "train.batch_size=128",
        "train.accumulation_steps=8",
        "train.num_envs=4",
        "hardware.num_envs=4",
        "hardware.worker_pointer_v2_fast_default_num_envs=4",
        "train.max_episodes=60",
        "seed=42",
    ]
    
    args = initialize_hydra_runtime(
        cli_args,
        target=configs,
        project_root=PROJECT_ROOT,
        default_experiment="initial_worker_pointer_v2_full_x",
        create_run_context=False,
    )
    print("Parsed args.run_id:", getattr(args, "run_id", None))
    print("Parsed configs.run_id:", getattr(configs, "run_id", None))
    
    ctx = run_context(configs, PROJECT_ROOT, create_dirs=False)
    print("Context run_id:", ctx.run_id)
    print("Context run_dir:", ctx.run_dir)
    print("Context checkpoint_dir:", ctx.checkpoint_dir)
    
    checkpoint_paths = resolve_checkpoint_paths(configs)
    print("checkpoint_paths['lightning_latest']:", checkpoint_paths["lightning_latest"])
    print("checkpoint_paths['lightning_best']:", checkpoint_paths["lightning_best"])
    
    resume_path = _resolve_resume_checkpoint_path(args, checkpoint_paths)
    print("Resolved resume_path:", resume_path)
    print("Resume path exists:", resume_path.is_file())
    
    ckpt = load_checkpoint(resume_path)
    next_ep = _resume_start_episode(ckpt.payload)
    print("Next episode will be:", next_ep)
    
    # Check async eval best state
    async_eval_best_json = ctx.checkpoint_dir / "async_eval" / "state" / "best.json"
    print("async_eval best.json path:", async_eval_best_json)
    print("async_eval best.json exists:", async_eval_best_json.is_file())
    if async_eval_best_json.is_file():
        best_data = json.loads(async_eval_best_json.read_text(encoding="utf-8-sig"))
        print("Existing best.json score:", best_data.get("selection_score"))
        print("Existing best.json episode:", best_data.get("episode"))
        print("Existing best.json best_path:", best_data.get("best_path"))

if __name__ == "__main__":
    main()
