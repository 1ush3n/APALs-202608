
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs

def resolve_project_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_latest_checkpoint(model_dir):
    model_dir = resolve_project_path(model_dir)
    list_of_files = list(model_dir.glob("*.pth"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=lambda path: path.stat().st_ctime)
    return latest_file

def generate_schedule(model_path=None):
    print("--- Generating Schedule (Deterministic) ---")
    
    data_path = getattr(configs, 'data_file_path', 'data/283.csv')
    env = AirLineEnv_Graph(data_path_or_dir=data_path)
    print(f"Dataset: {data_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    model = HBGATPN(configs).to(device)
    
    if model_path is None:
        model_dir = "checkpoints"
        model_path = find_latest_checkpoint(model_dir)
        
    if model_path is not None:
        model_path = resolve_project_path(model_path)

    if model_path and model_path.exists():
        print(f"Loading weights from: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e:
            print(f"Warning: Architecture mismatch for {model_path}. Proceeding with random weights.\nError details: {str(e)[:100]}...")
    else:
        print("Warning: No checkpoint found or specified. Using random weights.")
        
    agent = PPOAgent(model, configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, device, configs.batch_size)
    
    # Rollout
    state = env.reset()
    done = False
    total_reward = 0
    
    # Track Schedule Details
    # env.assigned_tasks already has (task_id, station_id, worker_id, start, end)
    # But internal_id. Need mapping back to original?
    # env.raw_data['id_map'] is {orig: internal}.
    # We need internal -> orig.
    
    id_map = env.raw_data['id_map']
    internal_to_orig = {v: k for k, v in id_map.items()}
    
    while not done:
        # Mask
        task_mask, station_mask, worker_mask = env.get_masks()
        
        # Action (Strictly Greedy for Final Output)
        action, _, _, _, _ = agent.select_action(
            state.to(device),
            mask_task=task_mask.to(device),
            mask_station_matrix=station_mask.to(device),
            mask_worker=worker_mask.to(device),
            deterministic=True,
            temperature=0.0
        )
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        if 'error' in info:
            print(f"Error during generation: {info['error']}")
            break
        
    print(f"--- Rollout Complete. Total Reward: {total_reward:.2f} ---")
    
    # 运行完毕，从环境中直接抽取真实的历史记录
    schedule_log = []
    for (t_id, s_id, team, start_time, end_time) in env.assigned_tasks:
        # 包含所有工序，即使是虚拟节点 (s_id == -1)，因为 verifier 需要完整拓扑
        # Try to find original ID if map exists, else use raw internal int
        if hasattr(env, 'raw_data') and 'task_df' in env.raw_data:
            try:
                original_id = env.raw_data['task_df'].iloc[t_id]['task_id']
            except Exception:
                original_id = t_id
        else:
            original_id = t_id
            
        schedule_log.append({
            "TaskID": t_id,
            "StationID": s_id + 1, # 1-based 适应物理站位习惯
            "Team": str(team), # 记录班组信息 (内部 ID 0-based)
            "Start": start_time,
            "End": end_time,
            "Duration": end_time - start_time # 计算出受效率影响的真实工时
        })
        
    df = pd.DataFrame(schedule_log)
    df = df.sort_values("Start")
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "final_schedule.csv"
    
    df.to_csv(out_file, index=False)
    print(f"Schedule saved to: {out_file}")
    
    return df

if __name__ == "__main__":
    generate_schedule()
