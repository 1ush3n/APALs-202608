"""
评估循环集成测试：验证环境重置(reset)、智能体动作选择(select_action)、
环境步进(step)三者协作正常，确保端到端推理链路通畅。
用于在代码修改后快速验证训练/评估主回路未被破坏。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs

try:
    print("Testing environment reset and step...")
    env = AirLineEnv_Graph(data_path_or_dir='data/283.csv', seed=42)
    obs = env.reset()
    
    print("Testing agent action selection...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(model, lr=configs.lr, gamma=configs.gamma, k_epochs=configs.k_epochs, eps_clip=configs.eps_clip, device=device)
    
    obs = obs.to(device)
    
    action, _, _, _, _ = agent.select_action(obs, mask_worker=torch.zeros(configs.n_w, dtype=torch.bool).to(device))
    print("Selected action:", action)
    
    obs_next, reward, done, info = env.step(action)
    print("Step passed, reward:", reward)
    
    print("All tests passed.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Fail:", e)
