import os
import torch
import numpy as np
from environment import AirLineEnv_Graph
from configs import configs

def init_env(args, seed=None):
    """
    统一环境初始化门面
    注意：您主干的图强化学习仍可以直接正常使用返回的 HeteroData，完全不会受到影响！
    """
    data_path = getattr(args, 'data_path', 'data/100.csv')
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.getcwd(), data_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集未找到，请检查路径: {data_path}")
        
    env_seed = seed if seed is not None else getattr(args, 'seed', 42)
    
    env = AirLineEnv_Graph(
        data_path=data_path,
        seed=env_seed
    )
    return env

def standardize_env_step(env, action):
    """
    标准化环境返回步骤，强制统一 reward 为标量 float，done 为 bool。
    防护不同 RL 基线由于 tensor / np.ndarray 类型泛滥引起的反向传播或广播警告。
    """
    state, reward, done, info = env.step(action)
    
    if isinstance(reward, torch.Tensor):
        reward = reward.item()
    elif isinstance(reward, np.ndarray):
        reward = reward.item()
    reward = float(reward)
    
    if isinstance(done, torch.Tensor):
        done = done.item()
    elif isinstance(done, np.ndarray):
        done = done.item()
    done = bool(done)
    
    return state, reward, done, info

def standardize_env_reset(env, randomize_duration=False):
    """
    标准化重置。同时为了支持不需要图神经网络的基准算法（DQN / BasicPPO），
    我们可以提供一个展平的辅助方法，而不改变原始 state 本身。
    """
    state = env.reset(randomize_duration=randomize_duration)
    return state

def extract_flat_state_for_baselines(env):
    """
    [专门供给非图神经网络基线算法（DQN, MLP PPO）使用的方法]
    从底层原生物理环境属性中直接抽取一维的 numpy state 向量。
    这样一来，原版图强化模型继续使用 PyG HeteroData，基线模型使用这个一维向量，互不打扰！
    """
    # 抽取任务完成度特征
    task_status_flat = env.task_status.flatten() # [N]
    
    # 抽取任务属性
    task_feat_flat = env.task_static_feat.flatten().numpy() # [N * TaskFeatDim]
    
    # 抽取工人空闲特征
    worker_free_flat = env.worker_free_time.flatten() # [W]
    worker_sync_flat = (worker_free_flat - env.current_time)
    worker_sync_flat = np.maximum(worker_sync_flat, 0)
    
    # [PADDING FIX] 强制将可能动态变化长度的工人数组锁定为配置好的上限长度，供 MLP 食用
    n_w_max = getattr(configs, 'n_w_max', 120)
    current_w = len(worker_sync_flat)
    if current_w < n_w_max:
        worker_sync_flat = np.pad(worker_sync_flat, (0, n_w_max - current_w), 'constant', constant_values=0)
    elif current_w > n_w_max:
        worker_sync_flat = worker_sync_flat[:n_w_max]
    
    # 拼接组装成一个庞大的一维状态
    # 确保 DQN 这种只吃固定一维向量的模型能够正常运算，且不会污染异构图结构。
    flat_state = np.concatenate([task_status_flat, task_feat_flat, worker_sync_flat])
    return flat_state
