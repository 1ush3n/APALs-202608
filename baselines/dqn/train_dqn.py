import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque

# [ENV FIX] 防止 OpenMP 运行时冲突导致崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加根路径以便导入外部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from args_parser import get_dqn_parser
from env_wrapper import init_env, standardize_env_reset, standardize_env_step, extract_flat_state_for_baselines
from utils.logger import init_logger, record_experiment_time
from utils.device_utils import get_available_device, clear_torch_cache
from utils.visualization import plot_gantt

# DQN网络（适配环境MultiDiscrete动作空间）
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim_list, hidden_dim=128):
        super(DQN, self).__init__()
        # 共享特征层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 动作分支（适配MultiDiscrete: [task, station, worker_leader]）
        self.task_head = nn.Linear(hidden_dim, action_dim_list[0])
        self.station_head = nn.Linear(hidden_dim, action_dim_list[1])
        self.worker_head = nn.Linear(hidden_dim, action_dim_list[2])
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        task_logits = self.task_head(x)
        station_logits = self.station_head(x)
        worker_logits = self.worker_head(x)
        return task_logits, station_logits, worker_logits

class DQNAgent:
    def __init__(self, state_dim, action_dim_list, args, device):
        self.device = device
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.action_dim_list = action_dim_list
        
        # 初始化网络
        self.model = DQN(state_dim, action_dim_list).to(device)
        self.target_model = DQN(state_dim, action_dim_list).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放池（限制大小，避免内存溢出）
        self.memory = deque(maxlen=getattr(args, 'memory_size', 10000))
        
    def select_action(self, state, env_for_demand=None):
        """
        带探索的动作选择，适配MultiDiscrete动作空间 (Cascaded Masking)
        """
        if np.random.rand() <= self.epsilon:
            # 随探索，强制联级拓扑合法
            if env_for_demand is not None:
                t_mask_raw, s_mask_raw, w_mask_global = env_for_demand.get_masks()
                
                # 1. Random Task
                t_valid = torch.where(~t_mask_raw)[0].cpu().numpy()
                if len(t_valid) == 0: return None # TOTAL DEADLOCK
                task = int(np.random.choice(t_valid))
                
                # 2. Random Station strictly relying on Task
                s_mask_t = s_mask_raw[task].cpu().numpy()
                s_valid = np.where(~s_mask_t)[0]
                station = int(np.random.choice(s_valid)) if len(s_valid) > 0 else 0
                
                # 3. Random Worker strictly relying on Task & Station
                req_skill = int(env_for_demand.task_static_feat[task, 1].item())
                worker_skills = env_for_demand.worker_skill_matrix.numpy()
                has_skill = worker_skills[:, req_skill] > 0.5
                worker_locks = env_for_demand.worker_locks
                valid_lock = (worker_locks == 0) | (worker_locks == station + 1)
                
                final_w_mask_np = w_mask_global.numpy() | (~has_skill) | (~valid_lock)
                w_valid = np.where(~final_w_mask_np)[0].tolist()
                worker = int(np.random.choice(w_valid)) if len(w_valid) > 0 else 0
                
                demand = int(env_for_demand.task_static_feat[task, 2].item())
                demand = max(1, demand)
                team = [worker]
                if demand > 1:
                    subs = [w for w in w_valid if w != worker]
                    team.extend(subs[:demand-1])
            else:
                task = np.random.randint(0, self.action_dim_list[0])
                station = np.random.randint(0, self.action_dim_list[1])
                worker = np.random.randint(0, self.action_dim_list[2])
                team = [worker]
        else:
            # 贪心动作（利用）
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                task_logits, station_logits, worker_logits = self.model(state_tensor)
                
                # [Deadlock Prevent] Cascaded Action Masking
                if env_for_demand is not None:
                    t_mask_raw, s_mask_raw, w_mask_global = env_for_demand.get_masks()
                    
                    # 1. Mask Task
                    t_mask = t_mask_raw.to(self.device).bool().unsqueeze(0)
                    if t_mask.all(): return None # TOTAL DEADLOCK
                    
                    task_logits = task_logits.masked_fill(t_mask, -1e9)
                    task = torch.argmax(task_logits).item()
                    
                    # 2. Mask Station conditioned on Task
                    s_mask = s_mask_raw[task].to(self.device).bool().unsqueeze(0)
                    station_logits = station_logits.masked_fill(s_mask, -1e9)
                    station = torch.argmax(station_logits).item()
                    
                    # 3. Mask Worker conditioned on Task & Station
                    req_skill = int(env_for_demand.task_static_feat[task, 1].item())
                    worker_skills = env_for_demand.worker_skill_matrix.numpy()
                    has_skill = worker_skills[:, req_skill] > 0.5
                    
                    worker_locks = env_for_demand.worker_locks
                    valid_lock = (worker_locks == 0) | (worker_locks == station + 1)
                    
                    final_w_mask_np = w_mask_global.numpy() | (~has_skill) | (~valid_lock)
                    w_mask = torch.tensor(final_w_mask_np, dtype=torch.bool).to(self.device).unsqueeze(0)
                    
                    worker_logits = worker_logits.masked_fill(w_mask, -1e9)
                    worker = torch.argmax(worker_logits).item()
                    
                    demand = int(env_for_demand.task_static_feat[task, 2].item())
                    demand = max(1, demand)
                    team = [worker]
                    if demand > 1:
                        w_valid = np.where(~final_w_mask_np)[0].tolist()
                        subs = [w for w in w_valid if w != worker]
                        team.extend(subs[:demand-1])
                        
                    if t_mask_raw[task].item() == True or s_mask_raw[task, station].item() == True:
                        print(f"CRITICAL: DQN Forced Invalid Action! Task={task}, Station={station}, Team={team}")
                        
                    
                else:
                    task = torch.argmax(task_logits).item()
                    station = torch.argmax(station_logits).item()
                    worker = torch.argmax(worker_logits).item()
                    team = [worker]
                
        return (task, station, team)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        """
        经验回放，分批训练避免内存溢出 (Refactored for Vectorization & Safety)
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # 批量解包 (Unzip)
        batch = [self.memory[idx] for idx in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        # [安全补丁: 维度断言] 防止因为异构环境抽出错误长度导致的隐性跑乱，这是避免静默失败的终极防御
        expected_state_dim = self.model.fc1.in_features
        assert state_tensor.shape == (batch_size, expected_state_dim), f"State shape mismatch: {state_tensor.shape} != {(batch_size, expected_state_dim)}"
        
        # 批量取得 actions (Task, Station, WorkerLeader)
        task_a = torch.tensor([a[0] for a in actions], dtype=torch.long).to(self.device)
        station_a = torch.tensor([a[1] for a in actions], dtype=torch.long).to(self.device)
        worker_a = torch.tensor([a[2][0] for a in actions], dtype=torch.long).to(self.device) # Only consider leader
        
        # 当前Q值预测 (Batched)
        task_logits, station_logits, worker_logits = self.model(state_tensor)
        
        # 提取被选中动作的 Q 值
        batch_idx = torch.arange(batch_size, device=self.device)
        q_task = task_logits[batch_idx, task_a]
        q_station = station_logits[batch_idx, station_a]
        q_worker = worker_logits[batch_idx, worker_a]
        
        q_current = (q_task + q_station + q_worker) / 3.0
        
        # 目标网络计算目标Q值 (Batched)
        with torch.no_grad():
            next_task_logits, next_station_logits, next_worker_logits = self.target_model(next_state_tensor)
            # 取最大 Q 值
            next_q_task = next_task_logits.max(dim=1)[0]
            next_q_station = next_station_logits.max(dim=1)[0]
            next_q_worker = next_worker_logits.max(dim=1)[0]
            next_q = (next_q_task + next_q_station + next_q_worker) / 3.0
            
        # 只有 not done 的部分才会累加 next_q
        q_target = reward_tensor + self.gamma * next_q * (1 - done_tensor)
        
        # 计算全局损失
        loss = self.loss_fn(q_current, q_target)
        
        # 全局一次反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

def train_dqn(args):
    # 初始化日志
    logger, exp_dir = init_logger(args, "dqn_baseline")
    start_time = time.time()
    
    try:
        # 设备初始化
        device = get_available_device()
        # 环境初始化（统一接口）
        env = init_env(args, seed=args.seed)
        
        # 状态维度适配（使用降维展平方法）
        standardize_env_reset(env)
        flat_state = extract_flat_state_for_baselines(env)
        state_dim = flat_state.shape[0]
        
        action_dim_list = [env.num_tasks, env.num_stations, env.num_workers]
        
        # 初始化Agent
        agent = DQNAgent(state_dim, action_dim_list, args, device)
        batch_size = getattr(args, 'batch_size', 32)
        
        # 训练指标
        episode_rewards = []
        episode_losses = []
        episode_makespans = []
        best_makespan = float('inf')
        
        # 训练循环
        logger.info(f"开始 DQN 训练，状态维度: {state_dim}，动作维度: {action_dim_list}，最大轮次: {args.max_episodes}")
        for ep in range(args.max_episodes):
            standardize_env_reset(env)
            state = extract_flat_state_for_baselines(env)
            done = False
            ep_reward = 0
            ep_loss = 0
            step_count = 0
            max_steps = env.num_tasks * 2  # 防止无限循环
            
            while not done and step_count < max_steps:
                step_count += 1
                # [Deadlock Fix: Wait-And-See]
                t_mask_raw, _, _ = env.get_masks()
                while t_mask_raw.all():
                    if not env.try_wait_for_resources():
                        # 彻底死锁
                        break
                    t_mask_raw, _, _ = env.get_masks()
                
                if t_mask_raw.all():
                    # 彻底死锁
                    reward = -100.0
                    done = True
                    agent.remember(state, (0, 0, [0]), reward, state, done)
                    ep_reward += reward
                    break
                    
                # 选择动作
                action = agent.select_action(state, env_for_demand=env)
                
                if action is None:
                    # [Safety Net] 如果 select_action 内部仍然发生计算错误返回了 None
                    reward = -100.0
                    done = True
                    agent.remember(state, (0, 0, [0]), reward, state, done)
                    ep_reward += reward
                    break
                    
                # 执行动作
                _, reward, done, info = standardize_env_step(env, action)
                next_state = extract_flat_state_for_baselines(env)
                
                # 存储经验
                agent.remember(state, action, reward, next_state, done)
                # 累加奖励
                ep_reward += reward
                # 经验回放
                loss = agent.replay(batch_size)
                ep_loss += loss
                # 更新状态
                state = next_state
            
            # 记录指标
            episode_rewards.append(ep_reward)
            episode_losses.append(ep_loss / step_count if step_count > 0 else 0)
            
            makespan = np.max(env.station_wall_clock) if len(env.assigned_tasks) == env.num_tasks else 99999.0
            episode_makespans.append(makespan)
            
            if makespan < best_makespan and len(env.assigned_tasks) == env.num_tasks:
                best_makespan = makespan
                best_sch = env.assigned_tasks.copy()
                # 存储最新最好成绩的 CSV 和 Gantt
                tasks_data = []
                for (tid, sid, team, start, end) in best_sch:
                     tasks_data.append({
                         'TaskID': tid,
                         'StationID': sid + 1,
                         'Team': str(team),
                         'Start': start,
                         'End': end,
                         'Duration': end - start
                     })
                df = pd.DataFrame(tasks_data)
                df.to_csv(os.path.join(exp_dir, f"Best_Schedule_DQN.csv"), index=False)
                plot_gantt(best_sch, os.path.join(exp_dir, f"Best_Gantt_DQN.png"))
                logger.info(f"✨ 新的最佳 DQN 调度已保存! Makespan: {best_makespan:.2f} -> {exp_dir}")
            
            # 每10轮打印日志
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                avg_makespan = np.mean(episode_makespans[-10:])
                logger.info(f"Episode {ep+1}/{args.max_episodes} | 平均奖励: {avg_reward:.2f} | 损失: {avg_loss:.4f} | Makespan: {avg_makespan:.2f} | Epsilon: {agent.epsilon:.4f}")
            
            # 每50轮更新目标网络
            if (ep + 1) % 50 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                # 清理缓存
                clear_torch_cache()
        
        # 保存模型
        model_path = os.path.join(exp_dir, "dqn_model.pth")
        torch.save(agent.model.state_dict(), model_path)
        logger.info(f"模型保存至: {model_path}")
        
        # 结果归档
        results = pd.DataFrame({
            'episode': range(1, args.max_episodes+1),
            'reward': episode_rewards,
            'loss': episode_losses,
            'makespan': episode_makespans
        })
        results['avg_reward_10'] = results['reward'].rolling(window=10).mean()
        results['avg_makespan_10'] = results['makespan'].rolling(window=10).mean()
        results.to_csv(os.path.join(exp_dir, "dqn_results.csv"), index=False)
        
    except Exception as e:
        logger.error(f"DQN训练失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        record_experiment_time(logger, start_time)
        clear_torch_cache()

if __name__ == "__main__":
    parser = get_dqn_parser()
    args = parser.parse_args()
    train_dqn(args)
