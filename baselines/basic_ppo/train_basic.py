import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd

# [ENV FIX] 防止 OpenMP 运行时冲突导致崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加根路径以便导入外部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from args_parser import get_basic_ppo_parser
from env_wrapper import init_env, standardize_env_reset, standardize_env_step, extract_flat_state_for_baselines
from utils.logger import init_logger, record_experiment_time
from utils.device_utils import get_available_device, clear_torch_cache
from utils.visualization import plot_gantt

# 基础PPO网络（仅MLP，无GAT/指针网络）
class BasicPPO(nn.Module):
    def __init__(self, state_dim, action_dim_list, hidden_dim=256):
        super(BasicPPO, self).__init__()
        # 共享MLP特征层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 策略头（适配MultiDiscrete动作空间）
        self.task_policy = nn.Linear(hidden_dim, action_dim_list[0])
        self.station_policy = nn.Linear(hidden_dim, action_dim_list[1])
        self.worker_policy = nn.Linear(hidden_dim, action_dim_list[2])
        # 价值头
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        # 策略输出（logits）
        task_logits = self.task_policy(features)
        station_logits = self.station_policy(features)
        worker_logits = self.worker_policy(features)
        # 价值输出
        value = self.value_head(features)
        return task_logits, station_logits, worker_logits, value

class BasicPPOAgent:
    def __init__(self, state_dim, action_dim_list, args, device):
        self.device = device
        self.lr = args.lr
        self.clip_epsilon = args.clip_epsilon
        self.gamma = getattr(args, 'gamma', 0.99)
        self.lamda = getattr(args, 'lamda', 0.95)
        self.action_dim_list = action_dim_list
        
        # 初始化网络
        self.model = BasicPPO(state_dim, action_dim_list).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 存储轨迹数据（每轮清空，避免内存溢出）
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []
    
    def select_action(self, state, env_for_demand=None):
        """
        选择动作并记录概率/价值
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            task_logits, station_logits, worker_logits, value = self.model(state_tensor)
            
            # [Deadlock Prevent] Cascaded Action Masking
            if env_for_demand is not None:
                t_mask_raw, s_mask_raw, w_mask_global = env_for_demand.get_masks()
                
                # 1. Mask Task
                t_mask = t_mask_raw.to(self.device).bool().unsqueeze(0)
                if t_mask.all(): return None # TOTAL DEADLOCK
                
                task_logits = task_logits.masked_fill(t_mask, -1e9)
                task_dist = Categorical(logits=task_logits)
                task_action = task_dist.sample()
                task_idx = task_action.item()
                
                # 2. Mask Station (Conditioned strictly on chosen task)
                s_mask = s_mask_raw[task_idx].to(self.device).bool().unsqueeze(0)
                station_logits = station_logits.masked_fill(s_mask, -1e9)
                station_dist = Categorical(logits=station_logits)
                station_action = station_dist.sample()
                station_idx = station_action.item()
                
                # 3. Mask Worker (Conditioned strictly on Task's Skill & Station's Lock)
                req_skill = int(env_for_demand.task_static_feat[task_idx, 1].item())
                worker_skills = env_for_demand.worker_skill_matrix.numpy()
                has_skill = worker_skills[:, req_skill] > 0.5
                
                worker_locks = env_for_demand.worker_locks
                valid_lock = (worker_locks == 0) | (worker_locks == station_idx + 1)
                
                w_mask_time = w_mask_global.numpy()
                final_w_mask_np = w_mask_time | (~has_skill) | (~valid_lock)
                w_mask = torch.tensor(final_w_mask_np, dtype=torch.bool).to(self.device).unsqueeze(0)
                
                worker_logits = worker_logits.masked_fill(w_mask, -1e9)
                worker_dist = Categorical(logits=worker_logits)
                worker_action = worker_dist.sample()
                
                mask_record = (t_mask, s_mask, w_mask)
            else:
                task_dist = Categorical(logits=task_logits)
                station_dist = Categorical(logits=station_logits)
                worker_dist = Categorical(logits=worker_logits)
                task_action = task_dist.sample()
                station_action = station_dist.sample()
                worker_action = worker_dist.sample()
                task_idx = task_action.item()
                mask_record = None
            
            # 记录log prob和value
            log_prob = task_dist.log_prob(task_action) + station_dist.log_prob(station_action) + worker_dist.log_prob(worker_action)
            value = value.item()
        
        task_idx = task_action.item()
        worker_idx = worker_action.item()
        
        if env_for_demand is not None:
            demand = int(env_for_demand.task_static_feat[task_idx, 2].item())
            demand = max(1, demand)
            team = [worker_idx]
            if demand > 1:
                if mask_record is not None:
                    # Retrieve the global valid worker mask calculated in the cascaded phase
                    final_w_mask_np = mask_record[2].cpu().numpy()[0]
                    w_valid = np.where(~final_w_mask_np)[0].tolist()
                    subs = [w for w in w_valid if w != worker_idx]
                    team.extend(subs[:demand-1])
                else:
                    team = [abs(worker_idx + i) % self.action_dim_list[2] for i in range(demand)]
        else:
             team = [worker_idx]
             
        action = (task_idx, station_action.item(), team)
        
        # 存储轨迹用于计算概率的 tuple 索引，不需要管队伍延展
        recorded_action = (task_idx, station_action.item(), worker_action.item())
        
        self.states.append(state_tensor)
        self.actions.append(recorded_action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask_record)
        
        return action
    
    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self):
        """
        计算GAE优势函数
        """
        advantages = []
        advantage = 0.0
        next_value = 0.0
        
        for reward, done, value in reversed(list(zip(self.rewards, self.dones, self.values))):
            td_error = reward + self.gamma * next_value * (1 - done) - value
            advantage = td_error + self.gamma * self.lamda * (1 - done) * advantage
            advantages.insert(0, advantage)
            next_value = value
        
        returns = np.array(advantages) + np.array(self.values)
        adv_array = np.array(advantages)
        adv_mean = np.mean(adv_array)
        adv_std = np.std(adv_array) if np.std(adv_array) > 1e-8 else 1e-8
        advantages = list((adv_array - adv_mean) / adv_std)
        return advantages, returns.tolist()
    
    def update(self, batch_size):
        """
        PPO更新，分批训练避免内存碎片
        """
        if len(self.states) < batch_size:
             self.clear_memory()
             return 0.0
             
        advantages, returns = self.compute_gae()
        
        # 转换为tensor
        states = torch.cat(self.states, dim=0).to(self.device).detach()
        log_probs_old = torch.cat(self.log_probs, dim=0).to(self.device).detach()
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # 分批更新 (多次 epochs 防止单样本浪费)
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        losses = []
        
        epochs = 4
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # 前向传播
                task_logits, station_logits, worker_logits, values = self.model(states[batch_idx])
                
                # 应用存储的合法性掩码重建相同概率约束
                batch_masks = [self.masks[i] for i in batch_idx]
                if batch_masks[0] is not None:
                    t_masks = torch.cat([m[0] for m in batch_masks], dim=0)
                    s_masks = torch.cat([m[1] for m in batch_masks], dim=0)
                    w_masks = torch.cat([m[2] for m in batch_masks], dim=0)
                    
                    task_logits = task_logits.masked_fill(t_masks, -1e9)
                    station_logits = station_logits.masked_fill(s_masks, -1e9)
                    worker_logits = worker_logits.masked_fill(w_masks, -1e9)
                
                # 重新计算动作概率
                task_dist = Categorical(logits=task_logits)
                station_dist = Categorical(logits=station_logits)
                worker_dist = Categorical(logits=worker_logits)
                
                # 取出批次动作
                batch_actions = [self.actions[i] for i in batch_idx]
                task_actions = torch.tensor([a[0] for a in batch_actions], dtype=torch.int64).to(self.device)
                station_actions = torch.tensor([a[1] for a in batch_actions], dtype=torch.int64).to(self.device)
                worker_actions = torch.tensor([a[2] for a in batch_actions], dtype=torch.int64).to(self.device)
                
                log_probs_new = task_dist.log_prob(task_actions) + station_dist.log_prob(station_actions) + worker_dist.log_prob(worker_actions)
                
                # PPO裁剪
                ratio = torch.exp(log_probs_new - log_probs_old[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values.squeeze(-1), returns[batch_idx])
                
                # 总损失
                entropy = (task_dist.entropy() + station_dist.entropy() + worker_dist.entropy()).mean()
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 反向传播 (梯度裁剪)
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                losses.append(total_loss.item())
        
        # 清空轨迹
        self.clear_memory()
        
        return np.mean(losses) if losses else 0.0

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.masks.clear()

def train_basic_ppo(args):
    # 初始化日志
    logger, exp_dir = init_logger(args, "basic_ppo_baseline")
    start_time = time.time()
    
    try:
        # 设备初始化
        device = get_available_device()
        # 环境初始化（统一接口）
        env = init_env(args, seed=args.seed)
        
        # 状态维度适配
        standardize_env_reset(env)
        flat_state = extract_flat_state_for_baselines(env)
        state_dim = flat_state.shape[0]
        
        action_dim_list = [env.num_tasks, env.num_stations, env.num_workers]
        
        # 初始化Agent
        agent = BasicPPOAgent(state_dim, action_dim_list, args, device)
        batch_size = getattr(args, 'batch_size', 64)
        
        # 训练指标
        episode_rewards = []
        episode_losses = []
        episode_makespans = []
        best_makespan = float('inf')
        
        logger.info(f"开始 Basic PPO (MLP) 训练，状态维度: {state_dim}，动作维度: {action_dim_list}，最大轮次: {args.max_episodes}")
        for ep in range(args.max_episodes):
            standardize_env_reset(env)
            state = extract_flat_state_for_baselines(env)
            
            done = False
            ep_reward = 0
            step_count = 0
            max_steps = env.num_tasks * 2  # 防止无限循环
            
            while not done and step_count < max_steps:
                step_count += 1
                
                # [Deadlock Fix: Wait-And-See]
                t_mask_raw, _, _ = env.get_masks()
                while t_mask_raw.all():
                    can_wait = env.try_wait_for_resources()
                    if not can_wait:
                        break
                    t_mask_raw, _, _ = env.get_masks()
                
                if t_mask_raw.all():
                    reward = -100.0
                    done = True
                    agent.store_reward(reward, done)
                    ep_reward += reward
                    break
                
                action = agent.select_action(state, env_for_demand=env)
                
                if action is None:
                    # [Safety Net]
                    reward = -100.0
                    done = True
                    agent.store_reward(reward, done)
                    ep_reward += reward
                    break
                    
                _, reward, done, info = standardize_env_step(env, action)
                next_state = extract_flat_state_for_baselines(env)
                
                agent.store_reward(reward, done)
                
                ep_reward += reward
                state = next_state
            
            # 记录指标
            episode_rewards.append(ep_reward)
            
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
                df.to_csv(os.path.join(exp_dir, f"Best_Schedule_BasicPPO.csv"), index=False)
                plot_gantt(best_sch, os.path.join(exp_dir, f"Best_Gantt_BasicPPO.png"))
                logger.info(f"✨ 新的最佳 BasicPPO 调度已保存! Makespan: {best_makespan:.2f} -> {exp_dir}")
            
            loss = agent.update(batch_size)
            episode_losses.append(loss)
            
            # 日志
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                avg_makespan = np.mean(episode_makespans[-10:])
                logger.info(f"Episode {ep+1:04d}/{args.max_episodes} | 奖励: {ep_reward:.2f} (Avg: {avg_reward:.2f}) | Loss: {avg_loss:.4f} | Makespan: {makespan:.2f} (Avg: {avg_makespan:.2f})")
                
            # 每100轮清理与保存一次
            if (ep + 1) % 100 == 0:
                clear_torch_cache()
                model_path = os.path.join(exp_dir, f"basic_ppo_model_ep{ep+1}.pth")
                torch.save(agent.model.state_dict(), model_path)
                
        # 保存最终模型
        model_path = os.path.join(exp_dir, "basic_ppo_model_final.pth")
        torch.save(agent.model.state_dict(), model_path)
        logger.info(f"最终模型保存至: {model_path}")
        
        # 结果归档
        results = pd.DataFrame({
            'episode': range(1, args.max_episodes+1),
            'reward': episode_rewards,
            'loss': episode_losses,
            'makespan': episode_makespans
        })
        results['avg_makespan_10'] = results['makespan'].rolling(window=10).mean()
        results.to_csv(os.path.join(exp_dir, "basic_ppo_results.csv"), index=False)
        
    except Exception as e:
        logger.error(f"Basic PPO 训练失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        record_experiment_time(logger, start_time)
        clear_torch_cache()

if __name__ == "__main__":
    parser = get_basic_ppo_parser()
    args = parser.parse_args()
    train_basic_ppo(args)
