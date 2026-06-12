import os
import sys
import time
import numpy as np
import pandas as pd

# 添加根路径以便导入外部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# [Hotfix] 修复多运行时 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from args_parser import get_heuristic_parser
from env_wrapper import init_env, standardize_env_reset, standardize_env_step
from utils.logger import init_logger, record_experiment_time
from utils.device_utils import clear_torch_cache
from utils.visualization import plot_gantt

def spt_policy(env):
    """
    SPT策略：优先选择耗时最短的就绪任务，在合法的站位中指派符合条件的工人
    """
    # 获取掩码（符合拓扑约束的可用任务）
    task_mask, station_mask, _ = env.get_masks()
    if task_mask.all():
        return None  # 无合法的符合拓扑的就绪任务
        
    ready_tasks = np.where(~task_mask.numpy())[0] if hasattr(task_mask, 'numpy') else np.where(~task_mask)[0]
    
    # 按耗时升序排序（SPT核心）
    task_durations = env.task_static_feat[ready_tasks, 0].numpy()
    sorted_task_idx = np.argsort(task_durations)
    
    # 尝试按 SPT 顺序找到第一个拥有可分配站位的任务
    selected_task = None
    selected_station = None
    for tid in ready_tasks[sorted_task_idx]:
        # 查找该任务的合法站位 (s_mask 为 False 的位)
        valid_stations = np.where(~station_mask[tid].numpy())[0] if hasattr(station_mask, 'numpy') else np.where(~station_mask[tid])[0]
        if len(valid_stations) > 0:
            selected_task = tid
            # 简单策略：从合法站位中随机选一个
            selected_station = int(np.random.choice(valid_stations))
            break
            
    if selected_task is None:
        return None # 暂时没有任务能找到空坑位
    
    # 选择符合技能的工人（严格对齐技能与岛屿锁定逻辑）
    task_skill = int(env.task_static_feat[selected_task, 1].item())
    num_workers_req = int(env.task_static_feat[selected_task, 2].item())
    
    # 找到在该站位下合法的有技能工人
    skilled_available = []
    for w in range(env.num_workers):
        if env.worker_skill_matrix[w, task_skill] > 0.5:
            # 站位锁定匹配 (0=未绑定, s+1=绑定到某岛屿)
            if env.worker_locks[w] == 0 or env.worker_locks[w] == (selected_station + 1):
                skilled_available.append(w)
    
    if len(skilled_available) < num_workers_req:
         return None # 暂时凑不齐人
         
    # 随机选择工人
    selected_workers = np.random.choice(skilled_available, size=num_workers_req, replace=False).tolist()

    action = (selected_task, selected_station, selected_workers) 
    return action

def run_spt(args):
    # 初始化日志
    logger, exp_dir = init_logger(args, "spt_heuristic")
    start_time = time.time()
    
    try:
        # 初始化环境（统一接口）
        env = init_env(args, seed=args.seed)
        
        # 实验指标初始化
        total_makespan = []
        total_reward = []
        total_duration = []
        best_schedules = []
        best_makespan = float('inf')
        
        # 运行SPT策略
        num_runs = getattr(args, 'num_runs', 5)
        for run in range(num_runs):
            logger.info(f"SPT策略运行轮次: {run+1}/{num_runs}")
            state = standardize_env_reset(env)
            done = False
            ep_reward = 0
            step_count = 0
            max_steps = env.num_tasks * 4  # 增加步数，因为包含等待步
            
            run_start_time = time.time()
            while not done and step_count < max_steps:
                step_count += 1
                action = spt_policy(env)
                if action is None:
                    # 尝试等待资源释放
                    if env.try_wait_for_resources():
                         continue
                    else:
                         logger.error(f"轮次{run+1}：检测到真实死锁，终止本轮")
                         break
                state, reward, done, info = standardize_env_step(env, action)
                ep_reward += reward
            
            run_duration = time.time() - run_start_time
            makespan = np.max(env.station_wall_clock) if done else 99999.0
            
            total_makespan.append(makespan)
            total_reward.append(ep_reward)
            total_duration.append(run_duration)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedules = env.assigned_tasks if makespan < 99999.0 else []
            logger.info(f"轮次{run+1} - Makespan: {makespan:.2f}, 总奖励: {ep_reward:.2f}")
        
        if best_schedules:
            plot_gantt(best_schedules, os.path.join(exp_dir, "SPT_gantt.png"))
            tasks_data = []
            for (tid, sid, team, start, end) in best_schedules:
                  # 获取原始 AO 号
                  task_ao = env.raw_data['task_df']['task_id'].iloc[tid] if 'task_df' in env.raw_data else str(tid)
                  # 获取真实的序号
                  real_id = env.raw_data['task_df']['序号'].iloc[tid] if 'task_df' in env.raw_data and '序号' in env.raw_data['task_df'].columns else tid
                  
                  tasks_data.append({
                      'TaskID': real_id,
                      'TaskAO': task_ao,
                      'StationID': sid + 1,
                      'Team': str(team),
                      'Start': start,
                      'End': end,
                      'Duration': end - start
                  })
            df = pd.DataFrame(tasks_data)
            df.to_csv(os.path.join(exp_dir, "SPT_schedule.csv"), index=False)
            logger.info(f"已导出最好一轮的甘特图与明细至: {exp_dir}")

        logger.info(f"平均Makespan: {np.mean(total_makespan):.2f}, 平均推理耗时: {np.mean(total_duration):.4f}秒")
        
    except Exception as e:
        logger.error(f"SPT实验执行失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理资源
        record_experiment_time(logger, start_time)
        clear_torch_cache()

if __name__ == "__main__":
    parser = get_heuristic_parser()
    args = parser.parse_args()
    run_spt(args)
