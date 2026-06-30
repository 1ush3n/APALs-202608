import numpy as np
import random
import time
import copy
import pandas as pd
import argparse
import sys
import os

# [Hotfix 2026-03-13] 修复 Matplotlib/PyTorch 混合加载导致的 OpenMP 多重运行时崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 动态添加根目录，以便导入环境和配置模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from environment import AirLineEnv_Graph
from configs import configs
from utils.visualization import plot_gantt

class GeneticAlgorithmScheduler:
    """
    针对飞机装配线的遗传算法基线调度器 (GA)
    
    采用两段式实数/整数编码机制：
    1. 任务优先序列 (Task Priority/Sequence)
    2. 站位与工人的指派映射 (Assignment Map)
    
    使用基于拓扑排序的安全交叉与变异机制保障约束不被破坏。
    """
    def __init__(self, env, pop_size=50, max_gen=100, cx_pb=0.8, mut_pb=0.2):
        self.env = env
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        
        self.num_tasks = env.num_tasks
        self.num_stations = env.num_stations
        self.num_workers = env.num_workers
        
        # 预先重置环境以获得拓扑字典
        self.env.reset()
        self.env.skip_obs_building = True
        
        # 预存所有任务的合法依赖关系以保障生成合法的拓扑序列
        self.predecessors = env.predecessors
        
    def _create_individual(self):
        """
        初始化单挑染色体：全优先级序列表示
        Chrom1 (seq_pref): 工序选择优先级 (shape: [num_tasks])
        Chrom2 (station_pref): 站位选择优先级 (shape: [num_tasks, num_stations])
        Chrom3 (team_pref): 工人选择优先级 (shape: [num_tasks, num_workers])
        """
        seq_pref = np.random.rand(self.num_tasks).tolist()
        station_pref = np.random.rand(self.num_tasks, self.num_stations).tolist()
        team_pref = np.random.rand(self.num_tasks, self.num_workers).tolist()
        
        return {'seq_pref': seq_pref, 'station_pref': station_pref, 'team_pref': team_pref}

    def _init_population(self):
        return [self._create_individual() for _ in range(self.pop_size)]
        
    def _evaluate_fitness(self, individual):
        """
        使用沙盒仿真模拟这条染色体的调度逻辑，返回适应度 (makespan)
        越小越好。
        """
        # 直接重置共享环境，避免巨大的 PyTorch/PyG deepcopy 开销
        sim_env = self.env
        sim_env.reset() # Soft reset
        
        seq_prefs = np.array(individual['seq_pref'])
        station_prefs = np.array(individual['station_pref'])
        team_prefs = np.array(individual['team_pref'])
        
        # [安全补丁: 强制标量化/NumPy化]
        # 把底层可能返回的多维 Tensor 全部降维剥离成纯净的 numpy array，以此彻底免疫类型混合运算导致的致命报错
        worker_skill_matrix = sim_env.worker_skill_matrix.numpy() if hasattr(sim_env.worker_skill_matrix, 'numpy') else sim_env.worker_skill_matrix
        worker_locks = sim_env.worker_locks.numpy() if hasattr(sim_env.worker_locks, 'numpy') else sim_env.worker_locks
        task_static_feat = sim_env.task_static_feat.numpy() if hasattr(sim_env.task_static_feat, 'numpy') else sim_env.task_static_feat
        
        done = False
        total_makespan = float('inf')
        total_balance_std = float('inf')
        
        # 强制步数容错
        max_limit = self.num_tasks * 3 
        step = 0
        
        while not done and step < max_limit:
            step += 1
            t_mask_raw, s_mask_raw, w_mask_raw = sim_env.get_masks()
            t_mask = t_mask_raw.numpy() if hasattr(t_mask_raw, 'numpy') else t_mask_raw
            s_mask = s_mask_raw.numpy() if hasattr(s_mask_raw, 'numpy') else s_mask_raw
            
            # [公平性对齐] 引入原地等待逻辑
            if t_mask.all():
                if sim_env.try_wait_for_resources():
                     continue 
                else:
                     return 999999.0, (99999.0, 9999.0, [])
            
            # 1. 任务选择 (基于 seq_pref)
            available_tasks = [i for i in range(self.num_tasks) if not t_mask[i]]
            best_task_id = max(available_tasks, key=lambda x: seq_prefs[x])
            
            # 2. 站位选择 (基于 station_pref)
            valid_stations = [s for s in range(self.num_stations) if not s_mask[best_task_id, s]]
            if not valid_stations:
                if sim_env.try_wait_for_resources(): continue
                return 999999.0, (99999.0, 9999.0, [])
                
            desired_station = max(valid_stations, key=lambda s: station_prefs[best_task_id, s])
                
            # 3. 定人员 (关键漏洞修复：每一轮决策都要获取环境中最新的 Lock 状态)
            task_type_idx = int(task_static_feat[best_task_id, 1].item())
            req_demand = max(1, int(task_static_feat[best_task_id, 2].item()))
            
            # 实时捕获环境快照 (实时反映已分配任务导致的占用情况)
            current_locks = sim_env.worker_locks
            current_skills = sim_env.worker_skill_matrix.numpy()
            
            skilled_available = []
            for w in range(self.num_workers):
                # A. 技能匹配
                if current_skills[w, task_type_idx] > 0.5:
                    # B. 站位锁定匹配 (0=未绑定, desired_station+1=已绑定到该目标岛屿)
                    if current_locks[w] == 0 or current_locks[w] == (desired_station + 1):
                        skilled_available.append(w)
            
            if len(skilled_available) < req_demand:
                 # 核心对齐：如果当前在该站位凑不齐人（可能是因为该站位的人正在干别的活），执行等待
                 if sim_env.try_wait_for_resources(): continue
                 return 999999.0, (99999.0, 9999.0, [])
                 
            # 择优组队
            prefs = team_prefs[best_task_id]
            skilled_available.sort(key=lambda w: prefs[w], reverse=True)
            selected_team = skilled_available[:req_demand]
            
            # 4. 执行演算
            action = (best_task_id, desired_station, selected_team)
            _, _, done, _ = sim_env.step(action)
            
        if done:
            total_makespan = np.max(sim_env.station_wall_clock)
            total_balance_std = np.std(sim_env.station_loads)
            
        # 以 makespan 为第一适应度 (越小越好)，balance 为次要 (与 RL 环境中的惩罚权重 1:1 绝对对齐)
        fitness = total_makespan + 1.0 * total_balance_std 
        return fitness, (total_makespan, total_balance_std, sim_env.assigned_tasks)

    def _crossover(self, p1, p2):
        """
        全量优先级均匀交叉 (Uniform Crossover)。
        由于全部转为浮点优先级数组，我们摒弃复杂的序列交叉，使用纯浮点的均匀混合。
        """
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        
        for i in range(self.num_tasks):
            # Task preference crossover
            if random.random() < 0.5:
                c1['seq_pref'][i], c2['seq_pref'][i] = p2['seq_pref'][i], p1['seq_pref'][i]
            
            # Station preference crossover
            for j in range(self.num_stations):
                if random.random() < 0.5:
                    c1['station_pref'][i][j], c2['station_pref'][i][j] = p2['station_pref'][i][j], p1['station_pref'][i][j]
                    
            # Team preference crossover
            for j in range(self.num_workers):
                 if random.random() < 0.5:
                    c1['team_pref'][i][j], c2['team_pref'][i][j] = p2['team_pref'][i][j], p1['team_pref'][i][j]
                    
        return c1, c2

    def _mutate(self, ind):
        """变异算子: 添加高斯扰动"""
        for i in range(self.num_tasks):
            # 偏好变异：添加扰动
            if random.random() < self.mut_pb:
                 ind['seq_pref'][i] += random.gauss(0, 0.2)
                 
            for j in range(self.num_stations):
                 if random.random() < self.mut_pb:
                      ind['station_pref'][i][j] += random.gauss(0, 0.2)
                      
            for j in range(self.num_workers):
                 if random.random() < self.mut_pb:
                      ind['team_pref'][i][j] += random.gauss(0, 0.2)
                
        return ind

    def run(self):
        print(f"--- 启动运筹学遗传算法 (GA) 基线 ---")
        print(f"配置: PopSize={self.pop_size}, MaxGen={self.max_gen}")
        
        pop = self._init_population()
        
        best_overall_individual = None
        best_overall_fitness = float('inf')
        best_overall_metrics = None
        
        start_t = time.time()
        
        for g in range(self.max_gen):
            fitnesses_and_metrics = []
            
            # 1. 评估种群
            for ind in pop:
                fit, metrics = self._evaluate_fitness(ind)
                fitnesses_and_metrics.append((fit, metrics, ind))
                
                if fit < best_overall_fitness:
                    best_overall_fitness = fit
                    best_overall_metrics = metrics
                    best_overall_individual = copy.deepcopy(ind)
                    
            # 2. 选择 (Tournament Selection)
            # 根据适应度排序 (从小到大，因越小越好)
            fitnesses_and_metrics.sort(key=lambda x: x[0])
            
            # Print stats
            best_g_fit = fitnesses_and_metrics[0][0]
            avg_g_fit = np.mean([x[0] for x in fitnesses_and_metrics if x[0] < 99999.0]) # 排除死锁异常的
            print(f"[Gen {g+1}/{self.max_gen}] Best Fit: {best_g_fit:.2f} (Makespan: {fitnesses_and_metrics[0][1][0]:.2f}) | Avg Fit: {avg_g_fit:.2f}")
            
            # 精英保留策略 (Elitism)
            next_pop = [x[2] for x in fitnesses_and_metrics[:int(self.pop_size * 0.1)]]
            
            # 3. 产生下一代
            while len(next_pop) < self.pop_size:
                # 锦标赛选择选出父亲母亲
                t_size = 3
                p1_candidates = random.sample(fitnesses_and_metrics, t_size)
                p2_candidates = random.sample(fitnesses_and_metrics, t_size)
                
                p1 = min(p1_candidates, key=lambda x: x[0])[2]
                p2 = min(p2_candidates, key=lambda x: x[0])[2]
                
                if random.random() < self.cx_pb:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                    
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                next_pop.extend([c1, c2])
                
            pop = next_pop[:self.pop_size] # 截断溢出
            
        time_elapsed = time.time() - start_t
        print(f"\n--- GA 基线运行结束 (耗时: {time_elapsed:.1f}s) ---")
        
        makespan, balance_std, assigned_tasks = best_overall_metrics
        print(f"最好成绩 -> Makespan: {makespan:.2f} h | Balance Std: {balance_std:.2f}")
        
        if assigned_tasks:
            os.makedirs("results", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 先保存 CSV 保证排程数据安全落地
            tasks_data = []
            for (tid, sid, team, start, end) in assigned_tasks:
                  # 获取原始 AO 号
                  task_ao = self.env.raw_data['task_df']['task_id'].iloc[tid] if 'task_df' in self.env.raw_data else str(tid)
                  # 获取真实的序号
                  real_id = self.env.raw_data['task_df']['序号'].iloc[tid] if 'task_df' in self.env.raw_data and '序号' in self.env.raw_data['task_df'].columns else tid
                  
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
            csv_path = f"results/GA_schedule_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[*] 成功保存排程明细至 -> {csv_path}")
            
            # 再绘制可能因环境报错的甘特图
            gantt_path = f"results/GA_gantt_{timestamp}.png"
            try:
                plot_gantt(assigned_tasks, gantt_path)
                print(f"[*] 成功保存甘特图至 -> {gantt_path}")
            except Exception as e:
                print(f"[!] 绘制甘特图失败: {e}")
            
            
        return makespan, balance_std, assigned_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径 (.csv)')
    parser.add_argument('--pop_size', type=int, default=30)
    parser.add_argument('--max_gen', type=int, default=20)
    args = parser.parse_args()
    
    data_path = args.data_path if args.data_path else configs.data_file_path
    if not os.path.exists(data_path) and os.path.exists(os.path.join(os.getcwd(), data_path)):
         data_path = os.path.join(os.getcwd(), data_path)
         
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=42)
    
    ga_solver = GeneticAlgorithmScheduler(env, pop_size=args.pop_size, max_gen=args.max_gen)
    ga_solver.run()
