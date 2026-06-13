# -*- coding: utf-8 -*-
"""
多进程向量化环境快照重建与同步切图单元测试。
主要验证 VectorEnv 异步多进程下的重建流程和特征提取一致性，确保在高频切图与训练下不崩溃。
"""

import sys
import io
import os
import math
from pathlib import Path
import torch
import numpy as np
import random
from typing import Dict, Any, List

# 修复 Linux 环境变量 OMP_NUM_THREADS 非法值导致的 libgomp 报错
omp_threads = os.environ.get("OMP_NUM_THREADS", "")
try:
    int(omp_threads)
except ValueError:
    os.environ["OMP_NUM_THREADS"] = "1"

# 强制将标准输出设为 UTF-8 防止 Windows 终端 Emoji 崩溃
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# 强制使用 pathlib 规范化跨平台项目根目录导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs import configs
from utils.vector_env import VectorEnv, EnvCreator

def set_seed(seed: int = 42) -> None:
    """
    全局锁种子函数，确保测试结果 100% 可复现。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_vector_env_rebuild_and_switch() -> None:
    # 锁定种子
    set_seed(42)
    
    # 覆写 configs 中的部分超参数，以加快并行化子进程初始化速度并对齐基准
    configs.n_w = 80
    configs.n_m = 5
    configs.max_slots_per_station = 3
    configs.enable_dynamic_events = False
    configs.enable_material_delay = False
    
    # 使用 pathlib 规范化数据集目录
    datasets_dir = PROJECT_ROOT / "data" / "random_datasets"
    assert datasets_dir.exists(), f"数据集目录不存在: {datasets_dir}"
    
    print("=" * 70)
    print("🚀 [TEST] 开始执行多进程向量环境快照重建与同步切图测试...")
    print("=" * 70)
    
    # 1. 实例化多进程向量环境 (使用 2 个并行子进程)
    print("\n[Step 1] 初始化 EnvCreator 与 VectorEnv...")
    make_env = EnvCreator(str(datasets_dir), seed_offset=100)
    vec_env = VectorEnv(make_env, num_envs=2)
    
    try:
        # 2. 调用 reset_all 并验证局部代理缓存状态
        print("\n[Step 2] 异步广播复位环境 (reset_all)...")
        obs_list = vec_env.reset_all(randomize_duration=True, randomize_workers=True)
        assert len(obs_list) == 2, f"期望返回 2 个观测状态，但得到 {len(obs_list)}"
        
        # 验证初始缓存属性
        for i, env_proxy in enumerate(vec_env.envs):
            assert env_proxy.num_tasks is not None, f"进程 {i} 静态任务数量 num_tasks 未正确缓存"
            assert env_proxy.ideal_makespan is not None, f"进程 {i} 理想完工时间 ideal_makespan 未正确缓存"
            assert env_proxy.mean_task_time is not None, f"进程 {i} 平均任务工时 mean_task_time 未正确缓存"
            assert env_proxy.current_time == 0.0, f"进程 {i} 初始 current_time 应该为 0.0"
            print(f"  -> 子环境 {i} 属性验证通过: Tasks={env_proxy.num_tasks}, Ideal Makespan={env_proxy.ideal_makespan:.2f}, Mean Task Time={env_proxy.mean_task_time:.2f}")

        # 3. 抽取并验证快照 (get_state_snapshot_all)
        print("\n[Step 3] 批量获取并校验状态快照...")
        snapshots = vec_env.get_state_snapshot_all()
        assert len(snapshots) == 2
        
        # 4. 验证 Proxy 本地就地延迟静态重建，并严格对比对数归一化与竞争特征
        print("\n[Step 4] 执行本地延迟重建 (rebuild_state_from_snapshot)...")
        for i, env_proxy in enumerate(vec_env.envs):
            snap = snapshots[i]
            
            # 使用代理的本地重建接口 (不经过 IPC)
            rebuilt_data = env_proxy.rebuild_state_from_snapshot(snap)
            
            # 校验重建后的节点维度
            num_rebuilt_tasks = rebuilt_data['task'].x.shape[0]
            num_rebuilt_workers = rebuilt_data['worker'].x.shape[0]
            num_rebuilt_stations = rebuilt_data['station'].x.shape[0]
            
            assert num_rebuilt_tasks == env_proxy.num_tasks, f"重建任务数 ({num_rebuilt_tasks}) 与代理缓存 ({env_proxy.num_tasks}) 不一致"
            assert num_rebuilt_workers == len(snap['worker_free_time']), f"重建工人数 ({num_rebuilt_workers}) 与快照不一致"
            assert num_rebuilt_stations == configs.n_m, f"重建站位数 ({num_rebuilt_stations}) 与配置不一致"
            
            # 5. 深入校验特征重建精准度
            station_x = rebuilt_data['station'].x
            task_x = rebuilt_data['task'].x
            worker_x = rebuilt_data['worker'].x
            
            # ① 校验站位负载相对竞争特征 [5] 与 [6] 维
            loads = snap['station_loads']
            sum_loads = np.sum(loads)
            max_load = np.max(loads)
            expected_load_comp_5 = loads / (sum_loads + 1e-6)
            expected_load_comp_6 = loads / (max_load + 1e-6)
            
            assert torch.allclose(station_x[:, 5], torch.tensor(expected_load_comp_5, dtype=torch.float), atol=1e-5), f"环境 {i} 站位负载相对占比特征 station_x[:, 5] 重建错误"
            assert torch.allclose(station_x[:, 6], torch.tensor(expected_load_comp_6, dtype=torch.float), atol=1e-5), f"环境 {i} 站位负载最大占比特征 station_x[:, 6] 重建错误"
            
            # ② 校验槽位释放时间特征是否引入对数归一化 [4] 维
            # 初始状态下，assigned_tasks 应该为空，所以槽位释放等待时间应为 0.0，对数 log1p(0) = 0
            assert torch.allclose(station_x[:, 4], torch.zeros(configs.n_m), atol=1e-5), f"环境 {i} 初始槽位等待时间不为 0"
            
            # ③ 校验物料等待时间 [17] 维与工人预估等待时间 [11] 维的对数归一化
            # 初始物料延迟和工人空闲都是 0，log1p(0) = 0
            assert torch.allclose(task_x[:, 17], torch.zeros(env_proxy.num_tasks), atol=1e-5), f"环境 {i} 初始任务物料等待时间特征不为 0"
            assert torch.allclose(worker_x[:, 11], torch.zeros(num_rebuilt_workers), atol=1e-5), f"环境 {i} 初始工人等待时间特征不为 0"
            
            print(f"  -> 子环境 {i} 重建特征精准校验成功 (包含对数特征与负载相对竞争特征)")

        # 6. 同步广播切图测试
        print("\n[Step 5] 异步广播同步切换数据集测试 (switch_dataset_all)...")
        # 随机指定一个新索引进行切图
        num_datasets = len(vec_env.envs[0].dataset_pool)
        print(f"  当前图纸池包含图纸数: {num_datasets}")
        
        # 随机切图
        next_idx = random.randint(0, num_datasets - 1)
        print(f"  -> 广播所有子进程同步切换至图纸索引: {next_idx}")
        vec_env.switch_dataset_all(next_idx)
        
        # 验证所有进程静态缓存是否已经完美对齐
        tasks_0 = vec_env.envs[0].num_tasks
        tasks_1 = vec_env.envs[1].num_tasks
        assert tasks_0 == tasks_1, f"切图后多进程任务数不一致: Env0={tasks_0}, Env1={tasks_1}"
        
        makespan_0 = vec_env.envs[0].ideal_makespan
        makespan_1 = vec_env.envs[1].ideal_makespan
        assert math.isclose(makespan_0, makespan_1, rel_tol=1e-5), f"切图后多进程理想完工时间不一致: Env0={makespan_0}, Env1={makespan_1}"
        
        print(f"  -> 广播切图静态属性同步验证通过. 新任务数: {tasks_0}, 新理想完工时间: {makespan_0:.2f}")

        # 7. 在新图纸上执行一轮步进与快照重建验证 (保障步进不报 KeyError/DimensionMismatch)
        print("\n[Step 6] 在新图纸上重新跑 reset 并仿真步进...")
        obs_list = vec_env.reset_all(randomize_duration=True, randomize_workers=True)
        masks_list = vec_env.get_masks_all()
        
        # 仿真随机步进 2 步
        for step in range(2):
            actions = []
            for i in range(2):
                t_mask, s_mask, w_mask = masks_list[i]
                # 寻找一个合法动作
                t_valid = torch.where(~t_mask)[0]
                if len(t_valid) == 0:
                    actions.append(None)
                    continue
                tid = t_valid[0].item()
                
                # 寻找合适站位
                s_valid = torch.where(~s_mask[tid])[0]
                if len(s_valid) == 0:
                    actions.append(None)
                    continue
                sid = s_valid[0].item()
                
                # 寻找合适工人 (此处简化找前 demand 个)
                raw_demand = obs_list[i]['task'].x[tid, -1].item()
                demand = max(1, int(raw_demand))
                
                # 自回归工人掩码重建
                worker_feats = obs_list[i]['worker'].x
                worker_skills = worker_feats[:, 1:11]
                task_type_idx = torch.argmax(obs_list[i]['task'].x[tid, 5:15]).item()
                
                # 找出具备技能的工人
                has_skill = worker_skills[:, task_type_idx] > 0.5
                skill_mask = ~has_skill
                s_act = sid + 1
                worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act)
                
                w_combined_mask = w_mask | skill_mask | lock_mask
                w_valid = torch.where(~w_combined_mask)[0]
                
                if len(w_valid) < demand:
                    actions.append(None)
                else:
                    team = w_valid[:demand].tolist()
                    actions.append((tid, sid, team))
            
            # 并行向前推进
            next_states, step_rewards, step_dones, infos = vec_env.step_all(actions)
            obs_list = next_states
            masks_list = vec_env.get_masks_all()
            print(f"  -> 仿真步进第 {step + 1} 步成功. 奖励: {[float(r) for r in step_rewards]}")
            
        # 在经历了步进之后，再次获取快照并验证延迟重建
        print("\n[Step 7] 仿真步进后再次获取快照并执行就地重建...")
        snapshots_post = vec_env.get_state_snapshot_all()
        for i, env_proxy in enumerate(vec_env.envs):
            snap = snapshots_post[i]
            # 如果发生了动作步进，检查 assigned_tasks 和 station_task_finish_times 是否已正确被快照捕获并成功重建
            rebuilt_data = env_proxy.rebuild_state_from_snapshot(snap)
            print(f"  -> 子环境 {i} 步进后就地重建 HeteroData 成功 (任务数: {rebuilt_data['task'].x.shape[0]})")
            
            # 特征对齐抽样校验
            station_x = rebuilt_data['station'].x
            # 负载不全为 0 时相对占比之和应为 1 (若有分配任务的话)
            if np.sum(snap['station_loads']) > 0:
                assert math.isclose(float(station_x[:, 5].sum()), 1.0, rel_tol=1e-4), "负载相对占比之和不为 1"
            
    finally:
        # 关闭多进程向量化环境，防止后台子进程残留 OOM
        print("\n[Final] 关闭多进程守护子进程...")
        vec_env.close()
        
    print("\n" + "=" * 70)
    print("🎉 恭喜！`tests/test_vector_env_rebuild.py` 所有单元测试通过！")
    print("=" * 70)

if __name__ == "__main__":
    test_vector_env_rebuild_and_switch()
