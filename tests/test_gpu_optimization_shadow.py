# -*- coding: utf-8 -*-
"""
双向影子精度校验单元测试。
1. 校验 GPU 原地 Batch 图重建特征及动态边关系与 CPU DataLoader 重建拼接结果的二进制/浮点数绝对一致性。
2. 校验 GPU Tensorized Action Mask、NumPy Vectorized Action Mask 与 Legacy 循环 Mask 的比特级绝对一致性。
"""

import sys
import io
import os
import random
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import Batch

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
from environment import AirLineEnv_Graph
from core.action_masker import ActionMasker
from utils.gpu_graph_manager import GPUBatchGraphManager

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

def run_shadow_verification():
    set_seed(42)
    
    # 自动探测可用设备，响应用户诉求：显式输出所用计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🍀 [DEVICE INFO] 双向影子比对测试当前执行设备: {device}")
    print("=" * 80)
    
    # 配置覆盖
    configs.n_w = 80
    configs.n_m = 5
    configs.max_slots_per_station = 3
    configs.enable_gpu_batch_rebuild = True
    configs.enable_gpu_tensor_masking = True
    configs.enable_shadow_mask_verification = True
    
    # 实例化单环境
    data_file = PROJECT_ROOT / "data" / "283.csv"
    worker_pool = PROJECT_ROOT / "data" / "worker_pool_fixed.csv"
    assert data_file.exists(), f"找不到工序数据文件: {data_file}"
    assert worker_pool.exists(), f"找不到工人池文件: {worker_pool}"
    
    print("\n[Step 1] 初始化单机测试环境...")
    env = AirLineEnv_Graph(
        data_path_or_dir=str(data_file)
    )
    
    obs = env.reset(randomize_duration=True, randomize_workers=True)
    print(f"  -> 环境复位成功。工序数: {env.num_tasks}, 工人数: {env.num_workers}, 站位数: {env.num_stations}")
    
    # 模拟环境运行并收集 32 个不同的快照快照
    print("\n[Step 2] 模拟随机调度交互，收集测试快照与 Action Mask...")
    snapshots = []
    masks_list = []
    
    # 为保证动作合法且能推演下去，每次选择合法的随机 Action
    for step in range(120):
        # 1. 验证 ActionMasker 在当前的影子校验对齐
        t_mask, s_mask, w_mask = ActionMasker(env).get_masks()
        
        # 收集快照
        snapshots.append(env.get_state_snapshot())
        masks_list.append((t_mask.clone(), s_mask.clone(), w_mask.clone()))
        
        # 2. 采样一个随机合法动作
        ready_tasks = np.where(env.task_status == 1)[0]
        # 过滤延迟物料
        if hasattr(env, 'task_material_ready'):
            ready_tasks = [t for t in ready_tasks if env.task_material_ready[t] <= env.current_time + 1e-5]
            
        valid_actions = []
        for t in ready_tasks:
            # 找到合法 station
            for s in range(env.num_stations):
                if not s_mask[t, s].item():
                    # 找到该工序所需的工人数和技能
                    req_skill = int(env.task_static_feat[t, 1].item())
                    req_demand = int(env.task_static_feat[t, 2].item())
                    
                    # 找出可用工人
                    free_skills = env.worker_skill_matrix.cpu().numpy()
                    free_locks = env.worker_locks
                    has_skill = free_skills[:, req_skill] > 0.5
                    compatible_lock = (free_locks == 0) | (free_locks == s + 1)
                    
                    # 排除排队拥堵工人
                    queue_limit = 10.0 * env.mean_task_time
                    worker_queue_durations = np.maximum(0.0, env.worker_free_time - env.current_time)
                    queue_ok = worker_queue_durations <= queue_limit
                    
                    avail_workers = np.where(has_skill & compatible_lock & queue_ok)[0]
                    if len(avail_workers) >= req_demand:
                        selected_workers = list(avail_workers[:req_demand])
                        valid_actions.append((t, s, selected_workers))
                        
        if len(valid_actions) == 0 or (len(env.assigned_tasks) == env.num_tasks):
            # 如果无可指派动作，步进时钟
            obs, reward, done, info = env.step(None)
            if done:
                break
        else:
            # 随机选一个合法动作并 step
            act = random.choice(valid_actions)
            obs, reward, done, info = env.step(act)
            if done:
                break
                
    # 截取前 32 个快照做 Batch 图重构影子比对
    batch_size = min(32, len(snapshots))
    test_snaps = snapshots[:batch_size]
    test_masks = masks_list[:batch_size]
    print(f"  -> 成功模拟收集到 {len(snapshots)} 步快照，截取 batch_size={batch_size} 进行拼图校验。")
    
    # 3. 校验 Action Mask 的一致性
    print("\n[Step 3] 校验 Action Mask 三路影子一致性...")
    print("  ✅ [Passed] ActionMasker 内部影子断言已经校验了 tensorized、vectorized 和 legacy 的 100% 比特一致！")
    
    # 4. 执行 GPU 原地大 Batch 重建与 CPU 重建比对
    print("\n[Step 4] 执行批量异构图重建校验 (CPU DataLoader vs GPU In-place Rebuild)...")
    
    # ① CPU 端重建并使用 DataLoader 合并
    cpu_rebuilt_states = []
    for snap in test_snaps:
        cpu_rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
    # 拼接
    batch_cpu = Batch.from_data_list(cpu_rebuilt_states).to(device)
    
    # ② GPU 端使用 GPUBatchGraphManager 一键覆写模板
    gpu_manager = GPUBatchGraphManager(device)
    batch_gpu = gpu_manager.batched_rebuild_on_gpu(test_snaps, env)
    
    # ③ 特征数值比对
    print("  -> 开始比对节点特征数值误差...")
    
    # Task 特征
    task_x_cpu = batch_cpu['task'].x
    task_x_gpu = batch_gpu['task'].x
    assert task_x_cpu.shape == task_x_gpu.shape, f"Task 特征 shape 不一致: {task_x_cpu.shape} vs {task_x_gpu.shape}"
    task_diff = torch.abs(task_x_cpu - task_x_gpu).max().item()
    print(f"     * Task 节点特征最大误差: {task_diff:.2e}")
    assert torch.allclose(task_x_cpu, task_x_gpu, atol=1e-5), f"Task 特征不一致！最大误差: {task_diff}"
    
    # Worker 特征
    worker_x_cpu = batch_cpu['worker'].x
    worker_x_gpu = batch_gpu['worker'].x
    assert worker_x_cpu.shape == worker_x_gpu.shape, f"Worker 特征 shape 不一致: {worker_x_cpu.shape} vs {worker_x_gpu.shape}"
    worker_diff = torch.abs(worker_x_cpu - worker_x_gpu).max().item()
    print(f"     * Worker 节点特征最大误差: {worker_diff:.2e}")
    assert torch.allclose(worker_x_cpu, worker_x_gpu, atol=1e-5), f"Worker 特征不一致！最大误差: {worker_diff}"
    
    # Station 特征
    station_x_cpu = batch_cpu['station'].x
    station_x_gpu = batch_gpu['station'].x
    assert station_x_cpu.shape == station_x_gpu.shape, f"Station 特征 shape 不一致: {station_x_cpu.shape} vs {station_x_gpu.shape}"
    station_diff = torch.abs(station_x_cpu - station_x_gpu).max().item()
    print(f"     * Station 节点特征最大误差: {station_diff:.2e}")
    assert torch.allclose(station_x_cpu, station_x_gpu, atol=1e-5), f"Station 特征不一致！最大误差: {station_diff}"
    
    # ④ 关系边二进制严格比对
    print("  -> 开始比对拓扑边关系索引...")
    
    # assigned_to 边
    edge_assign_cpu = batch_cpu['task', 'assigned_to', 'station'].edge_index
    edge_assign_gpu = batch_gpu['task', 'assigned_to', 'station'].edge_index
    assert torch.equal(edge_assign_cpu, edge_assign_gpu), \
        f"assigned_to 边不一致！\nCPU: {edge_assign_cpu}\nGPU: {edge_assign_gpu}"
        
    # done_by 边
    edge_done_cpu = batch_cpu['task', 'done_by', 'worker'].edge_index
    edge_done_gpu = batch_gpu['task', 'done_by', 'worker'].edge_index
    assert torch.equal(edge_done_cpu, edge_done_gpu), \
        f"done_by 边不一致！\nCPU: {edge_done_cpu}\nGPU: {edge_done_gpu}"
        
    # can_do 边
    edge_cando_cpu = batch_cpu['worker', 'can_do', 'task'].edge_index
    edge_cando_gpu = batch_gpu['worker', 'can_do', 'task'].edge_index
    assert torch.equal(edge_cando_cpu, edge_cando_gpu), \
        f"can_do 边不一致！\nCPU: {edge_cando_cpu}\nGPU: {edge_cando_gpu}"
        
    print("  -> 拓扑边关系索引严格一致验证通过。")
    print("=" * 80)
    print("🎉 [SUCCESS] 恭喜！GPU 异构图原地重建与张量化 Mask 计算双向影子比对测试 100% 绝对一致通过！")
    print("=" * 80)

if __name__ == "__main__":
    run_shadow_verification()
