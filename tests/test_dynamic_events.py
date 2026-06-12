"""
飞机脉动装配线（APAL）四维动态扰动事件测试套件。
依据 Harness-Driven Development (Test-First Protocol) 规范编写。
"""
import sys
import os
import numpy as np
import torch
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.event_engine import Event, EventType, EventQueue
from core.action_masker import ActionMasker
from environment import AirLineEnv_Graph
from configs import configs

DATA_PATH = os.path.join(ROOT_DIR, "data", "283.csv")

def setup_module(module):
    """全局初始化配置，确保基准图存在"""
    assert os.path.exists(DATA_PATH), f"Base dataset 283.csv not found at {DATA_PATH}"


def test_station_breakdown_and_recovery():
    """测试事件①：工位故障与恢复及槽位掩码"""
    # 强制开启配置
    configs.enable_station_breakdown = True
    configs.prob_station_breakdown_base = 1.0  # 100% 触发
    configs.prob_station_breakdown_max = 1.0
    configs.max_slots_per_station = 3
    configs.breakdown_lost_slots_min = 2
    configs.breakdown_lost_slots_max = 2

    env = AirLineEnv_Graph(data_path_or_dir=DATA_PATH, seed=42)
    state = env.reset(randomize_duration=True, randomize_workers=True)

    # 1. 验证初始化状态变量
    assert hasattr(env, "station_available_slots")
    assert len(env.station_available_slots) == env.num_stations
    for sid in range(env.num_stations):
        assert env.station_available_slots[sid] == 3

    # 2. 检查事件队列中是否存在故障事件
    breakdown_events = [item[3] for item in env.event_queue._queue if item[3].type == EventType.STATION_BREAKDOWN]
    assert len(breakdown_events) > 0, "Should inject STATION_BREAKDOWN events"
    
    # 3. 模拟触发一个故障事件
    ev = breakdown_events[0]
    sid = ev.data['station_id']
    lost = ev.data['lost_slots']
    duration = ev.data['duration']
    
    # 手动处理 breakdown 事件
    env.station_available_slots[sid] = max(1, env.station_available_slots[sid] - lost)
    assert env.station_available_slots[sid] == 1, "Slots should reduce from 3 to 1"
    
    # 验证 GNN 观测中的第 [7] 维度反射
    obs = env._get_observation()
    station_x = obs['station'].x
    # station_x[:, 7] 应为 available_slots / max_slots
    assert abs(station_x[sid, 7].item() - (1.0 / 3.0)) < 1e-5, "GNN should perceive the slot ratio reduction"

    # 4. 验证 ActionMasker 在达到缩减槽位上限时的屏蔽行为
    masker = ActionMasker(env)
    
    # 模拟在该站位正在运行 1 个任务
    env.station_task_finish_times[sid].append(env.current_time + 10.0)
    
    # 槽位已被占满 (1 >= 1)，其余任务不应被分配到该工位
    t_mask, s_mask, w_mask = masker.get_masks()
    ready_tasks = torch.where(~t_mask)[0]
    for tid in ready_tasks:
        # 该站位的分配掩码应当被设为 True (被屏蔽)
        assert s_mask[tid, sid].item() == True, f"Station {sid} is full (slots=1), task {tid} must be masked out"

    # 5. 模拟恢复事件
    env.station_available_slots[sid] = min(3, env.station_available_slots[sid] + lost)
    assert env.station_available_slots[sid] == 3, "Slots should recover to 3"
    
    # 再次验证 GNN 观测
    obs_recovered = env._get_observation()
    assert abs(obs_recovered['station'].x[sid, 7].item() - 1.0) < 1e-5, "GNN should perceive slot recovery"


def test_online_duration_perturbation():
    """测试事件②：工时在线随机扰动"""
    configs.enable_online_duration_perturb = True
    configs.online_perturb_prob_per_step = 1.0  # 100% 触发

    env = AirLineEnv_Graph(data_path_or_dir=DATA_PATH, seed=42)
    env.reset()

    # 获取初始的一个待调度任务工时
    tid = 0
    original_dur = env.task_static_feat[tid, 0].item()

    # 手动触发工时在线扰动逻辑
    env._try_inject_online_duration_perturb()
    
    # 检查队列中是否有 DURATION_PERTURB 事件
    perturb_events = [item[3] for item in env.event_queue._queue if item[3].type == EventType.DURATION_PERTURB]
    assert len(perturb_events) > 0, "Should inject DURATION_PERTURB event"
    
    ev = perturb_events[0]
    affected_tasks = ev.data['task_ids']
    factor = ev.data['perturb_factor']
    
    # 模拟处理该扰动
    for t in affected_tasks:
        if env.task_status[t] <= 1:
            env.task_static_feat[t, 0] *= factor
            env.base_task_x[t, 0:1] = env.task_static_feat[t, 0:1] / env.mean_task_time

    # 验证受影响任务的工时发生了期望倍数的改变
    for t in affected_tasks:
        obs = env._get_observation()
        # GNN 的 task_x[:, 0] 应实时更新
        gnn_dur = obs['task'].x[t, 0].item()
        expected_gnn_dur = env.task_static_feat[t, 0].item() / env.mean_task_time
        assert abs(gnn_dur - expected_gnn_dur) < 1e-5, "GNN task_x[:, 0] must dynamically reflect duration change"


def test_material_arrival_delay():
    """测试事件③：物料延迟到达及 ActionMasker 拦截机制"""
    configs.enable_material_delay = True
    configs.prob_material_delay_base = 1.0  # 全员延迟
    configs.prob_material_delay_max = 1.0
    configs.material_delay_min = 10.0
    configs.material_delay_max = 20.0

    env = AirLineEnv_Graph(data_path_or_dir=DATA_PATH, seed=42)
    env.reset(randomize_duration=True, randomize_workers=True)

    # 1. 验证初始化状态变量
    assert hasattr(env, "task_material_ready")
    assert (env.task_material_ready > 0).any(), "Some tasks must have material delay ready time"

    # 2. 检查事件队列
    material_events = [item[3] for item in env.event_queue._queue if item[3].type == EventType.MATERIAL_ARRIVE]
    assert len(material_events) > 0, "Should inject MATERIAL_ARRIVE events"

    # 3. 选取一个有延迟的任务，验证在当前时间 < 就绪时间时被 ActionMasker 拦截
    delayed_tasks = np.where(env.task_material_ready > env.current_time)[0]
    assert len(delayed_tasks) > 0
    tid = delayed_tasks[0]
    ready_time = env.task_material_ready[tid]

    masker = ActionMasker(env)
    
    # 模拟任务已处于就绪状态，但物料还没到
    env.task_status[tid] = 1  # Ready
    t_mask, s_mask, w_mask = masker.get_masks()
    assert t_mask[tid].item() == True, "Task with delayed material must be masked out"

    # 验证 GNN 的等待时间第 [17] 维特征反映
    obs = env._get_observation()
    expected_wait = (ready_time - env.current_time) / env.mean_task_time
    assert abs(obs['task'].x[tid, 17].item() - expected_wait) < 1e-5, "GNN should reflect exact normalized material wait time"

    # 4. 模拟时间推移物料到达，解除屏蔽
    env.current_time = ready_time + 1.0
    t_mask_after, _, _ = masker.get_masks()
    # 此时如果其他约束满足，就不应该再被物料拦截
    obs_after = env._get_observation()
    assert obs_after['task'].x[tid, 17].item() == 0.0, "GNN wait time must drop to 0 after arrival"


def test_worker_fatigue_and_lazy_recovery():
    """测试事件④：工人连续工作疲劳及空闲动态恢复 (Lazy Evaluation)"""
    configs.enable_worker_fatigue = True
    configs.fatigue_threshold_hours = 4.0
    configs.fatigue_decay_slope = 0.05
    configs.fatigue_efficiency_floor = 0.60
    configs.fatigue_recovery_ratio = 0.5

    env = AirLineEnv_Graph(data_path_or_dir=DATA_PATH, seed=42)
    env.reset()

    # 1. 验证初始化状态变量
    assert hasattr(env, "worker_cumulative_work")
    assert hasattr(env, "worker_fatigue_factor")
    assert hasattr(env, "worker_last_busy_end")
    assert (env.worker_cumulative_work == 0.0).all()
    assert (env.worker_fatigue_factor == 1.0).all()

    # 2. 模拟某工人工作 6.0 小时
    w = 0
    start_t = 0.0
    end_t = 6.0
    env._update_worker_fatigue(w, start_t, end_t)
    
    # 累计工作时间应为 6.0
    assert env.worker_cumulative_work[w] == 6.0
    # 疲劳因子计算：overtime = 6 - 4 = 2
    # decay = 0.05 * 2 / 8 = 0.0125
    # fatigue = 1.0 - 0.0125 = 0.9875
    assert abs(env.worker_fatigue_factor[w] - 0.9875) < 1e-5

    # 3. 测试 Lazy Evaluation 动态疲劳恢复
    # 设当前环境时间向后推移 4.0 小时 (工人闲置了 4.0 小时)
    env.current_time = 10.0  # 距离 last_busy_end[w]=6.0 过去了 4 小时
    
    # 动态查询该工人的当前疲劳度
    current_fatigue = env.get_current_fatigue_factor(w, env.current_time)
    # 闲置 4.0 小时，以 0.5 速率恢复疲劳：恢复 4 * 0.5 = 2.0 小时的累积工作时长
    # 累积时长变为 6.0 - 2.0 = 4.0 小时
    # 刚好等于 4.0 小时阈值，疲劳度应完美恢复为 1.0
    assert abs(current_fatigue - 1.0) < 1e-5, f"Fatigue should recover to 1.0 after 4h idle (got {current_fatigue:.4f})"
    
    # 4. 验证 GNN 能感知动态恢复后的值
    obs = env._get_observation()
    assert abs(obs['worker'].x[w, 21].item() - 1.0) < 1e-5, "GNN should read dynamically recovered fatigue factor"


def test_gnn_dimension_alignment():
    """验证新维度在特征数组中的对齐与拼接大小"""
    configs.task_feat_dim = 18
    configs.worker_feat_dim = 22
    configs.station_feat_dim = 15

    env = AirLineEnv_Graph(data_path_or_dir=DATA_PATH, seed=42)
    obs = env.reset()

    # 检查返回的异构图各张量维度
    assert obs['task'].x.shape[1] == 18, f"Task features should have 18 columns, got {obs['task'].x.shape[1]}"
    assert obs['worker'].x.shape[1] == 22, f"Worker features should have 22 columns, got {obs['worker'].x.shape[1]}"
    assert obs['station'].x.shape[1] == 15, f"Station features should have 15 columns, got {obs['station'].x.shape[1]}"
