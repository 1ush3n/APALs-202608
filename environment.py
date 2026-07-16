import gymnasium as gym
import numpy as np
import torch
import math
from torch_geometric.data import HeteroData
from gymnasium import spaces
import os
import pandas as pd
import heapq
import hashlib
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Any, Iterable
from pathlib import Path

from data_loader import load_data
from configs import configs
from core.event_engine import Event, EventType, EventQueue
from core.action_masker import ActionMasker
from core.constraints import Assignment, ConstraintEngine, ScheduleValidationReport
from core.time_comparison import release_time_tolerance, time_reached_scalar
from utils.resource_graph import (
    SkillHubTopology,
    apply_resource_graph,
    build_skill_features,
    build_skill_hub_topology,
    build_task_skill_edges,
    build_worker_skill_edges,
    worker_topology_key,
)
from utils.reschedule import (
    BaselineSchedule,
    RescheduleScenario,
    calculate_reschedule_objective_terms,
    calculate_reschedule_lower_bound,
    calculate_stability_metrics,
    load_baseline_schedule,
    load_reschedule_scenario,
    sample_task_delay_load_scenario,
    sample_task_delay_scenario,
)
from runtime.reschedule_manifest import resolve_manifest_entry_for_data

Action = Tuple[int, int, List[int]]


def _fill_station_macro_features(station_x: torch.Tensor, worker_locks: np.ndarray, is_free_bool: np.ndarray) -> None:
    """填充站位宏观策略特征：维度 1=站位绑定工人比, 2=全局自由工人比, 3=站内可用工人比。
    该函数被 _get_observation / rebuild_state_from_snapshot / vector_env 共享调用。"""
    num_workers = len(worker_locks)
    num_stations = station_x.shape[0]
    global_mobile_count = np.sum(worker_locks == 0)
    station_x[:, 2] = float(global_mobile_count) / num_workers
    for s in range(num_stations):
        bound_count = np.sum(worker_locks == s + 1)
        station_x[s, 1] = float(bound_count) / num_workers
        free_and_bound = np.sum((worker_locks == s + 1) & is_free_bool)
        station_x[s, 3] = float(free_and_bound) / num_workers


# ---------------------------------------------------------------------------
# 航空装配线环境 (AirLineEnv_Graph)
# ---------------------------------------------------------------------------
class AirLineEnv_Graph(gym.Env):
    """
    基于图的航空装配线强化学习环境。
    
    核心特性:
    1. 异构图状态: 包含 Task, Worker, Station 三种节点及其相互关系。
    2. 离散事件仿真: 时间推进基于事件(Event-Driven)，而非固定步长。
    3. 复杂约束: 包含工艺优先关系、技能匹配、站位空间约束。
    """
    
    # Gymnasium Metadata
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, data_path_or_dir="工序约束_50.xlsx", seed=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 设置私有随机种子以保证多环境并行复现性并防止血崩
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
            
        # P0 漏洞修复: 初始化最大时间锚点
        self.max_time = 1e6
        
        self.num_workers = configs.n_w
        self.num_stations = configs.n_m
        
        # 目录中只保留轻量文件描述符，完整图上下文按需加载。
        self.dataset_pool: list[dict[str, Any]] = []
        self._context_lru: OrderedDict[int, None] = OrderedDict()
        self._context_cache_size = max(1, int(getattr(configs, "dataset_context_cache_size", 2)))

        data_source = Path(data_path_or_dir)
        if data_source.is_dir():
            files = sorted(
                path for path in data_source.iterdir()
                if path.suffix.lower() in {".csv", ".xlsx"}
            )
            if not files:
                raise ValueError(f"目录 {data_source} 中未找到任何 csv 或 xlsx 文件。")
            for file_path in files:
                self._register_dataset(file_path)
        else:
            self._register_dataset(data_source)
            
        # 初始化激活索引
        self.active_dataset_idx = 0
        self._worker_skill_topology_cache: dict[tuple[int, bytes], torch.Tensor] = {}
        self._active_worker_topology_key: tuple[int, bytes] | None = None
        self.switch_dataset(0)
        
        # 动作空间: Tuple(Task, Station, Worker_List_Leader, Num_Workers)
        self.action_space = spaces.MultiDiscrete([self.num_tasks, self.num_stations, self.num_workers])
        
        # 事件队列 (Priority Queue Engine)
        self.event_queue = EventQueue(max_size=10000)
        self.baseline_schedule: BaselineSchedule | None = None
        self.reschedule_scenario: RescheduleScenario | None = None
        self.reschedule_start_time = 0.0
        self.reschedule_lower_bound = 0.0
        self.reschedule_takt_feasible = True
        
    def _register_dataset(self, file_path: str | Path) -> None:
        """注册数据集路径，不在环境初始化时加载 DataFrame 和图张量。"""
        self.dataset_pool.append({"file_path": str(Path(file_path).resolve())})

    def _load_and_build_context(self, file_path: str | Path) -> None:
        """兼容旧调用：解析轻量元数据，完整图仍在首次切换时构建。"""
        path = Path(file_path).resolve()
        raw_data = load_data(path)
        self.dataset_pool.append(
            {
                "file_path": str(path),
                "raw_data": raw_data,
                "num_tasks": int(raw_data["num_tasks"]),
            }
        )

    def _touch_context(self, idx: int) -> None:
        self._context_lru.pop(idx, None)
        self._context_lru[idx] = None
        while len(self._context_lru) > self._context_cache_size:
            evict_idx, _ = self._context_lru.popitem(last=False)
            if evict_idx == self.active_dataset_idx:
                self._context_lru[evict_idx] = None
                continue
            ctx = self.dataset_pool[evict_idx]
            file_path = ctx["file_path"]
            ctx.clear()
            ctx["file_path"] = file_path

    def _ensure_dataset_context(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < len(self.dataset_pool):
            raise IndexError(f"数据集索引越界: {idx}/{len(self.dataset_pool)}")
        ctx = self.dataset_pool[idx]
        if "raw_data" not in ctx:
            raw_data = load_data(Path(ctx["file_path"]))
            ctx["raw_data"] = raw_data
            ctx["num_tasks"] = int(raw_data["num_tasks"])

        self.raw_data = ctx["raw_data"]
        self.num_tasks = int(ctx["num_tasks"])
        if "base_data" not in ctx:
            self._build_static_context(ctx)
        self._touch_context(idx)
        return ctx
        
    def switch_dataset(self, idx: int):
        """无缝切换激活的图骨架"""
        self.active_dataset_idx = idx
        ctx = self._ensure_dataset_context(idx)
            
        # 3. 将激活上下文的属性挂载到当前环境实例，保持向下兼容
        self.base_data = ctx['base_data']
        self.base_task_x = ctx['base_task_x']
        self.base_worker_x = ctx['base_worker_x']
        self.base_station_x = ctx['base_station_x']
        self.task_static_feat = ctx['task_static_feat']
        self.worker_skill_matrix = ctx['worker_skill_matrix']
        self.predecessors = ctx['predecessors']
        self.successors = ctx['successors']
        self.num_preds = ctx['num_preds']
        self.fixed_stations = ctx['fixed_stations']
        self.mean_task_time = ctx['mean_task_time']
        self.ideal_station_load = ctx['ideal_station_load']
        self.ideal_makespan = ctx['ideal_makespan']
        self.total_base_workload = ctx['total_base_workload']
        self.base_durations = ctx['base_durations']
        self.max_allowed_stations = ctx['max_allowed_stations']
        self.constraint_engine = ctx['constraint_engine']
        self.is_critical = ctx['is_critical']
        self.baseline_schedule = None
        self.reschedule_scenario = None
        
        # 状态变量初始化
        self.current_time = 0.0
        self.task_status = np.zeros(self.num_tasks, dtype=int) 
        self.worker_free_time = np.zeros(self.num_workers, dtype=float) 
        self.worker_locks = np.zeros(self.num_workers, dtype=int)
        self.station_loads = np.zeros(self.num_stations, dtype=float)
        self.station_wall_clock = np.zeros(self.num_stations, dtype=float)

    @property
    def dataset_count(self) -> int:
        return len(self.dataset_pool)

    def get_dataset_descriptor(self, idx: int) -> dict[str, Any]:
        """返回可安全跨进程传输的轻量数据集信息。"""
        ctx = self.dataset_pool[idx]
        return {
            "dataset_idx": int(idx),
            "file_path": str(ctx["file_path"]),
            "num_tasks": None if "num_tasks" not in ctx else int(ctx["num_tasks"]),
        }

    def export_dataset_context(self, idx: int) -> dict[str, Any]:
        """按需导出主进程重建 observation 所需的静态上下文。"""
        ctx = self._ensure_dataset_context(idx)
        required = (
            "file_path",
            "num_tasks",
            "base_data",
            "base_task_x",
            "base_station_x",
            "task_skill_edge_index",
            "full_can_do_edge_index",
            "mean_task_time",
            "ideal_station_load",
        )
        return {key: ctx[key] for key in required}

    def _skill_hub_topology(
        self,
        task_skill_edge_index: torch.Tensor,
        worker_x: torch.Tensor,
        topology_key: tuple[int, bytes] | None,
    ) -> SkillHubTopology | None:
        """取得 Skill Hub 静态拓扑；legacy direct 模式不使用该缓存。"""
        if not bool(getattr(configs, "use_skill_hub", False)):
            return None
        key = topology_key or worker_topology_key(
            worker_x,
            int(configs.num_skill_types),
        )
        worker_edges = self._worker_skill_topology_cache.get(key)
        if worker_edges is None:
            worker_edges = build_worker_skill_edges(
                worker_x,
                int(configs.num_skill_types),
            )
            self._worker_skill_topology_cache[key] = worker_edges
        return SkillHubTopology(
            worker_to_skill=worker_edges,
            skill_to_task=task_skill_edge_index,
        )

    def _current_skill_hub_topology(self) -> SkillHubTopology | None:
        ctx = self.dataset_pool[self.active_dataset_idx]
        return self._skill_hub_topology(
            ctx["task_skill_edge_index"],
            self.base_worker_x,
            self._active_worker_topology_key,
        )
        
    def _build_static_context(self, ctx):
        """巧妙利用原有的 init_hetero_data 逻辑，并将产生的属性打包进 ctx"""
        # 解析固定站位约束 (Fixed Station Constraint)
        self.fixed_stations = -np.ones(self.num_tasks, dtype=int)
        if 'fixed_station' in self.raw_data['task_df'].columns:
            for idx, val in enumerate(self.raw_data['task_df']['fixed_station']):
                if pd.isna(val): continue
                s_idx = -1
                try:
                    val_str = str(val).lower().strip()
                    if val_str.startswith('station'):
                         s_idx = int(float(val_str.split()[-1])) - 1
                    elif val_str.startswith('s'):
                         s_idx = int(float(val_str[1:])) - 1
                    else:
                         s_idx = int(float(val_str)) - 1 
                except:
                    pass
                if 0 <= s_idx < self.num_stations:
                    self.fixed_stations[idx] = s_idx

        edge_array = self.raw_data['precedence_edges'].detach().cpu().numpy()
        self.constraint_engine = ConstraintEngine.build(
            num_tasks=self.num_tasks,
            num_stations=self.num_stations,
            edges=edge_array,
            durations=self.raw_data['task_df']['duration'].to_numpy(dtype=float),
            fixed_stations=self.fixed_stations,
        )
        self.fixed_stations = self.constraint_engine.fixed_stations.copy()
                    
        # 复用原有的庞大初始化逻辑
        self.init_hetero_data()
        self.constraint_engine = self.constraint_engine.with_max_allowed_stations(
            self.max_allowed_stations
        )
        
        # 将生成的张量打包进 ctx
        ctx['base_data'] = self.base_data
        ctx['base_task_x'] = self.base_task_x
        ctx['base_worker_x'] = self.base_worker_x
        ctx['base_station_x'] = self.base_station_x
        ctx['task_static_feat'] = self.task_static_feat
        ctx['worker_skill_matrix'] = self.worker_skill_matrix
        ctx['predecessors'] = self.predecessors
        ctx['successors'] = self.successors
        ctx['num_preds'] = self.num_preds
        ctx['fixed_stations'] = self.fixed_stations
        ctx['mean_task_time'] = self.mean_task_time
        ctx['ideal_station_load'] = self.ideal_station_load
        ctx['ideal_makespan'] = self.ideal_makespan
        ctx['total_base_workload'] = self.total_base_workload
        ctx['base_durations'] = self.base_durations
        ctx['max_allowed_stations'] = self.max_allowed_stations
        ctx['constraint_engine'] = self.constraint_engine
        ctx['is_critical'] = self.is_critical
        ctx['task_skill_edge_index'] = build_task_skill_edges(
            self.base_task_x,
            int(configs.num_skill_types),
        )
        
        # 预计算全量 can_do 边索引（基于 full_worker_skill_matrix，不依赖 reset 采样）
        if getattr(configs, "use_skill_hub", False):
            ctx['full_can_do_edge_index'] = torch.empty((2, 0), dtype=torch.long)
        else:
            n_w_full = self.full_worker_skill_matrix.shape[0]
            n_t_full = self.num_tasks
            w_all = torch.arange(n_w_full).repeat_interleave(n_t_full)
            t_all = torch.arange(n_t_full).repeat(n_w_full)
            skills_col = self.task_static_feat[:, 1].squeeze().long()
            task_skills = skills_col[t_all]
            valid_skill = (task_skills >= 0) & (task_skills < int(configs.num_skill_types))
            has_skill = torch.zeros_like(valid_skill)
            has_skill[valid_skill] = (
                self.full_worker_skill_matrix[w_all[valid_skill], task_skills[valid_skill]] == 1.0
            )
            ctx['full_can_do_edge_index'] = torch.stack([w_all[has_skill], t_all[has_skill]])

    def _resolve_project_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        return path if path.is_absolute() else Path(__file__).resolve().parent / path

    def _ensure_baseline_schedule(self) -> BaselineSchedule:
        if self.baseline_schedule is not None:
            return self.baseline_schedule
        manifest_entry = resolve_manifest_entry_for_data(
            configs,
            self.dataset_pool[self.active_dataset_idx]["file_path"],
        )
        if manifest_entry is not None:
            baseline_path = manifest_entry.baseline_schedule_path
        else:
            baseline_path = self._resolve_project_path(getattr(configs, "reschedule_baseline_schedule_path", "results/final_schedule.csv"))
        baseline = load_baseline_schedule(baseline_path)
        missing = set(range(self.num_tasks)) - set(baseline.tasks)
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(f"baseline 调度缺少当前数据集任务: {sample}")
        self.baseline_schedule = baseline
        return baseline

    def _build_reschedule_scenario(self, baseline: BaselineSchedule) -> RescheduleScenario:
        forced = getattr(self, "_forced_reschedule_scenario", None)
        if forced is not None:
            return forced
        scenario_path = str(getattr(configs, "reschedule_scenario_path", "") or "").strip()
        if scenario_path:
            return load_reschedule_scenario(self._resolve_project_path(scenario_path))
        if str(getattr(configs, "reschedule_train_scenario_mode", "config")) == "uniform_load":
            _level, scenario = sample_task_delay_load_scenario(
                baseline,
                rng=self.np_random,
            )
            return scenario
        return sample_task_delay_scenario(
            baseline,
            rng=self.np_random,
            min_start_ratio=float(getattr(configs, "reschedule_start_time_min_ratio", 0.15)),
            max_start_ratio=float(getattr(configs, "reschedule_start_time_max_ratio", 0.65)),
            task_prob=float(getattr(configs, "reschedule_delay_task_prob", 0.08)),
            delay_min=float(getattr(configs, "reschedule_delay_min", 5.0)),
            delay_max=float(getattr(configs, "reschedule_delay_max", 30.0)),
        )

    def _apply_reschedule_scenario(self) -> None:
        baseline = self._ensure_baseline_schedule()
        baseline_report = self.validate_assignments(
            (
                task.task_id,
                task.station_id,
                task.team,
                task.start,
                task.end,
            )
            for task in baseline.tasks.values()
        )
        if not baseline_report.is_legal:
            nonzero = {
                key: value
                for key, value in baseline_report.violations.items()
                if int(value) > 0
            }
            raise ValueError(f"重调度基线排程违反硬约束: {nonzero}")
        scenario = self._build_reschedule_scenario(baseline)
        self.reschedule_scenario = scenario
        self.reschedule_start_time = float(scenario.start_time)
        self.current_time = self.reschedule_start_time

        self.assigned_tasks = []
        self.task_station_map = {}
        self.task_end_times = -np.ones(self.num_tasks)
        self.task_status.fill(0)
        self.completed_preds = np.zeros(self.num_tasks, dtype=int)
        self.worker_free_time = np.zeros(self.num_workers, dtype=float)
        self.worker_locks = np.zeros(self.num_workers, dtype=int)
        self.station_loads.fill(0.0)
        self.station_wall_clock.fill(0.0)
        self.station_task_finish_times = [[] for _ in range(self.num_stations)]
        self.event_queue = EventQueue(max_size=10000)

        for task_id, release_time in scenario.task_release_times.items():
            if 0 <= task_id < self.num_tasks:
                self.task_material_ready[task_id] = max(float(release_time), self.reschedule_start_time)
                self.event_queue.push(Event(float(self.task_material_ready[task_id]), EventType.MATERIAL_ARRIVE, {'task_id': int(task_id)}))

        for task_id, base_task in baseline.tasks.items():
            if base_task.start > self.reschedule_start_time + 1e-9:
                continue
            if task_id < 0 or task_id >= self.num_tasks:
                continue
            team = [int(w) for w in base_task.team if int(w) < self.num_workers]
            sid = int(base_task.station_id)
            start = float(base_task.start)
            end = float(base_task.end)
            duration = max(0.0, end - start)
            self.task_status[task_id] = 2
            self.task_end_times[task_id] = end
            self.task_station_map[task_id] = sid
            self.assigned_tasks.append((task_id, sid, team, start, end))
            if sid >= 0:
                self.station_loads[sid] += duration * max(1, len(team))
                self.station_wall_clock[sid] = max(self.station_wall_clock[sid], end)
                if end > self.current_time + 1e-9:
                    heapq.heappush(self.station_task_finish_times[sid], end)
            for w in team:
                self.worker_free_time[w] = max(self.worker_free_time[w], end)
                if sid >= 0 and self.worker_locks[w] == 0:
                    self.worker_locks[w] = sid + 1
            if end > self.current_time + 1e-9:
                self.event_queue.push(Event(end, EventType.TASK_FINISH, {'task_id': task_id, 'worker_ids': team, 'station_id': sid}))
            else:
                for succ in self.successors[task_id]:
                    self.completed_preds[succ] += 1

        for task_id in range(self.num_tasks):
            if self.task_status[task_id] == 2:
                continue
            if self.completed_preds[task_id] == self.num_preds[task_id]:
                self.task_status[task_id] = 1

        durations_np = self.task_static_feat[:, 0].detach().cpu().numpy()
        self.reschedule_lower_bound = calculate_reschedule_lower_bound(
            baseline,
            self.task_status,
            durations_np,
            self.task_material_ready,
            current_time=self.current_time,
            num_stations=self.num_stations,
            station_slots=float(getattr(configs, "estimated_cmax_station_slots", 3.0)),
        )
        tolerance = float(getattr(configs, "reschedule_takt_tolerance", 1e-5))
        self.reschedule_takt_feasible = self.reschedule_lower_bound <= baseline.makespan + tolerance
        
    def init_hetero_data(self):
        """
        初始化异构图的静态特征 (Task, Worker)。
        包含由 'seed' 控制的随机初始化逻辑。
        """
        data = HeteroData()
        
        # ------------------
        # 1. 任务节点 (Task Nodes)
        # ------------------
        task_df = self.raw_data['task_df']
        # 特征: [Duration, SkillType, DemandWorkers]
        dur = torch.tensor(task_df['duration'].values, dtype=torch.float).unsqueeze(1)
        skill = torch.tensor(task_df['skill_type'].values, dtype=torch.float).unsqueeze(1)
        demand = torch.tensor(task_df['demand_workers'].values, dtype=torch.float).unsqueeze(1)
        # 物理工序至少需要 1 人；虚拟层级节点不占用工人。
        demand = torch.where(skill.ge(0), torch.clamp(demand, min=1.0), torch.zeros_like(demand))
        
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # ------------------
        # 2. 工人节点 (Worker Nodes)
        # ------------------
        # 使用 pathlib 规范化跨平台绝对路径，避免由于工作目录偏离导致子进程崩溃
        from pathlib import Path
        proj_root = Path(__file__).resolve().parent
        pool_path = proj_root / configs.worker_pool_path
        if not pool_path.exists():
            pool_path = Path(configs.worker_pool_path)
            
        full_worker_df = pd.read_csv(pool_path)
        self.n_w_max = len(full_worker_df)
        self.full_worker_efficiency = full_worker_df['efficiency'].values
        num_skill_types = int(configs.num_skill_types)
        skill_slots = int(getattr(configs, "worker_skill_feature_slots", num_skill_types))
        assert 0 < num_skill_types <= skill_slots, "有效工种数必须大于 0 且不超过工人技能槽位数"
        active_skill_columns = [f'skill_{i}' for i in range(num_skill_types)]
        missing_skill_columns = sorted(set(active_skill_columns) - set(full_worker_df.columns))
        if missing_skill_columns:
            raise ValueError(f"工人池缺少有效技能列：{missing_skill_columns}")
        active_skill_matrix = torch.tensor(
            full_worker_df[active_skill_columns].values,
            dtype=torch.float,
        )
        # 输入形状：[W, 5] -> [W, 10]；后 5 个槽位仅为特征布局兼容而补零。
        skill_padding = torch.zeros((self.n_w_max, skill_slots - num_skill_types))
        self.full_worker_skill_matrix = torch.cat([active_skill_matrix, skill_padding], dim=1)
        expected_worker_dim = 1 + skill_slots + 11
        assert int(configs.worker_feat_dim) == expected_worker_dim, (
            f"worker_feat_dim={configs.worker_feat_dim}，应为 {expected_worker_dim}"
        )
        
        # Default workers (used in eval)
        self.num_workers = configs.n_w
        
        # Initialize default sizes just to keep base shapes valid before first reset
        self.worker_efficiency = self.full_worker_efficiency[:self.num_workers]
        self.worker_skill_matrix = self.full_worker_skill_matrix[:self.num_workers]
        # 计算每种技能的最大需求人数，确保每种技能至少有这么多工人拥有，
        # 防止出现 "任务需要5人，但全场只有3个合格工人" 的死锁情况。
        self.worker_static_feat = torch.tensor(self.worker_efficiency, dtype=torch.float).unsqueeze(1)
        
        # [再次鲁棒性检查] Check and Clamp Demand
        # 双重保险：如果初始化后发现某技能工人总数仍少于某任务需求，强制降低该任务需求。
        skill_capacity = self.worker_skill_matrix[:, :num_skill_types].sum(dim=0)  # [5]
        
        clamped_count = 0
        for t in range(self.num_tasks):
            t_skill = int(skill[t].item())
            t_demand = int(demand[t].item())
            if t_skill < 0:
                continue
            if t_skill >= num_skill_types:
                raise ValueError(f"工序 {t} 的工种 {t_skill} 超出 [0, {num_skill_types - 1}]")
            
            cap = int(skill_capacity[t_skill].item())
            if cap == 0:
                # 理论上不应发生，除非逻辑错误。兜底处理。
                print(f"CRITICAL: Skill {t_skill} has 0 workers! Force assigning Worker 0.")
                self.worker_skill_matrix[0, t_skill] = 1.0
                skill_capacity[t_skill] += 1
                cap = 1
                
            if t_demand > cap:
                demand[t] = cap
                clamped_count += 1
                
        if clamped_count > 0:
            print(f"[Robustness] Auto-clamped demand for {clamped_count} tasks to match worker availability.")
            
        # 更新被 Clamp 后的特征
        self.task_static_feat = torch.cat([dur, skill, demand], dim=1)
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        
        # 计算全局的关键路径和最晚允许站位 (持久化静态特征，只计算一次)
        self.is_critical, cpm_makespan = self._calculate_cpm()
        self.max_allowed_stations = self._calculate_max_allowed_stations()
        
        # =====================================================================
        # [Scale Invariance] 尺度锚点定义 (用于抹平跨数据集的绝对数值差异)
        # =====================================================================
        # 1. 基础时间尺度 (T_base)
        valid_durs = dur[dur > 0]
        self.mean_task_time = torch.mean(valid_durs).item() if len(valid_durs) > 0 else 1.0
        
        # 2. 基础负荷尺度 (L_base)
        self.total_base_workload = torch.sum(dur * demand).item()
        self.ideal_station_load = self.total_base_workload / max(1, self.num_stations)
        
        # 3. 理想总完工时间 (M_ideal)
        # 结合平均法和 CPM 法：max(C_CPM, Sum(Duration_i) / Num_Stations)
        sum_durations = torch.sum(dur).item()
        avg_makespan = sum_durations / max(1.0, float(self.num_stations))
        self.ideal_makespan = max(float(cpm_makespan), avg_makespan)
        
        self.base_data = data
        self.obs_data = None # 将在 reset 中 clone
        
        # 预先分配静态底座张量；重调度模式会追加 baseline 参照特征。
        task_feat_dim = max(18, int(getattr(configs, "task_feat_dim", 18)))
        self.base_task_x = torch.zeros((self.num_tasks, task_feat_dim))
        # [Domain Randomization] 备份只读的基础工时分布，用于后续加噪
        # [Scale Invariance] 使用内生均值时间 T_base 进行归一化，而非硬编码 100.0
        self.base_durations = dur.clone() / self.mean_task_time  
        self.base_task_x[:, 0:1] = self.base_durations
        
        type_onehot = torch.zeros((self.num_tasks, num_skill_types))
        type_indices = skill.long().squeeze(1)
        valid_skill = (type_indices >= 0) & (type_indices < num_skill_types)
        type_onehot[valid_skill, type_indices[valid_skill]] = 1.0
        # 输入形状：[T, 5] -> task_x[:, 5:10]；虚拟节点保持全零技能编码。
        self.base_task_x[:, 5 : 5 + num_skill_types] = type_onehot
        self.base_task_x[:, 16:17] = demand
        
        # [Feature Upgrade] worker base feat + wait_time slot (11 dims padding now, total 22 dims)
        self.base_worker_x = torch.cat([self.worker_static_feat, self.worker_skill_matrix, torch.zeros((self.num_workers, 11))], dim=1)
        self._active_worker_topology_key = worker_topology_key(
            self.base_worker_x,
            int(configs.num_skill_types),
        )
        self._worker_skill_topology_cache[self._active_worker_topology_key] = (
            build_worker_skill_edges(self.base_worker_x, int(configs.num_skill_types))
        )
        # [Feature Upgrade] station base feat + slot_wait_time + relative loads (15 dims)
        self.base_station_x = torch.zeros((self.num_stations, 15))

        # 显式构建静态图骨架，后续 observation 只刷新动态特征和动态边。
        data['task'].x = self.base_task_x.clone()
        data['worker'].x = self.base_worker_x.clone()
        data['station'].x = self.base_station_x.clone()
        data['task', 'precedes', 'task'].edge_index = self.raw_data['precedence_edges'].clone().long()
        initial_topology = build_skill_hub_topology(
            self.base_task_x,
            self.base_worker_x,
            int(configs.num_skill_types),
        ) if bool(getattr(configs, "use_skill_hub", False)) else None
        apply_resource_graph(
            data,
            self.base_task_x,
            self.base_worker_x,
            configs,
            skill_hub_topology=initial_topology,
        )
        self.base_data = data
        
    def reset(self, randomize_duration: bool = False, randomize_workers: bool = False, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[HeteroData, Dict[str, Any]]:
        """
        重置环境状态以开始新的 Episode。
        如果在训练阶段开启 randomize_duration，则按 ±range 对静态工时进行伪装修改。
        如果在训练阶段开启 randomize_workers，则动态随机抽取固定工人池的一个子集（领域随机化）。
        """
        # Gymnasium seed forward compatibility
        super().reset(seed=seed, options=options)
        # 新版 gymnasium 会将 np_random 覆盖为 np.random.Generator，
        # 但环境内部大量使用 RandomState API（如 .rand()），因此显式回退
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
             
        # ====================
        # [Domain Randomization] Worker Pool Sampling
        # ====================
        if randomize_workers:
            min_w = configs.n_w_min
            max_w = configs.n_w
            self.num_workers = self.np_random.randint(min_w, max_w + 1)
            
            # [P1] 按工序技能需求强制执行工人覆盖保障
            # 原则: demand 绝不能改，必须通过采样保证人数 ≥ max_demand
            
            # Step 1: 统计每种技能在所有工序中的最大需求人数
            skill_max_demand = {}
            for t in range(self.num_tasks):
                t_skill = int(self.task_static_feat[t, 1].item())
                t_demand = int(self.task_static_feat[t, 2].item())
                if t_skill < 0:
                    continue
                if t_skill >= int(configs.num_skill_types):
                    raise ValueError(
                        f"工序 {t} 的工种 {t_skill} 超出 [0, {int(configs.num_skill_types) - 1}]"
                    )
                skill_max_demand[t_skill] = max(skill_max_demand.get(t_skill, 0), t_demand)
            
            # Step 2: 对每种技能，强制抽取至少 max_demand 个该技能的工人
            selected = set()
            for skill_id, need_count in skill_max_demand.items():
                capable = [w for w in range(self.n_w_max) 
                          if self.full_worker_skill_matrix[w, skill_id] == 1 and w not in selected]
                take = min(need_count, len(capable))
                if take > 0 and len(capable) > 0:
                    selected.update(self.np_random.choice(capable, take, replace=False))
            
            # Step 3: 随机填充剩余名额至 num_workers
            if len(selected) < self.num_workers:
                remaining = [w for w in range(self.n_w_max) if w not in selected]
                num_to_add = self.num_workers - len(selected)
                if len(remaining) >= num_to_add:
                    selected.update(self.np_random.choice(remaining, num_to_add, replace=False))
                elif len(remaining) > 0:
                    selected.update(remaining)
            else:
                # 强制采样后人数已超过随机目标，以实际人数为准
                self.num_workers = len(selected)
                
            w_indices = np.array(list(selected))
            self.np_random.shuffle(w_indices)
        else:
            self.num_workers = configs.n_w
            w_indices = np.arange(self.num_workers)
            
        self.worker_efficiency = self.full_worker_efficiency[w_indices]
        self.worker_skill_matrix = self.full_worker_skill_matrix[w_indices]
        self.worker_static_feat = torch.tensor(self.worker_efficiency, dtype=torch.float).unsqueeze(1)
        
        # 重建动态的 base_worker_x (维度随 num_workers 变化, 增加至 22 维以容纳疲劳因子)
        # 1(efficiency) + 10(skills) + 11(Padding: 1 wait, 1 free, 8 locks, 1 fatigue) = 22 dims
        self.base_worker_x = torch.cat([self.worker_static_feat, self.worker_skill_matrix, torch.zeros((self.num_workers, 11))], dim=1)
        self._active_worker_topology_key = worker_topology_key(
            self.base_worker_x,
            int(configs.num_skill_types),
        )
        if self._active_worker_topology_key not in self._worker_skill_topology_cache:
            self._worker_skill_topology_cache[self._active_worker_topology_key] = (
                build_worker_skill_edges(
                    self.base_worker_x,
                    int(configs.num_skill_types),
                )
            )
        
        # 重置运行状态张量
        self.current_time = 0.0
        self.task_status.fill(0) 
        self.worker_free_time = np.zeros(self.num_workers, dtype=float)
        self.worker_locks = np.zeros(self.num_workers, dtype=int)
        self.station_loads.fill(0.0)
        self.station_wall_clock.fill(0.0)
        self.event_queue = EventQueue(max_size=10000)
        
        # [Dynamic Events] 新增扰动状态变量初始化
        self.station_available_slots = np.full(self.num_stations, configs.max_slots_per_station, dtype=int)
        self.task_material_ready = np.zeros(self.num_tasks, dtype=float)
        self.worker_cumulative_work = np.zeros(self.num_workers, dtype=float)
        self.worker_fatigue_factor = np.ones(self.num_workers, dtype=float)
        self.worker_last_busy_end = np.zeros(self.num_workers, dtype=float)
        
        # [Dynamic Events] 事件注入域
        if getattr(configs, 'enable_dynamic_events', False):
            typical_horizon = 200.0 # 假设典型工程在 200 小时内
            
            # ① 工人缺勤事件注入
            prob_base_absent = getattr(configs, 'prob_worker_absent_base', 0.0)
            if randomize_duration or randomize_workers:
                prob_max_absent = getattr(configs, 'prob_worker_absent_max', 0.15)
                p_absent = self.np_random.uniform(prob_base_absent, max(prob_base_absent, prob_max_absent))
            else:
                p_absent = prob_base_absent
                
            if p_absent > 0:
                absent_workers = np.where(self.np_random.rand(self.num_workers) < p_absent)[0]
                dur_min = getattr(configs, 'absence_duration_min', 10.0)
                dur_max = getattr(configs, 'absence_duration_max', 50.0)
                for w in absent_workers:
                    leave_time = self.np_random.uniform(0.0, typical_horizon)
                    duration = self.np_random.uniform(dur_min, dur_max)
                    self.event_queue.push(Event(leave_time, EventType.WORKER_LEAVE, {'worker_id': int(w), 'duration': float(duration)}))
            
            # ② 工位故障与恢复事件注入
            if getattr(configs, 'enable_station_breakdown', False):
                prob_base_breakdown = getattr(configs, 'prob_station_breakdown_base', 0.0)
                if randomize_duration or randomize_workers:
                    prob_max_breakdown = getattr(configs, 'prob_station_breakdown_max', 0.10)
                    p_breakdown = self.np_random.uniform(prob_base_breakdown, max(prob_base_breakdown, prob_max_breakdown))
                else:
                    p_breakdown = prob_base_breakdown
                    
                if p_breakdown > 0:
                    breakdown_stations = np.where(self.np_random.rand(self.num_stations) < p_breakdown)[0]
                    dur_min_bd = getattr(configs, 'breakdown_duration_min', 5.0)
                    dur_max_bd = getattr(configs, 'breakdown_duration_max', 30.0)
                    lost_min = getattr(configs, 'breakdown_lost_slots_min', 1)
                    lost_max = getattr(configs, 'breakdown_lost_slots_max', 3)
                    for s in breakdown_stations:
                        breakdown_time = self.np_random.uniform(0.0, typical_horizon)
                        duration = self.np_random.uniform(dur_min_bd, dur_max_bd)
                        lost_slots = self.np_random.randint(lost_min, lost_max + 1)
                        # 保证故障后至少剩 1 个 slot
                        lost_slots = min(lost_slots, configs.max_slots_per_station - 1)
                        if lost_slots > 0:
                            self.event_queue.push(Event(breakdown_time, EventType.STATION_BREAKDOWN, 
                                                        {'station_id': int(s), 'lost_slots': int(lost_slots), 'duration': float(duration)}))
                            self.event_queue.push(Event(breakdown_time + duration, EventType.STATION_RECOVER, 
                                                        {'station_id': int(s), 'lost_slots': int(lost_slots)}))

            # ③ 物料延迟到达注入
            if getattr(configs, 'enable_material_delay', False):
                prob_base_delay = getattr(configs, 'prob_material_delay_base', 0.0)
                if randomize_duration or randomize_workers:
                    prob_max_delay = getattr(configs, 'prob_material_delay_max', 0.10)
                    p_delay = self.np_random.uniform(prob_base_delay, max(prob_base_delay, prob_max_delay))
                else:
                    p_delay = prob_base_delay
                    
                if p_delay > 0:
                    delayed_tasks = np.where(self.np_random.rand(self.num_tasks) < p_delay)[0]
                    delay_min = getattr(configs, 'material_delay_min', 5.0)
                    delay_max = getattr(configs, 'material_delay_max', 40.0)
                    for t in delayed_tasks:
                        delay_time = self.np_random.uniform(delay_min, delay_max)
                        self.task_material_ready[t] = delay_time
                        self.event_queue.push(Event(delay_time, EventType.MATERIAL_ARRIVE, {'task_id': int(t)}))
        
        # [Slot Model] - 记录每个站位中各并行工序的预计完成时间，用于计算等待延迟
        # 小顶堆：记录每个站位中各并行工序的预计完成时间，用于计算等待延迟
        self.station_task_finish_times = [[] for _ in range(self.num_stations)]
        
        self.assigned_tasks = [] 
        self.task_station_map = {} 
        self.task_end_times = -np.ones(self.num_tasks)
        
        # 预计算图拓扑 (前驱/后继)
        self.predecessors = {i: [] for i in range(self.num_tasks)}
        self.successors = {i: [] for i in range(self.num_tasks)}
        
        edge_index = self.raw_data['precedence_edges'].numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)
            
        self.num_preds = np.array([len(self.predecessors[i]) for i in range(self.num_tasks)])
        self.completed_preds = np.zeros(self.num_tasks, dtype=int)
        
        # 设定初始任务状态
        # 没有前驱的任务设为 Ready (1)
        for i in range(self.num_tasks):
            if self.num_preds[i] == 0:
                self.task_status[i] = 1 # Ready
            else:
                self.task_status[i] = 0 # Not Ready
                
        # 克隆 Observation 数据并重建稀疏边矩阵 (由于 worker数量波动)
        self.obs_data = self.base_data.clone()
        
        apply_resource_graph(
            self.obs_data,
            self.base_task_x,
            self.base_worker_x,
            configs,
            skill_hub_topology=self._current_skill_hub_topology(),
        )
        
        # 动态篡改工时
        if randomize_duration:
            rnd_range = configs.dur_random_range
            noise_np = self.np_random.uniform(1.0 - rnd_range, 1.0 + rnd_range, size=self.base_durations.shape)
            noise = torch.tensor(noise_np, dtype=torch.float, device=self.base_durations.device)
            perturbed_durations = self.base_durations * noise
            
            # 刷新模型底层观测到的图静态信息区 (Task_x[0])
            self.base_task_x[:, 0:1] = perturbed_durations
            # [Scale Invariance] 刷新用于仿真计算真实验收时间 (Step duration calculation)
            self.task_static_feat[:, 0] = (perturbed_durations * self.mean_task_time).squeeze()
        else:
            # 安全还原成纯净考题卷子
            self.base_task_x[:, 0:1] = self.base_durations
            self.task_static_feat[:, 0] = (self.base_durations * self.mean_task_time).squeeze()
            
        # [关键路径计算 (CPM)]
        # 用于后续计算 Blocking Penalty
        self.is_critical, _ = self._calculate_cpm()

        if getattr(configs, "enable_reschedule_mode", False):
            self._apply_reschedule_scenario()
            self._advance_time()
        
        return self._get_observation()

    def _topological_sort(self):
        """返回任务的拓扑排序列表"""
        in_degree = self.num_preds.copy()
        queue = [i for i in range(self.num_tasks) if in_degree[i] == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return topo_order

    def _calculate_cpm(self):
        """
        关键路径法 (Critical Path Method, CPM)。
        逻辑:
        1. 正向递推 (Forward Pass) -> 计算最早开始时间 (ES)
        2. 反向递推 (Backward Pass) -> 计算最晚开始时间 (LS)
        3. 关键任务判定: 如果 ES == LS (Slack == 0)，则是关键任务。
        """
        durations = self.task_static_feat[:, 0].numpy()
        num_tasks = self.num_tasks
        
        # 1. 拓扑排序
        topo_order = self._topological_sort()
        
        # 2. 正向递推 (ES)
        es = np.zeros(num_tasks)
        for u in topo_order:
            my_es = 0
            for p in self.predecessors[u]:
                my_es = max(my_es, es[p] + durations[p])
            es[u] = my_es
            
        max_makespan = 0
        for u in range(num_tasks):
            max_makespan = max(max_makespan, es[u] + durations[u])
            
        # 3. 反向递推 (LS)
        ls = np.full(num_tasks, max_makespan)
        for u in reversed(topo_order):
            my_lf = max_makespan
            if self.successors[u]:
                children_ls = [ls[v] for v in self.successors[u]]
                my_lf = min(children_ls)
            
            ls[u] = my_lf - durations[u]
            
        # 4. 判定关键任务
        slack = ls - es
        is_critical = (slack < 1e-5)
        return is_critical, max_makespan

    def _calculate_max_allowed_stations(self):
        """
        通过反向拓扑遍历计算每个任务被允许部署的“最晚站位”。
        这是为了防止 RL 环境将一个无关任务扔到了非常靠后的工位，
        结果发现其【依赖子任务】在更早的站位是被限死 (Fixed Node) 的，导致永恒死锁。
        """
        num_tasks = self.num_tasks
        max_allowed = np.full(num_tasks, self.num_stations - 1)
        
        # 将 Fixed Stations 初始化进 max_allowed 
        for t in range(num_tasks):
            if self.fixed_stations[t] != -1:
                max_allowed[t] = self.fixed_stations[t]
                
        # 拓扑排序 - 用于获取线性处理顺序
        topo_order = self._topological_sort()
                    
        # 沿着反向拓扑序更新最晚允许工位: 父节点的最晚工位不能晚于任何子节点的最晚工位
        for u in reversed(topo_order):
            for p in self.predecessors[u]:
                max_allowed[p] = min(max_allowed[p], max_allowed[u])
                
        return max_allowed

    def calculate_duration(self, task_id, team_indices, start_time_est=None):
        """
        非线性工时计算逻辑:
        T_real = (T_std * N_demand) / (Sum(Eff_i * Fatigue_i) * Synergy_Factor)
        
        Synergy Factor (协同系数): 
        人数越多，沟通成本越高，效率会有折扣。
        设定: 0.95 ^ (人数 - 1)
        """
        task_info = self.task_static_feat[task_id]
        t_std = task_info[0].item()
        n_demand = int(task_info[2].item())
        
        n_act = len(team_indices)
        if n_act == 0: return float('inf')
        
        # 效率求和 (加入动态疲劳系数计算)
        sum_efficiency = 0.0
        for w in team_indices:
            fatigue_f = self.get_current_fatigue_factor(w, start_time_est if start_time_est is not None else self.current_time)
            sum_efficiency += self.worker_efficiency[w] * fatigue_f
        
        # 协同折扣
        syn_factor = 0.95 ** (n_act - 1)
        
        effective_capacity = sum_efficiency * syn_factor
        
        t_real = (t_std * n_demand) / effective_capacity
        return t_real

    def _get_station_earliest_available_time(self, sid, min_start_time, duration):
        """
        寻找最早的起步时间 T (T >= min_start_time)，使得在 [T, T+duration] 期间，
        站位 sid 的并发任务数严格低于 allowed_slots。
        """
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        allowed_slots = self.station_available_slots[sid] if hasattr(self, 'station_available_slots') else max_slots
        
        intervals = [(at[3], at[4]) for at in self.assigned_tasks if at[1] == sid]
        if len(intervals) < allowed_slots:
            return min_start_time
            
        candidate_times = [min_start_time] + [ed for (_, ed) in intervals if ed >= min_start_time]
        candidate_times.sort()
        
        for t in candidate_times:
            test_start = t
            test_end = t + duration
            # 扫描在 [test_start, test_end) 期间与现有区间的重叠
            endpoints = []
            for (st, ed) in intervals:
                # 严格重叠条件
                if max(st, test_start) < min(ed, test_end) - 1e-5:
                    endpoints.append((max(st, test_start), 1))
                    endpoints.append((min(ed, test_end), -1))
            
            if not endpoints: return test_start
            
            endpoints.sort(key=lambda x: (x[0], x[1]))
            cur_overlap = 0
            is_valid = True
            for pos, val in endpoints:
                cur_overlap += val
                if cur_overlap >= allowed_slots:
                    is_valid = False
                    break
            
            if is_valid: return test_start
            
        return candidate_times[-1]

    def _get_estimated_cmax(self):
        """
        [Phase 7: Estimated Cmax]
        计算预估完工期 (Estimated Cmax)，用于指导单步截断的强化学习，防止智能体恶意推迟长耗时任务。
        Cmax_est = max( 当前最大完工期, 各站位平均完工期 + (未排队任务总标准耗时 / 站位数 * 预估槽位) )
        """
        curr_max = np.max(self.station_wall_clock)
        
        # 取未分配的任务：0=Wait, 1=Ready
        unassigned_mask = (self.task_status == 0) | (self.task_status == 1)
        unassigned_sum = self.task_static_feat[unassigned_mask, 0].sum().item()
        
        from configs import configs
        slots = configs.estimated_cmax_station_slots
        
        curr_mean = np.mean(self.station_wall_clock)
        lower_bound = curr_mean + (unassigned_sum / (self.num_stations * slots))
        
        return max(curr_max, lower_bound)

    def _reject_invalid_action(self, reason: str, details: Dict[str, Any]) -> Tuple[HeteroData, float, bool, Dict[str, Any]]:
        """将违反 APAL 硬约束的动作拒绝在环境边界之外。"""
        info = {'error': reason, 'invalid_action': True}
        info.update(details)
        reward = -50.0 * configs.reward_scale
        if getattr(self, 'skip_obs_building', False):
            return None, reward, False, info
        return self._get_observation(), reward, False, info

    def _validate_worker_team(self, task_id: int, station_id: int, team: List[int]) -> Optional[Dict[str, Any]]:
        """校验正工时工序的派工人数、技能、重复工人与跨站锁定约束。"""
        if task_id < 0 or task_id >= self.num_tasks:
            return {'reason': 'invalid_task_id', 'task_id': task_id}
        if self.task_status[task_id] != 1:
            return {'reason': 'task_not_ready_or_already_fixed', 'task_id': task_id, 'status': int(self.task_status[task_id])}
        if (
            hasattr(self, 'task_material_ready')
            and not time_reached_scalar(
                self.task_material_ready[task_id],
                self.current_time,
                release_time_tolerance(configs),
            )
        ):
            return {
                'reason': 'task_release_time_not_reached',
                'task_id': task_id,
                'release_time': float(self.task_material_ready[task_id]),
                'current_time': float(self.current_time),
            }
        if station_id < -1 or station_id >= self.num_stations:
            return {'reason': 'invalid_station_id', 'station_id': station_id}

        duration_raw = float(self.task_static_feat[task_id, 0].item())
        if duration_raw <= 1e-5:
            if station_id != -1 or team:
                return {
                    'reason': 'virtual_task_requires_no_resources',
                    'task_id': task_id,
                    'station_id': station_id,
                    'team': team,
                }
            return None

        station_invalid = self.constraint_engine.station_violation(
            task_id,
            station_id,
            self.task_station_map,
        )
        if station_invalid is not None:
            return station_invalid

        team = [int(w) for w in team]
        demand = max(1, int(self.task_static_feat[task_id, 2].item()))
        if len(team) < demand:
            return {'reason': 'insufficient_workers', 'task_id': task_id, 'required': demand, 'actual': len(team)}
        if len(team) > demand:
            return {'reason': 'excess_workers', 'task_id': task_id, 'required': demand, 'actual': len(team)}
        if len(team) != len(set(team)):
            return {'reason': 'duplicate_workers', 'task_id': task_id, 'team': team}

        req_skill = int(self.task_static_feat[task_id, 1].item())
        for w in team:
            if w < 0 or w >= self.num_workers:
                return {'reason': 'worker_out_of_range', 'task_id': task_id, 'worker_id': w, 'num_workers': self.num_workers}
            if self.worker_skill_matrix[w, req_skill] < 0.5:
                return {'reason': 'worker_skill_mismatch', 'task_id': task_id, 'worker_id': w, 'skill': req_skill}
            if self.worker_locks[w] != 0 and self.worker_locks[w] != station_id + 1:
                return {
                    'reason': 'worker_station_lock_mismatch',
                    'task_id': task_id,
                    'worker_id': w,
                    'locked_station': int(self.worker_locks[w]),
                    'target_station': station_id + 1,
                }
        return None

    def validate_assignments(
        self,
        assignments: Iterable[Assignment],
    ) -> ScheduleValidationReport:
        """使用统一约束引擎校验完整的环境内部排程。"""
        return self.constraint_engine.validate_schedule(
            assignments,
            demands=self.task_static_feat[:, 2].detach().cpu().numpy(),
            required_skills=self.task_static_feat[:, 1].detach().cpu().numpy(),
            worker_skill_matrix=self.worker_skill_matrix.detach().cpu().numpy(),
            max_slots_per_station=int(getattr(configs, 'max_slots_per_station', 3)),
        )

    def _reschedule_objective_terms(
        self,
        *,
        makespan: float,
        balance_std: float,
    ) -> dict[str, float]:
        """计算当前部分排程的统一归一化目标，不使用任何未来动作信息。"""
        if self.baseline_schedule is None:
            raise RuntimeError("重调度目标需要 baseline schedule")
        stability = calculate_stability_metrics(
            self.baseline_schedule,
            self.assigned_tasks,
            current_time=float(self.reschedule_start_time),
        )
        return calculate_reschedule_objective_terms(
            makespan=float(makespan),
            balance_std=float(balance_std),
            takt_h=float(self.baseline_schedule.makespan),
            takt_violation_h=None,
            start_deviation_mean_h=float(stability["start_deviation_mean_h"]),
            station_change_rate=float(stability["station_change_rate"]),
            team_change_rate=float(stability["team_change_rate"]),
            config_obj=configs,
            ideal_station_load=float(self.ideal_station_load),
        )

    def step(self, action: Action) -> Tuple[HeteroData, float, bool, Dict[str, Any]]:
        """
        执行一步动作。
        Action: (task_id, station_id, team_list)
        """
        task_id, station_id, team = action
        task_id = int(task_id)
        station_id = int(station_id)
        team = [int(w) for w in team]

        # 虚拟层级节点不占用站位和工人；兼容旧调用方传入的占位动作。
        if 0 <= task_id < self.num_tasks and not self.constraint_engine.physical_mask[task_id]:
            station_id = -1
            team = []

        invalid = self._validate_worker_team(task_id, station_id, team)
        if invalid is not None:
            return self._reject_invalid_action(invalid['reason'], invalid)

        if not self.constraint_engine.physical_mask[task_id]:
            finish_time = float(self.current_time)
            self.task_status[task_id] = 2
            self.task_end_times[task_id] = finish_time
            self.task_station_map[task_id] = -1
            self.assigned_tasks.append((task_id, -1, [], finish_time, finish_time))
            self.event_queue.push(
                Event(
                    finish_time,
                    EventType.TASK_FINISH,
                    {'task_id': task_id, 'worker_ids': [], 'station_id': -1},
                )
            )
            self._advance_time()
            done = len(self.assigned_tasks) == self.num_tasks
            observation = None if getattr(self, 'skip_obs_building', False) else self._get_observation()
            return observation, 0.0, done, {'virtual_task': True}
        
        # 记录执行前的 makespan 与平衡差 (Telescoping Sum Calculation Base)
        # 变更为具备下界预测能力的 Cmax_est
        prev_makespan = self._get_estimated_cmax()
        prev_std = np.std(self.station_loads)
        dense_objective_enabled = bool(
            getattr(configs, "enable_reschedule_mode", False)
            and self.baseline_schedule is not None
            and getattr(configs, "reschedule_use_objective_delta_reward", False)
        )
        prev_objective_terms = (
            self._reschedule_objective_terms(
                makespan=float(prev_makespan),
                balance_std=float(prev_std),
            )
            if dense_objective_enabled
            else None
        )
        
        # [Dynamic Events] 尝试注入工时扰动
        self._try_inject_online_duration_perturb()
        
        # ==========================================
        # [Forward Allocation Engine]
        # 计算该工序真正的起步时间：必须满足（现在，人齐，且工位有空）
        # ==========================================
        
        # 1. 团队集结完毕时间 (木桶原理)
        team_ready_time = self.current_time
        if team:
            team_ready_time = max([self.worker_free_time[w] for w in team])
            
        # [NEW] 2. 前置工序完成时间 (彻底解决拓扑时序错乱 Bug)
        pred_ready_time = self.current_time
        preds = self.predecessors.get(task_id, [])
        for p in preds:
            pred_ready_time = max(pred_ready_time, self.task_end_times[p])

        # 结合基本条件，计算进入工位的【最低期望起始点】
        min_start_bound = max(self.current_time, team_ready_time, pred_ready_time)
        
        # 3. 计算真实执行工时 (考虑工人在此起步时间时的疲劳恢复)
        duration = self.calculate_duration(task_id, team, start_time_est=min_start_bound)

        # [NEW] 4. 站位槽位腾出时间 (解决超并发重叠，使用 Sweep Line)
        station_ready_time = min_start_bound
        if station_id >= 0:
            station_ready_time = self._get_station_earliest_available_time(station_id, min_start_bound, duration)
            
        # 5. 四者取大，得到实际安排进时间表里的开工时刻
        start_time = max(min_start_bound, station_ready_time)
        finish_time = start_time + duration
        team_wait_h = max(0.0, team_ready_time - self.current_time)
        station_wait_h = max(0.0, station_ready_time - min_start_bound)
        worker_idle_ratio_before = float(np.mean(self.worker_free_time <= self.current_time)) if self.num_workers > 0 else 0.0
        total_slots_before = max(1, int(np.sum(self.station_available_slots)))
        busy_slots_before = 0
        for s in range(self.num_stations):
            busy_slots_before += sum(1 for finish_t in self.station_task_finish_times[s] if finish_t > self.current_time)
        station_slot_vacancy_ratio_before = max(0.0, float(total_slots_before - busy_slots_before) / float(total_slots_before))
        
        # 更新工人状态与站位绑定
        for w in team:
            # [Fatigue Update] 物理更新疲劳状态
            self._update_worker_fatigue(w, start_time, finish_time)
            
            self.worker_free_time[w] = finish_time
            if self.worker_locks[w] == 0 and station_id != -1:
                self.worker_locks[w] = station_id + 1
        
        if station_id != -1:
            # [Slot Model] 将本工序完成时间塞入站位的可用时间池
            # 不再维护已废弃的 station_active_tasks
            heapq.heappush(self.station_task_finish_times[station_id], finish_time)
            
            # 更新站位工作量总和 (Workload - 人.小时)
            self.station_loads[station_id] += duration * len(team) 
            
            # 更新真实的站位物理下班时间 (Wall-Clock Makespan)
            self.station_wall_clock[station_id] = max(self.station_wall_clock[station_id], finish_time)
        
        # 更新任务状态
        self.task_status[task_id] = 2 # 2=已调度
        self.task_end_times[task_id] = finish_time
        self.task_station_map[task_id] = station_id
        
        self.assigned_tasks.append((task_id, station_id, team, start_time, finish_time))
        # 2. 添加事件到队列
        self.event_queue.push(Event(finish_time, EventType.TASK_FINISH, 
                                    {'task_id': task_id, 'worker_ids': team, 'station_id': station_id}))
        
        # 3. 推进仿真时间 (离散事件引擎)
        self._advance_time()
        curr_makespan = self._get_estimated_cmax()
        curr_std = np.std(self.station_loads)
        
        delta_makespan = curr_makespan - prev_makespan
        delta_std = curr_std - prev_std
        
        # [Scale Invariance] Reward Reshaping
        # 修复 vloss 尖峰：分母改为理想总完工时间 (ideal_makespan)
        # 这样无论图大小，整个 episode 累积的 makespan_reward 总和都稳定在 -1.0 到 -2.0 左右 (即完工时间是理想时间的几倍)
        # 乘以 100.0 是为了保持单步数值量级与之前相近，防止之前配置的 reward_scale 过小导致梯度消失
        norm_delta_makespan = (delta_makespan / max(1e-5, self.ideal_makespan)) * 100.0
        
        # 负载方差原本就除以了 ideal_station_load（随数据集大小等比放大），原理正确，只需同样乘 100 保持系数平衡
        norm_delta_std = (delta_std / max(1.0, self.ideal_station_load)) * 100.0
        
        coef_makespan = configs.r_coef_makespan
        coef_std = configs.r_coef_std
        
        base_penalty = -(coef_makespan * norm_delta_makespan) - (coef_std * norm_delta_std)
        wait_penalty_raw = (
            getattr(configs, 'r_coef_wait', 0.0)
            * ((team_wait_h + station_wait_h) / max(1e-6, self.mean_task_time))
        )
        idle_penalty_raw = getattr(configs, 'r_coef_idle', 0.0) * worker_idle_ratio_before
        if getattr(configs, 'enable_resource_wait_penalty', False):
            base_penalty -= (wait_penalty_raw + idle_penalty_raw)
        
        # 关键路径激励 (Critical Path Incentive)
        # 如果调度的是关键路径任务且全局开关开启，我们在进度奖励上分配更高的权重，激励智能体优先排关键路径工序
        use_cpm = getattr(configs, 'enable_cpm_reward', True)
        is_task_critical = (self.is_critical[task_id] if (hasattr(self, 'is_critical') and self.is_critical is not None) else False) and use_cpm
        
        dense_term_deltas: dict[str, float] = {}
        curr_objective_terms: dict[str, float] | None = None
        dense_unclipped_reward = 0.0
        dense_reward_clipped = False
        if dense_objective_enabled:
            assert prev_objective_terms is not None
            curr_objective_terms = self._reschedule_objective_terms(
                makespan=float(curr_makespan),
                balance_std=float(curr_std),
            )
            dense_term_deltas = {
                key: float(curr_objective_terms[key] - prev_objective_terms[key])
                for key in curr_objective_terms
            }
            multiplier = float(getattr(configs, "reschedule_objective_delta_multiplier", 100.0))
            reward = -multiplier * float(sum(dense_term_deltas.values()))
            dense_unclipped_reward = float(reward)
            clip_limit = float(getattr(configs, "reschedule_objective_delta_clip", 50.0))
            if clip_limit > 0.0:
                reward = float(np.clip(reward, -clip_limit, clip_limit))
                dense_reward_clipped = not np.isclose(reward, dense_unclipped_reward)
        elif getattr(configs, 'use_dense_progress_reward', False):
            delta_progress = 1.0 / self.num_tasks
            progress_multiplier = 2.0 if is_task_critical else 1.0
            norm_delta_progress = delta_progress * 100.0 * progress_multiplier
            coef_progress = getattr(configs, 'r_coef_progress', 1.0)
            reward = base_penalty + (coef_progress * norm_delta_progress)
        else:
            critical_bonus = 0.5 if is_task_critical else 0.0
            reward = base_penalty + critical_bonus

        # 旧奖励沿用固定截断；统一目标差分使用自己的可配置截断。
        if not dense_objective_enabled:
            reward = np.clip(reward, -50.0, 50.0)
        
        # 全局奖励缩放乘数：把原始的巨大的 makespan 分差在底层压缩至 [-5, 5] 的健康小区间
        reward = reward * configs.reward_scale
        
        # F. 终局结算 (Final Cleansing)
        done = (len(self.assigned_tasks) == self.num_tasks)
        reschedule_info: dict[str, float] = {}
        if done and getattr(configs, "enable_reschedule_mode", False) and self.baseline_schedule is not None:
            final_makespan = float(np.max(self.station_wall_clock)) if len(self.station_wall_clock) > 0 else 0.0
            stability = calculate_stability_metrics(
                self.baseline_schedule,
                self.assigned_tasks,
                current_time=float(getattr(self, "reschedule_start_time", 0.0)),
            )
            takt = max(1e-6, float(self.baseline_schedule.makespan))
            feasible_scale = 1.0 if self.reschedule_takt_feasible else float(getattr(configs, "reschedule_infeasible_stability_relax", 0.35))
            start_pen = float(getattr(configs, "reschedule_stability_start_weight", 0.20)) * (stability["start_deviation_mean_h"] / takt)
            station_pen = float(getattr(configs, "reschedule_stability_station_weight", 0.10)) * stability["station_change_rate"]
            team_pen = float(getattr(configs, "reschedule_stability_team_weight", 0.05)) * stability["team_change_rate"]
            takt_violation = max(0.0, final_makespan - takt)
            takt_pen = float(getattr(configs, "reschedule_takt_violation_weight", 1.0)) * (takt_violation / takt)
            stability_penalty = feasible_scale * (start_pen + station_pen + team_pen)
            if not dense_objective_enabled:
                reward -= float((takt_pen + stability_penalty) * configs.reward_scale * 100.0)
            reschedule_info = {
                'reschedule_takt_h': float(takt),
                'reschedule_takt_violation_h': float(takt_violation),
                'reschedule_lower_bound_h': float(getattr(self, "reschedule_lower_bound", 0.0)),
                'reschedule_takt_feasible': float(bool(getattr(self, "reschedule_takt_feasible", True))),
                'reschedule_start_deviation_mean_h': stability["start_deviation_mean_h"],
                'reschedule_station_change_rate': stability["station_change_rate"],
                'reschedule_team_change_rate': stability["team_change_rate"],
                'reschedule_stability_penalty': float(stability_penalty),
            }
        
        # 收集物理诊断指标传回主进程，用于 TensorBoard 进行 Reward 分项拆解
        if dense_objective_enabled:
            multiplier = float(getattr(configs, "reschedule_objective_delta_multiplier", 100.0))
            makespan_penalty = float(
                dense_term_deltas.get("score_makespan", 0.0) * multiplier * configs.reward_scale
            )
            std_penalty = float(
                dense_term_deltas.get("score_balance", 0.0) * multiplier * configs.reward_scale
            )
        else:
            makespan_penalty = float(coef_makespan * norm_delta_makespan * configs.reward_scale)
            std_penalty = float(coef_std * norm_delta_std * configs.reward_scale)
        info = {
            'makespan_penalty': makespan_penalty,
            'std_penalty': std_penalty,
            'resource_wait_penalty_candidate': float(wait_penalty_raw * configs.reward_scale),
            'resource_idle_penalty_candidate': float(idle_penalty_raw * configs.reward_scale),
            'team_wait_h': float(team_wait_h),
            'station_wait_h': float(station_wait_h),
            'worker_idle_ratio_before': float(worker_idle_ratio_before),
            'station_slot_vacancy_ratio_before': float(station_slot_vacancy_ratio_before),
        }
        if dense_objective_enabled:
            info['reschedule_objective_score'] = float(sum((curr_objective_terms or {}).values()))
            info['reschedule_objective_delta'] = float(sum(dense_term_deltas.values()))
            info['reschedule_objective_reward_unscaled'] = dense_unclipped_reward
            info['reschedule_objective_reward_clipped'] = float(dense_reward_clipped)
            for key, value in dense_term_deltas.items():
                info[f'reschedule_delta_{key}'] = float(value)
        info.update(reschedule_info)
        if (
            not dense_objective_enabled
            and not getattr(configs, 'use_dense_progress_reward', False)
            and is_task_critical
        ):
            info['critical_bonus'] = float(critical_bonus * configs.reward_scale)
            
        if getattr(self, 'skip_obs_building', False):
            return None, reward, done, info
        return self._get_observation(), reward, done, info

    def _advance_time(self):
        """
        推进时间 current_time 到下一个事件点。
        处理逻辑:
        1. 处理所有 <= current_time 的事件 (Task Finish)，释放前驱。
        2. [Zero-Duration Logic]: 如果解锁了 0工时 任务，立即执行并完成，不推进时间。
        3. 检查是否有 Valid 任务可做。
           - 如果有 -> 返回控制权给 Agent。
           - 如果无 -> 跳跃到下一个事件发生的时间点。
        """
        while True:
            # 删除破坏性的 early return，因为在空队列情况下，仍然可能需要处理 0 工时任务
            
            # 由于 EventQueue 内部已控制容量，这里的长度告警可以移除或使用 len()
            if len(self.event_queue) > 10000:
                print("WARNING: Event queue limit exceeded! Forcing episode end to prevent OOM/Infinite Loop.")
                self.current_time = self.max_time
                self.event_queue.clear()
                return
                
            # 1. 处理所有已到期的事件
            while (
                not self.event_queue.is_empty()
                and time_reached_scalar(
                    self.event_queue.peek().time,
                    self.current_time,
                    release_time_tolerance(configs),
                )
            ):
                ev = self.event_queue.pop()
                if ev.type == EventType.TASK_FINISH:
                    tid = ev.data['task_id']
                    sid = ev.data['station_id']
                    # [Slot Model] 释放工位的历史使用记录 (将其从堆中清理)
                    # 由于我们使用 finish_time 推入，这里理论上不需要严苛清理，
                    # 只要为了防止 heap 无限膨胀而在完成时 pop 一次堆顶即可 (或者让其在下一次被覆写)
                    if sid >= 0:
                        if self.station_task_finish_times[sid]:
                            heapq.heappop(self.station_task_finish_times[sid])
                    
                    # 解锁后继
                    for succ in self.successors[tid]:
                        self.completed_preds[succ] += 1
                        if self.completed_preds[succ] == self.num_preds[succ]:
                            if self.task_status[succ] == 0:
                                self.task_status[succ] = 1 # Ready
                elif ev.type == EventType.WORKER_LEAVE:
                    w = ev.data['worker_id']
                    duration = ev.data['duration']
                    # 强行推迟可用时间（如果正忙，则等他忙完继续推迟；如果空闲，立即推迟）
                    self.worker_free_time[w] = max(self.worker_free_time[w], self.current_time) + duration
                    # 加入回归锚点，防止引擎死锁
                    self.event_queue.push(Event(self.worker_free_time[w], EventType.WORKER_RETURN, {'worker_id': w}))
                elif ev.type == EventType.WORKER_RETURN:
                    # 仅作为时钟锚点，什么也不需要做，ActionMasker 会自动发现他 free 了
                    pass
                elif ev.type == EventType.STATION_BREAKDOWN:
                    sid = ev.data['station_id']
                    lost = ev.data['lost_slots']
                    self.station_available_slots[sid] = max(1, self.station_available_slots[sid] - lost)
                elif ev.type == EventType.STATION_RECOVER:
                    sid = ev.data['station_id']
                    lost = ev.data['lost_slots']
                    self.station_available_slots[sid] = min(configs.max_slots_per_station, self.station_available_slots[sid] + lost)
                elif ev.type == EventType.DURATION_PERTURB:
                    # 已经在 step 起始处以即时形式进行了前瞻性修改以影响 duration 计算
                    pass
                elif ev.type == EventType.MATERIAL_ARRIVE:
                    # 仅作为时钟唤醒锚点，ActionMasker 在此时间点之后会自动解锁对应的物料限制
                    pass
            
            # 2. 0工时任务穿透逻辑 (Zero-Duration Penetration)
            # 必须立即处理掉所有 Ready 的 0工时任务
            ready_indices = np.where(self.task_status == 1)[0]
            zero_run_count = 0
            for t in ready_indices:
                dur = self.task_static_feat[t, 0].item()
                if dur < 1e-5: # Zero duration
                    if (
                        hasattr(self, 'task_material_ready')
                        and not time_reached_scalar(
                            self.task_material_ready[t],
                            self.current_time,
                            release_time_tolerance(configs),
                        )
                    ):
                        continue
                    # 立即完成
                    self.task_status[t] = 2 # Scheduled/Done
                    finish_time = self.current_time
                    self.task_end_times[t] = finish_time
                    self.task_station_map[t] = -1 # Virtual task
                    self.assigned_tasks.append((t, -1, [], finish_time, finish_time))
                    
                    # 加入事件队列 (为了统一触发 unlock 逻辑)
                    self.event_queue.push(Event(finish_time, EventType.TASK_FINISH, 
                                                {'task_id': t, 'worker_ids': [], 'station_id': -1}))
                    zero_run_count += 1
                    
            if zero_run_count > 0:
                self.total_zero_runs = getattr(self, 'total_zero_runs', 0) + zero_run_count
                if self.total_zero_runs > self.num_tasks * 2:
                    raise RuntimeError(f"Infinite loop detected in 0-duration task resolution at time {self.current_time}. Check task graph topology.")
                # 如果处理了 0工时任务，可能解锁了新任务，需要重新进入循环检查
                continue
            else:
                self.total_zero_runs = 0
            
            # 3. 检查是否需要 Agent 介入
            # 只有当存在 "可行 (Valid)" 任务时，才暂停并在 State 中返回。
            task_mask, _, _ = self.get_masks()
            
            if not task_mask.all():
                 # 至少有一个任务是 False (即 Valid)
                 break
            
            # 4. 如果没有 Valid 任务，则必须跳跃时间 (交由 _advance_time 内部或外部决定)
            if self.event_queue.is_empty():
                # 真正的环境空转末端，退出循环，让外部拿到掩码后再决定是否调用 try_wait_for_resources
                break
            
            # Jump to next event
            next_ev = self.event_queue.peek()
            self.current_time = next_ev.time
            # 循环会继续处理 next_ev

    def try_wait_for_resources(self):
        """
        [Deadlock Fix] 当外部(如 train.py)拿到全 False 的 mask 时调用此方法。
        主动将时间快进到下一个事件发生（释放工人或槽位），然后返回 True。
        如果连未来的事件也没有了，说明发生了真正的死锁，返回 False。
        """
        if self.event_queue.is_empty():
            return False  # 真正的死锁：无人可用，且也没有人正在干活
            
        next_ev = self.event_queue.peek()
        self.current_time = next_ev.time
        self._advance_time()  # 触发内部事件释放并尝试解锁新任务
        return True

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成动作掩码 (Action Masking)。
        逻辑已抽离至 core.action_masker 模块。
        """
        return ActionMasker(self).get_masks()

    def _fill_reschedule_task_features(self, task_x: torch.Tensor, *, current_time: float, task_release_ready: np.ndarray) -> None:
        if not getattr(configs, "enable_reschedule_mode", False):
            return
        if task_x.size(1) < 24 or self.baseline_schedule is None:
            return
        takt = max(1e-6, float(self.baseline_schedule.makespan))
        task_status_arr = self.task_status
        task_x[:, 18:24] = 0.0
        for task_id, base_task in self.baseline_schedule.tasks.items():
            if task_id < 0 or task_id >= task_x.size(0):
                continue
            task_x[task_id, 18] = float(base_task.start - current_time) / takt
            task_x[task_id, 19] = float(base_task.station_id + 1) / max(1, self.num_stations)
            task_x[task_id, 20] = float(len(base_task.team)) / max(1, self.num_workers)
            task_x[task_id, 21] = 1.0 if base_task.start <= self.reschedule_start_time + 1e-9 else 0.0
            task_x[task_id, 22] = 1.0 if task_release_ready[task_id] > self.reschedule_start_time + 1e-9 else 0.0
            if current_time > base_task.start + 1e-9 and task_status_arr[task_id] != 2:
                task_x[task_id, 23] = float(current_time - base_task.start) / takt

    def _get_observation(self):
        """
        [Phase 3.1: O(1) In-place Observation]
        构建异构图观测状态 (Observation)。
        彻底放弃在仿真步内的张量创建和拼接，转为 O(1) 预建内存片段的原地刷新。
        """
        data = self.base_data.clone()
        
        # 1. Task Features (In-place refresh)
        task_x = self.base_task_x.clone()
        task_x[:, 1:5] = 0.0 # reset status
        task_x[torch.arange(self.num_tasks), self.task_status + 1] = 1.0 # set status (offset by 1 to skip duration)
        
        # [Dynamic Events] 物料延迟到达特征写入第 [17] 维 (wait_time / mean_task_time)，使用对数归一化
        wait_times_t = np.maximum(0, self.task_material_ready - self.current_time)
        task_x[:, 17] = torch.log1p(torch.tensor(wait_times_t, dtype=torch.float) / self.mean_task_time)
        self._fill_reschedule_task_features(
            task_x,
            current_time=float(self.current_time),
            task_release_ready=self.task_material_ready,
        )
        
        data['task'].x = task_x
        
        # 2. Worker Features (In-place refresh)
        worker_x = self.base_worker_x.clone()
        
        # [Feature Upgrade: 连续时间特征支撑排队决策]
        # 计算工人的预估等待时间: max(0, worker_free_time - current_time) / self.mean_task_time，使用对数归一化
        wait_times_w = np.maximum(0, self.worker_free_time - self.current_time)
        # Efficiency(0), Skills(1~10), ProjectedWait(11)
        worker_x[:, 11] = torch.log1p(torch.tensor(wait_times_w, dtype=torch.float) / self.mean_task_time)
        
        is_free_bool = (self.worker_free_time <= self.current_time)
        worker_x[:, 12] = torch.tensor(is_free_bool, dtype=torch.float)
        
        # [Feature Upgrade] One-Hot Encode Lock state 
        worker_x[:, 13:21] = 0.0 # Clear
        lock_indices = torch.tensor(self.worker_locks, dtype=torch.long)
        lock_indices = torch.clamp(lock_indices, max=7) 
        worker_x[torch.arange(self.num_workers), 13 + lock_indices] = 1.0
        
        # [Dynamic Events] 工人疲劳因子写入第 [21] 维 (动态 Lazy 计算，确保无滞后)
        for w in range(self.num_workers):
            worker_x[w, 21] = self.get_current_fatigue_factor(w, self.current_time)
            
        data['worker'].x = worker_x
        apply_resource_graph(
            data,
            task_x,
            worker_x,
            configs,
            skill_hub_topology=self._current_skill_hub_topology(),
        )
        
        # 3. Station Features (In-place refresh)
        station_x = self.base_station_x.clone()
        # [Scale Invariance] 物理累积负荷基于理论均分值归一化，代替死板的 / 1000.0
        station_x[:, 0] = torch.tensor(self.station_loads, dtype=torch.float) / max(1.0, self.ideal_station_load)
        
        # [Feature Upgrade: Relative Load Competition]
        sum_loads = np.sum(self.station_loads)
        max_load = np.max(self.station_loads)
        station_x[:, 5] = torch.tensor(self.station_loads / (sum_loads + 1e-6), dtype=torch.float)
        station_x[:, 6] = torch.tensor(self.station_loads / (max_load + 1e-6), dtype=torch.float)
        
        # [Dynamic Events] 工位当前可用槽位占比写入第 [7] 维
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        station_x[:, 7] = torch.tensor(self.station_available_slots, dtype=torch.float) / max_slots
        
        # 计算站位槽位释放时间，使用对数归一化
        for s in range(self.num_stations):
            heap = self.station_task_finish_times[s]
            allowed_slots = self.station_available_slots[s]
            if len(heap) >= allowed_slots:
                wait_time_s = max(0, heap[0] - self.current_time)
            else:
                wait_time_s = 0.0
            station_x[s, 4] = math.log1p(wait_time_s / self.mean_task_time)
            
        # [Feature Upgrade] Macro Strategic Features for Path Planning
        _fill_station_macro_features(station_x, self.worker_locks, is_free_bool)
            
        data['station'].x = station_x
        
        # 4. Dynamic Edges (Assigned To)
        ts_src, ts_dst, tw_src, tw_dst = [], [], [], []
        for t_id, s_id, team, _, _ in self.assigned_tasks:
            if s_id != -1:
                ts_src.append(t_id)
                ts_dst.append(s_id)
                for w_id in team:
                    tw_src.append(t_id)
                    tw_dst.append(w_id)
                    
        if ts_src:
            t_s_edge = torch.tensor([ts_src, ts_dst], dtype=torch.long)
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if tw_src:
            t_w_edge = torch.tensor([tw_src, tw_dst], dtype=torch.long)
        else:
            t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        return data

    def get_state_snapshot(self):
        """生成状态轻量级切片以存入 Buffer。不包含 can_do_edge_index（从 ctx 缓存读取）。"""
        snapshot = {
            'task_status': self.task_status.copy(),
            'worker_free_time': self.worker_free_time.copy(),
            'worker_locks': self.worker_locks.copy(),
            'station_loads': self.station_loads.copy(),
            'station_wall_clock': self.station_wall_clock.copy(),
            'current_time': self.current_time,
            'assigned_tasks': list(self.assigned_tasks),
            'base_worker_x': self.base_worker_x.clone(),
            'dataset_idx': getattr(self, 'active_dataset_idx', 0),
            'worker_topology_key': self._active_worker_topology_key,
            # [Dynamic Events] 保存新增状态变量
            'station_available_slots': self.station_available_slots.copy(),
            'task_material_ready': self.task_material_ready.copy(),
            'worker_cumulative_work': self.worker_cumulative_work.copy(),
            'worker_fatigue_factor': self.worker_fatigue_factor.copy(),
            'worker_last_busy_end': self.worker_last_busy_end.copy(),
            'reschedule_start_time': float(getattr(self, 'reschedule_start_time', 0.0)),
            'reschedule_takt_feasible': bool(getattr(self, 'reschedule_takt_feasible', True)),
            'reschedule_lower_bound': float(getattr(self, 'reschedule_lower_bound', 0.0)),
        }
        if getattr(configs, "enable_reschedule_mode", False) and self.baseline_schedule is not None:
            base_start = np.zeros(self.num_tasks, dtype=float)
            base_station = -np.ones(self.num_tasks, dtype=int)
            base_team_size = np.zeros(self.num_tasks, dtype=float)
            base_frozen = np.zeros(self.num_tasks, dtype=float)
            for task_id, base_task in self.baseline_schedule.tasks.items():
                if 0 <= task_id < self.num_tasks:
                    base_start[task_id] = float(base_task.start)
                    base_station[task_id] = int(base_task.station_id)
                    base_team_size[task_id] = float(len(base_task.team))
                    base_frozen[task_id] = 1.0 if base_task.start <= self.reschedule_start_time + 1e-9 else 0.0
            snapshot.update({
                'baseline_makespan': float(self.baseline_schedule.makespan),
                'baseline_start': base_start,
                'baseline_station': base_station,
                'baseline_team_size': base_team_size,
                'baseline_frozen': base_frozen,
            })
        return snapshot
        
    def rebuild_state_from_snapshot(
        self,
        snapshot,
        *,
        reusable_state: HeteroData | None = None,
        reuse_resource_topology: bool = False,
    ):
        """
        基于快照恢复成 PyG 图结构，避免完整异构图深拷贝带来的极高缓存占用。
        """
        # 溯源：找到生成该快照时的底层骨架上下文，免疫维度错位崩溃
        ctx_idx = snapshot.get('dataset_idx', 0)
        ctx = self.dataset_pool[ctx_idx]
        if "base_data" not in ctx:
            active_idx = int(getattr(self, "active_dataset_idx", 0))
            restore_fields = {
                name: getattr(self, name)
                for name in (
                    "raw_data",
                    "num_tasks",
                    "base_data",
                    "base_task_x",
                    "base_worker_x",
                    "base_station_x",
                    "task_static_feat",
                    "worker_skill_matrix",
                    "predecessors",
                    "successors",
                    "num_preds",
                    "fixed_stations",
                    "constraint_engine",
                    "mean_task_time",
                    "ideal_station_load",
                    "ideal_makespan",
                    "total_base_workload",
                    "base_durations",
                    "max_allowed_stations",
                    "is_critical",
                    "full_worker_efficiency",
                    "full_worker_skill_matrix",
                    "worker_efficiency",
                    "worker_static_feat",
                    "_active_worker_topology_key",
                )
                if hasattr(self, name)
            }
            ctx = self._ensure_dataset_context(ctx_idx)
            if ctx_idx != active_idx:
                for name, value in restore_fields.items():
                    setattr(self, name, value)
        
        raw_worker_topology_key = snapshot.get("worker_topology_key")
        topology_digest = hashlib.sha256(
            repr(raw_worker_topology_key).encode("utf-8")
        ).hexdigest()
        topology_key = (
            f"skill={int(bool(getattr(configs, 'use_skill_hub', False)))};"
            f"bidir={int(bool(getattr(configs, 'skill_hub_bidirectional', False)))};"
            f"tasks={int(ctx['num_tasks'])};workers={int(len(snapshot['worker_free_time']))};"
            f"worker_topology={topology_digest}"
        )
        if reusable_state is not None:
            if not reuse_resource_topology:
                raise ValueError("传入 reusable_state 时必须显式启用 reuse_resource_topology")
            cached_key = getattr(reusable_state, "apal_resource_topology_key", None)
            if cached_key != topology_key:
                raise ValueError(
                    "可复用观测的静态拓扑与当前快照不一致: "
                    f"cached={cached_key!r}, current={topology_key!r}"
                )
            data = reusable_state
        else:
            data = ctx['base_data'].clone()
        
        task_x = ctx['base_task_x'].clone()
        task_x[:, 1:5] = 0.0
        task_x[torch.arange(ctx['num_tasks']), snapshot['task_status'] + 1] = 1.0
        
        # [Dynamic Events] 重建第 17 维物料准备特征，使用对数归一化
        snap_mat = snapshot.get('task_material_ready', np.zeros(ctx['num_tasks']))
        wait_times_t = np.maximum(0, snap_mat - snapshot['current_time'])
        task_x[:, 17] = torch.log1p(torch.tensor(wait_times_t, dtype=torch.float) / ctx['mean_task_time'])
        if task_x.size(1) >= 24 and 'baseline_start' in snapshot:
            takt = max(1e-6, float(snapshot.get('baseline_makespan', 1.0)))
            task_x[:, 18] = torch.tensor((snapshot['baseline_start'] - snapshot['current_time']) / takt, dtype=torch.float)
            task_x[:, 19] = torch.tensor((snapshot['baseline_station'] + 1) / max(1, self.num_stations), dtype=torch.float)
            task_x[:, 20] = torch.tensor(snapshot['baseline_team_size'] / max(1, len(snapshot['worker_free_time'])), dtype=torch.float)
            task_x[:, 21] = torch.tensor(snapshot['baseline_frozen'], dtype=torch.float)
            task_x[:, 22] = torch.tensor((snap_mat > snapshot.get('reschedule_start_time', 0.0) + 1e-9).astype(float), dtype=torch.float)
            cur_t = float(snapshot['current_time'])
            snap_status = snapshot['task_status']
            for task_id in range(task_x.size(0)):
                base_start = float(snapshot['baseline_start'][task_id])
                if cur_t > base_start + 1e-9 and snap_status[task_id] != 2:
                    task_x[task_id, 23] = float(cur_t - base_start) / takt
        
        data['task'].x = task_x
        
        snap_num_workers = len(snapshot['worker_free_time'])
        worker_x = torch.as_tensor(snapshot['base_worker_x']).clone()
        
        # [Feature Upgrade: Wait time rebuild]，使用对数归一化
        wait_times_w = np.maximum(0, snapshot['worker_free_time'] - snapshot['current_time'])
        worker_x[:, 11] = torch.log1p(torch.tensor(wait_times_w, dtype=torch.float) / ctx['mean_task_time'])
        
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, 12] = torch.tensor(is_free_bool, dtype=torch.float)
        
        worker_x[:, 13:21] = 0.0
        snap_locks = snapshot['worker_locks']
        lock_indices = torch.tensor(snap_locks, dtype=torch.long).clamp(max=7)
        worker_x[torch.arange(snap_num_workers), 13 + lock_indices] = 1.0
        
        # [Dynamic Events] 重建第 21 维工人疲劳因子 (Lazy 评估)
        snap_cum = snapshot.get('worker_cumulative_work', np.zeros(snap_num_workers))
        snap_last = snapshot.get('worker_last_busy_end', np.zeros(snap_num_workers))
        for w in range(snap_num_workers):
            cum_work = snap_cum[w]
            last_end = snap_last[w]
            if last_end > 0 and snapshot['current_time'] > last_end:
                idle_time = snapshot['current_time'] - last_end
                recovery_ratio = getattr(configs, 'fatigue_recovery_ratio', 0.5)
                cum_work = max(0.0, cum_work - idle_time * recovery_ratio)
            
            alpha = getattr(configs, 'fatigue_threshold_hours', 4.0)
            beta = getattr(configs, 'fatigue_decay_slope', 0.05)
            f_min = getattr(configs, 'fatigue_efficiency_floor', 0.60)
            overtime = max(0.0, cum_work - alpha)
            fatigue_f = max(f_min, 1.0 - beta * overtime / (alpha * 2))
            worker_x[w, 21] = fatigue_f
            
        data['worker'].x = worker_x
        if reusable_state is not None:
            if bool(getattr(configs, "use_skill_hub", False)):
                skill_x = build_skill_features(worker_x, int(configs.num_skill_types))
                if skill_x.size(1) != int(configs.skill_feat_dim):
                    raise ValueError(
                        f"skill_feat_dim 配置错误: {configs.skill_feat_dim}，"
                        f"实际需要 {skill_x.size(1)}"
                    )
                data["skill"].x = skill_x
        else:
            apply_resource_graph(
                data,
                task_x,
                worker_x,
                configs,
                skill_hub_topology=self._skill_hub_topology(
                    ctx["task_skill_edge_index"],
                    worker_x,
                    snapshot.get("worker_topology_key"),
                ),
            )
        
        station_x = ctx['base_station_x'].clone()
        station_x[:, 0] = torch.tensor(snapshot['station_loads'], dtype=torch.float) / max(1.0, ctx['ideal_station_load'])
        
        # 重建 Relative Load Competition 特征 (station_x[:, 5] 和 [:, 6])
        snap_loads = snapshot['station_loads']
        sum_loads = np.sum(snap_loads)
        max_load = np.max(snap_loads)
        station_x[:, 5] = torch.tensor(snap_loads / (sum_loads + 1e-6), dtype=torch.float)
        station_x[:, 6] = torch.tensor(snap_loads / (max_load + 1e-6), dtype=torch.float)
        
        # [Dynamic Events] 重建第 7 维槽位占用率与释放时间计算逻辑
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        snap_slots = snapshot.get('station_available_slots', np.full(self.num_stations, max_slots))
        station_x[:, 7] = torch.tensor(snap_slots, dtype=torch.float) / max_slots
        
        _fill_station_macro_features(station_x, snap_locks, is_free_bool)
        
        # 重建站位槽位释放时间特征 (station_x[s, 4])
        station_finish_lists = [[] for _ in range(self.num_stations)]
        for t_id, s_id, team, start_t, finish_t in snapshot['assigned_tasks']:
            if s_id != -1 and finish_t > snapshot['current_time']:
                station_finish_lists[s_id].append(finish_t)
                

        for s in range(self.num_stations):
            # 计算槽位释放时间并应用对数归一化
            lst = station_finish_lists[s]
            lst.sort()
            allowed_slots = snap_slots[s]
            if len(lst) >= allowed_slots:
                wait_time_s = max(0.0, lst[0] - snapshot['current_time'])
            else:
                wait_time_s = 0.0
            station_x[s, 4] = math.log1p(wait_time_s / ctx['mean_task_time'])
            
        data['station'].x = station_x
        
        ts_src, ts_dst, tw_src, tw_dst = [], [], [], []
        for t_id, s_id, team, _, _ in snapshot['assigned_tasks']:
            if s_id != -1:
                ts_src.append(t_id)
                ts_dst.append(s_id)
                for w_id in team:
                    tw_src.append(t_id)
                    tw_dst.append(w_id)
                    
        if ts_src:
            t_s_edge = torch.tensor([ts_src, ts_dst], dtype=torch.long)
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if tw_src:
            t_w_edge = torch.tensor([tw_src, tw_dst], dtype=torch.long)
        else:
            t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        data.apal_resource_topology_key = topology_key
        return data

    def get_current_fatigue_factor(self, worker_id: int, target_time: float) -> float:
        w = worker_id
        if not getattr(configs, 'enable_worker_fatigue', False):
            return 1.0
            
        cum_work = self.worker_cumulative_work[w]
        last_end = self.worker_last_busy_end[w]
        
        # 1. 动态恢复 (Lazy Evaluation)
        if last_end > 0 and target_time > last_end:
            idle_time = target_time - last_end
            recovery_ratio = getattr(configs, 'fatigue_recovery_ratio', 0.5)
            cum_work = max(0.0, cum_work - idle_time * recovery_ratio)
            
        # 2. 计算疲劳因子
        alpha = getattr(configs, 'fatigue_threshold_hours', 4.0)
        beta = getattr(configs, 'fatigue_decay_slope', 0.05)
        f_min = getattr(configs, 'fatigue_efficiency_floor', 0.60)
        
        overtime = max(0.0, cum_work - alpha)
        fatigue_factor = max(f_min, 1.0 - beta * overtime / (alpha * 2))
        return float(fatigue_factor)
        
    def _update_worker_fatigue(self, worker_id: int, start_time: float, finish_time: float):
        w = worker_id
        if not getattr(configs, 'enable_worker_fatigue', False):
            return
            
        # 1. 首先基于 start_time 更新并归档空闲恢复期间的累积工作量
        last_end = self.worker_last_busy_end[w]
        if last_end > 0 and start_time > last_end:
            idle_time = start_time - last_end
            recovery_ratio = getattr(configs, 'fatigue_recovery_ratio', 0.5)
            self.worker_cumulative_work[w] = max(0.0, self.worker_cumulative_work[w] - idle_time * recovery_ratio)
            
        # 2. 累加本次工作时长
        work_duration = finish_time - start_time
        self.worker_cumulative_work[w] += work_duration
        
        # 3. 更新忙碌结束时间锚点
        self.worker_last_busy_end[w] = finish_time
        
        # 4. 重新计算静态的 worker_fatigue_factor[w] 备用
        alpha = getattr(configs, 'fatigue_threshold_hours', 4.0)
        beta = getattr(configs, 'fatigue_decay_slope', 0.05)
        f_min = getattr(configs, 'fatigue_efficiency_floor', 0.60)
        
        overtime = max(0.0, self.worker_cumulative_work[w] - alpha)
        self.worker_fatigue_factor[w] = max(f_min, 1.0 - beta * overtime / (alpha * 2))

    def _try_inject_online_duration_perturb(self):
        if not getattr(configs, 'enable_online_duration_perturb', False):
            return
        prob = getattr(configs, 'online_perturb_prob_per_step', 0.02)
        if self.np_random.rand() < prob:
            # 随机选择 1~3 个尚未被调度的 Ready 或 Not Ready 任务进行扰动
            unassigned_tasks = np.where(self.task_status <= 1)[0]
            if len(unassigned_tasks) > 0:
                num_to_perturb = self.np_random.randint(1, min(4, len(unassigned_tasks) + 1))
                perturbed = self.np_random.choice(unassigned_tasks, num_to_perturb, replace=False)
                factor = self.np_random.uniform(0.8, 1.5)
                
                # 立即应用扰动
                for t in perturbed:
                    self.task_static_feat[t, 0] *= factor
                    # 刷新 base_task_x 特征以让 GNN 感知
                    self.base_task_x[t, 0] = self.task_static_feat[t, 0] / self.mean_task_time
                
                # 记录进事件队列
                self.event_queue.push(Event(self.current_time, EventType.DURATION_PERTURB, 
                                            {'task_ids': [int(t) for t in perturbed], 'perturb_factor': float(factor)}))
