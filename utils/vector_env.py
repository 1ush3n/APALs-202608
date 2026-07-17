import copy
import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait
from typing import Callable, List, Tuple, Any, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData

class EnvCreator:
    """
    一个可序列化 (Picklable) 的环境创建器。
    用于规避 Windows 平台 spawn 模式下闭包 (Local Function) 无法被序列化传输至子进程的问题。
    """
    def __init__(self, data_path_or_dir: str, seed_offset: int = 42, config_overrides: Optional[dict] = None):
        self.data_path_or_dir = data_path_or_dir
        self.seed_offset = seed_offset
        if config_overrides is None:
            try:
                from configs import configs
                config_overrides = configs.to_flat_dict()
            except Exception:
                config_overrides = {}
        self.config_overrides = dict(config_overrides)

    def __call__(self, index: int):
        if self.config_overrides:
            from configs import configs
            configs.update_from_dict(self.config_overrides)
        from environment import AirLineEnv_Graph, _fill_station_macro_features
        return AirLineEnv_Graph(data_path_or_dir=self.data_path_or_dir, seed=self.seed_offset + index)


class VectorEnvWorkerError(RuntimeError):
    """VectorEnv 子进程启动、通信或执行失败。"""


def _snapshot_to_ipc(snapshot: dict) -> dict:
    """只转换 snapshot 中可能触发 Torch 共享内存传输的张量。"""
    result = dict(snapshot)
    base_worker_x = result.get("base_worker_x")
    if torch.is_tensor(base_worker_x):
        result["base_worker_x"] = base_worker_x.detach().cpu().numpy()
    return result


def _masks_to_ipc(masks: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, ...]:
    return tuple(mask.detach().cpu().numpy() for mask in masks)


def _worker(
    conn,
    make_env_fn: Callable[[int], Any],
    index: int,
    worker_threads: int,
):
    """
    运行在独立子进程中的环境步进循环。
    通过 Pipe 与主进程的 EnvProxy 通信，避免主进程 GIL 争用。
    """
    worker_threads = max(1, int(worker_threads))
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    torch.set_num_threads(worker_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, worker_threads)))
    except RuntimeError:
        pass

    try:
        env = make_env_fn(index)
    except Exception:
        import traceback
        tb_str = traceback.format_exc()
        conn.send(("INIT_ERROR", tb_str))
        conn.close()
        return

    # 1. 成功初始化后，打包发送初始静态属性，由主进程 Proxy 本地缓存
    init_info = {
        'num_tasks': getattr(env, 'num_tasks', None),
        'ideal_makespan': getattr(env, 'ideal_makespan', None),
        'mean_task_time': getattr(env, 'mean_task_time', None),
        'dataset_count': int(getattr(env, 'dataset_count', 1)),
        'active_dataset_idx': int(getattr(env, 'active_dataset_idx', 0)),
        'dataset_descriptor': env.get_dataset_descriptor(getattr(env, 'active_dataset_idx', 0)),
        'worker_audit': {
            'pid': os.getpid(),
            'torch_num_threads': torch.get_num_threads(),
            'omp_num_threads': os.environ.get("OMP_NUM_THREADS"),
        },
    }
    conn.send(("INIT_OK", init_info))

    # 2. 持续循环监听主进程的物理驱动指令
    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            break
        try:
            
            if cmd == 'step':
                if data is None:
                    # 停滞或死锁处理，不步进，返回轻量级切片 snapshot 作为 obs 兜底
                    obs = env.get_state_snapshot()
                    reward = 0.0
                    done = True
                    info = {}
                else:
                    obs, reward, done, info = env.step(data)
                
                # 随每一次步进返回更新后的动态属性
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                info['dynamic_info'] = dynamic_info
                conn.send(("OK", (obs, reward, done, info)))
                
            elif cmd == 'reset':
                obs = env.reset(**data)
                
                # 域随机化后，很多静态和动态属性会改变，需回传主进程同步刷新缓存
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (obs, dynamic_info)))
                
            elif cmd == 'get_masks':
                masks = env.get_masks()
                conn.send(("OK", _masks_to_ipc(masks)))
                
            elif cmd == 'get_rollout_state':
                masks = _masks_to_ipc(env.get_masks())
                snap = _snapshot_to_ipc(env.get_state_snapshot())
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (masks, snap, dynamic_info)))
                
            elif cmd == 'step_snapshot':
                if data is None:
                    snap = _snapshot_to_ipc(env.get_state_snapshot())
                    conn.send(("OK", (snap, 0.0, True, {})))
                else:
                    old_skip_obs = getattr(env, 'skip_obs_building', False)
                    try:
                        env.skip_obs_building = True
                        _, reward, done, info = env.step(data)
                    finally:
                        env.skip_obs_building = old_skip_obs
                    snap = _snapshot_to_ipc(env.get_state_snapshot())
                    dynamic_info = {
                        'station_wall_clock': getattr(env, 'station_wall_clock', None),
                        'assigned_tasks': getattr(env, 'assigned_tasks', None),
                        'task_status': getattr(env, 'task_status', None),
                        'current_time': getattr(env, 'current_time', 0.0),
                    }
                    info['dynamic_info'] = dynamic_info
                    conn.send(("OK", (snap, reward, done, info)))
                
            elif cmd == 'try_wait_for_resources':
                res = env.try_wait_for_resources()
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (res, dynamic_info)))
                
            elif cmd == 'get_state_snapshot':
                snap = _snapshot_to_ipc(env.get_state_snapshot())
                conn.send(("OK", snap))
                
            elif cmd == 'switch_dataset':
                env.switch_dataset(data)
                # 切图后，更新对应的所有静态/动态特征骨架
                dynamic_info = {
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'dataset_count': int(getattr(env, 'dataset_count', 1)),
                    'active_dataset_idx': int(getattr(env, 'active_dataset_idx', 0)),
                    'dataset_descriptor': env.get_dataset_descriptor(data),
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", dynamic_info))
                
            elif cmd == 'rebuild_state_from_snapshot':
                # 兜底接口，通常只由本地 Proxy 直接执行，此处仅作向下兼容
                res = env.rebuild_state_from_snapshot(data)
                conn.send(("OK", res))
                
            elif cmd == 'initialize_dataset_context':
                idx = data
                conn.send(("OK", env.export_dataset_context(idx)))
                
            elif cmd == 'close':
                conn.send(("OK", None))
                conn.close()
                break
            else:
                conn.send(("ERROR", ValueError(f"Unknown command: {cmd}")))
        except Exception:
            import traceback
            conn.send(("ERROR", traceback.format_exc()))


class EnvProxy:
    """
    用于多进程 VectorEnv 的环境代理类。
    对主进程表现得与单环境一模一样，通过本地属性影子缓存实现超低延迟的同步属性查询。
    """
    def __init__(self, conn, process: mp.Process, idx: int, command_timeout_sec: float):
        self._conn = conn
        self._process = process
        self._idx = idx
        self._command_timeout_sec = float(command_timeout_sec)
        
        # 1. 静态属性缓存 (在初始化或切换数据集时写入)
        self.num_tasks: Optional[int] = None
        self.ideal_makespan: Optional[float] = None
        self.mean_task_time: Optional[float] = None
        self.dataset_count: int = 0
        self.active_dataset_idx: int = 0
        self.dataset_pool: List[Optional[dict]] = []
        
        # 2. 动态属性缓存 (在 reset, step 和 try_wait_for_resources 之后同步更新)
        self.station_wall_clock: Optional[np.ndarray] = None
        self.assigned_tasks: List[Any] = []
        self.task_status: Optional[np.ndarray] = None
        self.current_time: float = 0.0
        self._worker_skill_topology_cache: dict[tuple[int, bytes], torch.Tensor] = {}

    def update_static_properties(self, info: dict):
        if 'num_tasks' in info: self.num_tasks = info['num_tasks']
        if 'ideal_makespan' in info: self.ideal_makespan = info['ideal_makespan']
        if 'mean_task_time' in info: self.mean_task_time = info['mean_task_time']
        if 'dataset_count' in info:
            self.dataset_count = int(info['dataset_count'])
            while len(self.dataset_pool) < self.dataset_count:
                self.dataset_pool.append(None)
        if 'active_dataset_idx' in info:
            self.active_dataset_idx = int(info['active_dataset_idx'])
        descriptor = info.get('dataset_descriptor')
        if descriptor is not None:
            idx = int(descriptor['dataset_idx'])
            while len(self.dataset_pool) <= idx:
                self.dataset_pool.append(None)
            cached = self.dataset_pool[idx] or {}
            cached.update(descriptor)
            self.dataset_pool[idx] = cached

    def _recv(self, operation: str):
        if not self._conn.poll(self._command_timeout_sec):
            exit_code = self._process.exitcode
            state = "alive" if self._process.is_alive() else f"exitcode={exit_code}"
            raise TimeoutError(
                f"VectorEnv worker {self._idx} 执行 {operation} 超时 "
                f"({self._command_timeout_sec:.1f}s, {state})"
            )
        status, value = self._conn.recv()
        if status not in {"OK", "INIT_OK"}:
            raise VectorEnvWorkerError(
                f"VectorEnv worker {self._idx} 执行 {operation} 失败:\n{value}"
            )
        return value

    def update_dynamic_properties(self, info: dict):
        if 'station_wall_clock' in info: self.station_wall_clock = info['station_wall_clock']
        if 'assigned_tasks' in info: self.assigned_tasks = info['assigned_tasks']
        if 'task_status' in info: self.task_status = info['task_status']
        if 'current_time' in info: self.current_time = info['current_time']

    def reset(self, randomize_duration: bool = False, randomize_workers: bool = False, seed: Optional[int] = None):
        self._conn.send(('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers, 'seed': seed}))
        obs, dynamic_info = self._recv("reset")
        self.update_dynamic_properties(dynamic_info)
        self.update_static_properties(dynamic_info)
        return obs

    def step(self, action: Any):
        self._conn.send(('step', action))
        obs, reward, done, info = self._recv("step")
        if 'dynamic_info' in info:
            self.update_dynamic_properties(info.pop('dynamic_info'))
        return obs, reward, done, info

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._conn.send(('get_masks', None))
        return tuple(torch.from_numpy(mask) for mask in self._recv("get_masks"))

    def get_rollout_state(self):
        self._conn.send(('get_rollout_state', None))
        masks, snap, dynamic_info = self._recv("get_rollout_state")
        self.update_dynamic_properties(dynamic_info)
        return tuple(torch.from_numpy(mask) for mask in masks), snap

    def step_snapshot(self, action: Any):
        self._conn.send(('step_snapshot', action))
        snap, reward, done, info = self._recv("step_snapshot")
        if 'dynamic_info' in info:
            self.update_dynamic_properties(info.pop('dynamic_info'))
        return snap, reward, done, info

    def try_wait_for_resources(self) -> bool:
        self._conn.send(('try_wait_for_resources', None))
        res, dynamic_info = self._recv("try_wait_for_resources")
        self.update_dynamic_properties(dynamic_info)
        return res

    def get_state_snapshot(self) -> dict:
        self._conn.send(('get_state_snapshot', None))
        return self._recv("get_state_snapshot")

    def switch_dataset(self, idx: int):
        self._conn.send(('switch_dataset', idx))
        val = self._recv("switch_dataset")
        self.update_static_properties(val)
        self.update_dynamic_properties(val)

    def rebuild_state_from_snapshot(self, snapshot: dict) -> HeteroData:
        """
        基于快照恢复成 PyG 图结构。
        核心设计：此方法完全通过本地影子缓存直接进行数学与张量计算，不需要向子进程发送 IPC 信号。
        消除了 PPO 训练更新阶段高频通信带来的带宽和延迟延迟。
        """
        from configs import configs
        from environment import _fill_station_macro_features
        from worker_feature_layout import resolve_worker_feature_layout
        from utils.resource_graph import (
            SkillHubTopology,
            apply_resource_graph,
            build_worker_skill_edges,
            worker_topology_key,
        )
        
        ctx_idx = snapshot.get('dataset_idx', 0)
        while len(self.dataset_pool) <= ctx_idx:
            self.dataset_pool.append(None)
        ctx = self.dataset_pool[ctx_idx]
        
        # 防御机制：如果该数据集尚未在代理端加载 base_data，发送紧急 IPC 请求子进程就地初始化
        if ctx is None or 'base_data' not in ctx:
            self._conn.send(('initialize_dataset_context', ctx_idx))
            self.dataset_pool[ctx_idx] = self._recv("initialize_dataset_context")
            ctx = self.dataset_pool[ctx_idx]

        data = ctx['base_data'].clone()
        
        # 1. 重建任务节点特征
        task_x = ctx['base_task_x'].clone()
        task_x[:, 1:5] = 0.0
        task_x[torch.arange(ctx['num_tasks']), snapshot['task_status'] + 1] = 1.0
        
        snap_mat = snapshot.get('task_material_ready', np.zeros(ctx['num_tasks']))
        wait_times_t = np.maximum(0, snap_mat - snapshot['current_time'])
        task_x[:, 17] = torch.log1p(torch.tensor(wait_times_t, dtype=torch.float) / ctx['mean_task_time'])
        if task_x.size(1) >= 24 and 'baseline_start' in snapshot:
            snap_num_workers_for_task = len(snapshot['worker_free_time'])
            takt = max(1e-6, float(snapshot.get('baseline_makespan', 1.0)))
            task_x[:, 18] = torch.tensor((snapshot['baseline_start'] - snapshot['current_time']) / takt, dtype=torch.float)
            task_x[:, 19] = torch.tensor((snapshot['baseline_station'] + 1) / max(1, self.dataset_pool[ctx_idx]['base_station_x'].shape[0]), dtype=torch.float)
            task_x[:, 20] = torch.tensor(snapshot['baseline_team_size'] / max(1, snap_num_workers_for_task), dtype=torch.float)
            task_x[:, 21] = torch.tensor(snapshot['baseline_frozen'], dtype=torch.float)
            task_x[:, 22] = torch.tensor((snap_mat > snapshot.get('reschedule_start_time', 0.0) + 1e-9).astype(float), dtype=torch.float)
            cur_t = float(snapshot['current_time'])
            snap_status = snapshot['task_status']
            for task_id in range(task_x.size(0)):
                base_start = float(snapshot['baseline_start'][task_id])
                if cur_t > base_start + 1e-9 and snap_status[task_id] != 2:
                    task_x[task_id, 23] = float(cur_t - base_start) / takt
        
        data['task'].x = task_x
        
        # 2. 重建工人节点特征
        snap_num_workers = len(snapshot['worker_free_time'])
        worker_x = torch.as_tensor(snapshot['base_worker_x']).clone()
        worker_layout = resolve_worker_feature_layout(configs)
        assert worker_x.size(1) == worker_layout.total_dim, (
            f"快照工人特征维度错误: {worker_x.size(1)} != {worker_layout.total_dim}"
        )
        
        wait_times_w = np.maximum(0, snapshot['worker_free_time'] - snapshot['current_time'])
        worker_x[:, worker_layout.wait_idx] = torch.log1p(
            torch.tensor(wait_times_w, dtype=torch.float) / ctx['mean_task_time']
        )
        
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, worker_layout.free_idx] = torch.tensor(is_free_bool, dtype=torch.float)
        
        worker_x[:, worker_layout.lock_slice] = 0.0
        snap_locks = snapshot['worker_locks']
        lock_indices = torch.tensor(snap_locks, dtype=torch.long).clamp(max=7)
        worker_x[torch.arange(snap_num_workers), worker_layout.lock_start + lock_indices] = 1.0
        
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
            worker_x[w, worker_layout.fatigue_idx] = fatigue_f
            
        data['worker'].x = worker_x
        topology = None
        if bool(getattr(configs, "use_skill_hub", False)):
            topology_key = snapshot.get("worker_topology_key") or worker_topology_key(
                worker_x,
                int(configs.num_skill_types),
            )
            worker_edges = self._worker_skill_topology_cache.get(topology_key)
            if worker_edges is None:
                worker_edges = build_worker_skill_edges(
                    worker_x,
                    int(configs.num_skill_types),
                )
                self._worker_skill_topology_cache[topology_key] = worker_edges
            topology = SkillHubTopology(
                worker_to_skill=worker_edges,
                skill_to_task=ctx["task_skill_edge_index"],
            )
        apply_resource_graph(
            data,
            task_x,
            worker_x,
            configs,
            skill_hub_topology=topology,
        )
        
        # 3. 重建站位特征
        num_stations = len(snapshot['station_loads'])
        station_x = ctx['base_station_x'].clone()
        station_x[:, 0] = torch.tensor(snapshot['station_loads'], dtype=torch.float) / max(1.0, ctx['ideal_station_load'])
        
        # 重建 Relative Load Competition 特征 (station_x[:, 5] 和 [:, 6])
        snap_loads = snapshot['station_loads']
        sum_loads = np.sum(snap_loads)
        max_load = np.max(snap_loads)
        station_x[:, 5] = torch.tensor(snap_loads / (sum_loads + 1e-6), dtype=torch.float)
        station_x[:, 6] = torch.tensor(snap_loads / (max_load + 1e-6), dtype=torch.float)
        
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        snap_slots = snapshot.get('station_available_slots', np.full(num_stations, max_slots))
        station_x[:, 7] = torch.tensor(snap_slots, dtype=torch.float) / max_slots
        
        _fill_station_macro_features(station_x, snap_locks, is_free_bool)
        
        # 重建站位槽位释放时间特征 (station_x[s, 4])
        station_finish_lists = [[] for _ in range(num_stations)]
        for t_id, s_id, team, start_t, finish_t in snapshot['assigned_tasks']:
            if s_id != -1 and finish_t > snapshot['current_time']:
                station_finish_lists[s_id].append(finish_t)
                
        import math
        for s in range(num_stations):
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
        
        # 4. 重建指派关系边
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
        
        return data


class VectorEnv:
    """
    多进程向量化环境封装器 (Subprocess Vectorized Environment Wrapper)。
    利用多进程双向管道实现 CPU 并行步进，以最大化 Ryzen 9 7945HX 多核 CPU 的并行优势。
    """
    def __init__(
        self,
        make_env_fn: Callable[[int], Any],
        num_envs: int = 4,
        max_workers: Any = None,
        start_method: Optional[str] = None,
        worker_threads: Any = "auto",
        init_timeout_sec: float = 120.0,
        command_timeout_sec: float = 120.0,
    ):
        self.num_envs = num_envs
        self.closed = False
        self.start_method = start_method
        self.init_timeout_sec = float(init_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self._mp_ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self.worker_threads = self._resolve_worker_threads(worker_threads, num_envs)
        self.worker_audits: List[dict] = []
        
        # 创建双向管道
        self.parent_conns, self.child_conns = zip(*[self._mp_ctx.Pipe() for _ in range(num_envs)])
        
        # 启动守护子进程
        self.processes = []
        for i in range(num_envs):
            p = self._mp_ctx.Process(
                target=_worker,
                args=(self.child_conns[i], make_env_fn, i, self.worker_threads),
                daemon=True,
            )
            self.processes.append(p)
            p.start()
            
        # 收集子进程初始化和静态配置的反馈
        self.envs = []
        try:
            for i in range(num_envs):
                conn = self.parent_conns[i]
                process = self.processes[i]
                if not conn.poll(self.init_timeout_sec):
                    state = "alive" if process.is_alive() else f"exitcode={process.exitcode}"
                    raise TimeoutError(
                        f"VectorEnv worker {i} 初始化超时 "
                        f"({self.init_timeout_sec:.1f}s, {state})"
                    )
                status, info = conn.recv()
                if status != "INIT_OK":
                    raise VectorEnvWorkerError(f"VectorEnv worker {i} 初始化失败:\n{info}")
                proxy = EnvProxy(conn, process, i, self.command_timeout_sec)
                proxy.update_static_properties(info)
                self.envs.append(proxy)
                self.worker_audits.append(dict(info.get("worker_audit", {})))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve_worker_threads(worker_threads: Any, num_envs: int) -> int:
        if worker_threads is None or str(worker_threads).lower() == "auto":
            return max(1, (os.cpu_count() or 1) // max(1, int(num_envs)))
        try:
            return max(1, int(worker_threads))
        except (TypeError, ValueError):
            return max(1, (os.cpu_count() or 1) // max(1, int(num_envs)))

    def _recv_worker(self, index: int, operation: str):
        conn = self.parent_conns[index]
        process = self.processes[index]
        if not conn.poll(self.command_timeout_sec):
            state = "alive" if process.is_alive() else f"exitcode={process.exitcode}"
            raise TimeoutError(
                f"VectorEnv worker {index} 执行 {operation} 超时 "
                f"({self.command_timeout_sec:.1f}s, {state})"
            )
        status, value = conn.recv()
        if status != "OK":
            raise VectorEnvWorkerError(
                f"VectorEnv worker {index} 执行 {operation} 失败:\n{value}"
            )
        return value

    def _recv_workers_unordered(self, indices: List[int], operation: str) -> dict[int, Any]:
        """按 worker 就绪顺序收集结果，避免固定顺序 recv 放大慢 worker 的等待时间。"""
        pending = {int(index): self.parent_conns[int(index)] for index in indices}
        conn_to_index = {conn: index for index, conn in pending.items()}
        results: dict[int, Any] = {}
        deadline = time.monotonic() + self.command_timeout_sec

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                details = []
                for index in sorted(pending):
                    process = self.processes[index]
                    state = "alive" if process.is_alive() else f"exitcode={process.exitcode}"
                    details.append(f"{index}:{state}")
                raise TimeoutError(
                    f"VectorEnv workers 执行 {operation} 超时 "
                    f"({self.command_timeout_sec:.1f}s, pending={','.join(details)})"
                )

            ready = wait(list(pending.values()), timeout=remaining)
            if not ready:
                continue

            for conn in ready:
                index = conn_to_index[conn]
                status, value = conn.recv()
                pending.pop(index, None)
                if status != "OK":
                    raise VectorEnvWorkerError(
                        f"VectorEnv worker {index} 执行 {operation} 失败:\n{value}"
                    )
                results[index] = value

        return results

    def reset_all(self, randomize_duration: bool = False, randomize_workers: bool = False) -> List[Any]:
        """异步广播复位指令并同步收集子图状态"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers}))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "reset")
        results = []
        for i in range(self.num_envs):
            obs, dynamic_info = worker_results[i]
            self.envs[i].update_dynamic_properties(dynamic_info)
            self.envs[i].update_static_properties(dynamic_info)
            results.append(obs)
        return results

    def reset_indices(
        self,
        requests: dict[int, dict[str, Any]],
    ) -> dict[int, Any]:
        """按索引复位环境；每个环境可使用独立 seed。"""
        target_indices = sorted(int(index) for index in requests)
        if not target_indices:
            return {}
        for index in target_indices:
            self.parent_conns[index].send(("reset", dict(requests[index])))
        worker_results = self._recv_workers_unordered(target_indices, "reset")
        results: dict[int, Any] = {}
        for index in target_indices:
            obs, dynamic_info = worker_results[index]
            self.envs[index].update_dynamic_properties(dynamic_info)
            self.envs[index].update_static_properties(dynamic_info)
            results[index] = obs
        return results

    def step_all(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[dict]]:
        """异步发送物理步进指令并收集步进后的全部观测及奖励数据"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('step', actions[i]))
                
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "step")
        results = []
        for i in range(self.num_envs):
            obs, reward, done, info = worker_results[i]
            if 'dynamic_info' in info:
                self.envs[i].update_dynamic_properties(info.pop('dynamic_info'))
            results.append((obs, reward, done, info))
                
        next_states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return next_states, rewards, dones, infos

    def step_snapshot_all(self, actions: List[Any]) -> Tuple[List[dict], List[float], List[bool], List[dict]]:
        """异步步进并返回轻量 snapshot，主进程本地 rebuild 以消除 HeteroData 跨进程序列化开销"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('step_snapshot', actions[i]))
                
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "step_snapshot")
        results = []
        for i in range(self.num_envs):
            snap, reward, done, info = worker_results[i]
            if 'dynamic_info' in info:
                self.envs[i].update_dynamic_properties(info.pop('dynamic_info'))
            results.append((snap, reward, done, info))
                
        snapshots = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return snapshots, rewards, dones, infos

    def step_snapshot_indices(
        self,
        actions: dict[int, Any],
    ) -> dict[int, tuple[dict, float, bool, dict]]:
        """只步进指定环境，返回值按环境索引组织。"""
        target_indices = sorted(int(index) for index in actions)
        if not target_indices:
            return {}
        for index in target_indices:
            self.parent_conns[index].send(("step_snapshot", actions[index]))
        worker_results = self._recv_workers_unordered(target_indices, "step_snapshot")
        results: dict[int, tuple[dict, float, bool, dict]] = {}
        for index in target_indices:
            snapshot, reward, done, info = worker_results[index]
            if "dynamic_info" in info:
                self.envs[index].update_dynamic_properties(info.pop("dynamic_info"))
            results[index] = (snapshot, float(reward), bool(done), info)
        return results

    def get_masks_all(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """异步收集各进程环境的动作空间掩码"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('get_masks', None))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "get_masks")
        results = []
        for i in range(self.num_envs):
            masks = worker_results[i]
            results.append(tuple(torch.from_numpy(mask) for mask in masks))
        return results

    def get_masks_and_snapshots_all(self):
        """合并 IPC：一次命令返回 masks + snapshot + dynamic_info，替代 get_masks_all() + 逐环境 get_state_snapshot()"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('get_rollout_state', None))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "get_rollout_state")
        masks_list = []
        snapshots = []
        for i in range(self.num_envs):
            masks, snap, dynamic_info = worker_results[i]
            self.envs[i].update_dynamic_properties(dynamic_info)
            masks_list.append(tuple(torch.from_numpy(mask) for mask in masks))
            snapshots.append(snap)
        return masks_list, snapshots

    def get_rollout_state_indices(self, indices: List[int]) -> dict[int, tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]]:
        """批量获取指定环境的 masks 与 snapshot。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        for index in target_indices:
            self.parent_conns[index].send(('get_rollout_state', None))

        worker_results = self._recv_workers_unordered(target_indices, "get_rollout_state")
        results = {}
        for index in target_indices:
            masks, snap, dynamic_info = worker_results[index]
            self.envs[index].update_dynamic_properties(dynamic_info)
            results[index] = (tuple(torch.from_numpy(mask) for mask in masks), snap)
        return results

    def try_wait_for_resources_all(self) -> List[bool]:
        """异步收集并推送时间推移操作"""
        indexed = self.try_wait_for_resources_indices(list(range(self.num_envs)))
        return [indexed[i] for i in range(self.num_envs)]

    def try_wait_for_resources_indices(self, indices: List[int]) -> dict[int, bool]:
        """批量推进指定环境的资源等待。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        for index in target_indices:
            self.parent_conns[index].send(('try_wait_for_resources', None))

        worker_results = self._recv_workers_unordered(target_indices, "try_wait_for_resources")
        results = {}
        for index in target_indices:
            res, dynamic_info = worker_results[index]
            self.envs[index].update_dynamic_properties(dynamic_info)
            results[index] = bool(res)
        return results

    def get_state_snapshot_all(self) -> List[dict]:
        """获取所有环境当前的静态状态缓存"""
        return [env.get_state_snapshot() for env in self.envs]

    def switch_dataset_all(self, idx: int):
        """同步广播命令给所有并行子进程切换相同的数据集"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('switch_dataset', idx))
            
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "switch_dataset")
        for i in range(self.num_envs):
            val = worker_results[i]
            self.envs[i].update_static_properties(val)
            self.envs[i].update_dynamic_properties(val)

    def switch_dataset_indices(self, dataset_indices: dict[int, int]) -> None:
        """按环境索引切换不同数据集，支持多规模并行 rollout。"""
        target_indices = sorted(int(index) for index in dataset_indices)
        if not target_indices:
            return
        for index in target_indices:
            self.parent_conns[index].send(
                ("switch_dataset", int(dataset_indices[index]))
            )
        worker_results = self._recv_workers_unordered(
            target_indices,
            "switch_dataset",
        )
        for index in target_indices:
            value = worker_results[index]
            self.envs[index].update_static_properties(value)
            self.envs[index].update_dynamic_properties(value)

    def close(self):
        if self.closed:
            return
        for conn, process in zip(
            getattr(self, "parent_conns", ()),
            getattr(self, "processes", ()),
        ):
            if process.is_alive():
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        deadline = time.monotonic() + 2.0
        for p in getattr(self, "processes", ()):
            if p.is_alive():
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        for conn in getattr(self, "parent_conns", ()):
            try:
                conn.close()
            except Exception:
                pass
        self.closed = True
