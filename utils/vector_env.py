import copy
import multiprocessing as mp
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


def _worker(conn, make_env_fn: Callable[[int], Any], index: int):
    """
    运行在独立子进程中的环境步进循环。
    通过 Pipe 与主进程的 EnvProxy 通信，避免主进程 GIL 争用。
    """
    try:
        env = make_env_fn(index)
    except Exception as e:
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
        'dataset_pool': getattr(env, 'dataset_pool', None),
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
                conn.send(("OK", masks))
                
            elif cmd == 'get_rollout_state':
                masks = env.get_masks()
                snap = env.get_state_snapshot()
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (masks, snap, dynamic_info)))
                
            elif cmd == 'step_snapshot':
                if data is None:
                    snap = env.get_state_snapshot()
                    conn.send(("OK", (snap, 0.0, True, {})))
                else:
                    old_skip_obs = getattr(env, 'skip_obs_building', False)
                    try:
                        env.skip_obs_building = True
                        _, reward, done, info = env.step(data)
                    finally:
                        env.skip_obs_building = old_skip_obs
                    snap = env.get_state_snapshot()
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
                snap = env.get_state_snapshot()
                conn.send(("OK", snap))
                
            elif cmd == 'switch_dataset':
                env.switch_dataset(data)
                # 切图后，更新对应的所有静态/动态特征骨架
                dynamic_info = {
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'dataset_pool': getattr(env, 'dataset_pool', None),
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
                ctx = env.dataset_pool[idx]
                if 'base_data' not in ctx:
                    env._build_static_context(ctx)
                # 将序列化后的完整静态 ctx 缓存传回主进程
                conn.send(("OK", ctx))
                
            elif cmd == 'close':
                conn.send(("OK", None))
                conn.close()
                break
            else:
                conn.send(("ERROR", ValueError(f"Unknown command: {cmd}")))
        except Exception as e:
            conn.send(("ERROR", e))


class EnvProxy:
    """
    用于多进程 VectorEnv 的环境代理类。
    对主进程表现得与单环境一模一样，通过本地属性影子缓存实现超低延迟的同步属性查询。
    """
    def __init__(self, conn, idx: int):
        self._conn = conn
        self._idx = idx
        
        # 1. 静态属性缓存 (在初始化或切换数据集时写入)
        self.num_tasks: Optional[int] = None
        self.ideal_makespan: Optional[float] = None
        self.mean_task_time: Optional[float] = None
        self.dataset_pool: Optional[List[dict]] = None
        
        # 2. 动态属性缓存 (在 reset, step 和 try_wait_for_resources 之后同步更新)
        self.station_wall_clock: Optional[np.ndarray] = None
        self.assigned_tasks: List[Any] = []
        self.task_status: Optional[np.ndarray] = None
        self.current_time: float = 0.0

    def update_static_properties(self, info: dict):
        if 'num_tasks' in info: self.num_tasks = info['num_tasks']
        if 'ideal_makespan' in info: self.ideal_makespan = info['ideal_makespan']
        if 'mean_task_time' in info: self.mean_task_time = info['mean_task_time']
        if 'dataset_pool' in info: self.dataset_pool = info['dataset_pool']

    def update_dynamic_properties(self, info: dict):
        if 'station_wall_clock' in info: self.station_wall_clock = info['station_wall_clock']
        if 'assigned_tasks' in info: self.assigned_tasks = info['assigned_tasks']
        if 'task_status' in info: self.task_status = info['task_status']
        if 'current_time' in info: self.current_time = info['current_time']

    def reset(self, randomize_duration: bool = False, randomize_workers: bool = False, seed: Optional[int] = None):
        self._conn.send(('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers, 'seed': seed}))
        status, val = self._conn.recv()
        if status == "OK":
            obs, dynamic_info = val
            self.update_dynamic_properties(dynamic_info)
            self.update_static_properties(dynamic_info)
            return obs
        else:
            raise val

    def step(self, action: Any):
        self._conn.send(('step', action))
        status, val = self._conn.recv()
        if status == "OK":
            obs, reward, done, info = val
            if 'dynamic_info' in info:
                self.update_dynamic_properties(info['dynamic_info'])
                del info['dynamic_info']
            return obs, reward, done, info
        else:
            raise val

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._conn.send(('get_masks', None))
        status, val = self._conn.recv()
        if status == "OK":
            return val
        else:
            raise val

    def get_rollout_state(self):
        self._conn.send(('get_rollout_state', None))
        status, val = self._conn.recv()
        if status == "OK":
            masks, snap, dynamic_info = val
            self.update_dynamic_properties(dynamic_info)
            return masks, snap
        else:
            raise val

    def step_snapshot(self, action: Any):
        self._conn.send(('step_snapshot', action))
        status, val = self._conn.recv()
        if status == "OK":
            snap, reward, done, info = val
            if 'dynamic_info' in info:
                self.update_dynamic_properties(info['dynamic_info'])
                del info['dynamic_info']
            return snap, reward, done, info
        else:
            raise val

    def try_wait_for_resources(self) -> bool:
        self._conn.send(('try_wait_for_resources', None))
        status, val = self._conn.recv()
        if status == "OK":
            res, dynamic_info = val
            self.update_dynamic_properties(dynamic_info)
            return res
        else:
            raise val

    def get_state_snapshot(self) -> dict:
        self._conn.send(('get_state_snapshot', None))
        status, val = self._conn.recv()
        if status == "OK":
            return val
        else:
            raise val

    def switch_dataset(self, idx: int):
        self._conn.send(('switch_dataset', idx))
        status, val = self._conn.recv()
        if status == "OK":
            self.update_static_properties(val)
            self.update_dynamic_properties(val)
        else:
            raise val

    def rebuild_state_from_snapshot(self, snapshot: dict) -> HeteroData:
        """
        基于快照恢复成 PyG 图结构。
        核心设计：此方法完全通过本地影子缓存直接进行数学与张量计算，不需要向子进程发送 IPC 信号。
        消除了 PPO 训练更新阶段高频通信带来的带宽和延迟延迟。
        """
        from configs import configs
        from environment import _fill_station_macro_features
        
        ctx_idx = snapshot.get('dataset_idx', 0)
        ctx = self.dataset_pool[ctx_idx]
        
        # 防御机制：如果该数据集尚未在代理端加载 base_data，发送紧急 IPC 请求子进程就地初始化
        if 'base_data' not in ctx:
            self._conn.send(('initialize_dataset_context', ctx_idx))
            status, val = self._conn.recv()
            if status == "OK":
                self.dataset_pool[ctx_idx] = val
                ctx = self.dataset_pool[ctx_idx]
            else:
                raise val

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
        worker_x = snapshot['base_worker_x'].clone()
        
        wait_times_w = np.maximum(0, snapshot['worker_free_time'] - snapshot['current_time'])
        worker_x[:, 11] = torch.log1p(torch.tensor(wait_times_w, dtype=torch.float) / ctx['mean_task_time'])
        
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, 12] = torch.tensor(is_free_bool, dtype=torch.float)
        
        worker_x[:, 13:21] = 0.0
        snap_locks = snapshot['worker_locks']
        lock_indices = torch.tensor(snap_locks, dtype=torch.long).clamp(max=7)
        worker_x[torch.arange(snap_num_workers), 13 + lock_indices] = 1.0
        
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
        if 'can_do_edge_index' in snapshot:
            data['worker', 'can_do', 'task'].edge_index = snapshot['can_do_edge_index'].clone()
        else:
            full_ce = ctx['full_can_do_edge_index']
            mask = full_ce[0] < snap_num_workers
            data['worker', 'can_do', 'task'].edge_index = full_ce[:, mask].clone()
        
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
    ):
        self.num_envs = num_envs
        self.closed = False
        self.start_method = start_method
        self._mp_ctx = mp.get_context(start_method) if start_method else mp.get_context()
        
        # 创建双向管道
        self.parent_conns, self.child_conns = zip(*[self._mp_ctx.Pipe() for _ in range(num_envs)])
        
        # 启动守护子进程
        self.processes = []
        for i in range(num_envs):
            p = self._mp_ctx.Process(target=_worker, args=(self.child_conns[i], make_env_fn, i), daemon=True)
            self.processes.append(p)
            p.start()
            
        # 收集子进程初始化和静态配置的反馈
        self.envs = []
        for i in range(num_envs):
            status, info = self.parent_conns[i].recv()
            if status == "INIT_OK":
                proxy = EnvProxy(self.parent_conns[i], i)
                proxy.update_static_properties(info)
                self.envs.append(proxy)
            else:
                self.close()
                raise RuntimeError(f"Failed to initialize environment in worker {i}: {info}")

    def reset_all(self, randomize_duration: bool = False, randomize_workers: bool = False) -> List[Any]:
        """异步广播复位指令并同步收集子图状态"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers}))
        
        results = []
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                obs, dynamic_info = val
                self.envs[i].update_dynamic_properties(dynamic_info)
                self.envs[i].update_static_properties(dynamic_info)
                results.append(obs)
            else:
                raise val
        return results

    def step_all(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[dict]]:
        """异步发送物理步进指令并收集步进后的全部观测及奖励数据"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('step', actions[i]))
                
        results = []
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                obs, reward, done, info = val
                if 'dynamic_info' in info:
                    self.envs[i].update_dynamic_properties(info['dynamic_info'])
                    del info['dynamic_info']
                results.append((obs, reward, done, info))
            else:
                raise val
                
        next_states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return next_states, rewards, dones, infos

    def step_snapshot_all(self, actions: List[Any]) -> Tuple[List[dict], List[float], List[bool], List[dict]]:
        """异步步进并返回轻量 snapshot，主进程本地 rebuild 以消除 HeteroData 跨进程序列化开销"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('step_snapshot', actions[i]))
                
        results = []
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                snap, reward, done, info = val
                if 'dynamic_info' in info:
                    self.envs[i].update_dynamic_properties(info['dynamic_info'])
                    del info['dynamic_info']
                results.append((snap, reward, done, info))
            else:
                raise val
                
        snapshots = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return snapshots, rewards, dones, infos

    def get_masks_all(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """异步收集各进程环境的动作空间掩码"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('get_masks', None))
        
        results = []
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                results.append(val)
            else:
                raise val
        return results

    def get_masks_and_snapshots_all(self):
        """合并 IPC：一次命令返回 masks + snapshot + dynamic_info，替代 get_masks_all() + 逐环境 get_state_snapshot()"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('get_rollout_state', None))
        
        masks_list = []
        snapshots = []
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                masks, snap, dynamic_info = val
                self.envs[i].update_dynamic_properties(dynamic_info)
                masks_list.append(masks)
                snapshots.append(snap)
            else:
                raise val
        return masks_list, snapshots

    def try_wait_for_resources_all(self) -> List[bool]:
        """异步收集并推送时间推移操作"""
        results = [env.try_wait_for_resources() for env in self.envs]
        return results

    def get_state_snapshot_all(self) -> List[dict]:
        """获取所有环境当前的静态状态缓存"""
        return [env.get_state_snapshot() for env in self.envs]

    def switch_dataset_all(self, idx: int):
        """同步广播命令给所有并行子进程切换相同的数据集"""
        for i in range(self.num_envs):
            self.parent_conns[i].send(('switch_dataset', idx))
            
        for i in range(self.num_envs):
            status, val = self.parent_conns[i].recv()
            if status == "OK":
                self.envs[i].update_static_properties(val)
                self.envs[i].update_dynamic_properties(val)
            else:
                raise val

    def close(self):
        if self.closed:
            return
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        for conn in self.parent_conns:
            try:
                conn.close()
            except Exception:
                pass
        self.closed = True
