import torch
from torch_geometric.data import Batch
import numpy as np
from typing import Any
from configs import configs
from worker_feature_layout import resolve_worker_feature_layout
from utils.resource_graph import (
    apply_batched_resource_graph,
    build_worker_skill_edges,
    worker_topology_key,
)

class GPUBatchGraphManager:
    """
    预分配 GPU 批量异构图模板管理器。
    用于在 PPO 更新时，直接在 GPU 显存端批量覆写节点特征和动态边关系，
    从而彻底规避 CPU 端的 PyG DataLoader 拼接瓶颈和 CPU 图重建开销。
    """
    def __init__(self, device: torch.device, config=None):
        self.device = device
        self.config = config if config is not None else configs
        # 缓存 (dataset_idx, batch_size) 对应的 Batch 模板，防重复分配显存
        self.templates = {}
        self.worker_skill_topologies = {}

    def retain_dataset(self, dataset_idx: int) -> int:
        """仅保留当前数据集的模板，避免窄区间轮换时显存逐轮累积。"""
        stale_keys = [key for key in self.templates if key[0] != dataset_idx]
        for key in stale_keys:
            del self.templates[key]
        stale_topologies = [
            key for key in self.worker_skill_topologies if key[0] != dataset_idx
        ]
        for key in stale_topologies:
            del self.worker_skill_topologies[key]
        return len(stale_keys)

    def clear(self) -> int:
        """释放全部 GPU Batch 模板引用并返回清理数量。"""
        template_count = len(self.templates)
        self.templates.clear()
        self.worker_skill_topologies.clear()
        return template_count
        
    def get_batch_template(self, env: Any, batch_size: int, dataset_idx: int):
        if batch_size <= 0:
            raise ValueError("GPU Batch 模板的 batch_size 必须大于 0")
        key = (
            dataset_idx,
            batch_size,
            bool(getattr(self.config, "use_skill_hub", False)),
            bool(getattr(self.config, "skill_hub_bidirectional", False)),
        )
        if key not in self.templates:
            ctx = env.dataset_pool[dataset_idx]
            base_data = ctx['base_data']
            
            # 预先生成 batch_size 个静态大图，拼接成大 Batch 并常驻 GPU 显存
            data_list = [base_data.clone() for _ in range(batch_size)]
            batch_temp = Batch.from_data_list(data_list).to(self.device)
            self.templates[key] = batch_temp
        return self.templates[key]

    def _ensure_batch_vectors(
        self,
        batch: Batch,
        *,
        batch_size: int,
        num_tasks: int,
        num_workers: int,
        num_stations: int,
    ) -> Batch:
        """确保 GPU 原地重建后的异构 Batch 带有 PyG 节点 batch 向量。"""
        specs = {
            'task': num_tasks,
            'worker': num_workers,
            'station': num_stations,
        }
        if bool(getattr(self.config, "use_skill_hub", False)):
            specs['skill'] = int(getattr(self.config, "num_skill_types", 5))
        for node_type, nodes_per_graph in specs.items():
            storage = batch[node_type]
            expected_nodes = batch_size * nodes_per_graph
            actual_nodes = int(storage.x.size(0))
            if actual_nodes != expected_nodes:
                raise ValueError(
                    f"{node_type}.x 节点数与 batch 推断不一致: "
                    f"{actual_nodes} != {batch_size} * {nodes_per_graph}"
                )
            if not hasattr(storage, 'batch') or storage.batch is None or storage.batch.numel() != actual_nodes:
                storage.batch = torch.arange(batch_size, device=self.device, dtype=torch.long).repeat_interleave(nodes_per_graph)
        return batch

    def batched_rebuild_on_gpu(self, snapshots: list, env: Any):
        """
        核心全张量化 GPU 原地特征覆写与动态边装配算法。
        """
        if not snapshots:
            raise ValueError("GPU Batch 重建收到空 snapshots")
        required_keys = (
            'worker_free_time',
            'base_task_x',
            'base_worker_x',
            'current_time',
            'task_status',
            'worker_locks',
            'station_loads',
            'assigned_tasks',
        )
        for idx, snap in enumerate(snapshots):
            missing = [key for key in required_keys if key not in snap]
            if missing:
                raise KeyError(f"snapshot[{idx}] 缺少 GPU Batch 重建必需字段: {missing}")

        batch_size = len(snapshots)
        dataset_idx = snapshots[0].get('dataset_idx', 0)
        self.retain_dataset(dataset_idx)
        ctx = env.dataset_pool[dataset_idx]
        
        num_tasks = ctx['num_tasks']
        num_workers = len(snapshots[0]['worker_free_time'])
        num_stations = ctx['base_station_x'].shape[0]
        mean_task_time = ctx['mean_task_time']
        ideal_station_load = ctx['ideal_station_load']
        
        # 1. 取得 GPU 预分配的大 Batch 图模板
        batch_template = self.get_batch_template(env, batch_size, dataset_idx)
        t_device = self.device
        
        # 2. 覆盖静态特征以防污染
        # ① Task 静态特征
        base_task_x_batch = torch.cat(
            [torch.as_tensor(snap['base_task_x']) for snap in snapshots],
            dim=0,
        ).to(t_device)
        expected_task_shape = (batch_size * num_tasks, ctx['base_task_x'].size(1))
        if tuple(base_task_x_batch.shape) != expected_task_shape:
            raise ValueError(
                "GPU Batch 快照任务特征形状不一致: "
                f"actual={tuple(base_task_x_batch.shape)}, expected={expected_task_shape}"
            )
        batch_template['task'].x = base_task_x_batch
        # ② Worker 静态特征 (需根据每个 snapshot 实际被选中的工人 base_worker_x 拼接)
        base_worker_x_batch = torch.cat(
            [torch.as_tensor(snap['base_worker_x']) for snap in snapshots],
            dim=0,
        ).to(t_device)
        batch_template['worker'].x = base_worker_x_batch
        # ③ Station 静态特征
        batch_template['station'].x = ctx['base_station_x'].to(t_device).repeat(batch_size, 1)
        
        # 3. 收集 batch 中所有环境的状态数据
        statuses = []
        mat_readies = []
        current_times = []
        baseline_starts = []
        baseline_stations = []
        baseline_team_sizes = []
        baseline_frozen = []
        baseline_makespans = []
        reschedule_start_times = []
        
        worker_free_times = []
        worker_locks = []
        worker_cum_work = []
        worker_last_end = []
        
        station_loads = []
        station_slots = []
        
        # 收集工位等待时间矩阵 (batch_size, num_stations)
        station_wait_times = np.zeros((batch_size, num_stations), dtype=np.float32)
        
        # 收集动态边的连接索引
        ts_src, ts_dst = [], []
        tw_src, tw_dst = [], []
        
        max_slots = getattr(self.config, 'max_slots_per_station', 3)
        
        for b_i, snap in enumerate(snapshots):
            task_offset = b_i * num_tasks
            station_offset = b_i * num_stations
            worker_offset = b_i * num_workers
            
            cur_t = snap['current_time']
            current_times.append(cur_t)
            
            statuses.append(snap['task_status'])
            mat_readies.append(snap.get('task_material_ready', np.zeros(num_tasks)))
            if 'baseline_start' in snap:
                baseline_starts.append(snap['baseline_start'])
                baseline_stations.append(snap['baseline_station'])
                baseline_team_sizes.append(snap['baseline_team_size'])
                baseline_frozen.append(snap['baseline_frozen'])
                baseline_makespans.append(float(snap.get('baseline_makespan', 1.0)))
                reschedule_start_times.append(float(snap.get('reschedule_start_time', 0.0)))
            
            worker_free_times.append(snap['worker_free_time'])
            worker_locks.append(snap['worker_locks'])
            worker_cum_work.append(snap.get('worker_cumulative_work', np.zeros(num_workers)))
            worker_last_end.append(snap.get('worker_last_busy_end', np.zeros(num_workers)))
            
            station_loads.append(snap['station_loads'])
            allowed_slots = snap.get('station_available_slots', np.full(num_stations, max_slots))
            station_slots.append(allowed_slots)
            
            # can_do 边偏移与收集
            
            # 解析已指派的活动工序列表以计算槽位释放等待时间
            station_finish_lists = [[] for _ in range(num_stations)]
            for t_id, s_id, team, start_t, finish_t in snap['assigned_tasks']:
                if s_id != -1:
                    ts_src.append(t_id + task_offset)
                    ts_dst.append(s_id + station_offset)
                    for w_id in team:
                        tw_src.append(t_id + task_offset)
                        tw_dst.append(w_id + worker_offset)
                    
                    if finish_t > cur_t:
                        station_finish_lists[s_id].append(finish_t)
                        
            for s in range(num_stations):
                lst = station_finish_lists[s]
                lst.sort()
                lim = allowed_slots[s]
                if len(lst) >= lim:
                    station_wait_times[b_i, s] = max(0.0, lst[0] - cur_t)
                    
        # 4. 大 Tensor 打包与单次 H2D (Host-to-Device) 传输
        flat_status = torch.tensor(np.concatenate(statuses), dtype=torch.long, device=t_device)
        flat_mat_ready = torch.tensor(np.concatenate(mat_readies), dtype=torch.float32, device=t_device)
        flat_current_times_t = torch.tensor(current_times, dtype=torch.float32, device=t_device).repeat_interleave(num_tasks)
        
        flat_free_time = torch.tensor(np.concatenate(worker_free_times), dtype=torch.float32, device=t_device)
        flat_current_times_w = torch.tensor(current_times, dtype=torch.float32, device=t_device).repeat_interleave(num_workers)
        flat_locks = torch.tensor(np.concatenate(worker_locks), dtype=torch.long, device=t_device)
        flat_cum = torch.tensor(np.concatenate(worker_cum_work), dtype=torch.float32, device=t_device)
        flat_last = torch.tensor(np.concatenate(worker_last_end), dtype=torch.float32, device=t_device)
        
        flat_loads = torch.tensor(np.concatenate(station_loads), dtype=torch.float32, device=t_device)
        flat_slots = torch.tensor(np.concatenate(station_slots), dtype=torch.float32, device=t_device)
        flat_station_wait = torch.tensor(station_wait_times, dtype=torch.float32, device=t_device).view(-1)
        
        # 5. 原地特征快速覆写 (In-place Broadcasting)
        # ① Task 特征
        batch_template['task'].x[:, 1:5] = 0.0
        batch_template['task'].x[torch.arange(batch_size * num_tasks, device=t_device), flat_status + 1] = 1.0
        wait_t = torch.clamp(flat_mat_ready - flat_current_times_t, min=0.0)
        batch_template['task'].x[:, 17] = torch.log1p(wait_t / mean_task_time)
        if batch_template['task'].x.size(1) >= 24 and len(baseline_starts) == batch_size:
            flat_base_start = torch.tensor(np.concatenate(baseline_starts), dtype=torch.float32, device=t_device)
            flat_base_station = torch.tensor(np.concatenate(baseline_stations), dtype=torch.float32, device=t_device)
            flat_base_team = torch.tensor(np.concatenate(baseline_team_sizes), dtype=torch.float32, device=t_device)
            flat_base_frozen = torch.tensor(np.concatenate(baseline_frozen), dtype=torch.float32, device=t_device)
            flat_base_makespan = torch.tensor(baseline_makespans, dtype=torch.float32, device=t_device).repeat_interleave(num_tasks)
            flat_reschedule_start = torch.tensor(reschedule_start_times, dtype=torch.float32, device=t_device).repeat_interleave(num_tasks)
            batch_template['task'].x[:, 18] = (flat_base_start - flat_current_times_t) / torch.clamp(flat_base_makespan, min=1e-6)
            batch_template['task'].x[:, 19] = (flat_base_station + 1.0) / max(1, num_stations)
            batch_template['task'].x[:, 20] = flat_base_team / max(1, num_workers)
            batch_template['task'].x[:, 21] = flat_base_frozen
            batch_template['task'].x[:, 22] = (flat_mat_ready > flat_reschedule_start + 1e-9).float()
            delay_mask = (flat_current_times_t > flat_base_start + 1e-9) & (flat_status != 2)
            batch_template['task'].x[:, 23] = ((flat_current_times_t - flat_base_start) / torch.clamp(flat_base_makespan, min=1e-6)) * delay_mask.float()
        
        # ② Worker 特征
        worker_layout = resolve_worker_feature_layout(self.config)
        if batch_template['worker'].x.size(1) != worker_layout.total_dim:
            raise ValueError(
                "GPU 工人特征维度错误: "
                f"{batch_template['worker'].x.size(1)} != {worker_layout.total_dim}"
            )
        wait_w = torch.clamp(flat_free_time - flat_current_times_w, min=0.0)
        batch_template['worker'].x[:, worker_layout.wait_idx] = torch.log1p(wait_w / mean_task_time)
        batch_template['worker'].x[:, worker_layout.free_idx] = (
            flat_free_time <= flat_current_times_w
        ).float()
        
        batch_template['worker'].x[:, worker_layout.lock_slice] = 0.0
        flat_locks_clamped = torch.clamp(flat_locks, max=7)
        batch_template['worker'].x[
            torch.arange(batch_size * num_workers, device=t_device),
            worker_layout.lock_start + flat_locks_clamped,
        ] = 1.0
        
        # GPU 疲劳系数自适应计算
        fatigue_recovery_ratio = getattr(self.config, 'fatigue_recovery_ratio', 0.5)
        fatigue_threshold = getattr(self.config, 'fatigue_threshold_hours', 4.0)
        fatigue_decay = getattr(self.config, 'fatigue_decay_slope', 0.05)
        fatigue_floor = getattr(self.config, 'fatigue_efficiency_floor', 0.60)
        
        idle_time = torch.clamp(flat_current_times_w - flat_last, min=0.0)
        has_last = (flat_last > 0) & (flat_current_times_w > flat_last)
        
        cum_work = flat_cum.clone()
        cum_work[has_last] = torch.clamp(flat_cum[has_last] - idle_time[has_last] * fatigue_recovery_ratio, min=0.0)
        
        overtime = torch.clamp(cum_work - fatigue_threshold, min=0.0)
        fatigue_f = torch.clamp(1.0 - fatigue_decay * overtime / (fatigue_threshold * 2), min=fatigue_floor)
        batch_template['worker'].x[:, worker_layout.fatigue_idx] = fatigue_f
        
        # ③ Station 特征
        batch_template['station'].x[:, 0] = flat_loads / max(1.0, ideal_station_load)
        
        # 负载竞争占比
        loads_tensor = flat_loads.view(batch_size, num_stations)
        sum_loads = loads_tensor.sum(dim=1, keepdim=True)
        max_load = loads_tensor.max(dim=1, keepdim=True)[0]
        
        rel_loads = loads_tensor / (sum_loads + 1e-6)
        max_rel_loads = loads_tensor / (max_load + 1e-6)
        
        batch_template['station'].x[:, 5] = rel_loads.view(-1)
        batch_template['station'].x[:, 6] = max_rel_loads.view(-1)
        
        batch_template['station'].x[:, 7] = flat_slots / max_slots
        batch_template['station'].x[:, 4] = torch.log1p(flat_station_wait / mean_task_time)
        
        # 宏观锁与工人流动分布特征
        locks_tensor = flat_locks.view(batch_size, num_workers)
        is_free_tensor = (flat_free_time <= flat_current_times_w).view(batch_size, num_workers)
        
        mobile_count = (locks_tensor == 0).sum(dim=1, keepdim=True).float()
        batch_template['station'].x[:, 2] = (mobile_count / num_workers).repeat_interleave(num_stations).view(-1)
        
        for s in range(num_stations):
            s_act = s + 1
            bound_count = (locks_tensor == s_act).sum(dim=1, keepdim=True).float()
            free_bound_count = ((locks_tensor == s_act) & is_free_tensor).sum(dim=1, keepdim=True).float()
            
            station_indices = torch.arange(s, batch_size * num_stations, num_stations, device=t_device)
            batch_template['station'].x[station_indices, 1] = (bound_count / num_workers).squeeze(1)
            batch_template['station'].x[station_indices, 3] = (free_bound_count / num_workers).squeeze(1)
            
        # 6. 动态边关系写入
        # ① Assigned 关系边
        if ts_src:
            batch_template['task', 'assigned_to', 'station'].edge_index = torch.tensor([ts_src, ts_dst], dtype=torch.long, device=t_device)
            batch_template['station', 'has_task', 'task'].edge_index = torch.stack([
                batch_template['task', 'assigned_to', 'station'].edge_index[1],
                batch_template['task', 'assigned_to', 'station'].edge_index[0]
            ], dim=0)
        else:
            batch_template['task', 'assigned_to', 'station'].edge_index = torch.empty((2, 0), dtype=torch.long, device=t_device)
            batch_template['station', 'has_task', 'task'].edge_index = torch.empty((2, 0), dtype=torch.long, device=t_device)
            
        # ② Done_by 关系边
        if tw_src:
            batch_template['task', 'done_by', 'worker'].edge_index = torch.tensor([tw_src, tw_dst], dtype=torch.long, device=t_device)
        else:
            batch_template['task', 'done_by', 'worker'].edge_index = torch.empty((2, 0), dtype=torch.long, device=t_device)
            
        # ③ Can_do 关系边覆写
        worker_skill_edges = None
        task_skill_edges = None
        if bool(getattr(self.config, "use_skill_hub", False)):
            task_skill_edges = ctx["task_skill_edge_index"]
            worker_skill_edges = []
            for snap in snapshots:
                base_worker_x = torch.as_tensor(snap["base_worker_x"])
                topology_key = snap.get("worker_topology_key") or worker_topology_key(
                    base_worker_x,
                    int(self.config.num_skill_types),
                )
                cache_key = (int(dataset_idx), topology_key)
                edges = self.worker_skill_topologies.get(cache_key)
                if edges is None:
                    edges = build_worker_skill_edges(
                        base_worker_x,
                        int(self.config.num_skill_types),
                    )
                    self.worker_skill_topologies[cache_key] = edges
                worker_skill_edges.append(edges)

        apply_batched_resource_graph(
            batch_template,
            batch_template['task'].x,
            batch_template['worker'].x,
            batch_size=batch_size,
            num_tasks=num_tasks,
            num_workers=num_workers,
            config=self.config,
            task_skill_edges=task_skill_edges,
            worker_skill_edges=worker_skill_edges,
        )
        batch_template = self._ensure_batch_vectors(
            batch_template,
            batch_size=batch_size,
            num_tasks=num_tasks,
            num_workers=num_workers,
            num_stations=num_stations,
        )
        
        if batch_template is None:
            raise RuntimeError("GPU Batch 重建内部错误：Batch 模板为 None")
        return batch_template
