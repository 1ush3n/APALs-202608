import torch
import numpy as np
from typing import Tuple, Any

from core.time_comparison import (
    release_time_tolerance,
    time_reached_numpy,
    time_reached_scalar,
    time_reached_tensor,
)

class ActionMasker:
    """
    负责计算航空装配线强化学习环境的动作掩码 (Action Mask)。
    支持基于矩阵乘法点积的高吞吐量向量化算法，并植入影子防 Bug 对齐校验。
    """
    def __init__(self, env: Any):
        self.env = env
        
    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成动作掩码，融合影子校验模式。
        """
        env = self.env
        from configs import configs

        task_mask_mode = str(getattr(configs, "task_mask_mode", "resource_aware"))
        station_mask_mode = str(
            getattr(configs, "station_mask_mode", "resource_aware")
        )
        if task_mask_mode == "precedence_release_only" or station_mask_mode == "structural_only":
            return self._get_strict_masks(
                task_mask_mode=task_mask_mode,
                station_mask_mode=station_mask_mode,
            )
        
        # 1. Worker Queue 拥堵判定与可用性掩码
        enable_queue_mask = getattr(configs, 'enable_worker_queue_mask', False)
        max_ratio = getattr(configs, 'max_worker_queue_ratio', 10.0)
        
        if enable_queue_mask:
            queue_limit = max_ratio * env.mean_task_time
            worker_queue_durations = np.maximum(0.0, env.worker_free_time - env.current_time)
            queue_timeout = worker_queue_durations > queue_limit
            queue_ok = ~queue_timeout
        else:
            queue_ok = np.ones(env.num_workers, dtype=bool)
            
        enable_tensor_mask = getattr(configs, 'enable_gpu_tensor_masking', True)
        
        if enable_tensor_mask:
            task_mask_res, station_mask_res, worker_mask_res = self._get_masks_tensorized(queue_ok)
        else:
            task_mask_res, station_mask_res, worker_mask_res = self._get_masks_vectorized(queue_ok)
        
        # 2. 双路影子比对强校验 (防 Bug 护盾)
        if getattr(configs, 'enable_shadow_mask_verification', True):
            task_mask_l, station_mask_l, worker_mask_l = self._get_masks_legacy(queue_ok)
            task_mask_v, station_mask_v, worker_mask_v = self._get_masks_vectorized(queue_ok)
            
            # 使用 assert 验证绝对一致性
            # 无论返回哪一路，都需要确保在 CPU/GPU 各种计算下的结果都严格等于 Legacy 循环算出来的结果
            assert torch.equal(task_mask_res.cpu(), task_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Tensorized Task Mask 不一致！\nTensorized: {task_mask_res.cpu()}\n旧循环: {task_mask_l.cpu()}"
            assert torch.equal(station_mask_res.cpu(), station_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Tensorized Station Mask 不一致！\nTensorized: {station_mask_res.cpu()}\n旧循环: {station_mask_l.cpu()}"
            assert torch.equal(worker_mask_res.cpu(), worker_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Tensorized Worker Mask 不一致！\nTensorized: {worker_mask_res.cpu()}\n旧循环: {worker_mask_l.cpu()}"
                
            assert torch.equal(task_mask_v.cpu(), task_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Vectorized Task Mask 不一致！\nVectorized: {task_mask_v.cpu()}\n旧循环: {task_mask_l.cpu()}"
            assert torch.equal(station_mask_v.cpu(), station_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Vectorized Station Mask 不一致！\nVectorized: {station_mask_v.cpu()}\n旧循环: {station_mask_l.cpu()}"
            assert torch.equal(worker_mask_v.cpu(), worker_mask_l.cpu()), \
                f"🚨 [Mask Alignment Error] Vectorized Worker Mask 不一致！\nVectorized: {worker_mask_v.cpu()}\n旧循环: {worker_mask_l.cpu()}"
                
        return task_mask_res, station_mask_res, worker_mask_res

    def _get_strict_masks(
        self,
        *,
        task_mask_mode: str,
        station_mask_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """只按任务释放和站位永久结构生成 strict mask。"""
        env = self.env
        from configs import configs

        device = env.worker_skill_matrix.device
        task_mask = torch.ones(env.num_tasks, dtype=torch.bool, device=device)
        station_mask = torch.ones(
            (env.num_tasks, env.num_stations), dtype=torch.bool, device=device
        )
        worker_mask = torch.zeros(env.num_workers, dtype=torch.bool, device=device)

        ready_indices = np.where(env.task_status == 1)[0]
        if hasattr(env, "task_material_ready"):
            ready_indices = ready_indices[
                time_reached_numpy(
                    env.task_material_ready[ready_indices],
                    env.current_time,
                    release_time_tolerance(configs),
                )
            ]

        # strict 工序掩码：ready 任务不因当前工人/站位状态被屏蔽。
        if task_mask_mode == "precedence_release_only":
            task_mask[torch.as_tensor(ready_indices, device=device)] = False

        # structural station mask 只保留 min/max/fixed 永久工艺约束。
        for task_id in ready_indices:
            min_station = int(
                env.constraint_engine.minimum_station(
                    int(task_id), env.task_station_map
                )
            )
            fixed_station = int(env.fixed_stations[task_id])
            max_station = int(env.max_allowed_stations[task_id])
            if fixed_station != -1:
                candidates = (
                    [fixed_station]
                    if min_station <= fixed_station <= max_station
                    else []
                )
            else:
                candidates = list(
                    range(
                        max(0, min_station),
                        min(env.num_stations, max_station + 1),
                    )
                )
            for station_id in candidates:
                if 0 <= station_id < env.num_stations:
                    station_mask[task_id, station_id] = False

        if task_mask_mode != "precedence_release_only":
            valid_structural = ~station_mask.any(dim=1)
            task_mask[torch.as_tensor(ready_indices, device=device)] = ~valid_structural[
                torch.as_tensor(ready_indices, device=device)
            ]
        return task_mask, station_mask, worker_mask

    def _get_masks_tensorized(self, queue_ok: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 PyTorch GPU/CPU 张量广播与并行矩阵乘法的极速动作掩码计算算法
        """
        env = self.env
        from configs import configs
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        t_device = env.worker_skill_matrix.device
        
        # 初始化默认都是 Masked (True)
        task_mask = torch.ones(env.num_tasks, dtype=torch.bool, device=t_device)
        station_mask = torch.ones((env.num_tasks, env.num_stations), dtype=torch.bool, device=t_device)
        worker_mask = ~torch.from_numpy(queue_ok).to(t_device)
        
        task_status = torch.from_numpy(env.task_status).to(device=t_device, dtype=torch.long)
        ready_indices = torch.where(task_status == 1)[0]
        if len(ready_indices) == 0:
            return task_mask, station_mask, worker_mask
            
        # [Dynamic Events] 排除物料延迟到达的工序
        if hasattr(env, 'task_material_ready'):
            tolerance = release_time_tolerance(configs)
            # 在已 Ready 的工序中，只有物料就绪时间 <= 当前时间 的才算合法
            valid_mat_mask = time_reached_tensor(
                env.task_material_ready[ready_indices.detach().cpu().numpy()],
                env.current_time,
                tolerance,
                device=t_device,
            )
            ready_indices = ready_indices[valid_mat_mask]
            if len(ready_indices) == 0:
                return task_mask, station_mask, worker_mask
                
        # 1. 判定前驱任务所赋予的最小站位 min_stations
        min_stations_np = np.zeros(len(ready_indices), dtype=int)
        ready_indices_cpu = ready_indices.cpu().numpy()
        for idx, t in enumerate(ready_indices_cpu):
            min_stations_np[idx] = env.constraint_engine.minimum_station(
                int(t), env.task_station_map
            )
        min_stations = torch.from_numpy(min_stations_np).to(device=t_device, dtype=torch.long)
        
        fixed = torch.from_numpy(env.fixed_stations[ready_indices_cpu]).to(device=t_device, dtype=torch.long)
        max_allowed = torch.from_numpy(env.max_allowed_stations[ready_indices_cpu]).to(device=t_device, dtype=torch.long)
        
        # 2. 产生站位合法区间掩码: [len(ready_indices), num_stations]
        stations = torch.arange(env.num_stations, device=t_device)[None, :]
        in_range_normal = (stations >= min_stations[:, None]) & (stations <= max_allowed[:, None])
        in_range_fixed = (
            (stations == fixed[:, None])
            & (stations >= min_stations[:, None])
            & (stations <= max_allowed[:, None])
        )
        station_in_range = torch.where(fixed[:, None] != -1, in_range_fixed, in_range_normal)
        
        # 3. 判定工位槽位容量可用性: [num_stations]
        slots_avail_list = []
        for s in range(env.num_stations):
            allowed = env.station_available_slots[s] if hasattr(env, 'station_available_slots') else max_slots
            slots_avail_list.append(bool(len(env.station_task_finish_times[s]) < allowed))
        slots_avail = torch.from_numpy(np.array(slots_avail_list, dtype=bool)).to(t_device)
        
        # 4. 基于矩阵乘法计算各 Ready 工序在各 Station 上满足条件的可用工人数
        task_static_feat_device = env.task_static_feat.to(t_device)
        req_skills = task_static_feat_device[ready_indices, 1].long()
        req_demands = task_static_feat_device[ready_indices, 2].long()
        
        free_skills = env.worker_skill_matrix.to(t_device)
        valid_skills = req_skills >= 0
        safe_req_skills = req_skills.clamp(min=0)
        skills_mask = free_skills[:, safe_req_skills] > 0.5  # [num_workers, len(ready_indices)]
        skills_mask[:, ~valid_skills] = True
        
        free_locks = torch.from_numpy(env.worker_locks).to(device=t_device, dtype=torch.long)
        stations_1based = torch.arange(env.num_stations, device=t_device) + 1
        lock_compat = (free_locks[:, None] == 0) | (free_locks[:, None] == stations_1based[None, :])
        
        queue_ok_tensor = torch.from_numpy(queue_ok).to(t_device)
        compat_and_queue = lock_compat & queue_ok_tensor[:, None] # [num_workers, num_stations]
        
        # 矩阵点积计算：[len(ready_indices), num_stations]
        avail_count = torch.matmul(
            skills_mask.t().float(),
            compat_and_queue.float()
        )
        
        # 5. 校验工人数与物理容量约束
        valid_pair = station_in_range & slots_avail[None, :] & (avail_count >= req_demands[:, None])
        
        # 6. 装填掩码
        station_mask[ready_indices, :] = ~valid_pair
        task_mask[ready_indices] = ~valid_pair.any(dim=1)
        
        return task_mask, station_mask, worker_mask

    def _get_masks_vectorized(self, queue_ok: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 NumPy 矩阵广播与矩阵点积的向量化极速动作掩码校验算法
        """
        env = self.env
        from configs import configs
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        
        # 初始化默认都是 Masked (True)
        task_mask_np = np.ones(env.num_tasks, dtype=bool)
        station_mask_np = np.ones((env.num_tasks, env.num_stations), dtype=bool)
        
        ready_indices = np.where(env.task_status == 1)[0]
        if len(ready_indices) == 0:
            return (
                torch.from_numpy(task_mask_np).to(torch.bool),
                torch.from_numpy(station_mask_np).to(torch.bool),
                torch.from_numpy(~queue_ok).to(torch.bool)
            )
            
        free_skills = env.worker_skill_matrix.cpu().numpy()
        free_locks = env.worker_locks
        
        # [Dynamic Events] 排除物料延迟到达的工序
        if hasattr(env, 'task_material_ready'):
            tolerance = release_time_tolerance(configs)
            valid_release = time_reached_numpy(
                env.task_material_ready[ready_indices],
                env.current_time,
                tolerance,
            )
            ready_indices = ready_indices[valid_release]
            if len(ready_indices) == 0:
                return (
                    torch.from_numpy(task_mask_np).to(torch.bool),
                    torch.from_numpy(station_mask_np).to(torch.bool),
                    torch.from_numpy(~queue_ok).to(torch.bool)
                )
                
        # 1. 向量化计算前驱任务所能赋予的最小站位 min_stations
        min_stations = np.zeros(len(ready_indices), dtype=int)
        for idx, t in enumerate(ready_indices):
            min_stations[idx] = env.constraint_engine.minimum_station(
                int(t), env.task_station_map
            )
                
        fixed = env.fixed_stations[ready_indices]
        max_allowed = env.max_allowed_stations[ready_indices]
        
        # 2. 产生站位合法区间掩码: [len(ready_indices), num_stations]
        stations = np.arange(env.num_stations)[None, :]
        in_range_normal = (stations >= min_stations[:, None]) & (stations <= max_allowed[:, None])
        in_range_fixed = (
            (stations == fixed[:, None])
            & (stations >= min_stations[:, None])
            & (stations <= max_allowed[:, None])
        )
        station_in_range = np.where(fixed[:, None] != -1, in_range_fixed, in_range_normal)
        
        # 3. 判定工位槽位容量可用性: [num_stations]
        slots_avail = np.array([
            len(env.station_task_finish_times[s]) < (env.station_available_slots[s] if hasattr(env, 'station_available_slots') else max_slots)
            for s in range(env.num_stations)
        ], dtype=bool)
        
        # 4. 基于矩阵乘法计算各 Ready 工序在各 Station 上满足条件的可用工人数
        req_skills = env.task_static_feat[ready_indices, 1].long().cpu().numpy()
        req_demands = env.task_static_feat[ready_indices, 2].long().cpu().numpy()
        
        # skills_mask: [num_workers, len(ready_indices)]
        valid_skills = req_skills >= 0
        safe_req_skills = np.maximum(req_skills, 0)
        skills_mask = free_skills[:, safe_req_skills] > 0.5
        skills_mask[:, ~valid_skills] = True
        
        # lock_compat: [num_workers, num_stations]
        lock_compat = (free_locks[:, None] == 0) | (free_locks[:, None] == np.arange(env.num_stations) + 1)
        
        # compat_and_queue: [num_workers, num_stations]
        compat_and_queue = lock_compat & queue_ok[:, None]
        
        # 点积矩阵乘法，获取 [len(ready_indices), num_stations]
        avail_count = skills_mask.T.astype(int) @ compat_and_queue.astype(int)
        
        # 5. 校验工人数与物理容量约束
        valid_pair = station_in_range & slots_avail[None, :] & (avail_count >= req_demands[:, None])
        
        # 6. 装填掩码
        station_mask_np[ready_indices, :] = ~valid_pair
        task_mask_np[ready_indices] = ~np.any(valid_pair, axis=1)
        
        worker_mask = torch.from_numpy(~queue_ok).to(torch.bool)
        task_mask = torch.from_numpy(task_mask_np).to(torch.bool)
        station_mask = torch.from_numpy(station_mask_np).to(torch.bool)
        
        return task_mask, station_mask, worker_mask

    def _get_masks_legacy(self, queue_ok: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        旧版基于 CPU for 双重循环的动作掩码计算逻辑 (用作影子校验对齐基准)
        """
        env = self.env
        from configs import configs
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        
        worker_mask = torch.from_numpy(~queue_ok).to(torch.bool)
        
        task_mask = torch.ones(env.num_tasks, dtype=torch.bool)
        station_mask = torch.ones((env.num_tasks, env.num_stations), dtype=torch.bool)
        
        ready_indices = np.where(env.task_status == 1)[0]
        free_skills = env.worker_skill_matrix.cpu().numpy() 
        free_locks = env.worker_locks
        
        for t in ready_indices:
            if (
                hasattr(env, 'task_material_ready')
                and not time_reached_scalar(
                    env.task_material_ready[t],
                    env.current_time,
                    release_time_tolerance(configs),
                )
            ):
                continue

            min_station = env.constraint_engine.minimum_station(
                int(t), env.task_station_map
            )
            
            fixed = env.fixed_stations[t]
            req_skill = int(env.task_static_feat[t, 1].item())
            req_demand = int(env.task_static_feat[t, 2].item())
            
            valid_stations = False
            max_station = env.max_allowed_stations[t]
            station_range = (
                [fixed]
                if fixed != -1 and min_station <= fixed <= max_station
                else []
                if fixed != -1
                else list(range(min_station, min(env.num_stations, max_station + 1)))
            )
            has_skill = (
                free_skills[:, req_skill] > 0.5
                if req_skill >= 0
                else np.ones(env.num_workers, dtype=bool)
            )
            
            for s in station_range:
                if s < 0 or s >= env.num_stations: continue
                allowed_slots = env.station_available_slots[s] if hasattr(env, 'station_available_slots') else max_slots
                if len(env.station_task_finish_times[s]) >= allowed_slots:
                    continue 

                compatible_lock = (free_locks == 0) | (free_locks == s + 1)
                avail = np.sum(compatible_lock & has_skill & queue_ok)
                
                if avail >= req_demand:
                    station_mask[t, s] = False
                    valid_stations = True
            
            if valid_stations:
                task_mask[t] = False
                
        return task_mask, station_mask, worker_mask
