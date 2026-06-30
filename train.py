import os
omp_threads = os.environ.get("OMP_NUM_THREADS", "")
if omp_threads:
    try:
        if int(omp_threads) <= 0:
            os.environ["OMP_NUM_THREADS"] = "1"
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"

# 启用可扩展显存段以缓解动态图 GNN 变长 batch 的碎片化；峰值显存仍由 batch 控制。
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import sys
import io
import traceback
import argparse
import json
from pathlib import Path

if sys.platform == 'win32':
    # 强制将标准输出重定向为 UTF-8，防止 Windows 终端在输出 Emoji 时崩溃
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 添加项目根目录到路径

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs, load_training_config
from runtime.configuration import resolve_runtime_config
from runtime.artifacts import (
    checkpoint_paths as resolve_artifact_checkpoint_paths,
    resolve_path as resolve_artifact_path,
    sanitize_name,
)
from runtime.multiscale import (
    BenchmarkScore,
    InverseScaleDatasetSampler,
    apply_scale_profile_to_agent,
    build_dataset_candidates,
    parse_reference_makespans,
    score_multi_benchmark,
)
import pandas as pd
from baselines.heuristic.baseline_ga import GeneticAlgorithmScheduler
from utils.visualization import plot_gantt
import random
from utils.vector_env import VectorEnv, EnvCreator
from utils.reschedule import (
    calculate_reschedule_composite_score,
    calculate_stability_metrics,
    load_baseline_schedule,
    load_reschedule_scenarios,
    sample_task_delay_scenario,
    save_reschedule_scenarios,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_workspace_path(path_like, base_dir: Path = PROJECT_ROOT) -> Path:
    """将配置中的路径解析为跨平台绝对路径；绝对路径保持不变。"""
    return resolve_artifact_path(path_like, base_dir)


def sanitize_experiment_name(name: object) -> str:
    """将实验名压缩为安全目录名，避免不同配置的 checkpoint 互相覆盖。"""
    return sanitize_name(name)


def resolve_checkpoint_paths(config_obj=configs) -> dict[str, Path]:
    """按 experiment_name/checkpoint_root 解析当前实验的模型保存路径。"""
    paths = resolve_artifact_checkpoint_paths(config_obj, PROJECT_ROOT)
    model_dir = paths["model_dir"]
    best_model_dir = paths["legacy_best"].parent
    return {
        "model_dir": model_dir,
        "checkpoint_path": paths["legacy_latest"],
        "best_model_dir": best_model_dir,
        "best_model_path": paths["legacy_best"],
        "best_model_meta_path": paths["legacy_best_meta"],
    }


def resolve_tensorboard_log_root(config_obj=configs) -> Path:
    """严格使用配置中的 TensorBoard 根目录，不再按平台隐式改写。"""
    return Path(getattr(config_obj, "log_dir", "/root/tf-logs")).expanduser()


def write_best_model_meta(
    meta_path: Path,
    *,
    episode: int,
    eval_makespan: float,
    selection_metric: str = "eval_makespan",
    best_score: float | None = None,
    score_terms: dict[str, float] | None = None,
    constraint_metrics: dict[str, float] | None = None,
    config_obj=configs,
) -> None:
    """保存 best model 的可追溯元数据，方便服务器和本机定位模型来源。"""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode": int(episode),
        "selection_metric": selection_metric,
        "eval_makespan": float(eval_makespan),
        "best_score": None if best_score is None else float(best_score),
        "score_terms": score_terms or {},
        "constraint_metrics": constraint_metrics or {},
        "config_paths": list(getattr(config_obj, "config_paths", ())),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": sanitize_experiment_name(getattr(config_obj, "experiment_name", "default")),
        "data_file_path": getattr(config_obj, "data_file_path", ""),
        "train_data_path_or_dir": getattr(config_obj, "train_data_path_or_dir", ""),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def ensure_reschedule_baseline_available(config_obj=configs) -> Path | None:
    """重调度训练前确保 baseline CSV 存在；不存在时用初始模型生成一次。"""
    if not getattr(config_obj, "enable_reschedule_mode", False):
        return None
    baseline_path = resolve_workspace_path(getattr(config_obj, "reschedule_baseline_schedule_path", "results/final_schedule.csv"))
    if baseline_path.exists():
        return baseline_path
    model_path = resolve_workspace_path(getattr(config_obj, "reschedule_baseline_model_path", "checkpoints/initial_schedule/bestmodel/best_model.pth"))
    if not model_path.exists():
        raise FileNotFoundError(f"重调度 baseline 不存在，且找不到用于生成 baseline 的初始模型: {model_path}")

    backup = {
        "enable_reschedule_mode": getattr(config_obj, "enable_reschedule_mode", False),
        "task_feat_dim": getattr(config_obj, "task_feat_dim", 18),
        "randomize_durations": getattr(config_obj, "randomize_durations", False),
        "enable_dynamic_events": getattr(config_obj, "enable_dynamic_events", False),
    }
    try:
        setattr(config_obj, "enable_reschedule_mode", False)
        setattr(config_obj, "task_feat_dim", 18)
        setattr(config_obj, "randomize_durations", False)
        setattr(config_obj, "enable_dynamic_events", False)
        from scripts.generate_schedule import generate_schedule

        df = generate_schedule(model_path=str(model_path))
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        if baseline_path != PROJECT_ROOT / "results" / "final_schedule.csv":
            df.to_csv(baseline_path, index=False)
    finally:
        for key, value in backup.items():
            setattr(config_obj, key, value)
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline 自动生成后仍未找到: {baseline_path}")
    return baseline_path


def load_warm_start_weights_with_input_expansion(model: torch.nn.Module, model_path: Path, device: torch.device) -> dict[str, int]:
    """加载初始调度模型；当 task 输入维度扩展时复制旧权重的已有列。"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    target_state = model.state_dict()
    loaded_exact = 0
    loaded_expanded = 0
    skipped = 0

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key) if isinstance(source_state, dict) else None
        if source_tensor is None:
            skipped += 1
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = source_tensor.to(target_tensor.device, dtype=target_tensor.dtype)
            loaded_exact += 1
            continue
        if (
            source_tensor.ndim == target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] <= target_tensor.shape[1]
        ):
            patched = target_tensor.clone()
            patched[:, : source_tensor.shape[1]] = source_tensor.to(target_tensor.device, dtype=target_tensor.dtype)
            target_state[key] = patched
            loaded_expanded += 1
            continue
        skipped += 1

    model.load_state_dict(target_state, strict=True)
    return {"loaded_exact": loaded_exact, "loaded_expanded": loaded_expanded, "skipped": skipped}


def ensure_reschedule_eval_scenarios_available(config_obj=configs) -> Path | None:
    """确保重调度验证使用固定场景文件，而不是每次临时随机生成。"""
    if not getattr(config_obj, "enable_reschedule_mode", False):
        return None
    scenario_path = resolve_workspace_path(getattr(config_obj, "reschedule_eval_scenario_path", "results/reschedule_eval_scenarios.csv"))
    if scenario_path.exists():
        return scenario_path

    baseline_path = ensure_reschedule_baseline_available(config_obj)
    if baseline_path is None:
        return None
    baseline = load_baseline_schedule(baseline_path)
    num_scenarios = max(1, int(getattr(config_obj, "reschedule_eval_num_scenarios", 4)))
    seed = int(getattr(config_obj, "reschedule_eval_scenario_seed", 42))
    scenarios = []
    for idx in range(num_scenarios):
        scenario = sample_task_delay_scenario(
            baseline,
            rng=np.random.RandomState(seed + idx),
            min_start_ratio=float(getattr(config_obj, "reschedule_start_time_min_ratio", 0.15)),
            max_start_ratio=float(getattr(config_obj, "reschedule_start_time_max_ratio", 0.65)),
            task_prob=float(getattr(config_obj, "reschedule_delay_task_prob", 0.08)),
            delay_min=float(getattr(config_obj, "reschedule_delay_min", 5.0)),
            delay_max=float(getattr(config_obj, "reschedule_delay_max", 30.0)),
        )
        scenarios.append((f"eval_{idx:03d}", scenario))
    save_reschedule_scenarios(scenario_path, scenarios)
    print(f"固定重调度验证场景已生成: {scenario_path}")
    return scenario_path

# 设置全局随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def initialize_training_config(args, argv=None, system_name: str | None = None):
    """统一加载默认值、YAML、平台配置和命令行覆盖。"""
    _, loaded_paths, explicit_fields = resolve_runtime_config(
        args,
        target=configs,
        system_name=system_name,
    )
    args.explicit_config_fields = explicit_fields

    precision = str(configs.float32_matmul_precision)
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(f"float32_matmul_precision 无效: {precision}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(precision)

    print(
        "[Runtime] "
        f"platform={system_name or __import__('platform').system()} "
        f"configs={[str(Path(path)) for path in loaded_paths]} "
        f"num_envs={configs.num_envs} "
        f"worker_threads={configs.vector_env_worker_threads} "
        f"start_method={configs.vector_env_start_method} "
        f"amp={configs.lightning_precision} "
        f"matmul_precision={precision}",
        flush=True,
    )
    return configs

# ---------------------------------------------------------------------------
# 经验回放缓冲区 (Memory Buffer)
# ---------------------------------------------------------------------------
class Memory:
    """
    存储 PPO 训练所需的轨迹数据。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.is_truncated = []
        self.masks = [] # (task_mask, station_mask, worker_mask)
        self.values = [] # (state_value)
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.is_truncated[:]
        del self.masks[:]
        del self.values[:]

        # 显式释放残余对象，防 OOM 内存泄漏
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def refresh_env_observation(env):
    """
    在离散事件等待推进后重建当前观测，避免使用旧时刻的图状态。
    支持真实环境和 VectorEnv 的 EnvProxy。
    """
    if hasattr(env, "_get_observation"):
        return env._get_observation()
    if hasattr(env, "rebuild_state_from_snapshot") and hasattr(env, "get_state_snapshot"):
        return env.rebuild_state_from_snapshot(env.get_state_snapshot())
    raise TypeError(f"无法从环境类型 {type(env)!r} 刷新观测。")

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def _get_cpm_earliest_starts(ctx: dict) -> np.ndarray:
    """基于静态工序图计算 CPM 最早开工时间，用于 APAL 诊断指标。"""
    cached = ctx.get("_diag_cpm_earliest_starts")
    if cached is not None:
        return cached

    task_static_feat = ctx["task_static_feat"]
    durations = task_static_feat[:, 0].detach().cpu().numpy() if torch.is_tensor(task_static_feat) else np.asarray(task_static_feat)[:, 0]
    num_tasks = len(durations)
    predecessors = ctx["predecessors"]
    successors = ctx["successors"]
    def _neighbors(container, idx: int):
        return container.get(idx, []) if hasattr(container, "get") else container[idx]

    indegree = [len(_neighbors(predecessors, i)) for i in range(num_tasks)]
    queue = [i for i, deg in enumerate(indegree) if deg == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for nxt in _neighbors(successors, node):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    earliest = np.zeros(num_tasks, dtype=float)
    for node in topo_order:
        for pred in _neighbors(predecessors, node):
            earliest[node] = max(earliest[node], earliest[pred] + durations[pred])
    ctx["_diag_cpm_earliest_starts"] = earliest
    return earliest


def compute_apal_rollout_diagnostics(env_proxy, snapshot: dict, masks) -> dict:
    """从轻量 snapshot 计算 APAL 专属 rollout 诊断，不修改环境状态。"""
    task_mask, _, _ = masks
    current_time = float(snapshot["current_time"])
    worker_free_time = np.asarray(snapshot["worker_free_time"], dtype=float)
    station_slots = np.asarray(
        snapshot.get("station_available_slots", np.full(len(snapshot["station_loads"]), configs.max_slots_per_station)),
        dtype=float,
    )

    schedulable_tasks = float((~task_mask).sum().item())
    worker_waits = np.maximum(0.0, worker_free_time - current_time)
    avg_worker_wait_h = float(worker_waits.mean()) if worker_waits.size > 0 else 0.0
    worker_idle_ratio = float(np.mean(worker_free_time <= current_time)) if worker_free_time.size > 0 else 0.0

    active_counts = np.zeros(len(station_slots), dtype=float)
    station_next_finish = [[] for _ in range(len(station_slots))]
    for _, station_id, _, _, finish_time in snapshot["assigned_tasks"]:
        if station_id >= 0 and finish_time > current_time:
            active_counts[station_id] += 1.0
            station_next_finish[station_id].append(float(finish_time))

    free_slots = np.maximum(0.0, station_slots - active_counts)
    station_slot_vacancy_ratio = float(free_slots.sum() / max(1.0, station_slots.sum()))
    station_waits = []
    for sid, finishes in enumerate(station_next_finish):
        if len(finishes) >= max(1, int(station_slots[sid])):
            station_waits.append(max(0.0, min(finishes) - current_time))
        else:
            station_waits.append(0.0)
    avg_station_wait_h = float(np.mean(station_waits)) if station_waits else 0.0

    critical_offset_values = []
    dataset_idx = snapshot.get("dataset_idx", 0)
    pool = getattr(env_proxy, "dataset_pool", [])
    ctx = pool[dataset_idx] if dataset_idx < len(pool) and pool[dataset_idx] is not None else {}
    is_critical = np.asarray(ctx.get("is_critical", []), dtype=bool)
    if is_critical.size > 0:
        cpm_earliest = _get_cpm_earliest_starts(ctx)
        for task_id, _, _, start_time, _ in snapshot["assigned_tasks"]:
            if 0 <= task_id < len(is_critical) and is_critical[task_id]:
                critical_offset_values.append(max(0.0, float(start_time) - float(cpm_earliest[task_id])))

    return {
        "schedulable_tasks": schedulable_tasks,
        "avg_worker_wait_h": avg_worker_wait_h,
        "avg_station_wait_h": avg_station_wait_h,
        "avg_resource_wait_h": avg_worker_wait_h + avg_station_wait_h,
        "station_slot_vacancy_ratio": station_slot_vacancy_ratio,
        "worker_idle_ratio": worker_idle_ratio,
        "critical_start_offset_h": float(np.mean(critical_offset_values)) if critical_offset_values else np.nan,
    }


def select_actions_batch_compat(
    agent,
    *,
    obs_list,
    mask_task_list,
    mask_station_matrix_list,
    mask_worker_list,
    deterministic: bool,
    temperature: float,
    is_eval: bool,
):
    """兼容旧版 PPOAgent：缺少批量动作接口时退回逐环境采样。"""
    batch_selector = getattr(agent, "select_actions_batch", None)
    if callable(batch_selector):
        return batch_selector(
            obs_list=obs_list,
            mask_task_list=mask_task_list,
            mask_station_matrix_list=mask_station_matrix_list,
            mask_worker_list=mask_worker_list,
            deterministic=deterministic,
            temperature=temperature,
            is_eval=is_eval,
        )

    if not getattr(agent, "_warned_missing_batch_selector", False):
        print("WARNING: PPOAgent 缺少 select_actions_batch，已退回逐环境 select_action。请确认 ppo_agent.py 与 train.py 版本一致。")
        setattr(agent, "_warned_missing_batch_selector", True)

    results = []
    for obs, task_mask, station_mask, worker_mask in zip(
        obs_list,
        mask_task_list,
        mask_station_matrix_list,
        mask_worker_list,
    ):
        results.append(
            agent.select_action(
                obs,
                mask_task=task_mask,
                mask_station_matrix=station_mask,
                mask_worker=worker_mask,
                deterministic=deterministic,
                temperature=temperature,
                is_eval=is_eval,
            )
        )
    return results


def evaluate_model(
    env,
    agent,
    num_runs=1,
    temperature=None,
    writer=None,
    current_ep=0,
    scenario_names=None,
):
    """
    使用包含温度平滑的定制定向策略评估当前模型性能。
    在多情景（Standard、工时加噪、工人缺损、动态故障）固定扰动下进行评估，
    保证不同Episode评估考卷的100%一致性，并往 TensorBoard 写入详细分流数据。
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)

    # 评估期间必须关闭 Dropout 等机制
    agent.policy.eval()
    verbose_eval = bool(getattr(configs, "verbose_eval_progress", writer is None))
    
    # 备份当前 configs 状态，以防评估过程污染训练中的全局超参
    backup_dynamic_events = getattr(configs, 'enable_dynamic_events', False)
    backup_station_breakdown = getattr(configs, 'enable_station_breakdown', False)
    backup_material_delay = getattr(configs, 'enable_material_delay', False)
    
    scenarios = [
        # Scenario 0: Standard (纯净无扰动)
        {
            'name': '0_Standard',
            'rand_dur': False,
            'rand_w': False,
            'dyn_ev': False,
            'seed': None
        },
        # Scenario 1: Duration Noise (工时固定加噪)
        {
            'name': '1_DurationNoise',
            'rand_dur': True,
            'rand_w': False,
            'dyn_ev': False,
            'seed': None
        },
        # Scenario 2: Worker Perturbation (工人固定缺损)
        {
            'name': '2_WorkerNoise',
            'rand_dur': False,
            'rand_w': True,
            'dyn_ev': False,
            'seed': None
        },
        # Scenario 3: Dynamic Events (固定事件扰动)
        {
            'name': '3_DynamicEvents',
            'rand_dur': False,
            'rand_w': False,
            'dyn_ev': True,
            'seed': None
        }
    ]
    if scenario_names is not None:
        aliases = {
            "standard": "0_Standard",
            "duration_noise": "1_DurationNoise",
            "worker_noise": "2_WorkerNoise",
            "dynamic_events": "3_DynamicEvents",
        }
        selected = {aliases.get(str(name).lower(), str(name)) for name in scenario_names}
        scenarios = [scenario for scenario in scenarios if scenario["name"] in selected]
        if not scenarios:
            raise ValueError(f"未选择任何有效评估场景: {scenario_names}")
    base_eval_seed = int(getattr(configs, "seed", 42))
    runs_per_scenario = max(1, int(num_runs))
    for scenario_idx, scenario in enumerate(scenarios):
        scenario["seed"] = base_eval_seed + scenario_idx * runs_per_scenario
    scenario_results = []
    
    # 依次执行各情景推演
    for sc in scenarios:
        # 配置环境事件参数
        if sc['dyn_ev']:
            setattr(configs, 'enable_dynamic_events', True)
            setattr(configs, 'enable_station_breakdown', True)
            setattr(configs, 'enable_material_delay', True)
        else:
            setattr(configs, 'enable_dynamic_events', False)
            setattr(configs, 'enable_station_breakdown', False)
            setattr(configs, 'enable_material_delay', False)
            
        sc_makespans = []
        sc_balances = []
        sc_rewards = []
        sc_schedules = []
        sc_durations = []
        sc_worker_utils = []
        sc_station_utils = []

        for run_idx in range(runs_per_scenario):
            state = env.reset(randomize_duration=sc['rand_dur'], randomize_workers=sc['rand_w'], seed=sc['seed'] + run_idx)
            done = False
            total_reward = 0
            device = agent.device
             
            start_time = time.time()
            while not done:
                task_mask, station_mask, worker_mask = env.get_masks()
                if task_mask.all():
                    if env.try_wait_for_resources():
                         state = refresh_env_observation(env)
                         continue
                    done = True
                    break
                    
                action_ret = agent.select_action(
                    state.to(device), 
                    mask_task=task_mask.to(device), 
                    mask_station_matrix=station_mask.to(device),
                    mask_worker=worker_mask.to(device),
                    deterministic=(temperature == 0.0),
                    temperature=temperature,
                    is_eval=True
                )
                
                if action_ret[0] is None:
                    task_mask = torch.ones_like(task_mask) 
                    break
                    
                action, _, _, _, is_invalid = action_ret
                
                if getattr(configs, 'ablation_no_mask', False) and is_invalid:
                    task_mask = torch.ones_like(task_mask) 
                    break
                
                state, reward, done, _ = env.step(action)
                total_reward += reward
                 
            end_time = time.time()
            
            if len(env.assigned_tasks) != env.num_tasks:
                sc_makespans.append(env.ideal_makespan * 3.0) 
                sc_balances.append(env.ideal_station_load * 3.0)
                dynamic_penalty = configs.deadlock_penalty_constant
                sc_rewards.append(total_reward - (dynamic_penalty * configs.r_coef_makespan * configs.reward_scale * 4))
                sc_schedules.append([])
                sc_durations.append(end_time - start_time)
                sc_worker_utils.append(0.0)
                sc_station_utils.append(0.0)
            else:
                final_makespan = np.max(env.station_wall_clock)
                sc_makespans.append(final_makespan) 
                sc_balances.append(np.std(env.station_loads))     
                sc_rewards.append(total_reward)
                sc_schedules.append(env.assigned_tasks)
                sc_durations.append(end_time - start_time)
                
                worker_busy_time = 0.0
                station_busy_time = np.zeros(env.num_stations)
                for (tid, sid, team, start, end) in env.assigned_tasks:
                    dur = end - start
                    worker_busy_time += dur * len(team)
                    if sid >= 0:
                        station_busy_time[sid] += dur
                
                w_util = worker_busy_time / (env.num_workers * final_makespan) if final_makespan > 0 else 0.0
                max_slots = getattr(configs, 'max_slots_per_station', 3)
                s_util = np.sum(station_busy_time) / (env.num_stations * max_slots * final_makespan) if final_makespan > 0 else 0.0
                 
                sc_worker_utils.append(w_util)
                sc_station_utils.append(s_util)
            if verbose_eval:
                assigned_count = len(getattr(env, "assigned_tasks", []))
                complete = assigned_count == getattr(env, "num_tasks", assigned_count)
                print(
                    f"[Eval][RunResult] scenario={sc['name']} run={run_idx + 1}/{runs_per_scenario} "
                    f"complete={int(complete)} tasks={assigned_count}/{getattr(env, 'num_tasks', '?')} "
                    f"Mk={float(sc_makespans[-1]):.2f} Time={float(sc_durations[-1]):.2f}s",
                    flush=True,
                )
                 
        best_idx = np.argmin(sc_makespans)
        sc_res = {
            'name': sc['name'],
            'makespan': float(np.mean(sc_makespans)),
            'balance': float(np.mean(sc_balances)),
            'reward': float(np.mean(sc_rewards)),
            'schedule': sc_schedules[best_idx],
            'duration': float(np.mean(sc_durations)),
            'w_util': float(np.mean(sc_worker_utils)),
            's_util': float(np.mean(sc_station_utils))
        }
        scenario_results.append(sc_res)
        if verbose_eval:
            print(
                f"[Eval][ScenarioResult] scenario={sc['name']} "
                f"Mk={sc_res['makespan']:.2f} Bal={sc_res['balance']:.2f} "
                f"Reward={sc_res['reward']:.2f} AvgTime={sc_res['duration']:.2f}s",
                flush=True,
            )
         
        if writer is not None:
            writer.add_scalar(f'Eval_Scenario/{sc["name"]}_Makespan', sc_res['makespan'], current_ep)
            writer.add_scalar(f'Eval_Scenario/{sc["name"]}_Reward', sc_res['reward'], current_ep)
            writer.add_scalar(f'Eval_Scenario/{sc["name"]}_WorkerUtil', sc_res['w_util'], current_ep)
            writer.add_scalar(f'Eval_Scenario/{sc["name"]}_StationUtil', sc_res['s_util'], current_ep)
            
    # 还原 configs 全局设置
    setattr(configs, 'enable_dynamic_events', backup_dynamic_events)
    setattr(configs, 'enable_station_breakdown', backup_station_breakdown)
    setattr(configs, 'enable_material_delay', backup_material_delay)
    
    # 汇总均值作为主接口的评估反馈 (保证解包结构与旧接口 100% 对齐)
    avg_makespan = float(np.mean([r['makespan'] for r in scenario_results]))
    avg_balance = float(np.mean([r['balance'] for r in scenario_results]))
    avg_reward = float(np.mean([r['reward'] for r in scenario_results]))
    avg_duration = float(np.mean([r['duration'] for r in scenario_results]))
    avg_w_util = float(np.mean([r['w_util'] for r in scenario_results]))
    avg_s_util = float(np.mean([r['s_util'] for r in scenario_results]))
    
    # 最佳排程图纸返回 Standard (0_Standard) 下的结果，以方便正常绘制与检查
    best_sch = scenario_results[0]['schedule']
    if verbose_eval:
        print(
            "[Eval][Result] "
            f"Mk={avg_makespan:.2f} Bal={avg_balance:.2f} Reward={avg_reward:.2f} "
            f"AvgTime={avg_duration:.2f}s WUtil={avg_w_util * 100:.1f}% "
            f"SUtil={avg_s_util * 100:.1f}% BestTasks={len(best_sch)}",
            flush=True,
        )
    
    return avg_makespan, avg_balance, avg_reward, best_sch, avg_duration, avg_w_util, avg_s_util


def evaluate_initial_multi_benchmark(agent, config_obj=configs, writer=None, current_ep=0):
    """在多个固定初始排程基准集上评估，并计算归一化综合分。"""
    refs = parse_reference_makespans(getattr(config_obj, "multi_benchmark_reference_makespans", {}))
    rows: list[BenchmarkScore] = []
    device = agent.device
    was_training = bool(agent.policy.training)
    paths = list(getattr(config_obj, "multi_benchmark_data_paths", []))
    if not paths:
        raise ValueError("multi_benchmark_data_paths 不能为空")
    backups = {
        "enable_dynamic_events": getattr(config_obj, "enable_dynamic_events", False),
        "enable_station_breakdown": getattr(config_obj, "enable_station_breakdown", False),
        "enable_material_delay": getattr(config_obj, "enable_material_delay", False),
        "enable_online_duration_perturb": getattr(config_obj, "enable_online_duration_perturb", False),
        "enable_worker_fatigue": getattr(config_obj, "enable_worker_fatigue", False),
        "randomize_durations": getattr(config_obj, "randomize_durations", False),
    }
    for key in backups:
        setattr(config_obj, key, False)

    try:
        for raw_path in paths:
            data_path = resolve_workspace_path(raw_path)
            benchmark_name = data_path.stem
            if benchmark_name not in refs:
                raise ValueError(f"缺少基准 {benchmark_name} 的 reference makespan")
            benchmark_seed = int(getattr(config_obj, "seed", 42)) + len(rows)
            env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=benchmark_seed)
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=benchmark_seed)
            done = False
            invalid_step_count = 0
            start_time = time.time()

            agent.policy.eval()
            for _ in range(env.num_tasks * 3):
                if done:
                    break
                task_mask, station_mask, worker_mask = env.get_masks()
                if task_mask.all():
                    if env.try_wait_for_resources():
                        state = refresh_env_observation(env)
                        continue
                    invalid_step_count += 1
                    break
                action_ret = agent.select_action(
                    state.to(device),
                    mask_task=task_mask.to(device),
                    mask_station_matrix=station_mask.to(device),
                    mask_worker=worker_mask.to(device),
                    deterministic=True,
                    temperature=0.0,
                    is_eval=True,
                )
                if action_ret[0] is None:
                    invalid_step_count += 1
                    break
                action, _, _, _, is_invalid = action_ret
                if is_invalid:
                    invalid_step_count += 1
                    break
                state, _reward, done, info = env.step(action)
                if info.get("invalid_action", False):
                    invalid_step_count += 1
                    break

            complete = len(env.assigned_tasks) == env.num_tasks
            if complete and invalid_step_count == 0:
                makespan = float(np.max(env.station_wall_clock))
            else:
                makespan = float(env.ideal_makespan * 3.0)
            reference = float(refs[benchmark_name])
            row = BenchmarkScore(
                benchmark_name=benchmark_name,
                data_path=str(data_path),
                makespan=makespan,
                reference_makespan=reference,
                normalized_score=float(makespan / reference),
                complete=bool(complete),
                invalid_step_count=int(invalid_step_count),
                inference_time=float(time.time() - start_time),
            )
            rows.append(row)
            if writer is not None:
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_NormalizedScore", row.normalized_score, current_ep)
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_Makespan", row.makespan, current_ep)
                writer.add_scalar(f"MultiBenchmark/{benchmark_name}_InvalidSteps", row.invalid_step_count, current_ep)
    finally:
        for key, value in backups.items():
            setattr(config_obj, key, value)
        if was_training:
            agent.policy.train()
    return score_multi_benchmark(rows)


def _compute_assignment_utilization(env, final_makespan: float) -> tuple[float, float]:
    worker_busy_time = 0.0
    station_busy_time = np.zeros(env.num_stations)
    for _tid, sid, team, start, end in env.assigned_tasks:
        dur = max(0.0, float(end) - float(start))
        worker_busy_time += dur * len(team)
        if sid >= 0:
            station_busy_time[sid] += dur
    worker_util = worker_busy_time / (env.num_workers * final_makespan) if final_makespan > 0 else 0.0
    max_slots = getattr(configs, 'max_slots_per_station', 3)
    station_util = np.sum(station_busy_time) / (env.num_stations * max_slots * final_makespan) if final_makespan > 0 else 0.0
    return float(worker_util), float(station_util)


def _compute_reschedule_constraint_metrics(env) -> dict[str, float]:
    baseline = getattr(env, "baseline_schedule", None)
    start_time = float(getattr(env, "reschedule_start_time", 0.0))
    assigned_by_task = {int(item[0]): item for item in env.assigned_tasks}
    metrics = {
        "frozen_violation_count": 0.0,
        "release_violation_count": 0.0,
        "precedence_violation_count": 0.0,
        "worker_overlap_violation_count": 0.0,
        "station_slot_violation_count": 0.0,
        "skill_violation_count": 0.0,
        "demand_violation_count": 0.0,
        "duplicate_task_count": 0.0,
        "missing_task_count": 0.0,
        "takt_h": 0.0,
        "takt_violation_h": 0.0,
        "lower_bound_h": float(getattr(env, "reschedule_lower_bound", 0.0)),
        "takt_feasible": float(bool(getattr(env, "reschedule_takt_feasible", True))),
        "start_deviation_mean_h": 0.0,
        "station_change_rate": 0.0,
        "team_change_rate": 0.0,
    }

    task_ids = [int(item[0]) for item in env.assigned_tasks]
    metrics["duplicate_task_count"] = float(len(task_ids) - len(set(task_ids)))
    metrics["missing_task_count"] = float(max(0, env.num_tasks - len(set(task_ids))))

    if baseline is not None:
        metrics["takt_h"] = float(baseline.makespan)
        final_makespan = float(np.max(env.station_wall_clock)) if len(env.station_wall_clock) > 0 else 0.0
        metrics["takt_violation_h"] = max(0.0, final_makespan - float(baseline.makespan))
        stability = calculate_stability_metrics(baseline, env.assigned_tasks, current_time=start_time)
        metrics["start_deviation_mean_h"] = float(stability["start_deviation_mean_h"])
        metrics["station_change_rate"] = float(stability["station_change_rate"])
        metrics["team_change_rate"] = float(stability["team_change_rate"])

        for task_id, base_task in baseline.tasks.items():
            if base_task.start > start_time + 1e-9:
                continue
            assigned = assigned_by_task.get(int(task_id))
            if assigned is None:
                metrics["frozen_violation_count"] += 1.0
                continue
            _tid, sid, team, start, end = assigned
            if (
                int(sid) != int(base_task.station_id)
                or set(int(w) for w in team) != set(base_task.team)
                or abs(float(start) - float(base_task.start)) > 1e-5
                or abs(float(end) - float(base_task.end)) > 1e-5
            ):
                metrics["frozen_violation_count"] += 1.0

    if hasattr(env, "task_material_ready"):
        for task_id, _sid, _team, start, _end in env.assigned_tasks:
            release_time = float(env.task_material_ready[int(task_id)])
            if release_time > float(start) + 1e-5:
                metrics["release_violation_count"] += 1.0

    if hasattr(env, "raw_data") and "precedence_edges" in env.raw_data:
        edges = env.raw_data["precedence_edges"]
        if hasattr(edges, "detach"):
            edges_np = edges.detach().cpu().numpy()
        else:
            edges_np = np.asarray(edges)
        for src, dst in edges_np.T:
            pred = assigned_by_task.get(int(src))
            succ = assigned_by_task.get(int(dst))
            if pred is None or succ is None:
                continue
            if float(pred[4]) > float(succ[3]) + 1e-5:
                metrics["precedence_violation_count"] += 1.0

    intervals_by_worker: dict[int, list[tuple[float, float]]] = {}
    intervals_by_station: dict[int, list[tuple[float, float]]] = {}
    for task_id, sid, team, start, end in env.assigned_tasks:
        duration = float(end) - float(start)
        if duration <= 1e-8:
            continue
        demand = max(1, int(env.task_static_feat[int(task_id), 2].item()))
        if len(team) < demand:
            metrics["demand_violation_count"] += 1.0
        skill_id = int(env.task_static_feat[int(task_id), 1].item())
        for worker_id in team:
            if worker_id < 0 or worker_id >= env.num_workers or env.worker_skill_matrix[int(worker_id), skill_id] < 0.5:
                metrics["skill_violation_count"] += 1.0
            intervals_by_worker.setdefault(int(worker_id), []).append((float(start), float(end)))
        if sid >= 0:
            intervals_by_station.setdefault(int(sid), []).append((float(start), float(end)))

    for intervals in intervals_by_worker.values():
        intervals.sort()
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
            if prev_end > next_start + 1e-5:
                metrics["worker_overlap_violation_count"] += 1.0

    max_slots = int(getattr(configs, "max_slots_per_station", 3))
    for intervals in intervals_by_station.values():
        events: list[tuple[float, int]] = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda item: (item[0], item[1]))
        active = 0
        for _time_point, delta in events:
            active += delta
            if active > max_slots:
                metrics["station_slot_violation_count"] += 1.0
                break

    return metrics


def evaluate_reschedule_model(env, agent, num_runs=4, temperature=None, writer=None, current_ep=0):
    """
    专用于 APAL 预测-反应式重调度的评估。
    只评估工序延迟开始场景，不混入旧的工时噪声、工人扰动和动态事件评估。
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)

    agent.policy.eval()
    backups = {
        'enable_dynamic_events': getattr(configs, 'enable_dynamic_events', False),
        'enable_station_breakdown': getattr(configs, 'enable_station_breakdown', False),
        'enable_material_delay': getattr(configs, 'enable_material_delay', False),
        'enable_online_duration_perturb': getattr(configs, 'enable_online_duration_perturb', False),
        'enable_worker_fatigue': getattr(configs, 'enable_worker_fatigue', False),
        'randomize_durations': getattr(configs, 'randomize_durations', False),
    }
    setattr(configs, 'enable_dynamic_events', False)
    setattr(configs, 'enable_station_breakdown', False)
    setattr(configs, 'enable_material_delay', False)
    setattr(configs, 'enable_online_duration_perturb', False)
    setattr(configs, 'enable_worker_fatigue', False)

    makespans, balances, rewards, durations = [], [], [], []
    worker_utils, station_utils = [], []
    schedules = []
    constraint_rows = []
    score_rows = []
    scenario_path = ensure_reschedule_eval_scenarios_available(configs)
    if scenario_path is None:
        scenario_items = []
    else:
        scenario_items = load_reschedule_scenarios(scenario_path)
    if num_runs is not None:
        scenario_items = scenario_items[: max(1, int(num_runs))]

    try:
        base_seed = int(getattr(configs, "reschedule_eval_scenario_seed", 42))
        for idx, (scenario_id, scenario) in enumerate(scenario_items):
            setattr(env, "_forced_reschedule_scenario", scenario)
            state = env.reset(randomize_duration=False, randomize_workers=False, seed=base_seed + idx)
            done = False
            total_reward = 0.0
            invalid_step_count = 0
            start_wall = time.time()

            for _ in range(env.num_tasks * 3):
                if done:
                    break
                task_mask, station_mask, worker_mask = env.get_masks()
                if task_mask.all():
                    if env.try_wait_for_resources():
                        state = refresh_env_observation(env)
                        continue
                    break

                action_ret = agent.select_action(
                    state.to(agent.device),
                    mask_task=task_mask.to(agent.device),
                    mask_station_matrix=station_mask.to(agent.device),
                    mask_worker=worker_mask.to(agent.device),
                    deterministic=(temperature == 0.0),
                    temperature=temperature,
                    is_eval=True,
                )
                if action_ret[0] is None:
                    break
                action, _, _, _, is_invalid = action_ret
                if getattr(configs, 'ablation_no_mask', False) and is_invalid:
                    break
                state, reward, done, info = env.step(action)
                total_reward += float(reward)
                if info.get("invalid_action", False):
                    invalid_step_count += 1
                    break

            elapsed = time.time() - start_wall
            complete = len(env.assigned_tasks) == env.num_tasks
            if complete:
                final_makespan = float(np.max(env.station_wall_clock))
                balance = float(np.std(env.station_loads))
                worker_util, station_util = _compute_assignment_utilization(env, final_makespan)
                schedule = list(env.assigned_tasks)
            else:
                final_makespan = float(env.ideal_makespan * 3.0)
                balance = float(env.ideal_station_load * 3.0)
                worker_util, station_util = 0.0, 0.0
                schedule = []

            constraints = _compute_reschedule_constraint_metrics(env)
            constraints["scenario_id"] = scenario_id
            constraints["scenario_index"] = float(idx)
            constraints["reschedule_start_time"] = float(scenario.start_time)
            constraints["delayed_task_count"] = float(len(scenario.task_release_times))
            constraints["invalid_step_count"] = float(invalid_step_count)
            constraints["complete"] = float(complete)
            score_result = calculate_reschedule_composite_score(
                makespan=final_makespan,
                balance_std=balance,
                constraint_metrics=constraints,
                config_obj=configs,
                ideal_station_load=float(getattr(env, "ideal_station_load", 1.0)),
            )
            constraints["eligible"] = float(score_result.eligible)
            constraints["composite_score"] = float(score_result.score)
            constraints["selection_score"] = float(score_result.selection_score)
            constraints.update(score_result.terms)
            constraint_rows.append(constraints)
            score_rows.append(score_result)
            makespans.append(final_makespan)
            balances.append(balance)
            rewards.append(total_reward)
            durations.append(elapsed)
            worker_utils.append(worker_util)
            station_utils.append(station_util)
            schedules.append(schedule)
    finally:
        if hasattr(env, "_forced_reschedule_scenario"):
            delattr(env, "_forced_reschedule_scenario")
        for key, value in backups.items():
            setattr(configs, key, value)

    if score_rows:
        best_idx = int(np.argmin([score.selection_score for score in score_rows]))
    else:
        best_idx = int(np.argmin(makespans)) if makespans else 0
    avg_metrics = {
        key: float(np.mean([row[key] for row in constraint_rows])) if constraint_rows else 0.0
        for key in [
            "frozen_violation_count",
            "release_violation_count",
            "precedence_violation_count",
            "worker_overlap_violation_count",
            "station_slot_violation_count",
            "skill_violation_count",
            "demand_violation_count",
            "duplicate_task_count",
            "missing_task_count",
            "invalid_step_count",
            "complete",
            "reschedule_start_time",
            "delayed_task_count",
            "takt_h",
            "takt_violation_h",
            "lower_bound_h",
            "takt_feasible",
            "start_deviation_mean_h",
            "station_change_rate",
            "team_change_rate",
            "eligible",
            "composite_score",
            "selection_score",
            "score_makespan",
            "score_balance",
            "score_takt_violation",
            "score_start_stability",
            "score_station_change",
            "score_team_change",
        ]
    }
    avg_metrics["eligible_rate"] = avg_metrics.get("eligible", 0.0)

    avg_makespan = float(np.mean(makespans))
    avg_balance = float(np.mean(balances))
    avg_reward = float(np.mean(rewards))
    avg_duration = float(np.mean(durations))
    avg_w_util = float(np.mean(worker_utils))
    avg_s_util = float(np.mean(station_utils))
    best_sch = schedules[best_idx] if schedules else []

    if writer is not None:
        writer.add_scalar('RescheduleEval/Makespan', avg_makespan, current_ep)
        writer.add_scalar('RescheduleEval/Reward', avg_reward, current_ep)
        writer.add_scalar('RescheduleEval/WorkerUtil', avg_w_util, current_ep)
        writer.add_scalar('RescheduleEval/StationUtil', avg_s_util, current_ep)
        writer.add_scalar('RescheduleEval/CompositeScore', avg_metrics.get("composite_score", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/EligibleRate', avg_metrics.get("eligible_rate", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_Makespan', avg_metrics.get("score_makespan", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_Balance', avg_metrics.get("score_balance", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_TaktViolation', avg_metrics.get("score_takt_violation", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_StartStability', avg_metrics.get("score_start_stability", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_StationChange', avg_metrics.get("score_station_change", 0.0), current_ep)
        writer.add_scalar('RescheduleEval/Score_TeamChange', avg_metrics.get("score_team_change", 0.0), current_ep)
        for key, value in avg_metrics.items():
            writer.add_scalar(f'RescheduleEval/{key}', value, current_ep)

    evaluate_reschedule_model.last_metrics = avg_metrics
    evaluate_reschedule_model.last_scenario_metrics = constraint_rows
    return avg_makespan, avg_balance, avg_reward, best_sch, avg_duration, avg_w_util, avg_s_util

# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def train(args):
    try:
        from utils.report_generator import TrainingReporter
        reporter = TrainingReporter(log_dir=getattr(configs, 'report_dir', 'results/reports'))
        last_metrics = {}
        
        print("--- 开始训练 (Starting Training) ---")
        dynamic_flags = {
            "enable_dynamic_events": getattr(configs, "enable_dynamic_events", False),
            "enable_station_breakdown": getattr(configs, "enable_station_breakdown", False),
            "enable_material_delay": getattr(configs, "enable_material_delay", False),
            "enable_online_duration_perturb": getattr(configs, "enable_online_duration_perturb", False),
            "enable_worker_fatigue": getattr(configs, "enable_worker_fatigue", False),
            "enable_reschedule_mode": getattr(configs, "enable_reschedule_mode", False),
        }
        print(f"实验名称: {sanitize_experiment_name(getattr(configs, 'experiment_name', 'default'))}")
        print(f"动态扰动开关: {dynamic_flags}")
        
        # 1. 硬件自检与主计算设备公告
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("=" * 70)
        print(f"🖥️  系统硬件自检 | 激活计算设备: {device.type.upper()}")
        if device.type == 'cuda':
            print(f"🎮 显卡型号: {torch.cuda.get_device_name(0)}")
            print(f"🧬 CUDA 版本: {torch.version.cuda}")
            print(f"🎚️ 显存分配配置: 动态分配")
        else:
            print("⚠️ 警告: 未检测到可用 GPU，计算回退至 CPU 运行。")
        print("=" * 70)
        
        # 接管顶层强化学习伪随机核心
        seed_cfg = configs.seed
        set_seed(seed_cfg)
        baseline_path = ensure_reschedule_baseline_available(configs)
        if baseline_path is not None:
            print(f"重调度 baseline: {baseline_path}")
        eval_scenario_path = ensure_reschedule_eval_scenarios_available(configs)
        if eval_scenario_path is not None:
            print(f"固定重调度验证场景: {eval_scenario_path}")
        
        # 1. 初始化环境
        data_path = resolve_workspace_path(configs.data_file_path if configs.data_file_path else Path("data") / "3182.csv")
             
        # [Dataset Pool] 训练环境：直接投喂整个多图混合训练集目录
        train_dir = resolve_workspace_path(getattr(configs, 'train_data_path_or_dir', data_path))
        print(f"训练图纸池 (Dataset Pool): {train_dir}")
        
        num_envs = int(configs.num_envs) # DPPO 并行数量
        import platform
        plat = platform.system()
        start_method = getattr(configs, 'vector_env_start_method', 'auto')
        if start_method == "auto":
            raise RuntimeError("平台硬件配置必须显式指定 vector_env_start_method")
        print(f"初始化 DPPO 向量化环境，并行数量: {num_envs} (平台: {plat}, start_method: {start_method})")
        from utils.vector_env import EnvCreator
        make_env = EnvCreator(str(train_dir), seed_offset=int(configs.seed))
        vec_env = VectorEnv(
            make_env,
            num_envs=num_envs,
            start_method=start_method,
            worker_threads=getattr(configs, "vector_env_worker_threads", "auto"),
            init_timeout_sec=float(getattr(configs, "vector_env_init_timeout_sec", 120.0)),
            command_timeout_sec=float(getattr(configs, "vector_env_command_timeout_sec", 120.0)),
        )
        env = vec_env.envs[0] # 保留一个env引用用于 fallback 和 属性查询
        
        # [Validation] 验证环境：绑定单一的稳定基准图，防止评估基准浮动
        print(f"基准评估图 (Eval Graph): {data_path}")
        eval_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=int(configs.seed))
        multiscale_sampler = None
        current_multiscale_candidate = None
        if getattr(configs, "enable_multiscale_training", False):
            if train_dir.is_dir():
                multiscale_dataset_pool = [
                    {"file_path": str(path.resolve())}
                    for path in sorted(train_dir.iterdir())
                    if path.suffix.lower() in {".csv", ".xlsx"}
                ]
            else:
                multiscale_dataset_pool = [{"file_path": str(train_dir.resolve())}]
            candidates = build_dataset_candidates(
                multiscale_dataset_pool,
                min_ops=int(getattr(configs, "multiscale_min_ops", 200)),
                max_ops=int(getattr(configs, "multiscale_max_ops", 3100)),
                sampling_exponent=float(getattr(configs, "multiscale_sampling_exponent", 0.5)),
                min_updates=int(getattr(configs, "multiscale_min_updates", 600)),
                max_updates=int(getattr(configs, "multiscale_max_updates", 3300)),
            )
            multiscale_sampler = InverseScaleDatasetSampler(candidates, seed=int(getattr(configs, "seed", 42)))
            print("多规模 APAL 训练已启用，候选实例如下：")
            for item in candidates:
                print(
                    f"  idx={item.dataset_idx} ops={item.num_tasks} "
                    f"profile={item.profile.name} batch={item.profile.batch_size} "
                    f"k_epochs={item.profile.k_epochs} weight={item.sampling_weight:.6f} "
                    f"budget={item.scheduled_updates} file={Path(item.file_path).name}"
                )
        print("环境初始化完成.")
        
        # 2. 初始化设备与模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = HBGATPN(configs).to(device)
        
        # [GNN Compile Optimization] 若配置开启，利用 PyTorch 2.0 融合 GAT 算子
        if getattr(configs, 'use_compile', False):
            try:
                import platform
                if platform.system() == 'Windows':
                    print("ℹ️ Windows 环境检测：跳过 torch.compile（需 Linux + Triton）。")
                    print("   图编译将在 Linux 服务器上自动开启。")
                else:
                    model = torch.compile(model, dynamic=True)
                    print("🚀 成功激活 torch.compile 图算子融合！")
            except Exception as e:
                print(f"⚠️ 图编译失败，回退至原生未编译模式。Err: {e}")
                
        print("模型已加载至设备.")
        
        # Init Agent
        # Calculate Total Timesteps for Scheduler
        total_updates = int(configs.max_episodes / configs.update_every_episodes)

        agent = PPOAgent(
            model=model,
            lr=configs.lr,
            gamma=configs.gamma,
            k_epochs=configs.k_epochs,
            eps_clip=configs.eps_clip,
            device=device,
            batch_size=configs.batch_size,
            total_timesteps=total_updates,
            config=configs,
        )

        

        print(f"Agent Initialized. Total Scheduled Updates: {total_updates}")
        if (
            getattr(configs, "enable_reschedule_mode", False)
            and getattr(configs, "reschedule_warm_start", True)
            and not args.resume
        ):
            warm_path = resolve_workspace_path(getattr(configs, "reschedule_baseline_model_path", "checkpoints/initial_schedule/bestmodel/best_model.pth"))
            if warm_path.exists():
                warm_stats = load_warm_start_weights_with_input_expansion(agent.policy, warm_path, device)
                print(f"重调度 warm-start: {warm_path} | {warm_stats}")
            else:
                print(f"警告: 启用重调度 warm-start，但未找到初始模型 {warm_path}")
        
        # 3. 断点续训 (Resume Training)
        start_episode = 1
        checkpoint_paths = resolve_checkpoint_paths(configs)
        model_dir = checkpoint_paths["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_paths["checkpoint_path"]
        print(f"Checkpoint 目录: {model_dir}")
        
        if args.resume and checkpoint_path.exists():
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    agent.policy.load_state_dict(checkpoint['model_state_dict'])
                else: 
                     # Fallback if checkpoint is just a model_state_dict
                    agent.policy.load_state_dict(checkpoint)

                if 'optimizer_state_dict' in checkpoint:
                     try:
                         # 检查当前是否开启了SF，以及检查点中是否含有SF特有的 train_mode 标识
                         current_is_sf = getattr(configs, 'use_schedule_free', False)
                         param_groups = checkpoint['optimizer_state_dict'].get('param_groups', [])
                         is_sf_checkpoint = len(param_groups) > 0 and 'train_mode' in param_groups[0]
                         
                         if current_is_sf and not is_sf_checkpoint:
                             print("⚠️ 检查点中为普通 AdamW，当前开启了 ScheduleFree，已跳过优化器状态加载以防止崩溃。")
                         elif not current_is_sf and is_sf_checkpoint:
                             print("⚠️ 检查点中为 ScheduleFree，当前未开启 SF，跳过优化器状态加载。")
                         else:
                             agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     except Exception as opt_e:
                         print(f"⚠️ 无法恢复优化器状态: {opt_e}")
                if 'optimizer_adam_state_dict' in checkpoint and hasattr(agent, 'optimizer_adam'):
                    agent.optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
                
                if 'ema_model_state_dict' in checkpoint and hasattr(agent, 'ema_policy'):
                    agent.ema_policy.load_state_dict(checkpoint['ema_model_state_dict'])
                
                start_episode = checkpoint.get('episode', 0) + 1 if isinstance(checkpoint, dict) and 'episode' in checkpoint else 1
                print(f"恢复成功. 起始 Episode: {start_episode}")
            except Exception as e:
                print(f"⚠️ 恢复失败: 模型结构不匹配或缺少键值 (可能是 configs 修改了层数/维度). 跳过恢复。\\n报错信息截取: {str(e)[:100]}...")
        
        # 最佳模型记录
        best_makespan = float('inf')
        best_reschedule_score = float('inf')
        best_multi_benchmark_score = float('inf')
        best_model_dir = checkpoint_paths["best_model_dir"]
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_paths["best_model_path"]
        best_model_meta_path = checkpoint_paths["best_model_meta_path"]
        
        # 4. TensorBoard 设置
        run_name = f"{sanitize_experiment_name(getattr(configs, 'experiment_name', 'default'))}_ALB_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_root = resolve_tensorboard_log_root(configs)
        configs.log_dir = str(log_root)
        log_dir = log_root / run_name
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard 日志目录: {log_dir}")
        
        memory = Memory()
        
        # 5. 训练循环参数
        max_episodes = configs.max_episodes 
        update_every_episodes = configs.update_every_episodes
        eval_freq = configs.eval_freq
        dataset_rng = np.random.RandomState(int(configs.seed))
        
        print(f"开始 DPPO Episode 循环 (Max: {max_episodes}, Envs: {num_envs})...")
        use_fast_path = getattr(configs, 'use_rollout_snapshot_fastpath', True)
        use_profiler = getattr(configs, 'enable_rollout_profiler', False)
        print(f"Rollout 加速: fast_snapshot={use_fast_path}, profiler={use_profiler}, "
              f"shadow_mask_verify={getattr(configs, 'enable_shadow_mask_verification', False)}")
        
        # tqdm 负责进度展示；日志继续使用内置 print，避免遮蔽内置函数造成作用域错误。
        from tqdm import tqdm

        pbar = tqdm(
            range(start_episode, configs.max_episodes + 1),
            desc="🚀 APAL Training Progress",
            dynamic_ncols=True,
            unit="ep"
        )

        for ep in pbar:
            
            agent.policy.train()
            current_temp = configs.sample_temperature
            if multiscale_sampler is not None:
                current_multiscale_candidate = multiscale_sampler.sample()
                vec_env.switch_dataset_all(current_multiscale_candidate.dataset_idx)
                apply_scale_profile_to_agent(agent, current_multiscale_candidate.profile, configs)
                writer.add_scalar("Multiscale/NumTasks", current_multiscale_candidate.num_tasks, ep)
                writer.add_scalar("Multiscale/SamplingWeight", current_multiscale_candidate.sampling_weight, ep)
                writer.add_scalar("Multiscale/PPOEpochsPerRollout", agent.k_epochs, ep)
                writer.add_scalar("Multiscale/PPOBatchSize", agent.batch_size, ep)
                print(
                    f"[Multiscale] ep={ep} idx={current_multiscale_candidate.dataset_idx} "
                    f"ops={current_multiscale_candidate.num_tasks} "
                    f"profile={current_multiscale_candidate.profile.name} "
                    f"k_epochs={agent.k_epochs} batch={agent.batch_size}"
                )
            
            curr_curriculum_episodes = configs.curriculum_episodes
            apply_noise = configs.randomize_durations if ep > curr_curriculum_episodes else False
                
            states = vec_env.reset_all(randomize_duration=apply_noise, randomize_workers=apply_noise)
            dones = [False] * num_envs
            ep_rewards = [0.0] * num_envs

            # 为每个环境维护独立的轨迹细节
            ep_makespan_penalties = [0.0] * num_envs
            ep_std_penalties = [0.0] * num_envs
            ep_deadlock_penalties = [0.0] * num_envs
            ep_is_deadlock = [False] * num_envs
            ep_task_completions = [0.0] * num_envs

            # 为每个环境维护独立的轨迹缓冲区
            env_memories = [Memory() for _ in range(num_envs)]

            max_steps = max([e.num_tasks for e in vec_env.envs]) * 2 
            
            prof_mask_cum = prof_select_cum = prof_snapshot_cum = prof_step_cum = 0.0
            prof_deadlock_cum = 0.0
            prof_step_count = 0
            steps_per_sec = 0.0
            apal_diag_sums = {
                "schedulable_tasks": 0.0,
                "avg_worker_wait_h": 0.0,
                "avg_station_wait_h": 0.0,
                "avg_resource_wait_h": 0.0,
                "station_slot_vacancy_ratio": 0.0,
                "worker_idle_ratio": 0.0,
                "critical_start_offset_h": 0.0,
            }
            apal_diag_counts = {key: 0 for key in apal_diag_sums}
            reward_diag_sums = {
                "resource_wait_penalty_candidate": 0.0,
                "resource_idle_penalty_candidate": 0.0,
                "team_wait_h": 0.0,
                "station_wait_h": 0.0,
                "worker_idle_ratio_before": 0.0,
                "station_slot_vacancy_ratio_before": 0.0,
                "reschedule_takt_violation_h": 0.0,
                "reschedule_start_deviation_mean_h": 0.0,
                "reschedule_station_change_rate": 0.0,
                "reschedule_team_change_rate": 0.0,
                "reschedule_stability_penalty": 0.0,
            }
            reward_diag_count = 0
            
            prof_ep_t0 = time.perf_counter() if use_profiler else 0.0
            
            for t in range(max_steps):
                if all(dones):
                    break
                
                prof_mask_t0 = time.perf_counter() if use_profiler else 0.0
                    
                rollout_snapshots = None
                if use_fast_path:
                    masks_list, rollout_snapshots = vec_env.get_masks_and_snapshots_all()
                else:
                    masks_list = vec_env.get_masks_all()
                
                if use_profiler:
                    prof_mask_cum += time.perf_counter() - prof_mask_t0
                active_indices = [i for i, d in enumerate(dones) if not d]
                
                # Check deadlocks
                prof_dead_t0 = time.perf_counter() if use_profiler else 0.0
                for i in active_indices:
                    t_mask, s_mask, w_mask = masks_list[i]
                    if t_mask.all():
                        wait_start_time = float(getattr(vec_env.envs[i], 'current_time', 0.0))
                        wait_diag = None
                        wait_snapshot = None
                        if use_fast_path and rollout_snapshots is not None:
                            wait_snapshot = rollout_snapshots[i]
                        else:
                            wait_snapshot = vec_env.envs[i].get_state_snapshot()
                        try:
                            wait_diag = compute_apal_rollout_diagnostics(vec_env.envs[i], wait_snapshot, masks_list[i])
                        except Exception:
                            wait_diag = None

                        if vec_env.envs[i].try_wait_for_resources():
                            wait_end_time = float(getattr(vec_env.envs[i], 'current_time', wait_start_time))
                            wait_delta_h = max(0.0, wait_end_time - wait_start_time)
                            if wait_delta_h > 0.0:
                                station_wait_before = float(wait_diag.get("avg_station_wait_h", 0.0)) if wait_diag else 0.0
                                worker_wait_before = float(wait_diag.get("avg_worker_wait_h", 0.0)) if wait_diag else 0.0
                                if station_wait_before > 1e-9:
                                    reward_diag_sums["station_wait_h"] += wait_delta_h
                                elif worker_wait_before > 1e-9:
                                    reward_diag_sums["team_wait_h"] += wait_delta_h
                                else:
                                    reward_diag_sums["team_wait_h"] += wait_delta_h
                                reward_diag_count += 1

                            if use_fast_path:
                                masks_list[i], wait_snapshot = vec_env.envs[i].get_rollout_state()
                                states[i] = vec_env.envs[i].rebuild_state_from_snapshot(wait_snapshot)
                                if rollout_snapshots is not None:
                                    rollout_snapshots[i] = wait_snapshot
                            else:
                                states[i] = refresh_env_observation(vec_env.envs[i])
                                masks_list[i] = vec_env.envs[i].get_masks()
                            continue
                        
                        dynamic_penalty = configs.deadlock_penalty_constant
                        reward = -dynamic_penalty * configs.r_coef_makespan * configs.reward_scale 
                        dones[i] = True
                        if len(env_memories[i].rewards) > 0:
                            env_memories[i].rewards[-1] += reward
                            env_memories[i].is_terminals[-1] = True
                        ep_rewards[i] += reward
                        
                        # 记录死锁惩罚与状态
                        ep_deadlock_penalties[i] += reward
                        ep_is_deadlock[i] = True
                if use_profiler:
                    prof_deadlock_cum += time.perf_counter() - prof_dead_t0
                        
                active_indices = [i for i, d in enumerate(dones) if not d]
                if not active_indices:
                    break

                for i in active_indices:
                    diag_snapshot = (
                        rollout_snapshots[i]
                        if use_fast_path and rollout_snapshots is not None
                        else vec_env.envs[i].get_state_snapshot()
                    )
                    apal_diag = compute_apal_rollout_diagnostics(vec_env.envs[i], diag_snapshot, masks_list[i])
                    for key, value in apal_diag.items():
                        if np.isfinite(value):
                            apal_diag_sums[key] += float(value)
                            apal_diag_counts[key] += 1
                
                # Batch Inference for Active Envs
                actions = [None] * num_envs
                
                active_obs = [states[i] for i in active_indices]
                active_mask_task = [masks_list[i][0] for i in active_indices]
                active_mask_station = [masks_list[i][1] for i in active_indices]
                active_mask_worker = [masks_list[i][2] for i in active_indices]
                
                prof_select_t0 = time.perf_counter() if use_profiler else 0.0
                batch_results = select_actions_batch_compat(
                    agent,
                    obs_list=active_obs,
                    mask_task_list=active_mask_task,
                    mask_station_matrix_list=active_mask_station,
                    mask_worker_list=active_mask_worker,
                    deterministic=False,
                    temperature=current_temp,
                    is_eval=False,
                )
                if use_profiler:
                    prof_select_cum += time.perf_counter() - prof_select_t0
                
                if use_profiler:
                    _pst0 = time.perf_counter()
                for idx, i in enumerate(active_indices):
                    action, logprob, val, specific_station_mask, is_invalid = batch_results[idx]
                    t_mask, s_mask, w_mask = masks_list[i]
                    state_snapshot = (
                        rollout_snapshots[i]
                        if use_fast_path and rollout_snapshots is not None
                        else vec_env.envs[i].get_state_snapshot()
                    )
                    
                    if configs.ablation_no_mask and is_invalid:
                         dynamic_penalty = configs.deadlock_penalty_constant
                         reward = -dynamic_penalty * configs.r_coef_makespan * configs.reward_scale 
                         dones[i] = True      
                         
                         env_memories[i].states.append(state_snapshot) 
                         env_memories[i].actions.append(action)
                         lp_tensor = torch.tensor(logprob).to(device) if not isinstance(logprob, torch.Tensor) else logprob
                         env_memories[i].logprobs.append(lp_tensor)
                         env_memories[i].rewards.append(reward)
                         env_memories[i].is_terminals.append(True)
                         env_memories[i].is_truncated.append(False)
                         env_memories[i].masks.append((t_mask, s_mask, w_mask))
                         val_tensor = torch.tensor(val).to(device) if not isinstance(val, torch.Tensor) else val
                         env_memories[i].values.append(val_tensor)
                         ep_rewards[i] += reward
                         actions[i] = None
                          
                         # 记录死锁惩罚与状态 (消融无掩码情况)
                         ep_deadlock_penalties[i] += reward
                         ep_is_deadlock[i] = True
                    else:
                         actions[i] = action
                         
                         env_memories[i].states.append(state_snapshot) 
                         env_memories[i].actions.append(action)
                         lp_tensor = torch.tensor(logprob).to(device) if not isinstance(logprob, torch.Tensor) else logprob
                         env_memories[i].logprobs.append(lp_tensor)
                         env_memories[i].masks.append((t_mask, s_mask, w_mask))
                         val_tensor = torch.tensor(val).to(device) if not isinstance(val, torch.Tensor) else val
                         env_memories[i].values.append(val_tensor)
                if use_profiler:
                    prof_snapshot_cum += time.perf_counter() - _pst0
                
                # Parallel Step
                prof_step_t0 = time.perf_counter() if use_profiler else 0.0
                if use_fast_path:
                    next_snapshots, step_rewards, step_dones, infos = vec_env.step_snapshot_all(actions)
                    next_states = list(states)
                    for i in active_indices:
                        if actions[i] is not None:
                            next_states[i] = vec_env.envs[i].rebuild_state_from_snapshot(next_snapshots[i])
                else:
                    next_states, step_rewards, step_dones, infos = vec_env.step_all(actions)
                if use_profiler:
                    prof_step_cum += time.perf_counter() - prof_step_t0
                    prof_step_count += 1
                
                for i in active_indices:
                    if actions[i] is not None:
                        env_memories[i].rewards.append(step_rewards[i])
                        env_memories[i].is_terminals.append(step_dones[i])
                        env_memories[i].is_truncated.append(False)
                        ep_rewards[i] += step_rewards[i]
                        dones[i] = step_dones[i]
                        states[i] = next_states[i]
                        
                        # 累加奖励细节用于 TensorBoard 拆解监控
                        info = infos[i]
                        ep_makespan_penalties[i] += info.get('makespan_penalty', 0.0)
                        ep_std_penalties[i] += info.get('std_penalty', 0.0)
                        for key in reward_diag_sums:
                            reward_diag_sums[key] += float(info.get(key, 0.0))
                        reward_diag_count += 1
            
            # Episode Summary
            avg_reward = sum(ep_rewards) / num_envs
            
            makespans = []
            for i in range(num_envs):
                ep_makespan = np.max(vec_env.envs[i].station_wall_clock) if len(vec_env.envs[i].assigned_tasks) > 0 else 0.0
                makespans.append(ep_makespan)
                
                # 计算任务完成率 (百分比)
                num_tasks_done = len(vec_env.envs[i].assigned_tasks)
                total_tasks = vec_env.envs[i].num_tasks
                ep_task_completions[i] = (num_tasks_done / total_tasks) * 100.0
                
            avg_makespan = sum(makespans) / num_envs
            
            # 计算 Batch 汇总诊断指标
            deadlock_rate = sum(ep_is_deadlock) / num_envs
            avg_task_completion = sum(ep_task_completions) / num_envs
            avg_makespan_penalty = sum(ep_makespan_penalties) / num_envs
            avg_std_penalty = sum(ep_std_penalties) / num_envs
            avg_deadlock_penalty = sum(ep_deadlock_penalties) / num_envs
                
            writer.add_scalar('Reward/Episode_Avg', avg_reward, ep)
            writer.add_scalar('Train/WallClock_Makespan_Avg', avg_makespan, ep)
            
            # 详细诊断监控指标写入 TensorBoard
            writer.add_scalar('Train/Deadlock_Rate_Batch', deadlock_rate, ep)
            writer.add_scalar('Train/Avg_Task_Completion_Rate', avg_task_completion, ep)
            writer.add_scalar('RewardDetail/Makespan_Penalty_Mean', avg_makespan_penalty, ep)
            writer.add_scalar('RewardDetail/Load_Balance_Penalty_Mean', avg_std_penalty, ep)
            writer.add_scalar('RewardDetail/Deadlock_Penalty_Mean', avg_deadlock_penalty, ep)
            for key, total in apal_diag_sums.items():
                count = apal_diag_counts[key]
                if count > 0:
                    writer.add_scalar(f'APAL/{key}', total / count, ep)
            if reward_diag_count > 0:
                for key, total in reward_diag_sums.items():
                    writer.add_scalar(f'RewardDiagnostic/{key}', total / reward_diag_count, ep)
            
            status_strs = ["DEADLOCK" if len(vec_env.envs[i].assigned_tasks) < vec_env.envs[i].num_tasks else "COMPLETED" for i in range(num_envs)]
            prof_ep_total = (
                time.perf_counter() - prof_ep_t0
                if use_profiler
                else 0.0
            )
            if use_profiler and prof_step_count > 0:
                steps_per_sec = prof_step_count / max(prof_ep_total, 1e-6)

            # 动态更新进度条右侧的性能后缀，避免在大循环内频繁 print 刷屏
            pbar.set_postfix({
                "Rew": f"{avg_reward:.2f}",
                "Mk": f"{avg_makespan:.1f}",
                "DL": f"{deadlock_rate * 100:.0f}%",
                "SPS": f"{steps_per_sec:.1f}" if use_profiler and prof_step_count > 0 else "N/A"
            })
            
            # Rollout Profiler: 累计计时 + 平均 + 吞吐量
            if use_profiler and prof_step_count > 0:
                n = prof_step_count
                mask_ms    = (prof_mask_cum / n) * 1000
                deadlock_ms = (prof_deadlock_cum / n) * 1000
                select_ms  = (prof_select_cum / n) * 1000
                snapshot_ms = (prof_snapshot_cum / n) * 1000
                step_ms    = (prof_step_cum / n) * 1000
                total_per_step_ms = mask_ms + deadlock_ms + select_ms + snapshot_ms + step_ms
                if ep % configs.rollout_profile_interval == 0:
                    writer.add_scalar('Rollout/EpisodeTotal_s', prof_ep_total, ep)
                    writer.add_scalar('Rollout/Mask_ms', mask_ms, ep)
                    writer.add_scalar('Rollout/DeadlockCheck_ms', deadlock_ms, ep)
                    writer.add_scalar('Rollout/Select_ms', select_ms, ep)
                    writer.add_scalar('Rollout/Snapshot_ms', snapshot_ms, ep)
                    writer.add_scalar('Rollout/Step_ms', step_ms, ep)
                    writer.add_scalar('Rollout/TotalPerStep_ms', total_per_step_ms, ep)
                    writer.add_scalar('Rollout/EnvStepsPerSec', steps_per_sec, ep)
                    writer.add_scalar('Rollout/StepsPerEpisode', n, ep)
                    writer.add_scalar('Rollout/FastPathEnabled', 1.0 if use_fast_path else 0.0, ep)
                    if hasattr(torch.cuda, 'memory_allocated'):
                        writer.add_scalar('Rollout/GPUAllocatedGB', torch.cuda.memory_allocated() / 1e9, ep)
                    
                    print(f"  ⏱️ Profiler (ep{ep}): Total={prof_ep_total:.1f}s | "
                          f"Steps={n} | Steps/s={steps_per_sec:.1f} | "
                          f"PerStep: Mask={mask_ms:.1f}ms Deadlock={deadlock_ms:.1f}ms "
                          f"Select={select_ms:.1f}ms Snap={snapshot_ms:.1f}ms Step={step_ms:.1f}ms")
            
            
            # Combine memories
            for i in range(num_envs):
                if env_memories[i].is_terminals and not env_memories[i].is_terminals[-1]:
                    env_memories[i].is_truncated[-1] = True
                memory.states.extend(env_memories[i].states)
                memory.actions.extend(env_memories[i].actions)
                memory.logprobs.extend(env_memories[i].logprobs)
                memory.rewards.extend(env_memories[i].rewards)
                memory.is_terminals.extend(env_memories[i].is_terminals)
                memory.is_truncated.extend(env_memories[i].is_truncated)
                memory.masks.extend(env_memories[i].masks)
                memory.values.extend(env_memories[i].values)
            
            # PPO 更新
            if ep % update_every_episodes == 0:
                try:
                    metrics = agent.update(memory, vec_env.envs[0], current_ep=ep)
                    
                    if not getattr(agent, 'use_schedule_free', getattr(configs, 'use_schedule_free', False)):
                        progress = min(1.0, ep / configs.max_episodes)
                        min_lr = 1e-6
                        current_lr = configs.lr - progress * (configs.lr - min_lr)
                        for param_group in agent.optimizer.param_groups:
                            group_name = param_group.get('name', '')
                            if group_name == 'actor':
                                param_group['lr'] = current_lr * getattr(configs, 'actor_lr_multiplier', 1.0)
                            elif group_name == 'critic':
                                param_group['lr'] = current_lr * getattr(configs, 'critic_lr_multiplier', 1.0)
                            else:
                                param_group['lr'] = current_lr
                            
                    last_metrics = metrics
                    
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, ep)
                            
                except RuntimeError as e:
                    oom_text = str(e).lower()
                    is_cuda_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                        "out of memory" in oom_text
                        and any(token in oom_text for token in ("cuda", "gpu", "device"))
                    )
                    if is_cuda_oom:
                        raise RuntimeError(
                            "PPO 更新发生 CUDA OOM。optimizer 可能已经执行部分 step，"
                            "为保护 on-policy 语义，训练已终止；请降低 batch_size 或环境数。"
                        ) from e
                    raise
                finally:
                    memory.clear()
                    
                # [Dataset Pool] 交替课程学习：按设定的 PPO Update 频率切换图纸
                current_update_count = ep // update_every_episodes
                if multiscale_sampler is None and current_update_count % getattr(configs, 'switch_dataset_every_updates', 1) == 0:
                    if env.dataset_count > 1:
                        if getattr(configs, 'random_sample_dataset', True):
                            next_idx = dataset_rng.randint(0, env.dataset_count)
                        else:
                            next_idx = current_update_count % env.dataset_count
                        vec_env.switch_dataset_all(next_idx)
                        descriptor = env.dataset_pool[next_idx] or {}
                        print(
                            f"      [Narrow Pool] dataset={next_idx + 1}/{env.dataset_count}, "
                            f"file={descriptor.get('file_path', '按需加载')}, nodes={env.num_tasks}"
                        )
                
            # 定期评估与保存
              # [Validation Strategy]
            if ep % configs.eval_freq == 0:
                if getattr(configs, "enable_reschedule_mode", False):
                    makespan, balance, eval_reward, best_sch, eval_duration, w_util, s_util = evaluate_reschedule_model(
                        eval_env,
                        agent,
                        num_runs=max(1, int(getattr(configs, "reschedule_eval_num_scenarios", 4))),
                        temperature=configs.eval_temperature,
                        writer=writer,
                        current_ep=ep,
                    )
                    res_metrics = getattr(evaluate_reschedule_model, "last_metrics", {})
                else:
                    makespan, balance, eval_reward, best_sch, eval_duration, w_util, s_util = evaluate_model(
                        eval_env,
                        agent,
                        num_runs=1,
                        temperature=configs.eval_temperature,
                        writer=writer,
                        current_ep=ep,
                        scenario_names=tuple(configs.eval_scenarios),
                    )
                    res_metrics = {}
                multi_benchmark_result = None
                if (
                    not getattr(configs, "enable_reschedule_mode", False)
                    and getattr(configs, "enable_multi_benchmark_eval", False)
                ):
                    multi_benchmark_result = evaluate_initial_multi_benchmark(
                        agent,
                        config_obj=configs,
                        writer=writer,
                        current_ep=ep,
                    )
                    writer.add_scalar("MultiBenchmark/CompositeScore", multi_benchmark_result.composite_score, ep)
                    writer.add_scalar("MultiBenchmark/Eligible", float(multi_benchmark_result.eligible), ep)
                    print(
                        f"  [MultiBenchmark] score={multi_benchmark_result.composite_score:.6f} "
                        f"eligible={int(multi_benchmark_result.eligible)}"
                    )
                    for row in multi_benchmark_result.rows:
                        print(
                            f"    {row.benchmark_name}: mk={row.makespan:.2f} "
                            f"ref={row.reference_makespan:.2f} norm={row.normalized_score:.4f} "
                            f"complete={int(row.complete)} invalid={row.invalid_step_count}"
                        )
                
                reporter.add_record(ep, makespan, balance, w_util, s_util, best_sch, eval_reward)
                
                print(f"Epoch {ep:04d} [EVAL] | Mk={makespan:.2f} | Bal={balance:.2f} | WUtil={w_util*100:.1f}% | SUtil={s_util*100:.1f}%")
                if res_metrics:
                    scenario_metrics = getattr(evaluate_reschedule_model, "last_scenario_metrics", [])
                    eligible_count = int(round(res_metrics.get("eligible_rate", 0.0) * max(1, len(scenario_metrics))))
                    print(
                        "  [Resched] "
                        f"score={res_metrics.get('composite_score', 0.0):.4f} "
                        f"elig={eligible_count}/{len(scenario_metrics)} "
                        f"takt={res_metrics.get('takt_h', 0.0):.1f} "
                        f"tv={res_metrics.get('takt_violation_h', 0.0):.2f} "
                        f"sd={res_metrics.get('start_deviation_mean_h', 0.0):.2f} "
                        f"sc={res_metrics.get('station_change_rate', 0.0):.3f} "
                        f"tc={res_metrics.get('team_change_rate', 0.0):.3f} "
                        f"terms=({res_metrics.get('score_makespan', 0.0):.3f},"
                        f"{res_metrics.get('score_balance', 0.0):.3f},"
                        f"{res_metrics.get('score_takt_violation', 0.0):.3f},"
                        f"{res_metrics.get('score_start_stability', 0.0):.3f},"
                        f"{res_metrics.get('score_station_change', 0.0):.3f},"
                        f"{res_metrics.get('score_team_change', 0.0):.3f})"
                    )
                    bad_rows = [row for row in scenario_metrics if float(row.get("eligible", 0.0)) < 1.0]
                    for row in bad_rows[:3]:
                        print(
                            "  [BadScenario] "
                            f"{row.get('scenario_id', '?')} "
                            f"score={row.get('composite_score', 0.0):.4f} "
                            f"mk={row.get('makespan', 0.0):.2f} "
                            f"c={row.get('complete', 0.0):.0f} "
                            f"viol=fz{row.get('frozen_violation_count', 0.0):.0f}/"
                            f"rel{row.get('release_violation_count', 0.0):.0f}/"
                            f"pre{row.get('precedence_violation_count', 0.0):.0f}/"
                            f"wo{row.get('worker_overlap_violation_count', 0.0):.0f}/"
                            f"slot{row.get('station_slot_violation_count', 0.0):.0f}/"
                            f"skill{row.get('skill_violation_count', 0.0):.0f}/"
                            f"dem{row.get('demand_violation_count', 0.0):.0f}/"
                            f"dup{row.get('duplicate_task_count', 0.0):.0f}/"
                            f"miss{row.get('missing_task_count', 0.0):.0f}/"
                            f"inv{row.get('invalid_step_count', 0.0):.0f}"
                        )
                
                # 记录 Station Attention Weights
                # 监控 Critic 的注意力分布 (Gaze Variance)
                if configs.use_attention_critic:
                     s_var = getattr(agent.policy, 'last_s_var', 0.0)
                     writer.add_scalar('Critic/Gaze_Variance', s_var, ep)
                     print(f"      -> [Critic Gaze Variance]: {s_var:.6f}")
                
                writer.add_scalar('Eval/WallClock_Makespan', makespan, ep)
                writer.add_scalar('Eval/Workload_Balance_Std', balance, ep)
                writer.add_scalar('Eval/Average_Return', eval_reward, ep)
                writer.add_scalar('Eval/Inference_Time_sec', eval_duration, ep)
                writer.add_scalar('Eval/Worker_Utilization', w_util, ep)
                writer.add_scalar('Eval/Station_Utilization', s_util, ep)
                
                # Save Latest
                save_dict = {
                    'episode': ep,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict()
                }
                from runtime.checkpoints import build_checkpoint_metadata
                save_dict['apal_metadata'] = build_checkpoint_metadata(
                    configs,
                    episode=int(ep),
                )
                if hasattr(agent, 'optimizer_adam'):
                    save_dict['optimizer_adam_state_dict'] = agent.optimizer_adam.state_dict()
                if hasattr(agent, 'ema_policy'):
                    save_dict['ema_model_state_dict'] = agent.ema_policy.state_dict()
                
                torch.save(save_dict, checkpoint_path)
                
                # Save Best
                if getattr(configs, "enable_reschedule_mode", False):
                    current_score = float(res_metrics.get("composite_score", float("inf")))
                    can_save_best = bool(res_metrics.get("eligible_rate", 0.0) >= 1.0 - 1e-9 and current_score < best_reschedule_score)
                    selection_metric = "reschedule_composite_score"
                    score_terms = {
                        key: float(res_metrics.get(key, 0.0))
                        for key in [
                            "score_makespan",
                            "score_balance",
                            "score_takt_violation",
                            "score_start_stability",
                            "score_station_change",
                            "score_team_change",
                        ]
                    }
                    constraint_metrics = res_metrics
                elif multi_benchmark_result is not None:
                    current_score = float(multi_benchmark_result.composite_score)
                    can_save_best = bool(
                        multi_benchmark_result.eligible
                        and current_score < best_multi_benchmark_score
                    )
                    selection_metric = "multi_benchmark_normalized_makespan"
                    score_terms = {
                        row.benchmark_name: float(row.normalized_score)
                        for row in multi_benchmark_result.rows
                    }
                    constraint_metrics = {
                        "eligible": float(multi_benchmark_result.eligible),
                        "composite_score": float(multi_benchmark_result.composite_score),
                        "benchmarks": [
                            {
                                "benchmark_name": row.benchmark_name,
                                "data_path": row.data_path,
                                "makespan": float(row.makespan),
                                "reference_makespan": float(row.reference_makespan),
                                "normalized_score": float(row.normalized_score),
                                "complete": bool(row.complete),
                                "invalid_step_count": int(row.invalid_step_count),
                                "inference_time": float(row.inference_time),
                            }
                            for row in multi_benchmark_result.rows
                        ],
                    }
                    if current_multiscale_candidate is not None:
                        constraint_metrics["training_instance"] = {
                            "dataset_idx": int(current_multiscale_candidate.dataset_idx),
                            "file_path": current_multiscale_candidate.file_path,
                            "num_tasks": int(current_multiscale_candidate.num_tasks),
                            "profile": current_multiscale_candidate.profile.name,
                            "sampling_weight": float(current_multiscale_candidate.sampling_weight),
                            "scheduled_updates": int(current_multiscale_candidate.scheduled_updates),
                        }
                else:
                    current_score = makespan
                    can_save_best = bool(makespan < best_makespan)
                    selection_metric = "eval_makespan"
                    score_terms = {}
                    constraint_metrics = {}

                if can_save_best:
                    best_makespan = makespan
                    if getattr(configs, "enable_reschedule_mode", False):
                        best_reschedule_score = current_score
                    elif multi_benchmark_result is not None:
                        best_multi_benchmark_score = current_score
                    torch.save(
                        {
                            "model_state_dict": agent.policy.state_dict(),
                            "apal_metadata": build_checkpoint_metadata(
                                configs,
                                episode=int(ep),
                                eval_makespan=float(best_makespan),
                            ),
                        },
                        best_model_path,
                    )
                    write_best_model_meta(
                        best_model_meta_path,
                        episode=ep,
                        eval_makespan=best_makespan,
                        selection_metric=selection_metric,
                        best_score=current_score if selection_metric != "eval_makespan" else None,
                        score_terms=score_terms,
                        constraint_metrics=constraint_metrics,
                        config_obj=configs,
                    )
                    if getattr(configs, "enable_reschedule_mode", False):
                        print(
                            "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
                            f"New Best Reschedule Model Saved! Score: {best_reschedule_score:.6f}, Makespan: {best_makespan:.2f}"
                        )
                    elif multi_benchmark_result is not None:
                        print(
                            "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
                            f"New Best Multi-Benchmark Model Saved! Score: {best_multi_benchmark_score:.6f}, Makespan: {best_makespan:.2f}"
                        )
                    else:
                        print(f"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNew Best Model Saved! Makespan: {best_makespan}")
                    
                    # [Real-time Tracer] 实时快照抓拍最好成绩的排单策略
                    trace_dir = model_dir / "eval_traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = trace_dir / "best_schedule_trace.txt"
                    with trace_path.open("w", encoding="utf-8") as f:
                        for (tid, sid, team, start, end) in best_sch:
                            f.write(f"Station: {sid} \t| Task: {tid:04d} \t| Team: {team} \t| Start: {start:.1f} \t| End: {end:.1f}\n")
                            
                    if best_sch:
                        tasks_data = []
                        for (tid, sid, team, start, end) in best_sch:
                             tasks_data.append({
                                 'TaskID': tid,
                                 'StationID': sid + 1,
                                 'Team': str(team),
                                 'Start': start,
                                 'End': end,
                                 'Duration': end - start
                             })
                        import pandas as pd
                        df = pd.DataFrame(tasks_data)
                        df.to_csv(trace_dir / f"Ep_{ep}_Best_Schedule.csv", index=False)
                        from utils.visualization import plot_gantt
                        try:
                            plot_gantt(best_sch, str(trace_dir / f"Ep_{ep}_Gantt.png"))
                            print(f"📸 Real-time Schedule Trace exported to {trace_dir / f'Ep_{ep}_Gantt.png'}")
                        except Exception as e:
                            pass
                        
                # 生成阶段性报告
                if ep > 0 and ep % getattr(configs, 'generate_report_every_episodes', 100) == 0:
                    reporter.generate_report(current_ep=ep, metrics_dict=last_metrics)

        # =======================================================================
        # 6. 训练结束 - 终局性能测评与基线对比 (End of Training Evaluation)
        # =======================================================================
        print("\n" + "="*50)
        print("🎉 强化学习训练循环已结束！开始获取最强方案对比基线。")
        print("="*50)
        
        # 加载最好验证参数
        if best_model_path.exists():
             print(f"加载训练历史上最好的验证模型用于最终推演: {best_model_path}")
             try:
                 from runtime.checkpoints import load_checkpoint, load_policy_weights
                 load_policy_weights(
                     model,
                     load_checkpoint(best_model_path, map_location=device),
                 )
             except RuntimeError as e:
                 print(f"⚠️ 警告: 历史最佳模型 ({best_model_path}) 的结构与当前配置不匹配，无法加载。将继续使用当前最新的训练结果进行推演！")
             
        # 配置 PPO 最终推演
        print("\n>>> [1/2] 开始执行 PPO Agent 的终局推演...")
        # 重新实例环境，避免脏数据
        eval_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=int(configs.seed))
        if getattr(configs, "enable_reschedule_mode", False):
            ppo_makespan, ppo_balance, _, ppo_assigned, ppo_duration, *rest = evaluate_reschedule_model(
                eval_env,
                agent,
                num_runs=max(1, int(getattr(configs, "reschedule_eval_num_scenarios", 4))),
                temperature=configs.eval_temperature,
            )
        else:
            ppo_makespan, ppo_balance, _, ppo_assigned, ppo_duration, *rest = evaluate_model(
                eval_env,
                agent,
                num_runs=1,
                temperature=configs.eval_temperature,
            )

        # 配置 GA 基准对抗
        print("\n>>> [2/2] 开始执行 Genetic Algorithm (GA) 基线推演...")
        ga_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=int(configs.seed))
        ga_scheduler = GeneticAlgorithmScheduler(ga_env, pop_size=30, max_gen=20)
        ga_start = time.time()
        ga_makespan, ga_balance, ga_assigned = ga_scheduler.run()
        ga_duration = time.time() - ga_start
        
        # --- 报表总结生成 ---
        print("\n" + "#"*60)
        print("🚀 终局对比结果报告 (PPO vs GA) 🚀")
        print(f"指标说明：Makespan/Balance (越小越好), 推理耗时 (越快越好)")
        print("-" * 60)
        print(f"| 模型算法类型          | Makespan (h) | Balance Std | 推理耗时 (秒) |")
        print(f"|-----------------------|--------------|-------------|---------------|")
        print(f"| 经典运筹学: (GA 基线) | {ga_makespan:12.2f} | {ga_balance:11.2f} | {ga_duration:13.4f} |")
        print(f"| 强化学习: (HB-GAT-PN) | {ppo_makespan:12.2f} | {ppo_balance:11.2f} | {ppo_duration:13.4f} |")
        print("#"*60 + "\n")
        
        # 导出最佳 PPO 与 GA 细节到各自的文件夹及画图
        output_dir_ppo = resolve_workspace_path(Path("results") / "PPO")
        output_dir_ga = resolve_workspace_path(Path("results") / "GA")
        output_dir_ppo.mkdir(parents=True, exist_ok=True)
        output_dir_ga.mkdir(parents=True, exist_ok=True)
        
        def save_schedule(tasks, prefix_name, target_dir):
            if not tasks: return
            tasks_data = []
            for (tid, sid, team, start, end) in tasks:
                 tasks_data.append({
                     'TaskID': tid,
                     'StationID': sid + 1,
                     'Team': str(team),
                     'Start': start,
                     'End': end,
                     'Duration': end - start
                 })
            df = pd.DataFrame(tasks_data)
            target_path = Path(target_dir)
            df.to_csv(target_path / f"{prefix_name}_schedule.csv", index=False)
            plot_gantt(tasks, str(target_path / f"{prefix_name}_gantt.png"))
            
        print(f"正在向目录 ./results/PPO 与 ./results/GA 保存排程细节与甘特图...")
        save_schedule(ppo_assigned, "PPO_Final", output_dir_ppo)
        save_schedule(ga_assigned, "GA_Baseline", output_dir_ga)
        print("所有流程圆满结束！")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    if sys.platform == "win32":
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
    from args_parser import get_base_parser
    parser = get_base_parser()
    args = parser.parse_args()
    
    # 动态写入 configs 对象，由于各处都会 import configs，可实现全局透传
    # 先保持旧版 argparse 默认值行为，再加载 YAML，最后只让显式命令行参数覆盖 YAML。
    initialize_training_config(args)
    
    if args.trainer == "lightning":
        from train_lightning import run as run_lightning

        run_lightning(args, config_initialized=True)
    else:
        train(args)
