import os
omp_threads = os.environ.get("OMP_NUM_THREADS", "")
if omp_threads:
    try:
        if int(omp_threads) <= 0:
            os.environ["OMP_NUM_THREADS"] = "1"
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"

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
from configs import configs, load_config_files
import pandas as pd
from baselines.heuristic.baseline_ga import GeneticAlgorithmScheduler
from utils.visualization import plot_gantt
import random
from utils.vector_env import VectorEnv, EnvCreator

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_workspace_path(path_like, base_dir: Path = PROJECT_ROOT) -> Path:
    """将配置中的路径解析为跨平台绝对路径；绝对路径保持不变。"""
    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


def sanitize_experiment_name(name: object) -> str:
    """将实验名压缩为安全目录名，避免不同配置的 checkpoint 互相覆盖。"""
    raw = str(name or "default").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe or "default"


def resolve_checkpoint_paths(config_obj=configs) -> dict[str, Path]:
    """按 experiment_name/checkpoint_root 解析当前实验的模型保存路径。"""
    root = resolve_workspace_path(getattr(config_obj, "checkpoint_root", "checkpoints"))
    experiment_name = sanitize_experiment_name(getattr(config_obj, "experiment_name", "default"))
    model_dir = root / experiment_name
    best_model_dir = model_dir / "bestmodel"
    return {
        "model_dir": model_dir,
        "checkpoint_path": model_dir / "latest_checkpoint.pth",
        "best_model_dir": best_model_dir,
        "best_model_path": best_model_dir / "best_model.pth",
        "best_model_meta_path": best_model_dir / "best_model_meta.json",
    }


def write_best_model_meta(
    meta_path: Path,
    *,
    episode: int,
    eval_makespan: float,
    config_obj=configs,
) -> None:
    """保存 best model 的可追溯元数据，方便服务器和本机定位模型来源。"""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode": int(episode),
        "eval_makespan": float(eval_makespan),
        "config_paths": list(getattr(config_obj, "config_paths", ())),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": sanitize_experiment_name(getattr(config_obj, "experiment_name", "default")),
        "data_file_path": getattr(config_obj, "data_file_path", ""),
        "train_data_path_or_dir": getattr(config_obj, "train_data_path_or_dir", ""),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# 设置全局随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    ctx = env_proxy.dataset_pool[dataset_idx]
    is_critical = np.asarray(ctx.get("is_critical", []), dtype=bool)
    if is_critical.size > 0:
        cpm_earliest = _get_cpm_earliest_starts(ctx)
        for task_id, _, _, start_time, _ in snapshot["assigned_tasks"]:
            if 0 <= task_id < len(is_critical) and is_critical[task_id]:
                critical_offset_values.append(max(0.0, float(start_time) - float(cpm_earliest[task_id])))

    return {
        "schedulable_tasks": schedulable_tasks,
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


def evaluate_model(env, agent, num_runs=1, temperature=None, writer=None, current_ep=0):
    """
    使用包含温度平滑的定制定向策略评估当前模型性能。
    在多情景（Standard、工时加噪、工人缺损、动态故障）固定扰动下进行评估，
    保证不同Episode评估考卷的100%一致性，并往 TensorBoard 写入详细分流数据。
    """
    if temperature is None:
        temperature = getattr(configs, 'eval_temperature', 0.0)
        
    # 评估期间必须关闭 Dropout 等机制
    agent.policy.eval()
    
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
            'seed': 20260
        },
        # Scenario 1: Duration Noise (工时固定加噪)
        {
            'name': '1_DurationNoise',
            'rand_dur': True,
            'rand_w': False,
            'dyn_ev': False,
            'seed': 20261
        },
        # Scenario 2: Worker Perturbation (工人固定缺损)
        {
            'name': '2_WorkerNoise',
            'rand_dur': False,
            'rand_w': True,
            'dyn_ev': False,
            'seed': 20262
        },
        # Scenario 3: Dynamic Events (固定事件扰动)
        {
            'name': '3_DynamicEvents',
            'rand_dur': False,
            'rand_w': False,
            'dyn_ev': True,
            'seed': 20263
        }
    ]
    
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
        
        for _ in range(num_runs):
            state = env.reset(randomize_duration=sc['rand_dur'], randomize_workers=sc['rand_w'], seed=sc['seed'])
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
        
        # 1. 初始化环境
        data_path = resolve_workspace_path(configs.data_file_path if configs.data_file_path else Path("data") / "3182.csv")
             
        # [Dataset Pool] 训练环境：直接投喂整个多图混合训练集目录
        train_dir = resolve_workspace_path(getattr(configs, 'train_data_path_or_dir', data_path))
        print(f"训练图纸池 (Dataset Pool): {train_dir}")
        
        num_envs = getattr(configs, 'num_envs', 4) # DPPO 并行数量
        import platform
        plat = platform.system()
        if plat == "Linux":
            num_envs = configs.num_envs_linux
        elif plat == "Windows":
            num_envs = configs.num_envs_windows
        start_method = getattr(configs, 'vector_env_start_method', 'auto')
        if start_method == "auto":
            start_method = "forkserver" if plat == "Linux" else "spawn"
        print(f"初始化 DPPO 向量化环境，并行数量: {num_envs} (平台: {plat}, start_method: {start_method})")
        from utils.vector_env import EnvCreator
        make_env = EnvCreator(str(train_dir), seed_offset=42)
        vec_env = VectorEnv(make_env, num_envs=num_envs, start_method=start_method)
        env = vec_env.envs[0] # 保留一个env引用用于 fallback 和 属性查询
        
        # [Validation] 验证环境：绑定单一的稳定基准图，防止评估基准浮动
        print(f"基准评估图 (Eval Graph): {data_path}")
        eval_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=2026)
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
            total_timesteps=total_updates
        )

        

        print(f"Agent Initialized. Total Scheduled Updates: {total_updates}")
        
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
        best_model_dir = checkpoint_paths["best_model_dir"]
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_paths["best_model_path"]
        best_model_meta_path = checkpoint_paths["best_model_meta_path"]
        
        # 4. TensorBoard 设置
        run_name = f"{sanitize_experiment_name(getattr(configs, 'experiment_name', 'default'))}_ALB_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = resolve_workspace_path(configs.log_dir) / run_name
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard 日志目录: {log_dir}")
        
        memory = Memory()
        
        # 5. 训练循环参数
        max_episodes = configs.max_episodes 
        update_every_episodes = configs.update_every_episodes
        eval_freq = configs.eval_freq
        
        print(f"开始 DPPO Episode 循环 (Max: {max_episodes}, Envs: {num_envs})...")
        use_fast_path = getattr(configs, 'use_rollout_snapshot_fastpath', True)
        use_profiler = getattr(configs, 'enable_rollout_profiler', False)
        print(f"Rollout 加速: fast_snapshot={use_fast_path}, profiler={use_profiler}, "
              f"shadow_mask_verify={getattr(configs, 'enable_shadow_mask_verification', False)}")
        
        for ep in range(start_episode, configs.max_episodes + 1):
            
            agent.policy.train()
            current_temp = configs.sample_temperature
            
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
            apal_diag_sums = {
                "schedulable_tasks": 0.0,
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
                        if vec_env.envs[i].try_wait_for_resources():
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
            print(f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Avg Makespan: {avg_makespan:.1f} | Statuses: {status_strs}")
            
            # Rollout Profiler: 累计计时 + 平均 + 吞吐量
            if use_profiler and prof_step_count > 0:
                prof_ep_total = time.perf_counter() - prof_ep_t0
                n = prof_step_count
                mask_ms    = (prof_mask_cum / n) * 1000
                deadlock_ms = (prof_deadlock_cum / n) * 1000
                select_ms  = (prof_select_cum / n) * 1000
                snapshot_ms = (prof_snapshot_cum / n) * 1000
                step_ms    = (prof_step_cum / n) * 1000
                total_per_step_ms = mask_ms + deadlock_ms + select_ms + snapshot_ms + step_ms
                steps_per_sec = n / max(prof_ep_total, 1e-6)
                
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
                    
                    if not getattr(configs, 'use_schedule_free', False):
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
                    if "out of memory" in str(e) or "OOM" in str(e):
                        print(f"\n⚠️ [OOM 防护] 显存不足，自动清理缓存并跳过本轮更新 (Episode {ep})")
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
                finally:
                    memory.clear()
                    
                # [Dataset Pool] 交替课程学习：按设定的 PPO Update 频率切换图纸
                current_update_count = ep // update_every_episodes
                if current_update_count % getattr(configs, 'switch_dataset_every_updates', 1) == 0:
                    if len(env.dataset_pool) > 1:
                        import random
                        next_idx = random.randint(0, len(env.dataset_pool) - 1)
                        vec_env.switch_dataset_all(next_idx)
                        print(f"      🔄 [Alternating Training] 已切图至: {env.dataset_pool[next_idx]['file_path']} (Nodes: {env.num_tasks})")
                
            # 定期评估与保存
              # [Validation Strategy]
            if ep % configs.eval_freq == 0:
                makespan, balance, eval_reward, best_sch, eval_duration, w_util, s_util = evaluate_model(eval_env, agent, num_runs=1, temperature=configs.eval_temperature, writer=writer, current_ep=ep)
                
                reporter.add_record(ep, makespan, balance, w_util, s_util, best_sch, eval_reward)
                
                print(f"Epoch {ep:04d} [EVAL] | Makespan: {makespan:.2f} \t| Balance Std: {balance:.2f} \t| W-Util: {w_util*100:.1f}% \t| S-Util: {s_util*100:.1f}%")
                
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
                if hasattr(agent, 'optimizer_adam'):
                    save_dict['optimizer_adam_state_dict'] = agent.optimizer_adam.state_dict()
                if hasattr(agent, 'ema_policy'):
                    save_dict['ema_model_state_dict'] = agent.ema_policy.state_dict()
                
                torch.save(save_dict, checkpoint_path)
                
                # Save Best
                if makespan < best_makespan:
                    best_makespan = makespan
                    torch.save(agent.policy.state_dict(), best_model_path)
                    write_best_model_meta(
                        best_model_meta_path,
                        episode=ep,
                        eval_makespan=best_makespan,
                        config_obj=configs,
                    )
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
                 model.load_state_dict(torch.load(best_model_path, map_location=device))
             except RuntimeError as e:
                 print(f"⚠️ 警告: 历史最佳模型 ({best_model_path}) 的结构与当前配置不匹配，无法加载。将继续使用当前最新的训练结果进行推演！")
             
        # 配置 PPO 最终推演
        print("\n>>> [1/2] 开始执行 PPO Agent 的终局推演...")
        # 重新实例环境，避免脏数据
        eval_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=2026)
        ppo_makespan, ppo_balance, _, ppo_assigned, ppo_duration, *rest = evaluate_model(eval_env, agent, num_runs=1, temperature=configs.eval_temperature)
        
        # 配置 GA 基准对抗
        print("\n>>> [2/2] 开始执行 Genetic Algorithm (GA) 基线推演...")
        ga_env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=2026)
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
    setattr(configs, 'ablation_no_gat', args.ablation_no_gat)
    setattr(configs, 'ablation_no_pointer', args.ablation_no_pointer)
    setattr(configs, 'ablation_no_mask', args.ablation_no_mask)
    setattr(configs, 'data_file_path', args.data_path)
    setattr(configs, 'seed', args.seed)
    setattr(configs, 'max_episodes', args.max_episodes)

    if args.config:
        load_config_files(args.config, configs)
        setattr(configs, 'config_paths', tuple(args.config))
        argv = set(sys.argv[1:])
        if '--ablation_no_gat' in argv:
            setattr(configs, 'ablation_no_gat', True)
        if '--ablation_no_pointer' in argv:
            setattr(configs, 'ablation_no_pointer', True)
        if '--ablation_no_mask' in argv:
            setattr(configs, 'ablation_no_mask', True)
        if '--data_path' in argv:
            setattr(configs, 'data_file_path', args.data_path)
        if '--seed' in argv:
            setattr(configs, 'seed', args.seed)
        if '--max_episodes' in argv:
            setattr(configs, 'max_episodes', args.max_episodes)
    
    train(args)
