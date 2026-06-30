
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import platform

@dataclass
class Config:
    # ------------------
    # 路径配置 (Paths)
    # ------------------
    data_dir: str = "data"
    data_file_path: str = str(Path("data") / "283.csv") # 默认验证集基准图
    worker_pool_path: str = str(Path("data") / "worker_pool_fixed.csv")
    
    # ------------------
    # 环境与图相关 (Environment & Graph)
    # ------------------
    n_m: int = 5                         # 站位数量 (Stations)
    n_w_max: int = 100                   # 工人池总上限 (最大可配置的工人数量，固定池容量)
    n_w_min: int = 60                    # 每回合训练随机抽取的最小工人数 (Domain Randomization)
    n_w: int = 80                        # 每回合训练抽取的最大工人数，及验证(Eval)阶段固定的工人数
    max_slots_per_station: int = 3       # 每站位同时执行的最大工序数（物理工位槽）
    
    # ------------------
    # 模型超参数 (Model Hyperparameters)
    # ------------------
    hidden_dim: int = 128                # 隐藏层维度 (Embedding Size)
    num_gat_layers: int = 5              # GAT 层数 (Message Passing Depth)
    num_heads: int = 4                   # 多头注意力头数 (Attention Heads)
    use_leaky_relu: bool = True          # [新增] 是否在深层网络和Critic中使用 LeakyReLU 以防止梯度消失/ReLU死亡
    use_layer_norm: bool = False          # 兼容旧配置：不再直接控制所有 LayerNorm
    use_input_layer_norm: bool = True     # 输入嵌入层保持 LayerNorm，稳定异构原始特征尺度
    use_gat_layer_norm: bool = False      # GAT 消息传递层默认关闭 LayerNorm，保留 APAL 绝对时间/负载尺度
    use_head_layer_norm: bool = False     # 策略头与价值头 MLP 默认关闭 LayerNorm，避免改变旧版 head 输出尺度
    
    task_feat_dim: int = 18              # Task Node Input Features (17 -> 18, 新增物料等待时间)
    worker_feat_dim: int = 22            # Worker Node Input Features (21 -> 22, 新增疲劳系数)
    station_feat_dim: int = 15           # Station Node Input Features
    use_skill_hub: bool = True           # 是否用 Skill Hub 替代稠密 Worker->Task 技能边
    skill_hub_bidirectional: bool = True # Skill Hub 是否增加 Task->Skill->Worker 反向消息
    num_skill_types: int = 10            # APAL 工种数量
    skill_feat_dim: int = 16             # 技能 one-hot 10 维 + 六项资源统计
    
    # ------------------
    # 泛化性与域随机化 (Domain Randomization)
    # ------------------
    train_data_path_or_dir: str = "data/generated/initial_283"
    switch_dataset_every_updates: int = 1                 # 频繁切换以增强泛化能力
    random_sample_dataset: bool = True                   # 是否启用随机抽取数据集图纸，默认开启，可配置为 false 恢复顺序轮询
    dataset_context_cache_size: int = 2                   # 每个环境最多缓存的完整图上下文数量
    enable_multiscale_training: bool = False              # 是否启用多规模 APAL 反规模采样训练
    multiscale_min_ops: int = 200                         # 多规模训练纳入的最小工序数
    multiscale_max_ops: int = 3100                        # 多规模训练纳入的最大工序数
    multiscale_sampling_exponent: float = 0.5             # 反规模采样指数，0.5 对应 1/sqrt(n)
    multiscale_min_updates: int = 600                     # 次线性更新预算下界，仅用于调度记录
    multiscale_max_updates: int = 3300                    # 次线性更新预算上界，仅用于调度记录
    enable_multi_benchmark_eval: bool = False             # 是否用四基准归一化评分选择 best model
    multi_benchmark_data_paths: list[str] = field(default_factory=lambda: [
        "data/283.csv",
        "data/680.csv",
        "data/2338.csv",
        "data/3182.csv",
    ])
    multi_benchmark_reference_makespans: dict[str, float] = field(default_factory=dict)
    randomize_durations: bool = True                      # 开启随机工时扰动
    dur_random_range: float = 0.2                         # 扰动幅度
    curriculum_episodes: int = 0        # 训练前 N 轮强制关闭所有随机因子
    
    # ------------------
    # 动态事件 (Dynamic Events)
    # ------------------
    enable_dynamic_events: bool = True     # 是否在训练期间开启突发动态事件（域随机化的一部分）
    
    # ① 工人缺勤
    prob_worker_absent_base: float = 0.0   # 工人缺勤的基础概率（验证和推理时的默认值）
    prob_worker_absent_max: float = 0.15   # 训练时最大随机波动的缺勤概率
    absence_duration_min: float = 10.0      # 缺勤的最短时间 (小时)
    absence_duration_max: float = 50.0     # 缺勤的最长时间 (小时)
    
    # ② 工位故障与恢复
    enable_station_breakdown: bool = False  # 默认关闭以防干扰已有训练
    prob_station_breakdown_base: float = 0.0 # 每站故障基础概率 (eval时)
    prob_station_breakdown_max: float = 0.10 # 训练时每站最大概率
    breakdown_duration_min: float = 5.0     # 最短故障时间 (小时)
    breakdown_duration_max: float = 30.0    # 最长故障时间 (小时)
    breakdown_lost_slots_min: int = 1       # 最少损失槽位数
    breakdown_lost_slots_max: int = 5       # 最多损失槽位数

    # ③ 工时在线随机扰动
    enable_online_duration_perturb: bool = False # 默认关闭
    online_perturb_prob_per_step: float = 0.02   # 每步触发扰动的概率

    # ④ 物料延迟到达
    enable_material_delay: bool = False     # 默认关闭
    prob_material_delay_base: float = 0.0   # 物料延迟基础概率 (eval时)
    prob_material_delay_max: float = 0.10   # 训练时最大概率
    material_delay_min: float = 5.0         # 最短延迟 (小时)
    material_delay_max: float = 40.0        # 最长延迟 (小时)

    # ------------------
    # 预测-反应式重调度 (Baseline-guided Rescheduling)
    # ------------------
    enable_reschedule_mode: bool = False
    reschedule_baseline_schedule_path: str = "results/final_schedule.csv"
    reschedule_baseline_model_path: str = "checkpoints/initial_schedule/bestmodel/best_model.pth"
    reschedule_scenario_path: str = ""
    reschedule_eval_scenario_path: str = "results/reschedule_eval_scenarios.csv"
    reschedule_eval_scenario_seed: int = 42
    reschedule_start_time_min_ratio: float = 0.15
    reschedule_start_time_max_ratio: float = 0.65
    reschedule_delay_task_prob: float = 0.08
    reschedule_delay_min: float = 5.0
    reschedule_delay_max: float = 30.0
    reschedule_takt_tolerance: float = 1e-5
    reschedule_stability_start_weight: float = 0.20
    reschedule_stability_station_weight: float = 0.10
    reschedule_stability_team_weight: float = 0.05
    reschedule_takt_violation_weight: float = 1.0
    reschedule_infeasible_stability_relax: float = 0.35
    reschedule_warm_start: bool = True
    reschedule_eval_num_scenarios: int = 4

    # ⑤ 工人疲劳衰减与恢复
    enable_worker_fatigue: bool = False     # 默认关闭
    fatigue_threshold_hours: float = 4.0    # 疲劳阈值 (连续工作N小时后效率开始衰减)
    fatigue_decay_slope: float = 0.05       # 每超阈值1小时效率下降比例
    fatigue_efficiency_floor: float = 0.60  # 效率下限
    fatigue_recovery_ratio: float = 0.5     # 空闲恢复折算率 (空闲1h ≈ 恢复0.5h疲劳)
    # ------------------
    # PPO 训练超参数 (PPO Training)
    # ------------------
    num_envs: int = 4                      # DPPO 并行环境数量
    lr: float = 5e-5                       # 初始学习率
    actor_lr_multiplier: float = 0.5       # Actor 参数学习率倍率，用于放缓策略塌缩
    critic_lr_multiplier: float = 1.0      # Critic 参数学习率倍率，保持价值函数跟踪速度
    gamma: float = 0.999                   # 折扣因子
    k_epochs: int = 2                      # 每次更新循环次数
    eps_clip: float = 0.2                  # PPO Clip阈值
    eps_clip_end: float = 0.10             # PPO Clip 衰减下界，避免后期过早收窄到 0.05
    clip_v_grad_norm: float = 0.05          # 保护 Value Network 梯度的防破甲护盾
    batch_size: int = 32                    # 严防爆显存
    ppo_batch_size_cap: int = 0             # 0 表示不限制；平台配置可设置显存安全上限
    auto_oom_retry: bool = True             # CUDA OOM 后自动降低 PPO batch 重试
    skip_update_on_oom: bool = True         # 重试耗尽后回滚并跳过当前 PPO 更新
    oom_min_batch_size: int = 2             # OOM 自动降级的最小 PPO batch
    oom_max_retries: int = 1                # 首次 OOM 后仅将 batch 减半重试一次
    oom_transactional_updates: bool = True  # 更新前保存 CPU 快照，保证回滚语义完整
    r_coef_std: float = 0.5                # 解决坍缩效应
    
    estimated_cmax_station_slots: float = 3.0 
    
    c_policy: float = 1.0                  # Policy Loss 权重
    c_value: float = 0.5                   # Critic 价值损失权重
    
    r_coef_makespan: float = 1.0           # 宏观目标：Makespan 下班时间推移惩罚
    deadlock_penalty_multiplier: float = 1.5 # 死锁惩罚项 (相对于理想总完工时间的倍数)
    deadlock_penalty_constant: float = 50.0  # 恒定且温和的死锁罚分常数，不随图纸大小发生海啸式抖动
    reward_scale: float = 0.005            # 全局奖励缩放乘数
    enable_resource_wait_penalty: bool = False  # 默认只记录等待/空闲诊断，不直接改变奖励
    r_coef_wait: float = 0.05               # 等待惩罚候选系数：团队/站位等待时间 / 平均工时
    r_coef_idle: float = 0.02               # 空闲惩罚候选系数：可执行任务存在时的工人空闲率
    
    use_dense_progress_reward: bool = False  # 是否启用密集进度引导奖励
    r_coef_progress: float = 2.0            # 密集进度奖励的系数
    enable_cpm_reward: bool = False          # 是否启用关键路径CPM重塑奖励
    enable_worker_queue_mask: bool = False   # 是否启用限制工人排队拥堵的动作掩码
    max_worker_queue_ratio: float = 10.0     # 允许工人最大排队时间比例（排队时间 / 平均任务工时）
    use_shared_trunk: bool = False           # 是否开启共享 Actor/Critic 图表征网络骨干
    use_gradient_checkpointing: bool = False  # 是否开启图注意力编码层梯度检查点
    enable_shadow_mask_verification: bool = False  # [Rollout加速] 训练默认关闭影子比对；测试中临时覆写为 True
    use_compile: bool = False                # 是否开启图编译加速 (torch.compile, 需 Linux+Triton)
    
    # ------------------
    # Rollout 前向采样加速 (Rollout Fast Path)
    # ------------------
    use_rollout_snapshot_fastpath: bool = True  # 是否启用轻量 snapshot + 主进程本地 rebuild 替代跨进程传输完整 HeteroData
    enable_rollout_profiler: bool = True        # 是否开启 Rollout 计时分析器
    rollout_profile_interval: int = 10           # 每 N 个 episode 记录一次性能指标到 TensorBoard
    rollout_profile_cuda_sync: bool = False      # Profiler 中是否强制 CUDA 同步计时（训练时关闭以免拖慢速度）
    rollout_heartbeat_interval_sec: float = 0.0 # 0 表示关闭；需要诊断长时间阶段时再开启
    rollout_max_steps: int = 0 # 仅用于受控冒烟测试；0 表示不截断
    
    c_entropy: float = 0.0002                
    c_entropy_end: float = 0.00005            
    entropy_decay_episodes: int = 300   
    accumulation_steps: int = 16       
    gae_lambda: float = 0.9               
    
    max_episodes: int = 300             
    update_every_episodes: int = 1
    eval_freq: int = 1
    eval_temperature: float = 0.0         
    sample_temperature: float = 1.0        
    
    use_schedule_free: bool = True        # [核心开关] 开启 ScheduleFree 优化器防退化
    sf_warmup_steps: int = 30            # ScheduleFree 的预热步数 (建议为总更新次数的 5%-10%)
    use_ema: bool = False                 # 使用 ScheduleFree 时必须关闭传统的 EMA
    ema_decay: float = 0.995               
    
    enable_gpu_batch_rebuild: bool = True  # 是否启用 GPU 预分配 Batch 大图原地特征更新
    enable_gpu_tensor_masking: bool = True # 是否启用 GPU 张量化动作掩码并行计算
    
    kl_early_stop: float = 0.02            
    
    use_autoregressive_worker: bool = True  
    use_attention_critic: bool = True       
    
    ablation_no_mask: bool = False          
    ablation_no_gat: bool = False           
    ablation_no_pointer: bool = False       
    seed: int = 42                         
    
    # ------------------
    # 平台并行策略 (Platform Parallelism)
    # ------------------
    num_envs_linux: int = 16                 # Linux 服务器默认并行环境数
    num_envs_windows: int = 2               # Windows 本机默认并行环境数（低显存）
    vector_env_start_method: str = "auto"   # 多进程启动方式: "spawn" / "forkserver"
    vector_env_worker_threads: str = "auto"
    vector_env_init_timeout_sec: float = 120.0
    vector_env_command_timeout_sec: float = 120.0

    # ------------------
    # Lightning 训练编排
    # ------------------
    use_lightning: bool = True
    lightning_precision: str = "16-mixed"
    lightning_accelerator: str = "auto"
    lightning_devices: int = 1
    float32_matmul_precision: str = "high"
    eval_scenarios: list[str] = field(default_factory=lambda: ["standard"])
    verbose_eval_progress: bool = False

    # ------------------
    # 日志与监控 (Logging)
    # ------------------
    log_dir: str = "/root/tf-logs"
    experiment_name: str = "default"
    checkpoint_root: str = "checkpoints"
    config_paths: tuple[str, ...] = field(default_factory=tuple)
    
    # 报告生成 (Report Generation)
    generate_report_every_episodes: int = 100
    report_dir: str = "results/reports"
    result_dir: str = "results"
    
    def update_from_dict(self, kwargs: dict):
        """支持通过字典动态更新配置"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_flat_dict(self) -> dict:
        """导出当前扁平配置，便于调试和测试。"""
        return asdict(self)


def _flatten_config_tree(data: Mapping[str, Any], prefix: str = "") -> dict:
    """将分层 YAML 配置压平为当前 Config 兼容的键值对。"""
    flat = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat.update(_flatten_config_tree(value, prefix=f"{prefix}{key}."))
        else:
            flat[key] = value
    return flat


def _parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"').strip("'")


def _simple_yaml_load(text: str) -> dict:
    """最小 YAML 读取器，仅覆盖本项目分层配置使用的映射和列表语法。"""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, tuple[dict, str]] = {}

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        if content.startswith("- "):
            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            item = _parse_yaml_scalar(content[2:])
            if not isinstance(parent, list):
                container, key = last_key_at_indent[indent]
                new_list: list[Any] = []
                container[key] = new_list
                stack.append((indent, new_list))
                parent = new_list
            parent.append(item)
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        key, _, raw_value = content.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            assert isinstance(parent, dict)
            parent[key] = _parse_yaml_scalar(raw_value)
        else:
            assert isinstance(parent, dict)
            child: dict[str, Any] = {}
            parent[key] = child
            last_key_at_indent[indent + 2] = (parent, key)
            stack.append((indent, child))

    return root


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并映射，仅覆盖后加载配置明确给出的叶子字段。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mapping(current, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path, yaml_module: Any, visited: set[Path]) -> dict:
    """加载单个 YAML，并递归展开 defaults 列表。"""
    resolved = path.resolve()
    if resolved in visited:
        raise ValueError(f"检测到循环配置引用: {path}")
    visited.add(resolved)

    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    loaded = yaml_module.safe_load(text) if yaml_module is not None else _simple_yaml_load(text)
    loaded = loaded or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置文件顶层必须是映射结构: {path}")

    merged = {}
    defaults = loaded.get("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ValueError(f"defaults 必须是列表: {path}")
    for item in defaults:
        default_path = path.parent / str(item)
        merged = _deep_merge_mapping(
            merged,
            _load_yaml_mapping(default_path, yaml_module, visited),
        )

    body = {key: value for key, value in loaded.items() if key != "defaults"}
    merged = _deep_merge_mapping(merged, body)
    visited.remove(resolved)
    return merged


def load_config_files(paths: list[str] | tuple[str, ...], target: Config | None = None, strict: bool = True) -> Config:
    """
    加载一个或多个分层 YAML 配置文件，并写回当前 Config 单例。

    后加载的文件覆盖先加载的文件；YAML 的层级只用于组织，叶子键必须对应 Config 字段。
    """
    cfg = target if target is not None else configs
    if not paths:
        return cfg

    try:
        import yaml
    except ImportError:
        yaml = None

    merged: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        loaded = _load_yaml_mapping(path, yaml, set())
        merged = _deep_merge_mapping(merged, loaded)

    flat = _flatten_config_tree(merged)
    unknown = [key for key in flat if not hasattr(cfg, key)]
    if unknown and strict:
        raise KeyError(f"配置文件包含未知字段: {unknown}")
    cfg.update_from_dict({key: value for key, value in flat.items() if hasattr(cfg, key)})
    return cfg


def resolve_platform_hardware_config(
    system_name: str | None = None,
    project_root: Path | None = None,
) -> Path:
    """返回当前操作系统唯一对应的硬件配置。"""
    system = system_name or platform.system()
    root = project_root or Path(__file__).resolve().parent
    if system == "Windows":
        return root / "conf" / "hardware" / "windows_4060_low_memory.yaml"
    if system == "Linux":
        return root / "conf" / "hardware" / "linux_server.yaml"
    raise RuntimeError(f"不支持的训练平台: {system!r}；仅支持 Windows 和 Linux")


def load_training_config(
    experiment_paths: list[str] | tuple[str, ...],
    target: Config | None = None,
    *,
    system_name: str | None = None,
) -> tuple[Config, tuple[str, ...]]:
    """先加载实验配置，再追加当前平台硬件配置。"""
    cfg = target if target is not None else configs
    hardware_path = resolve_platform_hardware_config(system_name=system_name)
    paths = tuple(str(path) for path in experiment_paths) + (str(hardware_path),)
    load_config_files(paths, target=cfg)
    cfg.config_paths = paths
    return cfg, paths

# 全局单例实例化，保持向下兼容
configs = Config()
