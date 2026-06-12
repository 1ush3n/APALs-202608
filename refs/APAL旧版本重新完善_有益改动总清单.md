# APAL 项目从旧版本重新完善：有益改动总清单

## 1. 文档目的

本文档用于从 `docs` 中的旧版 APAL 调度代码重新开始开发时，选择性移植当前版本中已经证明有价值的改动。

核心原则：

1. 不整体复制当前版本，也不整体回退某个 Git commit。
2. 优先恢复旧版中已经取得较好 makespan 的稳定调度语义。
3. 分阶段移植正确性修复、性能优化、配置架构、验证体系和重调度功能。
4. 每移植一个独立功能，立即运行对应测试，避免一次性引入多个不可定位的变量。
5. 不重新引入已经证明会增加复杂度或破坏性能的跨尺度混训试验。

本文所称“已完成”，表示当前版本存在对应实现和测试，不表示该实现可以不经审查地直接复制到旧版。

---

## 2. 建议保留的总体架构

推荐保留以下主链路：

```text
分层 YAML 配置
    ↓
APAL 环境与硬约束动作掩码
    ↓
VectorEnv 并行采样
    ↓
轻量 snapshot IPC
    ↓
主进程批量动作选择
    ↓
GPU 原地重建图批次
    ↓
AMP PPO 更新
    ↓
固定验证场景与合法性检查
    ↓
按实验隔离保存 checkpoint、best model 和 TensorBoard
```

当前版本仍通过 `configs.py` 全局单例承接 YAML，因此它只是“分层 YAML 过渡架构”，并未完成彻底的依赖注入。重新完善时应保留 YAML 文件组织方式，但逐步让环境、模型、Agent 和 VectorEnv 显式接收配置对象。

---

## 3. 数据与 APAL 图建模

### 3.1 数据集名称校正

四个数据文件的新旧映射如下：

| 旧名称 | 当前名称 | 说明 |
|---|---|---|
| `290.csv` | `283.csv` | 文件内容一致，仅名称校正 |
| `715.csv` | `680.csv` | 文件内容一致，仅名称校正 |
| `2402.csv` | `2338.csv` | 文件内容一致，仅名称校正 |
| `3000.csv` | `3182.csv` | 文件内容一致，仅名称校正 |

所有默认路径、测试和命令应统一使用当前名称。不要同时保留两套文件名映射逻辑。

### 3.2 数据加载器正确性

应移植的修复：

- 支持 `Sub` 等显式紧前关系描述，不能把所有任务错误串联成单链。
- 正确构建允许并行分支的 DAG。
- 检查负工时、非法前驱、自环、越界 TaskID 和循环依赖。
- 保证 PyG `edge_index` 索引范围合法。
- 数据加载器输出的 `HeteroData` 在组成 batch 后必须具备各节点类型的 `batch` 向量。

这部分直接影响关键路径、可调度任务集合和最终 makespan，优先级高于网络调参。

验收测试：

```bash
python -m pytest tests/test_data_loader.py tests/test_engine_and_mask.py -q
```

---

## 4. 配置架构

### 4.1 分层 YAML

当前有价值的配置分层如下：

```text
conf/
├── model/
├── env/
├── reward/
├── train/
├── rollout/
├── hardware/
└── experiment/
```

实验配置只负责组合其他 YAML，例如：

```yaml
defaults:
  - ../model/hb_gat_pn.yaml
  - ../env/apal_default.yaml
  - ../env/initial_schedule.yaml
  - ../reward/default.yaml
  - ../train/ppo_default.yaml
  - ../rollout/fastpath.yaml
  - ../hardware/linux_server.yaml
```

推荐继续保持以下职责边界：

- `model`：网络维度、层数、LayerNorm、注意力结构。
- `env`：APAL 资源、数据路径、动态事件和重调度语义。
- `reward`：奖励项和综合评分权重。
- `train`：PPO 超参数、更新频率和 batch。
- `rollout`：fast snapshot、GPU rebuild、profiler。
- `hardware`：Windows/Linux 环境数、启动方式和线程数。
- `experiment`：组合配置、实验名称和输出隔离。

### 4.2 YAML 深度合并

这是必须保留的修复。

错误实现使用顶层 `dict.update()`。当后加载的 `batch_64.yaml` 只有：

```yaml
train:
  batch_size: 64
```

它会替换整个 `train` 节点，使 `update_every_episodes`、`eval_freq` 等字段丢失并回退到 Python 默认值。

正确实现必须递归合并嵌套映射，只覆盖叶子字段：

```text
ppo_default.train.update_every_episodes = 1
ppo_default.train.eval_freq = 1
batch_64.train.batch_size = 64

最终结果：
batch_size = 64
update_every_episodes = 1
eval_freq = 1
```

训练启动时应打印最终生效值：

```text
训练参数: batch_size=64, update_every_episodes=1, eval_freq=1
```

验收测试必须覆盖“同一嵌套节点中局部覆盖后，其他字段仍保留”。

### 4.3 实验输出隔离

应使用：

```yaml
experiment:
  experiment_name: initial_schedule
  checkpoint_root: checkpoints
```

输出结构：

```text
checkpoints/{experiment_name}/latest_checkpoint.pth
checkpoints/{experiment_name}/bestmodel/best_model.pth
checkpoints/{experiment_name}/bestmodel/best_model_meta.json
tf-logs/{experiment_name}_ALB_PPO_YYYYMMDD_HHMMSS/
```

`--resume` 只能恢复当前实验目录，初始调度和重调度不得共用 checkpoint。

### 4.4 下一版推荐目标

不要重新长期依赖全局 `configs` 单例。推荐逐步改为：

```python
env = AirLineEnv_Graph(data_path=data_path, cfg=cfg.env)
model = HBGATPN(model_cfg=cfg.model, env_cfg=cfg.env)
agent = PPOAgent(model=model, cfg=cfg.train)
vector_env = VectorEnv(make_env=make_env, cfg=cfg.rollout)
```

配置对象应为可序列化、带类型检查的 dataclass。高层训练逻辑不应直接依赖全局可变变量。

---

## 5. HB-GAT-PN 模型侧修复

### 5.1 LayerNorm 解耦

旧逻辑使用一个 `use_layer_norm` 同时控制所有 LayerNorm，导致关闭 GAT LayerNorm 时也错误关闭输入嵌入层 LayerNorm。

推荐配置：

```yaml
use_input_layer_norm: true
use_gat_layer_norm: false
use_head_layer_norm: false
```

物理意义：

- 输入嵌入 LayerNorm：缓解工时、时间、负载、技能等异构特征尺度差异。
- GAT LayerNorm：可能削弱 APAL 中绝对时间和资源负载尺度，默认关闭。
- Head LayerNorm：会改变策略头和价值头输出尺度，默认关闭以保持旧模型行为。

验收标准：

- Task、Worker、Station 输入投影后存在独立 LayerNorm。
- GAT 消息传递层和策略/价值头按各自开关构造。
- 三个开关互不影响。

### 5.2 Actor 与 Critic 分工

应保留共享图编码或明确的独立分支设计，但必须记录：

- Actor 输出任务、站位、工人团队的条件概率。
- Critic 只输出状态价值，不参与动作掩码。
- Actor 与 Critic 的学习率倍率独立。
- Critic 指标良好不代表 Actor 已收敛，二者必须分别观测。

当前训练经验表明，Critic 的 explained variance 可以接近 1，但策略综合成绩仍可能停滞。因此不能以 Critic 收敛代替策略质量评估。

---

## 6. PPO 闭环正确性修复

这些修复应作为一个整体优先移植。

### 6.1 等待资源后的状态同步

当动作掩码全不可用并执行 `try_wait_for_resources()` 后：

- 环境时间已经推进。
- 工人和站位可用时间已经变化。
- observation 与 mask 必须立即重新生成。

禁止继续使用等待前的旧图状态进行动作选择，否则会污染 on-policy 轨迹。

### 6.2 GAE 轨迹边界

不同环境、不同 episode、rollout 截断之间不得串联优势。

Memory 应保存：

```text
is_terminal
is_truncated
```

GAE 遇到任一边界都必须重置递推项。即使最后一步不是环境 terminal，只要 rollout 在此截断，也必须标记为 boundary。

### 6.3 PPO ratio 数值稳定

禁止直接裁剪当前策略的真实 logprob。

正确流程：

```text
log_ratio = current_logprob - old_logprob
safe_log_ratio = clamp(log_ratio, -20, 20)
ratio = exp(safe_log_ratio)
```

真实 `current_logprob` 继续用于熵和概率语义；只在指数运算前保护 `log_ratio`。

### 6.4 Value clipping

PPO 更新应保存 old value，并计算：

```text
unclipped_value_loss
clipped_value_loss
value_loss = max(unclipped_value_loss, clipped_value_loss)
```

历史数据没有 old value 时才允许回退到普通 MSE。

### 6.5 小样本统计稳定

`torch.var()` 和 `torch.std()` 在单元素 batch 上应使用：

```python
unbiased=False
```

否则会产生 degrees of freedom 警告和 NaN 指标。

### 6.6 必须保留的 PPO 指标

TensorBoard 至少记录：

```text
Policy/ApproxKL
Policy/ClipFraction
Policy/RatioMean
Policy/RatioStd
Loss/Policy
Loss/Value
Loss/Entropy
Critic/ExplainedVariance
Critic/Target_Returns_Mean
Critic/Advantage_Mean
Critic/Advantage_Std
PPO/GPURebuildFallbackCount
PPO/BatchVectorRepairCount
```

解释策略：

- `Loss/Policy` 上升本身不能证明策略退化。
- `RatioMean` 应接近 1。
- `ApproxKL` 和 `ClipFraction` 持续上升说明更新逐渐更激进。
- `GPURebuildFallbackCount` 和 `BatchVectorRepairCount` 应用于发现训练路径退化，不应长期忽略。

验收测试：

```bash
python -m pytest tests/test_ppo_loop_fixes.py tests/test_amp_cpu_compat.py -q
```

---

## 7. AMP、显存与设备兼容

### 7.1 CPU/GPU 统一 AMP

应根据实际设备创建 AMP 上下文：

```text
CUDA：torch.amp.autocast(device_type="cuda")
CPU：禁用 CUDA GradScaler，不得硬编码 CUDA autocast
```

`GradScaler` 必须只在支持的设备上启用。

### 7.2 低显存测试护栏

Windows RTX 4060 8GB 环境下，低显存测试建议限制：

```text
峰值 allocated <= 2.5GB
峰值 reserved <= 3.5GB
num_envs = 2
小图 = data/283.csv
小 batch、短轨迹
```

测试后执行：

```python
gc.collect()
torch.cuda.empty_cache()
```

但正式训练中不要每一步清缓存。缓存清理只应放在：

- 独立测试结束。
- 明确捕获 CUDA OOM 后。
- 大型评估或训练阶段切换且确认旧 tensor 已释放时。

### 7.3 OOM 异常识别

不要把所有 `RuntimeError` 都当成 OOM。只处理：

- `torch.cuda.OutOfMemoryError`
- 错误文本明确包含 CUDA/GPU/device 与 out of memory

其他异常必须原样抛出，否则会掩盖真实训练错误。

---

## 8. VectorEnv 与前向采样性能

这是当前版本中收益最明确的一组优化。

### 8.1 平台启动方式

推荐：

```text
Linux：fork
Windows：spawn
```

Linux 使用 `forkserver` 曾导致 worker 内 OpenMP/MKL 线程退化，前向采样 CPU 利用率显著下降。

### 8.2 Worker 线程分配

每个 worker 不应使用全部 CPU 核心。推荐：

```text
worker_threads = max(1, os.cpu_count() // num_envs)
```

worker 入口尽早设置：

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
torch.set_num_threads()
```

例如 16 核、8 个环境时，每个 worker 约 2 线程。

### 8.3 Fast snapshot IPC

旧路径每步通过 Pipe 传输完整 `HeteroData`，序列化和复制成本很高。

推荐路径：

```text
get_masks_and_snapshots_all()
step_snapshot_all()
rebuild_state_from_snapshot()
```

关键点：

- 一次 IPC 同时返回 mask 和轻量 snapshot。
- 子进程 step 时设置 `skip_obs_building=True`，避免构造完整图。
- 主进程使用静态图上下文和动态 snapshot 重建 observation。
- `assigned_tasks` 使用 `list()` 浅拷贝，不使用重复 `deepcopy()`。
- 静态 `can_do_edge_index` 不应每步 clone 或通过 Pipe 传输，应从图上下文缓存恢复。
- snapshot 中缺少 `can_do_edge_index` 时，GPU rebuild 必须从静态上下文读取，不能产生 `KeyError`。

### 8.4 GPU 原地批量重建

PPO update 中应优先使用 GPU 图模板缓存和原地重建：

```text
snapshots → batched_rebuild_on_gpu → PyG batch → PPO update
```

需要防御：

- GPU rebuild 返回 `None`。
- PyG NodeStorage 缺少 `batch`。
- batch 向量数量和节点数量不匹配。
- 旧 snapshot 缺少后来移除的静态字段。

### 8.5 批量动作选择

应保留 `select_actions_batch()`，避免逐环境重复执行完整 GNN 前向。

优化重点：

- 避免循环中频繁 `.item()`、`.cpu()` 和隐式 CUDA 同步。
- Task、Station 和 Worker logits 尽可能批量计算。
- 随机模式要求概率语义等价；确定性模式应与旧实现逐动作一致。
- 如果训练代码与 Agent 版本不一致，可提供显式兼容回退，但启动时必须打印警告。

### 8.6 Shadow mask 验证

训练默认关闭：

```yaml
enable_shadow_mask_verification: false
```

三套 mask 实现交叉校验只适合专项测试，不适合每个训练 step。

### 8.7 Profiler

保留以下采样指标：

```text
Rollout/EnvStepsPerSec
Rollout/Mask_ms
Rollout/DeadlockCheck_ms
Rollout/Select_ms
Rollout/Snapshot_ms
Rollout/Step_ms
Rollout/FastPathEnabled
Memory/Allocated_GB
Memory/Reserved_GB
```

Profiler 不应强制 `torch.cuda.synchronize()`，除非进行专项精确测量。

验收测试：

```bash
python -m pytest tests/test_vector_env_rebuild.py tests/test_vector_env_safety.py -q
python -m pytest tests/test_low_memory_gpu_safety.py -q
```

---

## 9. APAL 环境与硬约束

### 9.1 派工人数硬约束

“团队人数满足工序需求”必须是硬约束，而不是奖励惩罚。

应同时保证：

- 动作 mask 不允许选择人数不足的团队。
- 自回归工人选择必须选满需求人数。
- 每名工人满足技能要求。
- 同一工人不能在重叠时间段执行多个任务。
- 工人池抽样后，所有技能及最大需求人数仍有覆盖。

禁止通过降低任务 demand 来避免死锁，这会改变原问题物理语义。

### 9.2 站位和前驱约束

验证器至少检查：

```text
任务完整性
重复或缺失任务
紧前紧后
工序 release time
冻结任务一致性
工人时间重叠
站位槽位上限
技能匹配
需求人数
非法动作或无效 step
```

合法性检查必须独立于 reward。违反硬约束的方案不能因 makespan 较低而成为 best。

### 9.3 事件等待与零工时任务

应保留：

- 资源不可用时的事件推进。
- 零工时任务自动穿透。
- 零工时任务无限循环保护。
- 并行环境使用独立随机状态，避免 seed 冲突。

### 9.4 APAL 诊断指标

推荐记录：

```text
APAL/schedulable_task
APAL/avg_resource_wait_h
APAL/avg_station_wait_h
APAL/station_slot_vacancy_ratio
APAL/worker_idle_ratio
APAL/critical_start_offset_h
RewardDiagnostic/station_wait_h
```

`station_wait_h` 不能只读取动作后已经被覆盖的状态。应比较动作前后资源可用时间或使用实际 `station_ready_time - min_start_bound`。

---

## 10. 初始调度训练模式

为了先获得稳定高质量 baseline，初始调度训练应关闭动态事件：

```yaml
enable_dynamic_events: false
enable_station_breakdown: false
enable_material_delay: false
enable_online_duration_perturb: false
enable_worker_fatigue: false
```

可以保留：

```yaml
randomize_durations: true
dur_random_range: 0.2
randomize_workers: true
```

但随机工人必须保持固定人数和技能/需求覆盖，不能改变硬约束。

初始调度 best model 的选择规则：

```text
合法方案中 makespan 最低
```

实测经验：

- 当前研究中初始调度曾达到约 137 小时。
- 这说明旧版调度语义和当前 PPO/性能修复可以组合出有竞争力结果。
- 重新开发时应先复现这一量级，再引入重调度，不要同时调试两套任务。

---

## 11. 预测-反应式 APAL 重调度

### 11.1 Baseline 与节拍

重调度必须先有合法初始计划 CSV：

```text
results/final_schedule.csv
```

该文件应由已训练的初始调度模型确定性生成，并通过 schedule 验证器。

定义：

```text
takt = baseline makespan
```

不能使用随机策略或未训练权重生成 baseline，否则重调度评价失去意义。

### 11.2 当前 v1 扰动

当前有价值且已实现的扰动是：

```text
工序延迟开始 / operation release delay
```

不要将其继续命名为 material delay，因为它表达的是任务 release time，而不是物料对象状态。

### 11.3 冻结规则

在重调度触发时刻：

- 已完成任务绝对冻结。
- 已开始任务绝对冻结。
- 冻结任务的开始时间、结束时间、站位和团队不能变化。
- 未开始任务允许重排，但必须满足前驱、资源和 release time。

状态中可加入：

```text
baseline 开始时间偏移
baseline 站位
baseline 团队规模
是否冻结
是否受延迟影响
延迟强度
```

初始模型 warm-start 时，旧输入维度对应权重精确加载，新增特征权重单独初始化。

### 11.4 固定验证场景

训练场景可以随机生成，但评估必须固定：

```text
相同 baseline
相同 scenario CSV
相同扰动时刻
相同延迟任务
相同 release time
相同数据集和工人配置
```

`reschedule_eval_scenarios.csv` 一旦用于 PPO 与 GA 对比，不应在每次评估时重新随机生成。

### 11.5 共享综合评分

PPO 和 GA 必须调用同一个评分函数。

推荐分项：

```text
makespan_term
balance_term
takt_violation_term
start_stability_term
station_change_term
team_change_term
```

综合评分越低越好：

```text
score =
    makespan / takt
    + r_coef_std * balance_std / ideal_station_load
    + takt_violation_weight * takt_violation / takt
    + start_stability_weight * mean_start_deviation / takt
    + station_change_weight * station_change_rate
    + team_change_weight * team_change_rate
```

出现任一硬约束违规时：

```text
eligible = false
不得保存为 best model
```

重调度 best model 的选择规则：

```text
eligible_rate = 1.0 且平均综合评分最低
```

### 11.6 PPO 与 GA 同台比较

比较条件必须完全一致：

- 同一 baseline。
- 同一固定场景 CSV。
- 同一综合评分器。
- 同一硬约束检查器。
- 同一场景数量。
- 汇报逐场景和平均值，不能拿 PPO 单场景最好值与 GA 多场景平均值比较。

已有实验中，PPO 平均综合评分约 `1.358`，GA 平均综合评分约 `1.404`。该结果只能在 baseline、场景文件和配置完全一致时成立。

---

## 12. 训练、验证和可视化链路

建议保留并统一以下入口：

```text
train.py
scripts/generate_schedule.py
scripts/evaluate_model.py
scripts/evaluate_reschedule_model.py
scripts/evaluate_reschedule_ga.py
scripts/visualize_reschedule_comparison.py
```

每个入口都应支持：

```text
--config
--model_path
--output_dir
--seed
```

重调度可视化应同时显示：

- 上方 baseline 甘特图。
- 下方 reschedule 甘特图。
- 重调度触发时刻。
- baseline takt。
- 新 makespan。
- 逐任务开始时间偏移。
- 站位和团队是否变化。

同时输出：

```text
baseline_vs_reschedule.png
reschedule_schedule.csv
diff.csv
comparison_summary.csv
```

---

## 13. 跨平台与可复现性

### 13.1 Pathlib

所有路径应使用 `pathlib.Path`：

```python
project_root = Path(__file__).resolve().parent
output_path = project_root / "results" / "file.csv"
```

禁止手写 Windows 或 Linux 路径分隔符。

### 13.2 全局随机种子

入口统一锁定：

```text
random
numpy
torch
torch.cuda
每个仿真环境
```

同时设置确定性后端选项。VectorEnv 每个环境应使用可复现但互不相同的派生 seed。

### 13.3 Windows/Linux

推荐最小验证矩阵：

| 平台 | 环境数 | 启动方式 | 目标 |
|---|---:|---|---|
| Windows RTX 4060 8GB | 2 | spawn | 低显存正确性 |
| Linux 服务器 | 8 | fork | 并行吞吐 |

Linux 结果不能由 Windows 测试推断，反之亦然。

---

## 14. 测试体系

### 14.1 基础层

```bash
python -m pytest tests/test_data_loader.py tests/test_engine_and_mask.py -q
```

### 14.2 PPO 层

```bash
python -m pytest tests/test_ppo_loop_fixes.py tests/test_amp_cpu_compat.py -q
```

### 14.3 VectorEnv 层

```bash
python -m pytest tests/test_vector_env_rebuild.py tests/test_vector_env_safety.py -q
```

### 14.4 GPU 低显存层

```bash
python -m pytest tests/test_low_memory_gpu_safety.py -q
```

### 14.5 APAL 硬约束层

```bash
python -m pytest tests/test_worker_demand_hard_constraint.py -q
```

### 14.6 初始调度和重调度层

```bash
python -m pytest tests/test_initial_schedule_mode.py -q
python -m pytest tests/test_reschedule_task_delay.py tests/test_reschedule_ga.py -q
```

### 14.7 配置层

必须覆盖：

- 分层 YAML 正确加载。
- 未知字段报错。
- 后加载配置覆盖先加载配置。
- 嵌套节点执行深度合并。
- 实验输出目录隔离。
- Windows/Linux 路径解析。

---

## 15. 明确不要重新引入的内容

### 15.1 200–3500 工序单模型混训

不要重新引入：

- 单模型跨 200–3500 工序训练。
- 50 个跨尺度随机数据集池。
- 同一次 PPO update 混合不同图规模。
- 混合图失败后静默回退 CPU。

原因：

- 改变了原始研究问题。
- 增加显存和 batch 行为的不确定性。
- 破坏已经验证的 rollout 并行化。
- 难以区分泛化收益和训练分布漂移。

### 15.2 工人数幂律缩放

不要重新引入：

```text
worker_scaling_mode
worker_scale_coef
worker_scale_alpha
compute_worker_count()
```

当前阶段固定工人数量更利于复现实验和解释模型行为。

### 15.3 动态 batch 阶梯和 OOM retry

不要重新引入：

```text
compute_batch_size_from_staircase()
batchsize_staircase.yaml
同一次 update 内 OOM 后重试
```

原因：

- OOM 前可能已经完成部分 optimizer step。
- 重用同一 on-policy memory 会破坏更新语义。
- 容易误伤 rollout 并行化和 GPU rebuild。

如果后续确需自动 batch，应先做 update 前显存 probe，且不能进入 optimizer step 后再重试。

### 15.4 Linux forkserver 默认值

不要恢复 Linux `forkserver` 默认值。它曾导致 worker 线程退化和采样速度明显下降。

### 15.5 训练时开启 shadow mask

不要在正式训练中同时运行 tensorized、vectorized 和 legacy 三套 mask。

---

## 16. 尚未彻底解决的问题

### 16.1 动态事件物理语义未完全闭合

未来事件可能尚未进入当前动作的 earliest-start 计算。例如 Agent 在当前时间连续安排未来任务时，未来工人离开、站位故障等事件可能只在之后 `_advance_time()` 时处理。

这会产生“先承诺未来排程，再事后修正”的非平稳轨迹。

在重新开发初始调度时应继续关闭这些动态事件。重调度先只实现语义明确的 operation release delay。

### 16.2 训练主循环仍然过长

当前 `train.py` 同时负责：

- 配置加载。
- 环境构建。
- rollout。
- PPO update。
- 评估。
- checkpoint。
- GA 对比。
- 报告和可视化。

重新完善时应拆为 Runner/Service，但不要在恢复基准性能前同时引入大规模框架迁移。

### 16.3 尚未完成 Lightning 化

当前代码没有完成 PyTorch Lightning 训练体系。考虑到 PPO、VectorEnv 和自定义 on-policy memory 的特殊性，Lightning 化应在环境语义、PPO 正确性和性能基线稳定后单独进行。

### 16.4 `configs.py` 尚未彻底删除

当前 YAML 最终仍写入全局 `configs`。这是兼容过渡方案，不是最终架构。

---

## 17. 推荐重新实施顺序

### 阶段 A：恢复旧版性能基线

1. 使用旧版 APAL 环境、旧奖励和单数据集训练。
2. 统一当前数据文件名。
3. 加入 schedule 验证器。
4. 复现约 137 小时的初始调度水平。

验收：合法 makespan、固定 seed 可复现、与旧版结果接近。

### 阶段 B：移植 PPO 正确性

1. 等待后状态同步。
2. GAE boundary。
3. ratio 数值保护。
4. value clipping。
5. AMP CPU/GPU 兼容。

验收：专项测试全部通过，短训练无 NaN、Inf 和轨迹串联。

### 阶段 C：移植采样性能优化

1. VectorEnv。
2. Linux fork / Windows spawn。
3. worker 线程分配。
4. fast snapshot。
5. 批量动作选择。
6. GPU 原地重建。

验收：与旧路径动作和状态语义一致，`EnvStepsPerSec` 明显提升。

### 阶段 D：迁移 YAML 和实验隔离

1. 分层 YAML。
2. 深度合并。
3. 启动时打印最终配置。
4. checkpoint/TensorBoard 按实验隔离。
5. 逐步移除全局单例。

验收：修改任一 YAML 字段后可从启动日志和单元测试确认最终值。

### 阶段 E：补齐监控与硬约束

1. PPO KL/ratio/clip 指标。
2. APAL 资源等待和空闲指标。
3. 工人需求、技能、重叠、站位槽硬约束。
4. GPU fallback 和 batch repair 指标。

验收：任何非法方案均不能保存为 best。

### 阶段 F：重新加入重调度

1. 初始模型生成合法 baseline。
2. 固定 release-delay 场景。
3. 冻结任务和 release 硬约束。
4. PPO/GA 共享评分。
5. 固定场景评估。
6. baseline/reschedule 可视化。

验收：PPO 与 GA 使用完全相同的输入条件和评分口径。

---

## 18. 关键 Git 提交索引

以下提交适合用于查阅实现思路，不建议整提交 cherry-pick：

| 提交 | 主要内容 |
|---|---|
| `21f6ff9` | PPO 安全护栏、AMP、低显存测试、路径修复 |
| `7ca55f5` | rollout fast snapshot 接线 |
| `53e8456` | 分层 YAML、采样优化和多项训练修复 |
| `bfb782b` | Linux fork 与 worker CPU 利用率修复 |
| `338810f` | LayerNorm 配置拆分的早期实现 |
| `9367600` | 显式紧前关系与并行 DAG 修复 |
| `5d57b12` | PyG batch 向量缺失修复 |
| `895fb41` | 代码清理和共享逻辑抽取 |
| `392dc00` | 重调度综合评分、GA 和硬约束 |
| `b32f5f2` | 独立 PPO 重调度验证入口 |
| `29d6dce` | 重调度状态特征和固定场景增强 |
| `b63927e` | 多进程 IPC/共享内存泄漏修复 |
| `f125c87` | fast snapshot mask 类型修复 |
| `a45dac4` | 批量动作选择 CUDA 同步优化 |
| `b9396e6` | rollout CUDA 同步和 heartbeat 开销优化 |
| `eec459e` | mask tensor 操作与预分配优化 |
| `5084c0f` | OOM 识别误判修复 |

---

## 19. 最终迁移检查表

- [ ] 四个数据文件名统一。
- [ ] APAL DAG 和显式前驱解析正确。
- [ ] 工人需求、技能、重叠和站位槽为硬约束。
- [ ] LayerNorm 三类开关独立。
- [ ] 等待后 observation/mask 同步。
- [ ] GAE 不跨环境、episode 和 rollout boundary。
- [ ] PPO ratio 只裁剪 log-ratio。
- [ ] value clipping 使用 old value。
- [ ] AMP 在 CPU/CUDA 均能运行。
- [ ] Linux 使用 fork，Windows 使用 spawn。
- [ ] VectorEnv worker 线程按总核心数分配。
- [ ] fast snapshot 不传输完整 HeteroData。
- [ ] 静态边从上下文缓存恢复。
- [ ] 批量动作选择不频繁触发 CUDA 同步。
- [ ] YAML 使用递归深度合并。
- [ ] 启动日志打印最终 batch、update 和 eval 频率。
- [ ] checkpoint、best model 和 TensorBoard 按实验隔离。
- [ ] 初始调度关闭动态事件。
- [ ] baseline 由真实初始模型生成并通过验证器。
- [ ] 重调度使用固定场景。
- [ ] PPO 与 GA 共用综合评分和硬约束检查。
- [ ] 不重新引入跨 200–3500 工序混训。
- [ ] 不重新引入幂律工人数缩放。
- [ ] 不重新引入同 update 的 OOM retry。
- [ ] 每个阶段均有小规模、低显存自动测试。
