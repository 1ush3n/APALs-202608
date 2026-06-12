# 基于异构图注意力网络与指针网络的飞机脉动装配线智能调度系统 (HB-GAT-PN)

本项目是一个融合了**深度强化学习 (PPO)**、**异构图注意力机制 (Heterogeneous GAT)** 与 **自回归指针网络 (Pointer Network)** 的工业级调度仿真与智能排产系统。专为解决“工序-站位-工人”三层高度耦合、包含严格拓扑前置约束的飞机装配线动态调度难题而设计。

---

## 🚀 最新版本重大更新里程碑 (v6.0+)

经过近期深度的 Rollout 加速工程与 APAL 硬约束防御系统建设，本项目在**训练速度和调度安全**两个维度实现了关键跨越：

### 1. 动作合法性边界防线 (Constraint Validation)
在环境执行层引入了严格的**动作合法性验证**，彻底杜绝非法调度进入物理仿真引擎污染学习信号：
- **`_validate_worker_team()`**：在 `step()` 执行前对每个动作进行完整校验——工序/站位索引范围、工人数量是否满足需求、无重复工人、技能匹配、站位锁定约束。
- **`_reject_invalid_action()`**：统一的非法动作拒绝出口，返回标准化惩罚和诊断信息。
- **工人需求读取稳健化**：`ppo_agent.py` 新增 `get_task_demand()`，从 Task 特征第 16 列读取需求，附带 tensor shape 断言。

### 2. Rollout 前向采样全链路加速 (Phase 0+1)
针对 DPPO 向量化环境中采样速度缓慢的瓶颈，实施系统性低风险加速：
- **影子校验按需开启**：动作掩码双路影子比对默认关闭，Mask 获取耗时 **5.9ms → 0.9ms（6.6×）**。
- **Snapshot 极致轻量化**：不再克隆 72K `can_do` 边，不再 `deepcopy(assigned_tasks)`，Snapshot 耗时 **1.6ms → 0.1ms（16×）**。
- **合并 IPC + 平台自适应**：`get_rollout_state` 单次 pipe 返回 masks+snapshot；Linux 自动 `fork` 8 env，Windows `spawn` 2 env。
- **Rollout Profiler**：每 10 ep 向 TensorBoard 写入 Mask/Select/Snapshot/Step/EnvStepsPerSec 计时指标。
- **OMP_NUM_THREADS 修复**：空值不再强制单线程，Linux NumPy 恢复全核并行。

### 3. 异构图编码器数值安全 (Precision & Safety)
- **FP16 安全**：全链路 `mask_value` 从 `-1e9` → `-1e4`，防 AMP 溢出。
- **logprob/ratio 安全**：`compute_stable_log_ratio_and_ratio()` 仅 clamp `log_ratio`，不篡改真实 `logprob`。
- **LeakyReLU 防死亡**：Feature Embedder 和 GAT Encoder 全面采用 `LeakyReLU(0.1)`（可配置开关）。


## 🚀 历史版本更新档案 (v5.0)

经过早期深度的工程化重构与架构升级，本项目实现了从实验室算法向**健壮工业级系统**的跨越：

### 1. 动态张量构建与显存免疫 (Phase 3)
彻底废弃了原先极易引发 OOM（显存溢出）的硬编码预分配指针池（`edge_ts_mem`等）。全面采用基于 `assigned_tasks` 的**实时动态异构图连边构建**。环境重构不仅大幅提升了运行时的稳定性，更让模型面对上万节点规模的扩展成为了可能。

### 2. 调度验证闭环打通 (Phase 3)
标准化了推理脚本 `generate_schedule.py` 的输出列字段（统一为 `TaskID`, `StationID`, `Team`, `Start`, `End`, `Duration`），并完美缝合了虚拟门控节点的导出逻辑。**目前本系统生成的排程结果能够在最严苛的物理验证器 (`utils/verify_schedule.py`) 中取得 100% 完美的合法性验证！**消除了“工人分身术”和“前驱丢失”等所有隐患。

### 3. 工程化解耦与日志优化 (Phase 3.5 & 3.6)
- **输出净化**：大幅优化 PPO Agent 的 KL 散度（KL Divergence）阈值预警机制，从原先刷屏的 Batch 级警告，收敛为 Update 周期末端的静默汇总报告，保留了 Early Stopping 防退化机制的同时还给了终端一片清净。
- **目录规范**：彻底贯彻“代码与产物分离”原则。所有的神经网络权重（如 `best_model.pth`）、断点检查点以及推理快照已从源码级别的 `models/` 目录强制迁移至根目录下的 `checkpoints/`，并已更新至 `.gitignore`，保障了版本控制库的纯净。

### 4. 动态事件干扰与域随机化注入 (Phase 4: Dynamic Events)
这是本项目在工业价值上的一次重大突破。我们在完全**零特征新增**（不改变神经网络输入维度，完美兼容已有历史最优权重）的前提下，通过环境底层时钟把控（Time-Shifting）实现了突发干扰的注入：
- **工人缺勤 (Worker Absenteeism)**：基于 `configs.py` 中的概率分布，环境会在训练期间随机投下“缺勤地雷”。
- **引擎免疫死锁**：在 `core/event_engine.py` 引入了 `WORKER_LEAVE` 和 `WORKER_RETURN`。当工人离岗时，仿真引擎会自动将其可用时间强制推后，并追加回归事件作为时钟跳跃锚点，完美避免了车间空转等待导致的逻辑死锁。
- **无感适配**：动作掩码器和异构图（GNN）通过原有的等待时间（Wait Time）特征，全自动地感知到了这些缺勤状况并做出绕行决策。

*(注：系统已规划了面向未来的 “模型权重动态手术扩展协议 (Zero-Padding Weight Surgery)”，即便日后需要扩充物料延迟等全新维度特征，也能在不破坏已有最佳模型参数的前提下无损热更新。)*

---

## 📂 核心代码目录与架构全景

系统遵循标准的强化学习环境闭环结构：**数据加载** -> **离散事件仿真引擎** -> **神经网络前向推断** -> **反向传播与 PPO 策略更新**。

```text
Dynamic_APAL_RL_v1/
│
├── configs.py             # 系统的全局控制中枢（超参数、随机化概率与动态事件开关）
├── data_loader.py         # 任务工序拓扑与人员信息的统一解析器
├── environment.py         # 基于 Gymnasium 规范构建的带有时间推移 (Time-Shifting) 引擎的离散事件环境
├── ppo_agent.py           # 强化学习 PPO 算法核心（支持 SIL、KL熔断、自回归梯度累积）
├── train.py               # 模型的主训练循环
├── generate_schedule.py   # 模型推理、排程导出与快照生成脚本
│
├── core/
│   ├── action_masker.py   # 动作合法性过滤中枢（拦截未就绪任务与缺勤工人）
│   └── event_engine.py    # 驱动整个仿真时钟流转的 Priority Queue 事件堆核心
│
├── models/                # 神经网络源代码目录
│   └── hb_gat_pn.py       # 包含了从图卷积到自回归多头输出网络架构的设计
│
├── conf/                  # Hydra/OmegaConf 分层 YAML 配置（experiment/hardware/model/reward/rollout/train）
├── tests/                 # 26+ 个 pytest 单元测试与回归验证
├── scripts/               # 推理、评估、敏感性分析入口脚本
├── checkpoints/           # [自动生成] 模型权重断点续训库及排程甘特图快照
├── utils/                 
│   ├── vector_env.py      # 多进程 DPPO 向量化环境（支持 forkserver/spawn）
│   ├── gpu_graph_manager.py  # GPU 批量图构建与原地特征刷新
│   ├── report_generator.py   # 自动训练诊断报告生成
│   └── verify_schedule.py # 极其严苛的物理约束验证官
```

---

## 📜 核心机制深度解析

### 1. `configs.py` (全局配置文件)
此文件相当于系统的**中央控制台**，所有可调控的数值与超参数皆汇聚于此。
- **域随机化 (Domain Randomization) & 动态事件**: 包含了 `randomize_durations` (工时波动) 和 `enable_dynamic_events` / `prob_worker_absent_max` (随机缺勤配置)。使得网络能够经历千锤百炼，不仅学会排程，更学会在突发干扰下抢险救灾。
- **平台自适应**: `num_envs_linux=8` / `num_envs_windows=2`，自动选择 `forkserver` 或 `spawn` 多进程启动方式。
- **Rollout 加速开关**: `use_rollout_snapshot_fastpath`, `enable_rollout_profiler`, `enable_shadow_mask_verification=False`。

### 2. `environment.py` & `core/` (仿真引擎核心)
这是整个项目的物理与时间引擎核心，彻底摒弃了死板的步进式循环，改用真实的**离散事件仿真 (Discrete Event Simulation)**。
- **APAL 硬约束防线** (`_validate_worker_team` / `_reject_invalid_action`): 在 `step()` 中对所有传入动作执行严格合法性校验——工人技能匹配、站位锁定约束、人数需求满足、无重复工人。任何违规动作即被拒绝并返回惩罚信号，确保模型不会从非法调度中学习。
- **事件优先队列 (`EventQueue`)**: 环境的流转完全由离散事件（如 `TASK_FINISH`, `WORKER_LEAVE`, `WORKER_RETURN`）驱动。当全员忙碌或因缺勤无事可做时，时钟 `current_time` 会直接跃迁到队列中的下一个事件点。
- **异构图状态生成**: 在每个离散时间帧，收集所有状态拼装为 `torch_geometric` 格式的字典。利用巧妙的时钟差值 `max(0, worker_free_time - current_time)`，让 GNN 直接从时间维度读取繁忙与缺勤状态。

### 3. `ppo_agent.py` (PPO 强化大脑)
- **级联自回归 (Pointer Network 核心体现)**：选任务 → 定站位 → 选组员，逐层剥离多维庞大动作空间。
- **数值安全 Ratio 计算** (`compute_stable_log_ratio_and_ratio`): 仅在 `exp()` 前 clamp `log_ratio`，不篡改真实 `logprob`，保障 PPO 策略概率比的数学一致性。
- **保护机制**：包含 Value 网络梯度裁剪、KL 散度过高时的静默熔断、自适应熵衰减、以及梯度累积归一化。
- **向量化批推理** (`select_actions_batch`): 支持多环境 HeteroData 批量打包送入 GPU，单次前向完成所有环境的动作选择。

### 4. `utils/verify_schedule.py` (规则验证器)
利用**扫描线算法 (Sweep Line Algorithm)** 对生成的排程 CSV 进行严苛审计。它会逐一排查：任务拓扑图倒置、同一工人分身作业、无技能越权操作、以及超过物理卡位并发数等致命逻辑错误，确保模型产出的排单在实际工厂中 100% 落地可行。
---

## 实验与训练启动方法

> 配置重构说明：新的训练体系从 `conf/experiment/*.yaml` 读取分场景配置，并由 `apal_config/` 负责 YAML 合并、未知字段检查和实验路径解析。`configs.py` 当前仅作为迁移期兼容入口，后续会在核心模块全部显式接收配置对象后删除。

以下命令均默认在项目根目录执行。Windows 本机请优先使用 `rag_env` 的解释器：

```powershell
C:\Users\13575\miniconda3\envs\rag_env\python.exe
```

Linux 服务器如果已经进入对应环境，可直接使用：

```bash
python
```

### 1. 初始 APAL 调度训练

用于训练无动态扰动的初始调度模型，动态事件关闭，保留必要的 APAL 约束与 PPO 训练闭环。输出目录按 `experiment_name=initial_schedule` 隔离。

Windows：

```powershell
C:\Users\13575\miniconda3\envs\rag_env\python.exe train.py --config conf/experiment/initial_schedule.yaml
```

Linux：

```bash
python train.py --config conf/experiment/initial_schedule.yaml
```

断点恢复：

```bash
python train.py --config conf/experiment/initial_schedule.yaml --resume
```

主要产物：

- `checkpoints/initial_schedule/latest_checkpoint.pth`
- `checkpoints/initial_schedule/bestmodel/best_model.pth`
- `checkpoints/initial_schedule/bestmodel/best_model_meta.json`
- TensorBoard 日志：`tf-logs/initial_schedule_*` 或 Linux 下 `/root/tf-logs/initial_schedule_*`

### 2. 生成初始调度 baseline

重调度实验需要 `results/final_schedule.csv` 作为 baseline 初始计划。若该文件不存在，重调度训练会自动尝试用 `checkpoints/initial_schedule/bestmodel/best_model.pth` 生成。也可以手动生成：

```bash
python scripts/generate_schedule.py
```

注意：如果 `results/final_schedule.csv` 已存在，重调度训练会直接复用，不会自动覆盖。更换初始模型后，应先备份或删除旧 baseline，或修改 `reschedule_baseline_schedule_path` 指向新的 CSV。

### 3. 预测-反应式重调度 PPO 训练

用于训练以 baseline 为参照的 APAL 重调度策略。当前默认只处理工序 release 延迟场景，冻结已开始/已完成任务，按综合评分保存 best model。

Windows：

```powershell
C:\Users\13575\miniconda3\envs\rag_env\python.exe train.py --config conf/experiment/reschedule_task_delay.yaml
```

Linux：

```bash
python train.py --config conf/experiment/reschedule_task_delay.yaml
```

断点恢复：

```bash
python train.py --config conf/experiment/reschedule_task_delay.yaml --resume
```

主要产物：

- `checkpoints/reschedule_task_delay/latest_checkpoint.pth`
- `checkpoints/reschedule_task_delay/bestmodel/best_model.pth`
- `checkpoints/reschedule_task_delay/bestmodel/best_model_meta.json`
- 固定验证场景：`results/reschedule_eval_scenarios.csv`

重调度 best model 保存逻辑：

- 每次评估都会覆盖保存 `latest_checkpoint.pth`
- `best_model.pth` 只在 `eligible_rate=1.0` 且 `composite_score` 更低时保存
- `composite_score` 越低越好

### 4. GA 重调度基线评估

GA 与 PPO 重调度使用同一个 baseline、同一个固定验证场景 CSV、同一个综合评分函数和同一套硬约束检查。

推荐入口：

```bash
python scripts/evaluate_reschedule_ga.py --config conf/experiment/reschedule_task_delay.yaml --pop_size 30 --max_gen 20
```

快速 smoke：

```bash
python scripts/evaluate_reschedule_ga.py --config conf/experiment/reschedule_task_delay.yaml --pop_size 3 --max_gen 2 --num_runs 1
```

直接运行 GA 模块也可以：

```bash
python baselines/heuristic/reschedule_ga.py --config conf/experiment/reschedule_task_delay.yaml --pop_size 30 --max_gen 20
```

主要产物：

- `results/reschedule_ga/reschedule_ga_eval.csv`

对比 PPO 与 GA 时，应优先比较：

- `avg_score` / `RescheduleEval/CompositeScore`：越低越好
- `eligible_rate`：越高越好，最好为 `1.0`
- `avg_makespan`
- `score_makespan`
- `score_balance`
- `score_takt_violation`
- `score_start_stability`
- `score_station_change`
- `score_team_change`

### 5. 模型评估与排程导出

评估已有模型：

```bash
python scripts/evaluate_model.py --model_path checkpoints/initial_schedule/bestmodel/best_model.pth --test_data data/283.csv --num_runs 3 --temperature 0.0
```

生成确定性调度 CSV：

```bash
python scripts/generate_schedule.py
```

输出：

- `results/final_schedule.csv`
- 评估脚本会额外导出 `results/eval_*_schedule.csv` 与甘特图

四个窄规模模型统一评估：

```bash
python scripts/evaluate_initial_buckets.py \
  --models_root checkpoints \
  --output_dir results/initial_bucket_eval
```

该脚本分别在 `283/680/2338/3182.csv` 原始基准图上执行确定性标准场景，调用硬约束验证器，并汇总 `makespan / ideal_makespan`。

### 6. 低风险回归验证

基础数据、引擎与派工硬约束：

```bash
python -m pytest tests/test_data_loader.py tests/test_engine_and_mask.py tests/test_worker_demand_hard_constraint.py -q
```

重调度与 GA 基线：

```bash
python -m pytest tests/test_reschedule_task_delay.py tests/test_reschedule_ga.py -q
```

低显存 GPU 与 AMP 兼容：

```bash
python -m pytest tests/test_low_memory_gpu_safety.py tests/test_amp_cpu_compat.py -q
```

向量化环境安全：

```bash
python -m pytest tests/test_vector_env_rebuild.py tests/test_vector_env_safety.py -q
```

窄规模分桶、固定工人数和同质 PPO memory：

```bash
python -m pytest tests/test_narrow_bucket_training.py -q
```

### 7. 配置入口说明

- `conf/experiment/initial_schedule.yaml`：初始调度训练入口
- `conf/experiment/initial_schedule_283.yaml`：200–350 工序窄池，固定 80 名工人，batch 32
- `conf/experiment/initial_schedule_680.yaml`：550–850 工序窄池，固定 100 名工人，batch 16
- `conf/experiment/initial_schedule_2338.yaml`：2000–2750 工序窄池，固定 140 名工人，batch 8
- `conf/experiment/initial_schedule_3182.yaml`：2800–3500 工序窄池，固定 160 名工人，batch 4
- `conf/experiment/reschedule_task_delay.yaml`：预测-反应式重调度训练与 GA 对比入口
- `conf/env/initial_schedule.yaml`：关闭动态事件的初始调度环境配置
- `conf/env/reschedule_task_delay.yaml`：重调度 release 延迟场景配置
- `conf/hardware/windows_4060.yaml`：Windows RTX 4060 低显存配置
- `conf/hardware/windows_4060_low_memory.yaml`：四个窄池统一覆盖为 batch 4
- `conf/hardware/linux_server.yaml`：Linux 服务器并行配置

### 8. 窄规模训练池生成

正式训练不再混合 200–3500 工序的全范围数据。四个模型分别使用一个基准图和 10 个同模板窄区间变体。

生成全部训练池：

```bash
python scripts/generate_initial_buckets.py --bucket all --num_samples 10 --seed 42
```

重新生成指定分桶：

```bash
python scripts/generate_initial_buckets.py --bucket 283 --overwrite
```

生成目录中的 `manifest.json` 记录模板哈希、区间、种子和各 CSV 哈希。训练时工人数与 batch 均由对应实验 YAML 固定，不再运行时动态缩放。

### 9. 四个独立模型训练

Linux：

```bash
python train.py --config conf/experiment/initial_schedule_283.yaml
python train.py --config conf/experiment/initial_schedule_680.yaml
python train.py --config conf/experiment/initial_schedule_2338.yaml
python train.py --config conf/experiment/initial_schedule_3182.yaml
```

Windows RTX 4060 8GB 在实验配置后追加低显存覆盖：

```powershell
C:\Users\13575\miniconda3\envs\rag_env\python.exe train.py `
  --config conf/experiment/initial_schedule_283.yaml `
  --config conf/hardware/windows_4060_low_memory.yaml
```

每次 PPO update 内所有环境严格使用同一图和相同工人数；memory 清空后才同步切换到下一个同规模变体。

### 10. 评估已有模型

```bash
python scripts/evaluate_model.py \
  --model_path checkpoints/initial_schedule/bestmodel/best_model.pth \
  --test_data data/283.csv \
  --num_runs 3 \
  --temperature 0.0

python scripts/evaluate_reschedule_model.py \
  --config conf/experiment/reschedule_task_delay.yaml \
  --model_path checkpoints/reschedule_task_delay/bestmodel/best_model.pth \
  --num_runs 3 \
  --output_dir results/reschedule_ppo_eval
```

### 11. 重调度偏差可视化

用于把 baseline 初始计划与 PPO 重调度结果画在同一张图中，并导出逐任务偏差明细。

```bash
python scripts/visualize_reschedule_comparison.py \
  --config conf/experiment/reschedule_task_delay.yaml \
  --model_path checkpoints/reschedule_task_delay/bestmodel/best_model.pth \
  --scenario_id eval_000 \
  --output_dir results/reschedule_visual
```

主要输出：
- `results/reschedule_visual/eval_000_baseline_vs_reschedule.png`
- `results/reschedule_visual/eval_000_reschedule_schedule.csv`
- `results/reschedule_visual/eval_000_diff.csv`
- `results/reschedule_visual/comparison_summary.csv`

图中 baseline 位于上方，reschedule 位于下方；蓝色虚线表示重调度触发时刻，黑色虚线表示 baseline takt，红色线表示重调度 makespan。
