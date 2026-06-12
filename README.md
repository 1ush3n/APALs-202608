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
- **合并 IPC + 平台自适应**：`get_rollout_state` 单次 pipe 返回 masks+snapshot；Linux 自动 `forkserver` 8 env，Windows `spawn` 2 env。
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
