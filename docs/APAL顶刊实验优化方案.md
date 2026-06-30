# APAL 顶刊实验优化方案

本文基于 `docs/文献学习总结.md` 中对装配线综述、航空装配、GNN/RL 调度、随机调度和大规模图学习论文的学习结果，给出当前 APAL 调度论文的实验补强方案。目标不是继续堆叠模型，而是补齐顶刊审稿通常关注的证据链：工业问题定义、跨规模泛化、强 baseline、严格消融、动态扰动、工程效率和可复现性。

重要边界：

- APAL 不能等同于 FJSP/JSSP。FJSP/JSSP 文献只用于借鉴图表示、强化学习建模和实验设计。
- 航空装配文献主要支撑工业对象、动态扰动和实时重调度必要性。
- 所有“建议”均是对当前 APAL 论文的迁移性设计，不等同于原论文结论。

## 1. 当前实验基础与主要短板

当前项目已经具备较好的实验基础：

- 多规模主数据集：`283/680/2338/3182`。
- 主方法：HB-GAT-PN + PPO + Skill Hub。
- 学习型 baseline：Basic PPO、DQN。
- 启发式与搜索 baseline：SPT、LPT、EDD、CPM、MSL、Random、Beam、IG、SA、GA。
- 消融入口：no-GAT、no-pointer、no-mask、no-attention、no-skill-hub 等。
- 重调度入口：物料延迟场景、baseline schedule、固定验证场景、reschedule GA。
- 运行时统计：训练耗时、验证 wall time、推理耗时、SPS、显存/图结构相关统计。

主要短板是：

- 主结果目前偏重 makespan，工业可部署指标还不够完整。
- 消融结果存在统计口径不完全一致的问题，例如部分结果是单次验证或跨规模 fallback。
- 动态重调度实验还需要形成扰动强度曲线，而不是只做单一物料延迟案例。
- Skill Hub 的贡献需要同时从语义、边数、显存、rollout 吞吐和解质量证明。
- 顶刊常见的泛化实验、鲁棒性实验、复杂度表和 Pareto 质量-效率图还需要补齐。

## 2. 数据与场景补强

### 2.1 保留四个主规模数据集

现有 `283/680/2338/3182` 应作为主实验表的核心数据集。它们的作用是展示 APAL 方法在小、中、大规模工业图上的表现。

建议主表固定报告：

- makespan
- workload balance standard deviation
- worker utilization
- station utilization
- completion rate
- valid/invalid
- inference time

### 2.2 新增中等规模随机训练/测试数据

建议新增一组中等规模随机 APAL 数据集，例如：

| 规模 | 建议实例数 | 用途 |
|---:|---:|---|
| 400 | 5-10 | 训练分布补充 |
| 600 | 5-10 | 训练分布补充 |
| 800 | 5-10 | 验证泛化 |
| 1000 | 5-10 | 验证泛化 |

用途：

- 让模型不只记住单个 `680` 或单个固定规模。
- 支撑“小规模/中规模训练，大规模测试”的泛化叙事。
- 为多种子训练提供更稳定的训练分布。

### 2.3 新增技能瓶颈场景

Skill Hub 是当前方法的重要贡献，因此需要专门设计场景证明其价值。建议构造三类技能瓶颈数据：

| 场景 | 设计方式 | 目的 |
|---|---|---|
| 关键技能稀缺 | 降低某类关键技能工人比例 | 检验 worker-skill-task 建模能力 |
| 关键技能高需求 | 提高某类技能任务比例或需求人数 | 检验技能瓶颈下的资源分配 |
| 多技能耦合 | 增加工人技能重叠和任务技能多样性 | 检验 Skill Hub 是否优于 Worker-Task 直接边 |

核心指标：

- makespan
- skill bottleneck utilization
- invalid skill assignment count
- eligible rate
- worker utilization variance

### 2.4 新增动态扰动场景集

重调度实验不应只使用一个物料延迟案例。建议形成固定扰动场景库：

| 扰动类型 | 强度设置 | 核心指标 |
|---|---|---|
| 物料延迟 | 低/中/高延迟概率与幅度 | frozen violation、release violation、makespan |
| 工时扰动 | 低/中/高加工时间波动 | schedule change、completion rate |
| 工人不可用 | 低/中/高缺勤比例 | eligible rate、worker utilization |
| 站位异常 | 固定几个站位容量或时间窗口异常 | station utilization、slot violation |
| 混合扰动 | 物料延迟 + 工人不可用 | robustness score、response time |

建议每类扰动固定随机种子，生成可复现实验文件，避免每次验证场景变化导致结果不可比。

## 3. 主实验与 Baseline 体系

### 3.1 Baseline 分层

建议将 baseline 明确分成三层：

| 层级 | 方法 | 作用 |
|---|---|---|
| 规则类 | SPT、LPT、EDD、CPM、MSL、Random | 证明主方法优于常见人工 dispatching rule |
| 搜索类 | Beam、IG、SA、GA | 证明主方法在质量-时间权衡上有优势 |
| 学习类 | Basic PPO、DQN、HB-GAT-PN 消融 | 证明异构图表示和策略结构必要 |

搜索类 baseline 必须统一预算，例如 Beam 宽度、IG 迭代数、SA 迭代数、GA 种群和代数。否则运行时间和解质量不可公平比较。

### 3.2 主结果统计口径

建议主表采用以下规则：

- 主方法和核心学习型 baseline：至少 `5` 个随机种子，最好 `10` 个随机种子。
- 启发式规则：确定性规则可单次，随机规则至少 `5` 次。
- 搜索算法：若有随机性，至少 `5` 次；若耗时过高，可以放入补充实验，但必须说明预算。
- 单次补充结果、跨规模 fallback 结果不能与多种子均值混在同一主表中。

建议主表格式：

| 方法 | 283 | 680 | 2338 | 3182 | 平均排名 | 推理耗时 |
|---|---:|---:|---:|---:|---:|---:|
| Full HB-GAT-PN | mean ± std | mean ± std | mean ± std | mean ± std | - | - |

### 3.3 多指标结果表

顶刊审稿通常不会接受只报告 makespan。建议增加多指标表：

| 方法 | Makespan | Balance | Worker Util. | Station Util. | Complete | Valid | Infer Time |
|---|---:|---:|---:|---:|---:|---:|---:|

该表可选择 `680` 或 `3182` 作为代表规模，也可在 appendix 中给出四规模完整表。

## 4. 消融实验设计

### 4.1 结构消融

建议优先补齐以下消融：

| 消融 | 目的 |
|---|---|
| full | 完整 HB-GAT-PN + Skill Hub |
| no-GAT | 验证图编码器贡献 |
| no-pointer | 验证 Pointer 解码贡献 |
| no-mask | 验证硬约束 mask 的必要性 |
| no-attention-critic / no-attention-pooling | 验证 attention pooling 或 critic 结构贡献 |
| no-skill-hub | 验证 Worker-Skill-Task 分解贡献 |
| skill-hub unidirectional | 验证反向消息是否必要 |

每个消融至少报告：

- makespan
- valid rate
- completion rate
- inference time
- GPU memory peak
- rollout SPS

### 4.2 Skill Hub 专项消融

Skill Hub 是当前方法最有论文价值的结构之一，建议单独成表：

| 图结构 | 技能边数 | 总边数 | 显存峰值 | Rollout SPS | Makespan | Valid |
|---|---:|---:|---:|---:|---:|---:|
| Worker-Task direct | - | - | - | - | - | - |
| Worker-Skill-Task | - | - | - | - | - | - |
| Worker-Skill-Task + bidirectional | - | - | - | - | - | - |

该表同时证明：

- 语义层面：Skill Hub 更符合 APAL 的技能资源逻辑。
- 计算层面：Skill Hub 显著减少 dense Worker-Task 技能边。
- 效果层面：Skill Hub 不只是加速，也能保持或提升调度质量。

### 4.3 表征特征消融

建议补充特征层面的消融，以回应 RESCHED 类论文对复杂状态的质疑：

| 消融 | 目的 |
|---|---|
| 去除 worker skill 特征 | 检验技能信息是否必要 |
| 去除 station capacity/slot 特征 | 检验工位容量建模是否必要 |
| 去除 dynamic event 特征 | 检验动态扰动输入是否必要 |
| 简化任务动态特征 | 检验复杂状态是否过度设计 |

如果部分消融实现成本较高，可作为 appendix 或 supplementary experiments。

## 5. 泛化实验

### 5.1 跨规模泛化

建议做三类泛化实验：

| 训练设置 | 测试设置 | 目的 |
|---|---|---|
| 283 训练 | 680/2338/3182 测试 | 小到大泛化 |
| 400-800 随机实例训练 | 283/680/2338/3182 测试 | 分布泛化 |
| 680 训练 | 283/2338/3182 测试 | 当前主模型泛化能力 |

指标：

- normalized makespan
- valid rate
- completion rate
- inference time
- relative degradation

### 5.2 跨扰动泛化

重调度模型建议测试：

- 在低强度扰动训练，高强度扰动测试。
- 在物料延迟训练，工时扰动/人员扰动测试。
- 在单一扰动训练，混合扰动测试。

这类实验能证明模型不是只记住某一种重调度场景。

## 6. 鲁棒性与重调度实验

### 6.1 扰动强度曲线

建议每类扰动画曲线，而不是只报单点：

| x 轴 | y 轴 |
|---|---|
| 物料延迟强度 | makespan、frozen violation、eligible rate |
| 工时扰动比例 | makespan、schedule change、completion rate |
| 工人不可用比例 | eligible rate、worker utilization、invalid count |
| 站位异常强度 | station utilization、slot violation、response time |

### 6.2 重调度对比方法

建议至少比较：

- 当前 Reschedule PPO。
- Reschedule GA。
- 规则修复策略，例如 CPM/SPT repair。
- 从扰动后状态重新全量调度。
- 原 baseline 不调整，即 no-reschedule。

### 6.3 重调度指标

重调度指标建议固定为：

- makespan
- reschedule score
- frozen violation
- release/material violation
- demand violation
- schedule change count
- start time deviation
- station change count
- eligible rate
- response time

其中 `schedule change` 和 `response time` 是工业可部署性关键指标，建议进入正文表格。

## 7. 复杂度、效率与工程可部署性

### 7.1 图复杂度表

建议每个规模报告：

| 数据集 | Task | Worker | Station | Skill | Precedence 边 | Skill 边 | 总边数 |
|---|---:|---:|---:|---:|---:|---:|---:|

同时报告 Skill Hub 前后的边数缩减比例。

### 7.2 训练与推理效率表

建议报告：

- train wall time
- eval wall time
- eval inference time
- rollout SPS
- GPU memory peak
- CPU num_envs
- AMP 设置
- batch size

顶刊论文中这张表的作用是证明方法不是只在小图上有效，而是能在大规模 APAL 上实际运行。

### 7.3 质量-效率 Pareto 图

建议画一张 `makespan vs inference time` 的散点图：

- x 轴：inference time。
- y 轴：makespan 或 normalized makespan。
- 点：SPT、CPM、Beam、IG、SA、GA、Basic PPO、DQN、Full HB-GAT-PN。

这张图可以直接展示：主方法是否处于较优的质量-效率区域。

## 8. 论文图表清单

建议正文至少包含：

1. APAL 问题图：task-station-worker-skill-time 的约束关系。
2. 模型结构图：HB-GAT-PN + Skill Hub + Pointer + mask。
3. 主结果表：四规模 makespan。
4. 多指标表：balance、utilization、validity、runtime。
5. 消融表：结构模块和 Skill Hub。
6. 泛化表：训练规模到测试规模。
7. 重调度扰动曲线。
8. 质量-效率 Pareto 图。

建议附录包含：

1. 所有 baseline 参数。
2. 所有数据集统计。
3. 所有随机种子结果。
4. 所有重调度场景明细。
5. 训练命令和配置表。

## 9. 优先级排序

### 第一优先级：最影响顶刊可信度

1. 补齐 Skill Hub 专项消融。
2. 补齐主方法、Basic PPO、DQN、no-GAT、no-pointer、no-skill-hub 的多种子统计。
3. 统一 baseline 预算和统计口径。
4. 补充多指标表，不再只报 makespan。
5. 补充重调度扰动强度曲线。

### 第二优先级：增强论文说服力

1. 新增 400/600/800/1000 随机实例训练与测试。
2. 做跨规模泛化表。
3. 做质量-效率 Pareto 图。
4. 做技能瓶颈场景。
5. 做图复杂度和显存/SPS 表。

### 第三优先级：放入补充材料

1. 更细粒度的特征消融。
2. 多种混合扰动组合。
3. 不同 PPO 超参数敏感性。
4. 不同 GAT heads/hidden dim 结构对比。
5. 更多随机数据实例的完整明细。

## 10. 推荐执行路线

建议按以下顺序推进：

1. 固定现有四规模主实验口径，整理现有结果，区分多种子、单次和 fallback。
2. 跑 no-skill-hub 与 Skill Hub 专项消融，优先证明当前核心结构贡献。
3. 跑 5-10 种子主方法与关键 baseline，形成正文主表。
4. 跑统一预算的启发式/搜索 baseline，形成公平对比。
5. 生成 400/600/800/1000 随机实例，做跨规模泛化。
6. 构建固定动态扰动场景库，做重调度强度曲线。
7. 汇总复杂度、显存、SPS、推理耗时，形成工程效率表。
8. 根据结果选择 1-2 个代表案例画甘特图或 station load 图。

## 11. 最终验收标准

实验补强完成后，论文应能回答以下审稿问题：

- APAL 为什么不是普通 FJSP/JSSP？
- HB-GAT-PN 为什么比 flat PPO/DQN 和启发式规则更适合 APAL？
- Skill Hub 是否真的减少图规模，并保持或提升解质量？
- 模型是否能跨规模泛化？
- 模型在动态扰动和重调度中是否稳定？
- 相比搜索算法，主方法是否在质量-效率上有优势？
- 是否报告了工业可部署性指标，而不是只报告 makespan？
- 所有结果是否可复现、统计口径是否一致？

如果这些问题都能用表格、曲线和消融实验证据回答，当前 APAL 论文的实验部分会更接近顶刊要求。
