# 联合决策对比与消融实验重构：第一至第三阶段报告

## Material Passport

- Artifact ID：`APAL-JOINT-ABLATION-PHASE1-3-20260718`
- 类型：代码实验实现与可复现性验证报告
- 验证状态：`VERIFIED`
- 领域：飞机脉动装配线（APAL）初始调度
- 训练功能筛查数据：`data/scale_400_800_datasets/syn_403_77.csv`
- 固定验证数据：真实 `data/680.csv`
- 随机种子：42
- 框架：PyTorch Lightning、PPO、PyTorch Geometric
- 功能筛查限制：每个变体仅训练 3 次完整 PPO 更新，不用于判断方法效果

## 结论

第一至第三阶段已经完成。九个核心变体均能完成 3 次 Lightning PPO 更新，训练排程完成率均为 100%，并在真实 680 数据上完成 temperature=0 的确定性验证。九次验证产生的完整排程均通过统一约束引擎，且每个变体均生成 latest 与 best checkpoint。

本次重构改变了主方法的动作概率语义、actor/critic 池化参数、优化器参数分组和 checkpoint 结构。因此，旧主方法 checkpoint 不能继续用于正式实验，主方法与九个变体都必须重新训练。

## 第一阶段：统一实验架构

九个核心变体统一到同一套模型、环境、奖励、数据加载和 Lightning 训练入口：

1. `full_joint`：工序—工位—工人联合决策。
2. `operation_station`：学习工序与工位，工人由确定性最早完工补全器选择。
3. `operation_only`：仅学习工序，工位与工人均由确定性补全器选择。
4. `fixed_preallocation`：排程前固定 100% 人员工位绑定。
5. `static_topq`：固定 worker logits 下的无放回条件抽样，不使用已选团队嵌入。
6. `homogeneous_graphsage`：合并关系后使用共享 GraphSAGE。
7. `no_message_passing`：仅节点嵌入，不执行图消息传递。
8. `mean_max_pooling`：actor 使用 mean+max 全局池化。
9. `local_only`：actor 工序头不使用全局池化，critic 仍保留全局状态估计。

固定预分配在 rollout 前执行严格配额检查；无法覆盖工位—技能需求时抛出 `workforce_preallocation_infeasible`，不会把不可行配置伪装成训练死锁。

旧的 `ablation_no_gat`、`ablation_no_pointer` 和旧 checkpoint 格式已明确拒绝。当前 checkpoint 元数据格式为 v2，显式记录动作层级、人员绑定、团队选择、图编码器和 actor 上下文模式。

## 第二阶段：约束与回归验证

- 专用九变体测试：18 项通过。
- 降阶计算裁剪、PPO 信号与梯度测试：36 项通过。
- 第一次全项目回归：255 项直接通过；3 项失败均为旧接口测试断言。
- 更新旧测试契约后，失败模块定向复测：8 项通过。
- 第三阶段编排、Lightning checkpoint 完成率门控与九变体配置：13 项通过。

主要修复包括：

- 虚拟层级节点统一使用 `(task_id, -1, [])`，不再虚构工位或工人动作。
- 图 PPO 与图 DDQN 基线使用相同的虚拟节点规范动作。
- operation-only 不再计算 station head；operation-only 与 operation-station 不再执行 worker pointer 的 PPO 重算。
- 未使用的动作头和池化模块从可训练参数中冻结。
- actor attention 参数不再错误地按 critic 梯度阈值裁剪。
- 标准验证只有在完成率 100% 时才有资格保存 best checkpoint。
- 完整验证排程在计算 makespan 前再次经过统一约束引擎。

## 第三阶段：九变体功能筛查

共同条件：

- 训练文件：`syn_403_77.csv`，加载后含 715 个图工序节点。
- 验证文件：真实 `680.csv`，加载后含 438 个图工序节点。
- seed：42。
- 更新次数：3。
- 验证频率：仅第 3 次更新后验证一次。
- 验证温度：0。
- AMP：Lightning `16-mixed`。
- Windows 本机：2 个 spawn 向量环境；Linux 正式训练仍由平台配置使用 16 个环境。
- 筛查模型：hidden dim 32、1 层图编码器、共享 actor/critic trunk；所有变体保持相同宽度和深度。

| 变体 | 状态 | 可训练参数 | 平均 SPS | 总耗时（秒） | 680 验证 makespan |
|---|---:|---:|---:|---:|---:|
| full_joint | 通过 | 70,926 | 21.70 | 277.9 | 1553.40 |
| operation_station | 通过 | 62,346 | 24.33 | 255.2 | 471.50 |
| operation_only | 通过 | 55,976 | 25.83 | 245.0 | 545.65 |
| fixed_preallocation | 通过 | 70,926 | 22.33 | 270.0 | 1804.11 |
| static_topq | 通过 | 68,846 | 23.00 | 262.6 | 1127.32 |
| homogeneous_graphsage | 通过 | 38,574 | 28.17 | 207.0 | 1572.04 |
| no_message_passing | 通过 | 36,366 | 29.67 | 195.8 | 1191.34 |
| mean_max_pooling | 通过 | 71,820 | 23.80 | 255.1 | 681.28 |
| local_only | 通过 | 65,644 | 25.30 | 243.8 | 672.70 |

这些 makespan 仅用于证明确定性验证链路能完成合法排程。模型只训练了 3 次更新且只有一个种子，严禁据此声称某个变体优于主方法。

`mean_max_pooling` 日志提示 `rag_env` 未安装可选的 `torch-scatter`，PyG 的 max scatter 使用回退实现。因此本轮运行时可用于发现数量级差异，但正式运行时比较必须在所有变体共享的固定依赖环境中重新测量。

## 数据集角色与实验表述

- 真实 680 文件继续作为验证和 checkpoint 选择数据，不是 held-out test。
- 283、2338、3182 只在锁定 checkpoint 后进行跨规模评价。
- 由于真实实例之间存在来源和嵌套关系，283、2338、3182 不能表述为“来源独立 held-out test”。
- 当前训练生成数据与真实 AO 存在高重合，正式论文应披露该事实，并把结论限定为 cross-scale validation。

## 影响范围与重训要求

| 对象 | 是否必须重训/重跑 | 原因 |
|---|---|---|
| 主方法 full_joint | 必须重训 | 动作 log-prob、池化参数、优化器分组和 checkpoint v2 均已变化 |
| 八个核心消融变体 | 必须分别训练 | 以前没有语义独立且可审计的对应 checkpoint |
| 旧 `no_gat`/`no_pointer` checkpoint | 禁止复用 | 旧开关不能代表当前 GraphSAGE、无消息传递或 static top-q 语义 |
| Simple-HeteroGAT-PPO、Graph-DDQN-APAL | 建议重跑 | 虚拟节点动作已规范化；排程通常不变，但运行时和动作日志会变化 |
| SPT、LPT、EDD、CPM、MSL、Beam、IG、SA、GA | 本次修改不要求重跑 | 未改动其核心排程逻辑；若最终统一实验环境或数据再变化，则应整体重跑 |

正式训练应使用 Lightning、固定 YAML 配置、5 个随机种子和锁定的 680 checkpoint 选择规则。完成选模后，再一次性评价 283、2338、3182，避免跨规模实例参与模型选择。

