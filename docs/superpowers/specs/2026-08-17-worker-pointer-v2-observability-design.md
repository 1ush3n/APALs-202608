# Worker Pointer v2 可观测性设计

## 目标

以最小增量区分 APAL 训练退化的三类来源：随机状态难度、Worker Pointer 的实际选择质量，以及工序/工位/工人/GAT/Critic 模块的更新失衡。

## 范围

- 每次 rollout 汇总候选空间大小：可选工序数、所选工序的合法工位数、每个工人自回归步骤的合法工人数。
- 仅在启用动态 EFT 特征时，记录所选工人的 EFT 候选排名分位数；0 表示合法候选中 EFT 最小，1 表示最大。
- 每次 PPO 更新记录 GAT 编码器、工序头、工位头、Worker Pointer v2、Critic 的梯度范数与参数范数之比；动态 EFT 投影额外记录参数范数和梯度范数。
- 仅记录标量均值与分位数，不记录节点 embedding、边注意力或动作序列。
- 将重复或不适用字段停止写入：进度条别名 `Rew/Mk/SPS`、重复学习率 `Train/LearningRate`、未启用 APCF/蒸馏字段，以及 batched-v2 下无语义的 exact 重算误差。

## 不变式

- 不改变 APAL 动作、掩码、奖励、PPO 损失、优化器和异步验证流程。
- 诊断在 rollout 内仅保存小标量张量，结束时一次搬运 CPU。
- 异步验证继续写入独立 event 目录和 `async_eval_summary.csv`；不允许多进程写同一个主 event 文件。

## 验证

- 单元测试验证候选空间统计、EFT 排名与模块梯度分组的确定性数值。
- 回归 Worker Pointer v2、batched replay、配置加载与训练产物审计测试。
