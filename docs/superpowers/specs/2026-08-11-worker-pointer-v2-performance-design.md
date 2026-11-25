# WorkerPointer v2 性能验证与优化设计

## 目标

在不改变 WorkerPointer v2 的动作语义、PPO 序列概率或首次重算合同的前提下，逐项确认三项解码优化的真实性能收益；只对通过筛选的累积版本执行一次真实四环境、bf16、三更新确认运行。

## 范围与边界

- 只作用于 `team_selection_mode=autoregressive_pressure_v2` 与 `policy_action_scope=operation_station_worker`。
- 不修改 legacy `autoregressive`、CTG、APCF、正式 checkpoint、数据资产或正式结果台账。
- `main` 仅用作原始自回归的行为与本机端到端性能参照，不回合并其代码。
- 所有测试临时产物写入项目内 `.pytest_tmp_v2/`；真实确认 run 继续写入新的 `results/01_initial_main/initial_worker_pointer_v2_exploratory/<run_id>/`，并标记 `training_auto_eval_only`。

## 固定基准

基准脚本从既有、经 manifest 权威绑定的真实 400--800 工序训练池取得固定的一批冻结状态。每次测量固定：模型 checkpoint/随机种子、四环境批次、bf16 配置、任务与工人掩码、动作轨迹。

脚本同时记录：

- CUDA event 的 GPU 时间；
- `perf_counter` 加 CUDA 同步的端到端时间；
- 每个团队与每次工人选择的时间分布（中位数、P10、P90）；
- logits 等价性、采样动作一致性、worker log-prob 重算误差和梯度有限性。

常规 rollout 的异步 `ForwardMs` 不作为优化归因依据；最终是否放行仍使用完整 rollout 的端到端 SPS 与显存。

## 优化候选与顺序

### 1. GPU 原始特征复用

批量 rollout 已将 `batch_obs` 上传 GPU。v2 压力上下文必须直接使用按图切片后的 GPU 原始 task/worker 特征，而不是再次从 `obs_list[i]` 的 CPU 特征上传。工人技能同样在每个团队开始时一次性置于 GPU，禁止成员选择循环中的小型 H2D 拷贝。

此项不改变压力公式、掩码或模型输入值。

### 2. 单一解码缓存

新增仅在一个 `(observation, task, station, team)` 解码生命周期内存在的 `WorkerPointerV2DecodeCache`。它保存：

- `candidate_keys = v2_key_proj([worker_embedding, candidate_exposure, candidate_max_exposure])`；
- float32 的静态 query 特征：任务、工位、全局图上下文、长期压力、近期压力；
- 固定需求人数张量。

每一步仍重新计算 DeepSets 团队表示、团队技能消耗和选择进度，并保持：

`score = v2_attn(tanh(query + candidate_keys))`。

缓存不得跨动作、环境、模型参数版本或 PPO update 使用；PPO 重算必须以当前参数重新建缓存以保留正确梯度图。

### 3. 移除 v2 无用的 legacy 团队均值

v2 路径不创建或更新 `current_team_emb`。legacy 路径保持原样。v2 继续只通过 `WorkerPointerV2State` 更新 DeepSets 状态。

## 正确性合同

每项优化都必须先有失败测试，随后满足：

- 优化前后 float32 logits 在同一输入与 mask 下严格等价或在 `1e-6` 以内；
- 随机采样使用同一 RNG 状态时动作与 worker 序列完全一致；
- CPU PPO 首次重算 MaxAE 不超过 `1e-4`，GPU bf16 不超过 `1e-3`；
- v2 新模块梯度有限、至少一个非零，并仍位于 actor 优化器；
- legacy 路径不构造 v2 cache，旧 checkpoint 严格加载不受影响。

## 筛选与确认

每个单项优化先在短同步 microbenchmark 中与紧邻前一版本比较。只有端到端中位耗时呈明确下降、且所有正确性合同通过的优化才进入累积版本；无收益项不合入。

通过筛选的最终累积版本再运行一个全新、不可覆盖的三更新冷启动：真实 80 项 `variant_*.csv` manifest、seed=42、四环境、bf16、GradScaler 关闭。审计继续要求 100% 完成率、无非法动作/OOM/NaN/Inf、重算 MaxAE 不超过 `1e-3`。

最终与本机同口径 legacy Gate 1 run 比较后两次更新的中位 SPS 和峰值显存。只有重新通过 v2 不低于 legacy 85% SPS、额外显存不超过 512 MiB 且绝对峰值低于 7.5 GiB 时，才允许讨论 Gate 2 的 30 更新 burn-in。
