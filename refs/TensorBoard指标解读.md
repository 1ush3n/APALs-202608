# TensorBoard 指标完全解读手册

> 本手册覆盖项目中所有写入 TensorBoard 的监控指标（共 16 组），按前缀分组解释其含义、数据来源和趋势判断方法。

---

## 1. Reward/ — 训练奖励类

| 指标 | 含义 | 来源 | 健康趋势 |
|------|------|------|---------|
| `Reward/Episode_Avg` | 每个 Episode 中所有并行环境的平均累积奖励 | `train.py` — 各 env 的 `ep_reward` 求和/N | **稳步上升** → 策略在变好；剧烈抖动 → 奖励量级或价值函数不稳定 |
| `RewardDetail/Makespan_Penalty_Mean` | 每步 makespan（完工时间）增量惩罚的 episode 均值 | `info['makespan_penalty']` | **趋近于 0** → makespan 不再恶化；持续负值 → 排程质量差 |
| `RewardDetail/Load_Balance_Penalty_Mean` | 每步站位负载不均衡惩罚的 episode 均值 | `info['std_penalty']` | **趋近于 0** → 站位负载均匀 |
| `RewardDetail/Deadlock_Penalty_Mean` | 死锁惩罚的 episode 均值 | 死锁发生时注入的惩罚 | **始终为 0** → 无死锁；出现负值 → 发生了死锁 |
| `RewardDiagnostic/resource_wait_penalty_candidate` | 工人/站位等待时间的潜在惩罚 | `environment.py` `info` 字段 | 辅助诊断，正常应在小范围浮动 |
| `RewardDiagnostic/resource_idle_penalty_candidate` | 资源闲置的潜在惩罚 | `info` 字段 | 同上 |
| `RewardDiagnostic/team_wait_h` | 工人集结完毕的等待时长（小时） | `info` 字段 | 越小越好，持续 >5h 说明工人调度有问题 |
| `RewardDiagnostic/station_wait_h` | 站位槽位释放等待时长（小时） | `info` 字段 | 越小越好，持续 >10h 说明站位是瓶颈 |
| `RewardDiagnostic/worker_idle_ratio_before` | 执行动作前工人空闲占比 | snapshot 计算 | 反映资源利用水平 |
| `RewardDiagnostic/station_slot_vacancy_ratio_before` | 执行动作前站位空槽占比 | snapshot 计算 | 反映物理空间利用水平 |

---

## 2. Train/ — 训练绩效类

| 指标 | 含义 | 来源 | 健康趋势 |
|------|------|------|---------|
| `Train/WallClock_Makespan_Avg` | 每个 Episode 所有并行环境的平均最大完工时间（物理小时） | `np.max(env.station_wall_clock)` | **持续下降** → 策略在学习；停滞 → 需调整超参或进入局部最优 |
| `Train/Deadlock_Rate_Batch` | 该 Episode 中发生死锁的 env 占比 | `ep_is_deadlock` | **始终为 0**；>0.1 说明约束太紧或工人不够 |
| `Train/Avg_Task_Completion_Rate` | 任务完成率（已完成/总任务 × 100%） | `assigned_tasks` / `num_tasks` | **始终为 100%**（除非死锁） |

---

## 3. Loss/ — PPO 损失类

这些值记录的是**经过 `accumulation_steps` 反归一化**的原始损失（即真实梯度量级），在 `ppo_agent.py` update 阶段计算。

| 指标 | 含义 | 计算方式 | 健康趋势 |
|------|------|---------|---------|
| `Loss/Total` | PPO 总损失（pol+val+ent 的加权和） | `c_pol * policy_loss + value_loss + entropy_loss` | **逐步下降**但不能归零（归零 = 策略崩溃） |
| `Loss/Policy` | 策略损失 | `-min(surr1, surr2).mean()` | **小幅震荡**，绝对值稳定在 0.001–0.1 范围 |
| `Loss/Value` | 价值网络损失 | `c_val * MSE(value_pred, returns)` | **持续下降到稳定值**；若突然飙升 → 价值函数崩溃 |
| `Loss/Entropy` | 熵正则项 | `-c_ent * entropy.mean()` | **从负逐渐趋近于 0**（探索到利用的自然过渡） |

---

## 4. Entropy/ — 策略熵类

三个指针对应级联自回归策略的**三个决策分支**各自的策略熵（不确定性），于 PPO update 阶段计算。

| 指标 | 含义 | 健康范围 | 异常信号 |
|------|------|:---:|------|
| `Entropy/Task` | 工序选择的熵 | 0.5–3.0 | **<0.1** → 选择高度确定性，可能过早收敛；**>4.0** → 过于随机 |
| `Entropy/Station` | 站位选择的熵 | 0.3–1.5 | **<0.05** → 站位选择单一化（坍缩到某个站位） |
| `Entropy/WorkerTeam` | 工人选择的熵 | 1.0–3.5 | **<0.3** → 选人模式固化；**>5.0** → 工人选择纯随机 |

> 三个熵都随着训练进行呈**指数衰减**（由 `c_entropy / entropy_decay_episodes` 控制），这是预期行为。如果刚训练就全归零，说明 `c_entropy` 设太小。

---

## 5. Critic/ — 价值网络诊断类

这是判断 PPO 训练是否健康的**最重要的一组指标**。

| 指标 | 含义 | 健康标准 |
|------|------|:---:|
| `Critic/Explained_Variance` | 价值网络对 Returns 的解释力 | **>0.6 优秀**，0.3–0.6 一般，**<0.1 严重问题** |
| `Critic/Value_Predictions_Mean` | 价值网络的平均预测值 | 与 `Target_Returns_Mean` **匹配**（差值 <20%） |
| `Critic/Target_Returns_Mean` | GAE 计算的 Target Returns 均值 | 提供 Scale 参考 |
| `Critic/Absolute_Error_Mean` | 价值预测的绝对误差均值 | **逐步下降到稳定值** |
| `Critic/Advantage_Mean` | GAE 优势函数的均值 | **理想趋近于 0**（正值偏差说明多数动作"比预期好"，负值偏差说明"比预期差"） |
| `Critic/Advantage_Std` | 优势函数的标准差 | 反映本轮 rollout 中"好"与"坏"动作的区分度 |
| `Critic/Gaze_Variance` | 站位注意力权重的方差 | **持续上升**说明 Critic 正聚焦特定瓶颈站位 |

### 重点解读

**Explained Variance** 是最核心的诊断信号：

```
Explained_Variance ≈ 1 - Var(Returns - V_pred) / Var(Returns)

> 0.8  → Critic 拟合优秀
> 0.5  → 及格
> 0.2  → 需检查：学习率太大/太小？Returns 量级是否稳定？
< 0.0  → 价值函数逆拟合（比瞎猜还差），必须停机排查
```

---

## 6. Policy/ — 策略更新诊断类

| 指标 | 含义 | 来源 | 健康标准 |
|------|------|------|:---:|
| `Policy/ApproxKL` | 新旧策略的近似 KL 散度 | `(exp(log_ratio)-1) - log_ratio` | **0.001–0.02** 正常；**>0.02 触发 KL 熔断**（`loss *= 0.01`） |
| `Policy/ClipFraction` | 被 PPO clip 截断的样本比例 | `ratios 超出 [1-ε, 1+ε]` 的比例 | **0.05–0.2** 正常（ε=0.2 裁剪约 10-20%）；**>0.5** 说明 ε 太小或策略变化太剧烈 |
| `Policy/RatioMean` | 新/旧策略概率比 `π_new/π_old` 的均值 | `exp(log_ratio).mean()` | **1.0 附近**（无偏）；>1.05 或 <0.95 说明策略漂移 |
| `Policy/RatioStd` | 概率比的标准差 | — | <0.3 正常；>0.5 说明某些动作的策略变化极端 |
| `Policy/Meltdown_Count` | 本轮 update 中被 KL 熔断机制保护的 batch 数量 | KL > `kl_early_stop` 的 batch 数 | **0 理想**；偶尔 1-3 可接受；频繁 >5 需降低学习率 |

---

## 7. Training/ — 训练调度类

| 指标 | 含义 | 来源 |
|------|------|------|
| `Train/LearningRate` | 当前学习率 | `optimizer.param_groups[0]['lr']` |
| `Train/ActorLearningRate` | Actor 组学习率 | 同上（ScheduleFree 下与总 LR 一致） |
| `Train/CriticLearningRate` | Critic 组学习率 | `param_groups[1]`（若存在独立分组） |

---

## 8. Memory/ — 显存类

| 指标 | 含义 | 单位 | 来源 |
|------|------|:---:|------|
| `Memory/Allocated_GB` | PyTorch 已分配的显存 | GB | `torch.cuda.memory_allocated()` |
| `Memory/Reserved_GB` | PyTorch 缓存池预留的显存 | GB | `torch.cuda.memory_reserved()` |

---

## 9. PPO/ — 工程健壮性类

| 指标 | 含义 | 健康标准 |
|------|------|:---:|
| `PPO/GPURebuildFallbackCount` | GPU 批量重建回退次数 | **0**（回退说明数据形状不符预期） |
| `PPO/BatchVectorRepairCount` | Batch 向量修复次数 | **0**（修复说明有维度不一致的样本） |

---

## 10. APAL/ — 调度领域专属诊断类

这些指标在 rollout 每步由 `compute_apal_rollout_diagnostics()` 计算，直接从环境 snapshot 提取：

| 指标 | 含义 | 健康趋势 |
|------|------|---------|
| `APAL/schedulable_tasks` | 当前可调度（非 Mask）的任务数 | 稳定在 3–15 范围为合理，归零 = 即将死锁 |
| `APAL/avg_resource_wait_h` | 工人等待时间 + 站位等待时间的均值（小时） | **逐步下降**；持续 >20h 说明资源瓶颈严重 |
| `APAL/station_slot_vacancy_ratio` | 站位空槽占总槽位的比例 | 0.2–0.6 正常；**<0.1** 说明站位永远满负荷 |
| `APAL/worker_idle_ratio` | 空闲工人占比 | 0.1–0.5 正常；**<0.05** 说明工人永远在忙（存在过载风险） |
| `APAL/critical_start_offset_h` | 关键路径（CPM）上任务的实际开始时间与理论最早开始时间的偏移（小时） | **逐步下降** → 关键路径在加速 |

---

## 11. Eval/ — 评估类

每隔 `eval_freq` 个 episode 在固定种子验证集上评估：

| 指标 | 含义 | 来源 |
|------|------|------|
| `Eval/WallClock_Makespan` | 验证集的最大完工时间 | `np.max(env.station_wall_clock)` |
| `Eval/Workload_Balance_Std` | 站位负载标准差 | `np.std(station_loads)` |
| `Eval/Average_Return` | 评估 episode 的总奖励 | 累积 reward |
| `Eval/Inference_Time_sec` | 完整评估的推理耗时 | wall-clock 时间 |
| `Eval/Worker_Utilization` | 工人利用率 | 忙时/总工作时间 |
| `Eval/Station_Utilization` | 站位利用率 | 忙时/总工作时间 |
| `Eval_Scenario/{name}_Makespan` | 多情景评估（标准/加噪/缺勤/故障）的 makespan | 各情景独立评估 |
| `Eval_Scenario/{name}_Reward` | 对应情景的奖励 | — |
| `Eval_Scenario/{name}_WorkerUtil` | 对应情景的工人利用率 | — |
| `Eval_Scenario/{name}_StationUtil` | 对应情景的站位利用率 | — |

---

## 12. Rollout/ — 前向采样速度类

每隔 `rollout_profile_interval` 个 episode 记录（默认 10）：

| 指标 | 含义 | 健康值 |
|------|------|:---:|
| `Rollout/EpisodeTotal_s` | 该 episode 总耗时 | 越小越好 |
| `Rollout/Mask_ms` | 每步 mask 获取平均耗时 | <5ms |
| `Rollout/DeadlockCheck_ms` | 每步死锁检测平均耗时 | <1ms |
| `Rollout/Select_ms` | 每步 GPU 动作选择平均耗时 | 取决于图大小，80–200ms 正常 |
| `Rollout/Snapshot_ms` | 每步状态快照平均耗时 | <2ms |
| `Rollout/Step_ms` | 每步行进仿真平均耗时 | 5–20ms |
| `Rollout/TotalPerStep_ms` | 单步五者之和 | **优化的核心 KPI** |
| `Rollout/EnvStepsPerSec` | 每秒可执行的环境步数 | **吞吐量核心 KPI**，越高越好 |
| `Rollout/StepsPerEpisode` | 该 episode 的总步数 | 反映序列长短 |
| `Rollout/FastPathEnabled` | Fast Path 是否启用 | 1.0 = 开启 |
| `Rollout/GPUAllocatedGB` | GPU 当前已分配显存 | 单位 GB |

---

## 13. RescheduleEval/ — 预测-反应式重调度评估类

> **前置条件**：仅在 `enable_reschedule_mode=True` 时生效。
> 重调度模式会基于初始调度（baseline schedule）的完工时间作为 **节拍时间（Takt Time）**，
> 模拟生产执行到中期某时刻发生工序延迟，需要在剩余任务上重新调度。
> 评估使用固定种子（30300+），确保每次评估的扰动场景一致可比。

### 13.1 核心绩效

| 指标 | 含义 | 来源 | 健康趋势 |
|------|------|------|---------|
| `RescheduleEval/Makespan` | 重调度后的最终完工时间（小时） | `np.max(station_wall_clock)` | **趋近于或低于 baseline makespan**；高于 takt 说明重调度产生超期 |
| `RescheduleEval/Reward` | 重调度 episode 的总奖励（已扣除稳定性惩罚） | 累积 reward | **稳步上升**；负值越大说明超期/偏离越严重 |
| `RescheduleEval/WorkerUtil` | 工人利用率 | `_compute_assignment_utilization` | >0.5 优秀；<0.2 说明工人大量闲置 |
| `RescheduleEval/StationUtil` | 站位利用率 | 同上 | >0.4 优秀；<0.15 说明站位空置严重 |
| `RescheduleEval/complete` | 评估完成率（所有 scenarios 中成功分配的占比） | `complete` flag | **始终为 1.0**；（<1.0 说明有场景发生死锁） |
| `RescheduleEval/invalid_step_count` | 每次评估中非法动作步数 | info['invalid_action'] | **始终为 0** |

### 13.2 约束违规诊断

> ⚠️ 这组指标是重调度的 **"体检报告"**，任何指标 > 0 都意味着调度解违反了 APAL 硬约束。

| 指标 | 含义 | 健康标准 |
|------|------|:---:|
| `RescheduleEval/frozen_violation_count` | 已冻结（重调度起始时刻之前已开工）任务的排程被非法修改的次数 | **= 0** |
| `RescheduleEval/release_violation_count` | 工序在物料释放时间之前被提前安排的次数 | **= 0** |
| `RescheduleEval/precedence_violation_count` | 违反工艺优先关系（前驱未完成就开始后续任务）的次数 | **= 0** |
| `RescheduleEval/worker_overlap_violation_count` | 同一工人在两个任务时间重叠的违规次数 | **= 0** |
| `RescheduleEval/station_slot_violation_count` | 站位并发任务数超过 `max_slots_per_station` 的次数 | **= 0** |
| `RescheduleEval/skill_violation_count` | 工人不具备技能却被派工的次数 | **= 0** |
| `RescheduleEval/demand_violation_count` | 派工人数少于工序需求人数的次数 | **= 0** |
| `RescheduleEval/duplicate_task_count` | 同一工序被重复分配的次数 | **= 0** |
| `RescheduleEval/missing_task_count` | 评估结束时仍有未完成工序的数量 | **= 0**（=死锁或推理中断） |

### 13.3 节拍与可行下界

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `RescheduleEval/takt_h` | 初始调度的完工时间，即节拍时间（Takt Time） | `baseline.makespan` |
| `RescheduleEval/takt_violation_h` | 重调度 makespan 超出节拍的小时数 | `max(0, final_makespan - takt)` |
| `RescheduleEval/lower_bound_h` | 理论下界：frozen 任务结束时间 / 剩余工作量 / 最晚释放时间 三者取大 | `calculate_reschedule_lower_bound()` |
| `RescheduleEval/takt_feasible` | 下界是否 ≤ 节拍（1.0=可在节拍内完成，0.0=不可能） | — |

> **解读**：若 `takt_feasible = 0.0`，说明仅凭静态工作量就已不可能在原定节拍内完成，此时稳定性惩罚权重会按 `reschedule_infeasible_stability_relax` 放松，允许智能体更大胆地偏离 baseline 以求完工。

### 13.4 稳定性偏离度

| 指标 | 含义 | 计算方式 | 健康趋势 |
|------|------|---------|---------|
| `RescheduleEval/start_deviation_mean_h` | 可移动任务（重调度时刻之后才开工的）的开工时间与 baseline 的绝对偏差均值（小时） | `calculate_stability_metrics()` | **逐步下降后稳定**（>0 是必然的，因为有延迟） |
| `RescheduleEval/station_change_rate` | 可移动任务中被改到与原 baseline 不同站位的比例 | `/movable_count` | **趋近 0**（越低越"尊重"初始方案的站位安排） |
| `RescheduleEval/team_change_rate` | 可移动任务中被换了不同工人的比例 | `/movable_count` | **趋近 0**（越低越"尊重"初始方案的班组分配） |

> **解读**：这三个指标是重调度特有的"保真度"（Fidelity）衡量。
> `station_change_rate` 和 `team_change_rate` 越低，说明重调度方案越接近初始方案，
> 工厂现场的物理扰动就越小。但它们不能为 0——因为延迟本身就需要调整计划来吸收。

---

## 14. RewardDiagnostic/ — 重调度训练期诊断类

> 以下指标在训练期间的每个 Episode 由 `info` dict 累积写入，仅在 `enable_reschedule_mode=True` 时非零。

| 指标 | 含义 | 来源 | 健康趋势 |
|------|------|------|---------|
| `RewardDiagnostic/reschedule_takt_violation_h` | episode 末 makespan 超出节拍的小时数 | `info['reschedule_takt_violation_h']` | **趋近 0** |
| `RewardDiagnostic/reschedule_start_deviation_mean_h` | 可移动任务开工时间偏离 baseline 的均值 | `info['reschedule_start_deviation_mean_h']` | 稳定在小范围 |
| `RewardDiagnostic/reschedule_station_change_rate` | 站位变更率 | `info['reschedule_station_change_rate']` | 趋近 0 |
| `RewardDiagnostic/reschedule_team_change_rate` | 团队变更率 | `info['reschedule_team_change_rate']` | 趋近 0 |
| `RewardDiagnostic/reschedule_stability_penalty` | 稳定性惩罚项（已乘系数） | `info['reschedule_stability_penalty']` | **逐步下降** |

---

## 15. 训练全流程健康诊断速查表

| 症状 | 可能原因 | 优先查看的指标 |
|------|---------|------|
| Makespan 不下降 | 策略陷入局部最优 | `Policy/ApproxKL`（是否过早归零）、`Entropy/Task`（是否太低） |
| Loss 震荡剧烈 | 学习率太大或 batch size 太小 | `Train/LearningRate`、`Policy/RatioStd`、`Critic/Explained_Variance` |
| Value Loss 飙升 | Returns 量级不稳定 | `Critic/Target_Returns_Mean`、`Reward/Episode_Avg` |
| 死锁频繁 | 工人或站位约束过紧 | `Train/Deadlock_Rate_Batch`、`APAL/schedulable_tasks`、`APAL/worker_idle_ratio` |
| KL 频繁熔断 | 学习率太大或 epoch 太多 | `Policy/Meltdown_Count`、`Policy/ApproxKL` |
| 熵过早归零 | `c_entropy` 太小或衰减太快 | `Entropy/Task`、`Entropy/Station`、`Entropy/WorkerTeam` |
| GPU 速度慢 | 未挂载 Fast Path 或显存不够 | `Rollout/EnvStepsPerSec`、`Rollout/GPUAllocatedGB` |
| Critic 拟合失败 | Returns 方差大或网络容量不足 | `Critic/Explained_Variance`（<0.1 需立即停机） |
| 重调度约束违规 | 冻结任务/优先关系/技能被破坏 | `RescheduleEval/frozen_violation_count`、`RescheduleEval/precedence_violation_count`、`RescheduleEval/skill_violation_count` |
| 重调度节拍超期 | 扰动太大或策略不够激进 | `RescheduleEval/takt_violation_h`、`RescheduleEval/takt_feasible`；若 `takt_feasible=0` 说明客观上不可达 |
| 重调度稳定性偏离过大 | 策略"完全重排"而非"微调" | `RescheduleEval/station_change_rate`（>0.5 说明大量站位被改动）、`RescheduleEval/team_change_rate` |

---

## 16. 按训练阶段关注优先级

| 阶段 | 重点关注 | 次要关注 |
|:---|------|------|
| **第 1–10 个 Episode** | `Train/WallClock_Makespan_Avg`（基线）、`Train/Deadlock_Rate_Batch`（约束是否可行） | `Critic/Explained_Variance`（CRITIC 是否在学） |
| **第 10–100 个 Episode** | `Critic/Explained_Variance`（>0.3 才算步入正轨）、`Loss/Value`（持续下降） | `Policy/ApproxKL`、`Entropy/*` |
| **第 100–300 个 Episode** | `Eval/WallClock_Makespan`（验证集下降趋势）、`Policy/ClipFraction` | `APAL/*`（调度质量诊断） |
| **训练全程** | `Policy/Meltdown_Count`（不应频繁熔断）、`Rollout/EnvStepsPerSec`（吞吐是否稳定） | `Memory/*`（是否在泄漏） |
| **🧩 重调度模式** | `RescheduleEval/frozen_violation_count`（必须=0）、`RescheduleEval/takt_violation_h`（能否在节拍内完成） | `RescheduleEval/station_change_rate`（稳定性保真度）、`RescheduleEval/lower_bound_h`（下界可达性） |
