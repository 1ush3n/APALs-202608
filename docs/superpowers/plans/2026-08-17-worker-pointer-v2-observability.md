# Worker Pointer v2 可观测性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 APAL 的 Worker Pointer v2 增加低开销、可归因的训练标量，并清理重复或不适用日志。

**Architecture:** 在既有 `WorkerPointerV2Diagnostics` 中累计 rollout 级候选空间和 EFT 排名；在既有梯度遍历中按参数名分桶。Lightning 继续只消费标量字典，因此不改变训练边界。

**Tech Stack:** Python、PyTorch、PyTorch Lightning、TensorBoard、pytest。

## Global Constraints

- 仅使用现有依赖与现有诊断类；不写入节点级或边级大张量。
- 默认行为、PPO 数值语义、APAL 合法性和异步验证目录保持不变。
- 代码注释使用中文，所有新增行为先写 pytest。

---

### Task 1: Rollout 选择质量标量

**Files:**
- Modify: `training/worker_pointer_v2_diagnostics.py`
- Modify: `ppo_agent.py`
- Test: `tests/test_worker_pointer_v2.py`

**Interfaces:**
- Consumes: 每步 invalid mask、动态 EFT 特征和已选 worker。
- Produces: `record_action_space(...)`、`record_selection(..., eft_rank_percentile=...)` 与 `PointerV2/ActionSpace/*`、`PointerV2/EFT/*` 标量。

- [ ] **Step 1: 写失败测试**

```python
diagnostics.record_action_space(
    ready_task_count=torch.tensor([2.0]),
    legal_station_count=torch.tensor([3.0]),
    legal_worker_count=torch.tensor([4.0]),
)
diagnostics.record_selection(
    selected_exposure=torch.zeros((1, 12)),
    entropy=torch.tensor([0.5]),
    eft_rank_percentile=torch.tensor([0.25]),
)
metrics = diagnostics.finalize(require_coverage=False)
assert metrics["PointerV2/ActionSpace/ReadyTaskMean"] == 2.0
assert metrics["PointerV2/EFT/SelectedRankPercentileMean"] == 0.25
```

- [ ] **Step 2: 确认红灯**

Run: `python -m pytest tests/test_worker_pointer_v2.py::test_v2_diagnostics_summarize_action_space_and_eft_rank -q`

- [ ] **Step 3: 最小实现**

```python
rank = (legal_eft < selected_eft.unsqueeze(1)).sum(dim=1).float()
percentile = rank / (legal_count - 1).clamp_min(1).float()
```

只在动态 EFT 开启时记录排名；所有候选空间统计按 rollout 平均并输出 worker 候选 P10。

- [ ] **Step 4: 确认绿灯**

Run: `python -m pytest tests/test_worker_pointer_v2.py::test_v2_diagnostics_summarize_action_space_and_eft_rank -q`

### Task 2: 模块梯度与日志清理

**Files:**
- Modify: `ppo_agent.py`
- Test: `tests/test_worker_pointer_v2.py`

**Interfaces:**
- Consumes: `policy.named_parameters()`。
- Produces: `Gradient/{GATEncoder,TaskHead,StationHead,WorkerV2,Critic}GradToParamRatio` 与动态 EFT 投影范数；不再输出重复或无语义标量。

- [ ] **Step 1: 写失败测试**

```python
metrics = PPOAgent._collect_gradient_diagnostics(named_parameters)
assert metrics["worker_v2_grad_to_param"] > 0.0
assert metrics["dynamic_eft_proj_grad_norm"] > 0.0
```

- [ ] **Step 2: 确认红灯**

Run: `python -m pytest tests/test_worker_pointer_v2.py::test_v2_gradient_diagnostics_separate_dynamic_eft_projection -q`

- [ ] **Step 3: 最小实现**

按已有参数名前缀分类；使用 `||g|| / max(||theta||, 1e-12)`，不复制参数，也不假装它是 Schedule-Free 的真实参数位移。

- [ ] **Step 4: 确认绿灯**

Run: `python -m pytest tests/test_worker_pointer_v2.py::test_v2_gradient_diagnostics_separate_dynamic_eft_projection -q`

### Task 3: 回归与提交

**Files:**
- Modify: 本计划涉及文件及对应实验配置。

- [ ] **Step 1: 运行本地回归**

Run: `python -m pytest tests/test_worker_pointer_v2.py tests/test_worker_pointer_v2_batched_replay.py tests/test_config_loader.py tests/test_verify_worker_pointer_v2_training_run.py -q`

- [ ] **Step 2: 检查差异并提交**

Run: `git diff --check`，随后只暂存上述任务文件并提交。
