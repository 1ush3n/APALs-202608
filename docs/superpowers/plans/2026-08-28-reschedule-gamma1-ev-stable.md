# 重调度 gamma=1 与价值函数稳定性实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 R5 重调度协议、模型结构、奖励权重、暖启动和随机种子不变的前提下，验证 gamma=1、有效 PPO batch=64、actor 学习率略高于 critic 对策略分数和 `Critic/Explained_Variance` 的影响。

**Architecture:** 复用现有 `reschedule_task_delay_r5_gamma9995_aligned` 配置的全部协议设置，只新增一份独立 YAML；不修改训练代码、不修改旧实验目录、不覆盖旧 checkpoint。训练最多运行 60 个 episode，按 40 episode 作为首个决策点，持续读取 TensorBoard、CSV、checkpoint 和进程状态。

**Tech Stack:** Python、Hydra 原生 `key=value` 配置、PyTorch Lightning、PPO、CUDA、TensorBoard、AutoDL screen。

**Spec:** 用户于 2026-08-28 提出的下一次 APAL R5 重调度训练要求。

## Global Constraints

- 必须在 `rag_env` 环境中执行，服务器使用 CUDA。
- 必须保持 `num_envs=4`、3 个 CUDA 异步验证 worker、R5 协议和 batch=64。
- 只允许新增独立实验配置和运行目录；不得删除、覆盖或修改旧实验资产。
- 训练最多 60 个 episode；至少在 episode 40 附近进行一次趋势判断。
- 重点监控 `Critic/Explained_Variance`，并同时检查 reward、selection_score、makespan、loss、梯度、熵、KL、clip fraction、GradientCoverage、NonFiniteCount、OOM 和日志连续性。
- 发现 NaN/Inf、CUDA 致命错误、连续 OOM、异步 worker 持续失败或明确持续恶化时，先安全停止并保留所有文件；不得静默绕过校验。

### Task 1: 创建独立实验配置

**Files:**
- Create: `conf/experiment/reschedule_task_delay_r5_gamma1_ev_stable.yaml`

**Interfaces:**
- Consumes: 现有 `conf/experiment/reschedule_task_delay_r5.yaml` 的 R5 协议、模型和异步验证默认值。
- Produces: 可由 `python train.py experiment=reschedule_task_delay_r5_gamma1_ev_stable` 加载的独立配置。

- [ ] 保持奖励权重：makespan=0.20、balance=0、takt violation=3、start stability=4、station change=4、team change=0.3。
- [ ] 设置 `train.gamma=1.0`、`train.batch_size=64`、`train.num_envs=4`、`train.max_episodes=60`。
- [ ] 设置 `experiment.lr=5e-5`、`actor_lr_multiplier=1.0`、`critic_lr_multiplier=0.8`，得到 actor=`5e-5`、critic=`4e-5`。
- [ ] 保持 `reschedule_warm_start=true`、动态 EFT、3 个 CUDA worker 和 objective-delta reward。

### Task 2: 本地必要预检

**Files:**
- Test: `tests/test_config_loader.py`
- Test: `tests/test_ppo_batch_semantics.py`
- Test: `tests/test_async_evaluation.py`

**Interfaces:**
- Consumes: Task 1 的配置。
- Produces: 配置解析、batch 语义和 R5 CUDA 异步验证约束的通过证据。

- [ ] 使用 `D:\Conda\envs\rag_env\python.exe` 解析新配置，确认 gamma=1、batch=64、actor=5e-5、critic=4e-5、num_envs=4、worker=3。
- [ ] 运行必要测试：

```powershell
D:\Conda\envs\rag_env\python.exe -m pytest tests/test_config_loader.py tests/test_ppo_batch_semantics.py tests/test_async_evaluation.py -q
```

### Task 3: 服务器独立启动

**Files:**
- Remote create: `/root/autodl-tmp/APALs-202608-main-method/conf/experiment/reschedule_task_delay_r5_gamma1_ev_stable.yaml`
- Remote run: `/root/autodl-tmp/APALs-202608-main-method/runs/reschedule_task_delay_r5_gamma1_ev_stable/`

**Interfaces:**
- Consumes: Task 1 的配置、现有代码和已验证的 warm-start checkpoint。
- Produces: 新的 `run_manifest`、训练 TensorBoard、异步验证 TensorBoard、`last.ckpt`、可能的 `best.ckpt` 和日志。

- [ ] 先确认服务器 GPU、磁盘、旧进程和旧目录均未被覆盖。
- [ ] 用服务器已配置的 `rag_env`/base Python，通过 `screen` 后台启动：

```bash
cd /root/autodl-tmp/APALs-202608-main-method
screen -dmS apal_gamma1_ev bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && CUDA_VISIBLE_DEVICES=0 python -u train.py experiment=reschedule_task_delay_r5_gamma1_ev_stable seed=42 num_envs=4'
```

- [ ] 启动后立即确认日志打印的 run_id、CUDA、num_envs=4、batch_size=64、actor/critic 学习率、gamma=1 和 3 个异步 worker。

### Task 4: 在线比较与决策

**Files:**
- Read: 新 run 的全部 TensorBoard event、CSV、best.json、best.ckpt、last.ckpt。
- Compare: 旧 ent40、gamma9995 aligned、本次新 run。

**Interfaces:**
- Consumes: Task 3 的运行资产。
- Produces: episode 40 和 episode 60 的对比结论及是否保留该方向的决策。

- [ ] 每次检查枚举全部 scalar tag，记录最新值、窗口均值、范围、NaN/Inf 和趋势；单独列出 `Critic/Explained_Variance`。
- [ ] 检查 reward、selection_score、makespan、eligible、可行性、目标分量、loss、梯度、熵、KL、clip fraction、GradientCoverage、NonFiniteCount、OOM 和日志连续性。
- [ ] 不因单次未提升或普通平台期停止；只有自然完成、明确不可恢复故障或满足预先设定的持续退化证据才收尾。

## 自检

- 配置只新增 1 个文件，未修改训练实现。
- `critic_lr_multiplier` 明确存在于当前配置模型，且有效 actor/critic 学习率可由 `lr` 与倍率直接计算。
- `train.batch_size=64` 覆盖了自适应 batch 中间任务规模的回退路径。
- 旧实验目录和 checkpoint 不作为写入目标。
