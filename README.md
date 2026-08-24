# APAL 动态调度系统（HB-GAT-PN + PPO）

本项目面向 Aircraft Pulse Assembly Lines（APAL）初始调度与动态重调度，核心组件包括异构图注意力网络、指针网络、PPO、PyTorch Geometric 与 PyTorch Lightning。

## 环境

- Python 环境：`rag_env`
- Windows：自动加载 `conf/hardware/windows_4060_low_memory.yaml`，使用 `spawn`
- Linux：自动加载 `conf/hardware/linux_server.yaml`，使用 `forkserver`
- GPU：Lightning `16-mixed` AMP
- FP32 矩阵乘法：默认 `torch.set_float32_matmul_precision("high")`

```powershell
conda activate rag_env
pip install -r requirements.txt
```

PyTorch 应根据官方安装矩阵单独安装与显卡驱动兼容的 CUDA wheel。

## 生成训练池

生成 283、680、2338、3182 四档训练池，每档默认 32 个变体：

```powershell
python scripts/generate_initial_buckets.py bucket=all num_samples=32 seed=42
```

覆盖已有训练池：

```powershell
python scripts/generate_initial_buckets.py bucket=all num_samples=32 seed=42 overwrite=true
```

## 训练

训练入口会根据操作系统自动追加硬件配置，不需要手动传入 `conf/hardware/*.yaml`。

```powershell
python train.py experiment=initial_schedule_283
python train.py experiment=scale_400_800_schedule
python train.py experiment=initial_schedule_2338
python train.py experiment=initial_schedule_3182
```

命令行参数会覆盖 YAML 和自动平台配置。常用参数可直接传入：

```powershell
python train.py experiment=scale_400_800_schedule train.batch_size=256 train.num_envs=16 train.max_episodes=100
python train.py experiment=scale_400_800_schedule use_skill_hub=false
```

其他 `Config` 字段直接使用 Hydra `key=value` 覆盖：

```powershell
python train.py experiment=scale_400_800_schedule lr=0.00005 eval_scenarios=[standard]
```

分组名前缀会落到当前扁平 `Config` 字段：

```powershell
python train.py experiment=initial_schedule_283 train.batch_size=24 parallel.num_envs=4 artifacts.run_id=debug_260630-153000
```

配置优先级为：代码默认值 `<` 实验 YAML `<` 平台硬件 YAML `<` Hydra 命令行 `key=value`。

### APCF：锚点条件完整团队提议与反事实门控（初始调度）

APCF（Anchor-Conditioned Full Team Proposal with Counterfactual Gating）是初始
调度的主方法扩展，动作链为 `(o, s, H, P, z, T)`：锚点团队 `H` 由启发式生成，
提议器自回归生成完整合法团队 `P`，反事实门控 `z` 在两者间选择，随后执行
`T`（换人序列）。三大组件：

1. **反事实预训练数据**：确定性轨迹上采样决策点，强制替换候选团队后完整续排，
   记录相对收益 `y = (C(H) − C(P)) / max(C(H), ε)`；
2. **双头模型**：`AnchorConditionedTeamPointer`（自回归完整团队提议）与
   `AnchorProposalGate`（价值头 + 门控，未预训练时零初始化保证温度 0 必选锚点）；
3. **可追溯 PPO 微调**：`FrozenAnchorProposalTrace` 冻结决策点，重算对数概率必须
   与 rollout 期完全一致，且 z=0 时提议链仍计入 `log π`。

反事实数据构建（分位状态采样 + 候选预算：锚点/单换/双换/哈希代表）：

```powershell
python -X utf8 scripts/build_anchor_proposal_cf_data.py `
  --workers 6 `
  --worker-torch-threads 1
```

产物为 `data/initial_anchor_proposal_cf_v1/`：manifest.json（split 96/24/40、
候选预算、每样本 SHA-256、obs_pt 路径）+ 样本 npz/obs.pt。

`--workers` 以跨平台 `spawn` 方式按训练图并行构建；每个 worker 仅写私有临时目录，主进程按 manifest 固定顺序合并并生成唯一 manifest。`--workers=1` 保持单进程语义。建议在 16GB 内存设备上使用 6 个 worker，并固定 `--worker-torch-threads=1`，避免多进程下 PyTorch/OpenMP 线程过度订阅。

反事实预训练（Huber 相对收益回归 + 排序 BCE + 门控 CE + 正收益加权 BC，
仅训练双头、冻结编码器）：

```powershell
Generated APCF assets (staging directories, manifests, NPZ/PT samples, integrity reports, and pretraining checkpoints) are local/server artifacts and are never committed to Git. Before pretraining, independently validate the formal asset:

```powershell
python -X utf8 scripts/validate_anchor_proposal_cf_data.py `
  --dataset-dir data/initial_anchor_proposal_cf_v1 `
  --source-manifest data/scale_400_800_datasets/manifest_ctg_160_explicit_fiveskill_v1.json `
  --write-report data/initial_anchor_proposal_cf_v1/integrity_check.json
```

`--data-file data/680.csv` only applies the existing 100-worker initial-scheduling mapping; it is never a counterfactual sample source. Samples may only originate from the 160-graph manifest. `--max-candidates=4` limits anchor-trajectory candidate enumeration, while the independent label budget is anchor + at most two single swaps + at most two double swaps + one hash-selected double-swap representative (at most six teams after deduplication).

```powershell
python -X utf8 scripts/pretrain_anchor_proposal_cf.py `
  --manifest data/initial_anchor_proposal_cf_v1/manifest.json `
  --experiment conf/experiment/initial_anchor_proposal_cf_v1.yaml `
  --output checkpoints/apcf_pretrain_v1.ckpt
```

正式 PPO 微调必须显式提供预训练 checkpoint 与反事实 manifest，启动时会核验
checkpoint 的 model spec 为 APCF scope 且 manifest SHA-256 与当前配置一致：

```powershell
python train.py `
  experiment=initial_anchor_proposal_cf_v1 `
  anchor_proposal_pretrain_checkpoint_path=checkpoints/apcf_pretrain_v1.ckpt `
  anchor_proposal_cf_manifest_path=data/initial_anchor_proposal_cf_v1/manifest.json `
  train.num_envs=16 `
  train.max_episodes=300
```

探索下限按 `branch_floor_decay_fraction` 进度从 `0.20` 线性退火至 `0.02`；
每次成功 PPO 更新后推进（OOM 跳过时不推进）。APCF 冷启动会严格校验
反事实 manifest、预训练 checkpoint 的动作语义与 manifest SHA-256；续训只校验
latest PPO checkpoint 的 APCF 语义与 manifest，不重新加载预训练文件。Rollout 指标含
`APCF/RolloutDecisionCount`、`APCF/RolloutProposalAvailableRate`、
`APCF/RolloutHammingDistanceMean`、`APCF/RolloutRawProposalSelectRate`，以及
proposal pointer 对数概率/熵、预测反事实收益、gate 值和 raw 分支 logit 差，
可作为短轮筛选的诊断证据。


### 初始调度异步验证

Lightning 初始调度可将逐 episode 验证与 GPU 训练分离。单实例模式固定在 `async_eval_initial_data_path` 的 Standard 场景上按 Makespan 选择 best，且只有完整排程可以覆盖 best：

```bash
python train.py \
  experiment=initial_schedule_680 \
  experiment_name=initial_680_async_seed42 \
  async_eval_enabled=true \
  async_eval_initial_data_path=data/680.csv \
  async_eval_cpu_threads=4 \
  async_eval_queue_capacity=4 \
  async_eval_wait_on_finish=true \
  eval_freq=1 \
  eval_temperature=0.0 \
  seed=42
```

初始调度异步模式始终只验证 `async_eval_initial_data_path` 指定的固定 680 数据及其固定工人池，不执行 283/2338/3182 多基准验证，即使训练 YAML 启用了 `enable_multi_benchmark_eval=true`。有界队列满后训练会等待，因此不会丢失应参与 best 选择的 episode。异步结果位于当前运行的 `checkpoints/async_eval/`，异步 TensorBoard 位于其 `tensorboard/` 子目录。

### 命令行配置参数

可以通过以下命令查看当前版本支持的参数：

```bash
python train.py --help
```

常用直接参数如下：

| 参数 | 配置字段 | 示例 | 说明 |
|---|---|---|---|
| `experiment` | 实验配置 | `experiment=initial_schedule_283` | 对应 `conf/experiment/<name>.yaml` |
| `data_file_path` | `data_file_path` | `data_file_path=data/283.csv` | 验证和评估使用的数据集 |
| `train_data_path_or_dir` | `train_data_path_or_dir` | `train_data_path_or_dir=data/generated/initial_283` | 训练文件或训练池目录 |
| `seed` | `seed` | `seed=42` | 全局随机种子 |
| `train.max_episodes` | `max_episodes` | `train.max_episodes=300` | 最大训练 episode 数 |
| `train.num_envs` | `num_envs` | `train.num_envs=16` | 并行环境进程数 |
| `train.batch_size` | `batch_size` | `train.batch_size=16` | PPO mini-batch 大小 |
| `eval_freq` | `eval_freq` | `eval_freq=1` | 每隔多少个 episode 验证一次 |
| `log_dir` | `log_dir` | `log_dir=/root/tf-logs` | TensorBoard 日志根目录 |
| `run_id` | `run_id` | `run_id=initial_schedule_680_260630-153000` | 指定统一运行 ID；不指定时自动生成 |
| `runs_root` | `runs_root` | `runs_root=runs` | 统一运行目录根路径 |
| `use_skill_hub` | `use_skill_hub` | `use_skill_hub=true` | 启用 Skill Hub |
| `skill_hub_bidirectional` | `skill_hub_bidirectional` | `skill_hub_bidirectional=true` | 启用 Skill Hub 反向关系 |
| `policy_observation_scope` | `policy_observation_scope` | `policy_observation_scope=task` | 策略网络可见节点：`full`、`task` 或 `task_station`；默认 `full` |
| `resume` | 恢复训练 | `resume=true` | Lightning 读取当前 run 的最近断点 |
| `ablation_no_gat` | `ablation_no_gat` | `ablation_no_gat=true` | GAT 消融实验 |
| `ablation_no_pointer` | `ablation_no_pointer` | `ablation_no_pointer=true` | Pointer 消融实验 |
| `ablation_no_mask` | `ablation_no_mask` | 已禁用 | 主 PPO 必须保留动作硬约束 mask |

参数覆盖必须使用 `key=value`。例如 batch size 的推荐写法是：

```bash
train.batch_size=16
```

当前过渡实现仍会按叶子字段写入扁平 `Config`，因此下面写法也可用：

```bash
batch_size=16
```

`--batchsize` 不是有效参数。

### 使用 Hydra `key=value` 覆盖其他配置

所有 `Config` 字段统一使用：

```bash
key=value
```

每个配置项使用一个独立的 `key=value`，可以重复指定：

```bash
python train.py \
  experiment=initial_schedule_283 \
  train.batch_size=16 \
  train.num_envs=8 \
  lr=0.00005 \
  gamma=0.999 \
  accumulation_steps=8
```

不同数据类型的正确写法：

```bash
# 整数
k_epochs=2

# 浮点数
sample_temperature=1.0

# 布尔值必须写 true 或 false，不能写 0/1
enable_rollout_profiler=true
use_compile=false

# 字符串和路径
experiment_name=reschedule_task_delay
reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt

# 列表；Linux 建议用引号防止 shell 解释特殊字符
'eval_scenarios=[standard]'
```

PowerShell 列表示例：

```powershell
"eval_scenarios=[standard]"
```

### 使用 Hydra 风格 `key=value` 覆盖

训练、评估和排程生成入口同时支持直接追加 `key=value`：

```bash
python train.py \
  experiment=initial_schedule_283 \
  train.batch_size=24 \
  parallel.num_envs=8 \
  artifacts.runs_root=runs
```

当前实现仍落到扁平 `Config` 字段；`train.batch_size=24` 和 `batch_size=24` 等价。未知字段会直接报错，避免拼写错误被静默忽略。

常用 PPO 与性能配置示例：

```bash
python train.py \
  experiment=reschedule_task_delay \
  train.batch_size=16 \
  train.num_envs=16 \
  train.max_episodes=300 \
  eval_freq=1 \
  accumulation_steps=16 \
  skip_update_on_oom=true \
  enable_gpu_batch_rebuild=true \
  enable_rollout_profiler=true
```

结构性参数也可通过 `key=value` 指定，例如：

```bash
hidden_dim=128
num_gat_layers=5
num_heads=4
use_shared_trunk=false
```

加载 checkpoint 时，程序会自动识别其模型结构。若命令行显式指定的结构参数与 checkpoint 不一致，程序会报错并拒绝加载，避免静默使用错误结构。

恢复最近的 Lightning checkpoint：

```powershell
python train.py experiment=initial_schedule_283 resume=true
```

Windows GPU 冒烟测试（5 步 rollout、一次 PPO 更新和一次 Standard 验证）：

```powershell
python train.py experiment=smoke_lightning
```

历史训练循环仅用于回归对照：

```powershell
python train.py trainer=legacy experiment=initial_schedule_283
```

该命令会明确报错；legacy 训练入口已经移动到 `archive/legacy_train.py`，只用于历史对照阅读，不再作为当前训练入口。

## 重调度训练

重调度训练默认使用 `lightning` 入口。Lightning 会在启动时检查 baseline 排程、固定重调度验证场景和 warm-start 初始模型；legacy 入口仅保留给历史回归对照。

### 1. 准备初始调度模型

初始调度最优模型可以是 Lightning `.ckpt` 或 legacy `.pth`，不需要转换格式。建议统一放在：

```text
checkpoints/initial_schedule/
```

例如：

```text
checkpoints/initial_schedule/best_680.ckpt
```

模型必须与重调度使用的 APAL 数据集和图结构兼容。默认重调度配置使用 `data/283.csv`，因此应确认该模型能够在 `data/283.csv` 上生成完整可行排程。

> **重要：初始模型、baseline、重调度数据集和验证场景必须属于同一规模。**
>
> 例如，使用 `best_680.ckpt` 时，必须同时使用 680 数据、由该模型生成的 680 baseline，以及基于该 baseline 生成的 680 重调度验证场景。不能把 680 模型或 baseline 与默认的 `data/283.csv`、`results/final_schedule.csv`、`results/reschedule_eval_scenarios.csv` 混用。
>
> 如果规模不一致，可能出现 `c=1` 但 `elig=0/4`，并伴随大量 `fz`（冻结任务违规）和固定数量的 `dem`（需求人数违规）。这表示当前环境虽然完成了自己的任务，但验收使用的 baseline 并不属于同一张 APAL 图。

### 2. 生成 baseline 排程

使用初始调度模型生成确定性 baseline：

```powershell
python scripts/generate_schedule.py `
  experiment=initial_schedule_283 `
  model_path=checkpoints/initial_schedule/best_680.ckpt `
  output_path=results/final_schedule.csv
```

Linux 写法：

```bash
python scripts/generate_schedule.py \
  experiment=initial_schedule_283 \
  model_path=checkpoints/initial_schedule/best_680.ckpt \
  output_path=results/final_schedule.csv
```

生成后建议先验证 APAL 约束：

```powershell
python utils/verify_schedule.py data_path=data/283.csv schedule_path=results/final_schedule.csv
```

不同规模应使用独立文件名，避免意外复用。例如 680：

```bash
python scripts/generate_schedule.py \
  experiment=initial_schedule_680 \
  model_path=checkpoints/initial_schedule/best_680.ckpt \
  output_path=results/final_schedule_680.csv
```

不要让 680 重调度继续读取默认的：

```text
results/final_schedule.csv
results/reschedule_eval_scenarios.csv
```

应分别使用：

```text
results/final_schedule_680.csv
results/reschedule_eval_scenarios_680.csv
```

### 3. 开始重调度训练

```powershell
python train.py `
  experiment=reschedule_task_delay `
  reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt `
  reschedule_baseline_schedule_path=results/final_schedule.csv
```

Linux 写法：

```bash
python train.py \
  experiment=reschedule_task_delay \
  reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  reschedule_baseline_schedule_path=results/final_schedule.csv
```

启动日志应包含：

```text
[Reschedule] baseline=.../results/final_schedule.csv
[Reschedule] eval_scenarios=.../results/reschedule_eval_scenarios.csv
[Eval][Resched] ep=1 ...
```

如果 `results/reschedule_eval_scenarios.csv` 不存在，程序会根据 baseline 和配置的固定随机种子自动生成。训练期间每个 episode 都会在这些固定场景上验证，最优模型按重调度综合得分选择，并要求所有验证场景满足保存资格。

680 数据的完整示例：

```bash
python train.py \
  experiment=reschedule_task_delay \
  data_file_path=data/680.csv \
  train_data_path_or_dir=data/680.csv \
  n_w=100 \
  n_w_min=100 \
  reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  reschedule_baseline_schedule_path=results/final_schedule_680.csv \
  reschedule_eval_scenario_path=results/reschedule_eval_scenarios_680.csv
```

如果此前已经生成了其他规模的验证场景，必须更换 `reschedule_eval_scenario_path` 或删除旧文件后重新生成。程序只会在目标文件不存在时自动生成场景，不会自动判断已有场景是否属于当前 baseline。

### 4. 推荐论文实验：多实例重调度训练

论文主实验建议先准备 manifest：生成 30 个 400-600 工序随机训练实例，使用显式传入的初始调度模型为每个实例和真实 `283/680/2338/3182` 数据生成 baseline schedule，并为真实数据生成固定 low/medium/high 扰动场景。

```bash
python scripts/prepare_reschedule_data.py \
  experiment=reschedule_task_delay \
  initial_model_path=checkpoints/initial_schedule/bestmodel/best_model.pth \
  train_count=30 \
  min_ops=400 \
  max_ops=600 \
  seed=20260701 \
  output_manifest=data/reschedule_manifests/reschedule_400_600_seed20260701.json
```

训练时使用 manifest，环境会按当前随机数据集自动匹配自己的 baseline；`reschedule_baseline_model_path` 用于 warm start 初始化重调度策略，自动验证固定使用真实 680：

```bash
python train.py \
  experiment=reschedule_task_delay \
  train_data_path_or_dir=data/generated/reschedule_train_400_600 \
  reschedule_baseline_model_path=checkpoints/initial_schedule/bestmodel/best_model.pth \
  reschedule_manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701.json \
  reschedule_eval_instance_id=real_680 \
  train.batch_size=128
```

需要让 GPU 训练与验证解耦时，可显式开启异步 CPU 验证。训练进程独占 GPU，独立 worker 使用 4 个 CPU 线程，仅验证 manifest 中的 `real_680/medium_000`；有界队列最多保留 4 个待验证候选，队列满时训练会等待，因此不会丢弃任何应验证的 episode：

```bash
python train.py \
  experiment=reschedule_task_delay \
  experiment_name=reschedule_hgppo_objective_aligned_async_seed42 \
  train_data_path_or_dir=data/generated/reschedule_train_400_600 \
  reschedule_baseline_model_path=checkpoints/initial_schedule/bestmodel/best_model.pth \
  reschedule_manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701_physical_v2.json \
  reschedule_eval_instance_id=real_680 \
  async_eval_enabled=true \
  async_eval_instance_id=real_680 \
  async_eval_scenario_id=medium_000 \
  async_eval_cpu_threads=4 \
  async_eval_queue_capacity=4 \
  eval_freq=1 \
  seed=42
```

异步 worker 使用完整 Lightning checkpoint 恢复模型和 ScheduleFree 优化器状态，但确定性选动作时只执行 Actor，不计算无用的 Critic；观测缓存只复用静态图拓扑，动态节点特征和动态分配边仍逐步刷新。训练正常结束前会等待队列清空。进程或服务器中断后，用相同 `run_id` 和 `resume=true` 恢复时，会重新接管 `pending/running` 任务。

训练期单场景异步验证仅用于高频选择 best，不能代替论文中的正式评估。训练完成后仍须使用 `scripts/evaluate_reschedule_manifest.py` 在固定 low/medium/high 全场景上报告结果。

训练完成后评估真实多规模数据：

```bash
python scripts/evaluate_reschedule_manifest.py \
  experiment=reschedule_task_delay \
  model_path=checkpoints/reschedule_task_delay/bestmodel/best_model.pth \
  manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701.json \
  instance_ids=[real_283,real_680,real_2338,real_3182] \
  output_dir=results/reschedule_manifest_eval
```

与规则/搜索式 baseline 做公平比较时，也使用同一个 manifest 和同一个 `instance_ids`：

```bash
python scripts/evaluate_reschedule_rules.py \
  experiment=reschedule_task_delay \
  manifest_path=data/reschedule_manifests/reschedule_400_600_seed20260701.json \
  instance_ids=[real_680] \
  output_dir=results/reschedule_rules_real_680
```

### 5. 输出与续训

重调度主要输出：

| 输出 | 路径 |
|---|---|
| 最新 Lightning 断点 | `runs/reschedule_task_delay/<run_id>/checkpoints/last.ckpt` |
| 最优重调度模型 | `runs/reschedule_task_delay/<run_id>/checkpoints/best.ckpt` |
| 最终配置与运行清单 | `runs/reschedule_task_delay/<run_id>/configs/` |
| 固定验证场景 | `results/reschedule_eval_scenarios.csv` |
| TensorBoard 日志 | `runs/reschedule_task_delay/<run_id>/logs/tensorboard/` |
| 异步验证队列与逐轮结果 | `runs/reschedule_task_delay/<run_id>/checkpoints/async_eval/` |
| 异步验证 TensorBoard | `runs/reschedule_task_delay/<run_id>/checkpoints/async_eval/tensorboard/` |

如果显式使用 `artifact_layout=legacy`，才会写入旧的 `checkpoints/reschedule_task_delay/` 与 `/root/tf-logs/` 结构。`trainer=legacy` 现在会明确报错。

断点续训：

```bash
python train.py \
  experiment=reschedule_task_delay \
  reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  reschedule_baseline_schedule_path=results/final_schedule.csv \
  run_id=reschedule_task_delay_260630-153000 \
  resume=true
```

### 6. 评估重调度模型

```bash
python evaluate_reschedule_model.py \
  experiment=reschedule_task_delay \
  model_path=runs/reschedule_task_delay/<run_id>/checkpoints/best.ckpt
```

评估结果保存到：

```text
runs/reschedule_task_delay/<run_id>/eval/reschedule/reschedule_ppo_eval.csv
runs/reschedule_task_delay/<run_id>/eval/reschedule/reschedule_ppo_eval_summary.json
```

显式传 `output_dir=results/reschedule_ppo_eval` 时会写回旧目录。

## 运行日志

Lightning 每个 episode 输出一行精简 rollout 摘要：

```text
[Rollout] ep=1 R=-0.05 Mk=32.0 Done=2.1% SPS=11.5 ms=1.1/159.7/3.4/1.9
```

`ms` 依次表示：

1. IPC 与 Mask 获取耗时。
2. 策略前向和动作选择耗时。
3. 图状态重建耗时。
4. 环境 Step 耗时。

自动验证默认每个 episode 执行一次，只运行 Standard 场景，并打印 Makespan、Balance、Reward、人员/站位利用率及耗时。详细指标同时写入 Lightning/TensorBoard。开启初始调度异步验证后，训练进程只提交候选 checkpoint，CPU worker 按相同指标独立验证并原子更新 `best.ckpt`。

重调度默认启用统一目标差分密集奖励 `reschedule_use_objective_delta_reward=true`。每个动作的奖励来自验证综合目标在动作前后的变化，分项包括归一化 makespan、负载均衡、takt 超期、开始时间偏差、工位变化率和班组变化率；终局不再重复扣除这些项。TensorBoard 会额外记录 `Reward/ObjectiveDelta`、`Reward/ObjectiveFinalScore`、`Reward/ObjectiveClipFraction` 和 `Reward/Delta/*`。该奖励与旧重调度 reward 不同，正式实验必须使用新的 `experiment_name` 从初始调度模型 warm start，不能将旧 checkpoint 的续训曲线与新奖励训练曲线合并。

物料 release time 在张量化掩码、CPU 掩码、环境边界和合法性审计中统一使用 `float64` 与 `release_time_tolerance_hours=1e-5` 小时。评估默认采用 `eval_mask_mismatch_policy=fail`，掩码放行但环境拒绝时立即报告完整上下文；`recover` 仅供诊断性评估恢复 `task_release_time_not_reached`，训练阶段始终禁止恢复，避免污染 on-policy 轨迹。

重调度同步模式会在主进程执行固定场景验证，打印 `[Eval][Resched]`，并按 `reschedule_selection_score` 保存 best；只有所有固定验证场景满足资格时才允许覆盖 best。开启 `async_eval_enabled=true` 后，主进程改为打印 `[AsyncEval]` 入队信息，CPU worker 打印 `[AsyncEval][Done/Best]`，并按指定单场景的合法性和 `selection_score` 原子更新同一路径的 `best.ckpt`。

Lightning 每次 PPO 更新后覆盖保存最新 checkpoint；执行自动验证且选择指标改善时，覆盖保存最佳 checkpoint。终端会打印对应的 `[Checkpoint]` 记录。

新运行默认写入统一 `runs` 目录。若使用 `artifact_layout=legacy`，或 `resume=true` 但未指定 `run_id=...`，会回到旧 `checkpoints/results/tf-logs` 兼容路径。

Rollout 心跳默认关闭。需要诊断长时间阶段时，可在 rollout YAML 中设置：

```yaml
rollout:
  rollout_heartbeat_interval_sec: 30.0
```

## 输出文件与目录

除绝对路径外，以下相对路径均以项目根目录为基准。`<实验名>` 来自实验 YAML 的 `experiment_name`，例如 `initial_schedule_283`。新版本默认使用统一运行目录：

```text
runs/<实验名>/<实验名>_<YYMMDD-HHMMSS>/
```

例如：

```text
runs/initial_schedule_680/initial_schedule_680_260630-153000/
```

如果需要恢复某次新目录训练，建议显式传入对应 `run_id=...`：

```bash
python train.py experiment=initial_schedule_680 run_id=initial_schedule_680_260630-153000 resume=true
```

### 统一 Runs 输出

| 输出 | 路径 | 说明 |
|---|---|---|
| 最新 Lightning checkpoint | `runs/<实验名>/<run_id>/checkpoints/last.ckpt` | 每个 episode 覆盖保存 |
| 最优 Lightning checkpoint | `runs/<实验名>/<run_id>/checkpoints/best.ckpt` | 验证指标改善时覆盖保存 |
| Legacy 兼容 checkpoint | `runs/<实验名>/<run_id>/checkpoints/legacy/` | legacy 入口过渡使用 |
| TensorBoard event | `runs/<实验名>/<run_id>/logs/tensorboard/` | 训练、验证、rollout、OOM 和性能指标 |
| 最终配置 | `runs/<实验名>/<run_id>/configs/resolved_config.yaml` | YAML、平台配置和 CLI 覆盖后的最终配置 |
| 运行清单 | `runs/<实验名>/<run_id>/configs/run_manifest.json` | run_id、命令、Git commit 和 checkpoint 元数据 |
| 评估结果 | `runs/<实验名>/<run_id>/eval/` | `summary.json`、`schedule.csv`、`gantt.png` 等 |
| 附加产物 | `runs/<实验名>/<run_id>/artifacts/` | reports、traces 和后续论文实验中间文件 |
| Baseline 评估 | `runs/<实验名>/<run_id>/artifacts/baselines/` | Heuristic、Beam/IG/SA、文献适配 PPO/DQN baseline 的 metrics、schedule 和 run.log |
| Benchmark 结果 | `runs/<实验名>/<run_id>/artifacts/benchmark/` | runtime summary、raw 子进程输出和论文表格 CSV |

实验工具默认也写入 `runs`。若需要旧路径，显式传入 `output_dir=results/eval_logs`、`output_dir=results/runtime_benchmark/...`，或使用 `artifact_layout=legacy`。

生成所有运行的文件索引：

```bash
python scripts/index_runs.py --runs-root runs
```

该命令会生成 `runs/index.csv` 和 `runs/index.json`，汇总每个 run 的 `run_id`、配置来源、checkpoint、数据集、评估 summary、baseline metrics、benchmark rows 和核心指标。若某个 run 尚未执行手动评估，索引仍会保留运行清单，评估字段为空。

### Legacy 兼容路径

旧路径仍可读取，用于老 checkpoint、旧实验结果和未迁移脚本。

### Lightning 训练

| 输出 | 路径 | 说明 |
|---|---|---|
| 最新完整 checkpoint | `checkpoints/<实验名>/lightning/last.ckpt` | legacy artifact layout 下使用 |
| 最优完整 checkpoint | `checkpoints/<实验名>/lightning/best/best.ckpt` | legacy artifact layout 下使用 |
| TensorBoard event | `/root/tf-logs/<实验名>/version_N/events.out.tfevents.*` | rollout、PPO、验证、OOM 和性能指标 |
| Lightning 超参数 | `/root/tf-logs/<实验名>/version_N/hparams.yaml` | Lightning Logger 生成 |

查看日志：

```bash
tensorboard --logdir runs --bind_all
tensorboard --logdir /root/tf-logs --bind_all
```

### Legacy 训练

| 输出 | 路径 | 说明 |
|---|---|---|
| 最新训练断点 | `checkpoints/<实验名>/latest_checkpoint.pth` | 模型、优化器和 episode 状态 |
| 最优模型权重 | `checkpoints/<实验名>/bestmodel/best_model.pth` | 验证指标改善时覆盖保存 |
| 最优模型元数据 | `checkpoints/<实验名>/bestmodel/best_model_meta.json` | episode、Makespan、配置和数据路径 |
| TensorBoard event | `/root/tf-logs/<实验名>_ALB_PPO_<时间戳>/events.out.tfevents.*` | legacy 训练指标 |
| 阶段诊断报告 | `results/reports/report_ep<episode>_<时间戳>.md` | 按 `generate_report_every_episodes` 生成 |
| 最佳排程轨迹 | `checkpoints/<实验名>/eval_traces/Ep_<episode>_Best_Schedule.csv` | 训练过程中导出的排程 |
| 轨迹甘特图 | `checkpoints/<实验名>/eval_traces/Ep_<episode>_Gantt.png` | 对应排程可视化 |
| PPO 最终输出 | `results/PPO/PPO_Final_schedule.csv`、`PPO_Final_gantt.png` | 训练结束后的 PPO 推演 |
| GA 对照输出 | `results/GA/GA_Baseline_schedule.csv`、`GA_Baseline_gantt.png` | GA 基线排程 |

### 排程与评估

| 输出 | 路径 | 说明 |
|---|---|---|
| 默认最终排程 | `results/final_schedule.csv` | `scripts/generate_schedule.py` 生成，也是默认重调度 baseline |
| 重调度评估场景 | `results/reschedule_eval_scenarios.csv` | 默认重调度场景文件 |
| 手动评估排程 | 评估结果目录下的 `<前缀>_schedule.csv` | `evaluate_model.py` 输出 |
| 手动评估甘特图 | 同目录下的 `<前缀>_gantt.png` | 最佳评估排程可视化 |

评估和排程入口统一支持 Lightning `.ckpt`、legacy 完整 `.pth` 和裸 `state_dict`。加载时会读取 checkpoint 元数据；旧模型没有元数据时，会根据 `can_do`、`skill_emb`、`has_skill`、`required_by` 等权重键自动识别资源图模式。若 CLI 显式指定的模型结构与 checkpoint 冲突，程序会拒绝运行。

## 评估与排程

手动评估保留 Standard、工时扰动、人员扰动和动态事件四场景：

```powershell
python evaluate_model.py experiment=initial_schedule_283 model_path=checkpoints/<实验目录>/<模型文件> test_data=data/283.csv num_runs=3 temperature=0 output_dir=results/evaluation
```

生成排程：

```powershell
python scripts/generate_schedule.py experiment=initial_schedule_283 model_path=checkpoints/<实验目录>/<模型文件> output_path=results/final_schedule.csv
```

每次新运行会在 `runs/<实验名>/<run_id>/configs/` 写入 `resolved_config.yaml` 和 `run_manifest.json`，记录最终配置、配置来源、命令类型、run_id 和 Git commit。训练只负责 checkpoint 与 TensorBoard；GA 对比、排程 CSV 和甘特图由独立命令按需生成。

验证 APAL 排程约束：

```powershell
python utils/verify_schedule.py data_path=data/283.csv schedule_path=results/final_schedule.csv
```

## 测试

```powershell
python -m pytest -q
```

关键架构测试：

```powershell
python tests/run_preflight_tests.py
```

该前检查只执行真实图的初始化、图构建、模型前向和首步合法动作，不替代服务器上的正式训练与大规模验证。

## Skill Hub 资源图

默认使用 Skill Hub 压缩稠密的工人技能边，配置位于 `conf/model/hb_gat_pn.yaml`：

```yaml
model:
  use_skill_hub: true
  skill_hub_bidirectional: true
  num_skill_types: 5
  skill_feat_dim: 11
  worker_skill_feature_slots: 5
```

- `use_skill_hub: false`：保留原始 `Worker -> Task` 的 `can_do` 直接边，可用于旧模型和消融对照。
- `use_skill_hub: true`：使用 `Worker -> Skill -> Task` 压缩资源图。
- `skill_hub_bidirectional: true`：额外启用 `Task -> Skill -> Worker` 反向消息；设为 `false` 时仅保留正向链路。
- Skill Hub 与旧直接边互斥，不会同时参与消息传递。
- 新旧图结构的模型参数不兼容；加载 checkpoint 时必须使用训练该 checkpoint 时相同的图模式。
- 工人节点特征统一为 17 维：`[效率 | 5 个技能 | 等待 | 空闲 | 8 个工位锁状态 | 疲劳]`；历史 22 维 checkpoint 不可与当前五槽位模型混用。

## CUDA OOM 保护

- Windows 低显存配置将 PPO batch 上限设为 `4`；Linux 不设置上限，使用实验配置的默认 batch。
- GPU 图模板仅保留当前数据集，避免训练池轮换时显存逐轮累积。
- PPO 更新发生 CUDA OOM 时会完整回滚模型、优化器、AMP 缩放器和随机状态，然后直接跳过当前 PPO 更新。
- OOM 后不会自动降低 PPO batch size，也不会记录该轮 rollout、loss 或 eval 指标；下一轮继续使用原 batch size。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用于缓解显存碎片，但不能替代 batch 上限和事务回滚。

## Heuristic 与 GA 基线批量评估

本项目提供了一个能够自动在 `283.csv`, `680.csv`, `2338.csv`, `3182.csv` 等多个不同规模数据集上，批量运行各类启发式规则及遗传算法（GA）的综合评估工具。它实现了对不同规模数据集物理环境参数（如工人数 `n_w`）的自适应对齐，确保与主模型的评估环境完全一致（反作弊机制）。

### 支持的基线方法
1. **SPT**：最短工时优先 (Shortest Processing Time)
2. **LPT**：最长工时优先 (Longest Processing Time)
3. **Random**：随机就绪工序选择 (Random Dispatching)
4. **EDD**：最早可能开工时间优先 (Earliest Due Date/ES)
5. **CPM**：关键路径法优先 (Critical Path Method/LS 越早越优先)
6. **MSL**：最轻工位负载优先 (Minimum Station Load)
7. **GA**：遗传算法基准排程 (Genetic Algorithm)
8. **Beam / BeamSearch**：基于优先级编码的 Beam Search，保留少量候选解并逐层扰动扩展。
9. **IG / IteratedGreedy / DestroyRepair**：迭代贪婪 Destroy-and-Repair，优先破坏并修复关键路径和高影响工序。
10. **SA / SimulatedAnnealing**：模拟退火，在任务、工位和工人优先级邻域中搜索，并按温度接受劣解。

### 运行命令

在 `rag_env` 虚拟环境下，执行以下命令进行批量评估：

```powershell
python baselines/heuristic/run_all_baselines.py `
  --data_dir data `
  --datasets 283.csv 680.csv 2338.csv 3182.csv `
  --methods SPT LPT Random EDD CPM MSL GA `
  --ga_pop_size 30 `
  --ga_max_gen 20
```

Linux 下的写法：

```bash
python baselines/heuristic/run_all_baselines.py \
  --data_dir data \
  --datasets 283.csv 680.csv 2338.csv 3182.csv \
  --methods SPT LPT Random EDD CPM MSL GA \
  --ga_pop_size 30 \
  --ga_max_gen 20
```

运行新增的高级启发式基线：

```bash
python baselines/heuristic/run_all_baselines.py \
  --data_dir data \
  --datasets 283.csv 680.csv 2338.csv 3182.csv \
  --methods Beam IG SA \
  --beam_width 4 \
  --beam_branch_factor 4 \
  --beam_levels 8 \
  --beam_patience 4 \
  --ig_iterations 80 \
  --ig_destroy_ratio 0.10 \
  --ig_noise_sigma 0.25 \
  --sa_iterations 120 \
  --sa_initial_temp 0.05 \
  --sa_cooling 0.96 \
  --sa_min_temp 0.0001 \
  --balance_weight 1.0 \
  --seed 42
```

Windows PowerShell 写法：

```powershell
python baselines/heuristic/run_all_baselines.py `
  --data_dir data `
  --datasets 283.csv 680.csv 2338.csv 3182.csv `
  --methods Beam IG SA `
  --beam_width 4 `
  --beam_branch_factor 4 `
  --beam_levels 8 `
  --beam_patience 4 `
  --ig_iterations 80 `
  --ig_destroy_ratio 0.10 `
  --ig_noise_sigma 0.25 `
  --sa_iterations 120 `
  --sa_initial_temp 0.05 `
  --sa_cooling 0.96 `
  --sa_min_temp 0.0001 `
  --balance_weight 1.0 `
  --seed 42
```

快速冒烟测试可以先只跑 `283.csv`，并把迭代数降到很小：

```bash
python baselines/heuristic/run_all_baselines.py \
  --datasets 283.csv \
  --methods Beam IG SA \
  --beam_width 2 \
  --beam_branch_factor 1 \
  --beam_levels 1 \
  --beam_patience 1 \
  --ig_iterations 1 \
  --ig_destroy_ratio 0.05 \
  --sa_iterations 1 \
  --seed 42
```

### 文献适配学习型 baseline：L2D-PPO-APAL 与 Graph-DDQN-APAL

当前论文主线不再训练旧版 `BasicPPO` 和 `DQN`。这两个脚本保留为历史简化 baseline，但不建议进入正文主表。新的学习型对比算法使用与主方法一致的 APAL `HeteroData` 图观测、动作 mask、`env.step()` reward、训练数据目录和验证数据，只替换文献启发的学习框架。

`L2D-PPO-APAL` 是 learned dispatching rule / GNN-RL 思路的 APAL 适配版，仍用 PPO 优化，但 checkpoint 会标记为 `feature_mode=apal_hetero_graph`。训练命令：

```bash
python baselines/literature_ppo/train_l2d_ppo_apal.py \
  experiment=scale_400_800_schedule \
  train_data_path_or_dir=data/scale_400_800_datasets \
  data_file_path=data/680.csv \
  max_episodes=300 \
  train.batch_size=64 \
  seed=42 \
  output_dir=results/l2d_ppo_apal_680
```

`Graph-DDQN-APAL` 是图状态 Double DQN 的 APAL 适配版，使用在线网络选 next action、target 网络估计 target Q，并复用同一套 APAL mask 与 reward。训练命令：

```bash
python baselines/literature_dqn/train_graph_ddqn_apal.py \
  experiment=scale_400_800_schedule \
  train_data_path_or_dir=data/scale_400_800_datasets \
  data_file_path=data/680.csv \
  max_episodes=300 \
  train.batch_size=64 \
  seed=42 \
  output_dir=results/graph_ddqn_apal_680
```

DDQN 默认启用批量 replay：当前状态只执行一次批量图编码，next state 由 online network 批量选动作，再复用同一批图给 target network 估值。更新频率统一由 `ddqn_updates_per_transition` 控制，默认 `0.125`，即平均每 8 条新 transition 执行一次更新。`training_metrics.csv` 会记录 `effective_utd`、replay 各阶段耗时和 OOM 跳过次数；正式实验不建议设置 `max_replay_updates_per_episode`，否则会截断实际 UTD。

向量化 rollout 默认关闭，避免未经目标服务器基准验证就改变训练吞吐。Linux 服务器可显式启用 8 个环境；`max_episodes` 仍表示完成的轨迹总数，而不是并行轮数：

```bash
python baselines/literature_dqn/train_graph_ddqn_apal.py \
  experiment=scale_400_800_schedule \
  train_data_path_or_dir=data/scale_400_800_datasets \
  data_file_path=data/680.csv \
  max_episodes=300 \
  train.batch_size=64 \
  ddqn_updates_per_transition=0.125 \
  ddqn_enable_vector_env=true \
  ddqn_num_envs=8 \
  seed=42 \
  output_dir=results/graph_ddqn_apal_680
```

GPU 双模板 batch 重建是独立实验开关，默认关闭。只有完成同设备、同数据、同 transition 的等价性和显存基准后，才建议追加 `ddqn_enable_gpu_batch_rebuild=true`。该优化异常时会自动退回 CPU 构图，不会更改 batch size。

统一离线评估入口会根据 checkpoint 自动识别算法类型：

```bash
python baselines/literature/evaluate_literature_baseline.py \
  experiment=initial_schedule_680 \
  model_path=results/l2d_ppo_apal_680/l2d_ppo_apal_best.pth \
  datasets=[283.csv,680.csv,2338.csv,3182.csv] \
  num_runs=1 \
  temperature=0.0 \
  output_dir=results/eval_l2d_ppo_apal_all
```

评估结果会写入：

```text
results/eval_l2d_ppo_apal_all/<Method>/<dataset_name>/metrics.json
results/eval_l2d_ppo_apal_all/<Method>/<dataset_name>/schedule.csv
results/eval_l2d_ppo_apal_all/<Method>/<dataset_name>/runs_detail.csv
results/eval_l2d_ppo_apal_all/<Method>_summary.csv
```

若不传 `output_dir`，结果默认进入 `runs/<实验名>/<run_id>/artifacts/baselines/literature_eval/`。

两个文献适配训练脚本都会保存：

```text
<output_dir>/l2d_ppo_apal_latest.pth
<output_dir>/l2d_ppo_apal_best.pth
<output_dir>/l2d_ppo_apal_final.pth
<output_dir>/graph_ddqn_apal_latest.pth
<output_dir>/graph_ddqn_apal_best.pth
<output_dir>/graph_ddqn_apal_final.pth
```

继续训练时使用相同 `output_dir` 并追加 `resume=true`：

```bash
python baselines/literature_ppo/train_l2d_ppo_apal.py \
  experiment=scale_400_800_schedule \
  train_data_path_or_dir=data/scale_400_800_datasets \
  data_file_path=data/680.csv \
  max_episodes=300 \
  train.batch_size=64 \
  seed=42 \
  output_dir=results/l2d_ppo_apal_680 \
  resume=true
```

DDQN 默认启用精确恢复。除轻量的 best/latest/final checkpoint 外，还会生成：

```text
<output_dir>/graph_ddqn_apal_exact_resume.pth
<output_dir>/graph_ddqn_apal_exact_resume.replay.pt.gz
<output_dir>/graph_ddqn_apal_exact_resume.json
```

使用相同 `output_dir` 并传入 `resume=true` 时，会校验 SHA256 后恢复模型、target network、optimizer、AMP scaler、epsilon、replay buffer、UTD credit 和全部随机数状态。若精确恢复文件不存在，则兼容回退到 `graph_ddqn_apal_latest.pth`，此时 replay buffer 会重新预热。

DDQN replay 性能基准使用固定 transition，并对串行旧路径和批量新路径做交替配对测试：

```bash
python scripts/benchmark_graph_ddqn_replay.py \
  experiment=scale_400_800_schedule \
  train.batch_size=64 \
  benchmark_data_path=data/680.csv \
  benchmark_transitions=320 \
  benchmark_updates=50 \
  benchmark_repeats=5 \
  output_dir=results/05_efficiency_and_logs/ddqn_performance_optimization/linux_680_batch64
```

基准会保存逐次运行 CSV 和 JSON 汇总。正式采用优化前，应同时检查 loss 等价性、更新吞吐、峰值显存和端到端 episode 耗时；不能只比较 GPU 利用率。

### HB-GAT-PN 消融实验

正式消融统一使用主 PPO Lightning 入口。当前 CAC 对比消融推荐使用 `scale_400_800_schedule.yaml`：训练集为 `data/scale_400_800_datasets`，在线验证集为 `data/680.csv`。为避免每次在线验证跑四个大规模基准，消融训练阶段建议关闭 `enable_multi_benchmark_eval`，每个实验单独设置 `experiment_name`。

完整模型对照：

```bash
python train.py \
  experiment=scale_400_800_schedule \
  experiment_name=scale_400_800_full \
  enable_multi_benchmark_eval=false
```

常用三类结构消融：

节点范围对比消融的定义如下：`operation_only` 仅向策略网络编码工序节点，`operation_station` 仅编码工序和站位节点；工人、站位的真实负载、等待时间和可行性仍由环境用于动作掩码与完工时间计算，但不会作为这两种策略的图节点输入。主方法 `full_joint` 及其他方法保持 `policy_observation_scope=full`。

```bash
python train.py \
  experiment=scale_400_800_schedule \
  policy_action_scope=operation \
  policy_observation_scope=task \
  experiment_name=scale_400_800_operation_only \
  enable_multi_benchmark_eval=false

python train.py \
  experiment=scale_400_800_schedule \
  policy_action_scope=operation_station \
  policy_observation_scope=task_station \
  experiment_name=scale_400_800_operation_station \
  enable_multi_benchmark_eval=false
```

```bash
python train.py \
  experiment=scale_400_800_schedule \
  ablation_no_gat=true \
  experiment_name=scale_400_800_no_gat \
  enable_multi_benchmark_eval=false

python train.py \
  experiment=scale_400_800_schedule \
  ablation_no_pointer=true \
  experiment_name=scale_400_800_no_pointer \
  enable_multi_benchmark_eval=false

```

`ablation_no_mask=true` 在主 PPO 中会显式报错；合法性 mask 属于安全约束，不再作为消融变量。

也可以通过 `key=value` 做其他结构或训练机制消融：

```bash
python train.py \
  experiment=scale_400_800_schedule \
  experiment_name=scale_400_800_no_attention_critic \
  enable_multi_benchmark_eval=false \
  use_attention_critic=false \
  use_shared_trunk=true \
  use_autoregressive_worker=false
```

如果服务器上存在旧配置残留，或者需要显式确认训练和验证路径，可在命令中强制覆盖：

```bash
python train.py \
  experiment=scale_400_800_schedule \
  train_data_path_or_dir=data/scale_400_800_datasets \
  data_file_path=data/680.csv \
  enable_multi_benchmark_eval=false
```

若需要用四基准归一化综合评分保存 best model，则去掉 `enable_multi_benchmark_eval=false`，使用配置文件中的 `data/283.csv`、`data/680.csv`、`data/2338.csv`、`data/3182.csv` 进行验证。历史敏感性脚本已从自动测试集中移除，不应作为当前论文消融入口。

#### 常用参数说明：
- `--data_dir`：数据集文件所在的根目录（默认 `data`）。
- `--datasets`：待测试的 CSV 文件列表（默认 `283.csv 680.csv 2338.csv 3182.csv`）。
- `--methods`：待评估的算法列表。如果想要快速评估启发式规则并跳过较慢的 GA 和高级搜索，可以只保留 `SPT LPT Random EDD CPM MSL`。新增算法可写为 `Beam IG SA`，也兼容 `BeamSearch`、`IteratedGreedy`、`DestroyRepair`、`SimulatedAnnealing` 等别名。
- `--random_runs`：含有随机性规则的独立运行轮数（默认 `5`，如 `Random` 规则会运行 5 次求均值）。
- `--ga_pop_size` 和 `--ga_max_gen`：遗传算法的种群大小和迭代代数。
- `--balance_weight`：高级启发式 fitness 中负载均衡标准差的权重，fitness 为 `makespan + balance_weight * workload_balance_std`，默认 `1.0`。
- `--beam_width`：Beam Search 每层保留的候选解数量，越大越可能找到更好排程，但运行时间也会增加。
- `--beam_branch_factor`：Beam Search 每个候选解生成的邻域分支数量。
- `--beam_levels`：Beam Search 最大展开层数。
- `--beam_patience`：Beam Search 连续多少层无改进后提前停止。
- `--ig_iterations`：Iterated Greedy / Destroy-and-Repair 的迭代次数。
- `--ig_destroy_ratio`：每轮破坏并重新修复的工序比例，默认 `0.10`。
- `--ig_noise_sigma`：修复阶段优先级扰动强度。
- `--sa_iterations`：Simulated Annealing 的迭代次数。
- `--sa_initial_temp`：模拟退火初始温度。当前实现使用相对 `ideal_makespan` 的归一化 fitness 差值，因此该值不需要随数据集规模线性放大。
- `--sa_cooling`：降温系数，默认 `0.96`。
- `--sa_min_temp`：最低温度，默认 `0.0001`。

### 输出结果查看

评估完成后，结果将按照以下规则进行多层级、结构化归档：

1. **终端统计汇总表**：评估运行结束后，控制台会直接输出所有方法在各个数据集上的指标对比表格（包含 Makespan、工作量偏差 std、工人利用率、站位利用率、推理耗时以及是否死锁）。同时该汇总表格会被导出为 CSV 文件：
   - 默认汇总表格路径：`runs/<实验名>/<run_id>/artifacts/baselines/heuristic/baselines_summary.csv`
   - 旧路径兼容：显式传 `output_dir=results/eval_logs` 后写入 `results/eval_logs/baselines_summary.csv`
2. **各算法独立数据集的指标与排程**：每个算法在对应数据集下的评估明细将被详细记录在：
   - **度量指标**：`runs/<实验名>/<run_id>/artifacts/baselines/heuristic/<method>/<dataset_name>/metrics.json`
   - **排程明细**：`runs/<实验名>/<run_id>/artifacts/baselines/heuristic/<method>/<dataset_name>/schedule.csv`
   - **运行日志**：`runs/<实验名>/<run_id>/artifacts/baselines/heuristic/<method>/<dataset_name>/run.log`
