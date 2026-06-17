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
python scripts/generate_initial_buckets.py --bucket all --num_samples 32 --seed 42
```

覆盖已有训练池：

```powershell
python scripts/generate_initial_buckets.py --bucket all --num_samples 32 --seed 42 --overwrite
```

## 训练

训练入口会根据操作系统自动追加硬件配置，不需要手动传入 `conf/hardware/*.yaml`。

```powershell
python train.py --trainer lightning --config conf/experiment/initial_schedule_283.yaml
python train.py --trainer lightning --config conf/experiment/initial_schedule_680.yaml
python train.py --trainer lightning --config conf/experiment/initial_schedule_2338.yaml
python train.py --trainer lightning --config conf/experiment/initial_schedule_3182.yaml
```

命令行参数会覆盖 YAML 和自动平台配置。常用参数可直接传入：

```powershell
python train.py --config conf/experiment/initial_schedule_680.yaml --batch-size 16 --num-envs 8 --max-episodes 100
python train.py --config conf/experiment/initial_schedule_680.yaml --no-use-skill-hub
```

其他 `Config` 字段通过可重复的 `--set key=value` 覆盖：

```powershell
python train.py --config conf/experiment/initial_schedule_680.yaml --set lr=0.00005 --set eval_scenarios=[standard]
```

配置优先级为：代码默认值 `<` 实验 YAML `<` 平台硬件 YAML `<` 显式命令行参数 `<` `--set`。

### 命令行配置参数

可以通过以下命令查看当前版本支持的参数：

```bash
python train.py --help
```

常用直接参数如下：

| 参数 | 配置字段 | 示例 | 说明 |
|---|---|---|---|
| `--trainer` | 训练入口 | `--trainer lightning` | 可选 `lightning` 或 `legacy` |
| `--config` | YAML 配置 | `--config conf/experiment/initial_schedule_283.yaml` | 可重复使用，后加载的配置优先 |
| `--data-path` | `data_file_path` | `--data-path data/283.csv` | 验证和评估使用的数据集 |
| `--train-data-path` | `train_data_path_or_dir` | `--train-data-path data/generated/initial_283` | 训练文件或训练池目录 |
| `--seed` | `seed` | `--seed 42` | 全局随机种子 |
| `--max-episodes` | `max_episodes` | `--max-episodes 300` | 最大训练 episode 数 |
| `--num-envs` | `num_envs` | `--num-envs 16` | 并行环境进程数 |
| `--batch-size` | `batch_size` | `--batch-size 16` | PPO mini-batch 大小 |
| `--eval-freq` | `eval_freq` | `--eval-freq 1` | 每隔多少个 episode 验证一次 |
| `--log-dir` | `log_dir` | `--log-dir /root/tf-logs` | TensorBoard 日志根目录 |
| `--output-dir` | `result_dir` | `--output-dir results` | 评估、排程等结果根目录 |
| `--use-skill-hub` | `use_skill_hub` | `--use-skill-hub` | 启用 Skill Hub |
| `--no-use-skill-hub` | `use_skill_hub` | `--no-use-skill-hub` | 使用旧版 Worker-Task 直接边 |
| `--skill-hub-bidirectional` | `skill_hub_bidirectional` | `--skill-hub-bidirectional` | 启用 Skill Hub 反向关系 |
| `--no-skill-hub-bidirectional` | `skill_hub_bidirectional` | `--no-skill-hub-bidirectional` | 仅使用正向 Skill Hub |
| `--resume` | 恢复训练 | `--resume` | Lightning/legacy 分别读取各自最近断点 |
| `--ablation-no-gat` | `ablation_no_gat` | `--ablation-no-gat` | GAT 消融实验 |
| `--ablation-no-pointer` | `ablation_no_pointer` | `--ablation-no-pointer` | Pointer 消融实验 |
| `--ablation-no-mask` | `ablation_no_mask` | `--ablation-no-mask` | 动作掩码消融实验 |

参数名称必须包含分隔符。例如 batch size 的正确写法是：

```bash
--batch-size 16
```

兼容的下划线别名为：

```bash
--batch_size 16
```

`--batchsize` 不是有效参数。

### 使用 `--set` 覆盖其他配置

没有独立命令行参数的 `Config` 字段统一使用：

```bash
--set key=value
```

每个配置项使用一个独立的 `--set`，可以重复指定：

```bash
python train.py \
  --trainer lightning \
  --config conf/experiment/initial_schedule_283.yaml \
  --batch-size 16 \
  --num-envs 8 \
  --set lr=0.00005 \
  --set gamma=0.999 \
  --set accumulation_steps=8
```

不同数据类型的正确写法：

```bash
# 整数
--set k_epochs=2

# 浮点数
--set sample_temperature=1.0

# 布尔值必须写 true 或 false，不能写 0/1
--set enable_rollout_profiler=true
--set use_compile=false

# 字符串和路径
--set experiment_name=reschedule_task_delay
--set reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt

# 列表；Linux 建议用引号防止 shell 解释特殊字符
--set 'eval_scenarios=[standard]'
```

PowerShell 列表示例：

```powershell
--set "eval_scenarios=[standard]"
```

常用 PPO 与性能配置示例：

```bash
python train.py \
  --trainer legacy \
  --config conf/experiment/reschedule_task_delay.yaml \
  --batch-size 16 \
  --num-envs 16 \
  --max-episodes 300 \
  --eval-freq 1 \
  --set accumulation_steps=16 \
  --set auto_oom_retry=true \
  --set oom_max_retries=1 \
  --set enable_gpu_batch_rebuild=true \
  --set enable_rollout_profiler=true
```

结构性参数也可通过 `--set` 指定，例如：

```bash
--set hidden_dim=128
--set num_gat_layers=5
--set num_heads=4
--set use_shared_trunk=false
```

加载 checkpoint 时，程序会自动识别其模型结构。若命令行显式指定的结构参数与 checkpoint 不一致，程序会报错并拒绝加载，避免静默使用错误结构。

恢复最近的 Lightning checkpoint：

```powershell
python train.py --trainer lightning --config conf/experiment/initial_schedule_283.yaml --resume
```

Windows GPU 冒烟测试（5 步 rollout、一次 PPO 更新和一次 Standard 验证）：

```powershell
python train.py --trainer lightning --config conf/experiment/smoke_lightning.yaml
```

历史训练循环仅用于回归对照：

```powershell
python train.py --trainer legacy --config conf/experiment/initial_schedule_283.yaml
```

## 重调度训练

当前重调度训练必须使用 `legacy` 入口，因为 baseline 自动生成、固定重调度验证场景和初始调度模型 warm-start 尚未接入 Lightning 入口。

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
  --config conf/experiment/initial_schedule_283.yaml `
  --model-path checkpoints/initial_schedule/best_680.ckpt `
  --output-path results/final_schedule.csv
```

Linux 写法：

```bash
python scripts/generate_schedule.py \
  --config conf/experiment/initial_schedule_283.yaml \
  --model-path checkpoints/initial_schedule/best_680.ckpt \
  --output-path results/final_schedule.csv
```

生成后建议先验证 APAL 约束：

```powershell
python utils/verify_schedule.py --data_path data/283.csv --schedule_path results/final_schedule.csv
```

不同规模应使用独立文件名，避免意外复用。例如 680：

```bash
python scripts/generate_schedule.py \
  --config conf/experiment/initial_schedule_680.yaml \
  --model-path checkpoints/initial_schedule/best_680.ckpt \
  --output-path results/final_schedule_680.csv
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
  --trainer legacy `
  --config conf/experiment/reschedule_task_delay.yaml `
  --set reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt `
  --set reschedule_baseline_schedule_path=results/final_schedule.csv
```

Linux 写法：

```bash
python train.py \
  --trainer legacy \
  --config conf/experiment/reschedule_task_delay.yaml \
  --set reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  --set reschedule_baseline_schedule_path=results/final_schedule.csv
```

启动日志应包含：

```text
重调度 baseline: .../results/final_schedule.csv
固定重调度验证场景: .../results/reschedule_eval_scenarios.csv
重调度 warm-start: .../checkpoints/initial_schedule/best_680.ckpt
```

如果 `results/reschedule_eval_scenarios.csv` 不存在，程序会根据 baseline 和配置的固定随机种子自动生成。训练期间每个 episode 都会在这些固定场景上验证，最优模型按重调度综合得分选择，并要求所有验证场景满足保存资格。

680 数据的完整示例：

```bash
python train.py \
  --trainer legacy \
  --config conf/experiment/reschedule_task_delay.yaml \
  --data-path data/680.csv \
  --train-data-path data/680.csv \
  --set n_w=100 \
  --set n_w_min=100 \
  --set reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  --set reschedule_baseline_schedule_path=results/final_schedule_680.csv \
  --set reschedule_eval_scenario_path=results/reschedule_eval_scenarios_680.csv
```

如果此前已经生成了其他规模的验证场景，必须更换 `reschedule_eval_scenario_path` 或删除旧文件后重新生成。程序只会在目标文件不存在时自动生成场景，不会自动判断已有场景是否属于当前 baseline。

### 4. 输出与续训

重调度主要输出：

| 输出 | 路径 |
|---|---|
| 最新训练断点 | `checkpoints/reschedule_task_delay/latest_checkpoint.pth` |
| 最优重调度模型 | `checkpoints/reschedule_task_delay/bestmodel/best_model.pth` |
| 最优模型元数据 | `checkpoints/reschedule_task_delay/bestmodel/best_model_meta.json` |
| 固定验证场景 | `results/reschedule_eval_scenarios.csv` |
| TensorBoard 日志 | `/root/tf-logs/reschedule_task_delay_ALB_PPO_<时间戳>/` |

断点续训：

```bash
python train.py \
  --trainer legacy \
  --config conf/experiment/reschedule_task_delay.yaml \
  --set reschedule_baseline_model_path=checkpoints/initial_schedule/best_680.ckpt \
  --set reschedule_baseline_schedule_path=results/final_schedule.csv \
  --resume
```

### 5. 评估重调度模型

```bash
python evaluate_reschedule_model.py \
  --config conf/experiment/reschedule_task_delay.yaml \
  --model-path checkpoints/reschedule_task_delay/bestmodel/best_model.pth
```

评估结果保存到：

```text
results/reschedule_ppo_eval/reschedule_ppo_eval.csv
results/reschedule_ppo_eval/reschedule_ppo_eval_summary.json
```

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

自动验证默认每个 episode 执行一次，只运行 Standard 场景，并打印 Makespan、Balance、Reward、人员/站位利用率及耗时。详细指标同时写入 Lightning/TensorBoard。

Lightning 每次 PPO 更新后覆盖保存 `checkpoints/<实验名>/lightning/last.ckpt`；执行自动验证且 Makespan 改善时，覆盖保存 `checkpoints/<实验名>/lightning/best/best.ckpt`。终端会打印对应的 `[Checkpoint]` 记录。

Lightning 与 legacy 的 TensorBoard 日志根目录统一由 `log_dir` 控制，当前固定为 `/root/tf-logs`。Lightning 写入 `/root/tf-logs/<实验名>/version_N/`，legacy 写入 `/root/tf-logs/<实验名>_ALB_PPO_<时间戳>/`。

Rollout 心跳默认关闭。需要诊断长时间阶段时，可在 rollout YAML 中设置：

```yaml
rollout:
  rollout_heartbeat_interval_sec: 30.0
```

## 输出文件与目录

除 `log_dir` 等绝对路径外，以下相对路径均以项目根目录为基准。`<实验名>` 来自实验 YAML 的 `experiment_name`，例如 `initial_schedule_283`。

### Lightning 训练

| 输出 | 路径 | 说明 |
|---|---|---|
| 最新完整 checkpoint | `checkpoints/<实验名>/lightning/last.ckpt` | 每个 episode 覆盖保存，可用于 `--resume` |
| 最优完整 checkpoint | `checkpoints/<实验名>/lightning/best/best.ckpt` | Standard 验证 Makespan 改善时覆盖保存 |
| TensorBoard event | `/root/tf-logs/<实验名>/version_N/events.out.tfevents.*` | rollout、PPO、验证、OOM 和性能指标 |
| Lightning 超参数 | `/root/tf-logs/<实验名>/version_N/hparams.yaml` | Lightning Logger 生成 |

查看日志：

```bash
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
python evaluate_model.py --model-path checkpoints/<实验目录>/<模型文件> --test-data data/283.csv --num-runs 3 --temperature 0 --output-dir results/evaluation
```

生成排程：

```powershell
python scripts/generate_schedule.py --model-path checkpoints/<实验目录>/<模型文件> --output-path results/final_schedule.csv
```

每次训练会在 checkpoint 目录写入 `resolved_config.yaml` 和 `run_manifest.json`，记录最终配置、配置来源、命令类型和 Git commit。训练只负责 checkpoint 与 TensorBoard；GA 对比、排程 CSV 和甘特图由独立命令按需生成。

验证 APAL 排程约束：

```powershell
python utils/verify_schedule.py --data_path data/283.csv --schedule_path results/final_schedule.csv
```

## 测试

```powershell
python -m pytest -q
```

关键架构测试：

```powershell
python -m pytest -q tests/test_config_loader.py tests/test_lightning_architecture.py tests/test_vector_env_safety.py
```

## Skill Hub 资源图

默认使用 Skill Hub 压缩稠密的工人技能边，配置位于 `conf/model/hb_gat_pn.yaml`：

```yaml
model:
  use_skill_hub: true
  skill_hub_bidirectional: true
  num_skill_types: 10
  skill_feat_dim: 16
```

- `use_skill_hub: false`：保留原始 `Worker -> Task` 的 `can_do` 直接边，可用于旧模型和消融对照。
- `use_skill_hub: true`：使用 `Worker -> Skill -> Task` 压缩资源图。
- `skill_hub_bidirectional: true`：额外启用 `Task -> Skill -> Worker` 反向消息；设为 `false` 时仅保留正向链路。
- Skill Hub 与旧直接边互斥，不会同时参与消息传递。
- 新旧图结构的模型参数不兼容；加载 checkpoint 时必须使用训练该 checkpoint 时相同的图模式。

## CUDA OOM 保护

- Windows 低显存配置将 PPO batch 上限设为 `4`；Linux 不设置上限，使用实验配置的默认 batch。
- GPU 图模板仅保留当前数据集，避免训练池轮换时显存逐轮累积。
- PPO 更新首次发生 CUDA OOM 时会完整回滚模型、优化器、AMP 缩放器和随机状态，并将 batch 减半重试一次。
- 减半后再次 OOM 将直接回滚并跳过当前 PPO 更新；TensorBoard 会记录 `OOM/RetryCount`、`OOM/SkippedUpdate` 和 `OOM/EffectiveBatchSize`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用于缓解显存碎片，但不能替代 batch 上限和事务回滚。
