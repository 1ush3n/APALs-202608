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

Lightning 模型使用 `.ckpt`，legacy 模型使用 `.pth`，二者不能直接互换。加载模型时还必须保持与训练时相同的 Skill Hub 图配置。

## 评估与排程

手动评估保留 Standard、工时扰动、人员扰动和动态事件四场景：

```powershell
python evaluate_model.py --model_path checkpoints/<实验目录>/<模型文件> --test_data data/283.csv --num_runs 3 --temperature 0 --seed 42
```

生成排程：

```powershell
python scripts/generate_schedule.py
```

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
