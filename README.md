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

自动验证按 `eval_freq` 执行，默认只运行 Standard 场景，并打印 Makespan、Balance、Reward、人员/站位利用率及耗时。详细指标同时写入 Lightning/TensorBoard。

Rollout 心跳默认关闭。需要诊断长时间阶段时，可在 rollout YAML 中设置：

```yaml
rollout:
  rollout_heartbeat_interval_sec: 30.0
```

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
