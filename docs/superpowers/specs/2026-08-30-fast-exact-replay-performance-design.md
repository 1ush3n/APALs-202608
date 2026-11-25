# Fast-Exact Replay 性能与数值合同设计

## 目标

在不破坏 WorkerPointer v2 Fast-Exact 原有 physical-group replay identity 合同的前提下，定位并降低 PPO replay 性能回归：先增加可审计 profiling，再删除训练阶段独立 actor-only prepass，最后以可开关模式验证 dataset-homogeneous logical encoder batching。正式 V0/V1 训练在 parity gate 通过前不启用新 batching 模式。

## 当前事实与边界

- 当前正式参考路径是 `behavior_group_exact_gpu_template_v2`。
- `physical_group` 的成员、顺序、group size 和 replay action semantics 仍是 reference contract。
- `GPUExactBatchBuilder.build()` 一次调用不得混合 `dataset_idx`。
- Builder 的 layout template 与 dynamic worker-skill resource edges 分属不同语义；只有回归证明动态边每次完整覆写后，才允许拆分 cache key。
- P3 task/station head batching、P4 Worker Pointer token-major batching不在本设计范围内。
- Windows/Linux 均须使用 `pathlib`，深度学习路径继续使用现有 `rag_env` 环境和 AMP 约束。

## 分阶段设计

### P0：Fast-Exact profiling

在 `_run_v2_fast_exact_replay_update()` 内记录一次完整 PPO update 的：

- physical group count、logical batch count、group size mean/p50/p95；
- builder call count、build time、template hit/miss；
- actor encoder forward call/time；
- task/station/action-head time、Worker Pointer time；
- actor-only identity validation time；
- formal replay forward time；
- backward time、optimizer step time；
- replay samples/s。

Profiling 默认关闭；打开时在 CUDA 边界使用现有同步约定，避免把额外同步成本混入普通训练。指标通过现有 update metrics 和 benchmark 输出。

### P1：单次 formal forward 与当前 KL

保留一次独立的 no-grad actor-only validation，仅用于 first recompute identity contract。该 validation 不再写入 PPO 训练 KL cache。

每个 logical batch 改为：

1. 每个 reference physical group 只执行一次 formal actor+critic replay；
2. 暂存当前 replay 输出及其 autograd graph；
3. 对同一批 `total_lp.detach()` 与 old logprob 计算当前 KL、loss scale 和 epoch early-stop 统计；
4. 使用同一批输出计算 PPO policy/value/entropy loss 并 backward；
5. optimizer 更新后不复用任何旧参数 logits。

由于 GPUExactBatchBuilder 可能原位复用模板，暂存多个 graph 时必须隔离 replay batch，不能让后续 build 覆写前一批 graph 依赖的输入。

### P1.5：group-size benchmark

在相同 SHA、seed、模型和完整 PPO update 配置下测试 `num_envs=2/4/8/16`（资源不足时记录实际完成的最大值），比较 rollout wall time、PPO update wall time、replay samples/s、GPU utilization mean/p50/p90、peak allocated/reserved VRAM 和 Fast-Exact profile 指标。该阶段不得改变 replay 数值语义。

### P2：logical encoder batching

新增结构模式 `worker_pointer_v2_fast_replay_batching`，取值为 `physical_group` 或 `logical_batch_v1`，默认 `physical_group`。

`logical_batch_v1` 在每个 PPO logical batch 内：

1. 按 `dataset_idx` 分桶；
2. 同一 dataset 的多个 physical groups 展平为一次更大的 GPU builder/encoder batch；
3. 保留 `memory_index`、原始 group id、group position、logical position；
4. encoder 以大 batch 前向；
5. task/station/worker action heads仍按当前逐样本语义执行；
6. 输出按 memory index 恢复，使 reward、advantage、old logprob、mask 和 team trace 精确对齐。

该结构模式必须进入 ModelSpec 和 checkpoint `training_spec`，旧 checkpoint 缺失该字段时使用 reference 默认值，不改变旧模型加载行为。

### P2 parity gate

同一 rollout trace 同时执行 physical reference replay 与 logical batching replay，逐项比较 task/station/team/total logprob、state value、normalized entropy、mask 和 active-sample alignment。

- fp32：每项 MaxAE `<= 1e-4`；
- bf16：每项 MaxAE `<= 1e-3`。

测试覆盖数据集 283、680，多个 `num_envs`，异质 worker count、随机 worker topology、virtual task/station 和不同 team demand。未通过时保留 `logical_batch_v1` 为显式实验模式，不替换 reference。

### P2.1：template cache key

验证相同 dataset 和 node counts、不同 topology 下：

- layout template 可复用；
- dynamic worker-skill edges 每次完整覆写；
- 输出边与对应 snapshot topology 一致。

验证通过后，layout template key 不再携带整批 `topo_keys`；topology 继续由 `(dataset_idx, topo_key)` resource cache 管理。验证失败则保留旧 key。

## 错误处理

- dataset 混合、映射长度不符、group 不完整、非法 action mask、非有限 logits/梯度均继续 fail-closed。
- logical batching 发生 parity 失败、builder 错误或 OOM 时不得静默降级为 reference；由显式结构模式和 strict replay 约束报告失败。
- optimizer.step 前保持现有有限梯度检查和梯度裁剪。

## 验证与交付

每个阶段独立测试并独立提交。P0/P1 使用现有 Fast-Exact 小型集成测试扩展；P2 增加 builder cache-key、logical 映射和 parity regression tests；P1.5 使用现有 benchmark 脚本完成完整 update 记录。最终报告必须列出测试命令、通过/跳过原因、profile/benchmark 输出和 parity MaxAE；不启动正式 V0/V1 训练。

