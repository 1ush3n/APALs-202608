# Fast-Exact Replay 性能与数值合同实施计划

> **面向执行者：** 本计划按任务逐项执行；每项先写失败测试，再写最小实现，测试通过后独立提交。

**目标：** 在保持 `physical_group` Fast-Exact reference contract 的同时，增加可审计 profiling，消除训练阶段 stale KL prepass，并以 parity gate 验证可选的 dataset-homogeneous logical encoder batching。

**架构：** `physical_group` 继续作为默认参考路径。P1 在每个 logical batch 中只做一次 formal actor+critic replay，并从同一批输出计算 KL 与 PPO loss；P2 仅替换 encoder 的批量组织方式，action heads 继续使用现有逐样本语义。Builder 的 layout template cache 与 topology resource cache 只有在回归证明后才解耦。

**技术栈：** Python、PyTorch、PyTorch Geometric、pytest、Hydra/OmegaConf 风格配置、现有 Fast-Exact GPUExactBatchBuilder 和 benchmark 脚本。

**规格：** `docs/superpowers/specs/2026-08-30-fast-exact-replay-performance-design.md`

## 全局约束

- 默认 `worker_pointer_v2_fast_replay_batching` 必须为 `physical_group`。
- `GPUExactBatchBuilder.build()` 一次调用不得混合 `dataset_idx`。
- fp32 parity MaxAE 必须 `<= 1e-4`；bf16 parity MaxAE 必须 `<= 1e-3`。
- P3 task/station batching 与 P4 Worker Pointer token-major batching不在本计划内。
- 所有路径操作使用 `pathlib`；所有新增深度学习 forward 继续经过现有 AMP 上下文。
- 不修改用户已有的 `README.md`、实验配置、测试文件和 `_docx_work/`，除非本计划明确列出。
- 不启动正式 V0/V1 训练；只执行小型回归和明确的完整 PPO update benchmark。

---

### 任务 1：P0 profile 数据结构与统计函数

**文件：**
- 修改：`training/fast_exact_benchmark.py`
- 测试：`tests/test_fast_exact_benchmark.py`

**接口：**
- 消费：Fast-Exact update metrics 和 group size 序列。
- 产出：可复用的 group-size percentile 统计和 profile metric 汇总函数，不依赖 CUDA。

- [ ] 写失败测试：对空、单元素和多元素 group size 验证 mean/p50/p95；对零秒 update 验证 replay samples/s 为 0。
- [ ] 运行 `D:/Conda/envs/rag_env/python.exe -m pytest tests/test_fast_exact_benchmark.py -q`，确认新断言因函数缺失或结果缺失失败。
- [ ] 实现最小统计函数，使用现有 NumPy/stdlib，不添加依赖；保持现有 benchmark API 兼容。
- [ ] 重新运行同一测试文件，确认通过。
- [ ] 运行 `git diff --check`，提交 `test/feat: add fast-exact profile summaries`。

### 任务 2：P0 在 Fast-Exact update 中采集 profile

**文件：**
- 修改：`configs.py`
- 修改：`ppo_agent.py`
- 修改：`training/v2_fast_exact_batch.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_replay.py`
- 测试：`tests/test_fast_exact_benchmark.py`

**接口：**
- 消费：现有 `_run_v2_fast_exact_replay_update()`、`_replay_v2_fast_exact_group()` 和 `GPUExactBatchBuilder`。
- 产出：`V2/FastExact/Profile/*` 数值指标；builder build call/hit/miss 计数；默认关闭的 profile 开关。

- [ ] 写失败测试：开启 profile 后检查 physical/logical count、mean/p50/p95、builder/encoder/formal replay/backward/optimizer 调用计数和时间键存在且有限；关闭 profile 时检查训练语义仍可运行。
- [ ] 运行目标测试，确认 profile keys 尚未产生或 builder call counter 尚未存在。
- [ ] 在 `Config` 增加 `worker_pointer_v2_fast_exact_profile: bool = False`；在 builder 增加 `build_calls`，每次 `build()` 加一。
- [ ] 在 update 层用 `time.perf_counter()` 采集 build、precheck、formal replay、backward、optimizer；在 replay 层采集 encoder、action head、worker pointer 调用/时间；只在 profile 打开时做 CUDA 同步。
- [ ] 将 group size mean/p50/p95、template hit/miss、replay samples/s 写入 metrics；所有值转换为有限 Python float。
- [ ] 重新运行目标测试及 `tests/test_fast_exact_benchmark.py`，确认通过。
- [ ] 提交 `feat: profile fast-exact replay stages`。

### 任务 3：P1 先写当前 KL 语义的失败测试

**文件：**
- 修改：`tests/test_worker_pointer_v2_fast_exact_replay.py`

**接口：**
- 消费：现有单样本 Fast-Exact update helper。
- 产出：证明正式 PPO epoch 不再使用 actor-only prepass/cache，且每个 logical batch 的 formal replay 输出只 forward 一次。

- [ ] 将旧的 `test_fast_exact_actor_prepass_is_recomputed_after_optimizer_step` 改为断言：两 epoch 只包含一次独立 identity validation，正式 actor-only 调用不按 epoch 增长，`PrecheckReusedGroups == 0`。
- [ ] 新增计数测试：对 formal replay 调用记录每个 epoch 的 group forward 次数，确认每个 group 每个 PPO epoch恰好一次。
- [ ] 运行目标测试，确认它们因当前 `precheck_cache` 复用而失败。

### 任务 4：P1 实现单次 formal forward + 当前 KL

**文件：**
- 修改：`ppo_agent.py`
- 可能修改：`training/v2_fast_exact_batch.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_replay.py`

**接口：**
- 消费：P1 失败测试、factorized critic 分支、现有梯度检查。
- 产出：无 stale KL cache 的 Fast-Exact update；同一 formal outputs 同时用于 KL 和 PPO loss。

- [ ] 删除训练阶段 `precheck_cache` 的读取、写入和 optimizer-step 清理逻辑；identity validation 仍独立 no-grad 执行。
- [ ] 对每个 logical batch 构建 physical replay batch，执行一次 formal replay，保存 outputs 和用于 KL 的 detached total logprob；禁止为 KL 再次调用 replay。
- [ ] 在所有 formal outputs 已生成后计算 stable log ratio、KL、loss scale 和 early-stop；复用同一 outputs 计算 policy/value/entropy loss 与 backward。
- [ ] 若 Builder 模板原位复用会覆盖 autograd 输入，增加最小 batch isolation helper，仅在 P1 保留 graph 时克隆 batch、layout、raw slices 和 masks；不改变 reference 的节点/边值。
- [ ] 保留 factorized component logprob/value loss、gradient finite check、clip 和 optimizer 顺序。
- [ ] 运行 P1 目标测试；再运行 `tests/test_worker_pointer_v2_fast_exact_contract.py` 与 `tests/test_worker_pointer_v2_fast_exact_compat.py`。
- [ ] 提交 `fix: use current formal replay for fast-exact kl`。

### 任务 5：P1.5 benchmark 参数与记录

**文件：**
- 修改：`scripts/benchmark_worker_pointer_v2_fast_exact.py`
- 修改：`training/fast_exact_benchmark.py`（仅在缺少指标汇总时）
- 测试：`tests/test_fast_exact_benchmark.py`

**接口：**
- 消费：现有 `--mode v2_fast_exact`、`--data`、`--num-envs`、完整 update 流程。
- 产出：每个 env 数独立 JSON，含 rollout/update wall time、samples/s、GPU mean/p50/p90、显存和 P0 profile。

- [ ] 写失败测试：验证 benchmark result schema 包含 `num_envs`、`replay`、`gpu_utilization.update`、`profile` 和 builder stats。
- [ ] 运行 schema 测试确认缺少新字段时失败。
- [ ] 将 update profile 与 builder counters 复制到 benchmark JSON；保持 `pathlib` 路径和子进程隔离。
- [ ] 运行 schema 测试通过。
- [ ] 逐一执行完整合法 PPO update：`num_envs=2/4/8/16`，数据 `data/680.csv`；若硬件/时间无法完成，记录实际 exit code 和原因，不伪造结果。
- [ ] 提交 `feat: export fast-exact profile benchmark`。

### 任务 6：P2 配置、ModelSpec 和 checkpoint training_spec

**文件：**
- 修改：`configs.py`
- 修改：`runtime/modes.py`
- 修改：`runtime/configuration.py`
- 修改：`runtime/checkpoints.py`
- 修改：`conf/model/hb_gat_pn.yaml`
- 测试：`tests/test_worker_pointer_v2_fast_exact_config.py`
- 测试：`tests/test_checkpoint_metadata.py`

**接口：**
- 消费：结构模式字符串。
- 产出：`physical_group | logical_batch_v1` 配置校验、ModelSpec 字段和 checkpoint training_spec 字段。

- [ ] 写失败测试：默认值为 `physical_group`；非法模式被拒绝；`build_model_spec()` 与 `build_checkpoint_metadata()` 保存该模式；旧 metadata 缺字段时恢复为 `physical_group`。
- [ ] 运行配置/metadata 目标测试，确认字段不存在或校验缺失而失败。
- [ ] 增加常量和配置字段；仅 Fast-Exact 允许该配置；非 Fast-Exact 配置若显式设置非默认值则拒绝，避免语义漂移。
- [ ] 扩展 `ModelSpec`、build/load/validate 逻辑，保留旧 checkpoint 默认兼容。
- [ ] 更新 Hydra model 配置默认值；不改变现有 pilot 的 physical reference 行为。
- [ ] 运行目标测试及 `tests/test_config_loader.py`、`tests/test_lightning_architecture.py`。
- [ ] 提交 `feat: add fast-exact replay batching structural mode`。

### 任务 7：P2 logical encoder batching 映射测试

**文件：**
- 修改：`training/worker_pointer_v2_replay.py`
- 测试：新建 `tests/test_worker_pointer_v2_fast_exact_batching.py`

**接口：**
- 消费：logical batch 中的 physical group 索引、snapshot dataset_idx 和 memory indices。
- 产出：按 dataset 分桶且保持 group/member 顺序的纯 Python 映射函数。

- [ ] 写失败测试：输入跨 dataset 的 group 列表，输出按 dataset 分桶；每个 bucket 保存 flatten 后的 memory indices、original group index、group position、logical position；输入空组或重复 position 时拒绝。
- [ ] 运行新测试确认函数缺失而失败。
- [ ] 实现最小映射函数，不引入新类；使用 dataclass 仅在已有类型不足以表达映射时。
- [ ] 运行新测试通过，并覆盖异质 worker count 和 topology key 仅作为 metadata 不改变 memory 顺序。
- [ ] 提交 `test/feat: map dataset-homogeneous fast-exact batches`。

### 任务 8：P2 logical batching 接入 replay

**文件：**
- 修改：`ppo_agent.py`
- 修改：`training/worker_pointer_v2_replay.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_batching.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_replay.py`

**接口：**
- 消费：任务 6 的 structural mode 和任务 7 的 dataset bucket mapping。
- 产出：`logical_batch_v1` 一次 builder/encoder 处理同 dataset 的多个 physical groups；输出按 memory index 对齐；默认 reference 不变。

- [ ] 写失败测试：同一 logical batch 含两个 dataset 时 builder 被调用两次而非一次；同 dataset 多 physical groups 时 encoder forward call 数降为 dataset bucket 数；输出 memory indices 保持原顺序。
- [ ] 运行目标测试确认当前实现仍按 physical group 调用并失败。
- [ ] 在每个 logical batch 内按 snapshot `dataset_idx` 分桶，flatten group members；同 dataset bucket 调用一次 `_build_v2_fast_exact_group`/encoder，保留映射。
- [ ] 让现有 action-head replay 循环消费大 batch 的每个 sample，使用 batch ptr 和 raw slices；不改 task/station/worker head 的计算语义。
- [ ] 使用 memory index 恢复 PPO targets 和 outputs；所有 mask、team trace、advantage、old logprob 与 sample 对齐。
- [ ] 对 logical 模式增加结构指标：dataset bucket count、encoder call count、logical encoder sample count；physical 模式指标保持可比。
- [ ] 运行 batching/replay 目标测试及 Fast-Exact contract tests。
- [ ] 提交 `feat: batch fast-exact encoder by dataset`。

### 任务 9：P2 parity gate

**文件：**
- 修改：`tests/test_worker_pointer_v2_fast_exact_batching.py`
- 可能修改：`ppo_agent.py`（仅导出 parity 所需的现有 replay 输出）

**接口：**
- 消费：同一 memory/rollout trace、同一模型参数、physical 与 logical 两种模式。
- 产出：逐项 MaxAE 和 active-sample alignment 的可执行 gate。

- [ ] 写失败测试：构造同一 rollout 的 physical reference 与 logical replay，比较 task/station/team/total logprob、state value、normalized entropy、mask 和 memory index；覆盖 dataset 283/680 的可用 fixture。
- [ ] 先运行 parity 测试，确认新模式尚不存在或至少无法通过 gate。
- [ ] 实现最小 parity helper，fp32 阈值 `1e-4`、bf16 阈值 `1e-3`，失败时输出 dataset、num_envs、group、sample 和字段名。
- [ ] 运行 fp32 parity；CUDA 可用时运行 bf16 parity；CPU 环境对 bf16 明确 skip 并记录原因。
- [ ] 扩展覆盖异质 worker count、随机 topology、virtual task/station、不同 team demand 和多个 num_envs 配置。
- [ ] 提交 `test: gate fast-exact logical replay parity`。

### 任务 10：P2.1 template cache key 审查与优化

**文件：**
- 修改：`training/v2_fast_exact_batch.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_builder.py`
- 测试：`tests/test_worker_pointer_v2_fast_exact_batching.py`

**接口：**
- 消费：同 dataset、相同 node counts、不同 topology 的 snapshot。
- 产出：layout template 可复用而 dynamic worker-skill edges 每次完整覆写的 builder 行为。

- [ ] 写失败测试：相同 dataset/node counts 的 topology A/B 连续 build，断言输出 resource edges 分别匹配 A/B；断言修改后的 template hit/miss 统计符合预期。
- [ ] 运行测试确认当前完整 `topo_keys` key 产生 miss，从而使测试以预期方式失败。
- [ ] 从 template key 移除 `topo_keys`；保留 `(dataset_idx, topo_key)` topology resource cache；确保 `_rebuild_in_place()` 每次重新绑定当前 topology edge。
- [ ] 运行 builder、logical batching 和 parity 测试；若动态边覆盖不完整，撤销 key 缩减并保留原行为。
- [ ] 提交 `perf: separate fast-exact layout and topology caches`，或在 gate 失败时提交保留 key 的回归测试提交。

### 任务 11：最终全量验证与报告

**文件：**
- 可能修改：`README.md` 仅在用户明确要求记录 benchmark 时；默认不改。
- 新增：`artifacts/` 下 benchmark JSON 仅当仓库已有约定允许时；否则保留外部临时目录。

**接口：**
- 消费：所有阶段测试、P1.5 benchmark、P2 parity 输出。
- 产出：可审计的最终状态，不启动正式训练。

- [ ] 运行 `D:/Conda/envs/rag_env/python.exe -m pytest`，记录完整退出码和失败列表。
- [ ] 运行 Fast-Exact 相关测试集合：`tests/test_fast_exact_benchmark.py`、`tests/test_worker_pointer_v2_fast_exact_*.py`、`tests/test_checkpoint_metadata.py`、`tests/test_config_loader.py`。
- [ ] 复核 P1.5 每个完整 update 的 wall time、samples/s、GPU util、VRAM、physical groups、encoder/builder calls、template hit rate、parity MaxAE、fallback/OOM。
- [ ] 复核 `git diff --check`、`git status --short`，确认用户原有修改未被覆盖。
- [ ] 只在所有要求有证据时报告完成；未完成项逐项列出实际阻塞和命令输出。

