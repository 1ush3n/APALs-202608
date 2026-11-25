# APAL 项目分支归档与变更全景记录 (Branch Archive & Changelog)

**归档日期**: 2026-09-16  
**合并基准目标**: `main` (包含最终完整版本 `feat/mes-station-decoding`)  
**备份快照 Bundle**: `D:\APAL-all-branches-history-20260916.bundle`

---

## 1. 归档概述

在 APAL 项目研发过程中，为探索不同阶段的算法特性（如 CTG 启发式残差、Worker Pointer 架构重构、动态重调度超参数消融、站位贪心 MES 解码以及各阶段诊断），共产生了 20 个工作分支。

经审查分析，当前处于主导地位的最终版本位于 `feat/mes-station-decoding`，该版本已串联集成了历史演进中的所有主要特性与架构升级。
- **17 个分支** 为当前最新主干的历史祖先（Ancestor），其全部提交历史已经完整包含在最新提交中，合并后无需额外保留指针。
- **3 个分支** 包含独立诊断/局部基准测试性质的特化提交（Divergent），已通过 Git Format-Patch 完整导出至本目录，保留所有提交信息与代码变更。
- 全库所有分支指针与提交图谱已完整封包至 `D:\APAL-all-branches-history-20260916.bundle`，确保任何历史状态 100% 可逆。

---

## 2. 20 个分支全景清单

| 分支名称 | 顶端 Commit | 提交日期 | 关系状态 | 核心功能与演进内容 |
| :--- | :--- | :--- | :--- | :--- |
| `feat/mes-station-decoding` | `b68d3d8` | 2026-09-16 | **当前最新主线** | MES 站位最小前瞻贪心解码策略实现与 IEEE 期刊论文手稿 |
| `main-final` | `0ef5c83` | 2026-09-03 | 完整祖先 (Ancestor) | 20 次随机重调度评估脚本（有界并发）与主线收敛基准 |
| `codex/wp-v2-full-upgrade` | `e579a9a` | 2026-09-01 | 完整祖先 (Ancestor) | Worker Pointer V2 全面升级与 Task Delay R5 对齐配置 |
| `codex/full-joint-next-diagnostics` | `0136565` | 2026-08-30 | 完整祖先 (Ancestor) | 显式 partial-team 运行状态诊断与支撑 |
| `codex/gamma1-action-truncation-diagnostic` | `daa8d69` | 2026-08-29 | **分叉归档 (Patch)** | 评估 gamma1 动作截断对 R5 场景影响的诊断工具与测试 |
| `codex/reschedule-gamma1-ev-stable` | `44dbe52` | 2026-08-28 | 完整祖先 (Ancestor) | 重调度 gamma1 场景下稳定 Critic 的训练配置与实验脚本 |
| `codex/full-joint-diagnosis` | `2a04bc1` | 2026-08-28 | **分叉归档 (Patch)** | 全联合动作诊断探针、PPO clipping 与策略熵诊断模块 |
| `codex/reschedule-objective-reweighting` | `57df419` | 2026-08-27 | 完整祖先 (Ancestor) | 重调度多目标权重重平衡与防 OOM 批量调整 |
| `codex/dual-attention-context-screen` | `3a0106f` | 2026-08-26 | 完整祖先 (Ancestor) | 接入双头注意力筛查（Dual-Attention Context Screen）训练与评估入口 |
| `codex/operation-node-scope-ablation` | `93f899d` | 2026-08-24 | 完整祖先 (Ancestor) | 工序节点范围消融、统一主 PPO 动作 mask 与数值安全链路 |
| `codex/batched-vectorized-worker-pointer-v2` | `96df9ff` | 2026-08-18 | 完整祖先 (Ancestor) | 批量向量化 Worker Pointer V2 策略观测与快照 |
| `codex/local-performance-validation` | `7a227f9` | 2026-08-18 | **分叉归档 (Patch)** | 本地单机推理与环境交互性能基准测试记录 |
| `codex/apal-anchor-proposal-cf` | `06aca58` | 2026-08-14 | 完整祖先 (Ancestor) | CUDA 快速精确异步反事实锚点推荐（Anchor Proposal CF）验证 |
| `fix/schedulefree-checkpoint-eval-state` | `e50ac10` | 2026-08-05 | 完整祖先 (Ancestor) | ScheduleFree 优化器 checkpoint 评估状态修复与训练池冻结 |
| `feat/apal-ctg-prior-residual` | `6c6c65c` | 2026-08-05 | 完整祖先 (Ancestor) | 相对启发式先验（Relative Heuristic Prior）CTG 残差策略 |
| `feat/apal-conditional-resource-gate` | `f0a938d` | 2026-08-04 | 完整祖先 (Ancestor) | 条件资源门控（Conditional Resource Gate）与五技能协作约束 |
| `codex/ctg-candidate-audit` | `0f6da59` | 2026-08-07 | 完整祖先 (Ancestor) | CTG 候选时机审计工具与 margin-2 策略配置 |
| `feat/scale-400-800` | `600599d` | 2026-07-26 | 完整祖先 (Ancestor) | 400~800 节点大尺度实例初始调度模型与异步验证 |
| `codex/multiscale-apal-generalization` | `9c604f5` | 2026-06-17 | 完整祖先 (Ancestor) | Lightning 多基准 checkpoint 泛化评估支撑 |
| `main` (旧) | `144a8a3` | 2026-06-17 | 完整祖先 (Ancestor) | 历史仓库主分支初始保留基准点 |

---

## 3. 分叉诊断分支 Patch 详情

### (1) `codex/full-joint-diagnosis`
- **分叉基准**: `44dbe52` (codex/reschedule-gamma1-ev-stable)
- **提交范围**: `44dbe52..2a04bc1` (共 3 个 Commit)
- **对应补丁目录**: `archive/branch_records/full_joint_diagnosis/`
  - `0001-feat-add-full-joint-diagnostic-probes.patch`: 增加 `diagnostics/full_joint.py` 与 `scripts/diagnose_full_joint.py`
  - `0002-fix-apply-diagnostic-scenario-settings-per-run.patch`: 诊断场景参数独立隔离
  - `0003-feat-add-PPO-clipping-and-entropy-diagnostics.patch`: 增加 `scripts/diagnose_ppo_dynamics.py` 分析策略熵与剪切率
- **代码影响统计**: 6 个文件修改，新增 1749 行。

### (2) `codex/gamma1-action-truncation-diagnostic`
- **分叉基准**: `3955b8c`
- **提交范围**: `3955b8c..daa8d69` (共 1 个 Commit)
- **对应补丁目录**: `archive/branch_records/gamma1_action_truncation/`
  - `0001-feat-add-gamma1-action-truncation-r5-evaluation.patch`: 增加 `scripts/evaluate_action_truncation_r5.py` 与单元测试
- **代码影响统计**: 3 个文件修改，新增 283 行。

### (3) `codex/local-performance-validation`
- **分叉基准**: `2b808fa`
- **提交范围**: `2b808fa..7a227f9` (共 1 个 Commit)
- **对应补丁目录**: `archive/branch_records/local_performance_validation/`
  - `0001-chore-record-local-performance-validation.patch`: 记录单机 RTX 环境下的吞吐与延迟评测报告
- **代码影响统计**: 2 个文档，新增 171 行。

---

## 4. 历史恢复与补丁使用方法

### 方案 A: 通过 Patch 恢复特定诊断功能
若未来需要重新使用某个诊断脚本，可在任意分支上直接应用补丁：
```bash
# 查看补丁统计
git apply --stat archive/branch_records/full_joint_diagnosis/*.patch

# 应用补丁至当前工作区
git apply archive/branch_records/full_joint_diagnosis/*.patch

# 或者保留完整 commit 历史提交
git am archive/branch_records/full_joint_diagnosis/*.patch
```

### 方案 B: 通过 Git Bundle 完整恢复历史分支
在仓库根目录下，如需查看或检出归档时的任意分支状态：
```bash
# 从安全 Bundle 检出历史分支
git checkout -b recovered/full-joint-diagnosis codex/full-joint-diagnosis
# 或从 bundle 文件导入：
git fetch D:\APAL-all-branches-history-20260916.bundle "*:refs/bundle/*"
```
