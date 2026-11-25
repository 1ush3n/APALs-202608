# APAL 期刊论文逐步写作计划

> 当前版本优先级：`论文中文初稿.md` 与 `论文简明大纲_导师审阅版_v2.md` 为正文续写依据，`第二章补充文献审计记录.md` 与 `论断-实验-引用对应表.md` 约束缺口和证据强度。旧版 LaTeX、CAC 稿和早期创新点文档仅作素材参考。

## 本轮写作定位

- 论文类型：算法型研究论文。
- 当前章节范围：Introduction 已完成当前轮修订；Related Work 六节中文初稿已写入，下一轮逐节审阅并补齐正式参考文献表。
- 写作语言：中文初稿优先；正文结构和核心结论稳定后再形成英文稿。
- 期刊风格：暂按通用工程/智能制造期刊风格，不强行套 Nature 格式。

## 一句话论点

在 Aircraft Pulse Assembly Line 调度中，本文研究一种面向 APAL 多资源异构图的强化学习框架，通过异构图编码、Skill Hub、可行性掩码和自回归指针解码，序贯构造工序—物理站位—给定人数具体工人集合；当前只能将跨规模能力、模块必要性和重调度迁移写成待验证论点。

## 术语表

| 规范术语 | 首次定义 | 决定 |
|---|---|---|
| Aircraft Pulse Assembly Line (APAL) | 飞机脉动装配线 | 全文使用 APAL，不写成 FJSP 变体 |
| Heterogeneous Graph Pointer PPO (HGP-PPO) | 整体强化学习框架 | 当前工作名；正式投稿前统一确认 |
| HB-GAT-PN | policy network architecture | 当前工作名；与 HGP-PPO 的层级关系保持一致 |
| Skill Hub | worker-skill-operation 分解结构 | 作为结构组成；质量与效率贡献等待正式消融 |
| adaptive feasibility mask | 动态可行性掩码 | 写成硬约束嵌入机制，不写成普通工程技巧 |
| 脉动节拍 $T_{\mathrm{takt}}$ | 初始排程完成后，由考虑站内前序、并行执行和资源约束的各站位最晚完工时间最大值确定的整线脉动周期 | $T_{\mathrm{takt}}=\max_m C_m^{\mathrm{end}}=C_{\max}$；英文优先用 `pulse cycle time`，并说明本文不是预先给定的需求节拍；重调度保留该值作为基准节拍 |
| selection score | 含不可行强罚分的重调度选择指标 | 重调度主指标之一 |
| eligible rate | 硬约束合格率 | 所有重调度结果必须同步报告 |

## 章节任务图

1. Introduction：先定义 APAL 工业问题，再指出不能等同 FJSP/JSSP，最后引出 HGP-PPO。
2. Related Work：按装配线问题谱系、图学习调度、技能工人/异构资源、动态重调度组织。
3. Problem Formulation：给出初始调度和动态重调度两个模型。
4. Method：讲清异构图、Skill Hub、GAT、Pointer、Mask、PPO。
5. Experiments：已完成结果和待补结果分开写，避免过度声称。
6. Discussion and Conclusion：明确当前证据支持什么、尚未证明什么。

## 已使用本地材料

- `docs/CAC/CAC-v6.tex`
- `docs/CAC/main-v6.pdf` 的对应源文件
- `docs/文献学习总结.md`
- `docs/创新点定位与贡献表述.md`
- `docs/APAL顶刊实验优化方案.md`
- `docs/final_initial_experiment_results.md`
- `docs/reschedule_rule_experiment_results.md`
- `docs/参考文献/`

## 当前缺口

- 参考文献信息需要逐篇核验，当前 `references.tex` 只适合作为占位版。
- Skill Hub 的边数、显存、rollout throughput 和质量对比还未写成正式结果。
- no-pointer 消融目前统计口径不同，不能和 10 种子结果完全并列。
- 重调度主方法存在 `real_3182 / low_012` 工位槽位违规，必须先修复或在论文中如实报告。
- 需要补齐多指标表：balance、worker utilization、station utilization、complete rate、valid rate、inference time。
