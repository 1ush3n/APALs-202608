# 网页版 Gemini 任务模式交接提示词（APAL 论文高水准配图重绘任务）

> **使用说明**：请将下方分隔线之间的完整内容直接复制粘贴到**网页版 Gemini 任务模式**中，并允许其访问你的 GitHub 仓库 `https://github.com/1ush3n/APALs-202608`（`main` 分支）。提示词已完整保留至今全部技术约束、审美偏好、正反案例诊断、已通过模块与 5–10 轮精细打磨闭环规范。

---

你是顶刊/顶会级别的科研可视化与计算机系统/智能制造领域配图专家。请你接手我 GitHub 仓库 **`https://github.com/1ush3n/APALs-202608`**（分支：`main`）中的论文配图重绘任务（目标期刊：**IEEE Transactions on Systems, Man, and Cybernetics: Systems (IEEE TSMC)**，10 页双栏英文期刊手稿）。

---

## 一、仓库必读真相源与历史资产地图（先读再动图）

在提出任何设计方案或动笔绘图前，请先阅读并核对仓库中的以下真实材料：

1. **论文手稿与代码真相源（所有术语、符号、公式、数值必须以此为准，绝不允许编造）：**
   - 英文目标主稿：`docs/APAL_HGP_PPO_Journal_Manuscript/01_Manuscript_TSMC_English_10p/main_tsmc.tex` 与 `main_tsmc.pdf`
   - 中文对齐稿：`docs/APAL_HGP_PPO_Journal_Manuscript/02_Manuscript_Chinese_12p/main_12p.tex`、`docs/APAL_HGP_PPO_Journal_Manuscript/03_Markdown_Drafts/论文中文初稿_最新审阅对齐版.md`
   - 算法与环境核心代码：`models/hb_gat_pn.py`（异构图编码、类型注意力池化、操作-站位-工人指针与动态屏蔽）、`environment.py`（前向时间分配与物理约束推进）、`ppo_agent.py`（PPO 更新与 GAE）
   - 真实数据与算例 CSV：`output/figures_reference_revision/A1_source.csv`、`data/283.csv`、`data/r5_task_delay_v1/baselines/real/real_283_schedule.csv`、`data/r5_task_delay_v1/scenarios/real/real_283_scenarios.csv`、`data/r5_task_delay_v1/real_283_medium_early_repaired_schedule.csv`、`results/02_reschedule_main/r5_eval_full_x_bic_120ep/reschedule_eval_summary.json`
   - *(仓库体积说明：为控制 `main` 分支总追踪体积 `< 90 MB` 以顺利导入网页版 Gemini，大体积 `.pptx` 及重复 600 DPI 位图/旧编译 PDF 留在本地，所有 `.py` 绘图源码、轻量预览图 `previews/`、矢量 `pdf/`、`boards/` 及核心真实数据文件均已直接收录在 `main` 分支中。)*

2. **我原有的审美原则、16 模块精细打磨成果与术语审计：**
   - 初始交接规范：`output/figures_english_from_17/科研配图优化_给下一位AI.md`
   - **逐词术语审计与旧图致命缺陷清单**：`output/figures_english_from_17/refined_v8/PAPER_TERMINOLOGY_AUDIT_AND_INTEGRATION_ROADMAP.md`（**必读！**）
   - 17 页带批注原始母稿与预览：`output/figures_reference_revision/`（含 `A1_source.csv`、`A1_preview.png`、`B4_preview.png`）
   - **最新 16 模块打磨稿（v8，已历经 8 轮逐页核对）**：`output/figures_english_from_17/refined_v8/`（含预览 `previews_refined_v8/slide_01..16.png` 及源码 `scripts/`）
   - **合法开源航空总装现场照片（CC BY 2.0，可直接学术复用）**：`output/figures_english_from_17/refined_v8/assets/b1_apal_workcell_v4_ccby2.png`（授权信息见 `output/figures_english_from_17/refined_v8/assets/PHOTO_LICENSE_ATTRIBUTION.md`；**禁止使用任何 AI 生成的假飞机或版权不明图片**）

3. **已整合预览版与两套参考图库：**
   - 整合尝试 v1：`output/figures_english_from_17/paper_integrated_v1/`
     - `previews/fig1_apal_unfolding_delay.png` —— **【明确否决的反面教材】**
     - `previews/fig2_compact_b1_b5.png` —— 我认为它左半部分（实景产线→异构状态图）的构图逻辑与视觉质感**远比现版 Fig. 1 更适合作为论文 Fig. 1**
     - `previews/fig3_standalone_b5.png` —— 双塔编码器独立版（已修复 `LeakyReLU (0.1)` 与 `Disjoint weights` 居中问题，但仍缺解码部分）
     - `previews/fig_appA_b8_worker_pointer.png` —— **已高度认可的 Appendix A 工人指针两次抽取对比图（B8）**，后续 Fig. 1 / Fig. 2 的配色、字体与符号必须与其保持统一
   - **高分经典参考图集（我曾亲自打分）**：`output/figure_study_15/figure_reference_atlas.pdf` 与 `assets/`
     - ReSched Fig. 2（8.5 分，`a2.png`）：同一候选行贯穿输入、计算与输出
     - Wang & Gombolay RSS'20 Fig. 1（8.5 分，`a3.png`）：实景照片与图节点严格一一映射
     - DAN Fig. 3（8.0 分，`a4.png`）：双分支尺寸对称、上下游严格对齐
     - Smit AAAI'25 Fig. 1（7.5 分，`a5.png`）：全局位置与单步真实计算微放大结合
     - PipeDream Fig. 5（7.5 分，`b4.png`）：多资源对齐时间轴，用几何位置代替大段文字解释
   - 第二轮检索参考图库：`output/figures_english_from_17/reference_study_v2/`（`Top_Venue_Reference_Gallery_v2.pptx`、`boards/`、`REFERENCE_CATALOG_AND_REDESIGN_PROPOSALS.md`）。**注意我的诚实反馈**：里面部分图偏纯 CS/系统算法且略显“AI/GPT 一键生成感”，其中少数优质架构（如 AAMAS'24 `2D-Ptr` `C3_01`、HGS 三级掩码解码 `C3_02`、NeurIPS'23 `LEHD` 宏微分离 `C3_06`、NeurIPS'21 `MatNet` 双塔 `C3_07`）可借鉴结构思路，但**Fig. 1 绝不要照搬那批流水线格子或像 GPT 生成的示意图，必须重新搜寻制造业/航空装配/CPS/调度顶刊中真正严谨耐看的真实标杆图！**

---

## 二、核心决策与问题诊断（我已拍板的最终架构：方案 A）

`main_tsmc.tex` 是紧凑的 **10 页 IEEE TSMC 双栏手稿**（目前方法部分对应 **Fig. 1** [Section I] 与 **Fig. 2** [Section IV-A]，`Appendix A` 尚无配图）。经前期深度对齐，我已确定采用**“两张正文主图 + 附录 B8 选配”架构**：

### 1. 为什么旧图与现版 `fig1_apal_unfolding_delay.png` 彻底不可用？
- **论文里最早的旧 `fig1.png` / `fig2.png` 存在严重硬伤**（见 `PAPER_TERMINOLOGY_AUDIT_AND_INTEGRATION_ROADMAP.md`）：旧 `fig1.png` 有 4 处拼写错误、前后 P1/P3 工序标签自相矛盾、冻结时刻错写成 `t_r`（正文为 `\tau^r`）；旧 `fig2.png` 方法名错写成 `HB-GAT-PPO`、把 Critic 误标为 `Actor Branch`/`Actor Context`、把 `GATv2` 错写成 `GATConv`。
- **`paper_integrated_v1/previews/fig1_apal_unfolding_delay.png` 被永久否决**：左边是带重复小飞机图标的呆板表格方阵，右边是带大段文本框的 Matplotlib 22 行棒棒糖图，整体观感非常怪异且缺乏机制表达，无论怎么微调都不适合放进顶刊作为 Fig. 1。

### 2. 本次需要彻底重绘并打磨到顶刊水准的两张核心正文图：

- **【核心任务 1】全新 Fig. 1（Section I 跨双栏 `0.92\textwidth`）：物理 APAL 产线 ＋ 脉动节拍与物料延误问题语义 → 异构状态图 $G_t$**
  - **重要边界**：不要把右侧神经网络双塔编码器放进 Fig. 1（避免与 Fig. 2 重复）！它只讲**“物理问题与图建模从何而来”**，但要以 `fig2_compact_b1_b5.png` 左侧那种**“高清真实产线实景锚定 + 严谨学术图结构”**为质感基调，并把 APAL 核心问题机制有机融入其中：
    1. **物理实景与实体锚定**：以合法 CC-BY-2.0 波音总装现场图 `b1_apal_workcell_v4_ccby2.png` 为强视觉锚点，用极克制、清晰不杂乱的引线直接对应到 APAL 四类实体：Physical Station $S_m$（并发容量 $c_m$、负荷 $L_m$）、Operation $O_i$（前序 $\mathcal{P}(i)$、工期 $\tilde{p}_i$、所需人数 $q_i$、所需技能 $k_i$、释放时间 $r_i$）、Skill Node $K_k$（$|K|=5$）、Multi-Skilled Worker $W_w$（熟练度 $\eta_w$、5 维技能、等待、持久站位绑定 $\ell_w$）。
    2. **APAL 五站同步脉动机制 + 物料延误触发重调度语义（彻底告别老 Fig. 1 的矩阵+22 行棒棒糖！）**：用紧凑、优雅、非 AI 感的顶刊几何表达（如干净的双轨微型甘特/时空切面或精炼机制对比块），让审稿人在 Fig. 1 一眼看懂两件最本质的事：
       - **五站脉动耦合（Prop. 1）**：5 个工位 $S_1\text{–}S_5$ 按节拍周期 $H$ 同步转位（$\text{Window}(k,m)=[(k+m-1)H, (k+m)H)$），瓶颈工位决定全线节拍，同一架飞机绝不同时出现在两个站位；工人一经首次指派即绑定至对应工位，形成跨工序人力资源耦合。
       - **物料释放延误与稳定性重调度**：明确区分 **重调度触发时刻 $\tau^r$**（图中与论文务必写作 $\tau^r$，而非 $t_r$）与 **物料实际释放时刻 $r_i$**（$\tau^r \ne r_i$）；$t \le \tau^r$ 已提交前缀连同既有工人-站位绑定被冻结继承，未完工操作在保持约束下修复，体现延误如何经共享工人/前序传染导致排队等待 $B_i^r > r_i$。*(仅选用 3–4 个真实代表工序如 `O94, O111, O223` 或严格标注 `illustrative` 的清晰机制条，禁止堆砌 22 行密集数据散点！)*
    3. **四节点五关系异构状态图 $G_t = (O \cup M \cup W \cup K, E_t)$**：与实景形成一一颜色/形状对照，清晰区分实线静态关系（`precedes`, `requires`, `has-skill`）与虚线动态分配关系（`assigned-to`, `done-by`），并可借鉴顶会紧凑属性卡片风格简洁标出特征向量 $x_M, x_O, x_W, x_K$。

- **【核心任务 2】全新统一 Fig. 2（Section IV 跨双栏 `0.92\textwidth`）：HGP-PPO 编解码一体化与 PPO 更新完整框架图**
  - **重要要求**：**将编码（Encoder）与解码（Decoder）彻底融合为同一张完整算法图！** 解决目前 `B5`（只有双塔编码）与 `B6`/`B7`/`B8`（只有独立指针解码）割裂的问题，同时绝不靠缩字号硬塞导致拥挤。
  - **推荐构图逻辑（宏微分离：借鉴 `2D-Ptr` AAMAS'24 + `LEHD` NeurIPS'23 + `MatNet` NeurIPS'21 + `HGS` 三级指针解码）**：
    1. **共享输入与节点映射**：$G_t = (O \cup M \cup W \cup K, E_t) \to$ 节点类型投影 $f_X(x_v) \in \mathbb{R}^{128}$（`LN → LReLU`）。
    2. **独立双塔异构图编码器（Actor $\theta$ vs Critic $\phi$，严格 $\theta \cap \phi = \varnothing$）**：主流程保留清晰、对齐的 `L = 5` 层堆叠编码塔与中间 `Disjoint weights (θ ∩ φ = ∅)` 分割线；**禁止像旧 B5 那样把相同的 4-Head GATv2 内部结构在 Actor 和 Critic 里重复画两遍！** 改为在侧边/底部**单独拉出 1 个精细 Micro-Zoom 放大框**，清晰展示单层 Hetero-GATv2 的真实计算：`Linear W_s h_u + W_t h_v → LeakyReLU (0.1) → GATv2 Attention a_{r,h}^T → Per-Relation Softmax → 4-Head Mean + Relation Sum → Post-Norm LN → LReLU → Add (+)`。
    3. **类型注意力池化 readout**：`Node-Type Attention Pooling [M | O | W]` 输出 $g_\theta, g_\phi \in \mathbb{R}^{3d}$。**关键真相核对**：技能节点 $K$ 参与 GATv2 消息传递并融入 $H^O, H^W$，池化仅对 $M, O, W$ 三类节点聚合，**严禁虚构第四路技能池化**！
    4. **级联三级条件自回归指针解码器 $\pi_\theta(\tilde{a}_t \mid s_t)$**（必须让人看懂顺序因果条件与掩码计算，借鉴 `ReSched Fig. 2` 候选行追踪与 `HGS Fig. 4` 条件屏蔽，不画空洞的 “Pointer” 盒子）：
       - **Tier 1 工序指针 $i_t \in \mathcal{F}_t^O$**：仅前序 $\mathcal{P}(i)$ 完成 $\wedge$ 已释放 $\wedge$ 存在可行 `(Station, Worker)` 的候选通过；不可行项屏蔽为 $-10^4$ / $-\infty$（可用 1 个紧凑矩阵/候选条展示嵌入、logit、precedence mask 与 softmax）。
       - **Tier 2 站位指针 $m_t \in \{S_1,\dots,S_5\}$**：以选中 $h(i_t), k_{i_t}, q_{i_t}$ 为条件，在 5 个站位上按并发容量 $c_m$ 与可用工人数屏蔽并选择站位（并发空间下界不是唯一约束）。
       - **Tier 3 顺序工人团队指针 $\mathbf{w}_t = (w_t^{(1)},\dots,w_t^{(q_{i_t})}) \mapsto W_t$**：工人需求人数 $q_i$ 由工序给定（非网络预测）；显式画出顺序子步条件（$r = 1 \to 2$：部分团队池化 Sum/Mean/Max、动态最早完工时间 EFT、边际技能稀缺度、已选工人 $w_t^{(1)}$ 在下一步被动态 mask 置零），并在首次指派后建立持久工人-站位绑定（有序选择轨迹 $\mathbf{w}_t \mapsto$ 环境并行执行的无序团队 $W_t$）。
    5. **环境前向时间分配 + Critic 价值头 + PPO/GAE 耦合**：
       - **严格事实边界**：动作的实际开工/完工时间 $(B_i, C_i)$ **不是神经网络直接输出的**，而是离散联合动作 $a_t=(i_t,m_t,W_t)$ 送入 **Forward Allocation Engine / Environment** 后确定性推算资源日历所得！
       - **Critic 支路**：经 `MLP (3d → 64 → 1)` 输出 $V_\phi(s_t) \in \mathbb{R}$，与环境奖励共同产生 **TD Error $\delta_t$** 与 **GAE 优势 $\hat{A}_t$** 反馈至策略更新。
       - **PPO 符号硬约束**：策略概率比必须记作 **$r_t^{\text{PPO}}(\theta)$**，目标记作 **$L_{\text{policy}}(\theta)$**（含熵正则项 $-c_H\mathcal{H}(\pi_\theta)$），旧策略记作 **$\theta^-$**。**严禁写 $\rho_t$**（与公式(1)中的团队协同折减因子 $\rho=0.95$ 冲突！）；针对英文稿统一裁剪 vs checkpoint 分头裁剪的已知差异，保留共同有序动作回放与 GAE 主线并在图注或小注保持准确，不篡改代码或编造一致。

- **【配套基线】Appendix A 独立图**：保留并对齐已获认可的 `fig_appA_b8_worker_pointer.png`（完整展示双轮抽人数值演例与技能稀缺度计算，与正文 Fig. 2 形成“正文编解码总图 + 附录放大镜”呼应）。

---

## 三、绝对不能违背的科研绘图“红线”与真实工业美学标准

1. **去“AI 生成感 / GPT 一键感”十条铁律**：
   - 坚决采用**干净、扁平、高精度的 IEEE/NeurIPS 矢量工程制图风**：统一线宽层级、利落正交折线/圆角连接、克制科研配色（延续 `refined_v8` 与 `fig_appA_b8` 的学术蓝/青/赤陶/紫/琥珀色系）。
   - **禁止**：AI 插画式飞机、荧光渐变、厚重立体投影/玻璃拟态、无意义神经节点乱连交叉网、把整页做成微缩文字表格、巨大空白边框里放一行小字（文本框必须与字号匹配紧凑）。
   - **删减自检（Delete Test）**：删除某个图标或连线后，读者是否少理解了一条计算逻辑或约束？若只是“少了点科技感”，立刻删掉！
2. **禁止泄露内部开发与代码标识符**：图面与图注中**绝不出现** `B1/B3/B4/B5/B6/B8/A1-A6`、`GATv2Conv`、`concat=False`、`aggr=sum`、`BIC`、`.csv` 文件名、`ckpt`、或写死的 `Table VII config` 等错误表号。
3. **两套时钟严禁混用**：严格区分“单循环作业模板时域”“真实基准排程/重调度时域（如 $H^0=292.57\text{ h}$、$\tau^r=65.83\text{ h}$）”与“多架机绝对车间日历上的脉动展开”。
4. **实测数值 vs 计算示意严格区分**：若某处概率、注意力权重仅为教学演例，必须在英文图注或图内小字明确标注 `*illustrative*`，不得冒充实测训练记录。
5. **全英文图面，中文字号沟通**：图内所有标题、图例、轴标、公式、演例及 PPT 备注必须 **100% 英文**，且在 IEEE 双栏印刷宽度（`7.16 in` / `0.92\textwidth`）下字号等效 **≥ 7.5 pt** 清晰可读。

---

## 四、强制执行工作流：先寻顶刊真实标杆图 → 再经 5–10 轮闭环精细打磨

请**务必**严格按以下四个阶段逐步推进，**不要试图一轮直接定稿出最终图**：

### Phase 1：跨学科顶刊标杆参考图深度检索与“去 AI 味”筛选（先找参考，不急着画）
1. 针对 **Fig. 1（APAL 产线实景 + 脉动/延误 + 异构图）**，不要只看纯机器学习论文，重点检索 **IEEE TSMC、IEEE TASE、IEEE TII、RCIM（Robotics and Computer-Integrated Manufacturing）、Journal of Manufacturing Systems、EJOR、RSS、ICRA** 中的真实高质量论文 Fig. 1（例如航空/汽车/复杂产品装配线数字孪生与动态重调度框架图、实拍车间与知识图谱/Petri网/异构图映射图），过滤掉一切看起来像“AI 生图”或“粗陋流程图”的配图。
2. 针对 **Fig. 2（编解码一体化 HGP-PPO）**，结合仓库 `reference_study_v2` 里的优质结构样本（`2D-Ptr C3_01`、`HGS Fig. 4 C3_02`、`LEHD C3_06`、`MatNet C3_07`）与 `figure_study_15`（`ReSched Fig. 2`、`DAN Fig. 3`），再补充检索顶会/顶刊中兼顾“GNN 编码 + 多级自回归指针解码 + RL 训练反馈”的最紧凑高分图例。
3. **阶段交付并暂停对齐**：向我展示你筛选出的 Top 6–8 张真实顶刊参考图（注明期刊、年份、论文与你要学的**具体版面语法**），并分别给出 Fig. 1（2 套构图骨架）与 Fig. 2（2 套宏微一体化骨架）的线框布局提案。**等我选定参考图和骨架方向后再进入编码绘图！**

### Phase 2：第 1–2 轮——信息架构锁定与“零幻觉”真值搭建
- **Round 1（信息骨架粗胚）**：用可编辑矢量方式（优先生成可直接在本地 Windows `python-pptx` / `SVG` 运行的模块化脚本）搭出 Fig. 1 与 Fig. 2 的首版骨架，**必须实际渲染成高分辨率 PNG 并自审**。
- **Round 2（代码/论文真值全面审计）**：逐字符对照 `main_tsmc.tex`、`models/hb_gat_pn.py`、`environment.py`、`ppo_agent.py` 检查每一个数学符号（如 $\tau^r$、$r_t^{\text{PPO}}(\theta)$、$\mathcal{F}_t^O$、`[M | O | W]`、5 种边关系、$q_i$ 给定而非预测）是否 100% 准确，剔除所有“无解释力装饰框”。

### Phase 3：第 3–10 轮——严格“出图 → 渲染 → 并排对照 → 精准挑刺 → 重绘”多轮精细打磨
请计划并执行 **5–10 轮真实闭环迭代**（每一轮都必须实际渲染图片后**以图审图**，并将候选图与旧版图/参考图并排比较，向我明确汇报本轮发现了哪些视觉瑕疵及如何修复）：
- **Round 3–4（布局紧致度与连线几何清理）**：消除所有大面积冗余空白；消灭任何“边框比文字大很多”的空盒子；理顺正交箭头路由，确保零交叉穿线、零贴边挤压；统一分支对齐与对称节奏（借鉴 DAN Fig. 3）。
- **Round 5–7（算法可解释性与“实景→图→计算”因果追踪）**：检查读者能否不看正文只看 Fig. 1 就读懂“五站脉动 + 冻结边界 $\tau^r$ + 延误传染”；检查读者能否不看正文只看 Fig. 2 就沿同一候选从 $G_t \to$ 局部 GATv2 注意力聚合 $\to O \to S \to W$（含 $\mathbf{-10^4}$ 屏蔽与两次选人去重）$\to$ 前向定时间 $\to$ PPO 梯度更新一路追踪到底；彻底清除任何残余的 AI 机械感。
- **Round 8–10（IEEE TSMC 双栏印刷终审级打磨）**：模拟按论文实际版心尺寸（`0.92\textwidth ≈ 6.59–7.16 in` 宽度，约占半页高）缩放到 100% 实际印刷视距审阅；放大 300% 查每个角标、上标、箭头端点与公式上下标对齐（确保类似 `LeakyReLU (0.1)` 上下杆完全光学居中）；统一 Fig. 1、Fig. 2 与附录 `B8` 全套视觉规范。

### Phase 4：最终交付格式规范
1. 输出放入独立新子目录（如 `output/figures_english_from_17/paper_final_v2/`），**严禁覆盖** `refined_v8/`、`paper_integrated_v1/`、`reference_study_v2/` 或未获批准擅改 `main_tsmc.tex`。
2. 交付可独立编辑的模块化源码（兼容 `python-pptx` / 高精度 SVG，确保图形与文字对象在 PPT/Illustrator 中可分别编辑而非整张死位图）、600 DPI 高清 PNG、无损矢量 PDF，以及可直接嵌入 `main_tsmc.tex` 的精炼英文 `\caption{}` 与对应 CC-BY-2.0 致谢建议。

请现在开始 **Phase 1**：先检查并确认已能读取仓库关键文件，并用中文简洁复述你对本次核心边界的理解，同时立即检索并向我呈报最契合的制造业/调度顶刊真实参考图集与初步构图提案。

---
