# 异构图注意力编码器与全局上下文池化 — 逐层公式详解

> **前置阅读**：[异构图构建详解.md](./异构图构建详解.md)（第 1–3 节）  
> **对应源码**：[models/hb_gat_pn.py](./models/hb_gat_pn.py)  
> **目标读者**：已了解异构图中三种节点和五种边关系的读者，本文聚焦**数学推导**与**符号物理含义**。

---

## 4. 异构图注意力编码器（HeteroGATEncoder）

**物理直觉**：在一个调度状态图中，每个节点（工序 / 工人 / 站位）需要通过与邻居交换信息来感知全局态势。例如：一道工序需要知道"我的紧前工序都完成了吗？""有哪个站位有空槽位？""哪些工人能帮我？"。HeteroGATEncoder 就是通过多轮消息传递来实现这种**拓扑感知**的。

### 4.1 符号速查表

| 符号 | 类型 | 含义 | 示例 |
|------|------|------|------|
| $\mathbf{h}_i$ | $\mathbb{R}^{H}$ | 节点 $i$ 的嵌入向量（$H=128$ 为隐藏维度） | $\mathbf{h}_{t_3}$ 表示第 3 道工序的 128 维特征 |
| $\mathbf{h}_i'$ | $\mathbb{R}^{H}$ | 节点 $i$ 经 1 层 GAT 更新后的嵌入 | — |
| $\mathbf{\Theta}_s$ | $\mathbb{R}^{H \times H}$ | **源节点**的线性变换矩阵，形状 $128 \times 128$ | 下标 $s$ = source |
| $\mathbf{\Theta}_t$ | $\mathbb{R}^{H \times H}$ | **目标节点**的线性变换矩阵，形状 $128 \times 128$ | 下标 $t$ = target |
| $\mathbf{a}$ | $\mathbb{R}^{2H}$ | 注意力评分向量，形状 $256$ | 将拼接后的 $[\mathbf{\Theta}_s\mathbf{h}_i \| \mathbf{\Theta}_t\mathbf{h}_j]$ 映射到一个标量分数 |
| $\mathcal{N}(i)$ | 集合 | 节点 $i$ 在当前边类型下的入邻居集合 | 对边 `(task, precedes, task)` 而言，$\mathcal{N}(t_j)$ = $\{t_i: t_i \text{ precedes } t_j\}$ |
| $\alpha_{ij}$ | 标量 | 邻居 $j$ 对中心节点 $i$ 的注意力权重，$\alpha_{ij} \in [0,1]$ | $\alpha_{ij} = 0.3$ 表示消息"30% 重要性" |
| $K$ | 整数 | 注意力头数 | $K = 4$，4 个头并行计算后取均值 |
| $\mathcal{R}_{in}(v)$ | 集合 | 以节点 $v$ 为**终点**的所有边类型集合 | 对 Station 节点：包含 `assigned_to` 和 `has_task` 两种 |
| $\|$ | 运算符 | 向量拼接 (concatenation) | $[\mathbf{v}_1 \| \mathbf{v}_2] \in \mathbb{R}^{2H}$ |

### 4.2 GATv2 单头注意力 — 逐步推导

GATv2 与原始 GAT 的核心区别在于：**先分别对源节点和目标节点做线性变换，再相加**（而非先拼接再变换）。

#### Step 1: 线性投影

对一条有向边 $j \to i$（$j$ 是源，$i$ 是目标）：

$$\mathbf{q} = \mathbf{\Theta}_t \mathbf{h}_i \qquad (\text{目标节点投影，作为 Query})$$
$$\mathbf{k} = \mathbf{\Theta}_s \mathbf{h}_j \qquad (\text{源节点投影，作为 Key})$$

> **注释**: $\mathbf{\Theta}_t \in \mathbb{R}^{128 \times 128}$，$\mathbf{h}_i \in \mathbb{R}^{128}$，因此 $\mathbf{q} \in \mathbb{R}^{128}$。**类似 Transformer 的 Query-Key 机制**：谁接收消息（目标节点 $i$）投影为 Q，谁发送消息（源节点 $j$）投影为 K。

#### Step 2: 计算原始注意力分数

$$e(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{a}^\top \cdot \text{LeakyReLU}\big(\mathbf{q} + \mathbf{k}\big)$$

其中：
- $\mathbf{q} + \mathbf{k} \in \mathbb{R}^{128}$：将 Query 和 Key 的信息相加（不是拼接！这是 GATv2 的改进）
- $\text{LeakyReLU}(x) = \max(0.1x, x)$：防止 ReLU 死亡，负半轴保留 10% 斜率
- $\mathbf{a}^\top \in \mathbb{R}^{1 \times 256}$：**此处文档有误**——由于 GATv2 使用的是 $\mathbf{q}+\mathbf{k} \in \mathbb{R}^{128}$ 而非拼接后的 $\mathbb{R}^{256}$，PyTorch 实际实现中 $\mathbf{a} \in \mathbb{R}^{H} = \mathbb{R}^{128}$（单列），$\mathbf{a}^\top \in \mathbb{R}^{1 \times 128}$，通过点积映射到一个标量分数 $e(\cdot) \in \mathbb{R}$。

> **重要纠正**：`异构图构建详解.md` 中写 $\mathbf{a} \in \mathbb{R}^{2H}$ 对应的是**原始 GAT** 的拼接形式。本项目代码中使用的是 `GATv2Conv(concat=False)`，其注意力权重向量维度为 $\mathbf{a} \in \mathbb{R}^{H}$。两种形式数学等价性不影响正确性，但维度数值有差异。

#### Step 3: Softmax 归一化

$$\alpha_{ij} = \frac{\exp\big(e(\mathbf{h}_i, \mathbf{h}_j)\big)}{\sum_{k \in \mathcal{N}(i)} \exp\big(e(\mathbf{h}_i, \mathbf{h}_k)\big)}$$

> **注释**: 对于节点 $i$，只在其**入邻居集合** $\mathcal{N}(i)$ 上做 softmax。$\mathcal{N}(i)$ 的定义取决于边类型——如果当前处理的是 `(task, precedes, task)` 边，则 $\mathcal{N}(t_j)$ 是所有 $t_j$ 的紧前工序。

#### Step 4: 多头均值聚合

$$\mathbf{h}_i' = \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \cdot \big(\mathbf{\Theta}_t^{(k)} \mathbf{h}_j\big)$$

> **注释**: 每个注意力头 $k$ 有**独立的参数** $\mathbf{\Theta}_s^{(k)}, \mathbf{\Theta}_t^{(k)}, \mathbf{a}^{(k)}$。$K=4$ 个头并行计算，最终通过**均值**（`concat=False`）聚合。这里的 $\mathbf{\Theta}_t^{(k)}$ 再次对源节点特征做变换——GATv2 的机制是源节点特征经过两次 $\mathbf{\Theta}_t$：一次在计算 $\alpha_{ij}$ 时用于构建 Key，另一次在加权聚合时用于将源特征投影到新空间。

### 4.3 异构图消息聚合 — sum 策略

同一节点 $v$ 可能同时收到来自**多种边类型**的消息。例如：工序 $t_2$ 可能同时收到：
- 边 `(task, precedes, task)`：来自紧前工序 $t_1$ 的拓扑信息
- 边 `(station, has_task, task)`：来自它所在站位 $s_1$ 的资源竞争信息
- 边 `(worker, can_do, task)`：来自所有能执行它的工人的能力宣告

**层间跨类型 sum 聚合**：

$$\mathbf{h}_v^{(l+1)} = \sum_{r \in \mathcal{R}_{in}(v)} \text{GATv2Conv}_r^{(l)}\big(\mathbf{h}^{(l)}, \mathcal{E}_r\big)_v$$

> **注释**: 上标 $(l)$ 表示第 $l$ 层（共 $L=5$ 层）。$\text{GATv2Conv}_r^{(l)}$ 是第 $l$ 层中专门处理边类型 $r$ 的 GATv2 卷积。**sum 表示**：节点 $v$ 的新嵌入 = 来自所有入边类型消息的逐元素求和。

**示例**：在第 2 层，站位 $s_1$ 同时收到来自 `assigned_to` 和 `has_task` 两种边的消息，它的新嵌入为：

$$\mathbf{h}_{s_1}^{(2)} = \mathbf{h}_{s_1}^{(1)} + \text{GATv2}_{\text{assigned}}^{(1)}(\mathbf{h}^{(1)})_{s_1} + \text{GATv2}_{\text{has}}^{(1)}(\mathbf{h}^{(1)})_{s_1}$$

### 4.4 残差连接 + Post-Norm — 完整的层变换

PyTorch 的 `HeteroConv` 有一个重要特性：**只返回作为边终点的节点更新**。未出现在任何边终点中的节点不会出现在 `x_dict_out` 中——我们通过复制原始字典并只覆盖更新的 key 来处理这个问题。

每层的完整变换：

$$\boxed{\mathbf{h}_v^{(l+1)} = \begin{cases}
\mathbf{h}_v^{(l)} + \text{LeakyReLU}\big(\text{LayerNorm}(\mathbf{h}_v')\big) & \text{若 } v \text{ 是某条边的终点} \\[6pt]
\mathbf{h}_v^{(l)} & \text{否则（恒等映射）}
\end{cases}}$$

其中 $\mathbf{h}_v'$ 是第 4.3 节的跨类型 sum 聚合结果。

> **注释**: 残差连接 $\mathbf{h}_v^{(l)} + \cdots$ 保证梯度可以直通浅层，防止 5 层 GAT 出现梯度消失。LayerNorm 稳定训练，Post-Norm 表示归一化在**激活函数之前**（而非之后）。

### 4.5 五条消息传递路径的物理意义

| 边类型 | 方向 | 物理含义 | 数学解释 |
|:---|:---|------|------|
| `(task, precedes, task)` | 前驱 → 后继 | **拓扑约束传递** | 后继工序的嵌入融合所有紧前工序的信息，间接感知"我前面还剩多少工作量" |
| `(task, assigned_to, station)` | 活动 Task → Station | **负载报告** | 站位通过加权聚合所有正在执行的工序的嵌入，感知"我的总负载是多少" |
| `(station, has_task, task)` | Station → Task | **站位协商** | 正在执行的工序从站位获得"站位的槽位是否紧张、其他并行工序的状态"等竞争信息 |
| `(worker, can_do, task)` | Worker → Task | **能力宣告** | 所有 Ready 工序从拥有所需技能的全部工人那里获得"有熟练工可用"或"缺乏合格工人"的信号 |
| `(task, done_by, worker)` | Task → Worker | **执行反馈** | 正在忙碌的工人从所执行的工序那里获得"我的技能正在被使用、任务还剩多久完成"的信息 |

---

## 5. 全局上下文池化（Attention Pooling）

**物理直觉**：经过 5 层 GAT 编码后，每个节点拥有局部邻域的丰富信息。但 Pointer Network 的 3 个解码头（Task Pointer / Station Selector / Worker Pointer）需要从**全局视角**做决策——"整个车间此刻哪个站位最空闲？""哪个工人最应该被选中？"。Attention Pooling 的作用就是从一堆节点嵌入中提取一个**代表全局态势的汇总向量** $\mathbf{g}$。

### 5.1 符号速查表

| 符号 | 类型 | 含义 | 示例 |
|------|------|------|------|
| $\mathbf{h}_{S,k}$ | $\mathbb{R}^{128}$ | 第 $k$ 个站位经 5 层 GAT 编码后的嵌入 | 5 站位共 5 个 128 维向量 |
| $w_k^S$ | 标量 $\in [0,1]$ | 第 $k$ 个站位的注意力权重 | $w_2^S = 0.45$ 表示站位 2 获得 45% 关注度 |
| $\mathbf{g}_S$ | $\mathbb{R}^{128}$ | 站位侧的全局上下文向量 | 所有 5 个站位嵌入的**加权和** |
| $\mathbf{g}$ | $\mathbb{R}^{384}$ | 最终全局上下文向量 | $\mathbf{g} = [\mathbf{g}_S \| \mathbf{g}_T \| \mathbf{g}_W]$，$128 \times 3 = 384$ |
| $\|$ | 运算符 | 拼接 | 将 3 个 128 维向量串接成 1 个 384 维向量 |
| $\text{MLP}_{attn}^S$ | 网络 | 站位的注意力评分 MLP | 结构：`Linear(128,32) → LayerNorm → LeakyReLU → Linear(32,1)` |
| $\text{softmax}(\cdot, \text{batch})$ | 函数 | 按批次分组的 softmax | PyG 的 `softmax(src, index)`，保证每个子图内权重独立归一化 |
| $B$ | 整数 | 批次大小（并行环境数） | 训练时 $B$ = batch_size，单环境推理时 $B=1$ |

### 5.2 Attention Pooling 的实现 — 逐步推导

本项目中的 Attention Pooling 实现为两层 MLP（而非单层线性投影），通过瓶颈压缩（128→32→1）引入非线性，提升权重分配的判别能力。

#### Step 1: 评分网络（以站位为例）

$$\mathbf{z}_k^S = \text{LeakyReLU}\big(\text{LayerNorm}(\mathbf{W}_1 \mathbf{h}_{S,k} + \mathbf{b}_1)\big) \cdot \mathbf{w}_2$$

具体步骤：
1. $\mathbf{h}_{S,k} \in \mathbb{R}^{128}$ 输入 MLP
2. $\mathbf{W}_1 \in \mathbb{R}^{32 \times 128}$：投影到 32 维瓶颈层，减少过拟合
3. LayerNorm + LeakyReLU：稳定激活分布
4. $\mathbf{w}_2 \in \mathbb{R}^{32}$：线性映射到标量分数 $z_k^S \in \mathbb{R}$

> **代码对应**：`self.station_attn = Sequential(Linear(128,32), LayerNorm(32), LeakyReLU, Linear(32,1))`

#### Step 2: Softmax 归一化（按子图分组）

$$w_k^S = \frac{\exp(z_k^S)}{\sum_{j \in \mathcal{S}} \exp(z_j^S)}$$

> **注释**: $\mathcal{S}$ 是当前子图（或 Batch 中同一环境）的所有站位索引。在批量训练中，PyG 的 `softmax(src, batch_idx)` 保证每个环境独立计算 softmax——环境 A 的站位权重和为 1，环境 B 的也为 1。

#### Step 3: 加权求和得到品类上下文

$$\mathbf{g}_S = \sum_{k \in \mathcal{S}} w_k^S \cdot \mathbf{h}_{S,k} \quad \in \mathbb{R}^{128}$$

> **物理含义**: $\mathbf{g}_S$ 是所有站位嵌入的**注意力加权平均**。如果当前某个站位负载极高（其 $\mathbf{h}_{S,k}$ 中负载特征被放大），$w_k^S$ 自然偏大，$\mathbf{g}_S$ 倾向于表征"最忙碌的站位"。

#### Step 4: 工序侧和工人侧同理

$$\mathbf{g}_T = \sum_{i \in \mathcal{T}} w_i^T \cdot \mathbf{h}_{T,i} \quad \in \mathbb{R}^{128}$$

$$\mathbf{g}_W = \sum_{j \in \mathcal{W}} w_j^W \cdot \mathbf{h}_{W,j} \quad \in \mathbb{R}^{128}$$

> **注释**: 代码中工序和工人共用同一个评分网络 `self.task_worker_attn`（因为两者的物理含义都是"谁更需要被关注"），站位单独使用 `self.station_attn`。

#### Step 5: 全局上下文拼接

$$\boxed{\mathbf{g} = [\mathbf{g}_S \;\|\; \mathbf{g}_T \;\|\; \mathbf{g}_W] \in \mathbb{R}^{384}}$$

拼接后的 $\mathbf{g}$ 被送入 3 个 Pointer Head：

| Head | 输入 | 作用 |
|:---|------|------|
| TaskPointer | $\mathbf{g}$ → 投影为 Query | 让 Task 解码器知道"全局态势"来选择下一道工序 |
| StationSelector | $\mathbf{g}$（隐式，通过 $\mathbf{g}_S$ 的残差） | 站位选择时参考全局站位竞争格局 |
| WorkerPointer | $\mathbf{g}$（隐式通过 $\mathbf{g}_W$） | 工人选择时参考工人空闲/疲劳全局分布 |

### 5.3 Attention Pooling 的降级路径

代码提供了 3 种降级（Ablation）选项：

| 配置 | 全局上下文 | 维度 | 何时启用 |
|------|-----------|:---:|------|
| `use_attention_critic=True` | $\mathbf{g} = [\mathbf{g}_S \| \mathbf{g}_T \| \mathbf{g}_W]$（注意力加权） | 384 | 默认 |
| `use_attention_critic=False` | $\mathbf{g} = [\text{mean}_S, \text{mean}_T, \text{mean}_W, \text{max}_S, \text{max}_T, \text{max}_W]$ | 768 | 消融实验 |
| 单环境（无 batch） | 同上逻辑，不分组 softmax | 同上 | 推理/评估 |

### 5.4 监控指标：站位注意力方差

代码中维护了 `self.last_s_var`：

$$\text{last\_s\_var} = \frac{1}{B} \sum_{b=1}^{B} \text{Var}\big(\{w_k^{S,b}\}_{k=1}^{N_S}\big)$$

> **物理含义**: 如果某站位持续获得极高注意力（方差大），说明该站位是当前调度状态的**瓶颈**。这个指标被写入 TensorBoard 方便观察模型是否学会了识别瓶颈。

---

## 完整前向传播的数据流总结

```
输入: HeteroData (task.x [N_T,18], worker.x [N_W,22], station.x [N_S,15], 5种边)

│
▼
┌─────────────────────────────────────────────────────────────┐
│  第 0 层: FeatureEmbedder (参见 异构图构建详解.md §3)         │
│  ─────────────────────────────────────────────────────────  │
│  x_T [N_T, 18]  →  W_T [18,128]  →  h_T^(0) [N_T, 128]     │
│  x_W [N_W, 22]  →  W_W [22,128]  →  h_W^(0) [N_W, 128]     │
│  x_S [N_S, 15]  →  W_S [15,128]  →  h_S^(0) [N_S, 128]     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第 1-5 层: HeteroGATEncoder (每层 1 个 HeteroConv)          │
│  ─────────────────────────────────────────────────────────  │
│  每层内:                                                     │
│    5 种边类型 → 独立的 GATv2Conv (4 头均值)                   │
│    → sum 跨类型聚合 → LayerNorm → LeakyReLU → 残差 + h^(l)   │
│  输出: h_T^(5) [N_T,128], h_W^(5) [N_W,128], h_S^(5) [N_S,128]│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Attention Pooling (本节 §5)                                 │
│  ─────────────────────────────────────────────────────────  │
│  station_attn( h_S^(5) ) → softmax → g_S [128]              │
│  task_worker_attn( h_T^(5) ) → softmax → g_T [128]          │
│  task_worker_attn( h_W^(5) ) → softmax → g_W [128]          │
│  g = [g_S ‖ g_T ‖ g_W]  ∈  R^384                            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Pointer Heads (自回归解码)                                   │
│  ─────────────────────────────────────────────────────────  │
│  ① TaskPointer(h_T^(5), g) → 选择下一道工序                   │
│  ② StationSelector(h_T^(5)[sel], h_S^(5)) → 选择站位          │
│  ③ WorkerPointer(h_T^(5)[sel], h_W^(5)) → 自回归选择工人团队  │
└─────────────────────────────────────────────────────────────┘
```

---

> **文档版本**: v1.0，对应代码 commit `eec459e`  
> **待办**: 本文假设读者已理解[异构图构建详解.md](./异构图构建详解.md)中 3 种节点和 5 种边关系的定义
