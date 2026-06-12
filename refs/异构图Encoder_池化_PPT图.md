# 异构图编码器 + 上下文感知池化（PPT 用 Mermaid 图）

## 方案 A：横向总览流水线（推荐 PPT 用）

```mermaid
flowchart LR
    subgraph Input["📥 输入: HeteroData"]
        direction TB
        N1["🔵 Task 节点<br/>x_T ∈ R^(NT×18)"]
        N2["🟠 Worker 节点<br/>x_W ∈ R^(NW×22)"]
        N3["🟢 Station 节点<br/>x_S ∈ R^(NS×15)"]
        E1["5 类异构图边<br/>precedes | assigned_to<br/>can_do | done_by | has_task"]
    end

    subgraph Embed["📐 Feature Embedder"]
        direction TB
        F1["MLP_T: 18→128<br/>LeakyReLU"]
        F2["MLP_W: 22→128<br/>LeakyReLU"]
        F3["MLP_S: 15→128<br/>LeakyReLU"]
    end

    subgraph GAT["🧠 HeteroGATEncoder ×5"]
        direction TB
        L1["Layer 1: GATv2 Conv<br/>5 边类型 × 4 头<br/>┼ Post-Norm + Residual"]
        L2["Layer 2: GATv2 Conv<br/>┼ Post-Norm + Residual"]
        L3["..."]
        L4["Layer 5: GATv2 Conv<br/>┼ Post-Norm + Residual"]
    end

    subgraph Pool["🎯 Attention Pooling"]
        direction TB
        P1["softmax(MLP_attn(·))<br/>加权求和"]
        P2["g = [g_S ‖ g_T ‖ g_W]<br/>∈ R^384"]
    end

    subgraph Out["📤 输出"]
        direction TB
        O1["全局上下文 g<br/>送给 Pointer 解码器"]
    end

    N1 --> F1
    N2 --> F2
    N3 --> F3
    F1 & F2 & F3 --> L1
    E1 -.-> L1
    L1 --> L2 --> L3 --> L4
    L4 --> P1 --> P2 --> O1
```

---

## 方案 B：单层 GATv2 消息传递展开（适合讲解图卷积细节）

```mermaid
flowchart TB
    subgraph before["上一层的输出"]
        h_prev["h_T^(l) , h_W^(l) , h_S^(l)"]
    end

    subgraph edges["5 类边上的 GATv2Conv"]
        direction LR
        E_pre["(task, precedes, task)<br/>拓扑约束传递<br/>前驱 Task → 后继 Task"]
        E_as["(task, assigned_to, station)<br/>负载报告<br/>活动 Task → Station"]
        E_ht["(station, has_task, task)<br/>站位竞争协商<br/>Station → 活动 Task"]
        E_cd["(worker, can_do, task)<br/>能力宣告<br/>Worker → Task"]
        E_db["(task, done_by, worker)<br/>执行关系<br/>Task → Worker"]
    end

    subgraph formula["GATv2 注意力机制"]
        F1["α_ij = softmax( a^⊤ · LeakyReLU( Θ_s·h_i + Θ_t·h_j ) )"]
        F2["h_i' = Σ_j α_ij · Θ_t·h_j （4头均值）"]
    end

    subgraph agg["跨边类型聚合"]
        A1["h_v' = Σ_{r∈R} GATv2Conv_r(h_v)   （sum 聚合）"]
    end

    subgraph after["Post-Norm + 残差"]
        R1["h_v^(l+1) = h_v^(l) + LeakyReLU( LayerNorm( h_v' ) )"]
    end

    before --> edges --> formula --> agg --> after
    after -->|"5 层循环 × 共享边结构"| before
```

---

## 方案 C：Attention Pooling 展开（适合讲解全局上下文生成原理）

```mermaid
flowchart TB
    subgraph input["编码器输出  h_T^(L), h_W^(L), h_S^(L) ∈ R^128"]
        direction LR
        hT["h_T,1 ... h_T,NT"]
        hW["h_W,1 ... h_W,NW"]
        hS["h_S,1 ... h_S,NS"]
    end

    subgraph attn_T["工序注意力"]
        direction TB
        AT1["MLP_attn: 128→32→1<br/>→ w_i = softmax(score_i)"]
        AT2["g_T = Σ_i w_i · h_T,i"]
    end

    subgraph attn_W["工人注意力"]
        direction TB
        AW1["MLP_attn: 128→32→1<br/>→ w_j = softmax(score_j)"]
        AW2["g_W = Σ_j w_j · h_W,j"]
    end

    subgraph attn_S["站位注意力"]
        direction TB
        AS1["MLP_attn: 128→32→1<br/>→ w_k = softmax(score_k)"]
        AS2["g_S = Σ_k w_k · h_S,k"]
    end

    subgraph concat["全局上下文拼接"]
        GC["g = [ g_S ‖ g_T ‖ g_W ]<br/>维度: 128+128+128 = 384"]
    end

    subgraph monitor["📊 瓶颈监控"]
        M1["Var(w_k^S) → 检测站位不平衡度"]
    end

    hT --> AT1 --> AT2
    hW --> AW1 --> AW2
    hS --> AS1 --> AS2
    AT2 & AW2 & AS2 --> GC
    AS1 --> M1
```

---

## 方案 D：紧凑单图版（适合一页 PPT 放完整 Encoder 全貌）

```mermaid
flowchart LR
    subgraph A["1️⃣ 特征嵌入"]
        A1["Task<br/>18→128"]
        A2["Worker<br/>22→128"]
        A3["Station<br/>15→128"]
    end

    subgraph B["2️⃣ 异构图编码 (×5)"]
        B1["GATv2<br/>5边·4头<br/>sum聚合"]
        B2["Post-Norm<br/>+<br/>LeakyReLU"]
        B3["残差<br/>h+h'"]
    end

    subgraph C["3️⃣ Attention Pooling"]
        C1["加权<br/>softmax<br/>池化"]
        C2["拼接<br/>[S‖T‖W]"]
    end

    subgraph D["4️⃣ 输出"]
        D1["g ∈ R^384<br/>→ 3个Pointer<br/>解码器"]
    end

    A --> B1 --> B2 --> B3
    B3 -->|循环5次| B1
    B3 --> C1 --> C2 --> D1
```

---

## 推荐组合

| PPT 页面 | 使用方案 | 要点 |
|:---:|:---:|------|
| **第 1 页**（总览） | 方案 A 或 D | 展示从异构输入到全局上下文的完整数据流 |
| **第 2 页**（细节） | 方案 B | 展开 GATv2 单层的消息传递机制 + 公式 |
| **第 3 页**（池化） | 方案 C | 重点讲解 Attention Pooling 如何加权融合三类节点 |

> **渲染提示**: 复制对应方案代码到 [Mermaid Live Editor](https://mermaid.live) 可实时预览并导出 SVG/PNG。GitHub、Typora、VS Code（安装 Markdown Preview Mermaid 插件）均原生支持渲染。
