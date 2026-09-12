# Joint Scheduling and Workforce Allocation for Aircraft Pulsating Assembly Lines via Heterogeneous Graph Reinforcement Learning

**Abstract**: Operation planning in an aircraft pulsating assembly line (APAL) involves tightly coupled constraints, including precedence relationships, station spatial concurrency capacities, multi-skilled workers, and collaborative teamwork. Operation-station balancing and worker allocation interact intricately, while the combinatorial selection of concrete worker teams drastically expands the joint action space. To address these challenges, this paper proposes HGP-PPO, a reinforcement learning scheduling framework integrating heterogeneous graph neural networks with an autoregressive pointer decoder. First, the construction of a cyclic operation template is formulated as a sequential Markov decision process, utilizing four types of nodes—operations, stations, workers, and skills—along with their relations to capture complex topological structures and dynamic resource states. Second, a hierarchical operation-station-worker conditional autoregressive decoding mechanism is devised to sequentially select the operation, the physical station, and the concrete worker team of required size, strictly enforcing physical feasibility through dynamic action masking and a forward temporal allocation engine. Worker-station affiliations are endogenously established during the scheduling process to naturally harmonize personnel allocation with station workload demands. Furthermore, by augmenting state representations with baseline schedule and disturbance features followed by continued training, the framework is adapted to plan-level stability rescheduling under material delivery delays. Extensive experiments conducted on four real factory-derived instances ranging from 283 to 3,182 physical operations demonstrate that a single locked policy generates fully valid schedules without fine-tuning, achieving a 2.22% reduction in average cycle span compared to the adapted L2D-PPO-APAL baseline and exhibiting strong extrapolation to instances nearly four times the training scale. Across 36 paired material delay scenarios, the rescheduling model maintains a 100% feasibility rate, reducing the average composite objective by 7.00% and 3.01% against L2D-PPO-APAL and StabilityAwareRepair, respectively. Moreover, the model delivers repaired solutions within two minutes on the largest scale, showing orders-of-magnitude computational scalability over metaheuristic search methods. These results demonstrate that the proposed framework effectively coordinates multi-resource allocation in APALs, achieving a superior trade-off among cycle completion efficiency, plan stability, and real-time responsiveness under the investigated problem scope.

**Keywords**: Aircraft pulsating assembly line; Joint scheduling; Workforce allocation; Heterogeneous graph neural network; Proximal policy optimization; Stability rescheduling

---

## I. Introduction

Aircraft final assembly is a critical final stage of aircraft manufacturing, encompassing operations such as major component mating, piping installation, engine installation, avionics integration, and functional testing. Unlike conventional shop scheduling or automated mass-production lines, aircraft final assembly deals with large, highly customized products characterized by extensive process chains, dense precedence dependencies, long operation durations, restricted working space, and collaboration among multiple trades [11, 2, 25]. To improve production flow and establish a regular production rhythm, aircraft manufacturers widely adopt aircraft pulsating assembly lines (APALs). An APAL partitions the final assembly shop floor into several dedicated physical stations, through which aircraft in progress advance synchronously in a pulsating cadence. Within each pulsation cycle, each aircraft remains stationary at its assigned station while a designated set of operations is executed. Upon cycle completion, all aircraft along the line advance synchronously to their respective downstream stations until final roll-out [16].

This synchronous advancement mechanism allows local operation delays to affect the timing of line-wide station transfers. Line advancement is strictly dictated by the bottleneck station with the latest completion time. Delays caused by uneven task distribution, shortages of qualified workers, or station concurrency limits can postpone synchronous advancement when they extend the latest station completion time, requiring aircraft at stations that have already finished to wait. Consequently, jointly coordinating operation-to-station assignment, within-station sequencing, and resource allocation is essential to efficient and stable APAL operation [5, 6, 2].

In practical shop-floor settings, APAL scheduling involves three interdependent decisions for each operation: its execution sequence, its assigned physical station, and a specific collaborative worker team that satisfies the required skill qualifications and team size. Pre-assigning workers to dedicated stations before task distribution is determined can limit the ability to match workforce capabilities with evolving station workloads. The proposed method therefore determines worker–station assignments endogenously during schedule construction as workers are first selected for station-specific operations. This mechanism gives team selections lasting downstream consequences: each assignment not only determines an operation's execution duration and immediate resource consumption, but also binds previously unassigned workers to a specific station, thereby restricting the labor pool available to subsequent operations. Consequently, the execution efficiency of the current operation and the anticipated skill demands of downstream stations must be coordinated within the same unified decision process.

This joint decision problem presents several challenges:
1. Operations, stations, workers, and skills exhibit heterogeneous multi-relational topologies, operating within an environment characterized by rapid state transitions and dynamic resource calendars;
2. Selecting an optimal team from a multiskilled labor pool results in an exponential combinatorial action space of size $\binom{n_i}{q_i}$ for each operation;
3. Each intermediate decision changes the feasible choices available to subsequent decisions;
4. When shop-floor disturbances such as material arrival delays occur, rescheduling must preserve frozen schedule entries and existing worker–station bindings while repairing subsequent cyclic templates to restore feasibility and balance completion span against deviations from the baseline schedule [24, 12].

Fig. 1 summarizes the physical setting, the entities involved in joint decisions, and the plan repair process following material delays.

![Illustration of cyclic operation scheduling and stability-aware rescheduling for an aircraft pulsating assembly line](figures/fig1.png)

**Fig. 1. Illustration of cyclic operation scheduling and stability-aware rescheduling for an aircraft pulsating assembly line.** Aircraft advance through five physical stations in a pulsating rhythm. The cyclic operation template jointly determines the precedence-related execution relationships among operations, their assigned stations, and specific worker teams satisfying skill and headcount requirements. Following a material delay, the system locks committed entries at the schedule freeze boundary $\tau^r$, inherits worker bindings, and performs stability-aware rescheduling of the remaining operations.

To address the coupling between operation scheduling and workforce allocation in aircraft pulsating assembly lines, this paper proposes the HGP-PPO joint scheduling framework. The main contributions are as follows:

1. **Joint scheduling model with endogenous workforce allocation:** An APAL cyclic operation template model is formulated to incorporate precedence constraints, station concurrency limits, and collaborative operations performed by multi-skilled workers. The model integrates operation-to-station assignment, selection of specific worker teams, and worker–station assignment into a unified sequential construction process. Persistent station bindings established upon each worker’s first assignment explicitly capture how current team choices affect subsequent worker availability, enabling the coordinated determination of operation schedules and station workforce allocation.

2. **Heterogeneous graph representation of process dependencies and skill supply–demand relationships:** A heterogeneous graph comprising operation, station, worker, and skill nodes jointly represents process dependencies, skill supply–demand relationships, and dynamic assignments. Relation-specific attention encoding provides a shared state representation for operation-to-station assignment and team formation, enabling the policy to make joint decisions informed by both task structures and workforce states.

3. **Autoregressive joint decoding incorporating team states and skill scarcity:** A conditional operation–station–worker pointer mechanism decomposes the combinatorial selection of fixed-size teams into sequential member selections. Candidate workers are scored conditionally using partial team representations, tentative completion-time features, and the marginal scarcity of their additional skills. This mechanism avoids explicit enumeration of complete teams and uses dynamic masking and forward time allocation to enforce the scheduling constraints.

The framework is evaluated for both initial scheduling and rescheduling under material readiness delays. For initial scheduling, a single policy is held fixed and applied without fine-tuning to four test instances containing 283 to 3,182 physical operations. Comparisons with workforce pre-assignment, alternative graph representations, and decision architectures assess the role of the overall design. The framework is then adapted to stability-oriented cyclic template rescheduling by incorporating baseline schedule and disturbance features and continuing training, while preserving frozen operations and existing worker–station bindings. Evaluation across 36 paired disturbance scenarios considers schedule feasibility, completion efficiency, deviations from the baseline schedule, and online solution time. Together, these experiments assess cross-scale applicability within the investigated APAL data family and the framework’s applicability to both initial template construction and repair under material readiness delays.

The remainder of this paper is organized as follows. Section II reviews related work. Section III formulates the APAL joint scheduling and rescheduling models. Section IV presents the HGP-PPO architecture and decision mechanisms. Section V reports experiments across scales and ablation comparisons. Section VI concludes the paper and discusses its limitations.

## II. Related Work

### 2.1 Aircraft Assembly Scheduling and Workforce Allocation Under Multiple Resource Constraints

Assembly line balancing and scheduling, including ALBP, SALBP, and GALBP, provide the classical operations research foundations for operation-to-station assignment and workload balancing [5, 6, 4]. Recent studies have incorporated resource and workforce constraints to address complex aircraft final assembly settings, such as major component mating and intensive assembly operations. Bao et al. [2, 3] jointly optimized operation-to-station balancing and multiple resource allocations, primarily using staged heuristic search. P. Fang et al. [10] and D. Wang et al. [25] investigated flexible workforce allocation involving movement across stations in multistage assembly. Liu et al. [15] and Wen et al. [27] examined multiskilled workers and skilled-operator grades, respectively, at the level of dispatching within stations. These studies demonstrate the substantial influence of multiple resource constraints on assembly cycle duration. However, most decisions are made after operations have been assigned to stations, or concern only the number of workers moved and aggregate resource allocation. In contrast, this study addresses a global setting in which operation stations are not assigned in advance and workers are initially unbound, jointly selecting operations, physical stations, and specific worker teams of the required size within one decision process.

### 2.2 Graph Reinforcement Learning and Structured Actions in Production Scheduling

Graph neural networks (GNNs) learn structural representations of combinatorial optimization problems through shared message-passing functions [8]. In job-shop scheduling problems (JSSPs) and flexible job-shop scheduling problems (FJSPs), graph-based reinforcement learning methods encode operation dependencies and machine constraints to learn dispatching policies [29, 18, 26]. In aircraft assembly, our group's prior work [1] provided initial evidence of the effectiveness of graph-enhanced policies for APAL balancing. Wen et al. [28] combined a graph isomorphism network (GIN) with PPO to solve assembly scheduling problems involving skill grades and spatial capacity. However, the action spaces of mainstream reinforcement learning scheduling methods are largely limited to selecting the next operation or an operation–single-machine pair. To address the combinatorial selection of collaborative teams and dynamic worker–station assignments, the proposed framework represents process dependencies and skill supply–demand relationships through a heterogeneous graph and constructs teams using an operation–station–worker autoregressive decoder. Partial team states and skill-scarcity features inform successive member selections, while dynamic masks account for worker qualifications and station bindings.

### 2.3 Assembly Line Rescheduling Under Disturbances and Plan Stability

In dynamic manufacturing environments, predictive–reactive scheduling provides a standard paradigm in which a baseline schedule is prepared in advance and subsequently revised in response to disturbances [24, 23]. For supply chain uncertainty in aircraft moving assembly, Zhu et al. [30] adjusted operations that had not yet started in response to material delays. Hu et al. [12] proposed a two-stage inverse scheduling and resource reallocation strategy under processing-time fluctuations. Chen et al. [9] further used multiagent reinforcement learning to switch dispatching rules dynamically under material delays. In APALs, rescheduling must account for synchronous line advancement. This study focuses on plan-level cyclic template rebalancing under material readiness delays, preserving frozen operations and existing worker–station bindings. Baseline schedule and disturbance features are incorporated into the joint decision framework, which is further trained to balance completion efficiency against deviations in operation start times, station assignments, and team composition.

## III. APAL Joint Scheduling and Rescheduling Problem Formulation

### 3.1 APAL System Description, Core Assumptions, and Notation

Consider an APAL consisting of five fixed physical stations. The original assembly process topology is represented by a directed acyclic graph $G=(\mathcal{O},\mathcal{E})$, where $\mathcal{O}=\mathcal{O}^{+}\cup\mathcal{O}^{0}$. Here, $\mathcal{O}^{+}$ denotes physical operations with positive processing times, and $\mathcal{O}^{0}$ contains zero-duration nodes that retain only the process hierarchy. Zero-duration nodes consume no resources. Their precedence relationships are propagated through to equivalent precedence edges $\mathcal{E}^{+}$ between physical operations, forming the physical operation graph $G^+=(\mathcal{O}^+,\mathcal{E}^+)$. The immediate physical predecessors of operation $i$ are denoted by $\mathcal{P}(i)$. The system comprises a set of physical stations $\mathcal{M}=\{1,\dots,5\}$, a pool of multiskilled workers $\mathcal{W}$, and a set of skill types $\mathcal{K}$, with $|\mathcal{K}|=5$. Each operation $i\in\mathcal{O}^{+}$ has a baseline standard duration $p_i^{\mathrm{std}}>0$, a single required skill type $k_i\in\mathcal{K}$, and a fixed team size $q_i\in\mathbb{N}^{+}$. Each worker $w\in\mathcal{W}$ has a skill qualification vector $H_{wk}\in\{0,1\}$ and a baseline efficiency coefficient $\eta_w>0$. Except for a small number of initial positioning operations, operation stations are not specified in advance.

The core characteristics and operating assumptions are summarized as follows:

1. **Nonpreemptive operations**: Once an operation starts, it cannot be interrupted, and its station and executing team cannot be changed.
2. **Worker binding upon first selection**: All workers are globally unbound at the start of an instance. When worker $w$ is first assigned to an operation at station $m$, the worker is endogenously bound to that station and cannot serve other stations during that scheduling instance or any subsequent rescheduling triggered from it. Unbound workers remain available to any station satisfying the relevant conditions.
3. **Deterministic mapping from discrete actions to forward time allocation**: Scheduling decisions produce only discrete joint resource actions $(i,m_i,W_i)$, where $|W_i|=q_i$. Based on predecessor completion, worker availability, material readiness, and station concurrency slot capacity $c_m$, the Forward Allocation Engine deterministically maps each action to the earliest feasible start time $B_i$ and completion time $C_i$ on the template timeline.

**Table 2. Main Notation Used in Section III**

| Symbol | Physical meaning |
|---|---|
| $G^+=(\mathcal{O}^+,\mathcal{E}^{+})$ | Physical operation graph obtained by propagating precedence relationships through zero-duration nodes |
| $\mathcal{M}, \mathcal{W}, \mathcal{K}$ | Physical stations ($\lvert\mathcal{M}\rvert=5$), worker pool, and skill types ($\lvert\mathcal{K}\rvert=5$) |
| $p_i^{\mathrm{std}}, q_i, k_i$ | Standard duration, fixed team size, and required skill type of operation $i$ |
| $H_{wk}, \eta_w$ | Binary skill indicator and baseline efficiency coefficient of worker $w$ |
| $c_m$ | Maximum number of concurrent operation slots at station $m$ |
| $x_{im}, y_{iw}$ | Binary operation–station allocation and operation–worker assignment variables |
| $m_i, W_i$ | Physical station and specific collaborative team assigned to operation $i$, with $\lvert W_i\rvert=q_i$ |
| $B_i, C_i$ | Start and completion times of operation $i$ on the cyclic operation template timeline |
| $\ell_w$ | Station binding of worker $w$; $\ell_w=\varnothing$ indicates a globally unbound worker |
| $\pi^0, T_{\mathrm{ref}}^0$ | Initial baseline cyclic operation template and its actual reference cycle span, $T_{\mathrm{ref}}^0 := H^0 = C_{\max}(\pi^0)$ |
| $\tau^r, r_i$ | Rescheduling freeze time and updated material release time of operation $i$ |
| $\mathcal{O}^{\mathrm{frz}}, \mathcal{O}^{\mathrm{mov}}$ | Frozen operations ($B_i^0\le\tau^r$) and reschedulable operations ($B_i^0>\tau^r$) |
| $H, C_{\max}, S$ | Cyclic operation template span, template makespan, and composite rescheduling objective |

### 3.2 Initial APAL Cyclic Operation Scheduling Model

#### 3.2.1 Actual Processing Duration and Optimization Objective

The actual processing duration of an operation executed by a specific collaborative team depends on individual worker efficiencies and team collaboration effects. For a feasible team $W_i$ assigned to operation $i$ with $|W_i|=q_i$, the actual processing duration is defined as

$$
D_i(W_i) = \frac{p_i^{\mathrm{std}} q_i}{\rho^{q_i-1} \sum_{w\in W_i} \eta_w}, \qquad \rho = 0.95. \tag{1}
$$

Here, $\rho$ is the team collaboration discount factor. Initial scheduling constructs a steady-state cyclic operation template $\pi$ for repetitive production of a single aircraft model. Its formal objective is to minimize the cyclic operation template span, or cycle span, $H(\pi):=C_{\max}(\pi)$:

$$
\min_{\pi}\; C_{\max}(\pi) = \max_{m\in\mathcal{M}} C_m^{\mathrm{end}}(\pi) = \max_{i\in\mathcal{O}^{+}} C_i. \tag{2}
$$

The latest completion time at station $m$ is defined as $C_m^{\mathrm{end}}(\pi)=\max_{\{i\in\mathcal{O}^{+}\mid m_i=m\}}C_i$, or zero if no operations are assigned to that station. Once the baseline schedule $\pi^0$ has been generated and fixed, its completion span defines the reference horizon for subsequent rescheduling, $T_{\mathrm{ref}}^0:=H^0=C_{\max}(\pi^0)$, which is not redefined during rescheduling. The total labor-hour load at station $m$, $L_m=\sum_{m_i=m}q_iD_i(W_i)$, is used only for reward shaping within the algorithm and for workload balance diagnostics, rather than as a separate optimization objective.

#### 3.2.2 System Feasibility Constraints

An initial schedule must strictly satisfy the following physical constraints across multiple dimensions:

1. **Unique operation assignment and station monotonicity**:
   $$\sum_{m\in\mathcal{M}} x_{im} = 1, \quad m_i = \sum_{m\in\mathcal{M}} m x_{im}, \quad \forall i\in\mathcal{O}^{+}; \qquad m_j \ge m_i, \quad \forall (i,j)\in\mathcal{E}^{+}.$$
2. **Temporal precedence within the template**: Immediate predecessor completion constraints apply on the common template timeline, whether the operations are assigned to the same station or different stations:
   $$B_j \ge C_i, \qquad \forall (i,j)\in\mathcal{E}^{+}.$$
3. **Team size, skill qualifications, and station exclusivity**: The team size must equal the required headcount $q_i$, all members must possess skill $k_i$, and a worker cannot serve multiple stations:
   $$\sum_{w\in\mathcal{W}} y_{iw} = q_i, \quad y_{iw} \le H_{w k_i}, \quad \forall i\in\mathcal{O}^{+}, w\in\mathcal{W}; \qquad y_{iw} = y_{jw} = 1 \Longrightarrow m_i = m_j, \quad \forall i,j\in\mathcal{O}^{+}, w\in\mathcal{W}.$$
4. **Temporal and spatial concurrency capacity**: At any time, a worker can perform at most one operation, and the number of concurrent operations at a station cannot exceed its slot capacity $c_m$:
   $$\sum_{i\in\mathcal{O}^{+}} y_{iw} \mathbb{I}(B_i \le t < C_i) \le 1, \quad \forall w\in\mathcal{W}, t\ge0; \qquad \sum_{i\in\mathcal{O}^{+}} x_{im} \mathbb{I}(B_i \le t < C_i) \le c_m, \quad \forall m\in\mathcal{M}, t\ge0.$$
5. **Deterministic forward time allocation**: Let $c_t$ be the environment clock at the decision step. The lower bound on the start time of operation $i$ is $\underline{B}_i=\max\{c_t,\max_{j\in\mathcal{P}(i)}C_j,\max_{w\in W_i}a_w\}$, where $a_w$ is the earliest availability time of worker $w$. Starting from $\underline{B}_i$, the forward allocation engine searches for the earliest continuous interval during which the station concurrency satisfies $n_{m_i}(u)<c_{m_i}$, determines $B_i$, and sets $C_i=B_i+D_i(W_i)$.

### 3.3 Steady-State Cyclic Unfolding Across Aircraft and Physical Feasibility

**Proposition 1 (Physical feasibility of steady-state cyclic unfolding)**:
Given a cyclic operation template $\pi=(m,W,B,C)$ and cycle span $H=C_{\max}$, let aircraft be indexed by $k=0,1,2,\dots$. For aircraft $k$, the physical execution interval of operation $i$ on the absolute shop-floor timeline is defined by

$$
B_{i,k}^{\mathrm{phys}} = (k+m_i-1)H + B_i, \qquad C_{i,k}^{\mathrm{phys}} = (k+m_i-1)H + C_i.
$$

Under station monotonicity ($m_j\ge m_i$) and within-template temporal precedence ($B_j\ge C_i$), this unfolded production schedule strictly satisfies exclusive aircraft occupancy, technological precedence, worker mutual exclusion, and station capacity limits as the line advances.

**Proof and physical interpretation**:

1. **Exclusive aircraft occupancy and advancement rhythm**: The operations of aircraft $k$ at station $m$ are confined to the absolute time window $[(k+m-1)H,(k+m)H)$. The residence windows of consecutive aircraft at the same physical station do not overlap, ensuring exclusive spatial occupancy by the aircraft.
2. **Precedence feasibility**: For any physical precedence edge $(i,j)\in\mathcal{E}^{+}$, the execution time difference after unfolding satisfies
   $$B_{j,k}^{\mathrm{phys}} - C_{i,k}^{\mathrm{phys}} = (m_j - m_i)H + (B_j - C_i) \ge 0.$$
   This inequality holds because $m_j\ge m_i$ and $B_j\ge C_i$. In particular, for operations at different stations with $m_j>m_i$, $(m_j-m_i)H\ge H$. The predecessor operations of the same aircraft at upstream stations have therefore been completed in earlier pulsation cycles, and physical assembly precedence is naturally satisfied by station advancement. Uniformly imposing $B_j\ge C_i$ within a single template provides a **structured sufficient condition** for physically executable unfolding across aircraft.
3. **Scope of applicability**: This treatment eliminates the combinatorial complexity of tracking dependencies across cycles within a single-cycle template coordinate system, trading moderate conservatism for rigorous physical consistency and practical solvability.

### 3.4 Stability-Aware Cyclic Operation Template Rescheduling Under Material Delays

#### 3.4.1 Disturbance Definition and Operation Partitioning

The initial schedule $\pi^0$ serves as the baseline production template. When a material delay occurs, if the arrival of material for operation $i$ is delayed by $\Delta_i>0$ relative to its baseline start time, its updated release time is $r_i=B_i^0+\Delta_i$. At the schedule freeze time $\tau^r$, operations are partitioned into the frozen set $\mathcal{O}^{\mathrm{frz}}=\{i\in\mathcal{O}^{+}\mid B_i^0\le\tau^r\}$ and the reschedulable set $\mathcal{O}^{\mathrm{mov}}=\{i\in\mathcal{O}^{+}\mid B_i^0>\tau^r\}$.

#### 3.4.2 Strict Inheritance of Schedule Commitments and Worker Bindings

1. **Locking frozen operations**: For frozen operations already committed to shop-floor execution, the station, team, and timing are inherited exactly from the baseline schedule:
   $$(m_i^r, W_i^r, B_i^r, C_i^r) = (m_i^0, W_i^0, B_i^0, C_i^0), \quad \forall i\in\mathcal{O}^{\mathrm{frz}}.$$
2. **Complete inheritance of worker–station bindings**: The station assignments of all workers selected in the complete baseline schedule are locked and inherited ($\ell_w^r=m$); these workers cannot move across stations during rescheduling. Only workers never used in the baseline schedule remain unbound.
3. **Start-time lower bounds for reschedulable operations**: Reschedulable operations may change stations and teams, but their start times must satisfy $B_i^r\ge\max\{\tau^r,r_i\},\forall i\in\mathcal{O}^{\mathrm{mov}}$, along with all physical feasibility constraints in Section 3.2.

#### 3.4.3 Composite Stability Objective

Let the revised cycle span be $H^r:=C_{\max}^r$, and define the cycle-span overrun as $\Delta_{\mathrm{ref}}^r=\max\{0,C_{\max}^r-T_{\mathrm{ref}}^0\}$. With $N_{\mathrm{mov}}=|\mathcal{O}^{\mathrm{mov}}|$, define the mean start-time deviation as $\bar D_{\mathrm{start}}^r=\frac{1}{N_{\mathrm{mov}}}\sum_{i\in\mathcal{O}^{\mathrm{mov}}}|B_i^r-B_i^0|$, the station change rate as $R_{\mathrm{sta}}^r=\frac{1}{N_{\mathrm{mov}}}\sum\mathbb{I}(m_i^r\ne m_i^0)$, and the team change rate as $R_{\mathrm{team}}^r=\frac{1}{N_{\mathrm{mov}}}\sum\mathbb{I}(W_i^r\ne W_i^0)$. The multiobjective rescheduling evaluation function is defined as

$$
\min_{\pi^r\in\Pi_{\mathrm{feas}}^r}\; S(\pi^r) = 0.20\frac{C_{\max}^r}{T_{\mathrm{ref}}^0} + 3\frac{\Delta_{\mathrm{ref}}^r}{T_{\mathrm{ref}}^0} + 4\frac{\bar D_{\mathrm{start}}^r}{T_{\mathrm{ref}}^0} + 4 R_{\mathrm{sta}}^r + 0.3 R_{\mathrm{team}}^r. \tag{4}
$$

In this composite objective, $0.20C_{\max}^r/T_{\mathrm{ref}}^0$ maintains basic completion efficiency, $3\Delta_{\mathrm{ref}}^r/T_{\mathrm{ref}}^0$ penalizes the pulsation delay required to preserve synchronous advancement of the entire line, and the final three terms jointly control disruption caused by shop-floor adjustments. The weights are set to balance the typical magnitudes of the individual components observed in development scenarios.

**Scope of rescheduling**: Rescheduling in this study is defined as predictive–reactive, plan-level cyclic operation template rebalancing for subsequent production batches (Plan-Level Cyclic Template Rebalancing). It evaluates the ability of production management to restore plans under material disturbances. Transient physical flows of work in progress (WIP) across aircraft and buffer blocking during the transition from the old template to the new one are not explicitly simulated. Disturbances are restricted to known material arrival delays.

## IV. HGP-PPO Method

### 4.1 Unified Sequential MDP and Reinforcement Learning Framework

The initial scheduling and stability-aware rescheduling problems formulated in Section III are represented within a unified sequential Markov decision process (MDP), $\mathcal{M}^z=(\mathcal{S}^z,\mathcal{A},\mathcal{P},\mathcal{R}^z,\gamma)$, where $z\in\{\mathrm{init},\mathrm{res}\}$:

1. **State space and graph representation**: At decision step $t$, the environment state $s_t$ is represented by a heterogeneous graph $G_t$ with four node types. Operation nodes contain precedence, assignment status, and material release information; station nodes contain labor-hour loads and slot calendars; and worker nodes contain skills, efficiencies, and station bindings $\ell_w$. Rescheduling states additionally include baseline template start times and frozen-operation indicators.
2. **Hierarchical joint action decomposition**: Each resource decision produces a joint action $a_t=(i_t,m_t,W_t)$ specifying an operation, a physical station, and a specific worker team, with $|W_t|=q_{i_t}$. Internally, the policy autoregressively generates an ordered trajectory $\widetilde a_t=(i_t,m_t,\mathbf w_t)$, which the environment maps to the unordered set $W_t$ once team construction is complete.
3. **Dynamic feasibility masks and deterministic transitions**: At the operation level, the environment excludes operations whose predecessors are incomplete or whose materials are not ready. At the station level, it excludes choices that violate station monotonicity. At the worker level, it excludes workers lacking the required skill or already bound to other stations. After an action is selected, the forward allocation engine deterministically advances the state and updates the graph topology and availability calendars.
4. **Stage-specific rewards and parameter transfer**:
   - *Initial scheduling* uses a difference reward based on consecutive increments in the estimated makespan and the standard deviation of station labor-hour loads: $\widetilde r_t^{\mathrm{init}}=-(\widehat C_{t+1}-\widehat C_t)/C_{\mathrm{ref}}-0.5(\sigma_{L,t+1}-\sigma_{L,t})/L_{\mathrm{ref}}$, followed by scaling and the incorporation of a deadlock penalty;
   - *Rescheduling* directly uses the composite-objective difference reward $\widetilde r_t^{\mathrm{res}}=-[\widehat S(s_{t+1})-\widehat S(s_t)]$;
   - The framework first trains and fixes the initial scheduling policy, then initializes the rescheduling model with its parameters and fine-tunes it in the disturbance environment to transfer knowledge.

Fig. 2 presents the overall HGP-PPO architecture and solution process.

![Overall HGP-PPO framework](figures/fig2.png)

**Fig. 2. Overall HGP-PPO framework.** The APAL environment state is first converted into a heterogeneous graph with Operation, Station, Worker, and Skill nodes. The Actor and Critic use independent heterogeneous graph encoders. After attention-based context aggregation, the Actor sequentially selects an Operation, a Station, and a Worker team of the required size to form an executable joint action $(o,m,W)$. The Critic aggregates the graph state and estimates the state value $V(s)$. Executing the joint action produces the next state, reward, and trajectory, which PPO uses to update the policy and value networks.

#### 4.2 Heterogeneous Graph Representation and Global Encoding

**Four node types and state features.** At decision step $t$, the environment represents the APAL state as a heterogeneous graph $G_t=(\mathcal{V}_t,\mathcal{E}_t)$, where $\mathcal{V}_t=\mathcal{O}\cup\mathcal{M}\cup\mathcal{W}\cup\mathcal{K}$ comprises operation, station, worker, and skill nodes, respectively. The number of skill types is fixed at $|\mathcal{K}|=5$. Workers use five-dimensional multihot skill encodings, whereas physical operations use five-dimensional one-hot encodings. Zero-duration nodes have zero skill requirements and are excluded from the operation candidate set. Table 4-1 summarizes the input features of the four node types.

**Table 4-1. Input State Information for the Four Node Types**

| Node type | Main state information |
|---|---|
| Operation | Standard duration $p_i^{\mathrm{std}}$, scheduling status, five skill requirements, headcount $q_i$, and waiting time until material release; rescheduling additionally includes the baseline start time, baseline station, baseline team size, and frozen and delayed indicators |
| Station | Accumulated labor-hour load, proportions of bound and unbound workers, earliest slot waiting time, and available slots |
| Worker | Baseline efficiency, five skills, estimated waiting time, idle status, and station binding $\ell_w$ |
| Skill | Skill category and the number, current availability, and mean efficiency of workers possessing that skill |

Continuous features are normalized using scales derived from each instance. Let $\bar p=\frac{1}{|\mathcal{O}^{+}|}\sum_{i\in\mathcal{O}^{+}}p_i^{\mathrm{std}}$; processing durations are normalized as $\widetilde p_i=p_i^{\mathrm{std}}/\bar p$. Worker and station waiting times are logarithmically scaled. Station labor-hour loads are jointly represented relative to the ideal evenly distributed load, the total line load, and the current maximum load. Rescheduling time offsets are normalized by the baseline horizon $T_{\mathrm{ref}}^0$.

**Relation types and dynamic graph updates.** The heterogeneous graph defines the eight directed relations listed in Table 4-2. Technological precedence and skill supply–demand relations remain static within an instance, whereas operation–station and operation–worker relations are dynamically added as joint decisions are executed.

**Table 4-2. Heterogeneous Graph Relations and Update Rules**

| Relation | Direction | Update rule | Meaning |
|---|---|---|---|
| precedes | Operation $\rightarrow$ Operation | Static | Technological precedence dependency |
| assigned-to / has-task | Operation $\rightleftarrows$ Station | Dynamic | Bidirectional assignment relation between operations and stations |
| done-by | Operation $\rightarrow$ Worker | Dynamic | Assignment of specific workers to operations |
| has-skill / provided-by | Worker $\rightleftarrows$ Skill | Fixed within an episode | Bidirectional association of worker skill qualifications |
| requires / required-by | Operation $\rightleftarrows$ Skill | Static | Bidirectional association of operation skill requirements |

**Skill nodes and graph attention encoding.** Skill nodes bridge operation skill requirements with worker skill competencies, enabling operations sharing identical skill demands and workers possessing multiple qualifications to engage in message passing through explicit relational edges. In conjunction with process precedence and dynamic assignment relations, this representation establishes a unified graph context for the policy to evaluate immediate task demands and concurrent labor competition. Raw features of the four node types are projected into a common latent space through type-specific mappings and LeakyReLU: $\mathbf h_v^{(0)}=f_X(\mathbf x_v)$, $v\in\mathcal V^X$, $X\in\{O,M,W,K\}$.

Relation-specific GATv2 layers [7] are then used for message passing. For a directed edge $u\rightarrow v$ of relation $r$, at layer $\ell$ and attention head $h$, the unnormalized attention score and attention weight are
$$
e_{uv}^{(r,\ell,h)} = \mathbf a_{r,h}^{(\ell)\top}\operatorname{LeakyReLU}\left(\mathbf W_{s,r,h}^{(\ell)}\mathbf h_u^{(\ell-1)} + \mathbf W_{t,r,h}^{(\ell)}\mathbf h_v^{(\ell-1)}\right), \quad
\alpha_{uv}^{(r,\ell,h)} = \frac{\exp\left(e_{uv}^{(r,\ell,h)}\right)}{\sum_{z\in\mathcal N_r(v)}\exp\left(e_{zv}^{(r,\ell,h)}\right)}.
$$
After aggregating the multihead messages for relation $r$ as $\mathbf m_v^{(r,\ell)}=\frac{1}{H}\sum_{h=1}^H\sum_{u\in\mathcal N_r(v)}\alpha_{uv}^{(r,\ell,h)}\mathbf W_{s,r,h}^{(\ell)}\mathbf h_u^{(\ell-1)}$, the node embedding is updated by summation and a residual connection:
$$
\mathbf h_v^{(\ell)} = \mathbf h_v^{(\ell-1)} + \operatorname{LeakyReLU}\left(\sum_{r\in\mathcal R_{\mathrm{in}}(v)}\mathbf m_v^{(r,\ell)}\right).
$$

**Global attention pooling by node type.** To obtain a global context of fixed dimensionality across varying graph sizes, the Actor applies learned attention pooling separately to Operation, Station, and Worker nodes:
$$
\beta_v^X = \frac{\exp(a_X(\mathbf h_v))}{\sum_{u\in\mathcal V^X}\exp(a_X(\mathbf h_u))}, \quad \mathbf g_X = \sum_{v\in\mathcal V^X}\beta_v^X\mathbf h_v, \quad \mathbf g_t = [\mathbf g_M \,\|\, \mathbf g_O \,\|\, \mathbf g_W] \in \mathbb R^{3d}.
$$
The Critic uses a pooling layer of the same architecture with independent parameters to estimate the state value $V(s_t)$.

### 4.3 Hierarchical Autoregressive Joint Decoding

**Joint action probability factorization.** To avoid directly handling the combinatorial explosion in $\mathcal A(s_t)$, the policy decomposes a joint action into an operation choice, a conditional station choice, and $q_{i_t}$ conditional worker choices. Let $\mathbf w_t=(w_t^{(1)},\dots,w_t^{(q_{i_t})})$ denote the ordered worker generation sequence and $W_t=\{w_t^{(1)},\dots,w_t^{(q_{i_t})}\}$ the corresponding unordered team. Define the ordered trajectory action as $\widetilde a_t=(i_t,m_t,\mathbf w_t)$ and the unordered execution action as $a_t=(i_t,m_t,W_t)$. The policy factorizes as
$$
\pi_\theta(\widetilde a_t\mid s_t) = \pi_O(i_t\mid s_t)\,\pi_M(m_t\mid s_t,i_t)\prod_{r=1}^{q_{i_t}}\pi_W\left(w_t^{(r)}\mid s_t,i_t,m_t,\mathbf w_t^{(<r)}\right).
$$
Here, $\mathbf w_t^{(<r)}=(w_t^{(1)},\dots,w_t^{(r-1)})$ denotes the sequence of members already generated. Different member orders can map to the same unordered team. PPO computes the joint log-likelihood using the ordered trajectory $\widetilde a_t$ actually sampled, whereas physical environment execution and team change evaluation use the unordered set $W_t$.

**Operation and station pointers.** The operation pointer combines the global context $\mathbf g_t$ with operation embedding $\mathbf h_i$ to compute additive attention scores and normalizes them over the feasible operation set $\mathcal F_t^O$. The station selector then uses the selected operation embedding $\mathbf h_{i_t}$ as a query to score the five physical stations within the feasible set $\mathcal F_{i_t,t}^M$:
$$
u_{i,t}^O = \mathbf a_O^{\top}\tanh(\mathbf W_g\mathbf g_t + \mathbf W_O\mathbf h_i), \qquad \pi_O(i\mid s_t) = \frac{\exp(u_{i,t}^O)}{\sum_{j\in\mathcal F_t^O}\exp(u_{j,t}^O)}, \quad i\in\mathcal F_t^O;
$$
$$
u_{m,t}^M = \mathbf a_M^{\top}\tanh(\mathbf W_{OM}\mathbf h_{i_t} + \mathbf W_M\mathbf h_m), \qquad \pi_M(m\mid s_t,i_t) = \frac{\exp(u_{m,t}^M)}{\sum_{n\in\mathcal F_{i_t,t}^M}\exp(u_{n,t}^M)}, \quad m\in\mathcal F_{i_t,t}^M.
$$

**Fixed-size team construction and the Worker Pointer.** After the operation and station are selected, the Worker Pointer combines the current query $\mathbf q_t^{(r)}$ and candidate key $\mathbf k_{w,t}^{(r)}$ at each autoregressive substep $r\in\{1,\dots,q_{i_t}\}$ to score candidate workers and perform masked normalization:
$$
u_{w,t}^{(r)} = \mathbf v_W^{\top}\tanh\left(\mathbf q_t^{(r)} + \mathbf k_{w,t}^{(r)}\right), \qquad
\pi_W\left(w\mid s_t,i_t,m_t,\mathbf w_t^{(<r)}\right) = \frac{\exp(u_{w,t}^{(r)})}{\sum_{v\in\mathcal F_{W,t}^{(r)}}\exp(u_{v,t}^{(r)})}, \quad w\in\mathcal F_{W,t}^{(r)}.
$$
Infeasible candidates always receive zero probability. Candidate features incorporate supply–demand pressure for the five skills, dynamic EFT, and marginal scarcity. The network inputs, permutation-invariant pooling of partial teams, and scoring implementation are detailed in Appendix A.

**Dynamic feasibility masks and endogenous binding.** All three pointer levels maintain hard constraints through dynamic masks: infeasible action logits are set to $-\infty$.

1. **Operation mask** excludes operations with incomplete predecessors, materials that are not yet ready ($r_i>c_t$), or no feasible station–team combination;
2. **Station mask** excludes upstream stations according to the station monotonicity constraint imposed by predecessor operations ($m\ge\max_{j\in\mathcal P(i)}m_j$);
3. **Worker mask** excludes workers lacking the required skill ($H_{wk_i}=0$), already selected for the current team, or bound to another station.

A worker's station assignment is established endogenously and locked upon first selection:
$$
\ell_{w,t+1} = \begin{cases} m_t, & \ell_{w,t}=\varnothing \ \text{and}\ w\in W_t, \\ \ell_{w,t}, & \text{otherwise}. \end{cases}
$$
The worker subsequently remains available only to station $m_t$. Unbound workers remain available for selection across the entire line.

### 4.4 PPO Learning and Transfer

**Trajectory returns and generalized advantage estimation.** Policy updates use PPO [21] and GAE [20]. Let $d_t\in\{0,1\}$ indicate true task termination, either successful completion or deadlock, and let $u_t\in\{0,1\}$ indicate truncation upon exhaustion of the sampling budget. Define the continuation indicator as $b_t=(1-d_t)(1-u_t)$. The temporal-difference error and advantage estimate are
$$
\delta_t = r_t + \gamma b_t V_{\phi^-}(s_{t+1}) - V_{\phi^-}(s_t), \qquad \widehat A_t = \delta_t + \gamma\lambda b_t \widehat A_{t+1}, \qquad R_t = \widehat A_t + V_{\phi^-}(s_t).
$$
When true termination or nonterminal truncation occurs ($b_t=0$), both value bootstrapping and backward advantage propagation are stopped to avoid errors from joining trajectory segments.

**Autoregressive probability ratio and loss functions.** PPO operates on the likelihood of the complete ordered autoregressive trajectory $\widetilde a_t=(i_t,m_t,\mathbf w_t)$. The joint log-probability is the sum of the conditional log-probabilities across the three levels: $\ell_{t,\theta}=\log\pi_O(i_t\mid s_t)+\log\pi_M(m_t\mid s_t,i_t)+\sum_{r=1}^{q_{i_t}}\log\pi_W(w_t^{(r)}\mid\dots)$. The joint probability ratio between the new and old policies is $r_t^{\mathrm{PPO}}(\theta)=\exp(\ell_{t,\theta}-\ell_{t,\theta^-})$. With normalized advantage $\widetilde A_t$, the clipped PPO policy loss, Critic loss, and total loss are defined as
$$
\mathcal L_{\mathrm{policy}}(\theta) = -\mathbb E_t\left[\min\left(r_t^{\mathrm{PPO}}(\theta)\widetilde A_t,\, \operatorname{clip}(r_t^{\mathrm{PPO}}(\theta), 1-\epsilon, 1+\epsilon)\widetilde A_t\right)\right],
$$
$$
\mathcal L_V(\phi) = \mathbb E_t\left[(V_\phi(s_t)-R_t)^2\right], \qquad \mathcal L = \mathcal L_{\mathrm{policy}}(\theta) + c_V\mathcal L_V(\phi) - c_H\overline H,
$$
where $\overline H$ is a weighted sum of the conditional entropies of the three distribution levels, used to maintain effective exploration.

**Compatibility across scales and parameter transfer for rescheduling.** In HGP-PPO's heterogeneous graph attention and pointer network backbone, denoted HB-GAT-PN with the corresponding training algorithm HB-GAT-PPO, the network parameters are independent of the numbers of operations and workers. Continuous features are nondimensionalized using instance-specific scales, making the architecture inherently compatible with APAL inputs of different sizes. Rescheduling retains this backbone and augments Operation node states with baseline start times, baseline stations, and frozen indicators. It inherits the worker–station bindings of the initial schedule and uses the initial model weights as a warm start for fine-tuning in the disturbance environment under the composite stability objective $S(\pi^r)$, enabling effective knowledge transfer between tasks.


## V. Experiments and Results

### 5.1 Evaluation Questions and Experimental Protocol

The experiments establish a correspondence between claims, experiments, and evaluation criteria through five evaluation questions (EQs):

1. **EQ1 (Extrapolation across scales)**: Can a single fixed policy directly solve real factory instances containing 283 to 3182 physical operations without gradient updates or fine-tuning?
2. **EQ2 (Joint decoding quality)**: Is operation–station–worker-team autoregressive pointer decoding competitive with conventional dispatching rules, heuristic search, and baseline models in terms of cycle span $H=C_{\max}$?
3. **EQ3 (Key structural ablations)**: Do heterogeneous graph attention encoding, endogenous worker–station binding, and full three-level joint decision-making address the actual physical constraints of APALs?
4. **EQ4 (Online runtime scaling)**: How do end-to-end rescheduling runtimes of neural forward inference and conventional search performed separately for each instance scale with the number of operations?
5. **EQ5 (Recovery under material delays)**: Can the proposed architecture balance physical schedule feasibility, cycle completion performance, and stability relative to the baseline plan under material readiness delays?

**Data partitioning and information isolation.** The experiments assign mutually exclusive Training, Development, and Locked test roles. The Training set contains 24 synthetic structural variants with 400–800 physical operations and is used for gradient-based policy updates. The Development set contains six independent instances and 54 disturbance scenarios and is used only for hyperparameter tuning, early stopping, and checkpoint selection. The Locked test set contains the four original factory instances in Table 5-2 and 36 fixed delay scenarios. It is evaluated once after network hyperparameters, training budgets, and model states have been fixed, and does not inform any subsequent adjustments.

**Hard feasibility requirements and single-run evaluation.** Every schedule must satisfy all constraints on technological precedence, station monotonicity, the capacities of the five physical stations, required team sizes $q_i$, skill compatibility, worker temporal mutual exclusion, and worker–station binding. Rescheduling must additionally satisfy inheritance at the freeze boundary $\tau^r$, delayed material arrivals, and inheritance of existing worker bindings. All schedules are audited by both a common constraint engine and an independent schedule validator. Completeness ($R_{\mathrm{complete}}=1$) and feasibility ($R_{\mathrm{valid}}=1$) are mandatory prerequisites for quality ranking; penalized makespans are not used to produce misleading rankings.

### 5.2 Evaluation Instances, Baselines, and Statistical Procedures

**APAL evaluation instances.** Four sets of real APAL operation data supplied by an aircraft manufacturer are used; their physical sizes are listed in Table 5-2. All instances have five fixed physical stations and the same station concurrency capacities. Operation specialty codes are grouped into five scheduling skill categories. Each positive-duration physical operation requires a single skill type and a fixed team size $q_i$. Precedence relationships are propagated through zero-duration nodes to physical immediate-predecessor constraints as described in Section 3.1.

**Table 5-2. Locked Evaluation Instances From Real Factory APAL Data**

| Instance | Physical operations | Zero-duration nodes | Total nodes | Stations | Experimental role |
|---|---:|---:|---:|---:|---|
| 283 | 283 | 7 | 290 | 5 | Locked test |
| 680 | 680 | 35 | 715 | 5 | Locked test |
| 2338 | 2338 | 64 | 2402 | 5 | Locked test |
| 3182 | 3182 | 117 | 3299 | 5 | Locked test |

**Table 5-4. Comparison Methods, Decision Criteria, and Search Budgets**

| Category | Method | Decision criteria / search representation and budget |
|---|---|---|
| Dispatching rules | SPT [22] / LPT | Select operations by shortest / longest standard duration first |
| Dispatching rules | ES-priority / CPM [13] / MSL | Prioritize earliest start / critical-path latest start / minimum slack |
| Dispatching rules | Random | Random priorities with a fixed random seed |
| Search and metaheuristics | Beam Search [17] | Search over operation, station, and worker priorities; initial: width=4, branch=4, 8 levels; rescheduling: width=2, branch=2 |
| Search and metaheuristics | IG [19] | Destruction–reconstruction search; initial: 160 iterations, destroy=0.10; rescheduling: 3 iterations |
| Search and metaheuristics | SA [14] | Simulated annealing; initial: 120 iterations, $T_0=0.05$; rescheduling: 3 iterations |
| Learning-based baselines | L2D-PPO-APAL | Graph-based learned dispatching policy adapted to APAL |
| Learning-based baselines | Graph-DDQN-APAL | Graph-based value-function reinforcement learning adapted to APAL |
| Rescheduling reference | NoReschedule | Negative control: retain the baseline schedule without active repair |
| Heuristic repair | StabilityAware / CPM hybrid | Hybrid rules balancing start-time deviations and the critical-path horizon |

**Comparison methods and search budgets.** The comparison methods span three paradigms: dispatching rules, metaheuristic search, and deep reinforcement learning. Table 5-4 summarizes their main decision criteria and search budgets fixed in advance. All methods share the same physical parameter ($\rho=0.95$) and forward allocation engine. Detailed network hyperparameters for the proposed model—a hidden dimension of 128, five GATv2 layers with four heads, and 11.2M parameters—and the PPO training configuration are provided in Appendix Table B-1.

**Evaluation metrics and statistical inference.** For initial scheduling, cycle span $H=C_{\max}$ in (2) is the sole primary quality metric. Deterministic masked argmax decoding at zero temperature is used for the main comparison, supplemented by means and standard deviations over five random seeds. For rescheduling, subject to strict physical feasibility, the weighted composite objective $S$ defined in Section 3.4 and (4) is the primary ranking criterion. Its components—cycle span $C_{\max}^r$, baseline overrun $\Delta_{\mathrm{ref}}^r$, start-time deviation $\bar D_{\mathrm{start}}^r$, station change rate $R_{\mathrm{sta}}^r$, and team change rate $R_{\mathrm{team}}^r$—are also examined individually. Rescheduling experiments cover 36 paired disturbance scenarios: four instances $\times$ three delay levels $\times$ three freeze timings. A scenario-paired Friedman test evaluates overall ranking differences among 19 feasible methods. Two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction control the familywise error rate, and 10,000 stratified bootstrap resamples are used to estimate 95% confidence intervals, ensuring rigorous statistical inference.

### 5.3 Initial Scheduling Results and Performance Across Scales

Table 5-6 summarizes deterministic decoding results for the four real factory APAL instances. All results in this table correspond to complete schedules that pass the common feasibility checks, so their makespans can be compared directly under the same constraints. The proposed method uses a single fixed policy to process instances with 283 to 3182 physical operations, without gradient updates or instance-specific tuning during testing.

**Table 5-6. Initial Scheduling Makespan of Comparison Methods and the Workforce Preassignment Ablation on Four APAL Instances (h; Lower Is Better)**

| Category | Method | 283 | 680 | 2338 | 3182 | Equal-weight mean across four scales |
|---|---|---:|---:|---:|---:|---:|
| Graph reinforcement learning | Proposed method | 286.96 | **489.00** | 1209.96 | **1623.69** | **902.40** |
| Workforce allocation ablation | Worker–Station preassignment | 320.31 | 959.63 | 1605.13 | 2645.84 | 1382.73 |
| Learning-based baselines | L2D-PPO-APAL | 288.97 | 518.65 | **1079.44** | 1804.71 | 922.94 |
| Learning-based baselines | Graph-DDQN-APAL | **286.77** | 612.29 | 1263.66 | 2049.70 | 1053.11 |
| Search and metaheuristics | IG | 339.05 | 961.10 | 1797.57 | 2759.01 | 1464.18 |
| Search and metaheuristics | Beam Search | 333.43 | 984.30 | 1904.88 | 3272.54 | 1623.79 |
| Search and metaheuristics | SA | 311.78 | 1238.00 | 2417.03 | 3256.29 | 1805.77 |
| Dispatching rules | LPT | 292.63 | 637.56 | 1497.35 | 2039.50 | 1116.76 |
| Dispatching rules | Random | 310.62 | 872.52 | 1414.20 | 2170.15 | 1191.87 |
| Dispatching rules | SPT | 411.57 | 1702.77 | 2479.07 | 3346.37 | 1984.94 |
| Dispatching rules | MSL | 412.56 | 1608.50 | 2496.05 | 3455.45 | 1993.14 |
| Dispatching rules | ES-priority | 435.13 | 1773.41 | 2780.24 | 3729.75 | 2179.63 |
| Dispatching rules | CPM | 387.28 | 1597.92 | 2522.96 | 3430.06 | 1984.55 |

Under equal weighting of the four scales, the proposed method achieves the lowest mean makespan of 902.40 h, a 2.22% reduction from the 922.94 h of L2D-PPO-APAL. Graph-DDQN-APAL has a mean makespan of 1053.11 h. Fig. 3(a) shows how cycle span scales on a logarithmic axis. The individual results indicate that this average advantage does not reflect consistent superiority at every scale: Graph-DDQN-APAL achieves 286.77 h on the 283-operation instance, slightly below the proposed method's 286.96 h, while L2D-PPO-APAL achieves 1079.44 h on the 2338-operation instance, compared with 1209.96 h for the proposed method. The proposed method achieves the lowest makespans on the 680- and 3182-operation instances, at 489.00 and 1623.69 h, respectively. On the largest instance, its makespan is 181.02 h, or 10.03%, lower than that of L2D-PPO-APAL. This result supports its ability to maintain good schedule quality at the largest tested scale, but does not imply superiority at every scale.

Dispatching rules and search performed separately for each instance exhibit more pronounced degradation with increasing scale. Among the dispatching rules, LPT has the lowest mean makespan across the four scales, at 1116.76 h. Among search and metaheuristic methods, IG has the lowest mean, at 1464.18 h. Beam Search and SA achieve mean makespans of 1623.79 and 1805.77 h, respectively. Because these methods share the forward allocation engine and feasibility evaluation criteria, the differences primarily reflect the quality of the joint allocations generated by their priority rules or search representations under the fixed solution budgets. These results should not be interpreted as the performance limits of the corresponding search algorithms under larger computational budgets.

The fixed stochastic decoding settings provide supplementary information about output robustness. The proposed method's stochastic decoding results on the four instances are $274.97\pm3.77$, $517.50\pm12.89$, $1070.71\pm101.12$, and $1678.47\pm55.94$ h, respectively, with an equal-weight mean of 885.41 h across scales. Stochastic decoding produces lower mean makespans than deterministic decoding on the 283- and 2338-operation instances, but higher means on the 680- and 3182-operation instances. Thus, stochastic sampling does not consistently improve results across scales. Deterministic decoding remains the main comparison protocol, and stochastic results are used only to describe sampling-induced variation in policy outputs.

Overall, a single fixed policy generates feasible schedules for four substantially different instance sizes without changes to the network architecture or model parameters and achieves the lowest equal-weight mean makespan across the four scales. This evidence establishes extrapolation across scales within the same APAL data family under template-related conditions. Because the training instances retain the template lineage of the real data, the results are not interpreted as demonstrating generalization across factories or datasets.

### 5.4 Decision Architecture and Network Configuration Comparisons and Feasibility Boundaries

To distinguish the roles of workforce allocation, graph representation, and joint decision depth, the Worker–Station preassignment comparison is included alongside all initial scheduling methods in Table 5-6, while Table 5-7 compares the proposed method with three structural variants. All methods share the same environment constraints and feasibility evaluation criteria. The two action-level variants simultaneously reduce observation scope, graph inputs, and masks, and use a minimum-waiting rule to fill in resources that are not explicitly selected. They therefore compare complete decision architectures rather than identify the independent causal effect of an individual Pointer module.

**Table 5-7. Initial Scheduling Results and Rescheduling Feasibility Rates for Different Decision Architectures and Network Configurations**

| Method | Main change | 283 | 680 | 2338 | 3182 | Mean across four scales | Feasible initial scheduling instances | Rescheduling feasibility rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed method | Heterogeneous graph; joint operation–Station–worker decisions | 286.96 | 489.00 | 1209.96 | 1623.69 | **902.40** | 4/4 | 100% |
| Homogeneous GraphSAGE | Replace heterogeneous graph encoding with homogeneous graph encoding | 292.02 | 594.04 | 1323.22 | 1941.29 | 1037.64 | 4/4 | 100% |
| Joint operation–Station decisions | No explicit worker selection; restricted worker information | **267.40** | 565.26 | **1154.20** | 2073.34 | 1015.05 | 4/4 | 0% |
| Operation-only decisions | Stations and workers are selected by rules | 269.66 | 779.13 | Infeasible | Infeasible | — | 2/4 | 100% |

Note: Initial scheduling columns report makespan under deterministic decoding (h; lower is better). Means are computed only when all four instances are feasible. Penalty values associated with infeasible schedules are excluded from comparisons. Rescheduling feasibility rates are calculated over 36 disturbance scenarios across the four scales.

![Initial scheduling cycle spans across scales and decision architecture ablation comparisons](figures/fig3_initial_and_ablation.png)

**Fig. 3. Initial scheduling cycle spans across physical scales and decision architecture ablations.** (a) Cycle spans $H$ of six representative methods on the 283-, 680- (in-distribution), 2338-, and 3182-operation (extrapolation) instances, plotted on a logarithmic axis; solid lines with circular markers indicate the proposed method's cycle values. (b) Matrix of relative makespan degradation $\Delta J(\%)$ for decision architecture variants compared with the Full Model at each scale. Lower values are better; negative values indicate shorter local makespans at the expense of global feasibility.

The Worker–Station preassignment ablation fixes workers' station assignments before scheduling, restricting the policy to selecting members within those assignments. This variant produces feasible schedules at all four scales, but its mean deterministic makespan is 1382.73 h, 53.2% higher than the proposed method's 902.40 h. Under the fixed stochastic decoding settings, its results are $345.08\pm20.20$, $881.46\pm67.70$, $1679.23\pm87.50$, and $2695.56\pm57.30$ h, respectively. Under the current data and evaluation protocol, these results indicate that fixing Worker–Station relationships before scheduling limits the ability of workforce allocation to adapt to subsequent station workloads and skill requirements.

As shown in the ablation matrix in Fig. 3(b), homogeneous GraphSAGE generates feasible schedules at all four scales, but increases mean makespan from 902.40 to 1037.64 h, indicating that distinguishing relationships among operations, stations, workers, and skills helps maintain solution quality across scales. The operation–Station variant achieves shorter makespans at some scales but fails to schedule all remaining operations during rescheduling. The operation-only variant is infeasible on the two larger initial scheduling instances. Thus, local advantages of shallower decisions on small instances cannot replace feasibility evaluation across scales.

Overall, the full joint representation and decoder exhibit more consistent feasibility and better average performance in these experiments. The results support the overall design, but do not establish that every internal component is indispensable. Section 5.5 further compares the composite quality of the rescheduling methods.

### 5.5 Rescheduling Results Under Material Delays

Table 5-8 compares rescheduling results across 36 paired scenarios formed by four instance sizes, three delay intensities, and three trigger stages. The table includes all 19 feasible methods entering the Friedman analysis and retains NoReschedule as a negative control. The operation–Station variant, whose rescheduling feasibility rate is 0% in Table 5-7, is excluded from quality statistics. Lower composite scores $S$ are better.

**Table 5-8. Composite Scores and Mean Makespans for Rescheduling Under Material Delays**

| Category | Method | 283 | 680 | 2338 | 3182 | Mean $S$ | Mean makespan (h) | Feasibility rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graph reinforcement learning | Proposed method | 0.9572 / 297.7 | 0.7621 / 674.9 | 0.8627 / 1223.9 | 0.8224 / 1709.5 | **0.8511** | 976.5 | 100% |
| Model variant | Earlier model without baseline-assignment conditioning features | 0.9700 / 300.1 | 0.8242 / 682.5 | 0.9186 / 1239.2 | 1.0588 / 1786.6 | 0.9429 | 1002.1 | 100% |
| Structural comparison | Homogeneous GraphSAGE | 1.2236 / 318.9 | 1.0447 / 682.4 | 1.0310 / 1263.2 | 1.1599 / 1806.8 | 1.1148 | 1017.8 | 100% |
| Structural comparison | Operation-only decisions | 0.9592 / 296.9 | 1.2479 / 678.6 | 1.0825 / 1223.8 | 1.6577 / 1903.3 | 1.2368 | 1025.7 | 100% |
| Learning-based baselines | L2D-PPO-APAL | 0.9439 / 301.3 | 0.8441 / 663.3 | 0.8401 / 1223.9 | 1.0325 / 1764.9 | 0.9152 | 988.3 | 100% |
| Learning-based baselines | Graph-DDQN-APAL | 1.0414 / 306.6 | 1.0714 / 672.8 | 0.9394 / 1223.9 | 0.8716 / 1709.9 | 0.9810 | 978.3 | 100% |
| Metaheuristic search | BeamSearchRepair | 1.0575 / 322.9 | 1.0763 / 752.4 | 1.2596 / 1403.6 | 1.3626 / 1987.2 | 1.1890 | 1116.5 | 100% |
| Metaheuristic search | IteratedGreedyRepair | 1.0312 / 319.1 | 1.6770 / 783.1 | 1.6892 / 1454.1 | 1.3655 / 1987.5 | 1.4407 | 1135.9 | 100% |
| Metaheuristic search | SimulatedAnnealingRepair | 1.1431 / 323.6 | 2.2088 / 814.1 | 1.8923 / 1484.4 | 1.4394 / 2001.8 | 1.6709 | 1156.0 | 100% |
| Heuristic repair | HybridCPMStabilityRepair | 1.1337 / 294.2 | 1.0463 / 675.4 | 0.9952 / 1223.9 | 0.7560 / 1665.9 | 0.9828 | **964.9** | 100% |
| Heuristic repair | StabilityAwareRepair | 0.9913 / 310.4 | 0.7554 / 682.7 | 0.8045 / 1229.4 | 0.9591 / 1752.8 | 0.8776 | 993.8 | 100% |
| Heuristic repair | TaktAwareRepair | 0.9103 / 323.8 | 1.0759 / 775.7 | 1.0398 / 1307.4 | 0.9073 / 1816.4 | 0.9833 | 1055.8 | 100% |
| Heuristic repair | ReleaseAwareRepair | 0.7844 / 316.8 | 1.4584 / 770.5 | 1.1644 / 1369.4 | 1.1556 / 1901.5 | 1.1407 | 1089.5 | 100% |
| Heuristic repair | BottleneckSkillRepair | 0.9667 / 326.4 | 1.4732 / 768.0 | 1.4282 / 1404.7 | 0.8398 / 1827.5 | 1.1770 | 1081.7 | 100% |
| Heuristic repair | SPTRepair | 1.0188 / 333.3 | 0.9794 / 758.4 | 1.1850 / 1373.5 | 1.2038 / 1914.2 | 1.0967 | 1094.9 | 100% |
| Heuristic repair | FullRescheduleCPM | 1.3579 / 332.0 | 1.5415 / 778.9 | 1.5528 / 1384.3 | 1.1864 / 1875.6 | 1.4097 | 1092.7 | 100% |
| Heuristic repair | CPMRepair | 1.3513 / 331.1 | 1.5147 / 773.9 | 1.5968 / 1396.1 | 1.1838 / 1880.2 | 1.4116 | 1095.3 | 100% |
| Heuristic repair | LPTRepair | 1.5492 / 345.0 | 2.6828 / 836.3 | 1.9494 / 1456.0 | 1.3546 / 1922.1 | 1.8840 | 1139.9 | 100% |
| Heuristic repair | RandomRepair | 1.5427 / 351.7 | 3.2131 / 943.2 | 2.6982 / 1673.2 | 1.9854 / 2153.4 | 2.3599 | 1280.4 | 100% |
| Negative control | NoReschedule | — | — | — | — | Not applicable | Not applicable | 0% |

Note: Each instance column reports "$S$/makespan (h)." Within each scenario, $S$ is normalized using the baseline schedule horizon $T_{\mathrm{ref}}^0$ of the corresponding instance, and is then averaged with equal weights first within instances and subsequently across instances.

Compared with the final proposed model, the earlier model without baseline-assignment conditioning features, homogeneous GraphSAGE, and the operation-only variant have mean composite scores that are 10.79%, 30.98%, and 45.32% higher, respectively. The mean start-time deviations of homogeneous GraphSAGE and the operation-only variant are 65.93 and 111.94 h, both above the proposed method's 30.85 h. The comparison between the earlier and final models involves differences in both conditioning features and training stages. It is therefore treated as a comparison between model versions, without attributing the difference exclusively to a single component.

The proposed method maintains a 100% feasibility rate across all 36 scenarios and achieves the lowest mean composite score across scales, at 0.8511. Its median and interquartile mean (IQM) composite scores are 0.7415 and 0.7670, below those of StabilityAwareRepair (0.7574 and 0.7878) and L2D-PPO-APAL (0.7592 and 0.8042). Its average relative deviation (ARD) is 26.60%, the lowest among the 19 feasible methods. Its observed composite-score range is 1.0089, also the smallest among all feasible methods. The Friedman test indicates differences in overall rankings among the 19 feasible methods ($\chi_F^2(18)=369.20$, $p<0.001$). The proposed method has the lowest average rank of 4.56, followed by L2D-PPO-APAL and StabilityAwareRepair at 4.86 and 5.00, respectively. Together, these results indicate that the proposed method ranks among the best in both mean performance and robust central tendency, rather than obtaining a low mean solely from a few extreme scenarios.

The proposed method does not achieve the lowest score at every individual scale; its mean composite scores for the four scales range from 0.7621 to 0.9572. Its advantage lies primarily in the equal-weight mean across scales and in limiting poor outcomes, rather than in absolute superiority at a single scale. StabilityAwareRepair, the second-best method, has a mean composite score of 0.8776, followed by L2D-PPO-APAL and Graph-DDQN-APAL at 0.9152 and 0.9810, respectively.

Table 5-9 further decomposes the physical components directly relevant to these findings to identify the benefits and costs associated with a lower composite score.

**Table 5-9. Raw Rescheduling Components for Representative Methods**

| Method | $C_{\max}$ (h) | $\Delta_{\mathrm{ref}}$ (h) | $\bar D_{\mathrm{start}}$ (h) | $R_{\mathrm{sta}}$ (%) | $R_{\mathrm{team}}$ (%) | $S$ |
|---|---:|---:|---:|---:|---:|---:|
| Proposed method | 976.49 | 29.51 | **30.85** | 2.12 | 95.60 | **0.851100** |
| L2D-PPO-APAL | 988.33 | 42.11 | 45.54 | 1.93 | 95.87 | 0.915151 |
| Graph-DDQN-APAL | 978.32 | 29.63 | 56.31 | 2.02 | 96.05 | 0.980965 |
| BeamSearchRepair | 1116.52 | 167.19 | 108.85 | 0.54 | 0.70 | 1.188985 |
| HybridCPMStabilityRepair | **964.87** | **18.79** | 61.76 | 1.85 | 93.53 | 0.982793 |
| StabilityAwareRepair | 993.85 | 45.22 | 33.88 | 1.73 | 94.76 | 0.877564 |

Note: Physical components aggregated across scales are descriptive summaries only. The composite score $S$ is computed by normalization and weighting within each scenario, followed by equal-weight averaging at two levels. All five physical components of the proposed method have been fully checked against inference records for the 36 fixed disturbance scenarios. $R_{\mathrm{team}}$ is determined by a binary comparison of unordered team sets for reschedulable operations: changing any member counts as a team change for that operation. Thus, 95.60% means that 95.60% of reschedulable operations have at least one team member replaced. It does not mean that 95.60% of workers are replaced or that workers move between stations.

Relative to StabilityAwareRepair, the proposed method reduces mean makespan by 17.36 h, with shorter makespans in 32 scenarios and longer makespans in four. The mean overrun beyond the baseline schedule horizon decreases by 15.71 h, corresponding to 29 wins, three ties, and four losses. Both component differences remain significant after Holm correction over 25 physical-component comparisons (adjusted $p<0.001$). These findings indicate that the main benefits over the stability-aware rule concern completion efficiency and control of baseline-horizon overruns. Meanwhile, although mean start-time deviation decreases from 33.88 to 30.85 h, the scenario-level comparison yields only nine wins and 27 losses, and the corrected difference is not significant. Station and team change rates also increase from 1.73% and 94.76% to 2.12% and 95.60%, respectively. This comparison therefore reveals trade-offs among stability components rather than simultaneous dominance on all raw metrics.

Relative to L2D-PPO-APAL, the proposed method reduces mean makespan, baseline-horizon overrun, and start-time deviation by 11.84, 12.60, and 14.70 h, respectively, including a 32.3% reduction in mean start-time deviation. However, none of these component differences is significant after correction for multiple comparisons. HybridCPMStabilityRepair achieves the lowest mean makespan and overrun, but its start-time deviation reaches 61.76 h. BeamSearchRepair maintains low station and team change rates at the cost of longer makespans, greater overruns, and larger start-time deviations. Thus, the composite-score ranking reflects weighted trade-offs among multiple recovery objectives and cannot be replaced by any single component.

**Table 5-10. Paired Rescheduling Statistics Against the Two Closest Baselines**

| Baseline | Mean $S$ difference (proposed − baseline) | Relative reduction | Wins / ties / losses | $p_{\mathrm{Holm}}$ | $r_{\mathrm{rb}}$ | Stratified bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| L2D-PPO-APAL | −0.0640 | 7.00% | 19 / 0 / 17 | 0.7237 | −0.069 | [−0.1344, −0.0023] |
| StabilityAwareRepair | −0.0265 | 3.01% | 22 / 0 / 14 | 0.5498 | −0.210 | [−0.0652, 0.0110] |

Note: Negative differences and effect sizes indicate lower composite scores for the proposed method. Wins, ties, and losses are reported from the proposed method's perspective. $p_{\mathrm{Holm}}$ denotes the two-sided Wilcoxon result after correction for comparisons against 18 feasible baselines. Bootstrap intervals are based on 10,000 paired resamples within instances and equal weighting across the four instances.

Relative to L2D-PPO-APAL, the proposed method reduces the mean composite score by 7.00%, and the stratified bootstrap interval for the mean difference lies entirely below zero. However, the Holm-corrected Wilcoxon test does not indicate a consistent scenario-level rank advantage, and the 19 / 0 / 17 win–tie–loss distribution shows that the methods outperform each other in different scenarios. These statistical analyses address different questions: the stratified bootstrap estimates uncertainty in the mean difference under equal weighting of the four instances, whereas Wilcoxon–Holm tests paired rank differences across the 36 scenarios while controlling for 18 pairwise comparisons. Fig. 4 visualizes these scenario differences and win–tie–loss distributions through paired-difference heatmaps and instance-level mean recovery score plots. Relative to StabilityAwareRepair, the proposed method reduces the mean composite score by 3.01% and achieves a lower score in 22 scenarios, but the bootstrap interval includes zero and the corrected Wilcoxon result is not significant. The available evidence therefore supports a lower mean point estimate on the fixed test grid, but not a consistent scenario-level statistical advantage over both closest baselines.

![Paired-difference heatmaps for 36 material-delay template rescheduling scenarios and instance-level means](figures/fig4_reschedule_scenarios_heatmap.png)

**Fig. 4. Paired-difference heatmaps for 36 template rescheduling scenarios under dynamic material delays and instance-level mean recovery scores.** (a) Composite recovery scores $S$ for the four physical instance sizes and their equal-weight mean; lower is better. (b) Symmetric diverging heatmaps of paired score differences $d=S_{\mathrm{HGP}}-S_{\mathrm{Baseline}}$ between the proposed method and L2D-PPO-APAL or StabilityAwareRepair across 36 disturbance scenarios (four instances $\times$ three delay levels $\times$ three freeze timings). Blue indicates an advantage for the proposed method, and red indicates an advantage for the baseline. Win / Tie / Loss badges at the top summarize the pairwise comparisons.

**Table 5-11. End-to-End Rescheduling Time per Scenario Across Four Scales (s)**

| Method | 283 | 680 | 2338 | 3182 |
|---|---:|---:|---:|---:|
| Proposed method | **8.02** | **12.84** | **51.91** | **123.60** |
| BeamSearchRepair | 114.02 | 262.72 | 1787.00 | 3902.89 |
| IteratedGreedyRepair | 147.17 | 410.42 | 2416.80 | 4202.01 |
| SimulatedAnnealingRepair | 292.42 | 676.78 | 3445.39 | 4296.64 |

Note: Runtime includes representation construction, action selection or search, forward allocation, and feasibility validation, but excludes offline training and one-time model loading. Each proposed-method entry is the mean over the nine fixed scenarios for the corresponding instance. For the three search methods, each entry is a mean single-run time over 27 records per instance: nine scenarios multiplied by three fixed solver seeds. Hardware configurations differ across methods; the table therefore describes measured scaling trends rather than strict speedup ratios across execution environments.

![Online rescheduling runtime and scalability across four physical scales](figures/fig5_runtime_scalability.png)

**Fig. 5. Logarithmic comparison of end-to-end online rescheduling runtime and computational scalability across physical scales.** The proposed method, using neural forward inference, is compared with Beam Search, Iterated Greedy, and Simulated Annealing on instances containing 283 to 3182 physical operations. The vertical axis shows end-to-end solution time per scenario in seconds on a logarithmic scale.

End-to-end runtime increases with the number of operations. For the proposed method, the mean solution time over nine disturbance scenarios per instance ranges from 8.02 to 123.60 s across the four instances. As shown in Fig. 5, search performed separately for each instance incurs rapidly increasing runtime on large instances, requiring 3900–4300 s for a single repair at the 3182-operation scale. In contrast, through lightweight neural forward inference, the proposed method repairs the full-line plan in approximately 2 minutes even at this scale, demonstrating strong online responsiveness and computational scalability.


Overall, the results support the proposed method's ability to balance schedule feasibility, composite quality across scales, and completion efficiency under material delays. This conclusion is restricted to the cyclic operation template stability-aware rescheduling problem defined in this study and the 36 fixed scenarios. It does not cover transient physical WIP flows across aircraft during the transition between templates, nor does it imply that the proposed method is optimal at every scale or on every raw component.

## VI. Conclusions and Limitations

To address tightly coupled physical resources and constraints in APALs, including technological precedence, station concurrency, multiskilled collaborative teams, and worker–station binding, this paper proposes HGP-PPO, a deep reinforcement learning scheduling framework combining heterogeneous graph attention over four node types with a three-level autoregressive pointer decoder. The main findings and engineering implications are as follows:

1. **Joint decisions over multiple resources**: Scheduling is formulated as a hierarchical operation–station–worker autoregressive MDP. Dynamic action masks and a forward time allocation engine avoid the combinatorial explosion caused by team selection and enable endogenous worker–station binding and global coordination during schedule construction.
2. **Extrapolation across scales and rapid rescheduling**: On real factory instances containing 283 to 3182 operations, the model demonstrates strong generalization across scales without fine-tuning. Its mean cycle span across the four scales is 902.40 h, and it reduces cycle span by 10.03% relative to the baseline on the largest 3182-operation instance. Across 36 material-delay scenarios, it maintains a 100% physical feasibility rate and significantly outperforms alternatives on the composite recovery objective. Online repair of a single scenario at the largest scale takes approximately 2 minutes, indicating substantial value for real-time industrial response.

**Limitations and future work.** The method has several scope restrictions and limitations. First, the current model focuses on static construction of a single pulsation-cycle template and rescheduling on a discrete time horizon. It does not explicitly model transient physical transfers of WIP across aircraft or transient interstation congestion during synchronous station advancement. Second, the experiments assume deterministic standard operation durations and do not consider duration variability or random rework. Third, worker skills are simplified as binary coverage with fixed efficiencies, without detailed modeling of fatigue accumulation or learning effects. Future work will extend the approach to continuous transient simulation of the complete assembly process across multiple aircraft and investigate robust adaptive scheduling under uncertainty.

## Appendix A. Worker Pointer Architecture, Team State, and Dynamic Candidate Features

This appendix details the network layers, partial-team state representation, and dynamic feature calculations of the Worker Pointer introduced in Section 4.3.

**Network architecture and partial-team representation.** For the selected member sequence $\mathbf w_t^{(<r)}$, each selected worker is first projected through a linear layer as $\mathbf m_v=\mathbf W_{\mathrm{mem}}\mathbf h_v$. To ensure permutation invariance, the partial-team state concatenates Sum, Mean, and Max pooling features and fuses them through a fully connected layer:
$$
\mathbf h_{\mathrm{team},t}^{(r)} = \operatorname{LeakyReLU}\left(\mathbf W_{\mathrm{team}} \left[\sum_{v\in \mathbf w_t^{(<r)}} \mathbf m_v \,\|\, \frac{1}{\max(1, r-1)}\sum_{v} \mathbf m_v \,\|\, \max_{v\in \mathbf w_t^{(<r)}} \mathbf m_v\right]\right).
$$
The query $\mathbf q_t^{(r)}$ is obtained by concatenating and projecting the operation, station, global context, team state, aggregate pressure $\mathbf P_t$, skill consumption ratios, and worker selection progress: $\mathbf q_t^{(r)}=\mathbf W_q[\mathbf h_{i_t}\,\|\,\mathbf h_{m_t}\,\|\,\mathbf g_t\,\|\,\mathbf h_{\mathrm{team},t}^{(r)}\,\|\,\mathbf P_t\,\|\,\mathbf c_t^{(r)}\,\|\,\mathbf{prog}_t^{(r)}]+\mathbf b_q$. The candidate worker key combines the projection of the static embedding and pressure exposure with dynamic terms: $\mathbf k_{w,t}^{(r)}=\mathbf W_k[\mathbf h_w\,\|\,\mathbf e_w]+\mathbf W_{\mathrm{eft}}\mathbf f_{w,t}^{\mathrm{eft}}+\mathbf W_{\mathrm{sc}}\mu_{w,t}^{(r)}$. The scoring layer is $\mathbf v_W^{\top}\tanh(\mathbf q_t^{(r)}+\mathbf k_{w,t}^{(r)})$. Infeasible candidate scores are set to $-10^4$ before Softmax is applied.

**Global and near-term skill pressure.** Let $R_{jk}\in\{0,1\}$ indicate whether operation $j$ requires skill $k$, $H_{wk}\in\{0,1\}$ indicate whether worker $w$ possesses skill $k$, and $\widetilde p_j$ denote normalized standard duration. Let $\mathcal U_t$ be the set of unscheduled positive-duration operations, and let $\mathcal R_t\subseteq\mathcal U_t$ contain operations that are currently ready and eligible for action. Then

$$
D_{k,t}^{\mathrm{all}}=
\sum_{j\in\mathcal U_t}R_{jk}\widetilde p_jq_j,
\qquad
D_{k,t}^{\mathrm{near}}=
\sum_{j\in\mathcal R_t}R_{jk}\widetilde p_jq_j.
$$

Let $\widetilde a_{w,t}\ge0$ be the normalized waiting time of worker $w$ relative to the current time, and let $\mathcal W_t^q$ contain workers not excluded by the optional queuing rule. Then

$$
S_{k,t}^{\mathrm{all}}=
\sum_{w\in\mathcal W}H_{wk},
\qquad
S_{k,t}^{\mathrm{near}}=
\sum_{w\in\mathcal W_t^q}
H_{wk}\exp\!\left(-\frac{\widetilde a_{w,t}}{\tau_p}\right),
$$

$$
P_{k,t}^{h}=
\log\!\left(
1+\frac{D_{k,t}^{h}}{\max(S_{k,t}^{h},\varepsilon_s)}
\right),
\qquad h\in\{\mathrm{all},\mathrm{near}\},
$$

where $\tau_p>0$ is the waiting-time discount temperature and $\varepsilon_s>0$ prevents division by zero when supply is absent. Candidate worker pressure exposure is obtained by multiplying the worker's five-dimensional skill vector elementwise by $\mathbf P_t^{\mathrm{all}}$ and $\mathbf P_t^{\mathrm{near}}$, respectively.

**Dynamic EFT.** Let $A_t^{(r-1)}$ and $E_t^{(r-1)}$ denote the maximum waiting time and total efficiency of the partial team, respectively. Let $\widetilde a_{m_t,t}$ be the slot waiting time at the selected station and $\eta_w$ the baseline efficiency of candidate worker $w$. After adding candidate $w$, the normalized tentative completion quantity is

$$
\widehat C_{w,t}^{(r)}=
\max\!\left(
A_t^{(r-1)},\widetilde a_{w,t},\widetilde a_{m_t,t}
\right)
+\frac{\widetilde p_{i_t}q_{i_t}}
{\rho^{r-1}\left(E_t^{(r-1)}+\eta_w\right)},
\qquad \rho=0.95.
$$

The implementation further transforms this quantity into its difference from the current best feasible candidate and its standardized score within the feasible candidate set, which are used as ranking features by the Worker Pointer.

**Marginal scarcity of additional skills.** Let $C_{k,t}^{(r)}$ denote the cumulative consumption of skill $k$ by previously selected members. The marginal scarcity of candidate worker $w$'s additional skills is defined as

$$
\begin{aligned}
\mu_{w,t}^{(r)}=
\sum_{k\in\mathcal K}{}&H_{wk}(1-R_{i_tk})
\Bigg[
\log\!\left(
1+\frac{D_{k,t}^{\mathrm{all}}}
{\max(S_{k,t}^{\mathrm{all}}-C_{k,t}^{(r)}-H_{wk},\varepsilon_s)}
\right)\\
&-\log\!\left(
1+\frac{D_{k,t}^{\mathrm{all}}}
{\max(S_{k,t}^{\mathrm{all}}-C_{k,t}^{(r)},\varepsilon_s)}
\right)
\Bigg]_{+}.
\end{aligned}
$$

The factor $1-R_{i_tk}$ excludes the skill required by the current operation and measures only the incremental pressure on the reserves of the worker's additional skills caused by selecting that worker. Next-frontier pressure is derived from the five-dimensional near-term skill demand of operations that may become immediately available after the current action and is combined elementwise with the candidate worker's skill vector. It indicates near-term resource opportunity costs without changing any hard constraints or actual processing durations.

## Appendix B. Experimental Reproducibility and Statistical Audit Configuration

This appendix consolidates the key reproducibility information underlying the reported results. Implementation details concerning run directories and configuration management are omitted from the main method description.

**Table B-1. Key Reproducibility Settings of the Proposed Method**

| Category | Configuration item | Final setting |
|---|---|---|
| Dataset size | Training / Development / Locked test | 24 / 6 / 4 instances |
| Disturbance scenarios | Development / Locked test | 54 / 36; nine fixed scenarios per base instance |
| Model size | Trainable parameters | Approximately 11.2 M |
| Network architecture | Hidden dimension / relation attention layers / heads | 128 / 5 / 4 |
| PPO | Learning rate / $\gamma$ / GAE $\lambda$ | $1.0\times10^{-4}$ / 0.999 / 0.9 |
| PPO | Update epochs / clipping range | 2 / linearly decayed from 0.20 to 0.10 |
| Numerical processing | Difference amplification / clipping / global scaling | 100 / $[-50,50]$ / 0.005 |
| Loss regularization | Value loss coefficient / entropy coefficient | 0.5 / $2.0\times10^{-4}\to5.0\times10^{-5}$ |
| Optimization | Optimizer / weight decay | ScheduleFree AdamW / $1.0\times10^{-4}$ |
| Gradient control | Actor / Critic gradient norm limits | 0.5 / 0.05 |
| Rescheduling sampling | Batch size / parallel environments / gradient accumulation | 64 or 128 / 4 / 4 |
| Numerical precision | Training precision | BF16 mixed precision |
| Training budget | Initial scheduling / rescheduling | 60 epochs / 120 epochs (two stages) |
| Random settings | Rescheduling training seeds | 42 / 43 |
| Main evaluation decoding | Decoding rule / temperature | masked argmax / 0 |
| Supplementary decoding | Temperature / number of decoding runs | 0.01 / five predefined seeds |

The two rescheduling training stages have measured wall-clock times of 1 h 42 min 25 s and 1 h 43 min 46 s, respectively, in the same Linux CUDA environment, totaling 3 h 26 min 11 s. These times document the offline training cost of the proposed model and are not used for speed-ratio comparisons with runs performed on different hardware.

**Definitions of statistics and evaluation metrics.** To ensure rigorous and transparent statistical inference, the statistics used in rescheduling evaluation are defined as follows:

1. **Interquartile mean (IQM)**: After sorting the samples, the lowest 25% and highest 25% are removed, and the arithmetic mean of the middle 50% is computed. This statistic characterizes robust central tendency after excluding the influence of extreme values.
2. **Average relative deviation (ARD)**: For method $m$ over scenario set $\mathcal S$, where $|\mathcal S|=36$, define $\mathrm{ARD}_m=\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}\frac{S_{ms}-S_s^*}{S_s^*}\times100\%$. Here, $S_s^*=\min_k S_{ks}$ is the **best composite score observed among all feasible comparison methods in scenario $s$**, rather than a theoretical global optimum. ARD measures the method's mean gap relative to the current best-known performance.
3. **Paired rank-biserial correlation ($r_{\mathrm{rb}}$)**: As a nonparametric paired effect size for the two-sided Wilcoxon signed-rank test, this statistic is defined as the difference between the positive and negative rank sums divided by their total: $r_{\mathrm{rb}}=(W^+-W^-)/(W^++W^-)$. It ranges from $-1$ to $1$, with negative values indicating a more favorable overall rank distribution for the proposed method in the paired comparison.

Rescheduling statistics are computed from paired results for each of the 36 scenarios. Composite scores are first normalized by the corresponding instance's $T_{\mathrm{ref}}^0$, then aggregated with equal weights within and across instances. The Friedman test includes 19 methods that are feasible in all scenarios. Paired Wilcoxon tests apply Holm–Bonferroni correction across 18 baselines, and the stratified bootstrap uses 10,000 paired resamples within instances. The scenario-level result matrix, five raw components, statistical scripts, and consolidated statistical records are retained as separate audit materials.

## References

[1] Ai J., Qiao F., Liu J., et al. A graph-enhanced deep-reinforcement learning approach for the aircraft pulse assembly line balancing problem[C] // Proceedings of the 2025 China Automation Congress (CAC). 2025: 6623–6628.
[2] Bao Z., Chen L., Qiu K. An aircraft final assembly line balancing problem considering resource constraints and parallel task scheduling[J]. Computers & Industrial Engineering, 2023, 182: 109436.
[3] Bao Z., Chen L., Qiu K. A surrogate-assisted heuristic approach for the joint optimization of resource allocation and scheduling of an aircraft final assembly line[J]. Journal of Manufacturing Systems, 2023, 70: 102–116.
[4] Battaïa O., Dolgui A. A taxonomy of line balancing problems and their solution approaches[J]. International Journal of Production Economics, 2013, 142(2): 259–277.
[5] Becker C., Scholl A. A survey on problems and methods in generalized assembly line balancing[J]. European Journal of Operational Research, 2006, 168(3): 694–715.
[6] Boysen N., Fliedner M., Scholl A. A classification of assembly line balancing problems[J]. European Journal of Operational Research, 2007, 183(2): 674–693.
[7] Brody S., Alon U., Yahav E. How attentive are graph attention networks?[C] // International Conference on Learning Representations (ICLR). 2022.
[8] Cappart Q., Chételat D., Khalil E. B., et al. Combinatorial optimization and reasoning with graph neural networks[J]. Journal of Machine Learning Research, 2023, 24(130): 1–61.
[9] Chen B., Zhang J., Luo X., et al. A multi-agent graph attention deep reinforcement learning method for aircraft assembly dynamic scheduling[J]. Advanced Engineering Informatics, 2026, 68: 104565.
[10] Fang P., Yang J.-J., Jiang Y., et al. Flexible worker allocation in aircraft final assembly line using multiobjective evolutionary algorithms[J]. IEEE Transactions on Industrial Informatics, 2021, 17(11): 7468–7478.
[11] Fang S., Liu M., Xian G., et al. Aircraft moving assembly line: Pulsating and continuous[J]. Aeronautical Manufacturing Technology, 2021, 64(16): 40–50. (Fang Shuiliang, Liu Mengnan, Xian Guo, et al. Aircraft moving assembly line—Pulsating and continuous[J]. Aeronautical Manufacturing Technology, 2021, 64(16): 40–50. [In Chinese.])
[12] Hu J.-H., Zhu J.-Y., Sun Y.-N., et al. Network-based two-stage robust scheduling strategy for the aircraft assembly system with uncertain duration[J]. Computers & Industrial Engineering, 2024, 188: 109911.
[13] Kelley J. E., Jr. Critical-path planning and scheduling: Mathematical basis[J]. Operations Research, 1961, 9(3): 296–320.
[14] Kirkpatrick S., Gelatt C. D., Vecchi M. P. Optimization by simulated annealing[J]. Science, 1983, 220(4598): 671–680.
[15] Liu M., Li L., Liu X., et al. A multi-pass heuristic for multi-skilled worker scheduling in aircraft final assembly line with variable duration[J]. Journal of Systems Engineering and Electronics, 2025, 36(6): 1532–1547.
[16] Lu Z., Ren Y., Wang L., et al. A resource investment problem based on project splitting with time windows for aircraft moving assembly line[J]. Computers & Industrial Engineering, 2019, 135: 568–581.
[17] Ow P. S., Morton T. E. Filtered beam search in scheduling[J]. International Journal of Production Research, 1988, 26(1): 35–62.
[18] Park J., Chun J., Kim S. H., et al. Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning[J]. International Journal of Production Research, 2021, 59(11): 3342–3359.
[19] Ruiz R., Stützle T. A simple and effective iterated greedy algorithm for the permutation flowshop scheduling problem[J]. European Journal of Operational Research, 2007, 177(3): 2033–2049.
[20] Schulman J., Moritz P., Levine S., et al. High-dimensional continuous control using generalized advantage estimation[C] // International Conference on Learning Representations (ICLR). 2016.
[21] Schulman J., Wolski F., Dhariwal P., et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.
[22] Smith W. E. Various optimizers for single-stage production[J]. Naval Research Logistics Quarterly, 1956, 3(1–2): 59–66.
[23] Van de Vonder S., Demeulemeester E., Herroelen W. A classification of predictive-reactive project scheduling procedures[J]. Journal of Scheduling, 2007, 10(3): 195–207.
[24] Vieira G. E., Herrmann J. W., Lin E. Rescheduling manufacturing systems: A framework of strategies, policies, and methods[J]. Journal of Scheduling, 2003, 6(1): 39–62.
[25] Wang D., Qiao F., Guan L., et al. Human–machine collaborative optimization method for dynamic worker allocation in aircraft final assembly lines[J]. Computers & Industrial Engineering, 2024, 194: 110370.
[26] Wang R., Zhang G., Sun H., et al. Flexible job shop scheduling via dual attention network-based reinforcement learning[J]. IEEE Transactions on Neural Networks and Learning Systems, 2024, 35(9): 12513–12527.
[27] Wen X., Li X., Zhang Q., et al. A Q-learning improved particle swarm optimization for aircraft pulsating assembly line scheduling problem considering skilled operator allocation[J]. Computers, Materials & Continua, 2026, 86(1): 1–27.
[28] Wen X., Zhang H., Yang W., et al. Research on a worker skill level-aware pulsating assembly scheduling method for aircraft based on the GIN-PPO algorithm[J]. Computers & Industrial Engineering, 2026, 218: 112135.
[29] Zhang C., Song W., Cao Z., et al. Learning to dispatch for job shop scheduling via deep reinforcement learning[C] // Advances in Neural Information Processing Systems (NeurIPS). 2020, 33: 1621–1632.
[30] Zhu H., Gao L., Li X. Reactive scheduling for aircraft moving assembly line with material delivery disturbance[J]. International Journal of Production Research, 2021, 59(8): 2439–2456.
