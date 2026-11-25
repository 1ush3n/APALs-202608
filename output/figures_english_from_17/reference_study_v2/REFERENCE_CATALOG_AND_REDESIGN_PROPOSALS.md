# APAL Top-Venue Reference Atlas (`reference_study_v2`) & Redesign Blueprints

> **arXiv Terms of Use & License Notice**:  
> Per `D:\APAL-Dynamic-v4\.licenses\literature_search_arxiv_LICENSE.txt` and [arXiv API / Terms of Use](https://info.arxiv.org/help/api/index.html), all arXiv preprints were retrieved sequentially (minimum 3.6–3.7s delay per request) for academic visual reference only, with full source URLs preserved below. Reuse of any third-party figure artwork directly in a publication requires checking the specific copyright license of each individual paper (or redrawing original APAL figures inspired solely by layout grammar and information design).

---

## 1. Status of `paper_integrated_v1/` Centering Fix

In response to the pinned comment `"这里好像没有居中"` on `paper_integrated_v1/previews/fig3_standalone_b5.png` (and the mirrored geometry in `fig2_compact_b1_b5.png`), isolated scripts in `paper_integrated_v1/scripts/` (`build_fig3_standalone_b5.py`, `build_fig2_compact_b1_b5.py`, and `build_paper_deck.py`) were updated without altering `refined_v8/`:
- Aligned `LeakyReLU (0.1)` activation node and text label to exact vertical midpoints between `Linear` (`W_s h_u + W_t h_v`) and `GATv2 Attention` (equal `0.095 in` upper and lower stems in standalone `Fig. 3`).
- Centered the rotated `Disjoint weights (θ ∩ φ = ∅)` divider pill cleanly along the inter-tower axis.
- Full deck rebuilt and verified (**9/9 QA PASS**, visually inspected via rendered `fig3_standalone_b5.png` and `fig2_compact_b1_b5.png`).

---

## 2. Visual Diagnosis of Current `fig1_apal_unfolding_delay.png`

Before redrawing Fig. 1, visual inspection against top CS/IEEE venues highlights why the current `fig1_apal_unfolding_delay.png` feels weak:
1. **Panel (a) reads like a flat tabular box grid**: Uniform pastel rectangles (`Aircraft k`, `k+1`, ...) with tiny repeating plane thumbnails and box headers look like a slide diagram rather than a systems/scheduling figure (contrast with `C1_05` OctoPipe, `C1_07` Chimera, `C1_08` Hanayo, `C1_10` Zero Bubble).
2. **Panel (b) looks like a standalone Matplotlib lollipop chart**: Plotting all 22 delayed rows as thin stems with a large boxed text callout obscures structural precedence & resource-wait semantics and competes visually with experimental result plots (contrast with `C1_01` L-RHO, `C1_02` Zero Bubble Rollback, `C1_03` ISBJSSP, `C1_04` Train Rescheduling, `C1_09` Residual Scheduling).

---

## 3. Three Concrete Fig. 1 Redesign Blueprints

Review Boards `C1_Board_01.png`–`C1_Board_04.png` showcase 16 non-duplicate top-venue references. Three actionable redesign blueprints for **APAL Fig. 1**:

### Blueprint A [Recommended Primary Option]: Coupled Cyclic Pipeline + Rolling-Horizon Repair Loop
* **Primary Visual Models**: `C1_01` (L-RHO, ICLR'25), `C1_02` (Zero Bubble Rollback, ICLR'24 Oral), `C1_04` (Train Timetable Rescheduling), `C1_05` (OctoPipe, SC'26), `C1_10` (Zero Bubble Gantt).
* **Layout Structure**:
  * **Panel (a) — Station-by-Pulse Interlocking Pipeline (Top / Left)**: Instead of floating plane boxes, draw `S1..S5` as horizontal tracks across Pulse cycles `H` with crisp interlocking operation blocks (`Aircraft k`, `k+1`, ...), ONE highlighted diagonal aircraft path (`k`), and a compact right-hand station load/slot strip (`c_m`, skill diversity) inspired by Chimera (`C1_07`).
  * **Panel (b) — Trigger `tau_r` Disruption -> Stability-Preserving Repair (Bottom / Right)**:
    1. Top strip shows **Nominal vs Disruption at `tau_r = 65.83h`**: locked frozen prefix shaded cool grey on the left of a crisp dashed clock line (`tau_r`, `C1_09` style), and only **3–4 representative ops** (`O94`, `O111` material delay `+Delta_i` in amber, downstream `O223`) rather than 22 raw rows.
    2. Center inset mini-graph links **Material Delay -> Worker Contention Edge (`W5`)** directly to **Propagation Wait (`B_i^r > r_i`)** (`C1_04` red conflict -> green revised flow).
    3. Right/Bottom compact repaired timeline shows final makespan shift badge (`H^0 = 292.6h -> H^r = 305.9h`, `OctoPipe C1_05` pill style).

### Blueprint B: Three-Stage Topological Mutation (`G_0 -> G_tau -> G_r`) + Dual Gantt Band
* **Primary Visual Models**: `C1_03` (ISBJSSP Dynamic Disjunctive Graph), `C1_08` (Hanayo Wavefront), `C1_09` (Residual Scheduling Cursor).
* **Layout Structure**: Top row displays the graph-level meaning of disruption (`Nominal DAG -> Disruption Deletion/Delay at tau_r -> Reinstated Conflict-Free Repair`), and bottom row aligns compact Before/After multi-station Gantt strips across the same `tau_r` trigger line. Strongly emphasizes why simple right-shifting fails when multi-skill workers (`W_5`) are shared across aircraft pulses.

### Blueprint C: Compact 3-Column Problem–Mechanism–Objective Overview
* **Primary Visual Models**: `C1_13` (TARGETNET, NeurIPS'25), `C1_12` (AlpaServe, OSDI'23), `C1_14` (MatNet FFSP, NeurIPS'21), `C1_15` (PPipe, USENIX ATC'25).
* **Layout Structure**: Left column = APAL structural ingredients (5 Stations, Precedence, 5 Trade Skills, Multi-man `q_i`, Fatigue); Center column = Matrix-to-Gantt schedule with `PPipe`-style worker reservation probe intervals showing trigger `tau_r != r_i`; Right column = Stability vs Makespan recovery metrics. Ideal if space must be kept compact (single half-page horizontal strip).

---

## 4. Fig. 2 Seamless Fusion Recommendations (`C2_Board_01..02`)

Our `fig2_compact_b1_b5.png` is solid conceptually, and can reach higher polish by borrowing from `C2_01`–`C2_08`:
* **HybridNet (`C2_03`, IROS'22) & ScheduleNet (`C2_02`)**: Replace the chunky thick purple block arrow between Panel (a)'s shop-floor photo and `State Graph G_t` with hairline leader lines that project highlighted physical elements directly into typed graph nodes across a unified outer container seam.
* **OctoPipe (`C2_08`, SC'26)**: Present raw node feature definitions (`x_M, x_O, x_K, x_W`) as sleek color-matched schema cards docked neatly beside/below the graph rather than tiny color-bar legends.
* **ISBJSSP (`C2_07`)**: Add one subtle dashed callout circle around focus node `O_3` to distinguish static (`precedes`, `requires`) vs dynamic (`assigned-to`) neighborhood sets at a glance.

---

## 5. Fig. 3 Verdict: Can Encoder + Decoder (+ Critic) Be Unified on One Canvas?

**Yes.** Category 3 (`C3_Board_01..03`) proves top venues routinely present **Encoder + Decoder** in a single cohesive figure (`2D-Ptr` AAMAS'24 `C3_01`, `TARGETNET` NeurIPS'25 `C3_03`, `Decima` SIGCOMM'19 `C3_04`, `RL4CO` KDD'25 `C3_05`, `LEHD` NeurIPS'23 `C3_06`, `Wheatley` ICTAI'25 `C3_12`). However, concatenating B5 + B6 + B8 verbatim would overflow and shrink text because B5 currently duplicates the full multi-block `GATv2` anatomy inside both Actor and Critic towers.

Two clean architectural paths:
1. **Option 3A [Recommended if unifying Fig. 3]: Macro-to-Micro Unified Dual-Tower Enc-Dec (modeled on `2D-Ptr C3_01` + `LEHD C3_06` + `MatNet C3_07`)**
   * Keep **Macro Flow** crisp: `Shared State Graph G_t -> Actor Encoder (theta) & Critic Encoder (phi) separated by Disjoint Pill -> Pooling [M|O|W] -> Cascading 3-Tier Autoregressive Decoder (Op Ptr -> St Ptr -> Wk Ptr, modeled on HGS C3_02) + Value Head -> PPO Update`.
   * Extract the repeated internal layer detail (`Linear -> LReLU(0.1) -> GATv2 Attn -> Per-Relation Softmax -> Mean+Sum -> Post-Norm`) into **ONE single detached Micro-Block Zoom Inset** on the side/bottom (exactly like `LEHD Fig. 1` and `MatNet Fig. 2b`), freeing vertical room so the 3-Tier Pointer Decoder & conditional feasibility masks (`O -> S -> W`) sit comfortably inside the same figure without clutter.
2. **Option 3B [Alternative if full multi-draw trace needed]: Unified Macro Fig. 3 + Mechanistic Multi-Draw Trace Fig. 4 (modeled on `RL4CO C3_05` + `AM C3_09`)**
   * Main Fig. 3 presents the clean macro unified Encoder + 3-Tier Decoder + PPO loop, while a focused companion Fig. 4 / Appendix A keeps the full numerical `Draw 1 (r=1, W2=0.52) -> Mask -> Draw 2 (r=2, W5=0.64)` walkthrough (`B8 + B6`).

---

## 6. Complete 36-Figure Reference Catalog (Zero Overlap with `figure_study_15`)

### Category 1: Fig. 1 Problem Overview, Cyclic Pipeline & Dynamic Rescheduling (`assets/cat1_fig1_problem/`)

| ID | Venue | Fig | Paper Title | Source URL | Local Crop |
|---|---|---|---|---|---|
| `C1_01` | ICLR 2025 | Fig. 1 | Learning-Guided Rolling Horizon Optimization for Long-Horizon FJSP (L-RHO) | https://arxiv.org/abs/2502.15791 | [C1_01.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_01.png) |
| `C1_02` | ICLR 2024 Oral | Fig. 4 | Zero Bubble Pipeline Parallelism (Post-Validation Rollback) | https://arxiv.org/abs/2401.10241 | [C1_02.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_02.png) |
| `C1_03` | arXiv 2023 | Fig. 2 | Generating Dispatching Rules for Interrupting Swap-Allowed Blocking JSSP | https://arxiv.org/abs/2302.02506 | [C1_03.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_03.png) |
| `C1_04` | arXiv 2024 | Fig. 7 | RL for Scalable Train Timetable Rescheduling with Graph Representation | https://arxiv.org/abs/2401.06952 | [C1_04.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_04.png) |
| `C1_05` | SC 2026 | Fig. 3 | OctoPipe: Co-Optimizing Partitioning, Placement, and Scheduling | https://arxiv.org/abs/2509.23722 | [C1_05.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_05.png) |
| `C1_06` | SC 2026 | Fig. 4 | OctoPipe: Irregular Execution & Deadlock-Free Overlap Timeline | https://arxiv.org/abs/2509.23722 | [C1_06.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_06.png) |
| `C1_07` | SC 2021 | Fig. 2 | Chimera: Training Large Neural Networks with Bidirectional Pipelines | https://arxiv.org/abs/2107.06925 | [C1_07.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_07.png) |
| `C1_08` | SC 2023 | Fig. 3 | Hanayo: Harnessing Wave-like Pipeline Parallelism | https://arxiv.org/abs/2308.15762 | [C1_08.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_08.png) |
| `C1_09` | arXiv 2023 | Fig. 2 | Residual Scheduling: Step-by-Step Partial Construction with Clock Cursor | https://arxiv.org/abs/2309.15517 | [C1_09.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_09.png) |
| `C1_10` | ICLR 2024 Oral | Fig. 3 | Zero Bubble Pipeline Parallelism (ZB-H1 vs ZB-H2) | https://arxiv.org/abs/2401.10241 | [C1_10.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_10.png) |
| `C1_11` | ICML 2021 | Fig. 2 | TeraPipe: Token-Level Pipeline Parallelism for Large Language Models | https://arxiv.org/abs/2102.07988 | [C1_11.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_11.png) |
| `C1_12` | OSDI 2023 | Fig. 1 | AlpaServe: Statistical Multiplexing with Model Parallelism | https://arxiv.org/abs/2302.11665 | [C1_12.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_12.png) |
| `C1_13` | NeurIPS 2025 | Fig. 1 | Heterogeneous Graph Transformers for Mobile Multi-Agent Task Allocation (TARGETNET) | https://proceedings.neurips.cc | [C1_13.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_13.png) |
| `C1_14` | NeurIPS 2021 | Fig. 4 | Matrix Encoding Networks for Neural CO (FFSP Stage Matrices -> Gantt) | https://arxiv.org/abs/2106.11113 | [C1_14.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_14.png) |
| `C1_15` | USENIX ATC 2025 | Fig. 5 | PPipe: Video Analytics Serving via Pool-Based Pipeline Parallelism | https://arxiv.org/abs/2507.18748 | [C1_15.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_15.png) |
| `C1_16` | C&OR 2024 | Fig. 3 | Graph Neural Networks for Job Shop Scheduling Problems: A Survey | https://arxiv.org/abs/2406.14096 | [C1_16.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat1_fig1_problem/C1_16.png) |

### Category 2: Fig. 2 Physical Scene / Workcell -> Heterogeneous Graph -> GNN-RL (`assets/cat2_fig2_fusion/`)

| ID | Venue | Fig | Paper Title | Source URL | Local Crop |
|---|---|---|---|---|---|
| `C2_01` | arXiv / ICRA 2024 | Fig. 4 | Subassembly to Full Assembly: Assembly Sequence Planning via Graph RL (S2A) | https://arxiv.org/abs/2409.13620 | [C2_01.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_01.png) |
| `C2_02` | arXiv 2021 (KAIST) | Fig. 1 | ScheduleNet: Learn to Solve Multi-Agent Scheduling Problems with RL | https://arxiv.org/abs/2106.03051 | [C2_02.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_02.png) |
| `C2_03` | IROS 2022 | Fig. 1 | Learning Coordination Policies over Hetero Graphs for Human-Robot Teams (HybridNet) | https://arxiv.org/abs/2301.13279 | [C2_03.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_03.png) |
| `C2_04` | SIGCOMM 2019 | Fig. 4 | Learning Scheduling Algorithms for Data Processing Clusters (Decima) | https://people.csail.mit.edu/hongzi/content/publications/Decima-Sigcomm19.pdf | [C2_04.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_04.png) |
| `C2_05` | IEEE TNNLS 2024 | Fig. 2 | Flexible Job Shop Scheduling via Dual Attention Network Based RL (DAN) | https://arxiv.org/abs/2305.05119 | [C2_05.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_05.png) |
| `C2_06` | arXiv / IEEE 2024 | Fig. 2 | Learning-Enabled Flexible Job-Shop Scheduling with Transportation (HGS) | https://arxiv.org/abs/2402.08979 | [C2_06.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_06.png) |
| `C2_07` | arXiv 2023 | Fig. 3 | Dynamic Disruptions GNN-RL Framework for Interrupting Blocking JSSP | https://arxiv.org/abs/2302.02506 | [C2_07.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_07.png) |
| `C2_08` | SC 2026 | Fig. 6 | OctoPipe: Hardware + Execution DAG + Structured Node Schema Cards | https://arxiv.org/abs/2509.23722 | [C2_08.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat2_fig2_fusion/C2_08.png) |

### Category 3: Fig. 3 Unified Encoder + Autoregressive Pointer Decoder (`assets/cat3_fig3_encdec/`)

| ID | Venue | Fig | Paper Title | Source URL | Local Crop |
|---|---|---|---|---|---|
| `C3_01` | AAMAS 2024 | Fig. 2 | 2D-Ptr: 2D Array Pointer Network for Heterogeneous CVRP | https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1238.pdf | [C3_01.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_01.png) |
| `C3_02` | arXiv / IEEE 2024 | Fig. 4 | HGS Three-Stage Autoregressive Decoder (Operation -> Machine -> Vehicle) | https://arxiv.org/abs/2402.08979 | [C3_02.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_02.png) |
| `C3_03` | NeurIPS 2025 | Fig. 2 | TARGETNET: Simultaneous Task Allocation & Scheduling Policy + Edge-Attn Layer | https://proceedings.neurips.cc | [C3_03.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_03.png) |
| `C3_04` | SIGCOMM 2019 | Fig. 6 | Decima Policy Network (Hierarchical DAG GNN + Two-Head Softmax) | https://people.csail.mit.edu/hongzi/content/publications/Decima-Sigcomm19.pdf | [C3_04.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_04.png) |
| `C3_05` | KDD 2025 | Fig. 4.1 | RL4CO: Modularized Encoder f + Autoregressive Decoder g -> Env | https://arxiv.org/abs/2306.17100 | [C3_05.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_05.png) |
| `C3_06` | NeurIPS 2023 | Fig. 1 | Neural Combinatorial Optimization with Heavy Decoder (LEHD) | https://arxiv.org/abs/2310.07985 | [C3_06.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_06.png) |
| `C3_07` | NeurIPS 2021 | Fig. 2 | Matrix Encoding Networks (Dual-Tower F_A/F_B + Mixed-Score Zoom) | https://arxiv.org/abs/2106.11113 | [C3_07.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_07.png) |
| `C3_08` | AAAI 2024 | Fig. 2 | Equity-Transformer: Min-Max Routing via Multi-Agent Context Enc-Dec | https://arxiv.org/abs/2306.02689 | [C3_08.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_08.png) |
| `C3_09` | ICLR 2019 | Fig. 2 | Attention, Learn to Solve Routing Problems! (Unrolled t=1..4 Pointer Steps) | https://arxiv.org/abs/1803.08475 | [C3_09.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_09.png) |
| `C3_10` | ICLR 2024 | Fig. 3 | Deep RL Guided Improvement Heuristic for JSSP (L2S TPM+CAM -> Action Selection) | https://arxiv.org/abs/2211.10936 | [C3_10.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_10.png) |
| `C3_11` | NeurIPS 2023 | Fig. 1 | BQ-NCO: Direct Enc-Dec MDP vs Dynamic Step Re-Encoding MDP Contrast | https://arxiv.org/abs/2301.03313 | [C3_11.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_11.png) |
| `C3_12` | ICTAI 2025 | Fig. 3 | Learning to Solve RCPSP with Duration Uncertainty (Wheatley GNN + PPO Actor/Critic) | https://arxiv.org/abs/2511.13214 | [C3_12.png](file:///D:/APAL-Dynamic-v4/output/figures_english_from_17/reference_study_v2/assets/cat3_fig3_encdec/C3_12.png) |

---

## 7. Complete List of Retrieved Papers & URLs

**arXiv Preprints (39 verified on-topic papers; `2311.14449v3` discarded as an off-topic optical physics hit)**:
- `https://arxiv.org/abs/1803.08475` — Attention, Learn to Solve Routing Problems! (ICLR 2019)
- `https://arxiv.org/abs/2006.09503` — Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW, ICML 2021)
- `https://arxiv.org/abs/2009.03836` — GraSP-RL: GNN-based Scheduler for Production Planning
- `https://arxiv.org/abs/2010.16011` — POMO: Policy Optimization with Multiple Optima (NeurIPS 2020)
- `https://arxiv.org/abs/2102.07988` — TeraPipe: Token-Level Pipeline Parallelism (ICML 2021)
- `https://arxiv.org/abs/2106.01086` — Learning to Schedule Job-Shop Problems: Representation & Policy Learning (IJPR)
- `https://arxiv.org/abs/2106.03051` — ScheduleNet: Learn to Solve Multi-Agent Scheduling Problems with RL
- `https://arxiv.org/abs/2106.11113` — Matrix Encoding Networks for Neural Combinatorial Optimization (MatNet, NeurIPS 2021)
- `https://arxiv.org/abs/2107.06925` — Chimera: Bidirectional Pipelines (SC 2021)
- `https://arxiv.org/abs/2201.00548` — Hybrid Intelligence for Dynamic Job-Shop Scheduling (GRL)
- `https://arxiv.org/abs/2211.10936` — DRL Guided Improvement Heuristic for JSSP (L2S, ICLR 2024)
- `https://arxiv.org/abs/2212.10936` — A Memetic Algorithm with RL for Sociotechnical Production Scheduling (IEEE Access)
- `https://arxiv.org/abs/2301.03313` — BQ-NCO: Bisimulation Quotienting for Neural CO (NeurIPS 2023)
- `https://arxiv.org/abs/2301.13279` — HybridNet: Coordination Policies over Heterogeneous Graphs for Human-Robot Teams (IROS 2022)
- `https://arxiv.org/abs/2302.02506` — Generating Dispatching Rules for Interrupting Swap-Allowed Blocking JSSP
- `https://arxiv.org/abs/2302.11665` — AlpaServe: Statistical Multiplexing with Model Parallelism (OSDI 2023)
- `https://arxiv.org/abs/2305.05119` — Flexible Job Shop Scheduling via Dual Attention Network (DAN, IEEE TNNLS 2024)
- `https://arxiv.org/abs/2306.02689` — Equity-Transformer: Min-Max Routing via Sequential Generation (AAAI 2024)
- `https://arxiv.org/abs/2306.17100` — RL4CO: Extensive RL for Combinatorial Optimization Benchmark (KDD 2025)
- `https://arxiv.org/abs/2308.15762` — Hanayo: Wave-like Pipeline Parallelism (SC 2023)
- `https://arxiv.org/abs/2309.15517` — Residual Scheduling: A New RL Approach for Job Shop Scheduling
- `https://arxiv.org/abs/2310.07985` — Neural CO with Heavy Decoder (LEHD, NeurIPS 2023)
- `https://arxiv.org/abs/2401.06952` — RL for Scalable Train Timetable Rescheduling with Graph Representation
- `https://arxiv.org/abs/2401.10241` — Zero Bubble Pipeline Parallelism (ICLR 2024 Oral)
- `https://arxiv.org/abs/2401.11849` — Self-Labeling the Job Shop Scheduling Problem (SLIM, NeurIPS 2024)
- `https://arxiv.org/abs/2402.08979` — Learning-Enabled Flexible Job-Shop Scheduling with Transportation (HGS)
- `https://arxiv.org/abs/2405.01906` — Instance-Conditioned Adaptation for Large-Scale Routing (ICAM, IEEE TITS)
- `https://arxiv.org/abs/2406.14096` — Graph Neural Networks for Job Shop Scheduling Problems: A Survey (C&OR 2024)
- `https://arxiv.org/abs/2409.13571` — Scalable MARL for Factory-wide Dynamic Scheduling
- `https://arxiv.org/abs/2409.13620` — Subassembly to Full Assembly: Effective Assembly Sequence Planning via Graph RL (S2A)
- `https://arxiv.org/abs/2502.12617` — Graph-Enhanced DRL Framework for Aircraft Landing Problem
- `https://arxiv.org/abs/2502.15791` — Learning-Guided Rolling Horizon Optimization for Long-Horizon FJSP (L-RHO, ICLR 2025)
- `https://arxiv.org/abs/2503.07580` — BOPO: Best-Anchored and Objective-Guided Preference Optimization (ICML 2025)
- `https://arxiv.org/abs/2506.00260` — GrapheonRL: Constraint and Data-Aware Workflow Mapping & Scheduling (COMPSAC 2025)
- `https://arxiv.org/abs/2507.16635` — Multi-Agent Action Masked DRL for Industrial Assembly Line Balancing (J. Autom. Intell. 2025)
- `https://arxiv.org/abs/2507.18748` — PPipe: Video Analytics Serving via Pool-Based Pipeline Parallelism (USENIX ATC 2025)
- `https://arxiv.org/abs/2509.23722` — OctoPipe: Co-Optimizing Partitioning, Placement, and Scheduling (SC 2026)
- `https://arxiv.org/abs/2511.13214` — Wheatley: RCPSP with Duration Uncertainty using GNNs (ICTAI 2025)
- `https://arxiv.org/abs/2602.08052` — Graph-Enhanced DRL for Multi-Objective Unrelated Parallel Machine Scheduling

**Open-Access Proceedings PDFs (6 papers)**:
- `https://people.csail.mit.edu/hongzi/content/publications/Decima-Sigcomm19.pdf` (Decima, SIGCOMM 2019)
- `https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf` (L2D, NeurIPS 2020)
- `https://proceedings.neurips.cc` (HM-MATAS / TARGETNET, NeurIPS 2025, `Heterogeneous Graph Transformers for Simultaneous Mobile Multi-Robot Task Allocation and Scheduling`)
- `https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1238.pdf` (2D-Ptr, AAMAS 2024)
- `https://ojs.aaai.org/index.php/AAAI/article/view/29915` (`14546-SmitI.pdf`, AAAI 2025)
- `https://roboticsproceedings.org/rss16/p094.pdf` (Wang & Gombolay, RSS 2020)
