# APAL 16-Module Paper Terminology Audit & Figure Replacement / Integration Roadmap (`refined_v8`)

Ground-truth references audited (read-only):
- English IEEE TSMC manuscript: `docs/APAL_HGP_PPO_Journal_Manuscript/01_Manuscript_TSMC_English_10p/main_tsmc.tex`
- Chinese manuscript: `docs/APAL_HGP_PPO_Journal_Manuscript/02_Manuscript_Chinese_12p/main_12p.tex`
- Model implementation: `models/hb_gat_pn.py`
- Existing manuscript figures inspected: `docs/APAL_HGP_PPO_Journal_Manuscript/04_Figures_and_Assets/fig1.png`, `fig2.png`

---

## Part 1 — Forensic Audit of All 16 Modules Against `main_tsmc.tex`

### A. Seven Core Method Figures — Upgraded & Verified 100% Paper-Faithful in `refined_v8`

| Slide | Module | Non-Paper / Problematic Text Caught in `v7` | Fixed Paper-Faithful Text in `v8` | Ground Truth in `main_tsmc.tex` |
|---:|---|---|---|---|
| **06** | **B1** | `K₂ Trade Skill Hub` / `x_K: 5 trades, count, mean η`; edge `assigned`; `Skilled Worker`; `load L` | **`K₂ Skill Node (\|K\|=5)`** / `x_K: skill ID, count, cap., mean η`; edge **`assigned-to`**; **`Multi-Skilled Worker`** (`η_w, 5-d skills, wait, idle, ℓ_w`); `load L_m`; `p̃_i, status, k_i, r_i` | Tables I, II, III (`Skill`, `assigned-to`, `L_m, c_m, p̃_i, k_i, r_i, η_w, ℓ_w`). Note: `"Skill Hub"` only appeared in the Chinese draft (`main_12p.tex`), never `"Trade Skill Hub"` in English TSMC. |
| **08** | **B3** | Internal ID `Type-wise Readout (B4)`; PyG code kwargs `concat = False`, `aggr = sum`, `GATv2Conv`; redundant `(sum)` | **`Node-Type Attention Pooling`**; **`(H = 4 heads, head-averaged, relation-summed)`**; **`Head 1 (expanded GATv2 attention)`**; **`Multi-Head Mean (H = 4)`**; `Σ_{r∈ℛ} m_3^(r,ℓ)` | Section IV-B (`Node-type attention pooling`, `L = 5`, `H = 4`, relation-specific GATv2) |
| **10** | **B5** | `Policy π_θ(a_t\|s_t)`; `Type-wise Attention Pool`; informal `once`; `V_O ∪ V_M ∪ V_W ∪ V_K` | **`Policy π_θ(ã_t\|s_t)`**; **`Node-Type Attention Pooling [M \| O \| W]`**; **`Node Type Projection f_X (d = 128)`**; **`G_t = (O ∪ M ∪ W ∪ K, E_t)`**; `Op/St/Wk Ptr`, `Per-Relation Softmax` | Section IV-B line 247 (`V_t = O ∪ M ∪ W ∪ K`), line 309, Section IV-C line 321 (`π_θ(ã_t\|s_t)`) |
| **11** | **B6** | Non-paper `Feasible Set: A_O(s_t)`; programming syntax `−1e4`; borrowed ReSched term `O2O Mask (5×5)`; informal `Counter-example:` | **`Feasible Set: F_t^O = {O_3, O_4}`**; math notation **`−10^4`**; APAL term **`Precedence (5×5)`**; **`Mask Effect:`**; `P(i) done ∧ released ∧ (M, W) feasible`; `h(O_3), k_3, q_3 → St. Ptr` | Section IV-C lines 328–332 (`\mathcal{F}_t^O`), Appendix A line 690 (`-10^4`), Table I (`\mathcal{P}(i), k_i, q_i`) |
| **13** | **B8** | Sparse floating upper arch/title band above dotted `× q` frame; mixed `SoftMax`; informal `Feats` | Relocated **`(a) Substep r = 1: Select W₂ — [Shared Worker Pointer (r = 1 → 2)] → (b) Substep r = 2: Select W₅`** cleanly below the dotted frame; placed **`w_t^(1) = W₂`** badge directly in the center pocket at circle level eliminating top emptiness; standardized `Substep r`, `Masked Softmax`, `3. Dynamic State` | Section IV-C line 343 (`substep r`), Appendix A (`Partial-team pooling`, `Dynamic EFT`, `Marginal skill scarcity`) |
| **15** | **B10** | Awkward `Hidden Unit j (illustrative negative-return trace)`; non-paper `Skill hubs K (5)` / `folded in`; invented `.12–.31 (5 st.)`; mislabeled `GAE δ_t` | Formal module title **`Cross-Type Feature Fusion & Activation (Channel j)`**; **`Skill nodes K (5)` → `encoded in H^O, H^W`**; **`5 physical stations`** & **`multi-skilled pool`**; mathematically exact **`TD Error δ_t`**; `(a) Attention Pooling` | Section IV-B (`Node-type attention pooling`, `Skill nodes`), Section IV-D line 361–363 (`temporal-difference error δ_t`), Table IV (`283–3,182`) |
| **16** | **B11** | **Symbol collision** `ρ_t` (collides with Eq. 1 team coordination discount `ρ = 0.95`!); **internal debug leak** `(ckpt: per-head clip)`; **wrong hardcoded table number** `Table VII Config:` (actually Table XII in `main_tsmc.tex`!); `θ_old`, `L_π(θ)`, `TD Residual` | Exact paper notation **`r_t^PPO(θ)`**, **`θ^−`**, **`L_policy(θ)`**, **`TD Error`**, **`with entropy bonus −c_H H(π_θ)`**, and table-independent **`PPO Hyperparameters:`** | Section IV-D lines 361–372 (`r_t^{PPO}(θ)`, `θ^-`, `L_{policy}(θ)`, `TD error δ_t`), Appendix B Table XII (`tab:reproducibility`) |

### B. Nine Carried-Forward Data/Mechanism Modules (`A1, A2, A4–A6, B2, B4, B7, B9`) — Audit Verdict

All 9 carried-forward slides are **mathematically and terminologically consistent** with `main_tsmc.tex` (verified exact match to Proposition 1, Eq. 1–9, Section III-D, Tables I–III). However, they still retain **review-slide header/footer chrome** (since they have not yet been converted to pure publication plates). Before embedding any of them directly into a journal PDF, apply these quick cleanups:

| Slide | Module | Publication Readiness | Quick Cleanup Needed Only If Placed Directly in Paper |
|---:|---|---|---|
| **01** | **A1** | Terminology accurate (`2338`, `K0/K2/K4`, `standard duration`) | Strip slide header/footer & internal filename `A1_source.csv`; replace Chinese aerospace shorthand `"Direct successor AO"` → `"Direct successor operation"`. |
| **02** | **A2** | Exact match to Proposition 1 `[(k+m-1)H, (k+m)H)` | Strip slide header/footer & `real_283_schedule.csv` filename. |
| **03** | **A4** | Exact match (`283 physical operations`, `H^0 = 292.57 h`) | Strip slide chrome, cross-ref `"matching A5"`, and `.csv` filenames. |
| **04** | **A5** | Exact match to Section III-D (`τ^r, B_i^0, Δ_i, r_i, B_i^r`) | Strip slide chrome, `"as A4"`, and `.csv` filenames. |
| **05** | **A6** | Exact match to Eq. 1 & Eq. 9 (`p^std, D, B, C, c_5`) | Strip slide chrome & `.csv` filenames. |
| **07** | **B2** | Exact match to Table III (`precedes, assigned-to, done-by, requires, has-skill`) | Strip slide chrome & `.csv` filenames. |
| **09** | **B4** | Exact match to Section IV-B (`β^X, g_M/g_O/g_W, Skill nodes (K)`) | Strip slide chrome & bottom note (move illustrative note to LaTeX caption). |
| **12** | **B7** | Exact match to Section IV-C (`a_M^T tanh(W_OM h + W_M h_m)`) | Strip slide chrome & code footer `StationSelector in models/hb_gat_pn.py (rescheduling BIC ...)` (`BIC` is never defined in `main_tsmc.tex`). |
| **14** | **B9** | Exact match to Eq. 9 (`B_3 = max(...)`, `D_3(W_t)`, `c_3`) | Strip slide chrome & bottom note (move illustrative note to LaTeX caption). |

---

## Part 2 — Critical Defects Found in the Manuscript's Existing `fig1.png` and `fig2.png`

Inspecting `docs/APAL_HGP_PPO_Journal_Manuscript/04_Figures_and_Assets/fig1.png` and `fig2.png` revealed **severe publication-blocking errors** that MUST be fixed before IEEE TSMC submission:

1. **Existing `fig1.png` (Problem & Rescheduling Overview)**:
   - **4 spelling typos**: `"Precedence contraints"`, `"Task-delay distuption"`, `"Reschedule remaining opetaions"`, `"Woraker efficiency"`.
   - **Internal data contradictions**: Left precedence box labels **both** `APCJ00-0110` and `APCJ00-0120` as **`P1`**; `P3 (APCJ00-0140)` is shown as `Skill3, demand=2` in top-left but `Skill4, demand=3` in the middle panel.
   - **Symbol mismatch**: Uses `t_r` instead of the manuscript's freeze-boundary symbol **`τ^r`**.
2. **Existing `fig2.png` (Overall HGP-PPO Framework)**:
   - **Wrong method name**: Banner says **`HB-GAT-PPO`** instead of the paper's name **`HGP-PPO`**.
   - **Copy-paste mislabeling in Critic branch**: Bottom purple Critic enclosure is mislabeled **`"Actor Branch"`** at bottom-left, AND its pooled context box says **`"Actor Context"`** instead of **`"Critic Context"`**.
   - **Wrong layer name**: Encoder blocks say **`GATConv Layer L`** instead of **GATv2** (`main_tsmc.tex` Section IV-B cites Brody et al., ICLR 2022 GATv2).
   - **Generic iconography**: Uses black-box neural icons that do not illustrate masking, pointer scoring, or autoregressive team construction.

---

## Part 3 — Recommended Module Replacement & Integration Plan for the Paper

Because `main_tsmc.tex` is a tight **10-page IEEE TSMC double-column manuscript** (currently containing 5 figures: `Fig. 1` problem overview, `Fig. 2` overall framework, `Fig. 3` initial/ablation, `Fig. 4` rescheduling heatmap, `Fig. 5` runtime), we recommend a **surgical, zero-bloat replacement strategy** — keeping high-value standalone modules standalone and only integrating where a single double-column figure tells a unified story:

### Tier 1 — High-Priority Direct Replacements in `main_tsmc.tex` (Must-Do)
1. **Replace `Fig. 1` (Section I, Problem & Rescheduling Illustration)**:
   - **Recommended Integrated 2-Panel Composite (`A2 + A5`, or 3-Panel `A2 + A1_mini + A5`)**:
     - **(a) Cyclic Pulsating Flow (`A2`)**: Shows the 5 physical stations, Proposition 1 cyclic unfolding `[(k+m-1)H, (k+m)H)`, and 5 concurrent aircraft — directly grounding the caption's first half with real data instead of clip-art.
     - **(b) Material Delay Disruption & Stability-Aware Repair (`A5`)**: Shows baseline `B_i^0`, freeze boundary `τ^r`, direct delay `Δ_i`, release `r_i`, and repaired start `B_i^r` on real instance `283` — directly grounding the caption's second half with zero typos and exact `τ^r` notation.
   - *Alternative if you want real shop-floor visual impact in Fig. 1*: Put **`B1` (Workcell Photo → Heterogeneous State Graph `G_t`)** as **Fig. 1** (or **Fig. 2a**) to hook reviewers visually on Page 1–2!
2. **Replace `Fig. 2` (Section IV-A, Overall HGP-PPO Framework)**:
   - **Recommended Option 2A (Best Balance for 10p IEEE TSMC — Clean 2-Panel Integration `B1 + B5`)**:
     - **(a) Workcell-to-Graph Representation (`B1`)** on the left (or top) + **(b) Independent Actor–Critic Dual-Tower Architecture (`B5`)** on the right (or bottom).
     - Together they replace 100% of the buggy `fig2.png`, visually showcase **both** Contribution 2 (4-type heterogeneous graph from physical workcell) and the independent Actor–Critic GATv2 + 3-tier pointer + PPO coupling pipeline with zero generic neural-net icons!
   - **Recommended Option 2B (Keep Standalone — Cleanest Readability)**:
     - Make **`B1`** standalone **Fig. 2** (Section IV-B, Heterogeneous Graph Representation) and **`B5`** standalone **Fig. 3** (Section IV-A/B, Independent Actor–Critic Architecture), avoiding cramming two rich figures into one cramped row.
3. **Add `B8` Standalone in Section IV-C or Appendix A (`\label{app:worker_pointer}`) — HIGHEST ROI NEW FIGURE!**
   - **Do NOT compress `B8` into a composite!** `B8` (Autoregressive Worker Pointer showing `r = 1 → 2` state transition, `W_2 → used` mask update, Sum·Mean·Max partial-team pooling, and dynamic EFT / marginal skill scarcity) directly visualizes **Contribution 3** (the paper's core algorithmic novelty!) and Appendix A currently has **zero figures** alongside 11 dense equations. Placing `B8` standalone at the top of Appendix A (or Section IV-C) fits the existing page layout cleanly and elevates reviewer comprehension dramatically.

### Tier 2 — Optional Standalone Method Figures (Use for 12p Chinese Draft, Extended Journal Revision, or Thesis)
- **`B6` (Operation Pointer with Masking, ReSched style)**: Best kept **standalone** (`12.2 in` wide, high information density) in Section IV-C or thesis Chapter 3; integrating it with `B7`/`B8` in one figure would shrink text below IEEE 7pt minimum.
- **`B3` (Expanded 4-Head Hetero-GATv2 Layer)**: Best kept **standalone** in Section IV-B (or Supplementary Material) to detail GATv2 head averaging and relation summation.
- **`B10` (Critic Value Head)** & **`B11` (PPO Rollout & GAE Update)**: Best kept **standalone** in Appendix / Supplementary / Thesis (or paired as a 2-part supplementary figure `Fig. S1(a)-(b)` explaining Critic & PPO training details).
- **`A6` (Binding Constraint Breakdown)** & **`B9` (Forward Allocation Engine)**: Ideal standalone figures for thesis problem-modeling chapter or reviewer rebuttal clarifying deterministic forward time allocation.
