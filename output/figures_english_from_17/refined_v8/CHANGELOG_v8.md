# APAL 16-Module Figures — `refined_v8` Changelog

Built cleanly on top of saved milestone `refined_v7` (`APAL_Modules_English_Refined_v7.pptx` untouched) after a complete forensic terminology, notation, and publication-standard audit against `main_tsmc.tex`, `main_12p.tex`, and `models/hb_gat_pn.py`.

## Summary of All 7 Upgraded Core Figures (`B1, B3, B5, B6, B8, B10, B11`)

1. **Slide 6 (`B1`)**:
   - Replaced non-paper `K₂ Trade Skill Hub / x_K: 5 trades, count, mean η` with exact `main_tsmc.tex` Table II terminology: **`K₂ Skill Node (|K|=5)`** / **`x_K: skill ID, count, cap., mean η`**.
   - Aligned remaining callouts with Tables I–II: `S₃ Physical Station (load L_m, ratios, wait, c_m slots)`, `O₃ Operation (18d/24d: p̃_i, status, k_i, r_i)`, `W₅ Multi-Skilled Worker (η_w, 5-d skills, wait, idle, ℓ_w)`.
   - Fixed relation label `assigned` → **`assigned-to`** (exact Table III relation name) and cleanly separated `assigned-to` from `open station`.
2. **Slide 8 (`B3`)**:
   - Stripped internal slide reference `(B4)` → **`Node-Type Attention Pooling`** (exact Section IV-B term).
   - Stripped all PyTorch Geometric code identifiers (`concat = False`, `aggr = sum`, `GATv2Conv`) → **`(H = 4 heads, head-averaged, relation-summed)`**, **`Head 1 (expanded GATv2 attention)`**, **`Multi-Head Mean (H = 4)`**, and clean **`Relation Sum / Σ_{r∈ℛ} m_3^(r,ℓ)`**.
3. **Slide 10 (`B5`)**:
   - Aligned with Section IV-B & IV-C verbatim: **`Policy π_θ(ã_t|s_t)`** (ordered trajectory action), **`Node-Type Attention Pooling [M | O | W]`**, **`Node Type Projection f_X (d = 128)`** (removed informal `"once"`), **`Shared Heterogeneous State Graph G_t = (O ∪ M ∪ W ∪ K, E_t)`**, and consistent `Op/St/Wk Ptr`, `Per-Relation Softmax`.
4. **Slide 11 (`B6`)**:
   - Fixed feasible set symbol `A_O(s_t)` → **`F_t^O = {O_3, O_4}`** (Section IV-C line 328).
   - Replaced programming literal `−1e4` → mathematical **`−10^4`** (Appendix A line 690).
   - Replaced borrowed ReSched label `O2O Mask (5×5)` → APAL **`Precedence (5×5)`**, informal `Counter-example:` → **`Mask Effect:`**, and aligned bottom symbols to `P(i), k_3, q_3 → St. Ptr`.
5. **Slide 13 (`B8`)**:
   - **Major layout redesign solving both pinned upper-sparseness comments**: relocated `(a) Substep r = 1: Select W₂ — [Shared Worker Pointer (r = 1 → 2)] → (b) Substep r = 2: Select W₅` cleanly beneath the dotted `× q` frame with symmetric dashed navy stems.
   - Eliminated empty upper headroom and duplicate `After Draw 1` wording by placing **`w_t^(1) = W₂`** directly in the center pocket at circle height between Substeps 1 and 2, hugging the amber feedback wire tightly above Substep 1 probabilities, and aligning the 4-card bridge (`State Transition`, `1. Mask Update`, `2. Team Pooling`, `3. Dynamic State`).
6. **Slide 15 (`B10`)**:
   - Replaced weird `Hidden Unit j (illustrative negative-return trace)` with formal IEEE module title **`Cross-Type Feature Fusion & Activation (Channel j)`**.
   - Replaced non-paper `Skill hubs K (5) / folded in` → **`Skill nodes K (5) → encoded in H^O, H^W`**; replaced invented `.12–.31 (5 st.)` → **`5 physical stations`** and `skill-aware` → **`multi-skilled pool`**.
   - Fixed mathematical label `GAE δ_t` → exact **`TD Error δ_t`** (Section IV-D lines 361–363).
7. **Slide 16 (`B11`)**:
   - Resolved symbol collision with Eq. (1) coordination discount `ρ = 0.95` by adopting exact paper notation **`r_t^PPO(θ)`**, **`θ^−`**, **`L_policy(θ)`**, **`TD Error`**, and **`with entropy bonus −c_H H(π_θ)`**.
   - Removed internal debug leak `(ckpt: per-head clip)`.
   - Replaced wrong hardcoded **`Table VII Config:`** (actually Table XII in `main_tsmc.tex`) with table-independent **`PPO Hyperparameters:`**, and raised minor legend fonts to `>= 9.2pt`.
