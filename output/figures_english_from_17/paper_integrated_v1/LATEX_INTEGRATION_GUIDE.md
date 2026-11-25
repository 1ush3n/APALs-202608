# APAL HGP-PPO Paper-Integrated Figures v1 — LaTeX Integration & Review Guide

All deliverables in this folder (`D:/APAL-Dynamic-v4/output/figures_english_from_17/paper_integrated_v1/`) are strictly isolated and non-destructive. Existing manuscript files (`main_tsmc.tex`), old assets (`04_Figures_and_Assets/`), and module review decks (`refined_v8/`) remain completely untouched.

---

## 1. File Inventory & Exact Print Metrics (IEEEtran Double-Column `\textwidth = 7.16 in`)

| # | Slug (PDF in `pdf/`, 600 DPI PNG in `png_600dpi/`) | Role | Tight PDF Size (in) | Aspect Ratio | Height @ `7.16 in` (`\textheight` %) | Min Font (PPT / Print @ `7.16 in`) |
|---|---|---|---|---|---|---|
| 1 | `fig1_apal_unfolding_delay` | **New Fig. 1 (`A2 + A5`)** — replaces buggy `fig1.png` | `13.04 × 5.86` | `2.22 : 1` | `3.22 in` (`34.8%`) | `8.8 pt` / `~4.8 pt` |
| 2 | `fig2_compact_b1_b5` | **Option A Compact Fig. 2(a)–(b) (`B1 + B5`)** | `13.04 × 6.07` | `2.15 : 1` | `3.33 in` (`36.0%`) | `8.6 pt` / `~4.7 pt` |
| 3 | `fig2_standalone_b1` | **Option B High-Clarity Standalone Fig. 2 (`B1`)** | `12.36 × 5.84` | `2.12 : 1` | `3.38 in` (`36.6%`) | `9.6 pt` / `~5.6 pt` |
| 4 | `fig3_standalone_b5` | **Option B High-Clarity Standalone Fig. 3 (`B5`)** | `9.13 × 7.19` | `1.27 : 1` | `5.65 in` (`61.0%`; `4.63 in` @ `0.82\textwidth`) | `10.8 pt` / `~8.5 pt` |
| 5 | `fig_appA_b8_worker_pointer` | **New Standalone Method Fig. (`B8`)** — Appendix A / Sec. IV-C | `10.48 × 5.48` | `1.91 : 1` | `3.75 in` (`40.5%`) | `9.2 pt` / `~6.3 pt` |
| 6 | `extra_b3_hetero_gatv2` | Backup standalone (`B3`) — Relation-specific GATv2 Layer | `13.01 × 6.52` | `1.99 : 1` | `3.59 in` (`38.8%`) | `9.8 pt` / `~5.4 pt` |
| 7 | `extra_b6_op_pointer` | Backup standalone (`B6`) — Operation Pointer & Masking | `12.54 × 6.53` | `1.92 : 1` | `3.73 in` (`40.3%`) | `9.0 pt` / `~5.1 pt` |
| 8 | `extra_b10_critic_head` | Backup standalone (`B10`) — Critic Attention Pooling & Value Head | `12.45 × 4.81` | `2.59 : 1` | `2.77 in` (`29.9%`) | `9.8 pt` / `~5.6 pt` |
| 9 | `extra_b11_ppo_gae` | Backup standalone (`B11`) — PPO Replay, GAE & Dual Updates | `12.69 × 6.41` | `1.98 : 1` | `3.62 in` (`39.1%`) | `9.2 pt` / `~5.2 pt` |

> **Recommendation for IEEE LaTeX (`pdflatex` / `xelatex`)**: Always prefer the tight-cropped lossless **vector PDF** (`pdf/*.pdf`) over PNG so all equations, symbols, and node borders render at infinite resolution in the published PDF. The 600 DPI PNGs (`png_600dpi/*.png`) are provided for Word/Overleaf quick inspection or journal submission portals requesting raster fallback.

---

## 2. Visual Review Sheets

- **Before vs. After (`fig1.png`/`fig2.png`)**: `review_sheets/01_old_vs_new_fig1_and_fig2.png`
- **IEEE Double-Column (`7.16 in`) True-to-Scale Page Simulation**: `review_sheets/02_compact_vs_standalone_ieee_page_sim.png`
- **All 9 Tight-Cropped Exports Gallery**: `review_sheets/03_all_9_figures_gallery.png`
- **Editable Native Source Deck**: `APAL_Paper_Integrated_Figures_v1.pptx` (100% native PowerPoint vector shapes & rich text, Arial + Cambria, 0 CJK)

---

## 3. Copy-Paste Ready IEEE TSMC LaTeX Snippets

### 3.1 New Fig. 1 (`A2 + A5` Integrated — Replaces `main_tsmc.tex` lines 55–60)

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig1_apal_unfolding_delay.pdf}
\caption{Cyclic pulsating operation scheduling and stability-aware rescheduling on a five-station aircraft pulsating assembly line (APAL). (a) Five-station pulsating cyclic unfolding (Proposition~1): tracking Aircraft~$k$ along the navy diagonal traverses Stations $S_1$--$S_5$ across Pulses~1--5 in unfolded window $[(k+m-1)H,(k+m)H)$, while reading down Pulse~5 illustrates five distinct aircraft processed concurrently with verified operations and multi-skilled worker teams active at offset $t=0.30\,\mathrm{h}$. (b) Real-data material delay disruption on the 283-operation factory line triggered at freeze boundary $\tau^r=65.83\,\mathrm{h}$: 67 committed prefix operations ($t \le \tau^r$) retain baseline plans and worker bindings, while 216 remaining operations are reschedulable (22 directly delayed). Markers trace baseline start $B_i^0$ (circle), direct material delay $\Delta_i$ to release time $r_i$ (orange bar and diamond), and downstream resource propagation wait to repaired start $B_i^r$ (green bar and square), shifting cycle span from $H^0=292.57\,\mathrm{h}$ to $H^r=305.85\,\mathrm{h}$.}
\label{fig:1}
\end{figure*}
```

---

### 3.2 Option A (Compact Space-Saving): Single Fig. 2(a)–(b) (`B1 + B5` Seamless Fusion — Replaces `main_tsmc.tex` lines 238–243)

Best if the 10-page IEEE TSMC page budget is tight (occupies a single `~0.34\textheight` double-column banner, preserving existing `\label{fig:3}`–`\label{fig:5}` numbering).

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig2_compact_b1_b5.pdf}
\caption{Overall HGP-PPO framework for APAL scheduling and stability-aware rescheduling. (a) Mapping the physical APAL workcell (assembly floor photograph adapted from Edward Russell, Wikimedia Commons, CC~BY~2.0) to the four-type heterogeneous state graph $G_t=(\mathcal{O}\cup\mathcal{M}\cup\mathcal{W}\cup\mathcal{K},\mathcal{E}_t)$ with raw node features $\mathbf{x}_v$ (Tables~\ref{tab:node_features}--\ref{tab:relations}). (b) Independent dual-tower Actor ($\theta$) and Critic ($\phi$) architectures ($\theta\cap\phi=\emptyset$), each employing node-type projection $f_X$ ($d=128$), five relation-specific 4-head GATv2 encoder layers with post-norm residual updates, and node-type attention pooling $[\mathcal{M}\mid\mathcal{O}\mid\mathcal{W}]$. The Actor decodes an executable joint action $\tilde{a}_t=(i_t,m_t,W_t)$ through a three-tier autoregressive pointer policy $\pi_\theta(\tilde{a}_t\mid s_t)$, while the Critic estimates $V_\phi(s_t)$ to supply GAE advantages $\hat{A}_t$ for PPO policy updates.}
\label{fig:2}
\end{figure*}
```

---

### 3.3 Option B (High-Clarity Split — Recommended if Page Budget Allows): Standalone Fig. 2 (`B1`) + Standalone Fig. 3 (`B5`)

Maximizes printed font legibility (`5.6 pt` min on `B1`, `8.5 pt` min on `B5`!). Note that existing experimental figures (`\label{fig:3}`–`\label{fig:5}`) automatically renumber via LaTeX `\ref{...}` with no manual edits needed.

#### Standalone Fig. 2 (`B1` — Place in Section IV-B near Tables II–III)

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.98\textwidth]{figures/fig2_standalone_b1.pdf}
\caption{From the physical APAL workcell to the four-type heterogeneous state graph $G_t=(\mathcal{O}\cup\mathcal{M}\cup\mathcal{W}\cup\mathcal{K},\mathcal{E}_t)$ (Tables~\ref{tab:node_features}--\ref{tab:relations}). Left: real assembly floor workcell (photograph adapted from Edward Russell, Wikimedia Commons, CC~BY~2.0) annotated with representative Physical Station ($S_3$), Operation ($O_3$), Skill Node ($K_2$, $|\mathcal{K}|=5$), and Multi-Skilled Worker ($W_5$) entities and their state features $\mathbf{x}_v$ (operation features expand from 18-d to 24-d during rescheduling). Right: induced heterogeneous graph topology with static relations (\texttt{precedes}, \texttt{requires}, \texttt{has-skill}) and dynamic assignment edges (\texttt{assigned-to}), projected via $f_X(\mathbf{x}_v)\in\mathbb{R}^{128}$ into initial latent node embeddings $\mathbf{h}_v^{(0)}$.}
\label{fig:2}
\end{figure*}
```

#### Standalone Fig. 3 (`B5` — Place in Section IV-A / IV-C)

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.82\textwidth]{figures/fig3_standalone_b5.pdf}
\caption{Independent dual-tower HGP-PPO architecture with disjoint Actor ($\theta$) and Critic ($\phi$) parameters ($\theta\cap\phi=\emptyset$). Both towers project the shared heterogeneous state graph $G_t=(\mathcal{O}\cup\mathcal{M}\cup\mathcal{W}\cup\mathcal{K},\mathcal{E}_t)$ via $f_X$ ($d=128$) into five stacked 4-head relation-specific GATv2 layers with multi-head mean aggregation, relation summation, and post-norm residual connections, followed by node-type attention pooling $[\mathcal{M}\mid\mathcal{O}\mid\mathcal{W}]$ producing $\mathbf{g}_\theta,\mathbf{g}_\phi\in\mathbb{R}^{3d}$. The Actor sequentially decodes $(i_t,m_t,W_t)$ via three autoregressive pointer heads conditioned on $\mathbf{H}_\theta^{(5)}$, while the Critic MLP estimates $V_\phi(s_t)$ to compute GAE advantages $\hat{A}_t$ for PPO coupling.}
\label{fig:hgp_arch}
\end{figure*}
```

---

### 3.4 New Standalone Method Figure (`B8` — Default Appendix A `\label{app:worker_pointer}`, Movable to Section IV-C Anytime)

Because the figure canvas carries no hardcoded `Fig. X` number, this exact PDF can sit in Appendix A (`\label{app:worker_pointer}`, line 661) or be promoted directly into Section IV-C without touching the artwork.

```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{figures/fig_appA_b8_worker_pointer.pdf}
\caption{Autoregressive worker-team pointer unrolling across substeps $r=1\to 2$ for a two-worker operation $O_3$ assigned to station $S_3$ ($q_3=2$). In substep~$r=1$, the shared pointer evaluates all candidate workers under exact skill and station-binding feasibility masking ($-10^4$) and selects $w_t^{(1)}=W_2$. Between substeps, the causal state transition (center bridge) updates three coupled signals: (1)~self-masking $W_2\to\text{used}$ to prevent duplicate selection, (2)~partial-team pooling ($\text{Sum}\cdot\text{Mean}\cdot\text{Max}$) into $\mathbf{h}_{\mathrm{team}}\leftarrow\{W_2\}$, and (3)~recomputing candidate dynamic earliest finish time (EFT) and marginal skill scarcity features. In substep~$r=2$, the shared pointer re-scores the remaining eligible workers under the updated query $\mathbf{q}^{(2)}$ and selects $W_5$ to complete team $W_t=\{W_2,W_5\}$.}
\label{fig:worker_pointer}
\end{figure*}
```

---

## 4. Academic & Copyright Compliance Checklist

- **Photo License (CC BY 2.0)**: Any figure containing the Boeing 787 final assembly floor photograph (`fig2_compact_b1_b5` or `fig2_standalone_b1`) **must retain** the caption attribution:
  > *Assembly floor photograph adapted from Edward Russell (Wikimedia Commons, CC BY 2.0).*
  > Source URL: `https://commons.wikimedia.org/wiki/File:Final_assembly_of_the_2nd_787-10_(32910009961).jpg`
- **Terminology & Notation Audit**: 9/9 slides verified 100% aligned with `main_tsmc.tex` (`HGP-PPO`, `Skill Node`, exact Table III relation names `precedes`, `assigned-to`, `requires`, `has-skill`, exact symbols $\tau^r, B_i^0, r_i, B_i^r, \Delta_i, H^0, \mathcal{F}_t^O, -10^4, r_t^{\mathrm{PPO}}, \theta^-, \mathcal{L}_{\mathrm{policy}}, \delta_t$), **0 CJK characters**, **0 `.csv` filenames**, **0 code kwargs or internal slide IDs**.
