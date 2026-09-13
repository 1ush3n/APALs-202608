"""
Generate Figure 6: Rescheduling Main Results and 36-Scenario Paired Difference Heatmaps.

Figure structure:
  (a) Top row: Instance-level and overall average composite score S
      (5 panels: real_283, real_680, real_2338, real_3182, and 4-Instance Mean).
  (b) Bottom rows: 2 x 4 Paired Difference Heatmaps (d = S_HGP - S_baseline).
      Row 1: vs. L2D-PPO-APAL (Win/Tie/Loss: 19 / 0 / 17)
      Row 2: vs. StabilityAwareRepair (Win/Tie/Loss: 22 / 0 / 14)
      Columns: 4 physical instances.
      Each subpanel: 3x3 grid (Freeze Phase: Early, Middle, Late x Delay Severity: Low, Medium, High).

Outputs:
  - fig6_reschedule_scenarios_heatmap.pdf (vector)
  - fig6_reschedule_scenarios_heatmap.svg (editable vector)
  - fig6_reschedule_scenarios_heatmap.png (300 DPI preview)
  - fig6_reschedule_scenarios_heatmap_source.csv (complete raw source data)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_statistical_metrics import (
    load_fullx_bic,
    load_l2d,
    load_rules,
    INSTANCES,
    SCENARIO_ORDER,
)

FIGURES_DIR = PROJECT_ROOT / "docs" / "APAL_HGP_PPO_Journal_Manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Style Configuration
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
})

COLORS = {
    "HGP-PPO": "#0072B2",          # Deep Blue
    "L2D-PPO": "#D55E00",          # Vermilion / Deep Orange
    "StabilityAware": "#CC79A7",   # Purple / Reddish Purple
}

MARKERS = {
    "HGP-PPO": "o",
    "L2D-PPO": "s",
    "StabilityAware": "^",
}

INSTANCE_NAMES = {
    "real_283": "Instance 283\n(Below Training Scale)",
    "real_680": "Instance 680\n(Within Training Scale)",
    "real_2338": "Instance 2338\n(Above Training Scale)",
    "real_3182": "Instance 3182\n(Above Training Scale)",
}

SEVERITY_ORDER = ["low", "medium", "high"]
STAGE_ORDER = ["early", "middle", "late"]


def export_bundle(fig, base_name: str, source_df: pd.DataFrame):
    """Export PDF, SVG, PNG, and source CSV."""
    png_path = FIGURES_DIR / f"{base_name}.png"
    pdf_path = FIGURES_DIR / f"{base_name}.pdf"
    svg_path = FIGURES_DIR / f"{base_name}.svg"
    csv_path = FIGURES_DIR / f"{base_name}_source.csv"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    source_df.to_csv(csv_path, index=False)

    print(f"[Export] Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}\n  {csv_path}")


def main():
    print("=" * 70)
    print("Generating Figure 6: Rescheduling Results and 36-Scenario Heatmap")
    print("=" * 70)

    # 1. Load Data
    hgp = load_fullx_bic()
    l2d = load_l2d()
    sta = load_rules("StabilityAwareRepair")

    # Prepare Source DataFrame records
    source_records = []

    # Prepare arrays for heatmap: shape (4_instances, 3_severities, 3_stages)
    diff_l2d = np.zeros((len(INSTANCES), len(SEVERITY_ORDER), len(STAGE_ORDER)))
    diff_sta = np.zeros((len(INSTANCES), len(SEVERITY_ORDER), len(STAGE_ORDER)))

    # Instance-level means
    inst_means = {
        "HGP-PPO": {},
        "L2D-PPO": {},
        "StabilityAware": {},
    }

    for inst_idx, inst in enumerate(INSTANCES):
        hgp_scores = []
        l2d_scores = []
        sta_scores = []

        for sev_idx, sev in enumerate(SEVERITY_ORDER):
            for stg_idx, stg in enumerate(STAGE_ORDER):
                sc_id = f"{sev}_{stg}"
                s_hgp = hgp[(inst, sc_id)]["composite_score"]
                s_l2d = l2d[(inst, sc_id)]["composite_score"]
                s_sta = sta[(inst, sc_id)]["composite_score"]

                hgp_scores.append(s_hgp)
                l2d_scores.append(s_l2d)
                sta_scores.append(s_sta)

                d_l2d = s_hgp - s_l2d
                d_sta = s_hgp - s_sta

                diff_l2d[inst_idx, sev_idx, stg_idx] = d_l2d
                diff_sta[inst_idx, sev_idx, stg_idx] = d_sta

                source_records.append({
                    "instance_id": inst,
                    "scenario_id": sc_id,
                    "delay_severity": sev,
                    "freeze_phase": stg,
                    "S_HGP_PPO": s_hgp,
                    "S_L2D_PPO": s_l2d,
                    "S_StabilityAware": s_sta,
                    "diff_vs_L2D": d_l2d,
                    "diff_vs_StabilityAware": d_sta,
                })

        inst_means["HGP-PPO"][inst] = np.mean(hgp_scores)
        inst_means["L2D-PPO"][inst] = np.mean(l2d_scores)
        inst_means["StabilityAware"][inst] = np.mean(sta_scores)

    # 4-instance equal-weighted mean
    overall_means = {
        "HGP-PPO": np.mean([inst_means["HGP-PPO"][inst] for inst in INSTANCES]),
        "L2D-PPO": np.mean([inst_means["L2D-PPO"][inst] for inst in INSTANCES]),
        "StabilityAware": np.mean([inst_means["StabilityAware"][inst] for inst in INSTANCES]),
    }

    source_df = pd.DataFrame(source_records)

    # Calculate Win / Tie / Loss counts
    all_d_l2d = diff_l2d.flatten()
    all_d_sta = diff_sta.flatten()

    win_l2d = int(np.sum(all_d_l2d < -1e-6))
    loss_l2d = int(np.sum(all_d_l2d > 1e-6))
    tie_l2d = len(all_d_l2d) - win_l2d - loss_l2d

    win_sta = int(np.sum(all_d_sta < -1e-6))
    loss_sta = int(np.sum(all_d_sta > 1e-6))
    tie_sta = len(all_d_sta) - win_sta - loss_sta

    print(f"vs L2D: Win={win_l2d}, Tie={tie_l2d}, Loss={loss_l2d}")
    print(f"vs StabilityAware: Win={win_sta}, Tie={tie_sta}, Loss={loss_sta}")

    # Color normalization: Symmetric around 0 for diverging map
    max_abs_diff = max(np.max(np.abs(all_d_l2d)), np.max(np.abs(all_d_sta)))
    norm_limit = float(np.ceil(max_abs_diff * 10) / 10.0)
    norm = TwoSlopeNorm(vmin=-norm_limit, vcenter=0.0, vmax=norm_limit)

    # -------------------------------------------------------------------------
    # Layout Construction
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(13.2, 9.6))

    # Outer layout:
    # 0: Section (a) title
    # 1: Section (a) panels
    # 2: Section (b) title
    # 3: Section (b) heatmaps
    outer_gs = gridspec.GridSpec(
        4, 1,
        height_ratios=[0.05, 0.28, 0.05, 0.62],
        hspace=0.22,
        top=0.96, bottom=0.06, left=0.07, right=0.88
    )

    # -------------------------------------------------------------------------
    # SECTION (A) TITLE
    # -------------------------------------------------------------------------
    ax_title_a = fig.add_subplot(outer_gs[0])
    ax_title_a.axis("off")
    ax_title_a.text(
        0.0, 0.5,
        "(a) Instance-Level and Overall Mean Composite Score S (Lower is Better; Shaded Bar = 4-Instance Equal-Weighted)",
        fontsize=10.5, fontweight="bold", ha="left", va="center"
    )

    # -------------------------------------------------------------------------
    # SECTION (A) PANELS
    # -------------------------------------------------------------------------
    part_a_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5,
        subplot_spec=outer_gs[1],
        wspace=0.25,
        width_ratios=[1, 1, 1, 1, 1.15]
    )

    methods_order = ["StabilityAware", "L2D-PPO", "HGP-PPO"]
    y_pos = [0, 1, 2]

    for idx, inst in enumerate(INSTANCES):
        ax = fig.add_subplot(part_a_gs[idx])
        for y_idx, m in enumerate(methods_order):
            val = inst_means[m][inst]
            col = COLORS[m]
            marker = MARKERS[m]
            ax.plot([0.65, val], [y_idx, y_idx], color="#D8D8D8", lw=1.2, zorder=1)
            ax.scatter(val, y_idx, color=col, marker=marker, s=65, zorder=3, edgecolor="black", lw=0.6)
            # Numeric Label with smart placement
            ax.text(val + 0.02, y_idx, f"{val:.4f}", va="center", ha="left",
                    fontsize=8.0, fontweight="bold" if m == "HGP-PPO" else "normal",
                    color=col if m == "HGP-PPO" else "#333333")

        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(methods_order, fontweight="bold")
        else:
            ax.set_yticklabels([])

        ax.set_title(INSTANCE_NAMES[inst], fontsize=8.6, fontweight="bold", pad=6)
        ax.set_xlabel("Mean Score S", fontsize=8.2)
        ax.set_xlim(0.68, 1.18)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 5th Panel: Overall 4-Instance Mean
    ax_mean = fig.add_subplot(part_a_gs[4])
    for y_idx, m in enumerate(methods_order):
        val = overall_means[m]
        col = COLORS[m]
        marker = MARKERS[m]
        ax_mean.plot([0.65, val], [y_idx, y_idx], color="#BFBFBF", lw=1.5, zorder=1)
        ax_mean.scatter(val, y_idx, color=col, marker=marker, s=80, zorder=3, edgecolor="black", lw=0.7)
        ax_mean.text(val + 0.02, y_idx, f"{val:.4f}", va="center", ha="left",
                     fontsize=8.2, fontweight="bold", color=col)

    ax_mean.set_yticks(y_pos)
    ax_mean.set_yticklabels([])
    ax_mean.set_title("Overall Mean\n(4-Instance Equal)", fontsize=8.6, fontweight="bold", pad=6, color="#0072B2")
    ax_mean.set_xlabel("Equal-Weighted S", fontsize=8.2)
    ax_mean.set_xlim(0.68, 1.18)
    ax_mean.grid(axis="x", linestyle="--", alpha=0.4)
    ax_mean.spines["top"].set_visible(False)
    ax_mean.spines["right"].set_visible(False)
    ax_mean.patch.set_facecolor("#F0F7FB")
    ax_mean.patch.set_alpha(0.8)

    # -------------------------------------------------------------------------
    # SECTION (B) TITLE
    # -------------------------------------------------------------------------
    ax_title_b = fig.add_subplot(outer_gs[2])
    ax_title_b.axis("off")
    ax_title_b.text(
        0.0, 0.5,
        "(b) 36-Scenario Paired Difference Heatmaps: d = S(HGP-PPO) - S(Baseline)  [Negative / Blue Favors HGP-PPO; Symmetrical Scale]",
        fontsize=10.5, fontweight="bold", ha="left", va="center"
    )

    # -------------------------------------------------------------------------
    # SECTION (B) HEATMAPS
    # -------------------------------------------------------------------------
    part_b_gs = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=outer_gs[3],
        wspace=0.18, hspace=0.25
    )

    cmap = plt.colormaps["RdBu_r"]

    # Row 1: vs. L2D-PPO
    for idx, inst in enumerate(INSTANCES):
        ax = fig.add_subplot(part_b_gs[0, idx])
        data = diff_l2d[idx]

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

        # Cell annotations
        for r in range(3):
            for c in range(3):
                val = data[r, c]
                txt_col = "white" if abs(val) > 0.45 else "black"
                prefix = "+" if val > 0 else ""
                ax.text(c, r, f"{prefix}{val:.2f}", ha="center", va="center",
                        fontsize=8.0, fontweight="bold", color=txt_col)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([])
        ax.set_yticks([0, 1, 2])
        if idx == 0:
            ax.set_yticklabels(["Low", "Med", "High"], fontsize=8.0, fontweight="medium")
            ax.set_ylabel("vs. L2D-PPO-APAL\nDelay Severity", fontsize=8.8, fontweight="bold", labelpad=4)
        else:
            ax.set_yticklabels([])

        ax.set_title(f"Instance {inst.split('_')[1]}", fontsize=8.8, fontweight="bold", pad=4)

    # Row 2: vs. StabilityAwareRepair
    for idx, inst in enumerate(INSTANCES):
        ax = fig.add_subplot(part_b_gs[1, idx])
        data = diff_sta[idx]

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

        # Cell annotations
        for r in range(3):
            for c in range(3):
                val = data[r, c]
                txt_col = "white" if abs(val) > 0.45 else "black"
                prefix = "+" if val > 0 else ""
                ax.text(c, r, f"{prefix}{val:.2f}", ha="center", va="center",
                        fontsize=8.0, fontweight="bold", color=txt_col)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Early", "Mid", "Late"], fontsize=8.0)
        ax.set_xlabel("Planning Freeze Phase", fontsize=8.2)

        ax.set_yticks([0, 1, 2])
        if idx == 0:
            ax.set_yticklabels(["Low", "Med", "High"], fontsize=8.0, fontweight="medium")
            ax.set_ylabel("vs. StabilityAware\nDelay Severity", fontsize=8.8, fontweight="bold", labelpad=4)
        else:
            ax.set_yticklabels([])

    # -------------------------------------------------------------------------
    # Win / Tie / Loss Badges (Positioned to the right of Section b)
    # -------------------------------------------------------------------------
    fig.text(
        0.895, 0.45,
        f"vs. L2D-PPO-APAL\n"
        f"Win / Tie / Loss\n"
        f"  {win_l2d} / {tie_l2d} / {loss_l2d}\n"
        f"  (Win: {win_l2d/36*100:.1f}%)",
        fontsize=8.5, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#EBF3F9", edgecolor="#0072B2", lw=1.2)
    )

    fig.text(
        0.895, 0.18,
        f"vs. StabilityAware\n"
        f"Win / Tie / Loss\n"
        f"  {win_sta} / {tie_sta} / {loss_sta}\n"
        f"  (Win: {win_sta/36*100:.1f}%)",
        fontsize=8.5, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7EFF5", edgecolor="#CC79A7", lw=1.2)
    )

    # -------------------------------------------------------------------------
    # Shared Colorbar (Vertically spans Section b)
    # -------------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.965, 0.12, 0.015, 0.44])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Score Difference d\n(<0 favors HGP-PPO)", fontsize=8.0, labelpad=4)
    cbar.ax.tick_params(labelsize=7.5)

    # Export
    export_bundle(fig, "fig6_reschedule_scenarios_heatmap", source_df)
    plt.close(fig)
    print("[Figure 6] Generation complete.")


if __name__ == "__main__":
    main()
