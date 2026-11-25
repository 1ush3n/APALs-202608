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
    "real_283": "Medium\n(283 ops)",
    "real_680": "Large\n(680 ops)",
    "real_2338": "Industrial-I\n(2,338 ops)",
    "real_3182": "Industrial-II\n(3,182 ops)",
}

INSTANCE_SHORT_NAMES = {
    "real_283": "Medium",
    "real_680": "Large",
    "real_2338": "Industrial-I",
    "real_3182": "Industrial-II",
}

SEVERITY_ORDER = ["low", "medium", "high"]
STAGE_ORDER = ["early", "middle", "late"]


def export_bundle(fig, base_name: str, source_df: pd.DataFrame):
    """Export PDF, SVG, PNG, and source CSV."""
    ASSETS_DIR = PROJECT_ROOT / "docs" / "APAL_HGP_PPO_Journal_Manuscript" / "04_Figures_and_Assets"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in [FIGURES_DIR, ASSETS_DIR]:
        for b_name in [base_name, "fig4_reschedule_scenarios_heatmap"]:
            png_path = out_dir / f"{b_name}.png"
            pdf_path = out_dir / f"{b_name}.pdf"
            svg_path = out_dir / f"{b_name}.svg"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
            fig.savefig(svg_path, bbox_inches="tight")
    csv_path = FIGURES_DIR / f"{base_name}_source.csv"
    source_df.to_csv(csv_path, index=False)

    print(f"[Export] Saved {base_name} and fig4_reschedule_scenarios_heatmap to {FIGURES_DIR} and {ASSETS_DIR}")


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
    # Layout Construction (Scheme B: Left-Right Dual Block, Aspect Ratio ~2.7:1)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14.8, 5.5))
    methods_order = ["StabilityAware", "L2D-PPO", "HGP-PPO"]

    # Left: 31% width for (a) Consolidated Instance & Overall Mean Plot
    # Right: 69% width for (b) 2x4 Heatmaps + Badges + Colorbar
    gs_main = gridspec.GridSpec(
        1, 2,
        width_ratios=[0.31, 0.69],
        wspace=0.18,
        top=0.92, bottom=0.10, left=0.06, right=0.90
    )

    # =========================================================================
    # LEFT PANEL: (a) Consolidated Mean Plot across 4 Scales + Overall Mean
    # =========================================================================
    ax_left = fig.add_subplot(gs_main[0])
    y_labels = [
        "Medium\n(283 ops)",
        "Large\n(680 ops)",
        "Industrial-I\n(2,338 ops)",
        "Industrial-II\n(3,182 ops)",
        "Overall Mean\n(4-Instance Equal)",
    ]
    y_positions = [0, 1, 2, 3, 4.3]

    # Shaded background highlight for Overall Mean
    ax_left.axhspan(3.65, 4.95, facecolor="#F0F7FB", alpha=0.85, zorder=0)

    for idx, inst in enumerate(INSTANCES):
        y = y_positions[idx]
        vals = [inst_means[m][inst] for m in methods_order]
        ax_left.plot([min(vals), max(vals)], [y, y], color="#DCDCDC", lw=2.0, zorder=1)
        for m in methods_order:
            val = inst_means[m][inst]
            ax_left.scatter(
                val, y,
                color=COLORS[m], marker=MARKERS[m],
                s=65, zorder=3, edgecolor="black", lw=0.6,
                label=m if idx == 0 else ""
            )
            # Smart numeric placement to avoid collision
            offset_y = 0.16 if m == "StabilityAware" else (-0.16 if m == "L2D-PPO" else 0.0)
            ax_left.text(
                val, y + offset_y, f"{val:.4f}",
                ha="center", va="bottom" if offset_y >= 0 else "top",
                fontsize=7.0, color=COLORS[m],
                fontweight="bold" if m == "HGP-PPO" else "normal"
            )

    # 5th Row: Overall Mean
    y_ov = y_positions[4]
    ov_vals = [overall_means[m] for m in methods_order]
    ax_left.plot([min(ov_vals), max(ov_vals)], [y_ov, y_ov], color="#B0C8D8", lw=2.5, zorder=1)
    for m in methods_order:
        val = overall_means[m]
        ax_left.scatter(
            val, y_ov,
            color=COLORS[m], marker=MARKERS[m],
            s=80, zorder=3, edgecolor="black", lw=0.7
        )
        offset_y = 0.16 if m == "StabilityAware" else (-0.16 if m == "L2D-PPO" else 0.0)
        ax_left.text(
            val, y_ov + offset_y, f"{val:.4f}",
            ha="center", va="bottom" if offset_y >= 0 else "top",
            fontsize=7.3, color=COLORS[m], fontweight="bold"
        )

    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(y_labels, fontsize=8.2, fontweight="bold")
    ax_left.set_xlabel("Mean Composite Score S (Lower is Better)", fontsize=8.5, fontweight="bold")
    ax_left.set_title("(a) Instance & Overall Mean Score S", fontsize=9.2, fontweight="bold", pad=8)
    ax_left.set_xlim(0.70, 1.15)
    ax_left.grid(axis="x", linestyle="--", alpha=0.4)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(loc="lower right", frameon=True, fontsize=7.8, facecolor="white", framealpha=0.92, edgecolor="#D0D0D0")

    # =========================================================================
    # RIGHT BLOCK: (b) 2x4 Heatmaps (vs. L2D and vs. StabilityAware)
    # =========================================================================
    fig.text(
        0.395, 0.965,
        "(b) 36-Scenario Paired Difference Heatmaps: d = S(HGP-PPO) - S(Baseline)  [Negative / Blue Favors HGP-PPO]",
        fontsize=9.2, fontweight="bold", ha="left", va="top"
    )

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=gs_main[1],
        wspace=0.16, hspace=0.25
    )
    cmap = plt.colormaps["RdBu_r"]

    for idx, inst in enumerate(INSTANCES):
        # Row 0: vs. L2D-PPO
        ax0 = fig.add_subplot(gs_right[0, idx])
        im = ax0.imshow(diff_l2d[idx], cmap=cmap, norm=norm, aspect="auto")
        for r in range(3):
            for c in range(3):
                val = diff_l2d[idx, r, c]
                txt_col = "white" if abs(val) > 0.45 else "black"
                prefix = "+" if val > 0 else ""
                ax0.text(c, r, f"{prefix}{val:.2f}", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color=txt_col)
        ax0.set_xticks([0, 1, 2])
        ax0.set_xticklabels([])
        ax0.set_yticks([0, 1, 2])
        if idx == 0:
            ax0.set_yticklabels(["Low", "Med", "High"], fontsize=7.6)
            ax0.set_ylabel("vs. L2D-PPO\nDelay Severity", fontsize=8.0, fontweight="bold", labelpad=2)
        else:
            ax0.set_yticklabels([])
        ax0.set_title(INSTANCE_SHORT_NAMES[inst], fontsize=8.2, fontweight="bold", pad=3)

        # Row 1: vs. StabilityAwareRepair
        ax1 = fig.add_subplot(gs_right[1, idx])
        ax1.imshow(diff_sta[idx], cmap=cmap, norm=norm, aspect="auto")
        for r in range(3):
            for c in range(3):
                val = diff_sta[idx, r, c]
                txt_col = "white" if abs(val) > 0.45 else "black"
                prefix = "+" if val > 0 else ""
                ax1.text(c, r, f"{prefix}{val:.2f}", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color=txt_col)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["Early", "Mid", "Late"], fontsize=7.5)
        ax1.set_xlabel("Freeze Phase", fontsize=7.8)
        ax1.set_yticks([0, 1, 2])
        if idx == 0:
            ax1.set_yticklabels(["Low", "Med", "High"], fontsize=7.6)
            ax1.set_ylabel("vs. StabilityAware\nDelay Severity", fontsize=8.0, fontweight="bold", labelpad=2)
        else:
            ax1.set_yticklabels([])

    # -------------------------------------------------------------------------
    # Win / Tie / Loss Badges (Right Margin)
    # -------------------------------------------------------------------------
    fig.text(
        0.910, 0.68,
        f"vs. L2D-PPO-APAL\n"
        f"Win / Tie / Loss\n"
        f"  {win_l2d} / {tie_l2d} / {loss_l2d}\n"
        f"  (Win: {win_l2d/36*100:.1f}%)",
        fontsize=7.5, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EBF3F9", edgecolor="#0072B2", lw=1.0)
    )

    fig.text(
        0.910, 0.35,
        f"vs. StabilityAware\n"
        f"Win / Tie / Loss\n"
        f"  {win_sta} / {tie_sta} / {loss_sta}\n"
        f"  (Win: {win_sta/36*100:.1f}%)",
        fontsize=7.5, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7EFF5", edgecolor="#CC79A7", lw=1.0)
    )

    # -------------------------------------------------------------------------
    # Shared Colorbar
    # -------------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.970, 0.14, 0.010, 0.72])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Score Difference d\n(<0 favors HGP-PPO)", fontsize=7.2, labelpad=3)
    cbar.ax.tick_params(labelsize=6.8)

    # Export
    export_bundle(fig, "fig6_reschedule_scenarios_heatmap", source_df)
    plt.close(fig)
    print("[Figure 6/4] Generation complete with Scheme B layout.")


if __name__ == "__main__":
    main()
