"""
Generate Figure 3 (Compact Edition): Development-Set Selection Trajectories.

Tailored for IEEE Transactions single-column format:
- Width: 3.5 inches (~88.9 mm).
- Height: 4.2 inches (~106.7 mm).
- Two vertically stacked panels:
  (a) Initial Scheduling Dev-Set Trajectory (Instance 680, Epochs 2-60).
  (b) Template Rescheduling Dev-Set Trajectory (Continuous 120 Epochs).
- Eliminates separate flat feasibility track; inlines a compact feasibility badge.
- Selected checkpoints prominently highlighted.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "APAL_HGP_PPO_Journal_Manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Typography & styling for single-column IEEE figures
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.8,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
})

def main():
    print("=" * 70)
    print("Generating Figure 3 (Compact Edition): Dev-Set Trajectories")
    print("=" * 70)

    csv_path = FIGURES_DIR / "fig3_training_selection_trajectories_source.csv"
    df = pd.read_csv(csv_path)

    df_init = df[df["task"] == "initial_scheduling"].copy()
    df_res = df[df["task"] == "template_rescheduling"].copy()

    selected_ep_init = 44
    selected_val_init = float(df_init.loc[df_init["epoch"] == selected_ep_init, "metric_value"].iloc[0])

    selected_ep_res = 62
    selected_val_res = float(df_res.loc[df_res["epoch"] == selected_ep_res, "metric_value"].iloc[0])

    # Figure Canvas: 3.5 inches width (exact IEEE single-column)
    fig = plt.figure(figsize=(3.55, 4.3))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.15], hspace=0.38, left=0.15, right=0.96, top=0.92, bottom=0.09)

    # -------------------------------------------------------------------------
    # Panel (a): Initial Scheduling Trajectory
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    ax_a.plot(df_init["epoch"], df_init["metric_value"], color="#0072B2", lw=1.2, marker="o", markersize=3.0, label="$J_{\\mathrm{dev}}$ (Instance 680)")
    ax_a.scatter([selected_ep_init], [selected_val_init], s=55, facecolor="white", edgecolor="#D55E00", lw=1.5, marker="D", zorder=5)

    # Annotate Selected Checkpoint
    ax_a.annotate(
        f"Selected (Ep. {selected_ep_init})\n$J = {selected_val_init:.1f}\\text{{ h}}$",
        xy=(selected_ep_init, selected_val_init),
        xytext=(selected_ep_init - 26, selected_val_init + 250),
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9),
        fontsize=6.8, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF5EB", edgecolor="#D55E00", lw=0.6)
    )

    # In-plot 100% Feasibility Badge
    ax_a.text(
        0.03, 0.12, "Feasibility: 100%", transform=ax_a.transAxes,
        fontsize=6.5, fontweight="bold", color="#009E73",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAF8F2", edgecolor="#009E73", lw=0.5)
    )

    ax_a.set_ylabel("Dev Span $J_{\\mathrm{dev}}$ (h)", fontsize=7.8)
    ax_a.set_xlabel("Training Epoch", fontsize=7.5, labelpad=2)
    ax_a.set_title("(a) Initial Scheduling Trajectory (Dev 680)", fontsize=8.0, fontweight="bold", loc="left", pad=4)
    ax_a.set_ylim(420, 1350)
    ax_a.set_xlim(0, 62)
    ax_a.grid(True, linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # Panel (b): Template Rescheduling Trajectory
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])
    # Background shading for stages
    ax_b.axvspan(0, 60, facecolor="#F0F7FA", alpha=0.6, zorder=1)
    ax_b.axvspan(60, 120, facecolor="#FAF4F8", alpha=0.6, zorder=1)
    ax_b.axvline(60, color="#888888", linestyle="--", lw=0.8, zorder=2)

    # Stage labels
    ax_b.text(28, 4.35, "Stage 1: PPO", ha="center", fontsize=6.8, fontweight="bold", color="#0072B2")
    ax_b.text(90, 4.35, "Stage 2: BIC", ha="center", fontsize=6.8, fontweight="bold", color="#CC79A7")

    # Main curve
    ax_b.plot(df_res["epoch"], df_res["metric_value"], color="#0072B2", lw=1.2, marker="o", markersize=2.8, label="Score $S$", zorder=3)
    ax_b.scatter([selected_ep_res], [selected_val_res], s=55, facecolor="white", edgecolor="#D55E00", lw=1.5, marker="D", zorder=5)

    # Annotate Selected Checkpoint
    ax_b.annotate(
        f"Selected (Ep. {selected_ep_res})\n$S = {selected_val_res:.3f}$",
        xy=(selected_ep_res, selected_val_res),
        xytext=(selected_ep_res + 10, selected_val_res + 1.2),
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.9),
        fontsize=6.8, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF5EB", edgecolor="#D55E00", lw=0.6)
    )

    # In-plot 100% Feasibility Badge
    ax_b.text(
        0.03, 0.12, "Feasibility: 100%", transform=ax_b.transAxes,
        fontsize=6.5, fontweight="bold", color="#009E73",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAF8F2", edgecolor="#009E73", lw=0.5)
    )

    ax_b.set_ylabel("Dev Composite Score $S$", fontsize=7.8)
    ax_b.set_xlabel("Cumulative Epoch", fontsize=7.5, labelpad=2)
    ax_b.set_title("(b) Rescheduling Trajectory (120 Epochs)", fontsize=8.0, fontweight="bold", loc="left", pad=4)
    ax_b.set_ylim(0.5, 4.8)
    ax_b.set_xlim(0, 122)
    ax_b.grid(True, linestyle="--", alpha=0.35)

    # Save
    base_name = "fig3_training_selection_trajectories_compact"
    png_path = FIGURES_DIR / f"{base_name}.png"
    pdf_path = FIGURES_DIR / f"{base_name}.pdf"
    svg_path = FIGURES_DIR / f"{base_name}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Export] Saved compact Figure 3:\n  {png_path}\n  {pdf_path}\n  {svg_path}")

if __name__ == "__main__":
    main()
