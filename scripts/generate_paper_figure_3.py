"""
Generate Figure 3: Development-Set Selection Trajectories for Initial Scheduling and Rescheduling.

Structure:
  Two side-by-side panels:
  (a) Initial Scheduling:
      - Main plot: Dev-set makespan J_dev (h) on real_680 across epochs 0-60.
      - Lower track: Feasibility rate (%) across epochs.
      - Selected checkpoint: Epoch 44 (J_dev = 493.53 h) marked with diamond.
  (b) Template Rescheduling:
      - Main plot: Dev-set average composite score S on 3 validation scenarios across cumulative epochs 0-120.
      - Stage divider: Vertical dashed line at Epoch 60 separating Stage 1 and Stage 2.
      - Lower track: Feasibility rate (%) across epochs.
      - Selected checkpoint: Epoch 62 (S = 0.9382) marked with diamond.

Outputs:
  - fig3_training_selection_trajectories.pdf (vector)
  - fig3_training_selection_trajectories.svg (editable vector)
  - fig3_training_selection_trajectories.png (300 DPI preview)
  - fig3_training_selection_trajectories_source.csv (complete raw source data)
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def parse_reschedule_eval_log(log_path, epoch_offset=0):
    """Parse [AsyncEval][Done] lines from reschedule training log."""
    eval_by_ep = defaultdict(list)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[AsyncEval][Done]" in line:
                m = re.search(r"ep=(\d+)\s+score=([\d\.]+)\s+elig=(\d+)\s+mk=([\d\.]+)", line)
                if m:
                    ep = int(m.group(1)) + epoch_offset
                    eval_by_ep[ep].append({
                        "score": float(m.group(2)),
                        "elig": int(m.group(3)),
                        "mk": float(m.group(4))
                    })
    summary = []
    for ep in sorted(eval_by_ep.keys()):
        recs = eval_by_ep[ep]
        summary.append({
            "epoch": ep,
            "avg_score": sum(r["score"] for r in recs) / len(recs),
            "feasibility_rate": sum(r["elig"] for r in recs) / len(recs) * 100.0,
            "avg_makespan": sum(r["mk"] for r in recs) / len(recs),
            "eval_count": len(recs)
        })
    return pd.DataFrame(summary)


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
    print("Generating Figure 3: Development-Set Selection Trajectories")
    print("=" * 70)

    # 1. Load Initial Scheduling Data
    init_csv = PROJECT_ROOT / "results/01_initial_main/initial_worker_pointer_v2_full_x/initial_worker_pointer_v2_full_x_260904-123836/checkpoints/async_eval/results/async_eval_summary.csv"
    df_init_raw = pd.read_csv(init_csv)
    df_init = pd.DataFrame({
        "epoch": df_init_raw["episode"],
        "metric_value": df_init_raw["selection_score"],
        "feasibility_rate": df_init_raw["eligible"] * 100.0,
        "reward": df_init_raw["reward"],
    })
    selected_ep_init = 44
    selected_val_init = float(df_init.loc[df_init["epoch"] == selected_ep_init, "metric_value"].iloc[0])

    # 2. Load Rescheduling Data
    log_s1 = PROJECT_ROOT / "results/02_reschedule_main/train_full_x_bic.log"
    log_s2 = PROJECT_ROOT / "results/02_reschedule_main/train_full_x_bic_phase2.log"
    df_res_s1 = parse_reschedule_eval_log(log_s1, epoch_offset=0)
    df_res_s2 = parse_reschedule_eval_log(log_s2, epoch_offset=60)
    df_res = pd.concat([df_res_s1, df_res_s2], ignore_index=True).rename(columns={"avg_score": "metric_value"})

    selected_ep_res = 62
    selected_val_res = float(df_res.loc[df_res["epoch"] == selected_ep_res, "metric_value"].iloc[0])

    # Combine Source DataFrame
    df_source_init = df_init.copy()
    df_source_init["task"] = "initial_scheduling"
    df_source_init["target_instance"] = "real_680"
    df_source_init["metric_name"] = "J_dev_makespan_h"
    df_source_init["selected_checkpoint"] = df_source_init["epoch"] == selected_ep_init

    df_source_res = df_res.copy()
    df_source_res["task"] = "template_rescheduling"
    df_source_res["target_instance"] = "validation_0001 (3 scenarios)"
    df_source_res["metric_name"] = "dev_mean_composite_score_S"
    df_source_res["selected_checkpoint"] = df_source_res["epoch"] == selected_ep_res

    source_df = pd.concat([df_source_init, df_source_res], ignore_index=True)

    # -------------------------------------------------------------------------
    # Create Figure Layout
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(13.2, 6.2))
    outer_gs = gridspec.GridSpec(
        1, 2,
        wspace=0.26,
        left=0.07, right=0.96, top=0.90, bottom=0.10
    )

    # -------------------------------------------------------------------------
    # Panel (a): Initial Scheduling Trajectory
    # -------------------------------------------------------------------------
    panel_a_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[0],
        height_ratios=[3.2, 1.0],
        hspace=0.10
    )

    ax_init_main = fig.add_subplot(panel_a_gs[0])
    ax_init_feas = fig.add_subplot(panel_a_gs[1], sharex=ax_init_main)

    # Main plot
    ax_init_main.plot(
        df_init["epoch"], df_init["metric_value"],
        color="#0072B2", lw=1.6, marker="o", markersize=4.5,
        label="Dev-Set Selection Metric $J_{\\mathrm{dev}}$",
        zorder=3
    )

    # Selected Checkpoint Highlight
    ax_init_main.scatter(
        [selected_ep_init], [selected_val_init],
        s=120, facecolor="white", edgecolor="#D55E00", lw=2.2, marker="D",
        label=f"Selected Checkpoint (Ep. {selected_ep_init}, {selected_val_init:.2f} h)",
        zorder=5
    )

    # Annotation callout
    ax_init_main.annotate(
        f"Selected: Ep. {selected_ep_init}\n$J_{{\\mathrm{{dev}}}} = {selected_val_init:.2f}\\text{{ h}}$",
        xy=(selected_ep_init, selected_val_init),
        xytext=(selected_ep_init + 4, selected_val_init + 230),
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.3),
        fontsize=8.2, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF5EB", edgecolor="#D55E00", lw=1.0)
    )

    ax_init_main.set_ylabel("Dev Selection Score $J_{\\mathrm{dev}}$ (h)\n[Cycle Span on Dev 680]", fontsize=8.8)
    ax_init_main.set_title(
        "(a) Initial Scheduling Dev-Set Trajectory (Instance 680)\n"
        "[Evaluated Every 2 Epochs; Deterministic Decoding]",
        fontsize=9.2, fontweight="bold", loc="left", pad=6
    )
    ax_init_main.set_ylim(400, 1350)
    ax_init_main.grid(True, linestyle="--", alpha=0.35)
    ax_init_main.tick_params(labelbottom=False)
    ax_init_main.legend(loc="upper right", frameon=True, facecolor="#FAFAFA", framealpha=0.9)

    # Lower Feasibility Track
    ax_init_feas.plot(
        df_init["epoch"], df_init["feasibility_rate"],
        color="#009E73", lw=1.4, marker="s", markersize=3.5, zorder=3
    )
    ax_init_feas.set_ylabel("Feasible\nRate (%)", fontsize=7.8)
    ax_init_feas.set_xlabel("Training Epoch", fontsize=8.8)
    ax_init_feas.set_ylim(70, 105)
    ax_init_feas.set_yticks([80, 100])
    ax_init_feas.grid(True, linestyle="--", alpha=0.35)
    ax_init_feas.set_xlim(0, 62)

    # -------------------------------------------------------------------------
    # Panel (b): Template Rescheduling Trajectory
    # -------------------------------------------------------------------------
    panel_b_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[1],
        height_ratios=[3.2, 1.0],
        hspace=0.10
    )

    ax_res_main = fig.add_subplot(panel_b_gs[0])
    ax_res_feas = fig.add_subplot(panel_b_gs[1], sharex=ax_res_main)

    # Background shading for Stage 1 vs Stage 2
    ax_res_main.axvspan(0, 60, facecolor="#F0F7FA", alpha=0.6, zorder=1)
    ax_res_main.axvspan(60, 120, facecolor="#FAF4F8", alpha=0.6, zorder=1)
    ax_res_feas.axvspan(0, 60, facecolor="#F0F7FA", alpha=0.6, zorder=1)
    ax_res_feas.axvspan(60, 120, facecolor="#FAF4F8", alpha=0.6, zorder=1)

    # Vertical line separating stages
    ax_res_main.axvline(60, color="#666666", linestyle="--", lw=1.3, zorder=2)
    ax_res_feas.axvline(60, color="#666666", linestyle="--", lw=1.3, zorder=2)

    # Stage labels with clean badges
    ax_res_main.text(
        30, 3.65, "Stage 1: PPO Warm-Start\n(Epochs 1–60)",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#0072B2", alpha=0.9, lw=0.9)
    )
    ax_res_main.text(
        90, 3.65, "Stage 2: BIC Self-Play\n(Epochs 61–120)",
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#CC79A7",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#CC79A7", alpha=0.9, lw=0.9)
    )

    # Main plot
    ax_res_main.plot(
        df_res["epoch"], df_res["metric_value"],
        color="#0072B2", lw=1.6, marker="o", markersize=4.5,
        label="Dev-Set Average Score $S$",
        zorder=3
    )

    # Selected Checkpoint Highlight
    ax_res_main.scatter(
        [selected_ep_res], [selected_val_res],
        s=120, facecolor="white", edgecolor="#D55E00", lw=2.2, marker="D",
        label=f"Selected Checkpoint (Ep. {selected_ep_res}, $S = {selected_val_res:.4f}$)",
        zorder=5
    )

    # Annotation callout
    ax_res_main.annotate(
        f"Selected: Ep. {selected_ep_res}\n$S = {selected_val_res:.4f}$",
        xy=(selected_ep_res, selected_val_res),
        xytext=(selected_ep_res + 12, selected_val_res + 1.2),
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2),
        fontsize=8.2, fontweight="bold", color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5EB", edgecolor="#D55E00", lw=1.0)
    )

    ax_res_main.set_ylabel("Dev Composite Score $S$ (Lower is Better)\n[Mean over 3 Validation Scenarios]", fontsize=8.8)
    ax_res_main.set_title(
        "(b) Template Rescheduling Dev-Set Trajectory (Continuous 120 Epochs)\n"
        "[Stage 1: Initial Warm-Start | Stage 2: BIC Refinement]",
        fontsize=9.2, fontweight="bold", loc="left", pad=6
    )
    ax_res_main.set_ylim(0.5, 4.8)
    ax_res_main.grid(True, linestyle="--", alpha=0.35)
    ax_res_main.tick_params(labelbottom=False)
    ax_res_main.legend(loc="upper right", frameon=True, facecolor="#FAFAFA", framealpha=0.9)

    # Lower Feasibility Track
    ax_res_feas.plot(
        df_res["epoch"], df_res["feasibility_rate"],
        color="#009E73", lw=1.4, marker="s", markersize=3.5, zorder=3
    )
    ax_res_feas.set_ylabel("Feasible\nRate (%)", fontsize=7.8)
    ax_res_feas.set_xlabel("Cumulative Training Epoch", fontsize=8.8)
    ax_res_feas.set_ylim(70, 105)
    ax_res_feas.set_yticks([80, 100])
    ax_res_feas.grid(True, linestyle="--", alpha=0.35)
    ax_res_feas.set_xlim(0, 122)

    # Export
    export_bundle(fig, "fig3_training_selection_trajectories", source_df)
    plt.close(fig)
    print("[Figure 3] Generation complete.")


if __name__ == "__main__":
    main()
