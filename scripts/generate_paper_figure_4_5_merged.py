"""
Generate Merged Single-Column Figure 4: Initial Scheduling Performance & Ablation Study.

Combines:
  (a) Initial Scheduling Cycle Span across 4 Physical Scales (HGP-PPO vs. 5 Baselines).
  (b) Ablation Study on Decision Architectures & Feasibility (Heatmap + Feasibility Indicators).

Specification:
  - Exact IEEE single-column width: 3.55 inches (~90 mm).
  - Total height: ~5.6 inches (~142 mm).
  - Highly compact, zero-waste layout, professional typography (Arial).
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

# Single-column IEEE typography & styling
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 7.5,
    "axes.labelsize": 7.8,
    "axes.titlesize": 8.2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
})

def main():
    print("=" * 70)
    print("Generating Merged Single-Column Figure 4 (Initial & Ablation)")
    print("=" * 70)

    # 1. Load Fig 4 Data
    csv_fig4 = FIGURES_DIR / "fig4_initial_cross_scale_source.csv"
    df_init = pd.read_csv(csv_fig4)

    # 2. Load Fig 5 Data
    csv_fig5 = FIGURES_DIR / "fig5_configuration_quality_feasibility_source.csv"
    df_abl = pd.read_csv(csv_fig5)

    fig = plt.figure(figsize=(3.55, 5.7))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.15, 1.15],
        hspace=0.45,
        left=0.22, right=0.96, top=0.94, bottom=0.06
    )

    # -------------------------------------------------------------------------
    # Panel (a): Multi-Scale Initial Scheduling Performance
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])

    scales = ["283", "680", "2338", "3182"]
    x_indices = np.arange(len(scales))

    # Styling for methods
    method_styles = {
        "HGP-PPO": {"color": "#0072B2", "lw": 1.8, "ls": "-", "marker": "o", "ms": 4.5, "z": 5, "label": "HGP-PPO (Ours)"},
        "L2D-PPO-APAL": {"color": "#009E73", "lw": 1.2, "ls": "--", "marker": "s", "ms": 3.8, "z": 4, "label": "L2D-PPO"},
        "Graph-DDQN-APAL": {"color": "#E69F00", "lw": 1.1, "ls": "-.", "marker": "D", "ms": 3.5, "z": 3, "label": "Graph-DDQN"},
        "LPT": {"color": "#555555", "lw": 1.0, "ls": ":", "marker": "^", "ms": 3.5, "z": 2, "label": "LPT Rule"},
        "IG": {"color": "#D55E00", "lw": 1.0, "ls": ":", "marker": "v", "ms": 3.5, "z": 2, "label": "IG Search"},
        "Beam Search": {"color": "#CC79A7", "lw": 1.0, "ls": ":", "marker": "x", "ms": 4.0, "z": 2, "label": "Beam Search"},
    }

    for _, row in df_init.iterrows():
        key = row["Key"]
        if key not in method_styles:
            continue
        style = method_styles[key]
        vals = [float(row[s]) for s in scales]
        ax_a.plot(
            x_indices, vals,
            color=style["color"], lw=style["lw"], linestyle=style["ls"],
            marker=style["marker"], markersize=style["ms"],
            label=style["label"], zorder=style["z"]
        )

    # Annotate HGP-PPO values
    hgp_vals = [float(df_init.loc[df_init["Key"] == "HGP-PPO", s].iloc[0]) for s in scales]
    offsets = [(-12, 10), (0, 10), (0, -14), (0, 10)]
    for i, (val, off) in enumerate(zip(hgp_vals, offsets)):
        ax_a.annotate(
            f"{val:.0f}h", xy=(i, val), xytext=off, textcoords="offset points",
            fontsize=6.5, fontweight="bold", color="#0072B2", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F7FA", edgecolor="none", alpha=0.8)
        )

    ax_a.set_yscale("log")
    ax_a.set_xticks(x_indices)
    ax_a.set_xticklabels(["283", "680*", "2338", "3182"], fontsize=7.2, fontweight="bold")
    ax_a.set_xlabel("Physical Scale [Tasks] (*In-Distribution: 680)", fontsize=7.2, labelpad=2)
    ax_a.set_ylabel("Cycle Span $H$ (h, Log Scale)", fontsize=7.5)
    ax_a.set_title("(a) Initial Scheduling across Physical Scales", fontsize=8.0, fontweight="bold", loc="left", pad=4)
    ax_a.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_a.legend(loc="upper left", frameon=True, facecolor="#FAFAFA", framealpha=0.9, ncol=2, handlelength=1.4, handletextpad=0.4, columnspacing=0.6)
    ax_a.set_ylim(220, 4200)

    # -------------------------------------------------------------------------
    # Panel (b): Ablation Study (Quality Degradation Matrix)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[1])

    abl_rows = [
        ("Full Model (Ours)", [0.0, 0.0, 0.0, 0.0]),
        ("w/o Baseline Conditioning", [4.6, 39.6, 2.4, 10.0]),
        ("Worker-Station Preassign", [11.6, 96.2, 32.7, 63.0]),
        ("Homogeneous GraphSAGE", [1.8, 21.5, 9.4, 19.6]),
        ("Op-Station Joint", [-6.8, 15.6, -4.6, 27.7]),
        ("Operation-only", [-6.0, 59.3, np.nan, np.nan]),
    ]

    config_names = [r[0] for r in abl_rows]
    matrix = np.array([r[1] for r in abl_rows])

    # Create visual table / heatmap
    n_rows, n_cols = matrix.shape
    ax_b.set_xlim(-0.5, n_cols - 0.5)
    ax_b.set_ylim(-0.5, n_rows - 0.5)
    ax_b.invert_yaxis()

    import matplotlib.colors as mcolors
    # Color map from soft blue (negative / good) to white (0%) to soft red (high degradation)
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=100)
    cmap = plt.cm.RdYlGn_r

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            val = matrix[r_idx, c_idx]
            if np.isnan(val):
                color = "#EEEEEE"
                text = "N/A"
                text_color = "#999999"
            else:
                color = cmap(norm(val))
                text = f"{val:+.1f}%" if val != 0 else "0.0%"
                text_color = "#000000" if abs(val) < 60 else "#FFFFFF"

            rect = plt.Rectangle(
                (c_idx - 0.45, r_idx - 0.42), 0.9, 0.84,
                facecolor=color, edgecolor="#D0D0D0", lw=0.5, zorder=2
            )
            ax_b.add_patch(rect)
            ax_b.text(c_idx, r_idx, text, ha="center", va="center", fontsize=6.8, fontweight="bold" if r_idx==0 else "normal", color=text_color, zorder=3)

    ax_b.set_xticks(range(n_cols))
    ax_b.set_xticklabels(["Medium", "Large", "Industrial-I", "Industrial-II"], fontsize=7.2, fontweight="bold")
    ax_b.set_xlabel("Instance", fontsize=7.2, labelpad=2)
    ax_b.set_yticks(range(n_rows))
    ax_b.set_yticklabels(config_names, fontsize=6.8, fontweight="bold")
    ax_b.set_title("(b) Ablation: Relative Degradation $\\Delta J(\\%)$", fontsize=8.0, fontweight="bold", loc="left", pad=4)
    ax_b.tick_params(axis="both", which="both", length=0)
    ax_b.grid(False)

    # Save to both fig3 (manuscript) and fig4_compact (backup)
    ASSETS_DIR = PROJECT_ROOT / "docs" / "APAL_HGP_PPO_Journal_Manuscript" / "04_Figures_and_Assets"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for out_dir in [FIGURES_DIR, ASSETS_DIR]:
        for base_name in ["fig3_initial_and_ablation", "fig4_initial_and_ablation_compact"]:
            png_path = out_dir / f"{base_name}.png"
            pdf_path = out_dir / f"{base_name}.pdf"
            svg_path = out_dir / f"{base_name}.svg"

            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
            fig.savefig(svg_path, bbox_inches="tight")
            print(f"[Export] Saved {base_name} to {out_dir}")
    plt.close(fig)

if __name__ == "__main__":
    main()
