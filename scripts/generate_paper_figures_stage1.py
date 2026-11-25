"""
生成 APAL 论文第一阶段核心科研图表 (Fig. 4, Fig. 5, Fig. 7, Fig. 8)
输出格式: PDF (矢量投稿), SVG (可编辑矢量), PNG (300 DPI 预览), 以及源数据 CSV
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# 全局绘图规范配置 (IEEE / Nature 风格)
# ---------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.color'] = '#333333'
mpl.rcParams['ytick.color'] = '#333333'
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.major.size'] = 3.5
mpl.rcParams['ytick.major.size'] = 3.5
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = Path("docs/APAL_HGP_PPO_Journal_Manuscript/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一学术调色板 (Colorblind-safe)
COLORS = {
    'HGP-PPO': '#0072B2',              # 主方法: 高饱和深蓝
    'L2D-PPO-APAL': '#D55E00',         # 朱红/橙色
    'Graph-DDQN-APAL': '#009E73',      # 蓝绿色
    'Beam Search': '#882255',          # 深酒红
    'BeamSearchRepair': '#882255',     # 深酒红
    'StabilityAwareRepair': '#CC79A7', # 浅紫红
    'HybridCPMStabilityRepair': '#56B4E9', # 天蓝色
    'TaktAwareRepair': '#E69F00',      # 暖琥珀黄
    'IteratedGreedyRepair': '#44AA99', # 蓝绿色
    'IG': '#44AA99',                   # 蓝绿色
    'LPT': '#555555',                  # 炭灰
    'SA': '#117733',                   # 深森林绿
}

ASSETS_DIR = Path("docs/APAL_HGP_PPO_Journal_Manuscript/04_Figures_and_Assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig_bundle(fig, base_name: str, source_df: pd.DataFrame = None):
    names = [base_name]
    if base_name == "fig8_runtime_scalability":
        names.append("fig5_runtime_scalability")

    for b_name in names:
        for target_dir in [OUTPUT_DIR, ASSETS_DIR]:
            pdf_path = target_dir / f"{b_name}.pdf"
            svg_path = target_dir / f"{b_name}.svg"
            png_path = target_dir / f"{b_name}.png"
            csv_path = target_dir / f"{b_name}_source.csv"

            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
            fig.savefig(svg_path, dpi=300, bbox_inches='tight')
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            if source_df is not None:
                source_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Generated: {png_path.name}, {pdf_path.name}, {svg_path.name} in {target_dir}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 4: 初始模板质量与跨规模表现 (四面板水平点图)
# ---------------------------------------------------------------------------
def plot_fig4():
    instances = [283, 680, 2338, 3182]
    scales_info = [
        "283 (Downward Extrapol., <400)",
        "680 (In-Distribution, 400–800)",
        "2338 (Upward Extrapol., ~2.9×)",
        "3182 (Upward Extrapol., ~4.0×)"
    ]
    
    data = {
        'Method': ['HGP-PPO (Ours)', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'Beam Search', 'IG (Iterated Greedy)', 'LPT (Best Rule)'],
        'Key': ['HGP-PPO', 'L2D-PPO-APAL', 'Graph-DDQN-APAL', 'Beam Search', 'IG', 'LPT'],
        283: [286.96, 288.97, 286.77, 333.43, 339.05, 292.63],
        680: [489.00, 518.65, 612.29, 984.30, 961.10, 637.56],
        2338: [1209.96, 1079.44, 1263.66, 1904.88, 1797.57, 1497.35],
        3182: [1623.69, 1804.71, 2049.70, 3272.54, 2759.01, 2039.50],
        'Mean': [902.40, 922.94, 1053.11, 1623.79, 1464.18, 1116.76]
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.3), sharey=True, dpi=300)
    fig.subplots_adjust(wspace=0.16, top=0.85, bottom=0.16, left=0.18, right=0.98)

    methods = df['Method'].tolist()
    y_pos = np.arange(len(methods))[::-1]

    for idx, (inst, ax, scale_title) in enumerate(zip(instances, axes, scales_info)):
        vals = df[inst].values
        min_val = np.min(vals)

        for y in y_pos:
            ax.axhline(y, color='#EFEFEF', linestyle='-', linewidth=0.8, zorder=1)

        cur_min, cur_max = np.min(vals), np.max(vals)
        span = cur_max - cur_min
        ax.set_xlim(cur_min - span * 0.10, cur_max + span * 0.30)

        for i, (y, val, m_key) in enumerate(zip(y_pos, vals, df['Key'])):
            is_best = (val == min_val)
            is_ours = (m_key == 'HGP-PPO')

            color = COLORS[m_key]
            marker = 'o' if is_ours else ('s' if 'PPO' in m_key or 'DDQN' in m_key else ('D' if m_key == 'Beam Search' else ('^' if m_key == 'IG' else 'v')))
            size = 85 if is_ours else 55
            edgecolor = '#002E52' if is_ours else '#222222'
            lw = 1.4 if is_ours else 0.8

            ax.scatter(val, y, color=color, s=size, marker=marker, edgecolors=edgecolor, linewidths=lw, zorder=4)

            fontweight = 'bold' if is_best else 'normal'
            text_color = '#004A80' if (is_best and is_ours) else ('#111111' if is_best else '#555555')
            if (val - cur_min) / span > 0.82:
                ax.annotate(f"{val:.1f}", xy=(val, y), xytext=(-10, 0), textcoords='offset points',
                            va='center', ha='right', fontsize=8.8, fontweight=fontweight, color=text_color)
            else:
                ax.annotate(f"{val:.1f}", xy=(val, y), xytext=(11, 0), textcoords='offset points',
                            va='center', ha='left', fontsize=8.8, fontweight=fontweight, color=text_color)

        ax.set_title(scale_title, fontsize=9.5, fontweight='bold', pad=10, color='#1A252C')
        ax.set_xlabel(r"Cycle Span $H$ (h)", fontsize=9.5, fontweight='bold', labelpad=5)
        ax.tick_params(axis='both', labelsize=8.5)
        ax.grid(axis='x', color='#ECECEC', linestyle='--', linewidth=0.6, zorder=0)

        ax.set_ylim(-0.7, len(methods) - 0.3)
        best_idx = np.argmin(vals)
        best_name = methods[best_idx].split(' ')[0]
        ax.text(0.96, 0.05, f"Best: {best_name} ({min_val:.1f}h)", transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', ha='right', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#F8F9FA', edgecolor='#B0B0B0', linewidth=0.7))

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(methods, fontsize=9.5, fontweight='bold')

    fig.suptitle("Deterministic Initial Scheduling Performance across Scales (Table 5-6)", fontsize=11.5, fontweight='bold', y=0.98)
    save_fig_bundle(fig, "fig4_initial_cross_scale", df)


# ---------------------------------------------------------------------------
# Fig. 5: 决策配置质量与可行性 (左侧热图 + 右侧合法性矩阵)
# ---------------------------------------------------------------------------
def plot_fig5():
    rows = [
        'HGP-PPO (Full Model)',
        'w/o Baseline Conditioning',
        'Worker–Station Preassignment',
        'Homogeneous GraphSAGE',
        'Operation–Station Joint',
        'Operation-only'
    ]
    instances = ['283', '680', '2338', '3182']

    data = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [4.6, 39.6, 2.4, 10.0],
        [11.6, 96.2, 32.7, 63.0],
        [1.8, 21.5, 9.4, 19.6],
        [-6.8, 15.6, -4.6, 27.7],
        [-6.0, 59.3, np.nan, np.nan]
    ])

    initial_valid = ["4 / 4", "4 / 4", "4 / 4", "4 / 4", "4 / 4", "2 / 4"]
    resched_valid = ["100% (36/36)", "100% (36/36)", "N / R*", "100% (36/36)", "0% (0/36) [Collapse]", "100% (36/36)"]

    fig = plt.figure(figsize=(13.5, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 4, width_ratios=[4.2, 0.9, 1.4, 0.3], height_ratios=[1.0, 0.08],
                          wspace=0.12, hspace=0.35,
                          left=0.25, right=0.98, top=0.87, bottom=0.10)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_v1 = fig.add_subplot(gs[0, 1])
    ax_v2 = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[1, 0])

    norm = TwoSlopeNorm(vmin=-15.0, vcenter=0.0, vmax=80.0)
    cmap = mpl.colormaps['RdYlBu_r'].copy()
    cmap.set_bad(color='#EEEEEE')

    masked_data = np.ma.masked_invalid(data)
    im = ax_heat.imshow(masked_data, cmap=cmap, norm=norm, aspect='auto')

    for r in range(len(rows)):
        for c in range(len(instances)):
            val = data[r, c]
            if np.isnan(val):
                ax_heat.text(c, r, "Infeasible", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#666666')
                rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#E0E0E0', hatch='///', edgecolor='#AAAAAA', linewidth=0.6)
                ax_heat.add_patch(rect)
            else:
                prefix = "+" if val > 0 else ""
                txt = f"{prefix}{val:.1f}%" if abs(val) > 0.05 else "0.0%"
                text_color = '#FFFFFF' if abs(val) > 40 else '#111111'
                font_weight = 'bold' if (r == 0 or abs(val) > 20) else 'normal'
                ax_heat.text(c, r, txt, ha='center', va='center', fontsize=9.2, fontweight=font_weight, color=text_color)

    ax_heat.set_xticks(np.arange(len(instances)))
    ax_heat.set_xticklabels(instances, fontsize=10, fontweight='bold')
    ax_heat.set_xlabel("Instance Physical Scale", fontsize=9.5, fontweight='bold', labelpad=5)
    ax_heat.set_yticks(np.arange(len(rows)))
    ax_heat.set_yticklabels(rows, fontsize=9.5, fontweight='bold')
    ax_heat.set_title(r"Cycle Span Relative Difference $g_{v,j}$ (%)", fontsize=10.5, fontweight='bold', pad=8)

    cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"Relative Difference vs Full Model (%) [Blue: Shorter, Yellow: Baseline, Red: Worse]", fontsize=8.5, fontweight='bold', labelpad=3)

    ax_v1.set_title("Initial\nFeasible", fontsize=10, fontweight='bold', pad=8)
    ax_v1.set_xlim(-0.5, 0.5)
    ax_v1.set_ylim(-0.5, len(rows) - 0.5)
    ax_v1.invert_yaxis()
    ax_v1.axis('off')
    for r, val_str in enumerate(initial_valid):
        bg_col = '#EAF7EE' if '4 / 4' in val_str else '#FDEDEC'
        fg_col = '#1E8449' if '4 / 4' in val_str else '#C0392B'
        border_col = '#A9DFBF' if '4 / 4' in val_str else '#F5B7B1'
        rect = mpatches.Rectangle((-0.45, r - 0.42), 0.9, 0.84, facecolor=bg_col, edgecolor=border_col, linewidth=0.8)
        ax_v1.add_patch(rect)
        ax_v1.text(0, r, val_str, ha='center', va='center', fontsize=9.5, fontweight='bold', color=fg_col)

    ax_v2.set_title("Reschedule Legality\n(36 Scenarios)", fontsize=10, fontweight='bold', pad=8)
    ax_v2.set_xlim(-0.5, 0.5)
    ax_v2.set_ylim(-0.5, len(rows) - 0.5)
    ax_v2.invert_yaxis()
    ax_v2.axis('off')
    for r, val_str in enumerate(resched_valid):
        if '100%' in val_str:
            bg_col, fg_col, border_col = '#EAF7EE', '#1E8449', '#A9DFBF'
        elif '0%' in val_str:
            bg_col, fg_col, border_col = '#FDEDEC', '#922B21', '#E6B0AA'
        else:
            bg_col, fg_col, border_col = '#F4F6F7', '#566573', '#D5D8DC'
        rect = mpatches.Rectangle((-0.45, r - 0.42), 0.9, 0.84, facecolor=bg_col, edgecolor=border_col, linewidth=0.8)
        ax_v2.add_patch(rect)
        ax_v2.text(0, r, val_str, ha='center', va='center', fontsize=9.0, fontweight='bold', color=fg_col)

    fig.suptitle("Architectural Configuration Comparison: Quality & Feasibility Boundaries (Table 5-7)", 
                 fontsize=12, fontweight='bold', y=0.98)
    
    source_df = pd.DataFrame({
        'Configuration': rows,
        'Diff_283_pct': data[:, 0],
        'Diff_680_pct': data[:, 1],
        'Diff_2338_pct': data[:, 2],
        'Diff_3182_pct': data[:, 3],
        'Initial_Feasible': initial_valid,
        'Reschedule_Legality': resched_valid
    })
    save_fig_bundle(fig, "fig5_configuration_quality_feasibility", source_df)


# ---------------------------------------------------------------------------
# Fig. 7: 重调度效率与稳定性分项权衡 (2x3 面板水平点图)
# ---------------------------------------------------------------------------
def plot_fig7():
    data = {
        'Method': [
            'HGP-PPO (Ours)',
            'StabilityAwareRepair',
            'L2D-PPO-APAL',
            'Graph-DDQN-APAL',
            'HybridCPMStabilityRepair',
            'TaktAwareRepair',
            'BeamSearchRepair',
            'IteratedGreedyRepair (IG)'
        ],
        'Key': [
            'HGP-PPO',
            'StabilityAwareRepair',
            'L2D-PPO-APAL',
            'Graph-DDQN-APAL',
            'HybridCPMStabilityRepair',
            'TaktAwareRepair',
            'BeamSearchRepair',
            'IteratedGreedyRepair'
        ],
        'Cycle_Span_H': [976.49, 993.85, 988.33, 978.32, 964.87, 1055.84, 1116.52, 1135.94],
        'Delta_ref_h': [29.51, 45.22, 42.11, 29.63, 18.79, 106.50, 167.19, 186.61],
        'Start_Dev_h': [30.85, 33.88, 45.54, 56.31, 61.76, 74.90, 108.85, 148.31],
        'Station_Change_pct': [2.12, 1.73, 1.93, 2.02, 1.85, 1.67, 0.54, 0.65],
        'Team_Change_pct': [95.60, 94.76, 95.87, 96.05, 93.53, 2.06, 0.70, 0.93],
        'Score_S': [0.851100, 0.877564, 0.915151, 0.980965, 0.982793, 0.983300, 1.188985, 1.440700]
    }
    df = pd.DataFrame(data)

    panels = [
        ('Cycle_Span_H', r'(a) Cycle Span $H$ (h)', 'h'),
        ('Delta_ref_h', r'(b) Baseline Span Excess $\Delta_{\mathrm{ref}}$ (h)', 'h'),
        ('Start_Dev_h', r'(c) Mean Start Deviation $\bar{D}_{\mathrm{start}}$ (h)', 'h'),
        ('Station_Change_pct', r'(d) Station Change $R_{\mathrm{sta}}$ (%)', '%'),
        ('Team_Change_pct', r'(e) Team-set Change $R_{\mathrm{team}}$ (%)', '%'),
        ('Score_S', r'(f) Weighted Composite Score $S$', 'Score')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 6.8), sharey=True, dpi=300)
    fig.subplots_adjust(wspace=0.15, hspace=0.35, left=0.21, right=0.98, top=0.91, bottom=0.08)

    methods = df['Method'].tolist()
    y_pos = np.arange(len(methods))[::-1]

    axes_flat = axes.flatten()

    for ax, (metric_col, title, unit) in zip(axes_flat, panels):
        vals = df[metric_col].values
        min_val = np.min(vals)
        max_val = np.max(vals)
        val_span = max_val - min_val if max_val > min_val else 1.0

        for y in y_pos:
            ax.axhline(y, color='#F0F0F0', linestyle='-', linewidth=0.8, zorder=1)

        ax.set_xlim(min_val - val_span * 0.10, max_val + val_span * 0.26)
        ax.set_ylim(-0.7, len(methods) - 0.3)

        for y, val, m_key in zip(y_pos, vals, df['Key']):
            is_ours = (m_key == 'HGP-PPO')
            color = COLORS[m_key]
            marker = 'o' if is_ours else ('s' if 'PPO' in m_key or 'DDQN' in m_key else ('D' if 'Beam' in m_key or 'Greedy' in m_key else '^'))
            size = 80 if is_ours else 50
            edgecolor = '#002E52' if is_ours else '#222222'
            lw = 1.4 if is_ours else 0.8

            ax.scatter(val, y, color=color, s=size, marker=marker, edgecolors=edgecolor, linewidths=lw, zorder=4)

            txt = f"{val:.4f}" if unit == 'Score' else (f"{val:.2f}" if unit == '%' else f"{val:.1f}")
            is_best = (val == min_val)
            fw = 'bold' if (is_best or is_ours) else 'normal'
            tc = '#004A80' if is_ours else ('#111111' if is_best else '#555555')
            
            if (val - min_val) / val_span > 0.78:
                ax.annotate(txt, xy=(val, y), xytext=(-10, 0), textcoords='offset points',
                            va='center', ha='right', fontsize=8.5, fontweight=fw, color=tc)
            else:
                ax.annotate(txt, xy=(val, y), xytext=(10, 0), textcoords='offset points',
                            va='center', ha='left', fontsize=8.5, fontweight=fw, color=tc)

        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6, color='#222222')
        ax.set_xlabel(f"Lower is preferred ({unit})", fontsize=8.2, color='#444444', labelpad=4)
        ax.tick_params(axis='both', labelsize=8.5)
        ax.grid(axis='x', color='#EEEEEE', linestyle='--', linewidth=0.6, zorder=0)

    for r in range(2):
        axes[r, 0].set_yticks(y_pos)
        axes[r, 0].set_yticklabels(methods, fontsize=9.2, fontweight='bold')

    fig.suptitle("Efficiency–Stability Multi-Objective Trade-offs in Template Rescheduling (Table 5-9)", 
                 fontsize=11.5, fontweight='bold', y=0.98)
    save_fig_bundle(fig, "fig7_efficiency_stability_tradeoffs", df)


# ---------------------------------------------------------------------------
# Fig. 8: 端到端求解时间随规模变化 (对数纵轴折线图)
# ---------------------------------------------------------------------------
def plot_fig8():
    instances = [283, 680, 2338, 3182]
    
    data = {
        'Method': [
            'HGP-PPO (Ours, Online Inference)',
            'BeamSearchRepair (Search)',
            'IteratedGreedyRepair (Search)',
            'SimulatedAnnealingRepair (Search)'
        ],
        'Key': ['HGP-PPO', 'BeamSearchRepair', 'IG', 'SA'],
        283: [8.02, 114.02, 147.17, 292.42],
        680: [12.84, 262.72, 410.42, 676.78],
        2338: [51.91, 1787.00, 2416.80, 3445.39],
        3182: [123.60, 3902.89, 4202.01, 4296.64]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.15)

    x_vals = np.array(instances)
    markers = ['o', 's', '^', 'D']

    for i, row in df.iterrows():
        m_key = row['Key']
        label = row['Method']
        times = [row[inst] for inst in instances]
        is_ours = (m_key == 'HGP-PPO')

        color = COLORS[m_key] if m_key in COLORS else '#333333'
        lw = 2.4 if is_ours else 1.3
        marker = markers[i]
        ms = 8.5 if is_ours else 6.5

        ax.plot(x_vals, times, label=label, color=color, linewidth=lw, marker=marker,
                markersize=ms, markeredgecolor='#111111', markeredgewidth=0.8, zorder=4 if is_ours else 3)

        if is_ours:
            for x, y in zip(x_vals, times):
                ax.annotate(f"{y:.1f}s", xy=(x, y), xytext=(0, 9), textcoords='offset points',
                            fontsize=8.5, fontweight='bold', color='#004A80', ha='center')

    ax.set_yscale('log')
    ax.set_ylim(4, 10000)
    ax.set_xticks(instances)
    ax.set_xticklabels([str(x) for x in instances], fontsize=9.5, fontweight='bold')
    ax.set_xlabel("Instance Scale (Physical Operation Count)", fontsize=9.5, fontweight='bold', labelpad=6)
    ax.set_ylabel("End-to-End Solve Time (s, Log Scale)", fontsize=9.5, fontweight='bold', labelpad=6)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, color='#E0E0E0', zorder=0)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#F0F0F0', zorder=0)

    ax.annotate("HGP-PPO Online Inference:\n1–2 orders of magnitude faster\nthan metaheuristic search (8–124 s)",
                xy=(2338, 51.91), xytext=(1150, 16),
                arrowprops=dict(arrowstyle="->", color='#0072B2', lw=1.2),
                fontsize=8.5, fontweight='bold', color='#004A80',
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#EBF3FA", edgecolor="#0072B2", alpha=0.95))

    ax.legend(loc='upper left', fontsize=8.2, framealpha=0.95, edgecolor='#D0D0D0')
    ax.set_title("End-to-End Solve Time vs. Physical Operation Scale (Table XI)", fontsize=10.5, fontweight='bold', pad=10)

    save_fig_bundle(fig, "fig8_runtime_scalability", df)


if __name__ == '__main__':
    print("Starting generation of Paper Stage 1 Figures...")
    plot_fig4()
    plot_fig5()
    plot_fig7()
    plot_fig8()
    print("[SUCCESS] All 4 Stage 1 figures generated in:", OUTPUT_DIR.resolve())
