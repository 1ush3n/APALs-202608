"""Assemble the complete 9-slide APAL Paper Integrated Figures v1 PPTX deck.

Slides:
  1. fig1_apal_unfolding_delay  — New Fig. 1 (A2 + A5 integrated, ~2.28:1)
  2. fig2_compact_b1_b5         — Compact Fig. 2(a)-(b) (B1 + B5 fused, ~2.24:1)
  3. fig2_standalone_b1         — High-Clarity Standalone Fig. 2 (B1, ~2.15:1)
  4. fig3_standalone_b5         — High-Clarity Standalone Fig. 3 (B5, ~1.27:1)
  5. fig_appA_b8_worker_pointer — New Standalone Method Fig. (B8, Appendix A / Sec. IV-C)
  6. extra_b3_hetero_gatv2      — Clean standalone backup: B3 (Hetero-GATv2)
  7. extra_b6_op_pointer        — Clean standalone backup: B6 (Operation Pointer)
  8. extra_b10_critic_head      — Clean standalone backup: B10 (Critic Value Head)
  9. extra_b11_ppo_gae          — Clean standalone backup: B11 (PPO & GAE)
"""
from __future__ import annotations

import os
import sys
from pptx import Presentation
from pptx.util import Inches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
V8_SCRIPTS = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\refined_v8\scripts"

sys.path.insert(0, SCRIPT_DIR)
sys.path.append(V8_SCRIPTS)

from build_fig1_a2_a5 import build_fig1_a2_a5
from build_fig2_compact_b1_b5 import build_fig2_compact_b1_b5
from build_fig3_standalone_b5 import build_fig3_standalone_b5
from build_b1_v8 import build_b1_v8
from build_b8_v8 import build_b8_v8
from build_b3_v8 import build_b3_v8
from build_b6_v8 import build_b6_v8
from build_b10_v8 import build_b10_v8
from build_b11_v8 import build_b11_v8

OUT_PPTX = os.path.join(BASE_DIR, "APAL_Paper_Integrated_Figures_v1.pptx")

SLIDE_MANIFEST = [
    ("fig1_apal_unfolding_delay",  "New Fig. 1 (A2+A5): Pulsating Unfolding & Material Delay Repair", build_fig1_a2_a5),
    ("fig2_compact_b1_b5",         "Compact Fig. 2(a)-(b) (B1+B5): Workcell -> State Graph -> Dual-Tower HGP-PPO", build_fig2_compact_b1_b5),
    ("fig2_standalone_b1",         "Standalone Fig. 2 (B1): Physical APAL Workcell -> Heterogeneous State Graph", build_b1_v8),
    ("fig3_standalone_b5",         "Standalone Fig. 3 (B5): Independent Dual-Tower HGP-PPO Architecture", build_fig3_standalone_b5),
    ("fig_appA_b8_worker_pointer", "Appendix A / Sec. IV-C (B8): Autoregressive Worker Pointer (r = 1 -> 2)", build_b8_v8),
    ("extra_b3_hetero_gatv2",      "Extra Backup (B3): Relation-Specific Multi-Head GATv2 Layer", build_b3_v8),
    ("extra_b6_op_pointer",        "Extra Backup (B6): Operation Pointer & Feasibility Masking", build_b6_v8),
    ("extra_b10_critic_head",      "Extra Backup (B10): Critic Attention Pooling & Value Head", build_b10_v8),
    ("extra_b11_ppo_gae",          "Extra Backup (B11): PPO Replay, GAE & Dual-Optimizer Updates", build_b11_v8),
]


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for idx, (slug, desc, builder) in enumerate(SLIDE_MANIFEST, 1):
        slide = prs.slides.add_slide(blank_layout)
        builder(slide)
        print(f"[{idx}/9] Built {slug}: {desc}", flush=True)

    prs.save(OUT_PPTX)
    print(f"\nSaved native editable deck -> {OUT_PPTX}", flush=True)


if __name__ == "__main__":
    main()
