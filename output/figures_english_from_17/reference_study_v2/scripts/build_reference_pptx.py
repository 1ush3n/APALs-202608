import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
BOARDS_DIR = os.path.join(BASE, "boards")
MANIFEST = os.path.join(BASE, "crop_manifest_v2.json")
OUT_PPTX = os.path.join(BASE, "Top_Venue_Reference_Gallery_v2.pptx")

with open(MANIFEST, "r", encoding="utf-8") as f:
    ITEMS = json.load(f)

NAVY = RGBColor(20, 48, 82)
BLUE = RGBColor(22, 92, 148)
TEAL = RGBColor(16, 110, 92)
PURPLE = RGBColor(96, 54, 142)
AMBER = RGBColor(180, 95, 20)
SLATE = RGBColor(55, 70, 90)
LIGHT_BG = RGBColor(244, 247, 251)
CARD_BG = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(190, 202, 216)
MUTED = RGBColor(95, 110, 128)
DARK_TXT = RGBColor(22, 30, 42)
WHITE = RGBColor(255, 255, 255)

def set_bg(slide, rgb):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb

def add_rect(slide, x, y, w, h, fill=CARD_BG, border=BORDER_GRAY, border_pt=1.0, rounded=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.04
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if border is not None:
        shp.line.color.rgb = border
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    return shp

def add_txt(slide, x, y, w, h, text, size=11, bold=False, color=DARK_TXT, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Segoe UI"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb

def add_notes(slide, text):
    ns = slide.notes_slide
    ns.notes_text_frame.text = text

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.500)
blank = prs.slide_layouts[6]

# ---------------- SLIDE 1: COVER & OVERVIEW ----------------
s1 = prs.slides.add_slide(blank)
set_bg(s1, LIGHT_BG)
add_rect(s1, 0, 0, 13.333, 1.38, fill=NAVY, border=None)
add_txt(s1, 0.45, 0.20, 12.4, 0.50, "Top-Venue Scientific Figure Reference Atlas (v2)", size=24, bold=True, color=WHITE)
add_txt(
    s1, 0.45, 0.74, 12.4, 0.45,
    "36 Curated 300-DPI Figures from NeurIPS, ICLR, ICML, KDD, AAAI, AAMAS, SC, OSDI, USENIX ATC, SIGCOMM, IROS & IEEE TNNLS",
    size=12.5, bold=False, color=RGBColor(200, 222, 242)
)

cats = [
    ("Category 1  |  16 Figures (Slides 2-5)", "Fig. 1 Redesign: Problem Formulation, Cyclic Pipeline & Dynamic Rescheduling", BLUE,
     "Addresses the main user critique ('Fig. 1 still lacks top-CS polish'). Covers rolling-horizon frozen/active separation (L-RHO ICLR'25), rollback recovery (Zero Bubble ICLR'24), dynamic graph breakdown/repair (ISBJSSP, Train Resched), cyclic multi-stage waves (OctoPipe SC'26, Chimera, Hanayo, TeraPipe), and step-by-step partial Gantts (Residual Sched)."),
    ("Category 2  |  8 Figures (Slides 7-8)", "Fig. 2 Refinement: Physical Workcell -> Hetero Graph -> GNN-RL", TEAL,
     "Shows how top robotics, systems, and scheduling venues fuse physical domain scenes seamlessly into formal graph states and encoders: S2A volumetric-node links, ScheduleNet zoom-in timeline, HybridNet seamless dual-zone frame, Decima & DAN closed loops, HGS type-split encoders, and OctoPipe hardware+DAG+schema cards."),
    ("Category 3  |  12 Figures (Slides 9-11)", "Fig. 3 Integration Question: Unifying Dual-Tower Encoder + 3-Tier Decoder", PURPLE,
     "Directly answers whether APAL Fig. 3 can merge encoding and decoding on one canvas. Highlights 2D-Ptr (AAMAS'24) dual-tower encoder + joint pointer decoder, HGS 3-stage conditional mask decoder, TARGETNET (NeurIPS'25) macro+micro pairing, LEHD/MatNet zoom callouts, Decima factored softmax, and Wheatley PPO Actor+Critic.")
]
for idx, (tag, heading, col, body) in enumerate(cats):
    cx = 0.45 + idx * 4.22
    add_rect(s1, cx, 1.62, 4.00, 3.80, fill=CARD_BG, border=BORDER_GRAY, border_pt=1.2, rounded=True)
    add_rect(s1, cx, 1.62, 4.00, 0.44, fill=col, border=None)
    add_txt(s1, cx + 0.14, 1.67, 3.72, 0.34, tag, size=11.5, bold=True, color=WHITE)
    add_txt(s1, cx + 0.14, 2.18, 3.72, 0.68, heading, size=13, bold=True, color=NAVY)
    add_txt(s1, cx + 0.14, 2.94, 3.72, 2.34, body, size=10.5, color=SLATE)

add_rect(s1, 0.45, 5.66, 12.43, 1.52, fill=RGBColor(232, 240, 248), border=RGBColor(168, 190, 214), border_pt=1.1, rounded=True)
add_txt(s1, 0.65, 5.78, 12.0, 0.32, "Audit & Compliance Guarantees", size=12, bold=True, color=NAVY)
add_txt(
    s1, 0.65, 6.12, 12.0, 0.96,
    "- Strict Zero-Duplicate Rule: Every one of the 36 curated figures is disjoint from the 15 figures previously collected in figure_study_15.\n"
    "- Prior Integrated Deck Status: Centering fix in fig3_standalone_b5.png & fig2_compact_b1_b5.png rebuilt (9/9 QA PASS) without touching refined_v8/.\n"
    "- Actionable Output (Slides 6 & 12): Concrete Fig. 1 Redesign Blueprints (A/B/C) and Fig. 3 Encoder-Decoder Architecture Tradeoffs (Option 3A vs 3B).",
    size=10.2, color=SLATE
)
add_notes(s1, "Executive overview of the 36-figure Top-Venue Reference Atlas v2 prepared for APAL Fig. 1, Fig. 2, and Fig. 3 redesign discussions.")

# Helper for Board slides (2-5, 7-8, 9-11)
def add_board_slide(board_png, header_title, header_rgb, items_chunk, top_pick_id, synthesis_note):
    sl = prs.slides.add_slide(blank)
    set_bg(sl, LIGHT_BG)
    add_rect(sl, 0, 0, 13.333, 0.62, fill=header_rgb, border=None)
    add_txt(sl, 0.28, 0.12, 12.75, 0.38, header_title, size=15.5, bold=True, color=WHITE)

    # Board image sized proportionally: H = 6.62 in -> W = 9.807 in
    add_rect(sl, 0.20, 0.72, 9.86, 6.66, fill=CARD_BG, border=BORDER_GRAY, border_pt=1.0)
    sl.shapes.add_picture(board_png, Inches(0.22), Inches(0.74), height=Inches(6.62))

    # Right Analysis Sidebar (x=10.20..13.13, w=2.93)
    sx, sw = 10.20, 2.93
    add_rect(sl, sx, 0.72, sw, 6.66, fill=CARD_BG, border=BORDER_GRAY, border_pt=1.1, rounded=True)
    add_rect(sl, sx, 0.72, sw, 0.40, fill=header_rgb, border=None)
    add_txt(sl, sx + 0.10, 0.77, sw - 0.20, 0.30, "APAL Design Insights", size=11.5, bold=True, color=WHITE)

    y_cur = 1.20
    for it in items_chunk:
        is_top = (it["id"] == top_pick_id)
        box_fill = RGBColor(236, 245, 255) if is_top else RGBColor(248, 250, 252)
        box_bdr = BLUE if is_top else RGBColor(214, 222, 232)
        bpt = 1.4 if is_top else 0.8
        add_rect(sl, sx + 0.12, y_cur, sw - 0.24, 1.21, fill=box_fill, border=box_bdr, border_pt=bpt, rounded=True)
        badge = f"{it['id']} [TOP PICK]" if is_top else it["id"]
        tb = sl.shapes.add_textbox(Inches(sx + 0.16), Inches(y_cur + 0.04), Inches(sw - 0.32), Inches(1.13))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.04)
        tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.02)
        tf.margin_bottom = Inches(0.02)

        p1 = tf.paragraphs[0]
        p1.space_after = Pt(1.5)
        r1 = p1.add_run()
        r1.text = f"{badge} | {it['venue']}"
        r1.font.name = "Segoe UI"
        r1.font.size = Pt(8.8)
        r1.font.bold = True
        r1.font.color.rgb = header_rgb

        p2 = tf.add_paragraph()
        p2.space_after = Pt(2.0)
        r2 = p2.add_run()
        r2.text = it["short_title"]
        r2.font.name = "Segoe UI"
        r2.font.size = Pt(8.6)
        r2.font.bold = True
        r2.font.color.rgb = DARK_TXT

        tk = it["takeaway"]
        if len(tk) > 105:
            tk = tk[:102] + "..."
        p3 = tf.add_paragraph()
        r3 = p3.add_run()
        r3.text = tk
        r3.font.name = "Segoe UI"
        r3.font.size = Pt(7.8)
        r3.font.color.rgb = SLATE
        y_cur += 1.25

    # Bottom synthesis box inside sidebar
    add_rect(sl, sx + 0.12, 6.24, sw - 0.24, 1.02, fill=RGBColor(254, 247, 234), border=RGBColor(218, 172, 108), border_pt=1.0, rounded=True)
    add_txt(sl, sx + 0.18, 6.28, sw - 0.36, 0.22, "Board Synthesis for APAL:", size=8.8, bold=True, color=AMBER)
    add_txt(sl, sx + 0.18, 6.49, sw - 0.36, 0.72, synthesis_note, size=8.0, color=DARK_TXT)

    note_lines = [f"{it['id']} ({it['venue']}, {it['fig']}): {it['paper_title']} | {it['url']}\nTakeaway: {it['takeaway']}" for it in items_chunk]
    add_notes(sl, "\n\n".join(note_lines))

cat1 = [x for x in ITEMS if x["category"] == "cat1_fig1_problem"]
cat2 = [x for x in ITEMS if x["category"] == "cat2_fig2_fusion"]
cat3 = [x for x in ITEMS if x["category"] == "cat3_fig3_encdec"]

c1_meta = [
    ("C1_Board_01.png", "Category 1 (Board 1/4): Rolling-Horizon Repair & Disruption Graph Mutations", "C1_01",
     "Combine L-RHO's frozen vs active cycle (C1_01) with Train Rescheduling's red-to-green delay propagation link (C1_04)."),
    ("C1_Board_02.png", "Category 1 (Board 2/4): Multi-Stage Pipeline Unfolding & Wavefront Comparison", "C1_05",
     "Replace Fig. 1(a)'s flat grid with OctoPipe's stacked evolution (C1_05) or Hanayo/Chimera wave fronts + station load bars."),
    ("C1_Board_03.png", "Category 1 (Board 3/4): Progressive Timeline Cursors & Bubble Elimination", "C1_09",
     "Adopt Residual Scheduling's red tau_t clock dividing frozen prefix vs active candidate queue (C1_09) with Zero-Bubble clarity (C1_10)."),
    ("C1_Board_04.png", "Category 1 (Board 4/4): Problem-Solver Trio, Matrix-to-Gantt & Resource Reservations", "C1_13",
     "Use MatNet's matrix-to-Gantt link (C1_14) or PPipe's probe-vs-commit resource tracks (C1_15) for multi-worker station reservations."),
]
for idx, (fn, title, top_id, syn) in enumerate(c1_meta):
    add_board_slide(os.path.join(BOARDS_DIR, fn), title, BLUE, cat1[idx*4:(idx+1)*4], top_id, syn)

# ---------------- SLIDE 6: FIG. 1 REDESIGN BLUEPRINTS ----------------
s6 = prs.slides.add_slide(blank)
set_bg(s6, LIGHT_BG)
add_rect(s6, 0, 0, 13.333, 0.62, fill=NAVY, border=None)
add_txt(s6, 0.35, 0.12, 12.6, 0.38, "Fig. 1 Diagnosis & Three Concrete Top-Venue Redesign Blueprints", size=16.5, bold=True, color=WHITE)

add_rect(s6, 0.40, 0.78, 12.53, 1.05, fill=RGBColor(253, 242, 240), border=RGBColor(212, 120, 108), border_pt=1.2, rounded=True)
add_txt(s6, 0.56, 0.85, 12.2, 0.28, "Why Current fig1_apal_unfolding_delay.png Falls Short of NeurIPS / ICLR / SC Visual Quality:", size=11.5, bold=True, color=RGBColor(165, 42, 34))
add_txt(
    s6, 0.56, 1.14, 12.2, 0.62,
    "1) Panel (a) looks like a static PowerPoint table/matrix of uniform rectangles with repeated clip-art planes rather than a rhythmic multi-station unfolding workflow.\n"
    "2) Panel (b) is essentially a 22-row matplotlib lollipop plot wrapped in a large callout box — it lacks structural graph semantics and looks like a results chart instead of a problem formulation.",
    size=10.2, color=DARK_TXT
)

blueprints = [
    ("Blueprint A  [RECOMMENDED]", "L-RHO + Zero Bubble + Train Rescheduling Style", BLUE,
     "Best balance of mechanism depth & NeurIPS/ICLR clarity",
     "Left (a) — Cyclic Pulse Flow Strip (C1_05 / C1_07 / C1_10):\n"
     "Show 5 stations (S1-S5) along time pulses H using crisp interlocking F/B-style Gantt bars (not empty grey boxes) with ONE bold diagonal path tracing Aircraft k and a subtle right capacity bar (skill load / slots c_m).\n\n"
     "Right (b) — Two-Stage Disruption -> Stability Repair Loop (C1_01 + C1_02 + C1_04):\n"
     "Top-left mini-Gantt (Frozen prefix t <= tau_r shaded grey; O94/O111 hit by delay Delta_i in orange) -> Center precedence/worker-conflict mini-graph showing red delayed edge propagating -> Bottom-right Repaired Gantt comparing H0 vs Hr."),
    ("Blueprint B", "ISBJSSP Graph Mutation + Residual Clock Style", TEAL,
     "Emphasizes topological graph repair under dynamic disruption",
     "Top Strip — 3-Panel Graph State Evolution (C1_03):\n"
     "(i) Nominal Baseline DAG G0 (committed + active ops) -> (ii) Trigger tau_r Disruption (delayed material node O111 turns red; shared worker W5 contention edge highlighted) -> (iii) Stability-Preserving Repaired DAG G_r (prefix locked, suffix re-timed).\n\n"
     "Bottom Strip — Aligned Dual Gantt Tracks (C1_05 + C1_09):\n"
     "Vertical red dashed clock at tau_r=65.83h cleanly locks left committed blocks and highlights only 3-4 representative ripple ops (e.g., O94, O111, O223) rather than 22 noisy lollipop rows."),
    ("Blueprint C", "TARGETNET + MatNet + PPipe Systems Style", PURPLE,
     "Compact single-glance NeurIPS/OSDI Figure 1 problem statement",
     "Three-Column Horizontal Narrative (C1_13 + C1_14 + C1_15):\n"
     "Col 1 (APAL Constraints): Aircraft pulse cadence H + Precedence + Trade Skills K + Worker Fatigue/Multi-man q_i.\n\n"
     "Col 2 (Cyclic Schedule + Delay Trigger tau_r): Compact 5-station Gantt matrix (C1_14) overlaid with PPipe-style candidate-probe vs reserved worker bars (C1_15) illustrating why tau_r != r_i triggers cascading worker wait.\n\n"
     "Col 3 (Stability Objective): Side-by-side comparison badge of Frozen Stability vs Makespan Recovery (H0=292.6h vs Hr=305.9h).")
]
for idx, (btitle, bsub, col, badge, desc) in enumerate(blueprints):
    bx = 0.40 + idx * 4.24
    add_rect(s6, bx, 2.02, 4.05, 5.18, fill=CARD_BG, border=col, border_pt=1.5 if idx == 0 else 1.1, rounded=True)
    add_rect(s6, bx, 2.02, 4.05, 0.68, fill=col, border=None)
    add_txt(s6, bx + 0.14, 2.06, 3.77, 0.30, btitle, size=13, bold=True, color=WHITE)
    add_txt(s6, bx + 0.14, 2.36, 3.77, 0.28, bsub, size=9.5, color=RGBColor(228, 238, 250))
    add_rect(s6, bx + 0.14, 2.78, 3.77, 0.46, fill=RGBColor(238, 244, 250), border=None, rounded=True)
    add_txt(s6, bx + 0.20, 2.81, 3.65, 0.40, badge, size=9.0, bold=True, color=col, valign=MSO_ANCHOR.MIDDLE)
    add_txt(s6, bx + 0.14, 3.32, 3.77, 3.78, desc, size=9.4, color=DARK_TXT)

add_notes(s6, "Actionable diagnosis of current fig1_apal_unfolding_delay.png and 3 concrete redesign blueprints built directly from Category 1 top-venue references.")

# ---------------- SLIDES 7-8: CAT 2 ----------------
c2_meta = [
    ("C2_Board_01.png", "Category 2 (Board 1/2): Fusing Physical Workcell / 3D Assembly into Graph RL", "C2_03",
     "Keep fig2_compact_b1_b5's shop-floor photo on the left, but align node callout pills seamlessly into the State Graph seam like HybridNet (C2_03)."),
    ("C2_Board_02.png", "Category 2 (Board 2/2): Factory State -> Heterogeneous Encoders -> Structured Cards", "C2_08",
     "Borrow OctoPipe's structured type-colored attribute cards (C2_08) and ISBJSSP's single-node zoom inset (C2_07) to declutter x_M/x_O/x_K/x_W vectors."),
]
for idx, (fn, title, top_id, syn) in enumerate(c2_meta):
    add_board_slide(os.path.join(BOARDS_DIR, fn), title, TEAL, cat2[idx*4:(idx+1)*4], top_id, syn)

# ---------------- SLIDES 9-11: CAT 3 ----------------
c3_meta = [
    ("C3_Board_01.png", "Category 3 (Board 1/3): Unified Dual-Tower Encoders + Autoregressive Pointer Decoders", "C3_01",
     "2D-Ptr (C3_01) & HGS (C3_02) prove dual-tower encoding + cascading 3-tier pointer decoding integrate cleanly into a single coherent figure."),
    ("C3_Board_02.png", "Category 3 (Board 2/3): Macro+Micro Layer Callouts & Modular Enc-Dec Frameworks", "C3_06",
     "LEHD (C3_06) & MatNet (C3_07) show how to keep the Macro flow crisp while extracting ONE zoomed GATv2/Attention micro-callout on the side."),
    ("C3_Board_03.png", "Category 3 (Board 3/3): Unrolled Pointer Steps, Step Re-Encoding & Actor-Critic PPO", "C3_12",
     "Combine AM's step masking visual (C3_09) with Wheatley's compact GNN -> Policy/Value split + PPO dashed loop (C3_12)."),
]
for idx, (fn, title, top_id, syn) in enumerate(c3_meta):
    add_board_slide(os.path.join(BOARDS_DIR, fn), title, PURPLE, cat3[idx*4:(idx+1)*4], top_id, syn)

# ---------------- SLIDE 12: FIG 3 ENCODER-DECODER INTEGRATION VERDICT ----------------
s12 = prs.slides.add_slide(blank)
set_bg(s12, LIGHT_BG)
add_rect(s12, 0, 0, 13.333, 0.62, fill=PURPLE, border=None)
add_txt(s12, 0.35, 0.12, 12.6, 0.38, "Fig. 3 Architecture Decision: Should We Unify Encoder + Decoder (+ Critic)?", size=16.5, bold=True, color=WHITE)

add_rect(s12, 0.40, 0.78, 12.53, 0.90, fill=RGBColor(242, 236, 250), border=PURPLE, border_pt=1.2, rounded=True)
add_txt(s12, 0.56, 0.85, 12.2, 0.28, "Direct Evidence-Backed Answer from Category 3 Top-Venue Papers: YES — Unified Enc-Dec is Standard at NeurIPS / AAMAS / SIGCOMM.", size=11.5, bold=True, color=PURPLE)
add_txt(
    s12, 0.56, 1.15, 12.2, 0.46,
    "However, current B5 already spends ~55% height repeating identical 4-head GATv2 layers inside both Actor and Critic. "
    "Simply pasting B6+B8 onto B5 will overcrowd one column-width figure unless we reorganize Hierarchy + Micro-Callouts as shown below:",
    size=10.2, color=DARK_TXT
)

opts = [
    ("Option 3A  [RECOMMENDED FOR PAPER COHESION]", "Unified Macro-to-Micro Dual-Tower Enc + 3-Tier Decoder", BLUE,
     "Anchored to: 2D-Ptr (C3_01) + LEHD (C3_06) + MatNet (C3_07) + HGS (C3_02)",
     "Layout (Double-Column Width, Clean 3-Zone Flow):\n"
     "- Zone 1 (Dual-Tower Heterogeneous Encoders): Compact stacked x5 HGP-GATv2 tower blocks for Actor (theta) and Critic (phi) separated by the Disjoint Pill — move the detailed internal block diagram (Linear -> LReLU -> GATv2 -> Softmax -> LN) into ONE right/bottom Micro-Zoom Inset (just like LEHD Fig. 1 & MatNet Fig. 2b).\n"
     "- Zone 2 (Pooled Graph -> Cascading 3-Tier Autoregressive Decoder): Directly extend H_theta^(5) + g_theta into HGS-style chained heads: Op Ptr i_t -> Station Mask & Ptr m_t -> Worker Mask & Ptr W_t (with Draw 1 -> Draw 2 feasibility mask callout for multi-worker slots q_i > 1).\n"
     "- Zone 3 (Critic & PPO Coupling): Value MLP V_phi(s_t) feeding GAE advantage A_hat_t back across the top rail to update both decoder and encoder.\n\n"
     "Pros: One self-contained Section IV-A architecture figure telling the full APAL algorithm story; saves a full figure slot!"),
    ("Option 3B  [ALTERNATIVE IF FULL DETAIL NEEDED]", "Macro Architecture Fig. 3 + Enlarged 3-Tier Worker Trace Fig. 4", TEAL,
     "Anchored to: RL4CO (C3_05) + AM Decoder (C3_09) + Decima (C3_04) + Wheatley (C3_12)",
     "Layout (Two Focused Companion Figures):\n"
     "- Main Fig. 3 (Clean Unified Overview): Keeps our polished Dual-Tower B5 style (or simplifies Critic tower into a sleek mirrored block 'Shared Topology, Disjoint Weights') and expands the top Policy dashed box into explicit Op -> St -> Wk conditional mask arrows without showing multi-worker Draw 1 / Draw 2 numerical probabilities.\n"
     "- Dedicated Fig. 4 / Appendix A (Mechanistic Execution Trace): Uses our detailed B8 + B6 numerical probability comparison (Draw 1 r=1 selects W2 prob 0.52 -> Dynamic mask zeros W2 -> Draw 2 r=2 selects W5 prob 0.64) — matching AM Fig. 2 (C3_09) multi-step trace clarity.\n\n"
     "Pros: Zero micro-font crowding in single-column print; best readability if IEEE TSMC page limit allows an extra method figure.")
]
for idx, (otitle, osub, col, anchor, desc) in enumerate(opts):
    ox = 0.40 + idx * 6.36
    add_rect(s12, ox, 1.84, 6.16, 5.36, fill=CARD_BG, border=col, border_pt=1.5 if idx == 0 else 1.1, rounded=True)
    add_rect(s12, ox, 1.84, 6.16, 0.66, fill=col, border=None)
    add_txt(s12, ox + 0.16, 1.89, 5.84, 0.28, otitle, size=12.5, bold=True, color=WHITE)
    add_txt(s12, ox + 0.16, 2.18, 5.84, 0.26, osub, size=9.8, color=RGBColor(232, 240, 250))
    add_rect(s12, ox + 0.16, 2.60, 5.84, 0.34, fill=RGBColor(238, 244, 250), border=None, rounded=True)
    add_txt(s12, ox + 0.22, 2.65, 5.72, 0.24, anchor, size=9.2, bold=True, color=col)
    add_txt(s12, ox + 0.16, 3.04, 5.84, 4.04, desc, size=9.8, color=DARK_TXT)

add_notes(s12, "Comparison of Option 3A (single-figure Macro+Micro unified Encoder-Decoder) vs Option 3B (Unified macro Fig. 3 + dedicated mechanistic multi-draw trace Fig. 4).")

prs.save(OUT_PPTX)
print(f"Saved PPTX -> {OUT_PPTX} ({len(prs.slides)} slides)")
