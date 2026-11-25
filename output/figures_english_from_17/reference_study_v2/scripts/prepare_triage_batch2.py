import glob, os, shutil
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

DL_DIR = r"C:\Users\13575\.gemini\antigravity\brain\e39315b1-2e1d-4c54-833e-989799baf2ed\browser_sandbox\downloads"
RAW_DIR = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\raw_papers"
PAGE_DIR = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\rendered_pages"
TRIAGE_DIR = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\triage_sheets"

for src in sorted(glob.glob(os.path.join(DL_DIR, "*.pdf"))):
    dst = os.path.join(RAW_DIR, os.path.basename(src))
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)

def get_font(sz):
    for p in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

# Skip stems already processed into triage_b1
existing_pages = set(os.listdir(PAGE_DIR))
pdfs = sorted(glob.glob(os.path.join(RAW_DIR, "*.pdf")))
new_items = []

for pdf_p in pdfs:
    stem = os.path.splitext(os.path.basename(pdf_p))[0]
    if stem == "arxiv_2401_10241_zero_bubble" or stem == "p1238_1":
        continue
    # Check if already has p01 rendered
    if f"{stem}_p01.png" in existing_pages:
        continue
    try:
        doc = pdfium.PdfDocument(pdf_p)
    except Exception as e:
        print(f"BAD PDF {stem}: {e}")
        continue
    n_pages = len(doc)
    cand_pages = []
    for i in range(min(9, n_pages)):
        tp = doc[i].get_textpage()
        txt = tp.get_text_range()
        tp.close()
        if i in (0, 1) or ("Figure " in txt) or ("Fig." in txt) or ("FIG." in txt):
            cand_pages.append(i + 1)
    for pnum in cand_pages:
        out_png = os.path.join(PAGE_DIR, f"{stem}_p{pnum:02d}.png")
        if not os.path.exists(out_png):
            im = doc[pnum - 1].render(scale=165 / 72.0).to_pil()
            im.save(out_png)
        new_items.append((stem, pnum, out_png))
    doc.close()
    print(f"  NEW {stem[:42]:42s} | pages={n_pages:2d} | cand={cand_pages}")

f_hdr = get_font(22)
per_sheet = 6
cw, ch = 850, 1080
pad = 18
sheet_idx = 0
for s in range(0, len(new_items), per_sheet):
    chunk = new_items[s:s + per_sheet]
    sheet_idx += 1
    out_sheet = os.path.join(TRIAGE_DIR, f"triage_b2_{sheet_idx:02d}.png")
    canvas = Image.new("RGB", (3 * cw + 4 * pad, 2 * ch + 3 * pad), (240, 244, 248))
    draw = ImageDraw.Draw(canvas)
    for k, (stem, pnum, pg_path) in enumerate(chunk):
        r, c = k // 3, k % 3
        x0, y0 = pad + c * (cw + pad), pad + r * (ch + pad)
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(255, 255, 255), outline=(140, 155, 170), width=2)
        draw.rectangle([x0, y0, x0 + cw, y0 + 38], fill=(32, 54, 78))
        draw.text((x0 + 12, y0 + 7), f"{stem[:28]} | p{pnum}", fill=(255, 255, 255), font=f_hdr)
        im = Image.open(pg_path).convert("RGB")
        scale = min((cw - 12) / im.size[0], (ch - 48) / im.size[1])
        rs = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.Resampling.LANCZOS)
        canvas.paste(rs, (x0 + (cw - rs.size[0]) // 2, y0 + 42))
    canvas.save(out_sheet, optimize=True)
    print(f"Saved -> {out_sheet} ({len(chunk)} pages)")
