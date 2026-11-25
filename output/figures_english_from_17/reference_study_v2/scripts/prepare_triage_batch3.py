import glob, os, re, shutil
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

DL_DIR = r"C:\Users\13575\.gemini\antigravity\brain\e39315b1-2e1d-4c54-833e-989799baf2ed\browser_sandbox\downloads"
BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
RAW_DIR = os.path.join(BASE, "raw_papers")
PAGE_DIR = os.path.join(BASE, "rendered_pages")
TRIAGE_DIR = os.path.join(BASE, "triage_sheets")

new_ids = ["2409.13620v1.pdf", "2212.10936v4.pdf", "2009.03836v2.pdf", "2602.08052v1.pdf"]

def get_font(sz):
    for p in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

items = []
for fn in new_ids:
    src = os.path.join(DL_DIR, fn)
    dst = os.path.join(RAW_DIR, fn)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    stem = os.path.splitext(fn)[0]
    doc = pdfium.PdfDocument(dst)
    cand = []
    for i in range(min(12, len(doc))):
        tp = doc[i].get_textpage()
        txt = tp.get_text_range()
        tp.close()
        if i in (0, 1) or re.search(r"(Figure|Fig\.|FIG\.)\s*\d+", txt):
            cand.append(i + 1)
    for pn in cand:
        out_p = os.path.join(PAGE_DIR, f"{stem}_p{pn:02d}.png")
        im = doc[pn - 1].render(scale=165 / 72.0).to_pil()
        im.save(out_p)
        items.append((stem, pn, out_p))
    doc.close()
    print(f"NEW {stem:16s} | cand={cand}")

f_hdr = get_font(22)
per_sheet = 6
cw, ch, pad = 850, 1080, 18
for idx, s in enumerate(range(0, len(items), per_sheet), 1):
    chunk = items[s:s + per_sheet]
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
    out_sheet = os.path.join(TRIAGE_DIR, f"triage_b3_{idx:02d}.png")
    canvas.save(out_sheet, optimize=True)
    print(f"Saved -> {out_sheet} ({len(chunk)} pages)")
