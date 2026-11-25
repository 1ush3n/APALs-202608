import glob, os, re, pypdfium2 as pdfium

raw = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2\raw_papers"
for p in sorted(glob.glob(os.path.join(raw, "*.pdf"))):
    st = os.path.splitext(os.path.basename(p))[0]
    if st in ("arxiv_2401_10241_zero_bubble", "p1238_1", "2311.14449v3"):
        continue
    doc = pdfium.PdfDocument(p)
    figs = []
    title_line = ""
    for i in range(min(15, len(doc))):
        tp = doc[i].get_textpage()
        t = tp.get_text_range()
        tp.close()
        if i == 0:
            lines = [
                ln.strip() for ln in t.splitlines()
                if ln.strip() and "arXiv:" not in ln and "Published" not in ln and "Proceedings" not in ln
            ]
            title_line = " ".join(lines[:3])[:95]
        for m in re.finditer(r"(Figure|Fig\.|FIG\.)\s+(\d+(?:\.\d+)?)\s*[:.]", t):
            figs.append(f"p{i+1}:F{m.group(2)}")
    doc.close()
    print(f"{st[:34]:34s} | {title_line}")
    if figs:
        print("   ", ", ".join(dict.fromkeys(figs)))

print("\n--- PAPER_INTEGRATED_V1 PREVIEWS ---")
prev = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\paper_integrated_v1\previews"
for f in sorted(os.listdir(prev)):
    print(" ", f)
