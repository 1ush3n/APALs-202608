import os
import re
import shutil
import subprocess
from pptx import Presentation

BASE = r"D:\APAL-Dynamic-v4\output\figures_english_from_17\reference_study_v2"
PPTX_PATH = os.path.join(BASE, "Top_Venue_Reference_Gallery_v2.pptx")
OUT_DIR = os.path.join(BASE, "pptx_previews")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Strict CJK Audit (0 CJK chars across all slides and speaker notes)
prs = Presentation(PPTX_PATH)
cjk_re = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
cjk_hits = []
for idx, slide in enumerate(prs.slides, 1):
    for shp in slide.shapes:
        if shp.has_text_frame:
            t = shp.text_frame.text
            if cjk_re.search(t):
                cjk_hits.append((idx, "shape", t[:60]))
    if slide.has_notes_slide:
        nt = slide.notes_slide.notes_text_frame.text
        if cjk_re.search(nt):
            cjk_hits.append((idx, "notes", nt[:60]))

if cjk_hits:
    raise RuntimeError(f"CJK characters detected in PPTX: {cjk_hits}")
print(f"CJK AUDIT PASS: 0 non-English characters across {len(prs.slides)} slides and notes.")

# 2. Export slides via PowerShell PowerPoint COM (ASCII temp staging for reliability)
tmp_pptx = r"C:\Users\13575\AppData\Local\Temp\ref_gallery_v2_tmp.pptx"
tmp_out = r"C:\Users\13575\AppData\Local\Temp\ref_gallery_v2_pngs"
if os.path.exists(tmp_out):
    shutil.rmtree(tmp_out)
os.makedirs(tmp_out, exist_ok=True)
shutil.copy2(PPTX_PATH, tmp_pptx)

ps_code = f"""
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {{
    $pres = $pp.Presentations.Open("{tmp_pptx}", $true, $false, $false)
    for ($i = 1; $i -le $pres.Slides.Count; $i++) {{
        $fn = "{tmp_out}\\slide_" + $i.ToString("00") + ".png"
        $pres.Slides($i).Export($fn, "PNG", 1920, 1080)
    }}
}} finally {{
    if ($pres -ne $null) {{ $pres.Close() }}
    if ($pp -ne $null) {{ $pp.Quit() }}
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}}
"""
ps_path = os.path.join(BASE, "scripts", "com_export.ps1")
with open(ps_path, "w", encoding="utf-8") as f:
    f.write(ps_code)

res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_path], capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError(f"PowerPoint COM export failed ({res.returncode}):\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")

exported = sorted(os.listdir(tmp_out))
for fn in exported:
    shutil.copy2(os.path.join(tmp_out, fn), os.path.join(OUT_DIR, fn))
print(f"COM EXPORT PASS: {len(exported)} PNGs saved -> {OUT_DIR}")
