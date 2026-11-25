
$pp = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pres = $pp.Presentations.Open("C:\Users\13575\AppData\Local\Temp\ref_gallery_v2_tmp.pptx", $true, $false, $false)
    for ($i = 1; $i -le $pres.Slides.Count; $i++) {
        $fn = "C:\Users\13575\AppData\Local\Temp\ref_gallery_v2_pngs\slide_" + $i.ToString("00") + ".png"
        $pres.Slides($i).Export($fn, "PNG", 1920, 1080)
    }
} finally {
    if ($pres -ne $null) { $pres.Close() }
    if ($pp -ne $null) { $pp.Quit() }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
