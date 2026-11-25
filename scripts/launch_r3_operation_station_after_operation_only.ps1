param(
    [string]$OperationOnlyRunDir = "results/03_reschedule_main/r3/operation_only/s42_260728_1739",
    [string]$RunId = ("s42_{0}" -f (Get-Date -Format "yyMMdd_HHmm"))
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $ProjectRoot
$summaryPath = Join-Path $OperationOnlyRunDir "eval/summary.json"
$integrityPath = Join-Path $OperationOnlyRunDir "eval/integrity_check.json"
$launchScript = (Resolve-Path "scripts/launch_r3_operation_station_windows.ps1").Path
$logDir = "results/05_efficiency_and_logs/launch_logs"
$logPath = Join-Path $logDir ("r3_operation_station_after_operation_only_{0}.log" -f $RunId)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding UTF8
}

Write-Log "waiting_for_operation_only_summary=$summaryPath"
while ($true) {
    if ((Test-Path -LiteralPath $summaryPath) -and (Test-Path -LiteralPath $integrityPath)) {
        try {
            $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $integrity = Get-Content -LiteralPath $integrityPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([int]$summary.row_count -eq 72 -and [bool]$integrity.passed) { break }
            Write-Log "summary_present_but_not_ready row_count=$($summary.row_count) passed=$($integrity.passed)"
        } catch {
            Write-Log "summary_parse_pending error=$($_.Exception.Message)"
        }
    }
    Start-Sleep -Seconds 60
}

$stationRunDir = Join-Path "results/03_reschedule_main/r3/operation_station" $RunId
if (Test-Path -LiteralPath $stationRunDir) {
    Write-Log "operation_station_run_exists=$stationRunDir; refusing_duplicate_launch"
    exit 2
}
Write-Log "operation_only_r3_ready; launching_operation_station run_id=$RunId"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launchScript -RunId $RunId 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    Write-Log "operation_station_exit_code=$LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Log "operation_station_process_finished"
