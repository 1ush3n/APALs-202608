param(
    [Parameter(Mandatory = $true)][int]$WaitPid,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$EvalRoot,
    [Parameter(Mandatory = $true)][string]$LogPath
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13575\miniconda3\envs\rag_env\python.exe"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $LogPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding UTF8
}

Write-Log "waiting_for_pid=$WaitPid"
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$scenarioIds = "[low_000,medium_000,high_000]"
foreach ($instanceId in @("real_283", "real_680", "real_2338", "real_3182")) {
    $csv = Join-Path (Join-Path $EvalRoot "r3_temp000_seed42") (Join-Path $instanceId "reschedule_ppo_eval.csv")
    $summary = Join-Path (Join-Path $EvalRoot "r3_temp000_seed42") (Join-Path $instanceId "reschedule_ppo_eval_summary.json")
    if (!(Test-Path -LiteralPath $csv) -or !(Test-Path -LiteralPath $summary)) {
        Write-Log "temperature_000_incomplete instance=$instanceId; refusing_to_start_temperature_001"
        exit 2
    }
    $payload = Get-Content -LiteralPath $summary -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([int]$payload.scenario_count -ne 3) {
        Write-Log "temperature_000_wrong_scenario_count instance=$instanceId count=$($payload.scenario_count)"
        exit 2
    }
}

Write-Log "temperature_000_complete; start_temperature_001_seeds_42_46"
foreach ($seed in 42..46) {
    $outputDir = Join-Path $EvalRoot ("r3_temp001_seed{0}" -f $seed)
    $summaryPath = Join-Path $outputDir "reschedule_eval_summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        Write-Log "skip_existing_summary seed=$seed path=$summaryPath"
        continue
    }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Write-Log "start seed=$seed temperature=0.01 output=$outputDir"
    $modelArg = $ModelPath.Replace('\', '/')
    $manifestArg = $ManifestPath.Replace('\', '/')
    $outputArg = $outputDir.Replace('\', '/')
    $pythonCode = "import runpy,sys;from runtime.seed import set_seed;set_seed($seed);sys.argv=['scripts/evaluate_reschedule_manifest.py','experiment=reschedule_task_delay','model_path=$modelArg','manifest_path=$manifestArg','instance_ids=[real_283,real_680,real_2338,real_3182]','scenario_ids=$scenarioIds','seed=$seed','temperature=0.01','reschedule_eval_scenario_seed=42','reschedule_eval_use_cached_observation=true','reschedule_eval_skip_value_estimation=true','output_dir=$outputArg'];runpy.run_path('scripts/evaluate_reschedule_manifest.py',run_name='__main__')"
    & $Python -u -c $pythonCode 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "seed=$seed failed exit_code=$LASTEXITCODE; no_auto_retry"
        exit $LASTEXITCODE
    }
    Write-Log "completed seed=$seed"
}

Write-Log "all_temperature_001_seeds_completed"
$runDir = Split-Path -Parent $EvalRoot
& $Python -u "scripts/organize_reschedule_r3_sixrun.py" $runDir 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    Write-Log "organization_failed exit_code=$LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Log "organization_completed"
