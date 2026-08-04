param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$ManifestPath = "data/r4/m.json",
    [string]$LogPath = "results/05_efficiency_and_logs/launch_logs/r4_reschedule_sixrun_resume.log"
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13575\miniconda3\envs\rag_env\python.exe"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $LogPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding UTF8
}

foreach ($path in @("scripts/evaluate_reschedule_manifest.py", "scripts/organize_reschedule_r4_sixrun.py", $ManifestPath, $ModelPath)) {
    if (!(Test-Path -LiteralPath $path)) { throw "missing_path=$path" }
}

$runPath = (Resolve-Path $RunDir).Path
$evalRoot = Join-Path $runPath "eval"
New-Item -ItemType Directory -Force -Path $evalRoot | Out-Null
$modelArg = (Resolve-Path $ModelPath).Path.Replace('\', '/')
$manifestArg = (Resolve-Path $ManifestPath).Path.Replace('\', '/')
$scenarioIds = "[low_000,medium_000,high_000]"
$instances = @("real_283", "real_680", "real_2338", "real_3182")
$groups = @(
    [pscustomobject]@{ Name = "r4_temp000_seed42"; Temperature = 0.0; Seed = 42 }
    42..46 | ForEach-Object {
        [pscustomobject]@{ Name = ("r4_temp001_seed{0}" -f $_); Temperature = 0.01; Seed = $_ }
    }
)

Write-Log "resume_start run_dir=$runPath"
foreach ($group in $groups) {
    $groupName = [string]$group.Name
    $temperature = [string]$group.Temperature
    $seed = [int]$group.Seed
    $outputDir = Join-Path $evalRoot $groupName
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    foreach ($instanceId in $instances) {
        $instanceDir = Join-Path $outputDir $instanceId
        $csvPath = Join-Path $instanceDir "reschedule_ppo_eval.csv"
        $summaryPath = Join-Path $instanceDir "reschedule_ppo_eval_summary.json"
        if ((Test-Path -LiteralPath $csvPath) -and (Test-Path -LiteralPath $summaryPath)) {
            Write-Log "skip_existing_instance=$groupName/$instanceId"
            continue
        }
        $outputArg = $outputDir.Replace('\', '/')
        Write-Log "start_instance=$groupName/$instanceId temperature=$temperature seed=$seed output=$outputDir"
        $pythonCode = "import runpy,sys;from runtime.seed import set_seed;set_seed($seed);sys.argv=['scripts/evaluate_reschedule_manifest.py','experiment=reschedule_task_delay','model_path=$modelArg','manifest_path=$manifestArg','instance_ids=[$instanceId]','scenario_ids=$scenarioIds','seed=$seed','temperature=$temperature','reschedule_eval_scenario_seed=42','reschedule_eval_use_cached_observation=true','reschedule_eval_skip_value_estimation=true','output_dir=$outputArg'];runpy.run_path('scripts/evaluate_reschedule_manifest.py',run_name='__main__')"
        & $Python -u -c $pythonCode 2>&1 | Tee-Object -FilePath $LogPath -Append
        if ($LASTEXITCODE -ne 0) {
            Write-Log "instance_failed=$groupName/$instanceId exit_code=$LASTEXITCODE"
            exit $LASTEXITCODE
        }
        if (!(Test-Path -LiteralPath $csvPath) -or !(Test-Path -LiteralPath $summaryPath)) {
            Write-Log "instance_missing_output=$groupName/$instanceId"
            exit 3
        }
        Write-Log "completed_instance=$groupName/$instanceId"
    }
    Write-Log "completed_group=$groupName"
}

Write-Log "all_r4_groups_completed; organize"
& $Python -u "scripts/organize_reschedule_r4_sixrun.py" $runPath 2>&1 | Tee-Object -FilePath $LogPath -Append
if ($LASTEXITCODE -ne 0) {
    Write-Log "organization_failed exit_code=$LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Log "organization_completed"
