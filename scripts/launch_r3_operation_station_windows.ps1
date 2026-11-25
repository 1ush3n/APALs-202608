param(
    [string]$RunId = ("s42_{0}" -f (Get-Date -Format "yyMMdd_HHmm"))
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13575\miniconda3\envs\rag_env\python.exe"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $ProjectRoot
$manifest = "data/r3/m.json"
$warmstart = "checkpoints/init/g15.ckpt"
$trainData = "data/r3/t"
$outRoot = "results/03_reschedule_main/r3"
$runDir = Join-Path (Join-Path $outRoot "operation_station") $RunId
$logDir = "results/05_efficiency_and_logs/launch_logs"
$logPath = Join-Path $logDir ("r3_operation_station_{0}.log" -f $RunId)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($path in @("train.py", $manifest, $warmstart, $trainData)) {
    if (!(Test-Path -LiteralPath $path)) { throw "missing_path=$path" }
}

& $Python -u train.py `
    experiment=reschedule_task_delay `
    hardware=windows_4060_low_memory `
    experiment_name=operation_station `
    run_id=$RunId `
    runs_root=$outRoot `
    artifact_layout=runs `
    seed=42 `
    train_data_path_or_dir=$trainData `
    data_file_path=data/680.csv `
    reschedule_manifest_path=$manifest `
    reschedule_eval_instance_id=real_680 `
    reschedule_eval_scenario_seed=42 `
    reschedule_baseline_model_path=$warmstart `
    reschedule_warm_start=true `
    reschedule_use_objective_delta_reward=true `
    reschedule_objective_delta_multiplier=100.0 `
    reschedule_objective_delta_clip=50.0 `
    policy_action_scope=operation_station `
    train.max_episodes=300 `
    train.batch_size=64 `
    accumulation_steps=32 `
    adaptive_ppo_batch_by_tasks=true `
    adaptive_ppo_batch_small_task_max=530 `
    adaptive_ppo_batch_large_task_min=550 `
    adaptive_ppo_batch_small=128 `
    adaptive_ppo_batch_large=64 `
    lr=0.00015 `
    actor_lr_multiplier=1.0 `
    critic_lr_multiplier=0.5 `
    k_epochs=4 `
    eps_clip=0.25 `
    eps_clip_end=0.15 `
    kl_early_stop=0.015 `
    c_entropy=0.00010 `
    c_entropy_end=0.00002 `
    entropy_decay_episodes=180 `
    eval_freq=1 `
    eval_temperature=0.0 `
    sample_temperature=1.0 `
    async_eval_enabled=true `
    async_eval_device=cpu `
    async_eval_cpu_threads=2 `
    async_eval_queue_capacity=2 `
    async_eval_instance_id=real_680 `
    async_eval_scenario_id=medium_000 `
    async_eval_wait_on_finish=true `
    async_eval_failure_policy=fail `
    async_eval_max_retries=1 `
    async_eval_use_cached_observation=true `
    async_eval_submit_every_episodes=3 `
    async_eval_heartbeat_interval_sec=300 `
    skip_update_on_oom=true `
    oom_transactional_updates=true `
    2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) { throw "training_failed exit_code=$LASTEXITCODE" }
