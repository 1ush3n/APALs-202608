@echo off
setlocal
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8
cd /d "%~dp0.."
python -u baselines/heuristic/run_all_baselines.py --config conf/experiment/initial_schedule_680.yaml --data_dir data --datasets 2338.csv --methods SA --sa_iterations 120 --sa_initial_temp 0.05 --sa_cooling 0.96 --sa_min_temp 0.0001 --balance_weight 1.0 --seed 42 --output-dir results/tmp_sa_run_2338 > results/tmp_sa_run_2338/process.log 2>&1
set RC=%errorlevel%
echo EXIT_CODE=%RC%>>results/tmp_sa_run_2338/process.log
exit /b %RC%
