import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 动态添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 修复 OpenMP 多重运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from environment import AirLineEnv_Graph
from configs import configs, load_training_config, load_config_files
from runtime.artifacts import (
    resolve_run_output_dir,
    write_run_context_files,
    write_run_manifest,
)
from runtime.configuration import (
    add_common_config_arguments,
    parse_runtime_args,
    resolve_runtime_config,
)
from baselines.heuristic.baseline_ga import GeneticAlgorithmScheduler
from baselines.heuristic.advanced_schedulers import (
    AdvancedSchedulerBase,
    BeamSearchScheduler,
    IteratedGreedyScheduler,
    SimulatedAnnealingScheduler,
    build_metrics,
)

def compute_cpm_times(env):
    """
    基于关键路径法 (CPM) 正反向递推计算每个任务的最早开始时间 (ES) 和最晚开始时间 (LS)
    """
    durations = env.task_static_feat[:, 0].detach().cpu().numpy()
    num_tasks = env.num_tasks
    
    # 拓扑排序
    topo_order = env._topological_sort()
    
    # 正向递推 (最早开始时间 ES)
    es = np.zeros(num_tasks)
    for u in topo_order:
        my_es = 0
        for p in env.predecessors[u]:
            my_es = max(my_es, es[p] + durations[p])
        es[u] = my_es
        
    max_makespan = 0
    for u in range(num_tasks):
        max_makespan = max(max_makespan, es[u] + durations[u])
        
    # 反向递推 (最晚开始时间 LS)
    ls = np.full(num_tasks, max_makespan)
    for u in reversed(topo_order):
        my_lf = max_makespan
        if env.successors[u]:
            children_ls = [ls[v] for v in env.successors[u]]
            my_lf = min(children_ls)
        ls[u] = my_lf - durations[u]
        
    return es, ls

def select_heuristic_action(env, rule, es, ls):
    """
    根据给定的启发式规则，在当前就绪的任务和可用资源中选择并指派一个合法的动作
    """
    task_mask, station_mask, _ = env.get_masks()

    # 虚拟层级节点不占工位、不需要工人，必须先以资源空动作消费；否则把 skill=-1
    # 当作真实工种索引会产生“没有可用工人”的假死锁。
    virtual_ready_mask = (
        (np.asarray(env.task_status) == 1)
        & (~np.asarray(env.constraint_engine.physical_mask, dtype=bool))
    )
    if hasattr(env, "task_material_ready"):
        virtual_ready_mask &= np.asarray(env.task_material_ready) <= float(env.current_time) + 1.0e-9
    ready_virtual = np.where(virtual_ready_mask)[0]
    if ready_virtual.size > 0:
        selected_virtual = int(min(ready_virtual, key=lambda task_id: (ls[int(task_id)], int(task_id))))
        return (selected_virtual, -1, [])

    if task_mask.all():
        return None
        
    # 提取就绪任务列表
    ready_tasks = np.where(~task_mask.numpy())[0] if hasattr(task_mask, 'numpy') else np.where(~task_mask)[0]
    
    # 提取当前就绪任务的耗时
    durations = env.task_static_feat[ready_tasks, 0].detach().cpu().numpy()
    
    # 根据规则排序
    if rule == "SPT":
        # 耗时最短优先
        sorted_idx = np.argsort(durations)
    elif rule == "LPT":
        # 耗时最长优先
        sorted_idx = np.argsort(durations)[::-1]
    elif rule == "Random":
        # 随机挑选
        sorted_idx = np.random.permutation(len(ready_tasks))
    elif rule == "EDD":
        # 最早开工时间 (ES) 优先
        ready_es = es[ready_tasks]
        sorted_idx = np.argsort(ready_es)
    elif rule == "CPM":
        # 最晚开工时间 (LS) 优先 (最晚开工越早，紧急程度越高)
        ready_ls = ls[ready_tasks]
        sorted_idx = np.argsort(ready_ls)
    else: # 默认 SPT
        sorted_idx = np.argsort(durations)
        
    sorted_tasks = ready_tasks[sorted_idx]
    
    # 不能只检查优先级最高的一个任务：它可能暂时缺少技能工人，
    # 但同一时刻的其他就绪任务仍然可行。直接返回 None 会把正常的
    # 资源等待误判成死锁，尤其容易出现在 EDD/LPT 等规则中。
    for tid in sorted_tasks:
        valid_stations = np.where(~station_mask[tid].numpy())[0] if hasattr(station_mask[tid], 'numpy') else np.where(~station_mask[tid])[0]
        if len(valid_stations) == 0:
            continue

        # 先按站位负荷尝试；若最低负荷站位没有足够技能工人，继续尝试
        # 其他合法站位，而不是立即放弃当前任务。
        ordered_stations = sorted(
            (int(station_id) for station_id in valid_stations),
            key=lambda station_id: (float(env.station_loads[station_id]), station_id),
        )
        task_skill = int(env.task_static_feat[tid, 1].item())
        num_workers_req = int(env.task_static_feat[tid, 2].item())

        for selected_station in ordered_stations:
            skilled_available = []
            for worker_id in range(env.num_workers):
                if env.worker_skill_matrix[worker_id, task_skill] > 0.5:
                    if env.worker_locks[worker_id] == 0 or env.worker_locks[worker_id] == (selected_station + 1):
                        skilled_available.append(worker_id)

            if len(skilled_available) < num_workers_req:
                continue

            # 从可用工人中随机选择 (此处亦支持添加其他启发式策略)
            selected_workers = np.random.choice(skilled_available, size=num_workers_req, replace=False).tolist()
            return (int(tid), selected_station, selected_workers)

    # 所有就绪任务都暂时没有可行的站位/技能工人组合，才报告资源死锁。
    return None

def run_heuristic_eval(env, rule, num_runs=1, seed=42):
    """
    运行启发式规则的评估逻辑
    """
    np.random.seed(seed)
    
    run_makespans = []
    run_balances = []
    run_worker_utils = []
    run_station_utils = []
    run_durations = []
    run_schedules = []
    
    valid_count = 0
    deadlock_count = 0
    
    rule = str(rule).upper()
    # EDD 也必须使用统一可行性解码器；直接逐步贪心会在技能工人
    # 暂时不可用时误报死锁。解码器会保留 EDD 的 ES 优先级，同时
    # 处理资源等待和硬约束。
    safe_decoder_rules = {"SPT", "LPT", "EDD", "CPM", "MSL"}

    for run in range(num_runs):
        run_seed = seed + run

        if rule in safe_decoder_rules:
            start_time = time.time()
            scheduler = AdvancedSchedulerBase(env, seed=run_seed)
            solution = scheduler.build_rule_solution(rule, run_seed)
            result = scheduler.decoder.decode(solution, run_seed)
            inference_time = time.time() - start_time
            complete = bool(result.complete)

            if complete:
                valid_count += 1
                makespan = float(result.makespan)
                balance = float(result.balance_std)
                worker_busy_time = 0.0
                station_busy_time = np.zeros(env.num_stations)
                for _, sid, team, start, end in result.assigned_tasks:
                    duration = float(end) - float(start)
                    worker_busy_time += duration * len(team)
                    if sid >= 0:
                        station_busy_time[int(sid)] += duration
                w_util = worker_busy_time / (env.num_workers * makespan) if makespan > 0 else 0.0
                max_slots = getattr(configs, "max_slots_per_station", 3)
                s_util = (
                    np.sum(station_busy_time) / (env.num_stations * max_slots * makespan)
                    if makespan > 0
                    else 0.0
                )
                run_makespans.append(makespan)
                run_balances.append(balance)
                run_worker_utils.append(float(w_util))
                run_station_utils.append(float(s_util))
                run_schedules.append(result.assigned_tasks)
            else:
                deadlock_count += 1
                run_makespans.append(float(result.makespan))
                run_balances.append(float(result.balance_std))
                run_worker_utils.append(0.0)
                run_station_utils.append(0.0)
                run_schedules.append([])
            run_durations.append(inference_time)
            continue

        env.reset(randomize_duration=False, randomize_workers=False, seed=run_seed)
        es, ls = compute_cpm_times(env)
        
        done = False
        step_count = 0
        max_steps = env.num_tasks * 4
        
        start_time = time.time()
        while not done and step_count < max_steps:
            step_count += 1
            action = select_heuristic_action(env, rule, es, ls)
            if action is None:
                if env.try_wait_for_resources():
                    continue
                else:
                    # 遭遇真实死锁
                    break
            env.step(action)
            
        end_time = time.time()
        inference_time = end_time - start_time
        
        complete = len(env.assigned_tasks) == env.num_tasks
        if complete:
            valid_count += 1
            makespan = np.max(env.station_wall_clock)
            balance = np.std(env.station_loads)
            
            # 计算资源利用率
            worker_busy_time = 0.0
            station_busy_time = np.zeros(env.num_stations)
            for (tid, sid, team, start, end) in env.assigned_tasks:
                dur = end - start
                worker_busy_time += dur * len(team)
                if sid >= 0:
                    station_busy_time[sid] += dur
                    
            w_util = worker_busy_time / (env.num_workers * makespan) if makespan > 0 else 0.0
            max_slots = getattr(configs, 'max_slots_per_station', 3)
            s_util = np.sum(station_busy_time) / (env.num_stations * max_slots * makespan) if makespan > 0 else 0.0
            
            run_makespans.append(makespan)
            run_balances.append(balance)
            run_worker_utils.append(w_util)
            run_station_utils.append(s_util)
            run_durations.append(inference_time)
            run_schedules.append(env.assigned_tasks)
        else:
            deadlock_count += 1
            run_makespans.append(env.ideal_makespan * 3.0)
            run_balances.append(env.ideal_station_load * 3.0)
            run_worker_utils.append(0.0)
            run_station_utils.append(0.0)
            run_durations.append(inference_time)
            run_schedules.append([])
            
    best_idx = np.argmin(run_makespans) if run_makespans else 0
    metrics = {
        "makespan": float(np.mean(run_makespans)),
        "workload_balance_std": float(np.mean(run_balances)),
        "worker_utilization": float(np.mean(run_worker_utils)),
        "station_utilization": float(np.mean(run_station_utils)),
        "inference_time": float(np.mean(run_durations)),
        "valid": 1.0 if valid_count == num_runs else 0.0,
        "deadlock_count": deadlock_count,
        "completion_rate": float(valid_count / num_runs)
    }
    return metrics, run_schedules[best_idx]

def run_ga_eval(env, pop_size, max_gen, seed=42):
    """
    运行遗传算法 (GA) 评估逻辑
    """
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    # 实例化已有的遗传算法基线调度器
    ga_solver = GeneticAlgorithmScheduler(env, pop_size=pop_size, max_gen=max_gen)
    makespan, balance_std, assigned_tasks = ga_solver.run()
    duration = time.time() - start_time
    
    complete = len(assigned_tasks) == env.num_tasks
    if complete:
        # 计算资源利用率
        worker_busy_time = 0.0
        station_busy_time = np.zeros(env.num_stations)
        for (tid, sid, team, start, end) in assigned_tasks:
            dur = end - start
            worker_busy_time += dur * len(team)
            if sid >= 0:
                station_busy_time[sid] += dur
                
        w_util = worker_busy_time / (env.num_workers * makespan) if makespan > 0 else 0.0
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        s_util = np.sum(station_busy_time) / (env.num_stations * max_slots * makespan) if makespan > 0 else 0.0
        
        metrics = {
            "makespan": float(makespan),
            "workload_balance_std": float(balance_std),
            "worker_utilization": float(w_util),
            "station_utilization": float(s_util),
            "inference_time": float(duration),
            "valid": 1.0,
            "deadlock_count": 0,
            "completion_rate": 1.0
        }
    else:
        metrics = {
            "makespan": float(env.ideal_makespan * 3.0),
            "workload_balance_std": float(env.ideal_station_load * 3.0),
            "worker_utilization": 0.0,
            "station_utilization": 0.0,
            "inference_time": float(duration),
            "valid": 0.0,
            "deadlock_count": 1,
            "completion_rate": 0.0
        }
    return metrics, assigned_tasks

def run_advanced_eval(env, method, args, seed=42):
    """
    运行高级启发式基线，并复用现有 baseline 指标结构。
    """
    method_key = method.lower()
    start_time = time.time()

    if method_key in {"beam", "beamsearch", "beam_search"}:
        scheduler = BeamSearchScheduler(
            env,
            beam_width=args.beam_width,
            branch_factor=args.beam_branch_factor,
            levels=args.beam_levels,
            patience=args.beam_patience,
            seed=seed,
            balance_weight=args.balance_weight,
        )
    elif method_key in {"ig", "iteratedgreedy", "iterated_greedy", "destroyrepair", "destroy_and_repair"}:
        scheduler = IteratedGreedyScheduler(
            env,
            iterations=args.ig_iterations,
            destroy_ratio=args.ig_destroy_ratio,
            noise_sigma=args.ig_noise_sigma,
            seed=seed,
            balance_weight=args.balance_weight,
        )
    elif method_key in {"sa", "simulatedannealing", "simulated_annealing"}:
        scheduler = SimulatedAnnealingScheduler(
            env,
            iterations=args.sa_iterations,
            initial_temp=args.sa_initial_temp,
            cooling=args.sa_cooling,
            min_temp=args.sa_min_temp,
            seed=seed,
            balance_weight=args.balance_weight,
        )
    else:
        raise ValueError(f"未知高级启发式方法: {method}")

    makespan, balance_std, assigned_tasks = scheduler.run()
    duration = time.time() - start_time
    metrics = build_metrics(env, makespan, balance_std, assigned_tasks, duration)
    search_diagnostics = scheduler.search_diagnostics()
    metrics.update(search_diagnostics)
    if assigned_tasks:
        metrics["failure_type"] = None
    elif search_diagnostics["candidate_deadlock_count"] > 0:
        metrics["failure_type"] = "no_legal_schedule_with_deadlocks"
    elif search_diagnostics["illegal_candidate_count"] > 0:
        metrics["failure_type"] = "all_candidates_illegal"
        # 完整但非法的排程不是资源死锁，必须单独报告。
        metrics["deadlock_count"] = 0
    else:
        metrics["failure_type"] = "no_complete_schedule"
    if not assigned_tasks:
        print(
            f"[方法失败但评估继续] method={method}, "
            f"failure_type={metrics['failure_type']}, diagnostics={search_diagnostics}"
        )
    return metrics, assigned_tasks

def save_eval_results(method, dataset_name, metrics, assigned_tasks, env, output_root):
    """
    将评估结果以结构化的形式导出到文件
    保存目录：<output_root>/<method>/<dataset_name>/
    包含文件：metrics.json, schedule.csv, run.log
    """
    output_dir = Path(output_root) / method / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存 metrics.json
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    # 2. 保存 schedule.csv
    if assigned_tasks:
        tasks_data = []
        for (tid, sid, team, start, end) in assigned_tasks:
            task_ao = env.raw_data['task_df']['task_id'].iloc[tid] if 'task_df' in env.raw_data else str(tid)
            real_id = env.raw_data['task_df']['序号'].iloc[tid] if 'task_df' in env.raw_data and '序号' in env.raw_data['task_df'].columns else tid
            tasks_data.append({
                'TaskID': real_id,
                'TaskAO': task_ao,
                'StationID': sid + 1,
                'Team': str(team),
                'Start': start,
                'End': end,
                'Duration': end - start
            })
        df = pd.DataFrame(tasks_data)
        df.to_csv(output_dir / "schedule.csv", index=False)
        
    # 3. 保存 run.log
    with open(output_dir / "run.log", "w", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Metrics:\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n")
        
    print(f"      [Export Complete] -> {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="APAL 基线 Heuristic/GA 批量评估工具")
    parser.add_argument("--data_dir", type=str, default="data", help="测试数据集所在的目录")
    parser.add_argument("--datasets", type=str, nargs="+", default=["283.csv", "680.csv", "2338.csv", "3182.csv"], help="待测试的 CSV 文件列表")
    parser.add_argument("--methods", type=str, nargs="+", default=["SPT", "LPT", "Random", "EDD", "CPM", "MSL", "GA"], help="待评估的方法列表")
    parser.add_argument("--ga_pop_size", type=int, default=30, help="GA 种群大小")
    parser.add_argument("--ga_max_gen", type=int, default=20, help="GA 最大迭代代数")
    parser.add_argument("--balance_weight", type=float, default=1.0, help="高级启发式 fitness 中负载均衡标准差的权重")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam Search 保留的候选解数量")
    parser.add_argument("--beam_branch_factor", type=int, default=4, help="Beam Search 每个候选解生成的分支数量")
    parser.add_argument("--beam_levels", type=int, default=8, help="Beam Search 最大展开层数")
    parser.add_argument("--beam_patience", type=int, default=4, help="Beam Search 无改进提前停止轮数")
    parser.add_argument("--ig_iterations", type=int, default=80, help="Iterated Greedy 迭代次数")
    parser.add_argument("--ig_destroy_ratio", type=float, default=0.10, help="Iterated Greedy 每轮破坏的工序比例")
    parser.add_argument("--ig_noise_sigma", type=float, default=0.25, help="Iterated Greedy 修复扰动强度")
    parser.add_argument("--sa_iterations", type=int, default=120, help="Simulated Annealing 迭代次数")
    parser.add_argument("--sa_initial_temp", type=float, default=0.05, help="Simulated Annealing 初始温度")
    parser.add_argument("--sa_cooling", type=float, default=0.96, help="Simulated Annealing 降温系数")
    parser.add_argument("--sa_min_temp", type=float, default=1e-4, help="Simulated Annealing 最小温度")
    parser.add_argument("--random_runs", type=int, default=5, help="Random 规则的多轮评测次数")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    parser.add_argument("--config", type=str, default=None, help="加载实验配置文件 (YAML) 以保证运行环境的物理参数与主模型百分之百完全一致")
    add_common_config_arguments(parser)
    args = parse_runtime_args(parser)
    
    if args.config:
        resolve_runtime_config(args, target=configs)
        print(f"[*] 成功加载实验配置 YAML: {args.config}")
        print(f"[*] 全局环境参数已同步: n_w={configs.n_w}, n_m={configs.n_m}, max_slots_per_station={configs.max_slots_per_station}")
    else:
        resolve_runtime_config(args, target=configs)
    output_root, context = resolve_run_output_dir(
        configs,
        PROJECT_ROOT,
        default_legacy_dir="results/eval_logs",
        run_subdir="baselines/heuristic",
        explicit_dir=getattr(args, "output_dir", None),
        section="artifacts",
    )
    manifest_extra = {
        "run_type": "baseline",
        "artifact_kind": "heuristic_baselines",
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "output_dir": str(output_root.resolve()),
    }
    if context is not None:
        write_run_context_files(context, configs, command="run_all_baselines", extra=manifest_extra)
    else:
        write_run_manifest(output_root, configs, command="run_all_baselines", extra=manifest_extra)
    
    print("="*60)
    print("      APAL 批量基线评估工具 (Heuristics & Genetic Algorithm)")
    print("="*60)
    print(f"数据集目录: {args.data_dir}")
    print(f"数据集列表: {args.datasets}")
    print(f"评估基准方法: {args.methods}")
    print(f"随机策略轮数: {args.random_runs}，GA 参数: Pop={args.ga_pop_size}, Gen={args.ga_max_gen}")
    print("="*60)
    
    summary_data = []
    
    data_dir_path = PROJECT_ROOT / args.data_dir
    
    for dataset_file in args.datasets:
        dataset_path = data_dir_path / dataset_file
        dataset_name = dataset_path.stem
        
        if not dataset_path.exists():
            print(f"[Warning] 数据集不存在，已跳过: {dataset_path}")
            continue
            
        # 1. 动态对齐和加载全局/特定规模环境参数
        if args.config:
            resolve_runtime_config(args, target=configs)
        else:
            default_env_yaml = PROJECT_ROOT / "conf" / "env" / "apal_default.yaml"
            if default_env_yaml.exists():
                load_training_config([str(default_env_yaml)])
                
        # 加载特定数据集对应的 initial_bucket_xxx.yaml 以应用其规模化工人数参数
        scale_yaml_path = PROJECT_ROOT / "conf" / "env" / f"initial_bucket_{dataset_name}.yaml"
        if scale_yaml_path.exists():
            load_config_files([str(scale_yaml_path)])
            print(f"      [Auto-Config] 成功为数据集 {dataset_file} 加载特定规模配置: {scale_yaml_path.name} (n_w={configs.n_w})")
        else:
            print(f"      [Auto-Config] 数据集 {dataset_file} 未找到特定规模配置，继续使用全局 n_w={configs.n_w}")
            
        print(f"\n[Dataset] 开始评估数据集: {dataset_file} ...")
        # 实例化环境
        env = AirLineEnv_Graph(data_path_or_dir=str(dataset_path), seed=args.seed)
        
        for method in args.methods:
            print(f"  [Method: {method}] Running...")
            
            try:
                method_key = method.lower()
                if method == "GA":
                    # GA 评估
                    metrics, schedule = run_ga_eval(env, pop_size=args.ga_pop_size, max_gen=args.ga_max_gen, seed=args.seed)
                elif method_key in {"beam", "beamsearch", "beam_search", "ig", "iteratedgreedy", "iterated_greedy", "destroyrepair", "destroy_and_repair", "sa", "simulatedannealing", "simulated_annealing"}:
                    metrics, schedule = run_advanced_eval(env, method=method, args=args, seed=args.seed)
                else:
                    # 启发式规则评估
                    runs = args.random_runs if method == "Random" else 1
                    metrics, schedule = run_heuristic_eval(env, rule=method, num_runs=runs, seed=args.seed)
                
                # 导出结果
                save_eval_results(method, dataset_name, metrics, schedule, env, output_root)
                
                # 记录汇总表格
                summary_data.append({
                    "Dataset": dataset_name,
                    "Method": method,
                    "Makespan": f"{metrics['makespan']:.2f}",
                    "BalanceStd": f"{metrics['workload_balance_std']:.2f}",
                    "WorkerUtil": f"{metrics['worker_utilization'] * 100:.1f}%",
                    "StationUtil": f"{metrics['station_utilization'] * 100:.1f}%",
                    "Time(s)": f"{metrics['inference_time']:.4f}",
                    "Valid": "Yes" if metrics["valid"] > 0.5 else "No (Deadlock)"
                })
            except Exception as e:
                print(f"  [Error] 方法 {method} 在数据集 {dataset_name} 上运行失败: {e}")
                import traceback
                traceback.print_exc()

    # 打印最终统计汇总表
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("                        最终评估结果汇总表")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_summary.to_string(index=False))
        print("="*80)
        
        # 保存汇总表到 csv
        summary_csv = output_root / "baselines_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"[*] 汇总结果已导出至: {summary_csv}")

if __name__ == "__main__":
    main()
