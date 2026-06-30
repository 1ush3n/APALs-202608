"""
参数灵敏度分析工具。

对 configs.py 中的关键超参数进行网格搜索或单变量扫描，
量化每个参数对最终调度性能（Makespan、负载均衡、收敛速度）的影响程度。

用法:
    python scripts/sensitivity_analysis.py --param lr --values 1e-5,5e-5,1e-4,5e-4
    python scripts/sensitivity_analysis.py --param gamma --values 0.99,0.995,0.999
    python scripts/sensitivity_analysis.py --all  # 运行所有预设参数网格

输出:
    results/sensitivity/ 目录下生成 CSV 表格和对比图表
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import json
import time
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import configs
from configs import Config
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from train import Memory

# ============================================================================
# 预设扫描网格：定义每个参数的建议测试值
# ============================================================================
PRESET_PARAM_GRID = {
    'lr': {
        'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'description': 'Learning Rate / 学习率',
        'short_episodes': 50,
    },
    'gamma': {
        'values': [0.95, 0.99, 0.995, 0.999],
        'description': 'Discount Factor / 折扣因子',
        'short_episodes': 50,
    },
    'eps_clip': {
        'values': [0.05, 0.1, 0.2, 0.3, 0.4],
        'description': 'PPO Clip Threshold / PPO裁剪阈值',
        'short_episodes': 50,
    },
    'k_epochs': {
        'values': [2, 4, 8, 16],
        'description': 'Update Epochs per Batch / 每批更新轮数',
        'short_episodes': 50,
    },
    'gae_lambda': {
        'values': [0.8, 0.9, 0.95, 0.98, 1.0],
        'description': 'GAE Lambda / 广义优势估计衰减因子',
        'short_episodes': 50,
    },
    'hidden_dim': {
        'values': [64, 128, 256],
        'description': 'Hidden Dimension / 隐藏层维度',
        'short_episodes': 50,
    },
    'num_gat_layers': {
        'values': [2, 3, 5, 7],
        'description': 'GAT Layers / 图注意力层数',
        'short_episodes': 50,
    },
    'num_heads': {
        'values': [2, 4, 8],
        'description': 'Attention Heads / 注意力头数',
        'short_episodes': 50,
    },
    'c_policy': {
        'values': [0.01, 0.06, 0.1, 0.5],
        'description': 'Policy Loss Weight / 策略损失权重',
        'short_episodes': 50,
    },
    'c_value': {
        'values': [1.0, 5.0, 10.0, 20.0],
        'description': 'Value Loss Weight / 价值损失权重',
        'short_episodes': 50,
    },
    'c_entropy': {
        'values': [1e-5, 1e-4, 1e-3, 1e-2],
        'description': 'Entropy Coefficient / 熵正则化系数',
        'short_episodes': 50,
    },
    'reward_scale': {
        'values': [0.001, 0.005, 0.01, 0.05],
        'description': 'Reward Scale / 奖励缩放',
        'short_episodes': 50,
    },
    'dur_random_range': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.5],
        'description': 'Duration Randomization Range / 工时扰动幅度',
        'short_episodes': 50,
    },
}

# ============================================================================
# 核心灵敏度分析引擎
# ============================================================================

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run_short_training(
    param_name: str,
    param_value: Any,
    base_config: Config,
    data_path: str,
    device: torch.device,
    short_episodes: int = 50,
) -> Dict[str, Any]:
    """
    使用修改后的参数值执行短训练，返回评估指标。

    关键设计：project 中所有模块（environment, ppo_agent, hb_gat_pn）均通过
    ``from configs import configs`` 读取模块级全局单例，因此参数变更必须直接
    作用于 configs.configs 全局对象，而非 deepcopy 后的局部副本。
    """
    import configs as cfg_module
    _cfg = cfg_module.configs

    old_value = getattr(_cfg, param_name, None)
    setattr(_cfg, param_name, param_value)

    if param_name in ('hidden_dim', 'num_gat_layers', 'num_heads'):
        _cfg.max_episodes = short_episodes * 2
    else:
        _cfg.max_episodes = short_episodes

    _cfg.eval_freq = min(2, max(1, short_episodes // 5))

    set_seed(_cfg.seed)

    try:
        env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=_cfg.seed)
        eval_env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=_cfg.seed)

        try:
            model = HBGATPN(_cfg).to(device)
        except Exception as e:
            return {'error': f'Model init failed: {e}'}

        total_updates = max(1, _cfg.max_episodes // _cfg.update_every_episodes)
        agent = PPOAgent(
            model=model, lr=_cfg.lr, gamma=_cfg.gamma,
            k_epochs=_cfg.k_epochs, eps_clip=_cfg.eps_clip,
            device=device, batch_size=_cfg.batch_size,
            total_timesteps=total_updates,
        )

        eval_makespans = []
        eval_balances = []
        eval_worker_utils = []
        eval_station_utils = []
        episode_rewards = []
        memory = Memory()

        for ep in range(1, _cfg.max_episodes + 1):
            agent.policy.train()
            env.raw_data = env.dataset_pool[env.active_dataset_idx]['raw_data']
            env.num_tasks = env.dataset_pool[env.active_dataset_idx]['num_tasks']

            state = env.reset(
                randomize_duration=_cfg.randomize_durations,
                randomize_workers=_cfg.randomize_durations
            )
            done = False
            ep_reward = 0.0
            max_steps = env.num_tasks * 2

            for _ in range(max_steps):
                if done:
                    break
                t_mask, s_mask, w_mask = env.get_masks()
                if t_mask.all():
                    if env.try_wait_for_resources():
                        continue
                    done = True
                    break

                env_state_snapshot = env.get_state_snapshot()

                action_ret = agent.select_action(
                    state.to(device),
                    mask_task=t_mask.to(device),
                    mask_station_matrix=s_mask.to(device),
                    mask_worker=w_mask.to(device),
                    deterministic=False,
                )
                if action_ret[0] is None:
                    done = True
                    break

                action, logprob, val, _, _ = action_ret

                memory.states.append(env_state_snapshot)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.masks.append((t_mask, s_mask, w_mask))
                memory.values.append(val)

                state, reward, done, _ = env.step(action)
                ep_reward += reward

                memory.rewards.append(reward)
                memory.is_terminals.append(done)

            episode_rewards.append(ep_reward)

            if ep % _cfg.update_every_episodes == 0:
                if len(memory.actions) > 0:
                    agent.update(memory, env, current_ep=ep)
                memory.clear()

            if ep % _cfg.eval_freq == 0:
                agent.policy.eval()
                eval_state = eval_env.reset(randomize_duration=False)

                eval_makespan = None
                eval_balance = None
                eval_w_util = None
                eval_s_util = None

                for _ in range(max_steps):
                    t_mask, s_mask, w_mask = eval_env.get_masks()
                    if t_mask.all():
                        if eval_env.try_wait_for_resources():
                            continue
                        if len(eval_env.assigned_tasks) != eval_env.num_tasks:
                            eval_makespan = eval_env.ideal_makespan * 3.0
                            eval_balance = eval_env.ideal_station_load * 3.0
                            eval_w_util = 0.0
                            eval_s_util = 0.0
                        break

                    action_ret = agent.select_action(
                        eval_state.to(device),
                        mask_task=t_mask.to(device),
                        mask_station_matrix=s_mask.to(device),
                        mask_worker=w_mask.to(device),
                        deterministic=True,
                        temperature=0.0,
                        is_eval=True,
                    )
                    if action_ret[0] is None:
                        eval_makespan = eval_env.ideal_makespan * 3.0
                        eval_balance = eval_env.ideal_station_load * 3.0
                        eval_w_util = 0.0
                        eval_s_util = 0.0
                        break

                    eval_action, _, _, _, _ = action_ret
                    eval_state, _, eval_done, _ = eval_env.step(eval_action)
                    if eval_done:
                        break

                if eval_makespan is None:
                    if len(eval_env.assigned_tasks) == eval_env.num_tasks:
                        fm = np.max(eval_env.station_wall_clock)
                        eval_makespan = fm
                        eval_balance = np.std(eval_env.station_loads)
                        w_busy = sum(
                            (end - start) * len(team)
                            for (_, _, team, start, end) in eval_env.assigned_tasks
                        )
                        eval_w_util = w_busy / (eval_env.num_workers * fm) if fm > 0 else 0.0
                        s_busy = sum(
                            end - start
                            for (_, sid, _, start, end) in eval_env.assigned_tasks if sid >= 0
                        )
                        max_s = getattr(_cfg, 'max_slots_per_station', 3)
                        eval_s_util = s_busy / (eval_env.num_stations * max_s * fm) if fm > 0 else 0.0
                    else:
                        eval_makespan = eval_env.ideal_makespan * 3.0
                        eval_balance = eval_env.ideal_station_load * 3.0
                        eval_w_util = 0.0
                        eval_s_util = 0.0

                eval_makespans.append(eval_makespan)
                eval_balances.append(eval_balance)
                eval_worker_utils.append(eval_w_util)
                eval_station_utils.append(eval_s_util)

        final_makespan = eval_makespans[-1] if eval_makespans else eval_env.ideal_makespan * 3.0
        best_makespan = min(eval_makespans) if eval_makespans else eval_env.ideal_makespan * 3.0
        final_balance = eval_balances[-1] if eval_balances else eval_env.ideal_station_load * 3.0
        avg_reward_last5 = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards) if episode_rewards else 0.0
        final_w_util = eval_worker_utils[-1] if eval_worker_utils else 0.0
        final_s_util = eval_station_utils[-1] if eval_station_utils else 0.0

        convergence_speed = 99999.0
        if eval_makespans:
            threshold = best_makespan * 1.1
            for i, ms in enumerate(eval_makespans):
                if ms <= threshold:
                    convergence_speed = (i + 1) * _cfg.eval_freq
                    break

        return {
            'param_value': param_value,
            'final_makespan': final_makespan,
            'best_makespan': best_makespan,
            'final_balance': final_balance,
            'avg_reward_last5': avg_reward_last5,
            'final_w_util': final_w_util,
            'final_s_util': final_s_util,
            'convergence_speed_ep': convergence_speed,
            'eval_history': eval_makespans,
        }
    finally:
        if old_value is not None:
            setattr(_cfg, param_name, old_value)


def run_single_param_sweep(
    param_name: str,
    values: List[Any],
    data_path: str,
    device: torch.device,
    short_episodes: int = 50,
) -> pd.DataFrame:
    """对单个参数进行扫描"""
    results = []
    print(f"\n{'='*60}")
    print(f"Parameter Sweep: {param_name}")
    print(f"Values: {values}")
    print(f"{'='*60}")

    for i, val in enumerate(values):
        print(f"\n[{i+1}/{len(values)}] Testing {param_name} = {val} ...")
        t0 = time.time()
        result = run_short_training(
            param_name, val, configs.configs, data_path, device,
            short_episodes=short_episodes,
        )
        elapsed = time.time() - t0
        result['elapsed_sec'] = elapsed
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Makespan={result['best_makespan']:.1f}, "
                  f"Balance={result['final_balance']:.2f}, "
                  f"W-Util={result['final_w_util']*100:.1f}%, "
                  f"Time={elapsed:.1f}s")
        results.append(result)

    return pd.DataFrame(results)


def plot_sensitivity(df: pd.DataFrame, param_name: str, output_dir: str):
    """生成灵敏度分析图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_vals = df['param_value'].values
    x_labels = [str(v) for v in param_vals]

    metrics = [
        ('best_makespan', 'Best Makespan (h)', 'lower is better'),
        ('final_balance', 'Workload Balance Std', 'lower is better'),
        ('final_w_util', 'Worker Utilization', 'higher is better'),
        ('final_s_util', 'Station Utilization', 'higher is better'),
        ('convergence_speed_ep', 'Convergence Speed (episodes)', 'lower is better'),
        ('avg_reward_last5', 'Avg Reward (last 5 eps)', 'higher is better'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (col, ylabel, note) in zip(axes, metrics):
        if col in df.columns and not df[col].isna().all():
            valid = df[col].notna()
            if valid.sum() > 0:
                ax.plot(
                    range(len(param_vals)),
                    df[col].values,
                    'o-', linewidth=2, markersize=8, color='#2196F3'
                )
                ax.set_xticks(range(len(param_vals)))
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_title(f'{ylabel} ({note})', fontsize=11)
        ax.grid(True, alpha=0.3)

    desc = PRESET_PARAM_GRID.get(param_name, {}).get("description", "").split("/")[0].strip()
    fig.suptitle(
        f'Sensitivity Analysis: {param_name}\n'
        f'({desc})',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    save_path = output_dir / f'sensitivity_{param_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {save_path}")


def generate_summary_report(all_results: Dict[str, pd.DataFrame], output_dir: str):
    output_dir = Path(output_dir)
    """生成汇总报告"""
    summary_rows = []
    for param_name, df in all_results.items():
        if df.empty or 'error' in df.columns:
            continue
        if 'best_makespan' not in df.columns:
            continue

        valid = df[df['best_makespan'] < 99998.0]
        if valid.empty:
            continue

        best_row = valid.loc[valid['best_makespan'].idxmin()]
        worst_row = valid.loc[valid['best_makespan'].idxmax()]

        makespan_range = worst_row['best_makespan'] - best_row['best_makespan']
        sensitivity_pct = (makespan_range / best_row['best_makespan'] * 100) if best_row['best_makespan'] > 0 else 0

        summary_rows.append({
            'Parameter': param_name,
            'Description': PRESET_PARAM_GRID.get(param_name, {}).get('description', ''),
            'Best Value': best_row['param_value'],
            'Best Makespan': f"{best_row['best_makespan']:.1f}",
            'Worst Value': worst_row['param_value'],
            'Worst Makespan': f"{worst_row['best_makespan']:.1f}",
            'Range': f"{makespan_range:.1f}",
            'Sensitivity %': f"{sensitivity_pct:.1f}%",
            'Recommended': best_row['param_value'],
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Sensitivity %', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)

    summary_path = output_dir / 'sensitivity_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nSummary report: {summary_path}")

    txt_path = output_dir / 'sensitivity_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Parameter Sensitivity Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Parameters ranked by sensitivity (highest impact first):\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"  {row['Parameter']:20s} | Sensitivity: {row['Sensitivity %']:>6s} | "
                    f"Best={row['Best Value']} | Worst={row['Worst Value']} | "
                    f"Range={row['Range']}h\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Recommendations:\n")
        for _, row in summary_df.iterrows():
            f.write(f"  {row['Parameter']}: use {row['Recommended']}\n")

    print(f"Text report: {txt_path}")
    return summary_df


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parameter Sensitivity Analysis for APAL-RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/sensitivity_analysis.py --param lr --values 1e-5,5e-5,1e-4
  python scripts/sensitivity_analysis.py --param gamma --values 0.99,0.995,0.999
  python scripts/sensitivity_analysis.py --all  # Run full preset grid
  python scripts/sensitivity_analysis.py --quick-test  # Quick 10-ep test
        '''
    )
    parser.add_argument('--param', type=str, help='Parameter name to sweep')
    parser.add_argument('--values', type=str, help='Comma-separated values, e.g. "1e-4,5e-4,1e-3"')
    parser.add_argument('--all', action='store_true', help='Run all preset parameter grids')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with only 10 episodes per run')
    parser.add_argument('--data', type=str, default='data/283.csv', help='Dataset path for evaluation')
    parser.add_argument('--output', type=str, default='results/sensitivity', help='Output directory')
    parser.add_argument('--episodes', type=int, default=None, help='Override short episode count')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")

    if args.quick_test:
        short_episodes = 10
        print("[Quick Test Mode] Using 10 episodes per run")
    else:
        short_episodes = args.episodes

    all_results = {}

    if args.all:
        for param_name, param_info in PRESET_PARAM_GRID.items():
            eps = short_episodes or param_info['short_episodes']
            df = run_single_param_sweep(
                param_name, param_info['values'], args.data, device, short_episodes=eps
            )
            df.to_csv(
                output_dir / f'raw_{param_name}.csv',
                index=False, encoding='utf-8-sig'
            )
            if 'error' not in df.columns or df['error'].isna().all():
                plot_sensitivity(df, param_name, output_dir)
            all_results[param_name] = df

    elif args.param and args.values:
        try:
            values_str = args.values.split(',')
            values = []
            for v in values_str:
                v = v.strip()
                if '.' in v or 'e' in v.lower():
                    values.append(float(v))
                else:
                    values.append(int(v))
        except ValueError:
            print("Error: --values must be comma-separated numbers")
            sys.exit(1)

        eps = short_episodes or 50
        df = run_single_param_sweep(args.param, values, args.data, device, short_episodes=eps)
        df.to_csv(
            output_dir / f'raw_{args.param}.csv',
            index=False, encoding='utf-8-sig'
        )
        if 'error' not in df.columns or df['error'].isna().all():
            plot_sensitivity(df, args.param, output_dir)
        all_results[args.param] = df

    else:
        parser.print_help()
        print("\nAvailable preset parameters:")
        for name, info in PRESET_PARAM_GRID.items():
            print(f"  {name:25s} -> {info['description']}")
            print(f"  {'':25s}    values: {info['values']}")
        sys.exit(0)

    if all_results:
        generate_summary_report(all_results, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
