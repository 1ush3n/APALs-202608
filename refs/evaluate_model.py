import torch
import numpy as np
import argparse
import sys
import pandas as pd
from pathlib import Path

# 确保能找到根目录下的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from configs import configs
from utils.visualization import plot_gantt
from train import evaluate_model

def resolve_project_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(args):
    """
    统一的模型评估与零样本泛化测试脚本
    """
    print("--- 开始模型评估与泛化测试 ---")
    
    # 1. 确定数据路径 (支持评估默认路径或零样本泛化路径)
    data_path = resolve_project_path(args.test_data if args.test_data else configs.data_file_path)
    
    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return
        
    print(f"数据路径: {data_path}")
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=args.seed)
    print("环境初始化完成.")
    
    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = HBGATPN(configs).to(device)
    
    # 解析相对路径到根目录
    checkpoint_path = resolve_project_path(args.model_path)
        
    if not checkpoint_path.exists():
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return

    print(f"加载模型: {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    agent = PPOAgent(model, configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, device, configs.batch_size)
    
    # 3. 执行评估
    print(f"正在执行推理 (Runs: {args.num_runs}, Temperature: {args.temperature})...")
    makespan, balance, eval_reward, best_sch, eval_duration, w_util, s_util = evaluate_model(
        env, agent, num_runs=args.num_runs, temperature=args.temperature
    )
    
    print("\n--- 评估结果汇总 ---")
    print(f"平均 Makespan: {makespan:.2f} h")
    print(f"平均 Balance Std: {balance:.2f}")
    print(f"平均 Worker Utilization: {w_util*100:.1f}%")
    print(f"平均 Station Utilization: {s_util*100:.1f}%")
    print(f"平均 Reward: {eval_reward:.4f}")
    
    # 4. 导出结果 (收拢到 results/ 目录)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = data_path.stem
    prefix = f"eval_{dataset_name}"
    
    if best_sch and len(best_sch) > 0:
        tasks_data = []
        for (tid, sid, team, start, end) in best_sch:
            tasks_data.append({
                'TaskID': tid,
                'StationID': sid + 1,
                'Team': str(team),
                'Start': start,
                'End': end,
                'Duration': end - start
            })
        
        df_res = pd.DataFrame(tasks_data)
        csv_path = results_dir / f"{prefix}_schedule.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"详细排程表已保存至: {csv_path}")
        
        png_path = results_dir / f"{prefix}_gantt.png"
        plot_gantt(best_sch, png_path)
        print(f"甘特图已保存至: {png_path}")
    else:
        print("未生成排程记录（发生死锁）。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型权重路径 (例如 best_model.pth)')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据集路径，留空则使用 configs.py 默认')
    parser.add_argument('--num_runs', type=int, default=3, help='重复评估次数')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度。0表示完全贪婪执行。')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    main(args)
