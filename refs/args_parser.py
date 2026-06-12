import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description="APAL 强化学习项目基础参数")
    parser.add_argument("--config", action="append", default=[], help="分层 YAML 配置文件路径；可重复传入，后者覆盖前者")
    parser.add_argument("--data_path", type=str, default="data/283.csv", help="数据文件路径（默认 data/283.csv）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证复现性）")
    parser.add_argument("--max_episodes", type=int, default=3000, help="最大训练轮数（默认 3000）")
    parser.add_argument("--log_dir", type=str, default="tf-logs", help="日志保存目录（默认 tf-logs）")
    parser.add_argument("--result_dir", type=str, default="results", help="结果归档目录（默认 results）")
    parser.add_argument("--ablation_no_gat", action="store_true", help="消融实验：禁用 GAT 模块")
    parser.add_argument("--ablation_no_pointer", action="store_true", help="消融实验：禁用指针网络")
    parser.add_argument("--ablation_no_mask", action="store_true", help="消融实验：禁用硬掩码，改用软惩罚")
    parser.add_argument("--resume", action="store_true", help="是否自动恢复最新 checkpoint")
    return parser


def get_dqn_parser():
    parser = get_base_parser()
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="探索率初始值")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="探索率最小值")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="探索率衰减率")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--memory_size", type=int, default=10000, help="经验回放池大小")
    return parser


def get_basic_ppo_parser():
    parser = get_base_parser()
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 裁剪系数")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO 更新批量大小")
    return parser


def get_heuristic_parser():
    parser = get_base_parser()
    parser.add_argument("--num_runs", type=int, default=1, help="启发式策略运行轮数")
    return parser


def get_generalization_parser():
    parser = get_base_parser()
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="预训练模型权重路径")
    parser.add_argument("--test_data", type=str, default="data/ABC.csv", help="泛化测试数据集路径")
    return parser
