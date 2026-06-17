from __future__ import annotations

import argparse

from runtime.configuration import add_common_config_arguments


def get_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APAL 强化学习训练")
    add_common_config_arguments(parser)
    parser.add_argument(
        "--ablation-no-gat", "--ablation_no_gat",
        dest="ablation_no_gat", action="store_true", default=None,
    )
    parser.add_argument(
        "--ablation-no-pointer", "--ablation_no_pointer",
        dest="ablation_no_pointer", action="store_true", default=None,
    )
    parser.add_argument(
        "--ablation-no-mask", "--ablation_no_mask",
        dest="ablation_no_mask", action="store_true", default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trainer", choices=("lightning", "legacy"), default="lightning",
        help="Lightning 是主训练入口；legacy 仅用于历史兼容。",
    )
    return parser


def get_dqn_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--memory_size", type=int, default=10000)
    return parser


def get_basic_ppo_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    return parser


def get_heuristic_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.add_argument("--num_runs", type=int, default=1)
    return parser


def get_generalization_parser() -> argparse.ArgumentParser:
    parser = get_base_parser()
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--test_data", type=str, default="data/ABC.csv")
    return parser
