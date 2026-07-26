"""验证主方法临时筛查模型的独立入口。

该入口仅用于 ``scripts/train_main_screen.py`` 产生的 SCG checkpoint。
它复用标准初始调度评估的全部数据、温度、排程导出和运行清单口径，
仅将策略模型类替换为 ``ScaleGatedContextHBGATPN``；不修改正式评估入口。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_model as standard_evaluator
from experiments.main_screen.screen_models import ScaleGatedContextHBGATPN


def main() -> int:
    """以 SCG 模型类运行与标准初始调度完全相同的验证协议。"""
    standard_evaluator.HBGATPN = ScaleGatedContextHBGATPN
    return standard_evaluator.cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
