"""验证主方法临时筛查模型的独立入口。

该入口复用标准初始调度评估的全部数据、温度、排程导出和运行清单口径，
仅按 ``screen_model`` 替换策略模型类；不修改正式评估入口。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_model as standard_evaluator
from models.hb_gat_pn import HBGATPN
from experiments.main_screen.screen_models import (
    DualAttentionContextHBGATPN,
    ScaleGatedContextHBGATPN,
)


def _resolve_screen_model_class(screen_model: str) -> type[HBGATPN]:
    normalized = str(screen_model).strip().lower()
    model_classes: dict[str, type[HBGATPN]] = {
        "full": HBGATPN,
        "scg": ScaleGatedContextHBGATPN,
        "dual_attention": DualAttentionContextHBGATPN,
    }
    try:
        return model_classes[normalized]
    except KeyError as exc:
        raise ValueError("screen_model 仅允许 full、scg 或 dual_attention") from exc


def _extract_screen_model(argv: list[str] | None) -> tuple[str, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    screen_model = "scg"
    forwarded: list[str] = []
    for token in raw_args:
        key, separator, value = token.partition("=")
        if key == "screen_model" and separator:
            screen_model = value
        else:
            forwarded.append(token)
    return screen_model, forwarded


def main(argv: list[str] | None = None) -> int:
    """以筛查模型类运行与标准初始调度完全相同的验证协议。"""
    screen_model, forwarded = _extract_screen_model(argv)
    standard_evaluator.HBGATPN = _resolve_screen_model_class(screen_model)
    return standard_evaluator.cli_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
