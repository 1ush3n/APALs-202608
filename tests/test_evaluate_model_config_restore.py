"""正式验证入口的 checkpoint 配置恢复回归测试。

背景：已提交版 evaluate_model.py 曾直接用 CLI 实验 YAML 的默认配置构建模型/环境
（lightning_precision=16-mixed），而训练期异步验证使用 checkpoint 保存的训练配置
（bf16-mixed）。fp16/bf16 的 autocast 不同导致同一权重排出不同排程
（ent40 real_680：正式 525.51 vs 训练期异步 458.08）。
本测试锁定 restore_checkpoint_saved_config 的语义，防止该问题回归。
"""

from __future__ import annotations

from types import SimpleNamespace

from configs import Config
from evaluate_model import restore_checkpoint_saved_config


def _default_config() -> Config:
    cfg = Config()
    # CLI 实验 YAML 默认链解析后即为该默认值；本测试模拟"未恢复训练配置"的场景。
    assert cfg.lightning_precision == "16-mixed"
    assert cfg.seed == 42
    return cfg


def test_restores_bf16_training_config_over_cli_defaults() -> None:
    """核心回归：bf16 训练的 checkpoint 在 CLI 默认 16-mixed 下必须恢复为 bf16-mixed。"""
    cfg = _default_config()
    checkpoint = SimpleNamespace(
        metadata={
            "config": {
                "lightning_precision": "bf16-mixed",
                "randomize_durations": False,
            }
        },
        model_spec=SimpleNamespace(),
    )

    restore_checkpoint_saved_config(cfg, checkpoint, explicit_fields=set())

    assert cfg.lightning_precision == "bf16-mixed"
    assert cfg.randomize_durations is False


def test_cli_explicit_fields_win_over_saved_config() -> None:
    """CLI 显式覆盖字段（explicit_fields）优先于 checkpoint 保存的训练配置。"""
    cfg = _default_config()
    checkpoint = SimpleNamespace(
        metadata={
            "config": {
                "seed": 7,
                "lightning_precision": "bf16-mixed",
            }
        },
        model_spec=SimpleNamespace(),
    )

    restore_checkpoint_saved_config(cfg, checkpoint, explicit_fields={"seed"})

    assert cfg.seed == 42  # CLI 显式值不被覆盖
    assert cfg.lightning_precision == "bf16-mixed"  # 非显式字段仍恢复


def test_missing_config_metadata_is_noop() -> None:
    """checkpoint 缺少 apal_metadata.config（旧格式/非本仓库）时恢复为无操作。"""
    cfg = _default_config()
    checkpoint = SimpleNamespace(metadata={}, model_spec=SimpleNamespace())

    restore_checkpoint_saved_config(cfg, checkpoint, explicit_fields=set())

    assert cfg.lightning_precision == "16-mixed"


def test_unknown_saved_config_fields_are_ignored() -> None:
    """saved_config 中出现目标 Config 不存在的字段时静默忽略，不抛异常。"""
    cfg = _default_config()
    checkpoint = SimpleNamespace(
        metadata={
            "config": {
                "lightning_precision": "bf16-mixed",
                "no_such_field_xyz": 123,
            }
        },
        model_spec=SimpleNamespace(),
    )

    restore_checkpoint_saved_config(cfg, checkpoint, explicit_fields=set())

    assert cfg.lightning_precision == "bf16-mixed"
    assert not hasattr(cfg, "no_such_field_xyz")
