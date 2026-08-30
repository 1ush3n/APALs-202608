# -*- coding: utf-8 -*-
"""Fast-Exact 性能基准指标契约测试。

覆盖 WorkerPointer v2 Fast-Exact 验收门所需指标的计算逻辑：
- GPU 利用率采样统计（均值 / P50 / P90）；
- PPO replay samples/s 与 update 总耗时分解；
- GPU 模板命中率（含空命中安全）；
- 三组基准对比行的结构化输出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.benchmark_worker_pointer_v2_fast_exact as benchmark_script
import training.fast_exact_benchmark as benchmark_runtime
from training.fast_exact_benchmark import (
    compute_replay_performance,
    compute_template_hit_rate,
    summarize_group_sizes,
    summarize_utilization,
)


def test_summarize_utilization_reports_mean_p50_p90() -> None:
    samples = [0.0, 25.0, 50.0, 75.0, 100.0]
    stats = summarize_utilization(samples)
    assert stats["mean"] == pytest.approx(50.0)
    assert stats["p50"] == pytest.approx(50.0)
    assert stats["p90"] == pytest.approx(90.0, abs=0.5)


def test_summarize_utilization_empty_samples() -> None:
    stats = summarize_utilization([])
    assert stats == {
        "available": False,
        "sample_count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
    }


def test_compute_replay_samples_per_sec() -> None:
    performance = compute_replay_performance(
        update_metrics={
            "V2/FastExact/BehaviorReplaySamples": 4096,
            "V2/FastExact/ReplayUpdateSeconds": 10.0,
            "V2/FastExact/PhysicalGroupCount": 256,
            "V2/FirstContractTotalMaxAE": 0.0005,
            "V2/FirstContractTotalMAE": 0.0001,
        },
        sample_count=4096,
        k_epochs=2,
    )
    assert performance["unique_samples"] == 4096
    assert performance["effective_replay_samples"] == 8192
    assert performance["replay_samples_per_sec"] == pytest.approx(819.2)
    assert performance["update_seconds"] == pytest.approx(10.0)
    assert performance["physical_groups"] == 256
    assert performance["first_contract_total_max_ae"] == pytest.approx(0.0005)
    assert performance["first_contract_total_mae"] == pytest.approx(0.0001)


def test_compute_replay_performance_guards_zero_seconds() -> None:
    performance = compute_replay_performance(
        update_metrics={
            "V2/FastExact/BehaviorReplaySamples": 100,
            "V2/FastExact/ReplayUpdateSeconds": 0.0,
        },
        sample_count=100,
    )
    assert performance["replay_samples_per_sec"] == pytest.approx(0.0)


def test_template_hit_rate_mixed() -> None:
    assert compute_template_hit_rate(hits=8, misses=2) == pytest.approx(0.8)


def test_template_hit_rate_no_misses() -> None:
    assert compute_template_hit_rate(hits=9, misses=0) == pytest.approx(1.0)


def test_template_hit_rate_all_misses() -> None:
    assert compute_template_hit_rate(hits=0, misses=7) == pytest.approx(0.0)


def test_group_size_summary_reports_mean_p50_and_p95() -> None:
    summary = summarize_group_sizes([1, 2, 3, 4])

    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)


def test_group_size_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="group_sizes"):
        summarize_group_sizes([])


def test_measure_operation_samples_the_operation_interval() -> None:
    """GPU 采样必须覆盖被测操作，而不是在操作开始前结束。"""
    events: list[str] = []

    class _Sampler:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> list[float]:
            events.append("stop")
            return [25.0, 75.0]

    clock_values = iter((10.0, 12.5))
    result, elapsed, samples = benchmark_runtime.measure_operation(
        lambda: events.append("operation") or "ok",
        sampler=_Sampler(),
        synchronize=lambda: events.append("sync"),
        clock=lambda: next(clock_values),
    )

    assert result == "ok"
    assert elapsed == pytest.approx(2.5)
    assert samples == [25.0, 75.0]
    assert events == ["sync", "start", "operation", "sync", "stop"]


@pytest.mark.parametrize(
    ("mode", "platform_name", "expected"),
    (
        ("v2_legacy", "Windows", 4),
        ("v2_legacy", "Linux", 4),
        ("v2_cpu_wide", "Windows", 4),
        ("v2_cpu_wide", "Linux", 16),
        ("v2_fast_exact", "Windows", 4),
        ("v2_fast_exact", "Linux", 16),
    ),
)
def test_resolve_benchmark_num_envs_uses_explicit_platform_defaults(
    mode: str,
    platform_name: str,
    expected: int,
) -> None:
    assert (
        benchmark_runtime.resolve_benchmark_num_envs(
            mode,
            platform_name=platform_name,
            override=None,
        )
        == expected
    )


def test_run_modes_in_subprocesses_uses_one_process_per_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父进程必须为每个模式启动独立 worker，并只汇总落盘结果。"""
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        assert Path(cwd) == benchmark_script.PROJECT_ROOT
        assert check is True
        calls.append(list(command))
        mode = command[command.index("--worker-mode") + 1]
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(
            json.dumps({"mode": mode, "worker_pid": len(calls)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(benchmark_script.subprocess, "run", fake_run)
    args = argparse.Namespace(
        num_envs=None,
        batch_size=256,
        max_steps=0,
        updates=1,
        rollout_episodes=2,
        seed=42,
        warmup=True,
    )
    modes = ["v2_legacy", "v2_cpu_wide", "v2_fast_exact"]

    rows = benchmark_script.run_modes_in_subprocesses(
        args,
        data_path=tmp_path / "data.csv",
        modes=modes,
        result_dir=tmp_path,
    )

    assert [row["mode"] for row in rows] == modes
    assert [row["worker_pid"] for row in rows] == [1, 2, 3]
    assert len(calls) == 3
    assert all(command[0] == sys.executable for command in calls)
    assert all("--worker-mode" in command for command in calls)


def test_run_mode_samples_rollout_and_update_during_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单模式 worker 应分别在 rollout 与 PPO update 执行期间采样。"""
    events: list[str] = []

    class _Sampler:
        def start(self) -> None:
            events.append("sample_start")

        def stop(self) -> list[float]:
            events.append("sample_stop")
            return [50.0]

    class _Agent:
        batch_size = 256
        k_epochs = 2
        accumulation_steps = 16

        def update(self, memory, env, *, current_ep):
            events.append("update")
            return {
                "V2/FastExact/BehaviorReplaySamples": 2.0,
                "V2/FastExact/PhysicalGroupCount": 1.0,
                "V2/FirstContractTotalMaxAE": 0.0,
                "V2/FirstContractTotalMAE": 0.0,
            }

    class _Service:
        device = benchmark_script.torch.device("cpu")
        agent = _Agent()
        _fast_exact_builder = None

        def _collect_episode(self, episode):
            events.append("rollout")
            return None, SimpleNamespace(steps_per_second=10.0)

        def collect(self, update_index):
            return SimpleNamespace(
                memory=SimpleNamespace(states=[{}, {}]),
                env=object(),
                episode=update_index,
            )

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(benchmark_script, "NvidiaSmiSampler", _Sampler)
    monkeypatch.setattr(
        benchmark_script,
        "_apply_mode_overrides",
        lambda *args, **kwargs: 4,
    )
    monkeypatch.setattr(
        benchmark_script,
        "_build_service",
        lambda *args, **kwargs: (_Service(), object()),
    )
    monkeypatch.setattr(
        benchmark_script.torch.cuda,
        "is_available",
        lambda: False,
    )

    row = benchmark_script.run_mode(
        "v2_fast_exact",
        num_envs_override=None,
        batch_size=256,
        max_steps=0,
        rollout_episodes=1,
        updates=1,
        seed=42,
        data_path=tmp_path / "data.csv",
        warmup=False,
    )

    assert events == [
        "sample_start",
        "rollout",
        "sample_stop",
        "sample_start",
        "update",
        "sample_stop",
        "close",
    ]
    assert row["gpu_utilization"]["rollout"]["sample_count"] == 1
    assert row["gpu_utilization"]["update"]["sample_count"] == 1
    assert row["replay"]["effective_replay_samples"] == 4.0


def test_run_mode_exports_fast_exact_profile_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Agent:
        batch_size = 256
        k_epochs = 1
        accumulation_steps = 1

        def update(self, memory, env, *, current_ep):
            return {
                "V2/FastExact/BehaviorReplaySamples": 2.0,
                "V2/FastExact/PhysicalGroupCount": 1.0,
                "V2/FirstContractTotalMaxAE": 0.0,
                "V2/FirstContractTotalMAE": 0.0,
                "V2/FastExact/Profile/EncoderCalls": 1.0,
                "V2/FastExact/Profile/ReplaySamplesPerSec": 2.0,
            }

    class _Service:
        device = benchmark_script.torch.device("cpu")
        agent = _Agent()
        _fast_exact_builder = None

        def collect(self, update_index):
            return SimpleNamespace(
                memory=SimpleNamespace(states=[{}, {}]),
                env=object(),
                episode=update_index,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(benchmark_script, "NvidiaSmiSampler", lambda: SimpleNamespace(
        start=lambda: None,
        stop=lambda: [],
    ))
    monkeypatch.setattr(
        benchmark_script,
        "_apply_mode_overrides",
        lambda *args, **kwargs: 2,
    )
    monkeypatch.setattr(
        benchmark_script,
        "_build_service",
        lambda *args, **kwargs: (_Service(), object()),
    )
    monkeypatch.setattr(benchmark_script.torch.cuda, "is_available", lambda: False)

    row = benchmark_script.run_mode(
        "v2_fast_exact",
        num_envs_override=None,
        batch_size=256,
        max_steps=0,
        rollout_episodes=0,
        updates=1,
        seed=42,
        data_path=tmp_path / "data.csv",
        warmup=False,
    )

    assert row["num_envs"] == 2
    assert row["profile"]["EncoderCalls"] == 1.0
    assert row["profile"]["ReplaySamplesPerSec"] == 2.0


def test_fast_exact_benchmark_enables_profile_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_update(values: dict[str, object]) -> None:
        captured.update(values)

    monkeypatch.setattr(benchmark_script.configs, "update_from_dict", capture_update)

    benchmark_script._apply_mode_overrides(
        "v2_fast_exact",
        num_envs_override=2,
        batch_size=256,
        max_steps=0,
        seed=42,
        data_path=tmp_path / "680.csv",
    )

    assert captured["worker_pointer_v2_fast_exact_profile"] is True
