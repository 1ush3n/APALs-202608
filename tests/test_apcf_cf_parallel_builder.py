from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_plan_graph_jobs_preserves_split_and_manifest_order() -> None:
    """并行任务计划必须保持协议 split 与 manifest 内顺序，不能按完成时间重排。"""
    from scripts.build_anchor_proposal_cf_data import _plan_graph_jobs

    split = {
        "pretrain": [
            {"file": "syn_a.csv", "sha256": "a" * 64},
            {"file": "syn_b.csv", "sha256": "b" * 64},
        ],
        "frozen_diagnostic": [
            {"file": "syn_c.csv", "sha256": "c" * 64},
        ],
        "ppo_only": [{"file": "syn_d.csv", "sha256": "d" * 64}],
    }

    jobs = _plan_graph_jobs(split, max_graphs=0)

    assert [(job.index, job.split, job.file_name) for job in jobs] == [
        (0, "pretrain", "syn_a.csv"),
        (1, "pretrain", "syn_b.csv"),
        (2, "frozen_diagnostic", "syn_c.csv"),
    ]


def test_merge_graph_artifacts_rewrites_paths_in_plan_order(tmp_path: Path) -> None:
    """主进程合并时必须按任务序号写入，并把 worker 相对路径改为正式样本路径。"""
    from scripts.build_anchor_proposal_cf_data import (
        GraphBuildJob,
        GraphBuildResult,
        _merge_graph_artifacts,
    )

    output_dir = tmp_path / "asset"
    worker_a = tmp_path / "worker_a"
    worker_b = tmp_path / "worker_b"
    for worker_dir, stem in ((worker_a, "a"), (worker_b, "b")):
        (worker_dir / "samples").mkdir(parents=True)
        (worker_dir / "samples" / f"obs_{stem}.pt").write_bytes(b"observation")
        (worker_dir / "samples" / f"{stem}.npz").write_bytes(b"sample")
    output_dir.mkdir()
    (output_dir / "samples").mkdir()

    job_a = GraphBuildJob(0, "pretrain", "syn_a.csv", "a" * 64, worker_a)
    job_b = GraphBuildJob(1, "frozen_diagnostic", "syn_b.csv", "b" * 64, worker_b)
    # 特意反转传入顺序，模拟较晚任务先完成。
    rows = _merge_graph_artifacts(
        output_dir,
        [
            GraphBuildResult(job_b, [{"obs_pt": "samples/obs_b.pt", "npz_path": "samples/b.npz"}]),
            GraphBuildResult(job_a, [{"obs_pt": "samples/obs_a.pt", "npz_path": "samples/a.npz"}]),
        ],
    )

    assert [row["csv_sha256"] for row in rows] == ["a" * 64, "b" * 64]
    assert [row["split"] for row in rows] == ["pretrain", "frozen_diagnostic"]
    assert (output_dir / "samples" / "obs_a.pt").is_file()
    assert (output_dir / "samples" / "a.npz").is_file()
    assert (output_dir / "samples" / "obs_b.pt").is_file()
    assert (output_dir / "samples" / "b.npz").is_file()


def test_merge_graph_artifacts_reuses_shared_observation_file(tmp_path: Path) -> None:
    """同一状态的多个候选共用观测文件时，合并不得重复移动或报缺失。"""
    from scripts.build_anchor_proposal_cf_data import (
        GraphBuildJob,
        GraphBuildResult,
        _merge_graph_artifacts,
    )

    output_dir = tmp_path / "asset"
    worker_dir = tmp_path / "worker"
    (worker_dir / "samples").mkdir(parents=True)
    (worker_dir / "samples" / "obs_shared.pt").write_bytes(b"observation")
    (worker_dir / "samples" / "candidate_a.npz").write_bytes(b"candidate_a")
    (worker_dir / "samples" / "candidate_b.npz").write_bytes(b"candidate_b")
    (output_dir / "samples").mkdir(parents=True)
    job = GraphBuildJob(0, "pretrain", "syn_a.csv", "a" * 64, worker_dir)

    rows = _merge_graph_artifacts(
        output_dir,
        [
            GraphBuildResult(
                job,
                [
                    {"obs_pt": "samples/obs_shared.pt", "npz_path": "samples/candidate_a.npz"},
                    {"obs_pt": "samples/obs_shared.pt", "npz_path": "samples/candidate_b.npz"},
                ],
            )
        ],
    )

    assert len(rows) == 2
    assert all(row["obs_pt"] == "samples/obs_shared.pt" for row in rows)
    assert (output_dir / "samples" / "obs_shared.pt").is_file()
    assert not worker_dir.exists()


def _run_build_args(tmp_path: Path) -> SimpleNamespace:
    """构造 run_build 所需的参数；源 manifest 与映射数据均为临时文件。"""
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "explicit_fiveskill_v1",
                "files": [
                    {"file": "syn_x.csv", "sha256": hashlib.sha256(b"x").hexdigest()}
                ],
            }
        ),
        encoding="utf-8",
    )
    data_file = tmp_path / "ref.csv"
    data_file.write_bytes(b"ref")
    return SimpleNamespace(
        manifest=str(manifest),
        data_file=str(data_file),
        output_dir=str(tmp_path / "asset"),
        max_graphs=0,
        max_episode_steps=1200,
        max_candidates=4,
        workers=1,
        worker_torch_threads=1,
        seed=42,
    )


def test_run_build_workers_tmp_outside_asset_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker 临时分片必须位于系统临时目录，而不是正式资产目录内。"""
    import scripts.build_anchor_proposal_cf_data as build_mod

    workers_tmp = tmp_path / "workers_tmp"
    workers_tmp.mkdir()
    monkeypatch.setattr(
        build_mod,
        "tempfile",
        SimpleNamespace(mkdtemp=lambda prefix: str(workers_tmp)),
    )
    # 空任务计划：跳过真实 CSV 构建，只验证临时目录位置与清理。
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [])

    result = build_mod.run_build(_run_build_args(tmp_path))

    assert result["samples"] == 0
    output_dir = tmp_path / "asset"
    assert not (output_dir / ".workers").exists()
    assert not workers_tmp.exists(), "构建结束后 worker 临时目录必须被清理"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "manifest.json",
        "samples",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command_args"]["worker_tmp_dir"] == str(workers_tmp)
    assert str(workers_tmp) not in str(output_dir)


def test_run_build_worker_exception_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker 异常时，主进程 finally 必须仍清理系统临时目录并传播异常。"""
    import scripts.build_anchor_proposal_cf_data as build_mod

    workers_tmp = tmp_path / "workers_tmp"
    workers_tmp.mkdir()
    monkeypatch.setattr(
        build_mod,
        "tempfile",
        SimpleNamespace(mkdtemp=lambda prefix: str(workers_tmp)),
    )
    job = build_mod.GraphBuildJob(
        index=0,
        split="pretrain",
        file_name="syn_x.csv",
        csv_sha256="x" * 64,
    )
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [job])

    # 让 run_build 的 CSV 存在性检查通过：把 data/scale_400_800_datasets/ 下
    # 不存在的路径重定向到临时假文件，从而真正执行到被 monkeypatch 的 worker。
    real_workspace_path = build_mod._workspace_path
    fake_csv = tmp_path / "syn_x.csv"
    fake_csv.write_bytes(b"csv")

    def _fake_workspace_path(value: str | Path) -> Path:
        path = real_workspace_path(value)
        if "scale_400_800_datasets" in str(path) and not path.is_file():
            return fake_csv
        return path

    monkeypatch.setattr(build_mod, "_workspace_path", _fake_workspace_path)

    def _boom(request: build_mod.GraphBuildRequest) -> None:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(build_mod, "_build_graph_worker", _boom)
    monkeypatch.setattr(
        build_mod,
        "_run_graph_preflight",
        lambda requests, workers: [
            build_mod.GraphPreflightResult(requests[0].job, (), 0)
        ],
    )
    monkeypatch.setattr(build_mod, "_validate_preflight_results", lambda results: results)

    with pytest.raises(RuntimeError, match="worker boom"):
        build_mod.run_build(_run_build_args(tmp_path))

    output_dir = tmp_path / "asset"
    assert not (output_dir / ".workers").exists()
    assert not workers_tmp.exists(), "异常路径也必须清理 worker 临时目录"


def test_run_build_parallel_pool_empty_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """workers>1 时 spawn 池正常启停；空任务计划下不产生任何正式残留。"""
    import scripts.build_anchor_proposal_cf_data as build_mod

    workers_tmp = tmp_path / "workers_tmp"
    workers_tmp.mkdir()
    monkeypatch.setattr(
        build_mod,
        "tempfile",
        SimpleNamespace(mkdtemp=lambda prefix: str(workers_tmp)),
    )
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [])
    args = _run_build_args(tmp_path)
    args.workers = 2

    result = build_mod.run_build(args)

    assert result["samples"] == 0
    assert not workers_tmp.exists()
    assert not (tmp_path / "asset" / ".workers").exists()


def test_run_build_reuses_existing_empty_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.build_anchor_proposal_cf_data as build_mod

    output_dir = tmp_path / "asset"
    output_dir.mkdir()
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [])
    result = build_mod.run_build(_run_build_args(tmp_path))
    assert result["samples"] == 0
    assert (output_dir / "manifest.json").is_file()


def test_run_build_rejects_nonempty_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.build_anchor_proposal_cf_data as build_mod

    output_dir = tmp_path / "asset"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [])
    with pytest.raises(FileExistsError, match="empty"):
        build_mod.run_build(_run_build_args(tmp_path))


def test_run_build_preserves_worker_and_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_anchor_proposal_cf_data as build_mod

    workers_tmp = tmp_path / "workers_tmp"
    workers_tmp.mkdir()
    monkeypatch.setattr(
        build_mod,
        "tempfile",
        SimpleNamespace(mkdtemp=lambda prefix: str(workers_tmp)),
    )
    job = build_mod.GraphBuildJob(0, "pretrain", "syn_x.csv", "x" * 64)
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [job])
    real_workspace_path = build_mod._workspace_path
    fake_csv = tmp_path / "syn_x.csv"
    fake_csv.write_bytes(b"csv")

    def fake_workspace_path(value: str | Path) -> Path:
        path = real_workspace_path(value)
        if "scale_400_800_datasets" in str(path) and not path.is_file():
            return fake_csv
        return path

    monkeypatch.setattr(build_mod, "_workspace_path", fake_workspace_path)
    monkeypatch.setattr(
        build_mod,
        "_run_graph_preflight",
        lambda requests, workers: [
            build_mod.GraphPreflightResult(requests[0].job, (), 0)
        ],
    )
    monkeypatch.setattr(build_mod, "_validate_preflight_results", lambda results: results)
    monkeypatch.setattr(
        build_mod,
        "_build_graph_worker",
        lambda request: (_ for _ in ()).throw(RuntimeError("worker boom")),
    )
    monkeypatch.setattr(
        build_mod.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup boom")),
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        build_mod.run_build(_run_build_args(tmp_path))
    message = " ".join(str(exc) for exc in captured.value.exceptions)
    assert "worker boom" in message
    assert "cleanup boom" in message
    assert str(workers_tmp) in message


def test_run_build_raises_cleanup_error_without_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_anchor_proposal_cf_data as build_mod

    workers_tmp = tmp_path / "workers_tmp"
    workers_tmp.mkdir()
    monkeypatch.setattr(
        build_mod,
        "tempfile",
        SimpleNamespace(mkdtemp=lambda prefix: str(workers_tmp)),
    )
    monkeypatch.setattr(build_mod, "_plan_graph_jobs", lambda split, max_graphs: [])
    monkeypatch.setattr(
        build_mod.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup boom")),
    )

    with pytest.raises(RuntimeError, match="cleanup boom"):
        build_mod.run_build(_run_build_args(tmp_path))


def test_preflight_rejects_graph_without_four_distinct_states() -> None:
    from scripts.build_anchor_proposal_cf_data import (
        GraphBuildJob,
        GraphPreflightResult,
        _validate_preflight_results,
    )

    job = GraphBuildJob(0, "pretrain", "syn_x.csv", "x" * 64)
    result = GraphPreflightResult(job, (("x" * 64, 1, 0, 0, (0,)),), 7)
    with pytest.raises(ValueError, match="4"):
        _validate_preflight_results([result])


def test_replayed_state_must_match_preflight_key() -> None:
    from scripts.build_anchor_proposal_cf_data import _assert_state_key_matches

    expected = ("x" * 64, 3, 0, 0, (0,))
    actual = ("x" * 64, 3, 0, 0, (1,))
    with pytest.raises(RuntimeError, match="preflight"):
        _assert_state_key_matches(expected, actual)
