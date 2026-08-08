from __future__ import annotations

from pathlib import Path


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
