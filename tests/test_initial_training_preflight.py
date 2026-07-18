"""当前五技能初始调度训练的快速前检查。

该套件只覆盖真实数据的“环境初始化 → 图构建 → 前向 → 首步合法动作”，
不在本机执行正式训练或大规模评价。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.graph_baseline import select_graph_action
from baselines.literature.common import rollout_step_limit, save_literature_checkpoint
from baselines.literature_dqn.train_graph_ddqn_apal import GraphDDQNAPAL
from baselines.literature_ppo.train_l2d_ppo_apal import SimpleHeteroGATPPO
from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.checkpoints import ModelSpec, apply_checkpoint_model_spec
from runtime.reschedule_eval import load_warm_start_weights_with_input_expansion
from tests.runtime_safety import seed_everything, temporary_config
from worker_feature_layout import resolve_worker_feature_layout


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "283": PROJECT_ROOT / "data" / "283.csv",
    "680": PROJECT_ROOT / "data" / "680.csv",
    "2338": PROJECT_ROOT / "data" / "2338.csv",
    "3182": PROJECT_ROOT / "data" / "3182.csv",
}


def _base_overrides() -> dict[str, object]:
    """返回不受其他测试残留状态影响的当前初始调度配置。"""

    return {
        "n_m": 5,
        "n_w": 80,
        "n_w_min": 60,
        "task_feat_dim": 18,
        "worker_feat_dim": 17,
        "station_feat_dim": 15,
        "num_skill_types": 5,
        "worker_skill_feature_slots": 5,
        "skill_feat_dim": 11,
        "use_skill_hub": True,
        "skill_hub_bidirectional": True,
        "ablation_no_gat": False,
        "ablation_no_mask": False,
        "ablation_no_pointer": False,
        "use_attention_critic": True,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_online_duration_perturb": False,
        "enable_worker_fatigue": False,
        "use_schedule_free": False,
    }


def _make_env(data_path: Path) -> tuple[AirLineEnv_Graph, object]:
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=42)
    return env, env.reset(randomize_duration=False, randomize_workers=False, seed=42)


def _assert_worker_and_skill_contract(env: AirLineEnv_Graph, observation: object) -> None:
    worker_layout = resolve_worker_feature_layout(configs)
    task_x = observation["task"].x
    worker_x = observation["worker"].x
    physical_mask = env.task_static_feat[:, 1].ge(0)
    task_skills = env.task_static_feat[:, 1].long()

    assert worker_layout.num_skill_types == worker_layout.skill_slots == 5
    assert worker_layout.total_dim == 17
    assert worker_x.shape == (env.num_workers, worker_layout.total_dim)
    assert observation["skill"].x.shape == (5, 11)
    assert torch.all(worker_x[:, worker_layout.lock_slice].sum(dim=1) == 1.0)
    assert torch.all(torch.argmax(worker_x[:, worker_layout.lock_slice], dim=1) == 0)
    assert torch.allclose(
        worker_x[:, worker_layout.fatigue_idx],
        torch.ones(env.num_workers, dtype=worker_x.dtype),
    )

    task_skill_x = task_x[:, 5 : 5 + worker_layout.num_skill_types]
    assert torch.all(task_skill_x[~physical_mask] == 0.0)
    assert torch.all(task_skill_x[physical_mask].sum(dim=1) == 1.0)
    assert set(task_skills[physical_mask].tolist()) == set(range(worker_layout.num_skill_types))

    task_skill_edges = observation["skill", "required_by", "task"].edge_index
    incoming_count = torch.bincount(task_skill_edges[1], minlength=env.num_tasks)
    assert torch.equal(incoming_count, physical_mask.to(dtype=torch.long))

    worker_skills = worker_x[:, worker_layout.skill_slice] > 0.5
    for skill_id in range(worker_layout.num_skill_types):
        demands = env.task_static_feat[task_skills == skill_id, 2]
        assert demands.numel() > 0
        assert int(worker_skills[:, skill_id].sum().item()) >= int(demands.max().item())


@pytest.mark.parametrize("dataset_name", tuple(DATASETS))
def test_all_real_datasets_satisfy_five_skill_graph_contract(dataset_name: str) -> None:
    seed_everything(42)
    with temporary_config(configs, _base_overrides()):
        env, observation = _make_env(DATASETS[dataset_name])
        _assert_worker_and_skill_contract(env, observation)


def _assert_action_is_legal(env: AirLineEnv_Graph, action: tuple[int, int, list[int]]) -> None:
    task_idx, station_idx, team = action
    task_skill = int(env.task_static_feat[task_idx, 1].item())
    demand = int(env.task_static_feat[task_idx, 2].item())
    if task_skill >= 0:
        assert 0 <= station_idx < env.num_stations
        assert task_skill in range(int(configs.num_skill_types))
        assert len(team) == demand
        assert len(team) == len(set(team))
        assert all(env.worker_skill_matrix[worker_idx, task_skill] > 0.5 for worker_idx in team)
    else:
        # 层级虚拟节点是零工时工艺推进节点，不适用物理工序的技能/人数约束。
        assert demand == 0
        assert station_idx == -1
        assert team == []
    _, reward, _, info = env.step(action)
    assert np.isfinite(float(reward))
    assert not bool(info.get("invalid_action", False)), info
    assert not bool(info.get("deadlock", False)), info


@pytest.mark.parametrize(
    ("variant", "overrides"),
    (
        ("full", {}),
        ("no_message_passing", {"graph_encoder_mode": "none"}),
        ("local_only", {"actor_context_mode": "local_only"}),
    ),
)
def test_main_method_and_active_ablations_complete_a_legal_first_step(
    variant: str,
    overrides: dict[str, object],
) -> None:
    seed_everything(42)
    config_overrides = _base_overrides()
    config_overrides.update(overrides)
    with temporary_config(configs, config_overrides):
        env, observation = _make_env(DATASETS["283"])
        model = HBGATPN(configs).eval()
        with torch.inference_mode():
            encoded, context = model(observation)
        assert torch.isfinite(encoded["task"]).all(), variant
        assert torch.isfinite(encoded["worker"]).all(), variant
        assert torch.isfinite(context).all(), variant

        task_mask, station_mask, worker_mask = env.get_masks()
        agent = PPOAgent(
            model=model,
            lr=float(configs.lr),
            gamma=float(configs.gamma),
            k_epochs=int(configs.k_epochs),
            eps_clip=float(configs.eps_clip),
            device=torch.device("cpu"),
            batch_size=2,
            total_timesteps=1,
            config=configs,
        )
        action, *_ = agent.select_action(
            observation,
            mask_task=task_mask,
            mask_station_matrix=station_mask,
            mask_worker=worker_mask,
            deterministic=True,
            temperature=0.0,
            is_eval=True,
        )
        assert action is not None, variant
        _assert_action_is_legal(env, action)


@pytest.mark.parametrize("model_type", (SimpleHeteroGATPPO, GraphDDQNAPAL))
def test_literature_graph_baselines_accept_current_five_skill_graph(model_type: type) -> None:
    seed_everything(42)
    with temporary_config(configs, _base_overrides()):
        env, observation = _make_env(DATASETS["283"])
        model = model_type(configs).eval()
        with torch.inference_mode():
            encoded, context = model(observation)
        assert torch.isfinite(encoded["task"]).all()
        assert torch.isfinite(encoded["worker"]).all()
        assert torch.isfinite(context).all()

        result = select_graph_action(
            model,
            observation,
            masks=env.get_masks(),
            device=torch.device("cpu"),
            deterministic=True,
            temperature=0.0,
        )
        assert result.action is not None
        _assert_action_is_legal(env, result.action)


def test_current_runtime_rejects_historical_22_dim_checkpoint_spec() -> None:
    with temporary_config(configs, _base_overrides()):
        legacy_spec = ModelSpec(
            resource_graph_mode="skill_hub_bidirectional",
            worker_feat_dim=22,
        )
        with pytest.raises(ValueError, match="历史 22 维 checkpoint"):
            apply_checkpoint_model_spec(configs, legacy_spec)


def test_literature_rollout_step_limit_only_truncates_explicit_smoke_runs() -> None:
    with temporary_config(configs, {**_base_overrides(), "rollout_max_steps": 0}):
        assert rollout_step_limit(283) == 566
    with temporary_config(configs, {**_base_overrides(), "rollout_max_steps": 5}):
        assert rollout_step_limit(283) == 5


def test_literature_checkpoint_records_current_five_skill_layout(tmp_path: Path) -> None:
    with temporary_config(configs, _base_overrides()):
        checkpoint_path = tmp_path / "baseline.pth"
        args = SimpleNamespace(
            train_data_path_or_dir=str(DATASETS["283"]),
            data_path=str(DATASETS["283"]),
        )
        save_literature_checkpoint(
            checkpoint_path,
            algorithm="test",
            literature_family="test",
            model=torch.nn.Linear(1, 1),
            best_makespan=1.0,
            args=args,
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert payload["worker_feat_dim"] == 17
        assert payload["num_skill_types"] == 5
        assert payload["worker_skill_feature_slots"] == 5
        assert payload["worker_feature_layout_version"] == "five_skill_v2"


def test_reschedule_warm_start_rejects_legacy_checkpoint_format(tmp_path: Path) -> None:
    with temporary_config(configs, _base_overrides()):
        target_model = HBGATPN(configs)
        historical_state = target_model.state_dict()
        worker_key = "embedder.worker_emb.0.weight"
        historical_state[worker_key] = torch.zeros(
            (historical_state[worker_key].shape[0], 22),
            dtype=historical_state[worker_key].dtype,
        )
        checkpoint_path = tmp_path / "historical_22_dim.pth"
        torch.save({"model_state_dict": historical_state}, checkpoint_path)
        with pytest.raises(ValueError, match="checkpoint 格式不兼容"):
            load_warm_start_weights_with_input_expansion(
                target_model,
                checkpoint_path,
                torch.device("cpu"),
            )
