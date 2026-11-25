"""条件式团队门控的 best-anchor 教师生命周期管理。"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from runtime.checkpoints import (
    LoadedCheckpoint,
    build_model_spec,
    load_checkpoint,
    load_policy_weights,
)
from runtime.initial_checkpoint_selection import (
    load_initial_checkpoint_selection_manifest,
    sha256_file,
)
from runtime.paths import resolve_workspace_path


@dataclass(frozen=True)
class TeacherState:
    """写入训练 checkpoint 的、可审计的教师身份。"""

    sha256: str
    selection_score: float
    source: str
    version: int
    updates_since_reload: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "selection_score": self.selection_score,
            "source": self.source,
            "version": self.version,
            "updates_since_reload": self.updates_since_reload,
        }


class BestAnchorTeacherManager:
    """只在 PPO 更新边界加载、冻结并替换已验证的最佳策略。"""

    def __init__(
        self,
        *,
        config: Any,
        device: torch.device,
        model_factory: Callable[[], torch.nn.Module],
        checkpoint_dir: Path | None,
        make_schedulefree_optimizer: Callable[[torch.nn.Module], Any] | None,
        use_schedule_free: bool,
    ) -> None:
        self.config = config
        self.device = device
        self.model_factory = model_factory
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.make_schedulefree_optimizer = make_schedulefree_optimizer
        self.use_schedule_free = bool(use_schedule_free)
        self.teacher: torch.nn.Module | None = None
        self.state: TeacherState | None = None
        self._freshly_activated = False
        self._expected_spec = build_model_spec(config)
        self._expected_protocol_id = ""
        self._expected_manifest_sha256 = ""
        manifest_path = str(getattr(config, "checkpoint_selection_manifest_path", "")).strip()
        if manifest_path:
            manifest = load_initial_checkpoint_selection_manifest(manifest_path)
            self._expected_protocol_id = manifest.protocol_id
            self._expected_manifest_sha256 = manifest.sha256
        self._load_external_if_requested()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "best_anchor_distill_enabled", False))

    @property
    def active(self) -> bool:
        return self.teacher is not None and self.state is not None

    def _load_external_if_requested(self) -> None:
        path_text = str(
            getattr(self.config, "best_anchor_distill_external_checkpoint_path", "")
        ).strip()
        if not path_text:
            return
        external_protocol = str(
            getattr(self.config, "best_anchor_distill_external_protocol_id", "")
        ).strip()
        external_manifest = str(
            getattr(self.config, "best_anchor_distill_external_manifest_sha256", "")
        ).strip().lower()
        if external_protocol != self._expected_protocol_id:
            raise ValueError("外部 best-anchor 教师的选择协议与当前训练不一致")
        if external_manifest != self._expected_manifest_sha256:
            raise ValueError("外部 best-anchor 教师的 manifest 哈希与当前训练不一致")
        self._activate(
            path=resolve_workspace_path(path_text),
            selection_score=float(self.config.best_anchor_distill_external_selection_score),
            source="external",
        )

    def _validate_checkpoint(self, checkpoint: LoadedCheckpoint) -> None:
        if checkpoint.model_spec != self._expected_spec:
            raise ValueError(
                "best-anchor 教师的模型规格与学生不一致；"
                "动作范围、五技能特征或编码结构不能混用"
            )

    def _activate(self, *, path: Path, selection_score: float, source: str) -> None:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"best-anchor 教师 checkpoint 不存在: {path}")
        checkpoint = load_checkpoint(path, map_location="cpu")
        self._validate_checkpoint(checkpoint)
        model = self.model_factory().to(self.device)
        load_policy_weights(model, checkpoint, strict=True)
        if self.use_schedule_free:
            if self.make_schedulefree_optimizer is None:
                raise RuntimeError("ScheduleFree 教师缺少优化器重建器")
            optimizer_states = checkpoint.payload.get("optimizer_states", [])
            if not optimizer_states:
                raise ValueError("ScheduleFree best-anchor checkpoint 缺少 optimizer_states")
            optimizer = self.make_schedulefree_optimizer(model)
            optimizer.load_state_dict(optimizer_states[0])
            optimizer.eval()
            del optimizer
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        teacher_sha = sha256_file(path)
        previous_version = 0 if self.state is None else self.state.version
        self.teacher = model
        self.state = TeacherState(
            sha256=teacher_sha,
            selection_score=float(selection_score),
            source=str(source),
            version=previous_version + 1,
            updates_since_reload=0,
        )
        self._freshly_activated = True
        print(
            "NNNNNNNNNN [BestAnchor] 教师已加载 "
            f"version={self.state.version} source={source} "
            f"score={selection_score:.6f} sha={teacher_sha[:12]}",
            flush=True,
        )

    def refresh_from_run_best(self) -> bool:
        """读取异步 worker 的提交标记；不完整发布仅延后，不替换旧教师。"""
        if not self.enabled or self.checkpoint_dir is None:
            return False
        state_path = self.checkpoint_dir / "async_eval" / "state" / "best.json"
        best_path = self.checkpoint_dir / "best.ckpt"
        if not state_path.is_file() or not best_path.is_file():
            return False
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            if str(raw.get("selection_protocol_id", "")) != self._expected_protocol_id:
                return False
            if str(raw.get("selection_manifest_sha256", "")).lower() != self._expected_manifest_sha256:
                return False
            expected_sha = str(raw.get("best_checkpoint_sha256", "")).lower()
            actual_sha = sha256_file(best_path)
            if expected_sha != actual_sha:
                return False
            score = float(raw["selection_score"])
            if not math.isfinite(score):
                return False
            current_score = float("inf") if self.state is None else self.state.selection_score
            minimum = float(getattr(self.config, "best_anchor_distill_min_improvement", 0.0))
            if score >= current_score - minimum:
                return False
            self._activate(path=best_path, selection_score=score, source="run_best")
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def on_update_started(self) -> dict[str, float]:
        reloaded = self.refresh_from_run_best()
        if self.state is not None and not reloaded and not self._freshly_activated:
            self.state = TeacherState(
                sha256=self.state.sha256,
                selection_score=self.state.selection_score,
                source=self.state.source,
                version=self.state.version,
                updates_since_reload=self.state.updates_since_reload + 1,
            )
        self._freshly_activated = False
        return {
            "Distill/Enabled": 1.0 if self.active else 0.0,
            "Distill/TeacherReloaded": 1.0 if reloaded else 0.0,
            "Distill/TeacherVersion": 0.0 if self.state is None else float(self.state.version),
            "Distill/TeacherScore": 0.0 if self.state is None else float(self.state.selection_score),
        }

    def current_lambda(self) -> float:
        if self.state is None:
            return 0.0
        start = float(self.config.best_anchor_distill_lambda_start)
        end = float(self.config.best_anchor_distill_lambda_end)
        ramp = max(1, int(self.config.best_anchor_distill_ramp_updates))
        progress = min(1.0, self.state.updates_since_reload / ramp)
        return start + (end - start) * progress

    def checkpoint_state(self) -> dict[str, Any] | None:
        return None if self.state is None else self.state.as_dict()

    def restore_checkpoint_state(self, raw: object) -> None:
        """恢复时确认教师仍是同一经验证锚点，拒绝悄然换成别的模型。"""
        if not isinstance(raw, dict):
            return
        expected_sha = str(raw.get("sha256", "")).strip().lower()
        if not expected_sha:
            return
        if self.state is None:
            self.refresh_from_run_best()
        if self.state is None or self.state.sha256 != expected_sha:
            actual = "<none>" if self.state is None else self.state.sha256
            raise RuntimeError(
                "恢复训练时 best-anchor 教师不一致；"
                f"expected={expected_sha} actual={actual}"
            )
        self.state = TeacherState(
            sha256=self.state.sha256,
            selection_score=float(raw.get("selection_score", self.state.selection_score)),
            source=str(raw.get("source", self.state.source)),
            version=int(raw.get("version", self.state.version)),
            updates_since_reload=int(
                raw.get("updates_since_reload", self.state.updates_since_reload)
            ),
        )
