from __future__ import annotations

from dataclasses import dataclass, field

import torch

from models.worker_pointer_context import WorkerPressureContext


@dataclass
class WorkerPointerV2Diagnostics:
    """按 rollout 有界收集 v2 诊断；只在 finalize 时把张量搬到 CPU。"""

    num_skills: int = 5
    _pressure_all: list[torch.Tensor] = field(default_factory=list)
    _pressure_near: list[torch.Tensor] = field(default_factory=list)
    _demand_all: list[torch.Tensor] = field(default_factory=list)
    _demand_near: list[torch.Tensor] = field(default_factory=list)
    _supply_all: list[torch.Tensor] = field(default_factory=list)
    _supply_near: list[torch.Tensor] = field(default_factory=list)
    _zero_supply_all: list[torch.Tensor] = field(default_factory=list)
    _zero_supply_near: list[torch.Tensor] = field(default_factory=list)
    _selected_exposures: list[torch.Tensor] = field(default_factory=list)
    _entropies: list[torch.Tensor] = field(default_factory=list)
    _team_consumptions: list[torch.Tensor] = field(default_factory=list)
    _ready_task_counts: list[torch.Tensor] = field(default_factory=list)
    _legal_station_counts: list[torch.Tensor] = field(default_factory=list)
    _legal_worker_counts: list[torch.Tensor] = field(default_factory=list)
    _selected_eft_rank_percentiles: list[torch.Tensor] = field(default_factory=list)
    _legal_relative_eft: list[torch.Tensor] = field(default_factory=list)
    _legal_zscore_eft: list[torch.Tensor] = field(default_factory=list)
    _host_elapsed_ms: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self._pressure_all.clear()
        self._pressure_near.clear()
        self._demand_all.clear()
        self._demand_near.clear()
        self._supply_all.clear()
        self._supply_near.clear()
        self._zero_supply_all.clear()
        self._zero_supply_near.clear()
        self._selected_exposures.clear()
        self._entropies.clear()
        self._team_consumptions.clear()
        self._ready_task_counts.clear()
        self._legal_station_counts.clear()
        self._legal_worker_counts.clear()
        self._selected_eft_rank_percentiles.clear()
        self._legal_relative_eft.clear()
        self._legal_zscore_eft.clear()
        self._host_elapsed_ms.clear()

    @property
    def buffered_element_count(self) -> int:
        tensors: list[torch.Tensor] = [
            *self._pressure_all,
            *self._pressure_near,
            *self._demand_all,
            *self._demand_near,
            *self._supply_all,
            *self._supply_near,
            *self._zero_supply_all,
            *self._zero_supply_near,
        ]
        tensors.extend(self._selected_exposures)
        tensors.extend(self._entropies)
        tensors.extend(self._team_consumptions)
        tensors.extend(self._ready_task_counts)
        tensors.extend(self._legal_station_counts)
        tensors.extend(self._legal_worker_counts)
        tensors.extend(self._selected_eft_rank_percentiles)
        tensors.extend(self._legal_relative_eft)
        tensors.extend(self._legal_zscore_eft)
        return sum(tensor.numel() for tensor in tensors)

    def record_context(
        self,
        context: WorkerPressureContext,
        *,
        host_elapsed_ms: float,
    ) -> None:
        self._pressure_all.append(context.pressure_all.detach())
        self._pressure_near.append(context.pressure_near.detach())
        self._demand_all.append(context.demand_all.detach())
        self._demand_near.append(context.demand_near.detach())
        self._supply_all.append(context.supply_all.detach())
        self._supply_near.append(context.supply_near.detach())
        self._zero_supply_all.append(context.zero_supply_all.detach())
        self._zero_supply_near.append(context.zero_supply_near.detach())
        self._host_elapsed_ms.append(float(host_elapsed_ms))

    def record_selection(
        self,
        *,
        selected_exposure: torch.Tensor,
        entropy: torch.Tensor,
        dynamic_eft_features: torch.Tensor | None = None,
        selected_worker_index: int | torch.Tensor | None = None,
        worker_invalid_mask: torch.Tensor | None = None,
    ) -> None:
        assert selected_exposure.shape[-1] == 12
        self._selected_exposures.append(selected_exposure.detach())
        self._entropies.append(entropy.detach().reshape(-1))
        if dynamic_eft_features is None:
            return
        assert selected_worker_index is not None and worker_invalid_mask is not None
        assert dynamic_eft_features.ndim == 3 and dynamic_eft_features.shape[-1] == 2
        batch_size, num_workers, _ = dynamic_eft_features.shape
        assert worker_invalid_mask.shape == (batch_size, num_workers)
        selected = torch.as_tensor(
            selected_worker_index,
            device=dynamic_eft_features.device,
            dtype=torch.long,
        ).reshape(-1)
        if selected.numel() == 1:
            selected = selected.expand(batch_size)
        assert selected.shape == (batch_size,)
        assert torch.all((selected >= 0) & (selected < num_workers))
        batch_indices = torch.arange(batch_size, device=dynamic_eft_features.device)
        assert not bool(worker_invalid_mask[batch_indices, selected].any())
        relative_eft = dynamic_eft_features[:, :, 0]
        selected_eft = relative_eft[batch_indices, selected].unsqueeze(1)
        legal = ~worker_invalid_mask
        lower = ((relative_eft < selected_eft) & legal).sum(dim=1).float()
        equal = ((relative_eft == selected_eft) & legal).sum(dim=1).float()
        # 并列候选采用平均秩，单一合法候选的排名固定为 0。
        rank = lower + 0.5 * (equal - 1.0)
        legal_count = legal.sum(dim=1).float()
        percentile = rank / (legal_count - 1.0).clamp_min(1.0)
        self._selected_eft_rank_percentiles.append(percentile.detach())
        self._legal_relative_eft.append(relative_eft[legal].detach())
        self._legal_zscore_eft.append(dynamic_eft_features[:, :, 1][legal].detach())

    def record_action_space(
        self,
        *,
        ready_task_count: torch.Tensor | None = None,
        legal_station_count: torch.Tensor | None = None,
        legal_worker_count: torch.Tensor | None = None,
    ) -> None:
        """记录策略可见的候选规模；各阶段样本数可不同。"""
        for value, target in (
            (ready_task_count, self._ready_task_counts),
            (legal_station_count, self._legal_station_counts),
            (legal_worker_count, self._legal_worker_counts),
        ):
            if value is not None:
                assert value.numel() > 0
                target.append(value.detach().reshape(-1).float())

    def record_team(self, team_consumption: torch.Tensor) -> None:
        assert team_consumption.shape[-1] == self.num_skills
        self._team_consumptions.append(team_consumption.detach())

    @staticmethod
    def _cpu_cat(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values, dim=0).detach().float().cpu()

    @staticmethod
    def _summaries(prefix: str, values: torch.Tensor) -> dict[str, float]:
        assert values.ndim == 2
        metrics: dict[str, float] = {}
        quantiles = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
        for skill_index in range(values.shape[1]):
            column = values[:, skill_index]
            q50, q90, q99 = torch.quantile(column, quantiles)
            skill_prefix = f"{prefix}/Skill{skill_index}"
            metrics.update(
                {
                    f"{skill_prefix}/Count": float(column.numel()),
                    f"{skill_prefix}/Mean": float(column.mean()),
                    f"{skill_prefix}/P50": float(q50),
                    f"{skill_prefix}/P90": float(q90),
                    f"{skill_prefix}/P99": float(q99),
                    f"{skill_prefix}/Max": float(column.max()),
                }
            )
        return metrics

    @staticmethod
    def _scalar_quantile_summaries(prefix: str, values: torch.Tensor) -> dict[str, float]:
        assert values.ndim == 1 and values.numel() > 0
        q50, q90, q99 = torch.quantile(
            values,
            torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32),
        )
        return {
            f"{prefix}/LegalMean": float(values.mean()),
            f"{prefix}/LegalP50": float(q50),
            f"{prefix}/LegalP90": float(q90),
            f"{prefix}/LegalP99": float(q99),
            f"{prefix}/LegalMax": float(values.max()),
        }

    def finalize(self, *, require_coverage: bool) -> dict[str, float]:
        if not self._pressure_all:
            if require_coverage:
                raise RuntimeError("WorkerPointer v2 首轮 rollout 未产生压力上下文")
            return {}

        pressure_all = self._cpu_cat(self._pressure_all)
        pressure_near = self._cpu_cat(self._pressure_near)
        demand_all = self._cpu_cat(self._demand_all)
        demand_near = self._cpu_cat(self._demand_near)
        supply_all = self._cpu_cat(self._supply_all)
        supply_near = self._cpu_cat(self._supply_near)
        zero_supply_all = self._cpu_cat(self._zero_supply_all).bool()
        zero_supply_near = self._cpu_cat(self._zero_supply_near).bool()
        finite_tensors = (pressure_all, pressure_near, demand_all, demand_near, supply_all, supply_near)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in finite_tensors):
            self.reset()
            raise RuntimeError("WorkerPointer v2 压力诊断出现 NaN 或 Inf")

        if require_coverage:
            missing_all = torch.nonzero((demand_all > 0).sum(dim=0) == 0).reshape(-1).tolist()
            missing_near = torch.nonzero((demand_near > 0).sum(dim=0) == 0).reshape(-1).tolist()
            zero_near = torch.nonzero(
                ((demand_near > 0) & zero_supply_near).any(dim=0)
            ).reshape(-1).tolist()
            zero_all = torch.nonzero(
                ((demand_all > 0) & zero_supply_all).any(dim=0)
            ).reshape(-1).tolist()
            if zero_near:
                self.reset()
                raise RuntimeError(f"近期需求存在零有效供给，技能索引={zero_near}")
            if zero_all:
                self.reset()
                raise RuntimeError(f"长期需求存在零有效供给，技能索引={zero_all}")
            if missing_all or missing_near:
                self.reset()
                raise RuntimeError(
                    "WorkerPointer v2 压力覆盖不足："
                    f"长期缺失={missing_all}，近期缺失={missing_near}"
                )

        metrics = self._summaries("PointerV2/PressureAll", pressure_all)
        metrics.update(self._summaries("PointerV2/PressureNear", pressure_near))
        metrics["PointerV2/ContextHostMs"] = float(sum(self._host_elapsed_ms))
        if self._selected_exposures:
            exposure = self._cpu_cat(self._selected_exposures)
            metrics["PointerV2/SelectedExposureMean"] = float(exposure.mean())
            metrics["PointerV2/SelectedExposureMax"] = float(exposure.max())
        if self._entropies:
            entropy = self._cpu_cat([value.reshape(-1, 1) for value in self._entropies])
            metrics["PointerV2/WorkerEntropyMean"] = float(entropy.mean())
            metrics["PointerV2/TeamEntropy"] = float(entropy.sum())
        if self._team_consumptions:
            consumption = self._cpu_cat(self._team_consumptions)
            metrics["PointerV2/TeamConsumptionMean"] = float(consumption.mean())
            metrics["PointerV2/TeamConsumptionMax"] = float(consumption.max())
        if self._ready_task_counts:
            ready = self._cpu_cat([value.reshape(-1, 1) for value in self._ready_task_counts])
            metrics["PointerV2/ActionSpace/ReadyTaskMean"] = float(ready.mean())
        if self._legal_station_counts:
            station = self._cpu_cat([value.reshape(-1, 1) for value in self._legal_station_counts])
            metrics["PointerV2/ActionSpace/LegalStationMean"] = float(station.mean())
        if self._legal_worker_counts:
            worker = self._cpu_cat([value.reshape(-1, 1) for value in self._legal_worker_counts])
            metrics["PointerV2/ActionSpace/LegalWorkerMean"] = float(worker.mean())
            metrics["PointerV2/ActionSpace/LegalWorkerP10"] = float(
                torch.quantile(worker.reshape(-1), 0.1)
            )
        if self._selected_eft_rank_percentiles:
            eft_rank = self._cpu_cat(
                [value.reshape(-1, 1) for value in self._selected_eft_rank_percentiles]
            )
            metrics["PointerV2/EFT/SelectedRankPercentileMean"] = float(eft_rank.mean())
        if self._legal_relative_eft:
            metrics.update(
                self._scalar_quantile_summaries(
                    "RelativeEFT", self._cpu_cat(self._legal_relative_eft)
                )
            )
        if self._legal_zscore_eft:
            metrics.update(
                self._scalar_quantile_summaries(
                    "ZScoreEFT", self._cpu_cat(self._legal_zscore_eft)
                )
            )
        self.reset()
        return metrics


__all__ = ["WorkerPointerV2Diagnostics"]
