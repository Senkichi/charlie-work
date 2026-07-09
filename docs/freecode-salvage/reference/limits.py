"""Planner limits loader (PLN-02b).

Single source of truth for static capacity seed values. Runtime quota
observations override seed values by exact 4-tuple key
`(provider_id, model_id, dimension, window_semantics)` (D-12). The actual
merge math lives in `freecode.planner.capacity.seed_capacity_from_quota_snapshot`
(D-17 pure); this module is the bridge between the YAML/Pydantic surface and
the capacity arithmetic.

This module MUST NOT import from `freecode.state.*` — enforced by
`tests/test_planner_limits.py::test_limits_does_not_import_state`.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from freecode.planner.capacity import (
    QuotaSnapshot,
    seed_capacity_from_quota_snapshot,
)
from freecode.planner.models import CapacityModel

ConfidenceLevel = Literal["low", "medium", "high"]
Dimension = Literal["requests", "tokens"]
WindowSemantics = Literal["per_minute", "per_hour", "per_day"]


class PlannerWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dimension: Dimension
    window_semantics: WindowSemantics
    window_seconds: int = Field(ge=1)
    limit: float = Field(ge=0.0)


class PlannerLimitSeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    context_window_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    capability_tags: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    citation: str = Field(min_length=1)
    windows: list[PlannerWindow]

    @model_validator(mode="after")
    def _unique_window_keys(self) -> PlannerLimitSeed:
        if not self.windows:
            raise PydanticCustomError(
                "planner.limits.empty_windows",
                f"{self.provider_id}/{self.model_id} must declare at least one window",
            )
        seen: set[tuple[str, str]] = set()
        for w in self.windows:
            key = (w.dimension, w.window_semantics)
            if key in seen:
                raise PydanticCustomError(
                    "planner.limits.duplicate_window",
                    f"{self.provider_id}/{self.model_id} has duplicate window {key}",
                )
            seen.add(key)
        return self


class PlannerLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = Field(ge=1, le=1)
    models: list[PlannerLimitSeed]


def load_planner_limit_seeds() -> PlannerLimits:
    """Load and validate the bundled `freecode/planner/limits.yaml`."""
    root = resources.files("freecode.planner")
    path = root.joinpath("limits.yaml")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            "Invalid limits.yaml: root must be a mapping with `version` and `models` keys"
        )
    return PlannerLimits.model_validate(raw)


def seeds_to_capacity_models(
    limits: PlannerLimits, *, now_iso: str
) -> list[CapacityModel]:
    """Flatten PlannerLimits -> CapacityModel rows, one per (seed x window)."""
    out: list[CapacityModel] = []
    for seed in limits.models:
        for w in seed.windows:
            out.append(
                CapacityModel(
                    provider_id=seed.provider_id,
                    model_id=seed.model_id,
                    dimension=w.dimension,
                    window_semantics=w.window_semantics,
                    window_seconds=w.window_seconds,
                    limit_value=w.limit,
                    safety_margin=0.0,
                    confidence=seed.confidence,
                    source="seed",
                    citation=seed.citation,
                    last_adjustment_reason=None,
                    updated_at=now_iso,
                )
            )
    return out


def merge_runtime_observations(
    seeds: Iterable[CapacityModel],
    snapshots: Iterable[QuotaSnapshot],
    *,
    now_iso: str,
) -> list[CapacityModel]:
    """Override matching 4-tuple keys with runtime observations (D-12)."""
    return seed_capacity_from_quota_snapshot(snapshots, seeds, now_iso=now_iso)
