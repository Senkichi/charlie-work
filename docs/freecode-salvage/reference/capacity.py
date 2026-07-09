"""Pure capacity arithmetic for the provider utilization planner.

This module MUST NOT perform any I/O. In particular:
  * No imports from `freecode.state.*` (enforced by `tests/test_planner_capacity_choke_guard.py`).
  * No `open()`, `pathlib.Path.read_*`, or HTTP calls.
  * No mutation — `tighten_after_429` returns a new frozen `CapacityModel` via
    `dataclasses.replace`, never mutating its input (D-13/D-15).

Decision references: D-13 (safety_margin only; limit_value preserved), D-14
(0.05 step, 0.5 cap), D-15 (replace + reason + updated_at), D-16 (calendar-
aligned windows), D-17 (zero DB writes).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from freecode.planner.models import CapacityModel

SAFETY_MARGIN_STEP: float = 0.05
SAFETY_MARGIN_CAP: float = 0.5


@dataclass(frozen=True)
class QuotaSnapshot:
    """Single observation of a provider's runtime quota (used by seed merge)."""

    provider_id: str
    model_id: str
    dimension: str
    window_semantics: str
    window_seconds: int
    limit_value: float
    confidence: str
    citation: str
    observed_at: str


def usable_limit(limit_value: float, safety_margin: float) -> float:
    """Effective limit after safety-margin headroom is reserved."""
    if safety_margin < 0.0 or safety_margin > SAFETY_MARGIN_CAP:
        raise ValueError(
            f"safety_margin {safety_margin} outside [0.0, {SAFETY_MARGIN_CAP}]"
        )
    return limit_value * (1.0 - safety_margin)


def tighten_after_429(
    model: CapacityModel, *, reason: str, now_iso: str
) -> CapacityModel:
    """Bump safety_margin by SAFETY_MARGIN_STEP, hard-capped at SAFETY_MARGIN_CAP.

    Per D-13, `limit_value` is the prose audit anchor and is NEVER rewritten —
    the call to `replace` deliberately omits `limit_value` from its kwargs.
    """
    new_margin = min(model.safety_margin + SAFETY_MARGIN_STEP, SAFETY_MARGIN_CAP)
    return replace(
        model,
        safety_margin=new_margin,
        last_adjustment_reason=reason,
        updated_at=now_iso,
    )


def window_for_timestamp(
    now: datetime, window_semantics: str, window_seconds: int
) -> tuple[str, str]:
    """Return (start_iso, end_iso) for the calendar-aligned window containing `now`.

    `window_seconds` is accepted for parity with `CapacityModel.window_seconds`
    but is not used to compute the boundaries — alignment is by calendar unit (D-16).
    """
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("now must be UTC-aware (tzinfo with offset=0)")
    # window_seconds is documented; deliberately not validated against semantics here.
    del window_seconds

    if window_semantics == "per_day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif window_semantics == "per_hour":
        start = now.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
    elif window_semantics == "per_minute":
        start = now.replace(second=0, microsecond=0)
        end = start + timedelta(minutes=1)
    else:
        raise ValueError(f"unsupported window_semantics: {window_semantics!r}")
    return (
        start.replace(microsecond=0).isoformat(),
        end.replace(microsecond=0).isoformat(),
    )


def seed_capacity_from_quota_snapshot(
    snapshots: Iterable[QuotaSnapshot],
    existing_seeds: Iterable[CapacityModel],
    *,
    now_iso: str,
) -> list[CapacityModel]:
    """Merge runtime observations into seed list by 4-tuple key (D-12).

    Order: existing seed order first (with overrides applied in place); then any
    observation rows whose key was not in the seed list, appended in input order.
    """
    seeds = list(existing_seeds)
    snap_list = list(snapshots)
    snap_by_key: dict[tuple[str, str, str, str], QuotaSnapshot] = {}
    for snap in snap_list:
        key = (snap.provider_id, snap.model_id, snap.dimension, snap.window_semantics)
        snap_by_key[key] = snap  # later snapshot wins on duplicate keys in input
    consumed: set[tuple[str, str, str, str]] = set()
    out: list[CapacityModel] = []
    for seed in seeds:
        key = (seed.provider_id, seed.model_id, seed.dimension, seed.window_semantics)
        if key in snap_by_key:
            snap = snap_by_key[key]
            out.append(
                replace(
                    seed,
                    limit_value=snap.limit_value,
                    confidence=snap.confidence,
                    citation=snap.citation,
                    source="runtime",
                    updated_at=now_iso,
                )
            )
            consumed.add(key)
        else:
            out.append(seed)
    # Append observation-only rows preserving snapshot input order.
    for snap in snap_list:
        key = (snap.provider_id, snap.model_id, snap.dimension, snap.window_semantics)
        if key in consumed:
            continue
        consumed.add(key)
        out.append(
            CapacityModel(
                provider_id=snap.provider_id,
                model_id=snap.model_id,
                dimension=snap.dimension,
                window_semantics=snap.window_semantics,
                window_seconds=snap.window_seconds,
                limit_value=snap.limit_value,
                safety_margin=0.0,
                confidence=snap.confidence,
                source="runtime",
                citation=snap.citation,
                last_adjustment_reason=None,
                updated_at=now_iso,
            )
        )
    return out
