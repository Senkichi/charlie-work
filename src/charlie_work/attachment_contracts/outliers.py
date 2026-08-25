"""Saturation: archetype-relative outlier test on bound-member counts.

Binding constraint: this module never reads a line count. The only measure is
member_count (bound-member count) per AttachmentPoint, tested against the
distribution of same-kind, non-ledger attachment points already in the repo.

No constant size threshold appears anywhere in this module. The boundary is
derived per-repo, per-kind, per-scan from Q3 + 1.5*IQR (the standard Tukey
outlier fence) over that kind's own population.
"""

from __future__ import annotations

import math

from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    Kind,
    SaturationVerdict,
)

# Statistical-validity floor, not a size threshold.
#
# Q1/Q3 (and therefore IQR and the Tukey fence) are not meaningfully defined
# below a handful of observations: with n < 4, nearest-rank quartiles degenerate
# (Q1 and Q3 collapse onto the same or adjacent order statistics, so IQR is ~0
# and the fence either falsely saturates everything above the median or forbids
# saturation entirely by accident of arithmetic, not by any property of the
# data). FLOOR=4 is the smallest population size at which nearest-rank Q1/Q3
# are computed from genuinely distinct order statistics for both quartiles.
# Below FLOOR, a kind's population is too small for an outlier test to mean
# anything, so nothing of that kind is ever saturated.
FLOOR = 4


def _nearest_rank_quartile(sorted_values: list[int], quartile: float) -> float:
    """Nearest-rank quartile: deterministic across platforms, no interpolation.

    Uses the ceiling-rank convention: rank = ceil(quartile * n), 1-indexed,
    clamped into [1, n]. This avoids the linear-interpolation ambiguity that
    makes different quartile methods disagree at small n.
    """
    n = len(sorted_values)
    rank = max(1, min(n, math.ceil(quartile * n)))
    return float(sorted_values[rank - 1])


def _quartiles(counts: list[int]) -> tuple[float, float]:
    """Return (q1, q3) via deterministic nearest-rank quartiles."""
    ordered = sorted(counts)
    q1 = _nearest_rank_quartile(ordered, 0.25)
    q3 = _nearest_rank_quartile(ordered, 0.75)
    return q1, q3


def saturate(points: tuple[AttachmentPoint, ...], kind: Kind) -> tuple[SaturationVerdict, ...]:
    """Compute saturation verdicts for every non-ledger AttachmentPoint of `kind`.

    Ledger attachment points (is_linear_ledger=True) are excluded from the
    distribution entirely (they never contribute to Q1/Q3) and are never
    saturated themselves — they get no verdict at all. Points of other kinds
    are not in this call's population; call once per kind.
    """
    same_kind = tuple(p for p in points if p.kind == kind)
    non_ledger = tuple(p for p in same_kind if not p.is_linear_ledger)
    counts = [p.member_count for p in non_ledger]
    population = len(counts)

    if population < FLOOR:
        return tuple(
            SaturationVerdict(
                point=p,
                saturated=False,
                q3=0.0,
                iqr=0.0,
                boundary=0.0,
                population=population,
            )
            for p in non_ledger
        )

    q1, q3 = _quartiles(counts)
    iqr = q3 - q1
    boundary = q3 + 1.5 * iqr

    return tuple(
        SaturationVerdict(
            point=p,
            saturated=p.member_count > boundary,
            q3=q3,
            iqr=iqr,
            boundary=boundary,
            population=population,
        )
        for p in non_ledger
    )


def saturate_all(
    points: tuple[AttachmentPoint, ...], kinds: tuple[Kind, ...]
) -> tuple[SaturationVerdict, ...]:
    """Run saturate() across every kind present, concatenating verdicts.

    `kinds` should be the caller's derived set of kinds actually seen in
    `points` (e.g. sorted(set(p.kind for p in points))) — this function does
    not hardcode or otherwise enumerate the Kind literal's members.
    """
    verdicts: list[SaturationVerdict] = []
    for kind in kinds:
        verdicts.extend(saturate(points, kind))
    return tuple(verdicts)
