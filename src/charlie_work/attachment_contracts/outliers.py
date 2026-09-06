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
from typing import Mapping

from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    Kind,
    KindStats,
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


def _eligible_points(
    points: tuple[AttachmentPoint, ...], kind: Kind
) -> tuple[AttachmentPoint, ...]:
    """Same-kind, non-ledger, non-trivial, >=1-member attachment points.

    Shared by ``saturate`` (live fence) and ``saturate_with_fence`` (frozen
    fence): eligibility is a property of the points, not of which fence is
    applied, so the two paths must filter identically (issue #1614).
    """
    same_kind = tuple(p for p in points if p.kind == kind)
    # An AP with zero bound members is not evidence about the god-object
    # distribution for its kind (round-2 review finding #9): a population
    # padded with empty classes/Protocols/etc. drags Q1 down to 0, collapsing
    # the IQR and pulling the Tukey fence in on everything else. Such points
    # never carry a verdict of their own -- they cannot be evidence of
    # saturation either (member_count=0 can never exceed a boundary >= 0).
    #
    # Round-2 review finding #9 (regression): filtering zero-member points
    # alone was not enough -- 220/383 non-zero `class` APs are SINGLE-member
    # (Protocols, one-method dataclasses, Fake*/Test* doubles), and removing
    # only the zero tail compressed Q1 upward onto that single-member mass,
    # pulling the fence DOWN (5.0 -> 3.5) onto legitimate multi-method
    # classes. The real defect is that `class` is not one archetype:
    # structurally-trivial classes (Protocol bases, Exception subclasses,
    # empty @dataclass shells, Fake*/Test* doubles -- see archetypes.py) are
    # excluded from the population the same way ledgers are, regardless of
    # their member count.
    return tuple(
        p
        for p in same_kind
        if not p.is_linear_ledger and not p.is_structurally_trivial and p.member_count >= 1
    )


def saturate(points: tuple[AttachmentPoint, ...], kind: Kind) -> tuple[SaturationVerdict, ...]:
    """Compute saturation verdicts for every non-ledger AttachmentPoint of `kind`.

    Ledger attachment points (is_linear_ledger=True) are excluded from the
    distribution entirely (they never contribute to Q1/Q3) and are never
    saturated themselves — they get no verdict at all. Points of other kinds
    are not in this call's population; call once per kind.

    This is the LIVE fence: ``boundary = q3 + 1.5*iqr`` is derived from the
    CURRENT population. ``check_tree`` and ``compare`` use
    ``saturate_with_fence`` against a frozen ``KindStats`` instead (issue
    #1614); this function remains the recomputation path used by
    ``generate``, ``--refreeze``, ``backtest``, and ``redirect.suggest``.
    """
    eligible = _eligible_points(points, kind)
    counts = [p.member_count for p in eligible]
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
            for p in eligible
        )

    q1, q3 = _quartiles(counts)
    iqr = q3 - q1
    boundary = q3 + 1.5 * iqr

    if iqr == 0.0:
        # Degenerate fence (finding #9): with zero spread in the population,
        # Tukey's rule collapses to a strict `> Q3` test with no statistical
        # tolerance at all -- any point one member above a totally uniform
        # population would "saturate" on arithmetic alone, not on being an
        # actual outlier. Report the real q3/boundary for visibility but
        # treat nothing as saturated until the population actually spreads.
        return tuple(
            SaturationVerdict(
                point=p,
                saturated=False,
                q3=q3,
                iqr=0.0,
                boundary=boundary,
                population=population,
            )
            for p in eligible
        )

    return tuple(
        SaturationVerdict(
            point=p,
            saturated=p.member_count > boundary,
            q3=q3,
            iqr=iqr,
            boundary=boundary,
            population=population,
        )
        for p in eligible
    )


def saturate_with_fence(
    points: tuple[AttachmentPoint, ...], kind: Kind, stats: KindStats
) -> tuple[SaturationVerdict, ...]:
    """Saturate live points of ``kind`` against a FROZEN ``KindStats`` (issue #1614).

    Eligibility is filtered exactly as in ``saturate`` (via ``_eligible_points``)
    so the frozen and live paths cannot drift on which points enter the test.
    The fence itself is NOT recomputed: every verdict carries the frozen
    ``q3``/``iqr``/``boundary``/``population``. The FLOOR and degenerate-fence
    (iqr==0) guards are reproduced against the FROZEN decision (``stats``
    captures the population/iqr at freeze time), so a kind that was below FLOOR
    or degenerate at freeze time saturates nothing until an explicit
    ``--refreeze`` -- even if the live population has since grown past FLOOR.
    That is the intended freeze semantics: the exit criterion for shrinking a
    saturated class is a frozen target, not a moving one.
    """
    eligible = _eligible_points(points, kind)

    if stats.population < FLOOR or stats.iqr == 0.0:
        # Frozen decision was "no meaningful fence" -- nothing of this kind
        # saturates regardless of how the live population has shifted, until
        # an explicit re-baseline / --refreeze recomputes the statistics.
        return tuple(
            SaturationVerdict(
                point=p,
                saturated=False,
                q3=stats.q3,
                iqr=stats.iqr,
                boundary=stats.boundary,
                population=stats.population,
            )
            for p in eligible
        )

    return tuple(
        SaturationVerdict(
            point=p,
            saturated=p.member_count > stats.boundary,
            q3=stats.q3,
            iqr=stats.iqr,
            boundary=stats.boundary,
            population=stats.population,
        )
        for p in eligible
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


def saturate_all_with_fences(
    points: tuple[AttachmentPoint, ...], kind_stats: Mapping[Kind, KindStats]
) -> tuple[SaturationVerdict, ...]:
    """Run ``saturate_with_fence`` across every frozen kind, concatenating verdicts.

    Only kinds present in ``kind_stats`` receive verdicts -- a kind that
    appeared in the live tree but has no frozen statistics (e.g. a brand-new
    archetype since the last freeze) gets no verdict at all, so it cannot
    produce a block finding until an explicit re-baseline / ``--refreeze``
    records its fence. That is the intended freeze semantics (issue #1614):
    the fence is a frozen target, and a kind with no frozen target has no
    target to test against. ``kind_stats`` is iterated in sorted-key order
    for deterministic output.
    """
    verdicts: list[SaturationVerdict] = []
    for kind in sorted(kind_stats):
        verdicts.extend(saturate_with_fence(points, kind, kind_stats[kind]))
    return tuple(verdicts)
