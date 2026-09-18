"""Hand-computed quartile/boundary tests for outliers.saturate.

All expected values below are computed by hand in the docstrings/comments —
no test re-derives its expectation from the module under test.
"""

from __future__ import annotations

from charlie_work.attachment_contracts.model import AttachmentPoint, KindStats
from charlie_work.attachment_contracts.outliers import (
    FLOOR,
    saturate,
    saturate_all,
    saturate_all_with_fences,
    saturate_with_fence,
)


def _point(
    identity: str,
    count: int,
    kind: str = "class",
    ledger: bool = False,
    trivial: bool = False,
) -> AttachmentPoint:
    return AttachmentPoint(
        kind=kind,  # type: ignore[arg-type]
        identity=identity,
        file=f"src/{identity}.py",
        members=tuple(f"m{i}" for i in range(count)),
        is_linear_ledger=ledger,
        is_structurally_trivial=trivial,
    )


def test_floor_is_four() -> None:
    assert FLOOR == 4


def test_below_floor_nothing_saturated() -> None:
    # n=3 < FLOOR: outlier test is statistically meaningless, so nothing
    # saturates regardless of how skewed the values are.
    points = (_point("a", 1), _point("b", 2), _point("c", 100))
    verdicts = saturate(points, "class")
    assert len(verdicts) == 3
    assert all(v.saturated is False for v in verdicts)
    assert all(v.population == 3 for v in verdicts)
    assert all(v.boundary == 0.0 and v.q3 == 0.0 and v.iqr == 0.0 for v in verdicts)


def test_at_floor_saturates_extreme_outlier() -> None:
    # counts sorted = [1, 2, 3, 10], n=4 (== FLOOR, exactly meets it).
    # q1 rank = ceil(0.25*4) = 1 -> sorted[0] = 1
    # q3 rank = ceil(0.75*4) = 3 -> sorted[2] = 3
    # iqr = 3 - 1 = 2; boundary = 3 + 1.5*2 = 6
    # 10 > 6 -> saturated; 1, 2, 3 are all <= 6 -> not saturated.
    points = (_point("a", 1), _point("b", 2), _point("c", 3), _point("d", 10))
    verdicts = {v.point.identity: v for v in saturate(points, "class")}
    assert verdicts["a"].saturated is False
    assert verdicts["b"].saturated is False
    assert verdicts["c"].saturated is False
    assert verdicts["d"].saturated is True
    assert verdicts["d"].q3 == 3.0
    assert verdicts["d"].iqr == 2.0
    assert verdicts["d"].boundary == 6.0
    assert verdicts["d"].population == 4


def test_exact_tie_at_boundary_is_not_saturated() -> None:
    # counts sorted = [2, 5, 6, 12], n=4.
    # q1 rank = 1 -> sorted[0] = 2
    # q3 rank = 3 -> sorted[2] = 6
    # iqr = 4; boundary = 6 + 1.5*4 = 12
    # The point with member_count == 12 sits EXACTLY on the boundary.
    # saturated(p) uses strict '>' so an exact tie must NOT saturate.
    points = (_point("a", 2), _point("b", 5), _point("c", 6), _point("d", 12))
    verdicts = {v.point.identity: v for v in saturate(points, "class")}
    assert verdicts["d"].boundary == 12.0
    assert verdicts["d"].point.member_count == 12
    assert verdicts["d"].saturated is False, "exact tie at boundary must not saturate"


def test_five_points_one_extreme_outlier() -> None:
    # counts sorted = [1, 2, 3, 4, 20], n=5.
    # q1 rank = ceil(1.25) = 2 -> sorted[1] = 2
    # q3 rank = ceil(3.75) = 4 -> sorted[3] = 4
    # iqr = 2; boundary = 4 + 3 = 7
    # 20 > 7 -> saturated; all others <= 7 -> not saturated.
    points = (
        _point("a", 1),
        _point("b", 2),
        _point("c", 3),
        _point("d", 4),
        _point("e", 20),
    )
    verdicts = {v.point.identity: v for v in saturate(points, "class")}
    assert verdicts["e"].saturated is True
    assert verdicts["e"].boundary == 7.0
    for ident in ("a", "b", "c", "d"):
        assert verdicts[ident].saturated is False


def test_ledger_points_excluded_from_distribution_and_never_saturated() -> None:
    # The ledger AP has a huge member_count (50) but must not enter the
    # distribution (so it cannot skew q1/q3 for the real population) and
    # must never itself be saturated -- it gets no verdict at all.
    non_ledger = (_point("a", 1), _point("b", 2), _point("c", 3), _point("d", 4))
    ledger = _point("migration_ledger", 50, ledger=True)
    points = non_ledger + (ledger,)

    verdicts = saturate(points, "class")
    identities = {v.point.identity for v in verdicts}
    assert "migration_ledger" not in identities
    assert len(verdicts) == 4

    # Population must reflect only the 4 non-ledger points, not 5.
    assert all(v.population == 4 for v in verdicts)

    # Sanity: without the ledger's 50 polluting q3, boundary stays small.
    # sorted=[1,2,3,4]; q1 rank=1->1; q3 rank=3->3; iqr=2; boundary=3+3=6.
    for v in verdicts:
        assert v.boundary == 6.0


def test_structurally_trivial_points_excluded_from_distribution_and_never_saturated() -> None:
    # Round-2 review finding #9 (regression): single-member trivial classes
    # (Protocols, Fake* doubles, etc.) must not enter the population -- they
    # get no verdict at all, the same treatment as ledgers.
    real = (_point("a", 1), _point("b", 2), _point("c", 3), _point("d", 10))
    trivial = tuple(_point(f"trivial{i}", 1, trivial=True) for i in range(20))
    points = real + trivial

    verdicts = saturate(points, "class")
    identities = {v.point.identity for v in verdicts}

    assert identities == {"a", "b", "c", "d"}
    assert all(v.population == 4 for v in verdicts)
    verdict_by_id = {v.point.identity: v for v in verdicts}
    assert verdict_by_id["d"].boundary == 6.0
    assert verdict_by_id["d"].saturated is True


def test_saturate_only_considers_matching_kind() -> None:
    class_points = (_point("a", 1, kind="class"), _point("b", 2, kind="class"))
    other_points = (_point("c", 100, kind="typer_app"),)
    verdicts = saturate(class_points + other_points, "class")
    assert {v.point.identity for v in verdicts} == {"a", "b"}


def test_zero_member_points_excluded_from_population_and_get_no_verdict() -> None:
    # Round-2 review finding #9: 148/531 real class APs are zero-member
    # (Protocols, `class Foo: pass`, empty dataclasses); they anchor Q1 at 0
    # and collapse the IQR, dragging the boundary down onto legitimate
    # multi-method classes. A zero-member AP is not evidence about the
    # god-object distribution and must not enter the population or receive a
    # verdict of its own (member_count=0 could never be saturated anyway).
    zero_member = tuple(_point(f"empty{i}", 0) for i in range(5))
    real = (_point("a", 1), _point("b", 2), _point("c", 3), _point("d", 10))
    points = zero_member + real

    verdicts = saturate(points, "class")
    identities = {v.point.identity for v in verdicts}

    assert identities == {"a", "b", "c", "d"}
    assert all(v.population == 4 for v in verdicts)
    # Same boundary as test_at_floor_saturates_extreme_outlier: the 5
    # zero-member points must not have polluted q1/q3.
    verdict_by_id = {v.point.identity: v for v in verdicts}
    assert verdict_by_id["d"].boundary == 6.0
    assert verdict_by_id["d"].saturated is True


def test_iqr_zero_degenerate_fence_never_saturates() -> None:
    # Round-2 review finding #9: with zero spread (q1 == q3), Tukey's fence
    # collapses to a strict `> Q3` test with no tolerance at all. A uniform
    # population of 4-member points plus one 5-member point would otherwise
    # "saturate" on pure arithmetic, not on being a real outlier.
    points = (
        _point("a", 4),
        _point("b", 4),
        _point("c", 4),
        _point("d", 4),
        _point("e", 5),
    )
    verdicts = {v.point.identity: v for v in saturate(points, "class")}
    assert all(v.saturated is False for v in verdicts.values())
    assert verdicts["e"].iqr == 0.0
    assert verdicts["e"].q3 == 4.0


def test_saturate_all_concatenates_across_kinds() -> None:
    points = (
        _point("a1", 1, kind="class"),
        _point("a2", 2, kind="class"),
        _point("a3", 3, kind="class"),
        _point("a4", 30, kind="class"),
        _point("b1", 1, kind="typer_app"),
        _point("b2", 2, kind="typer_app"),
        _point("b3", 3, kind="typer_app"),
        _point("b4", 30, kind="typer_app"),
    )
    verdicts = saturate_all(points, ("class", "typer_app"))
    saturated_idents = {v.point.identity for v in verdicts if v.saturated}
    assert saturated_idents == {"a4", "b4"}
    assert len(verdicts) == 8


# ---------------------------------------------------------------------------
# saturate_with_fence / saturate_all_with_fences (issue #1614): frozen fence
# ---------------------------------------------------------------------------


def test_saturate_with_fence_uses_frozen_boundary_not_live_population() -> None:
    # Issue #1614 positive control: a population whose LIVE fence would move
    # when a median point is added, but the FROZEN fence must not.
    #
    # Freeze-time population (n=8): counts sorted = [1,2,3,4,6,8,16,20].
    #   q1 rank = ceil(0.25*8) = 2 -> sorted[1] = 2
    #   q3 rank = ceil(0.75*8) = 6 -> sorted[5] = 8
    #   iqr = 6; boundary = 8 + 1.5*6 = 17
    #   P (16) is NOT saturated (16 > 17 is False); big (20) IS saturated.
    freeze_points = (
        _point("a", 1),
        _point("b", 2),
        _point("c", 3),
        _point("d", 4),
        _point("e", 6),
        _point("f", 8),
        _point("P", 16),
        _point("big", 20),
    )
    live_verdicts = {v.point.identity: v for v in saturate(freeze_points, "class")}
    assert live_verdicts["big"].saturated is True
    assert live_verdicts["P"].saturated is False
    assert live_verdicts["P"].boundary == 17.0

    frozen = KindStats(kind="class", q3=8.0, iqr=6.0, boundary=17.0, population=8)

    # After adding one median-sized module (count=8, at the old q3), the LIVE
    # fence drops: n=9, sorted=[1,2,3,4,6,8,8,16,20],
    #   q1 rank = ceil(2.25) = 3 -> sorted[2] = 3
    #   q3 rank = ceil(6.75) = 7 -> sorted[6] = 8
    #   iqr = 5; boundary = 8 + 7.5 = 15.5  ->  P (16) IS saturated live.
    augmented = freeze_points + (_point("median", 8),)
    live_aug = {v.point.identity: v for v in saturate(augmented, "class")}
    assert live_aug["P"].saturated is True, "positive control: live fence moved and saturated P"
    assert live_aug["P"].boundary == 15.5

    # The FROZEN fence must NOT move: P stays not-saturated against boundary=17.
    frozen_verdicts = {
        v.point.identity: v for v in saturate_with_fence(augmented, "class", frozen)
    }
    assert frozen_verdicts["P"].saturated is False
    assert frozen_verdicts["P"].boundary == 17.0
    assert frozen_verdicts["P"].population == 8  # frozen population, not live 9
    assert frozen_verdicts["big"].saturated is True
    # The newly-added median module is tested against the frozen fence too.
    assert frozen_verdicts["median"].saturated is False


def test_saturate_with_fence_reproduces_floor_guard_against_frozen_population() -> None:
    # A kind that was below FLOOR at freeze time saturates nothing against the
    # frozen fence, even once the live population has grown past FLOOR -- the
    # frozen decision was "no meaningful fence", and only a refreeze revisits
    # it. (population=3 < FLOOR=4 -> boundary=0, nothing saturated.)
    frozen = KindStats(kind="class", q3=0.0, iqr=0.0, boundary=0.0, population=3)
    points = (
        _point("a", 1),
        _point("b", 2),
        _point("c", 3),
        _point("d", 4),
        _point("big", 50),
    )
    verdicts = saturate_with_fence(points, "class", frozen)
    # boundary=0.0 would naively saturate everything with member_count >= 1
    # via `> 0.0`; the frozen-population FLOOR guard must prevent that.
    assert all(v.saturated is False for v in verdicts)
    assert all(v.boundary == 0.0 for v in verdicts)


def test_saturate_with_fence_reproduces_degenerate_iqr_guard() -> None:
    # A kind that was degenerate (iqr==0) at freeze time saturates nothing
    # against the frozen fence, even if a live point now exceeds the frozen
    # boundary -- the frozen decision was "no spread, no tolerance".
    frozen = KindStats(kind="class", q3=4.0, iqr=0.0, boundary=4.0, population=4)
    points = (
        _point("a", 4),
        _point("b", 4),
        _point("c", 4),
        _point("d", 4),
        _point("e", 5),  # 5 > frozen boundary 4, but degenerate guard blocks it
    )
    verdicts = {v.point.identity: v for v in saturate_with_fence(points, "class", frozen)}
    assert all(v.saturated is False for v in verdicts.values())
    assert verdicts["e"].boundary == 4.0


def test_saturate_with_fence_excludes_ledger_and_trivial_like_saturate() -> None:
    # Eligibility is shared with saturate (_eligible_points): ledger and
    # structurally-trivial points get no verdict under the frozen fence too.
    frozen = KindStats(kind="class", q3=3.0, iqr=2.0, boundary=6.0, population=4)
    real = (_point("a", 1), _point("b", 2), _point("c", 3), _point("d", 10))
    ledger = _point("ledger", 50, ledger=True)
    trivial = _point("trivial", 1, trivial=True)
    verdicts = {
        v.point.identity: v for v in saturate_with_fence(real + (ledger, trivial), "class", frozen)
    }
    assert "ledger" not in verdicts
    assert "trivial" not in verdicts
    assert verdicts["d"].saturated is True  # 10 > 6


def test_saturate_all_with_fences_only_yields_verdicts_for_frozen_kinds() -> None:
    # A kind present in the live tree but absent from kind_stats (a brand-new
    # archetype since the last freeze) gets no verdict -- it has no frozen
    # target to test against and cannot produce a block finding until a
    # refreeze records its fence.
    fences = {"class": KindStats(kind="class", q3=3.0, iqr=2.0, boundary=6.0, population=4)}
    points = (
        _point("a", 1, kind="class"),
        _point("b", 2, kind="class"),
        _point("c", 3, kind="class"),
        _point("d", 10, kind="class"),
        _point("newkind", 100, kind="typer_app"),
    )
    verdicts = saturate_all_with_fences(points, fences)
    idents = {v.point.identity for v in verdicts}
    assert "newkind" not in idents
    assert {v.point.kind for v in verdicts} == {"class"}
