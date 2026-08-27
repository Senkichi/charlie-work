"""Hand-computed quartile/boundary tests for outliers.saturate.

All expected values below are computed by hand in the docstrings/comments —
no test re-derives its expectation from the module under test.
"""

from __future__ import annotations

from charlie_work.attachment_contracts.model import AttachmentPoint
from charlie_work.attachment_contracts.outliers import FLOOR, saturate, saturate_all


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
