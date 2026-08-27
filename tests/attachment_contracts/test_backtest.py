"""Tests for backtest.py's pure logic (select_samples, evaluate, criteria).

Round-1 review noted this module had zero unit test coverage, and that gap is
named as the reason findings #1 and #3 shipped -- these tests close it for
the pure (non-git) half. `run_backtest`, `load_commit_log`, and the worktree
orchestration remain exercised only via the real `backtest` CLI command
against an actual git checkout (see `docs/plans/attachment-contracts-
backtest-report.md`), not here.
"""

from __future__ import annotations

from charlie_work.attachment_contracts.backtest import (
    COUNTEREXAMPLE_MODULES,
    CommitRef,
    SampleResult,
    _criterion_counterexamples_clean,
    _cluster_b_informational,
    evaluate,
    select_samples,
)
from charlie_work.attachment_contracts.excludes import Excludes
from charlie_work.attachment_contracts.model import AttachmentPoint

# ---------------------------------------------------------------------------
# select_samples
# ---------------------------------------------------------------------------


def _ref(sha: str, date: str, changed_file_count: int = 1) -> CommitRef:
    return CommitRef(sha=sha, date=date, label="", changed_file_count=changed_file_count)


def test_select_samples_picks_first_commit_of_each_recent_month() -> None:
    commits = (
        _ref("aaa1", "2026-06-01"),
        _ref("aaa2", "2026-06-15"),
        _ref("bbb1", "2026-07-01"),
        _ref("bbb2", "2026-07-20"),
    )
    result = select_samples(commits, months=2, anchor_shas=())
    assert [r.sha for r in result] == ["aaa1", "bbb1"]
    assert [r.label for r in result] == ["2026-06", "2026-07"]


def test_select_samples_includes_anchors_outside_the_month_window() -> None:
    commits = (
        _ref("aaa1", "2026-01-01"),
        _ref("recent1", "2026-08-01"),
    )
    result = select_samples(commits, months=1, anchor_shas=("aaa1",))
    shas = {r.sha for r in result}
    assert "aaa1" in shas
    assert "recent1" in shas
    anchor_entry = next(r for r in result if r.sha == "aaa1")
    assert anchor_entry.label == "anchor"


def test_select_samples_deduplicates_by_sha() -> None:
    commits = (_ref("aaa1", "2026-06-01"),)
    result = select_samples(commits, months=1, anchor_shas=("aaa1",))
    assert len(result) == 1


def test_select_samples_sorted_chronologically() -> None:
    commits = (
        _ref("b", "2026-07-01"),
        _ref("a", "2026-06-01"),
    )
    result = select_samples(commits, months=2, anchor_shas=())
    assert [r.date for r in result] == ["2026-06-01", "2026-07-01"]


# ---------------------------------------------------------------------------
# select_samples + Excludes (G3 wiring, finding #5)
# ---------------------------------------------------------------------------


def test_select_samples_skips_codemod_commit_for_month_pick() -> None:
    codemod = _ref("codemod1", "2026-06-01", changed_file_count=50)
    normal = _ref("normal1", "2026-06-10", changed_file_count=2)
    excludes = Excludes()
    assert excludes.is_codemod_commit(50) is True

    result = select_samples((codemod, normal), months=1, anchor_shas=(), excludes=excludes)

    assert [r.sha for r in result] == ["normal1"]


def test_select_samples_skips_blame_ignored_sha_for_month_pick() -> None:
    ignored = _ref("ignoreme", "2026-06-01", changed_file_count=1)
    normal = _ref("normal1", "2026-06-10", changed_file_count=1)
    excludes = Excludes(blame_ignore_shas=frozenset({"ignoreme"}))

    result = select_samples((ignored, normal), months=1, anchor_shas=(), excludes=excludes)

    assert [r.sha for r in result] == ["normal1"]


def test_select_samples_anchors_are_never_excluded() -> None:
    # Anchors are explicit spec-named checkpoints -- must survive exclusion
    # filtering even if they happen to be codemod-shaped.
    codemod_anchor = _ref("anchor1", "2026-06-01", changed_file_count=999)
    excludes = Excludes()

    result = select_samples(
        (codemod_anchor,), months=0, anchor_shas=("anchor1",), excludes=excludes
    )

    assert [r.sha for r in result] == ["anchor1"]


# ---------------------------------------------------------------------------
# _criterion_counterexamples_clean: coverage floor (finding #3a)
# ---------------------------------------------------------------------------


def _point(kind, identity, file, count) -> AttachmentPoint:
    return AttachmentPoint(
        kind=kind, identity=identity, file=file, members=tuple(f"m{i}" for i in range(count))
    )


def _sample(sha: str, points: tuple[AttachmentPoint, ...], saturated: frozenset) -> SampleResult:
    return SampleResult(
        ref=_ref(sha, "2026-06-01"),
        points=points,
        saturated_keys=saturated,
        scanned_files=frozenset(p.file for p in points),
    )


def test_counterexamples_clean_fails_when_coverage_below_floor() -> None:
    # Only 1 of 13 counterexample modules ever appeared in the AP inventory
    # -- below the coverage floor, so "zero hits" must NOT count as a pass.
    module = COUNTEREXAMPLE_MODULES[0]
    points = (_point("class", "X", f"src/{module}", 3),)
    sample = _sample("s1", points, frozenset())

    result = _criterion_counterexamples_clean((sample,))

    assert result.passed is False
    assert "INCONCLUSIVE" in result.detail


def test_counterexamples_clean_passes_when_coverage_meets_floor_and_zero_hits() -> None:
    # >= half of the 13 counterexamples queried, none saturated.
    points = tuple(_point("class", "X", f"src/{m}", 3) for m in COUNTEREXAMPLE_MODULES[:7])
    sample = _sample("s1", points, frozenset())

    result = _criterion_counterexamples_clean((sample,))

    assert result.passed is True
    assert "zero false-positive" in result.detail


def test_counterexamples_clean_counts_bare_function_module_toward_coverage() -> None:
    # Round-2 review finding #8: a bare-function module (no class/router
    # archetype -> zero AttachmentPoints) that WAS actually walked by the
    # scanner must count toward coverage. Previously coverage was measured by
    # "produced an AP", which made this scenario permanently untestable for
    # 10 of the 13 spec-named counterexamples regardless of how many samples
    # covered them -- pinning the whole criterion at FAIL.
    bare_modules = COUNTEREXAMPLE_MODULES[:7]
    sample = SampleResult(
        ref=_ref("s1", "2026-06-01"),
        points=(),  # no archetype matched any of these bare-function modules
        saturated_keys=frozenset(),
        scanned_files=frozenset(f"src/{m}" for m in bare_modules),
    )

    result = _criterion_counterexamples_clean((sample,))

    assert result.passed is True
    assert "zero false-positive" in result.detail


def test_counterexamples_clean_fails_on_any_actual_hit_regardless_of_coverage() -> None:
    module = COUNTEREXAMPLE_MODULES[0]
    points = tuple(_point("class", "X", f"src/{m}", 3) for m in COUNTEREXAMPLE_MODULES[:7])
    saturated_key = ("class", f"src/{module}", "X")
    sample = _sample("s1", points, frozenset({saturated_key}))

    result = _criterion_counterexamples_clean((sample,))

    assert result.passed is False
    assert "false positives" in result.detail


# ---------------------------------------------------------------------------
# _cluster_b_informational: sees bare-function modules via scanned_files
# (finding #3b -- previously structurally guaranteed to read 0)
# ---------------------------------------------------------------------------


def test_cluster_b_counts_files_scanned_with_no_archetype_match() -> None:
    sample = SampleResult(
        ref=_ref("s1", "2026-06-01"),
        points=(),  # bare-function module: no class/typer/blueprint/ledger AP at all
        saturated_keys=frozenset(),
        scanned_files=frozenset({"src/pkg/bare_util.py"}),
    )

    hits, detail = _cluster_b_informational((sample,))

    assert hits == 1
    assert "src/pkg/bare_util.py" in detail


def test_cluster_b_zero_when_every_scanned_file_has_an_archetype() -> None:
    points = (_point("class", "X", "src/pkg/a.py", 3),)
    sample = SampleResult(
        ref=_ref("s1", "2026-06-01"),
        points=points,
        saturated_keys=frozenset(),
        scanned_files=frozenset({"src/pkg/a.py"}),
    )

    hits, _detail = _cluster_b_informational((sample,))

    assert hits == 0


# ---------------------------------------------------------------------------
# evaluate(): honest sample-window reporting (finding #3c)
# ---------------------------------------------------------------------------


def test_evaluate_reports_requested_vs_available_months() -> None:
    sample = _sample("s1", (), frozenset())
    verdict = evaluate((sample,), anchor_shas=(), requested_months=6)

    assert verdict.requested_months == 6
    # sample built via _ref/_sample has label="anchor" excluded but "" is a
    # (degenerate) non-anchor label here, so it counts as one "month" bucket.
    assert verdict.available_month_labels == ("",)
