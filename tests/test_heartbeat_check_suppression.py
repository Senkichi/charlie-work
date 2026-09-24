"""Suppression-registry tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
the issue #1361 acceptance scenarios (missing file, name-only match,
repo/substring narrowing, expiry boundary, unmatched accounting,
malformed fails-closed) plus the seeded-registry and
stale-issue-mentions integration checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _REAL_PR824_BODY_EXCERPT,
    _load_heartbeat_check,
    _make_repo,
    _stale_mention_gh_dispatch,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# Suppression registry (issue #1361)
# ---------------------------------------------------------------------------
# Seven scenarios per the issue's acceptance criteria:
#   1. empty/missing registry -> byte-identical to pre-#1361 behavior
#   2. active entry matched by check name only -> SUPPRESSED, verbatim detail
#   3. match narrowed by repo
#   4. match narrowed by substring
#   5. expiry boundary: an entry expiring today is expired
#   6. unmatched accounting in the summary line
#   7. malformed entry fails closed


def test_suppression_registry_missing_file_is_empty_and_noop(
    hb: ModuleType, tmp_path: Path
) -> None:
    """AC1: a missing registry file loads as zero entries, not an error, and
    a Report with zero suppressions behaves exactly like pre-#1361 code --
    every other test in this file constructs `hb.Report()` with no
    suppressions arg for exactly this reason."""
    entries, err = hb.load_suppression_registry(tmp_path / "does-not-exist.yaml")
    assert entries == []
    assert err is None

    report = hb.Report()
    assert report.suppressions == []
    report.anom("some-check", "some detail")
    assert report.lines == ["ANOMALY some-check: some detail"]
    assert report.anomaly
    assert report.suppression_summary() is None


def test_suppression_active_entry_matched_by_check_name_only(hb: ModuleType) -> None:
    """AC2: an active, unscoped (no repo, no match substring) entry converts
    ANOMALY into SUPPRESSED and the detail text survives verbatim."""
    entry = hb.SuppressionEntry(
        check="some-check", issue=42, expires="2099-01-01", repo=None, match=""
    )
    report = hb.Report(suppressions=[entry])
    report.anom("some-check", "42 things went sideways")

    assert report.lines == [
        "SUPPRESSED some-check: [#42 until 2099-01-01] 42 things went sideways"
    ]
    assert not report.anomaly


def test_suppression_narrowed_by_repo(hb: ModuleType) -> None:
    """AC3 (this issue's repo-narrowing scenario): an entry scoped to one
    repo does not suppress the same check name emitted for a different
    repo, but does suppress it for the repo it names."""
    entry = hb.SuppressionEntry(
        check="some-check", issue=7, expires="2099-01-01", repo="owner/repo-a"
    )
    report = hb.Report(suppressions=[entry])

    report.anom("some-check owner/repo-b", "detail for repo b")
    assert report.lines[-1] == "ANOMALY some-check owner/repo-b: detail for repo b"
    assert report.anomaly

    report.anom("some-check owner/repo-a", "detail for repo a")
    assert (
        report.lines[-1]
        == "SUPPRESSED some-check owner/repo-a: [#7 until 2099-01-01] detail for repo a"
    )


def test_suppression_narrowed_by_substring(hb: ModuleType) -> None:
    """`match` narrows suppression to details containing that substring;
    an unrelated detail for the same check still surfaces as ANOMALY."""
    entry = hb.SuppressionEntry(
        check="some-check", issue=9, expires="2099-01-01", match="known flaky condition"
    )
    report = hb.Report(suppressions=[entry])

    report.anom("some-check", "unrelated failure, never seen before")
    assert report.lines[-1] == "ANOMALY some-check: unrelated failure, never seen before"
    assert report.anomaly

    report.anom("some-check", "known flaky condition: count=3")
    assert (
        report.lines[-1]
        == "SUPPRESSED some-check: [#9 until 2099-01-01] known flaky condition: count=3"
    )


def test_suppression_expiry_boundary_expiring_today_is_expired(hb: ModuleType) -> None:
    """AC7 boundary case: an entry whose `expires` date equals the run's
    current date is already expired (inclusive), not suppressed for one
    more day -- the resurfaced line is annotated and flips the exit code."""
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    entry = hb.SuppressionEntry(check="some-check", issue=5, expires="2026-09-30")
    report = hb.Report(suppressions=[entry], now=now)

    report.anom("some-check", "still happening")

    assert report.lines == [
        "ANOMALY some-check: [suppression #5 EXPIRED 2026-09-30] still happening"
    ]
    assert report.anomaly


def test_suppression_expiry_boundary_day_before_is_still_active(hb: ModuleType) -> None:
    now = datetime(2026, 9, 29, 23, 59, tzinfo=timezone.utc)
    entry = hb.SuppressionEntry(check="some-check", issue=5, expires="2026-09-30")
    report = hb.Report(suppressions=[entry], now=now)

    report.anom("some-check", "still happening")

    assert report.lines == ["SUPPRESSED some-check: [#5 until 2026-09-30] still happening"]
    assert not report.anomaly


def test_suppression_unmatched_accounting_in_summary(hb: ModuleType) -> None:
    """AC5/AC7: the summary line accounts active/expired-by-date entries
    plus, orthogonally, how many matched nothing this run -- a signal that
    the underlying condition cleared and the entry is a deletion candidate."""
    now = datetime(2026, 8, 21, tzinfo=timezone.utc)
    matched_active = hb.SuppressionEntry(check="check-a", issue=1, expires="2099-01-01")
    unmatched_active = hb.SuppressionEntry(check="check-b", issue=2, expires="2099-01-01")
    matched_expired = hb.SuppressionEntry(check="check-c", issue=3, expires="2026-08-01")
    report = hb.Report(suppressions=[matched_active, unmatched_active, matched_expired], now=now)

    report.anom("check-a", "detail a")
    report.anom("check-c", "detail c")

    assert report.suppression_summary() == "active=2 expired=1 unmatched=1"


def test_suppression_no_suppressions_summary_is_none(hb: ModuleType) -> None:
    report = hb.Report()
    assert report.suppression_summary() is None


def test_suppression_malformed_yaml_fails_closed(hb: ModuleType, tmp_path: Path) -> None:
    """AC4/AC7: unparseable YAML is a load error, entries come back empty
    (fail closed -- nothing this run can be silently suppressed)."""
    path = tmp_path / "heartbeat-suppressions.yaml"
    path.write_text("check: [unterminated\n", encoding="utf-8")

    entries, err = hb.load_suppression_registry(path)

    assert entries == []
    assert err is not None
    assert "YAML parse error" in err


def test_suppression_malformed_entry_missing_required_field_fails_closed(
    hb: ModuleType, tmp_path: Path
) -> None:
    """AC4/AC7: a syntactically valid YAML list whose entry is missing a
    required field (`expires`) is malformed too -- fail closed rather than
    silently defaulting the missing field."""
    path = tmp_path / "heartbeat-suppressions.yaml"
    path.write_text(
        "- check: some-check\n  issue: 1\n  note: missing expires\n",
        encoding="utf-8",
    )

    entries, err = hb.load_suppression_registry(path)

    assert entries == []
    assert err is not None
    assert "entry 0" in err

    # Wired through Report exactly like every other ANOMALY: fail-closed
    # means this ANOMALY is additive, and with entries == [] nothing else
    # this run can be suppressed.
    report = hb.Report(suppressions=entries)
    report.anom("suppression-registry", err)
    assert report.anomaly
    assert report.lines == [f"ANOMALY suppression-registry: {err}"]


def test_suppression_malformed_entry_bad_repo_type_fails_closed(
    hb: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "heartbeat-suppressions.yaml"
    path.write_text(
        "- check: some-check\n  issue: 1\n  expires: '2099-01-01'\n  repo: 123\n",
        encoding="utf-8",
    )
    entries, err = hb.load_suppression_registry(path)
    assert entries == []
    assert err is not None
    assert "repo" in err


def test_suppression_integration_with_check_stale_open_issue_mentions(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """End-to-end through the real per-repo check: the seeded-registry shape
    (base check name + repo, no embedded repo suffix in `check`) suppresses
    the actual stale-open-issue-mentions anomaly for the matching repo, and
    the exit-code-relevant `anomaly` flag stays False."""
    repo = _make_repo(hb, tmp_path)  # slug="owner/repo"
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "t",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    entry = hb.SuppressionEntry(
        check="stale-open-issue-mentions",
        issue=1361,
        expires="2099-01-01",
        repo="owner/repo",
        match="referenced by merged work",
    )
    report = hb.Report(suppressions=[entry])
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert report.lines[-1].startswith(
        "SUPPRESSED stale-open-issue-mentions owner/repo: [#1361 until 2099-01-01]"
    )
    assert "#817" in report.lines[-1]
    assert "PR #824" in report.lines[-1]


def test_seeded_registry_loads_and_matches_both_fleet_repos(hb: ModuleType) -> None:
    """The registry checked into the repo (scripts/heartbeat-suppressions.yaml)
    parses cleanly as a single fleet-wide entry with a live (non-expired, as
    of this test's authorship) tracking issue: `repo` is omitted so the
    suppression covers every current and future fleet repo by construction
    (issue #1860 -- the prior per-repo enumeration missed Senkichi/jobcannon
    and Senkichi/swole when they onboarded). The leaf name predates #1860
    and is kept verbatim: the collect-only gate (issue #1538) fails test
    renames fail-closed, and it stays accurate since the one fleet-wide
    entry still matches both repos it names, along with every other."""
    registry_path = Path(__file__).parent.parent / "scripts" / "heartbeat-suppressions.yaml"
    entries, err = hb.load_suppression_registry(registry_path)

    assert err is None
    assert len(entries) == 1
    (entry,) = entries
    assert entry.check == "stale-open-issue-mentions"
    assert entry.repo is None
    assert entry.issue == 1361
    assert entry.expires == "2026-09-30"


def test_seeded_registry_actually_suppresses_the_real_per_repo_check_name(
    hb: ModuleType,
) -> None:
    """Closes the gap the two tests above leave open individually: parsing
    the checked-in registry and matching *some* check name each prove half
    the path. This proves the file on disk suppresses the exact string
    `check_stale_open_issue_mentions` emits for each real fleet repo
    (f"stale-open-issue-mentions {repo.slug}") -- not a stand-in owner/repo
    slug, and not a hypothetical shape. With `repo` omitted (issue #1860)
    it must also cover the repos the per-repo enumeration missed
    (Senkichi/jobcannon, Senkichi/swole) and any slug a future onboarding
    adds, so the entry can never again lag the fleet roster. If a future
    edit bakes the repo suffix into the seeded `check:` field instead of
    using the separate `repo:` field, this test fails loudly; the tests
    above would not."""
    registry_path = Path(__file__).parent.parent / "scripts" / "heartbeat-suppressions.yaml"
    entries, err = hb.load_suppression_registry(registry_path)
    assert err is None

    for slug in (
        "Senkichi/charlie-work",
        "Senkichi/job-cannon",
        "Senkichi/jobcannon",
        "Senkichi/swole",
        "Senkichi/a-repo-not-yet-onboarded",
    ):
        report = hb.Report(suppressions=entries)
        check = f"stale-open-issue-mentions {slug}"
        detail = (
            "18 open issue(s) referenced by merged work with no closure path: "
            "#817 (PR #824 fix/817-fleet-health-latch) (open=18)"
        )
        report.anom(check, detail)

        assert not report.anomaly, f"registry failed to suppress slug {slug}"
        assert report.lines[-1].startswith(f"SUPPRESSED {check}: [#1361 until 2026-09-30]")
