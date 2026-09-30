"""Issue #2048 stale-open-issue-mention exclusion tests for
``scripts/heartbeat_check.py``.

Seam-split sibling of ``test_heartbeat_check_stale_issue_mentions.py`` (the
#1556 Track-1 convention): the file's #902-era tests stay there; the #2048
exclusion machinery -- non-closing-context classification, parked/active
label exemption, bot-author exemption, and the excluded-count facts -- is
covered here so both modules stay under the 800-line cap.
"""

from __future__ import annotations

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
# Issue #2048 exclusions: a mention stops counting as closure evidence when
# (1) it sits in a non-closing context, (2) the issue carries a parked or
# active label, or (3) the mentioning PR is bot-authored.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Refs #700",
        "Ref #700",
        "Related #700",
        "See #700 for the design notes",
        "follow-up #700",
        "followup #700",
        "follow up to #700",
        "deferred #700",
        "deferral for #700",
        "part of #700",
        "filed as #700",
        "Refs: #700",
        "Refs #700, #701",  # enumeration stays in the qualifying clause
        "(refs #700)",  # paren clause still carries the qualifier
        "Refs #700\n\nFixed for real in #701",
    ],
    ids=[
        "refs",
        "ref",
        "related",
        "see",
        "follow-up-hyphen",
        "followup-bare",
        "follow-up-spaced",
        "deferred",
        "deferral",
        "part-of",
        "filed-as",
        "refs-colon",
        "refs-enumeration",
        "refs-parenthesized",
        "refs-cross-clause",
    ],
)
def test_issue_mention_occurrences_non_closing_contexts(hb: ModuleType, body: str) -> None:
    """Every keyword shape issue #2048 enumerates classifies non-closing."""
    occurrences = hb.issue_mention_occurrences(body)
    assert (700, True) in occurrences


def test_issue_mention_occurrences_for_issue_817_positive_control(hb: ModuleType) -> None:
    """Issue #2048's own regression shape: ``For issue #817:`` carries no
    non-closing qualifier, so it must keep flagging."""
    assert hb.issue_mention_occurrences("For issue #817: the deferred refactor") == [(817, False)]
    # A qualifier AFTER the mention never marks it non-closing either.
    assert hb.issue_mention_occurrences("Fixes #817, deferred cleanup remains") == [(817, False)]


def test_issue_mention_occurrences_per_occurrence_not_per_issue(hb: ModuleType) -> None:
    """One body may both reference and genuinely fix the same issue; the
    qualifying occurrence wins for that issue's evidence."""
    occurrences = hb.issue_mention_occurrences("Refs #5 for background.\nActually fixed #5.")
    assert (5, True) in occurrences
    assert (5, False) in occurrences


def _merged_pr(number: int, body: str, **extra: Any) -> dict[str, Any]:
    pr = {
        "number": number,
        "headRefName": "fix/unrelated",
        "title": "t",
        "body": body,
        "closingIssuesReferences": [],
        "mergedAt": "2026-09-30T00:00:00Z",
    }
    pr.update(extra)
    return pr


def test_check_stale_open_issue_mentions_excludes_refs_mentions(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #2048 exclusion 1, end to end: a `Refs #N` body mention of an
    open, unlabeled issue does not flag -- and the exclusion is counted."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700, 701],
        merged_prs=[_merged_pr(9, "Refs #700 (deferred re-key, filed as #701)")],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "stale_mentions=0" in report.lines[-1]
    assert "excluded_refs=2" in report.lines[-1]


def test_check_stale_open_issue_mentions_reports_excluded_counts_in_anomaly(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The facts line keeps the exclusions auditable even when findings
    remain."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700, 817],
        merged_prs=[_merged_pr(9, "Refs #700. For issue #817: done for real.")],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "#700" not in report.lines[-1]
    assert "excluded_refs=1" in report.lines[-1]


@pytest.mark.parametrize(
    "labels",
    [
        ["blocked"],
        ["needs-design"],
        ["human-action"],
        ["tracker"],
        ["umbrella"],
        ["epic"],
        ["agent:queued"],
        ["agent:in-progress"],
        ["agent:operator-queue"],
    ],
    ids=[
        "blocked",
        "needs-design",
        "human-action",
        "tracker",
        "umbrella",
        "epic",
        "agent-queued",
        "agent-in-progress",
        "agent-operator-queue",
    ],
)
def test_check_stale_open_issue_mentions_excludes_parked_and_active_labels(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path, labels: list[str]
) -> None:
    """Issue #2048 exclusion 2: parked labels (default config set incl. the
    tracker names) and every ``agent:*`` lifecycle label exempt the issue."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700],
        open_labels={700: labels},
        merged_prs=[_merged_pr(9, "Fixed for real in #700")],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "excluded_labels=1" in report.lines[-1]


def test_check_stale_open_issue_mentions_unlabeled_issue_still_flags(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The check's reason to exist survives: an issue with zero labels is
    never exempt by label."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[_merged_pr(824, _REAL_PR824_BODY_EXCERPT)],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "excluded_labels=0" in report.lines[-1]


def test_check_stale_open_issue_mentions_armed_issue_still_flags(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """``automated-ready`` (the armed marker) is neither parked nor active:
    an armed issue whose merged work never closed it is exactly the gap this
    check exists to catch."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        open_labels={817: ["automated-ready"]},
        merged_prs=[_merged_pr(824, _REAL_PR824_BODY_EXCERPT)],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]


def test_check_stale_open_issue_mentions_configured_lifecycle_rename_exempts(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A repo that renames a lifecycle label off the ``agent:`` prefix still
    gets exclusion 2 -- the set is derived from the repo's ``labels:``
    config section, not hard-coded."""
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("labels:\n  in_progress: fleet-working\n", encoding="utf-8")
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700],
        open_labels={700: ["fleet-working"]},
        merged_prs=[_merged_pr(9, "Fixed for real in #700")],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "excluded_labels=1" in report.lines[-1]


def test_check_stale_open_issue_mentions_configured_parked_labels(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """``heartbeat: stale_mention_parked_labels`` is authoritative: a
    configured name exempts, and a default name absent from the configured
    list stops exempting."""
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text(
        "heartbeat:\n  stale_mention_parked_labels: [custom-park]\n",
        encoding="utf-8",
    )
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700, 701],
        open_labels={700: ["custom-park"], 701: ["blocked"]},
        merged_prs=[
            _merged_pr(9, "Fixed for real in #700"),
            _merged_pr(10, "Fixed for real in #701"),
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    # #700 exempt (configured parked label); #701 still flags: with a
    # configured list the default set no longer applies, and "blocked" is
    # not in it -- the knob is authoritative, not additive.
    assert report.anomaly
    assert "#701" in report.lines[-1]
    assert "#700" not in report.lines[-1]
    assert "excluded_labels=1" in report.lines[-1]


def test_check_stale_open_issue_mentions_reports_config_load_error(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A malformed ``orchestrator.config.yaml`` is surfaced, not swallowed:
    the check still runs (``parked_label_names`` falls back to the mirrored
    default set, ``lifecycle_label_names`` contributes nothing) and the
    facts line carries a degraded note, same posture as the commit-scan
    degradation."""
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("- not a mapping\n", encoding="utf-8")
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700],
        merged_prs=[_merged_pr(9, "Fixed for real in #700")],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#700" in report.lines[-1]
    assert "config load degraded" in report.lines[-1]
    assert "expected a mapping" in report.lines[-1]


@pytest.mark.parametrize(
    "author",
    [
        {"login": "dependabot[bot]"},
        {"login": "renovate[bot]"},
        {"login": "dependabot"},
        {"login": "renovate"},
        {"login": "someone", "is_bot": True},
        {"login": "someone", "type": "Bot"},
    ],
    ids=[
        "dependabot-bot-login",
        "renovate-bot-login",
        "dependabot-bare",
        "renovate-bare",
        "is_bot-flag",
        "type-Bot",
    ],
)
def test_check_stale_open_issue_mentions_excludes_bot_authors(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path, author: dict[str, Any]
) -> None:
    """Issue #2048 exclusion 3: every bot-author shape exempts the whole
    PR's mentions, branch-derived ones included."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700],
        merged_prs=[_merged_pr(9, "Bump deps, see changelog for #700", author=author)],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "excluded_bots=1" in report.lines[-1]


@pytest.mark.parametrize(
    "author",
    [
        {"login": "senkichi"},
        {"login": "octocat"},
        None,
    ],
    ids=["human-login", "other-human", "deleted-account"],
)
def test_check_stale_open_issue_mentions_human_authors_not_excluded(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path, author: Any
) -> None:
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[700],
        merged_prs=[_merged_pr(9, "Fixed for real in #700", author=author)],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#700" in report.lines[-1]


def test_check_stale_open_issue_mentions_fetches_labels_and_author(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Structural: the exclusion inputs are actually fetched -- ``labels``
    on the issue-list call and ``author`` on the pr-list call."""
    repo = _make_repo(hb, tmp_path)
    captured: list[list[str]] = []
    _stale_mention_gh_dispatch(monkeypatch, hb, open_numbers=[1], merged_prs=[], captured=captured)
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    issue_calls = [a for a in captured if a[:2] == ["issue", "list"]]
    pr_calls = [a for a in captured if a[:2] == ["pr", "list"]]
    assert issue_calls and pr_calls
    assert "number,labels" in issue_calls[0]
    assert "author" in pr_calls[0][pr_calls[0].index("--json") + 1]


def test_stale_mention_parked_labels_default_mirrors_heartbeat_config(
    hb: ModuleType,
) -> None:
    """Drift guard: the stdlib-only script cannot import
    ``HeartbeatConfig``, so it carries a mirrored default -- this test keeps
    the two from drifting (same treatment as ``_slugify_branch``'s
    cross-check). ``hb`` re-exports the sibling's constant, so no second
    module load is needed."""
    from charlie_work.config import HeartbeatConfig

    assert hb.STALE_MENTION_PARKED_LABELS_DEFAULT == frozenset(
        HeartbeatConfig().stale_mention_parked_labels
    )
