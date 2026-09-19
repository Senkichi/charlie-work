"""Stale-open-issue-mentions tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``_mentioned_issue_numbers`` / ``_branch_issue_number`` /
``get_merged_commit_messages`` helpers and
``check_stale_open_issue_mentions`` (issue #902) with the real
PR #824 / commit-740484f captured payload excerpts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _REAL_PR824_BODY_EXCERPT,
    _gh_dispatch,
    _load_heartbeat_check,
    _make_repo,
    _stale_mention_gh_dispatch,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# check_stale_open_issue_mentions (issue #902)
#
# Real captured payload text, not paraphrased: PR #824's body starts with
# this exact prose (fetched via `gh pr view 824 --json body`, 2026-08-04).
# PR #824's branch (`fix/817-fleet-health-latch`) has no `agent/issue`
# prefix and its `closingIssuesReferences` came back `[]`, so this sentence
# was the only place the PR declared its target -- and it isn't a closing
# keyword, so `linked_issue_number` never binds on it either.
# ---------------------------------------------------------------------------


def test_mentioned_issue_numbers_extracts_817_from_real_pr824_body(hb: ModuleType) -> None:
    """Extraction stage, run against real (not synthetic) payload text.

    Proves the regex fires on GitHub's actual body shape, not just on data
    this test's author modeled after it -- the failure mode a purely
    synthetic fixture cannot rule out.
    """
    assert hb._mentioned_issue_numbers(_REAL_PR824_BODY_EXCERPT) == {817}


def test_branch_issue_number_extracts_817_from_real_pr824_branch(hb: ModuleType) -> None:
    assert hb._branch_issue_number("fix/817-fleet-health-latch") == 817


def test_branch_issue_number_ignores_agent_issue_prefixed_branch(hb: ModuleType) -> None:
    """No digit immediately after the slash -- already covered by the normal
    branch-prefix bind path, so a miss here is not a gap for this check."""
    assert (
        hb._branch_issue_number("agent/issue-414-feat-review-line-content-carry-forward") is None
    )


def test_mentioned_issue_numbers_suppresses_negated_reference(hb: ModuleType) -> None:
    """Issue #902 criterion 6 (negation half).

    The 32-char lookback window (matching `charlie_work.github`'s own
    documented tradeoff for the same negation guard) deliberately biases
    toward over-suppression: a match is only exempt from the negation check
    when no negation word appears anywhere in the preceding 32 characters,
    even across a clause boundary. That is a known, accepted cost -- a
    missed report here is safe (this check is advisory-only and re-scans
    every beat), not a correctness bug.
    """
    assert hb._mentioned_issue_numbers("This does not fix #817.") == set()
    # A reference with no negation word anywhere nearby is unaffected.
    assert hb._mentioned_issue_numbers("A completely unrelated change fixes #900.") == {900}


def test_mentioned_issue_numbers_suppresses_quoted_reference(hb: ModuleType) -> None:
    """Issue #902 criterion 6 (quoting half) -- #790's exact incident shape:
    a literal, quoted example inside prose must not be treated as live."""
    text = 'The bug looked the same as "Fixes #649" from before.'
    assert hb._mentioned_issue_numbers(text) == set()


def test_mentioned_issue_numbers_strips_fenced_code_blocks(hb: ModuleType) -> None:
    text = "See below:\n```\nraise ValueError('#404')\n```\nBut really see #817."
    assert hb._mentioned_issue_numbers(text) == {817}


def _init_test_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)


def _commit(repo: Path, filename: str, message: str) -> None:
    (repo / filename).write_text(filename, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def test_get_merged_commit_messages_parses_real_local_history(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Builds a real (throwaway) git repo and reads it back with the real
    `git log` plumbing -- not a mocked subprocess -- for the parsing stage.

    Self-contained by construction: the repo is created fresh under
    `tmp_path` and never reads this checkout's own history, so unlike the
    #866-shaped test below, it has no dependency on CI's clone depth
    (verified 2026-08-04 after that test failed in CI on a shallow clone).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_test_git_repo(repo)
    _commit(repo, "a.txt", "first commit\n\nrefs #123")
    _commit(repo, "b.txt", "second commit, unrelated")

    ok, commits, err = hb.get_merged_commit_messages(repo, limit=10)
    assert ok, err
    assert len(commits) == 2
    messages = [message for _, message in commits]
    assert any("refs #123" in message for message in messages)
    assert any("second commit, unrelated" in message for message in messages)


def test_get_merged_commit_messages_error_on_non_git_dir(hb: ModuleType, tmp_path: Path) -> None:
    ok, commits, err = hb.get_merged_commit_messages(tmp_path, limit=10)
    assert not ok
    assert commits == []
    assert err


# Real (not paraphrased) excerpt of PR #864's squash-merge commit body on
# `origin/main` (commit 740484f), captured verbatim via
# `git show -s --format=%B 740484f` (2026-08-04). #864 was filed and merged
# for a different issue (loop-pass staleness); this final bullet -- one of
# three sub-commits GitHub squashed into 740484f -- is the ONLY place
# anywhere that #866 is referenced. Neither "issue" nor a closing keyword
# precedes the reference, so only a bare `#N` scan over commit messages
# (never a PR title/body scan) can find it.
_REAL_PR864_SQUASH_COMMIT_EXCERPT = """\
* feat(heartbeat): surface error-level events with no consumer (refs #866)

self_deploy_alarm and every other member of instrumentation._ERROR_KINDS
(PR #865's supervisor_zero_pass_alarm included) are emitted, classified
error-level, documented, and unit-tested -- but nothing in the codebase
ever reads them back. A human had to manually open events.db and know
which kind string to search for.

check_error_events(report, repo, baseline) closes that gap by filtering
each repo's events.db on the level column persisted at write time by
_classify_level, so coverage is derived rather than a restated kind list
-- new alarm kinds are picked up with zero changes here. Missing/unreadable
events.db degrades to a reported ANOMALY (never an exception), which is
the deliberate point of disagreement with check_loop_pass_freshness's
"missing db = OK, no history yet" convention: this check's whole job is
"did any alarm fire," so an unreadable db is a repo it cannot vouch for.
Timestamps are compared in Python against baseline, never in SQL, per the
same ISO T/Z-vs-SQLite-space-format trap check_loop_pass_freshness already
guards against."""


def test_mentioned_issue_numbers_extracts_866_from_real_pr864_commit_excerpt(
    hb: ModuleType,
) -> None:
    """Extraction stage against the real PR #864 squash-commit text (see
    `_REAL_PR864_SQUASH_COMMIT_EXCERPT` above for provenance)."""
    assert 866 in hb._mentioned_issue_numbers(_REAL_PR864_SQUASH_COMMIT_EXCERPT)


def test_get_merged_commit_messages_extracts_866_from_real_squash_commit_shape(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Issue #902's #866 reproduction, through the REAL `git log` pipeline,
    against a throwaway repo this test owns.

    An earlier version of this test asserted against THIS checkout's own
    git history (commit 740484f, 19 commits back from `origin/main` HEAD at
    filing time) and passed locally. It failed in CI: `.github/workflows/ci.yml`'s
    `actions/checkout@v4` step sets no explicit `fetch-depth`, which defaults
    to a depth-1 (single-commit) shallow clone there, so commit 740484f was
    never fetched and unreachable from `git log` on the runner -- confirmed
    by grepping the workflow file for `fetch-depth` (absent) rather than
    assumed. Seeding an owned throwaway repo with the real commit-message
    text removes the ambient-history dependency while still exercising the
    real `git log` subprocess call end-to-end and the real payload shape --
    the actual failure mode this check exists to catch (#866's fix is only
    traceable via a commit message, never a PR title/body) requires running
    the true plumbing, not mocking it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_test_git_repo(repo)
    _commit(repo, "unrelated.txt", "unrelated setup commit, PR #1")
    _commit(repo, "heartbeat.py", _REAL_PR864_SQUASH_COMMIT_EXCERPT)

    ok, commits, err = hb.get_merged_commit_messages(repo, limit=10)
    assert ok, err
    matches = [sha for sha, message in commits if 866 in hb._mentioned_issue_numbers(message)]
    assert matches, "expected the seeded commit to reference #866"


def test_check_stale_open_issue_mentions_catches_817_824_reproduction(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 1: the #817/#824 reproduction, verbatim shape."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "fix: feed recovery observations into fleet health digest",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "PR #824" in report.lines[-1]


def test_check_stale_open_issue_mentions_catches_866_864_reproduction_via_commit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 2: #866's fix rode inside PR #864, a PR for a
    different issue, with no reference anywhere in #864's own title/body --
    only in one of its commit messages."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[866],
        merged_prs=[
            {
                "number": 864,
                "headRefName": "fix/heartbeat-loop-pass-staleness",
                "title": "feat(heartbeat): detect fleet loop-pass stall that log freshness cannot see",
                "body": "Adds check_loop_pass_freshness. No mention of any other issue here.",
                "closingIssuesReferences": [],
                "mergedAt": "2026-08-01T02:13:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        hb,
        "get_merged_commit_messages",
        lambda root, limit: (
            True,
            [
                ("740484f", "feat(heartbeat): loop-pass stall (#864)"),
                (
                    "e93fe13",
                    "feat(heartbeat): surface error-level events with no consumer (refs #866)",
                ),
            ],
            "",
        ),
    )

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#866" in report.lines[-1]
    assert "commit e93fe13" in report.lines[-1]


def test_check_stale_open_issue_mentions_no_label_filter_or_state_json(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 3 (corrected framing, 2026-08-04): an issue with
    NO LABELS AT ALL must still be reported. Enforced structurally: the
    candidate-set query must carry no `--label` filter, and the check must
    never read `state.json` -- both #817 and #866 have zero labels, and
    `workflow.py`'s `_merged_pr_referenced_issue_numbers` deliberately only
    ever considers the `ready`-labelled set, by design (see that function's
    docstring). This check's whole reason to exist is to not share that
    constraint.
    """
    repo = _make_repo(hb, tmp_path)
    captured: list[list[str]] = []
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[],
        captured=captured,
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    def _fail_if_called() -> None:
        raise AssertionError("check_stale_open_issue_mentions must never read state.json")

    monkeypatch.setattr(hb, "load_state", _fail_if_called)

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)  # would raise if load_state were called

    issue_list_calls = [args for args in captured if args[:2] == ["issue", "list"]]
    assert issue_list_calls, "expected an `issue list` call"
    assert "--label" not in issue_list_calls[0]
    assert "automated-ready" not in issue_list_calls[0]


def test_check_stale_open_issue_mentions_ok_when_issue_closed(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 4 (closed half): a closed issue never appears in
    the open-issue candidate set, so a PR mentioning it produces no report
    even though the text-matching machinery would otherwise fire."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[],  # 817 is closed -- absent from the open-issue query
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

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "stale_mentions=0" in report.lines[-1]


def test_check_stale_open_issue_mentions_ok_when_no_reference(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 4 (unreferenced half): an open issue with no
    merged-PR or commit reference anywhere produces no report."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[900],
        merged_prs=[
            {
                "number": 1,
                "headRefName": "fix/unrelated-cleanup",
                "title": "unrelated",
                "body": "nothing to see here",
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-01T00:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_never_issues_mutating_gh_calls(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 5: never auto-closes an issue or mutates a
    label. Enforced structurally -- every captured `gh` invocation this
    check makes must be a `list` (read) subcommand."""
    repo = _make_repo(hb, tmp_path)
    captured: list[list[str]] = []
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
        captured=captured,
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly  # sanity: a real finding did occur
    assert captured, "expected at least one gh call"
    for args in captured:
        assert args[1] == "list", f"non-read gh subcommand invoked: {args}"


def test_check_stale_open_issue_mentions_negated_reference_not_reported(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 6 (negation)."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 900,
                "headRefName": "fix/900-something-else",
                "title": "t",
                "body": "This change does not fix #817, it is unrelated.",
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_quoted_reference_not_reported(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 6 (quoting) -- #790's exact incident shape."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[649],
        merged_prs=[
            {
                "number": 788,
                "headRefName": "fix/negated-phrase-bug",
                "title": "t",
                "body": 'Demonstrates the negated-phrase bug: the same as "Fixes #649" from before.',
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_output_bounded(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Guard against unbounded output: many matches are capped and summarized."""
    repo = _make_repo(hb, tmp_path)
    numbers = list(range(1, 26))  # 25 distinct matches, cap is 20
    body = " ".join(f"#{n}" for n in numbers)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=numbers,
        merged_prs=[
            {
                "number": 1,
                "headRefName": "fix/many-refs",
                "title": "t",
                "body": body,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    line = report.lines[-1]
    detail = line.split("closure path: ", 1)[1].split(" (open=", 1)[0]
    parts = detail.split("; ")
    # 20 shown findings plus one "+N more" summary entry -- never all 25.
    assert len(parts) == hb.STALE_MENTION_REPORT_CAP + 1
    assert parts[-1] == "+5 more"


def test_check_stale_open_issue_mentions_anomaly_on_open_issue_list_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return False, None, "gh exploded"

    _gh_dispatch(monkeypatch, hb, handler)
    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)
    assert report.anomaly
    assert "gh exploded" in report.lines[-1]


def test_check_stale_open_issue_mentions_anomaly_on_merged_pr_list_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if args[:2] == ["issue", "list"]:
            return True, [], ""
        return False, None, "merged pr list exploded"

    _gh_dispatch(monkeypatch, hb, handler)
    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)
    assert report.anomaly
    assert "merged pr list exploded" in report.lines[-1]


def test_check_stale_open_issue_mentions_degrades_gracefully_when_git_log_fails(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A local `git log` failure is noted, not fatal -- the gh-sourced
    (branch/title/body) findings still get reported."""
    repo = _make_repo(hb, tmp_path)
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
    monkeypatch.setattr(
        hb, "get_merged_commit_messages", lambda root, limit: (False, [], "git log failed")
    )

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "commit-message scan degraded" in report.lines[-1]
