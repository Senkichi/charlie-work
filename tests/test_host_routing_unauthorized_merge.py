"""Host routing in the unauthorized-merge tripwire (issue #2910, spec C2).

Two orchestrators on two hosts can serve one repository. The review decision
that covers a merged worker PR lives in the state directory of the host that
reviewed it, so the other host can only ever read ``missing``. The tripwire must
leave a merge whose linked issue the other host owns to that host, and must
still fail towards reporting whenever it cannot prove that.

The proof is an exact per-issue ``issue_view`` read, never the 500-capped
``issue_list`` -- a foreign issue outside the newest-500 window is still proven.
A PR once proven foreign is remembered durably and never re-read.

Every "not reported" test has a positive control: the same merge, with the
issue owned here (or routing off), is reported.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire, _merged_worker_pr

from charlie_work.config import OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import (
    UNAUTHORIZED_MERGE_DETECTED_KEY,
    UNAUTHORIZED_MERGE_FOREIGN_KEY,
    OrchestratorApp,
)

_PR = 103
_ISSUE = 102
_KIND = "unauthorized_merge_detected"
_FOREIGN_KIND = "unauthorized_merge_foreign_dropped"


class _SpyGitHub(FakeGitHub):
    """FakeGitHub that records ``issue_view``/``issue_list`` calls and can fail them.

    ``issue_list_window`` simulates the real client's ``--limit 500`` cap: when
    set, ``issue_list`` answers only issues inside that window no matter what
    ``self.issues`` holds, while ``issue_view`` still answers by number.
    """

    def __init__(
        self, *, fail_issue_view: bool = False, issue_list_window: set[int] | None = None
    ) -> None:
        super().__init__()
        self.fail_issue_view = fail_issue_view
        self.issue_list_window = issue_list_window
        self.issue_view_calls: list[int] = []
        self.issue_list_states: list[Any] = []

    def issue_view(self, number: int):
        self.issue_view_calls.append(number)
        if self.fail_issue_view:
            raise GitHubError("gh unavailable")
        return super().issue_view(number)

    def issue_list(self, labels=None, state=None):
        self.issue_list_states.append(state)
        issues = super().issue_list(labels, state)
        if self.issue_list_window is not None:
            return [issue for issue in issues if issue.get("number") in self.issue_list_window]
        return issues


def _config(host: str | None, *, default_host: bool = True) -> OrchestratorConfig:
    base = OrchestratorConfig()
    return replace(base, dispatch=replace(base.dispatch, host=host, default_host=default_host))


def _issue(number: int, labels: set[str], *, state: str = "CLOSED") -> dict[str, Any]:
    # CLOSED by default: GitHub closes the issue when its PR merges, so by the
    # time the tripwire sees a merged PR its issue is normally no longer open.
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "",
        "labels": [{"name": name} for name in sorted(labels)],
        "state": state,
    }


def _app(
    tmp_path: Path, config: OrchestratorConfig, gh: FakeGitHub, issues: list[dict[str, Any]]
) -> tuple[OrchestratorApp, Any]:
    gh.issues = issues
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    return OrchestratorApp(tmp_path, paths, config, gh), paths


def _recorded(paths: Any) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """(state-record PR keys, state-ring events, events.db rows) of the finding."""
    state = load_state(paths.state_file)
    record = state.get(UNAUTHORIZED_MERGE_DETECTED_KEY) or {}
    ring = [e for e in state.get("events", []) if e["kind"] == _KIND]
    return list(record), ring, query_events(paths.state_file, kind=_KIND)


def _foreign_record(paths: Any) -> dict[str, Any]:
    state = load_state(paths.state_file)
    record = state.get(UNAUTHORIZED_MERGE_FOREIGN_KEY)
    return record if isinstance(record, dict) else {}


@pytest.mark.parametrize(
    ("host", "default_host", "issue_labels", "case"),
    [
        ("workstation", True, {"host:macbook"}, "default host, issue labelled for the Mac"),
        ("macbook", False, {"host:workstation"}, "Mac, issue labelled for the workstation"),
        ("macbook", False, set(), "non-default host, unlabelled issue belongs to the default"),
    ],
)
def test_merge_of_another_hosts_issue_is_not_reported(
    tmp_path: Path, host: str, default_host: bool, issue_labels: set[str], case: str
) -> None:
    gh = _SpyGitHub()
    app, paths = _app(
        tmp_path, _config(host, default_host=default_host), gh, [_issue(_ISSUE, issue_labels)]
    )

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert detected == [], case
    assert gh.issue_view_calls == [_ISSUE], f"the proof is the per-issue read ({case})"
    assert gh.issue_list_states == [], f"the capped list is never read ({case})"
    state_keys, ring, db_rows = _recorded(paths)
    assert state_keys == [], f"no state entry expected ({case})"
    assert ring == [] and db_rows == [], f"no {_KIND} event expected ({case})"


@pytest.mark.parametrize(
    ("host", "default_host", "issue_labels"),
    [
        ("workstation", True, {"host:workstation"}),
        ("workstation", True, set()),
        ("macbook", False, {"host:macbook"}),
    ],
)
def test_merge_of_a_locally_owned_issue_without_a_decision_is_still_reported(
    tmp_path: Path, host: str, default_host: bool, issue_labels: set[str]
) -> None:
    config = _config(host, default_host=default_host)
    gh = _SpyGitHub()
    app, paths = _app(tmp_path, config, gh, [_issue(_ISSUE, issue_labels)])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [(d["pr"], d["issue"], d["decision"]) for d in detected] == [(_PR, _ISSUE, "missing")]
    state_keys, ring, db_rows = _recorded(paths)
    assert state_keys == [str(_PR)]
    assert len(ring) == 1 and len(db_rows) == 1


def test_foreign_issue_outside_the_capped_issue_list_window_is_still_dropped(
    tmp_path: Path,
) -> None:
    """A backlog issue outside the newest-500 window still proves foreign.

    ``gh issue list`` is capped at the 500 newest issues, so a repo past that
    cap cannot see a backlog issue through the list -- the false positive from
    #2910 would return. The proof is ``issue_view(number)``, which has no
    window. If the read is reverted to ``issue_list(state="all")`` this test
    fails: the window hides the issue and the finding reports.
    """
    gh = _SpyGitHub(issue_list_window={7})
    app, _ = _app(tmp_path, _config("workstation"), gh, [_issue(_ISSUE, {"host:macbook"})])
    assert _ISSUE not in gh.issue_list_window, "the capped window must not see the issue"

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert detected == []
    assert gh.issue_view_calls == [_ISSUE]


@pytest.mark.parametrize(
    ("host", "default_host", "issue_labels", "expected_prs", "case"),
    [
        (
            "workstation",
            True,
            {"host:macbook", "host:workstation"},
            [_PR],
            "two host labels: the issue belongs to neither host, so it stays reported",
        ),
        (
            "macbook",
            False,
            {"host:macbook", "host:workstation"},
            [_PR],
            "same contested issue seen from the other host",
        ),
        (
            "workstation",
            True,
            {"host:macbook"},
            [],
            "positive control: one foreign label is positive proof and drops",
        ),
    ],
)
def test_a_two_host_label_issue_keeps_the_finding_on_both_hosts(
    tmp_path: Path,
    host: str,
    default_host: bool,
    issue_labels: set[str],
    expected_prs: list[int],
    case: str,
) -> None:
    gh = _SpyGitHub()
    app, _ = _app(
        tmp_path, _config(host, default_host=default_host), gh, [_issue(_ISSUE, issue_labels)]
    )

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [d["pr"] for d in detected] == expected_prs, case


def test_a_proven_foreign_pr_is_remembered_and_never_re_read(tmp_path: Path) -> None:
    """The per-issue proof is paid once per PR, then recorded durably (#2910 rework).

    A foreign merge is re-detected as a candidate on every pass -- it is
    neither baselined nor acked -- so the durable record under
    ``UNAUTHORIZED_MERGE_FOREIGN_KEY`` is what keeps the proof from costing an
    ``issue_view`` per pass. The record survives a fresh orchestrator on the
    same state directory, and the proof event fires once, on the pass that
    proved it.
    """
    config = _config("workstation")
    gh = _SpyGitHub()
    app, paths = _app(tmp_path, config, gh, [_issue(_ISSUE, {"host:macbook"})])
    merged = [_merged_worker_pr(_PR, _ISSUE, "sha-103")]

    assert app._detect_unauthorized_merges(merged) == []
    assert gh.issue_view_calls == [_ISSUE]
    record = _foreign_record(paths)
    assert set(record) == {str(_PR)}
    assert record[str(_PR)]["issue"] == _ISSUE
    assert record[str(_PR)]["proven_at"]

    # Second pass, same orchestrator: the durable record answers, not the API.
    assert app._detect_unauthorized_merges(merged) == []
    assert gh.issue_view_calls == [_ISSUE], "a remembered PR must not be re-read"

    # A fresh orchestrator on the same state directory: still no re-read.
    gh2 = _SpyGitHub()
    gh2.issues = gh.issues
    app2 = OrchestratorApp(tmp_path, paths, config, gh2)
    assert app2._detect_unauthorized_merges(merged) == []
    assert gh2.issue_view_calls == [], "the durable record outlives the process"

    # The proof event fired once, on the first pass only.
    assert len(query_events(paths.state_file, kind=_FOREIGN_KIND)) == 1


def test_a_failed_proof_is_not_remembered(tmp_path: Path) -> None:
    """An unreadable issue keeps the finding and is retried next pass."""
    gh = _SpyGitHub(fail_issue_view=True)
    app, paths = _app(tmp_path, _config("workstation"), gh, [_issue(_ISSUE, {"host:macbook"})])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [d["pr"] for d in detected] == [_PR]
    assert gh.issue_view_calls == [_ISSUE], "the read was attempted, then failed"
    assert _foreign_record(paths) == {}, "an unproven PR must not be durably exempted"
    assert _recorded(paths)[0] == [str(_PR)]

    # The read recovers next pass and proves foreign then.
    gh.fail_issue_view = False
    assert app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")]) == []
    assert gh.issue_view_calls == [_ISSUE, _ISSUE]
    assert set(_foreign_record(paths)) == {str(_PR)}


def test_issue_view_returning_nothing_is_not_proven_foreign(tmp_path: Path) -> None:
    """An issue the lookup cannot find keeps the finding (fail towards reporting)."""
    gh = _SpyGitHub()
    app, _ = _app(tmp_path, _config("workstation"), gh, [_issue(7, {"host:macbook"})])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [d["pr"] for d in detected] == [_PR]
    assert gh.issue_view_calls == [_ISSUE]


def test_with_routing_off_the_finding_is_reported_exactly_as_before(tmp_path: Path) -> None:
    """``dispatch.host`` unset: a ``host:`` label on the issue changes nothing and costs no read."""
    config = _config(None)
    gh = _SpyGitHub()
    app, paths = _app(tmp_path, config, gh, [_issue(_ISSUE, {"host:macbook"})])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert detected == [
        {
            "pr": _PR,
            "issue": _ISSUE,
            "head": f"agent/issue-{_ISSUE}-fix",
            "decision": "missing",
            "reviewed_head_sha": None,
            "live_head_sha": "sha-103",
            "review_dispatch_enabled": config.review_dispatch.enabled,
        }
    ]
    assert gh.issue_view_calls == [] and gh.issue_list_states == [], (
        "routing off must not add an issue read"
    )
    state_keys, ring, db_rows = _recorded(paths)
    assert state_keys == [str(_PR)] and len(ring) == 1 and len(db_rows) == 1


def test_pr_that_links_no_issue_stays_in_scope_without_an_issue_read(tmp_path: Path) -> None:
    gh = _SpyGitHub()
    app, paths = _app(tmp_path, _config("workstation"), gh, [_issue(_ISSUE, {"host:macbook"})])
    # Worker-prefixed branch with no issue number and a body with no closing keyword.
    unlinked = {
        **_merged_worker_pr(_PR, _ISSUE, "sha-103"),
        "headRefName": "agent/issue-hotfix",
        "body": "no closing keyword here",
    }

    detected = app._detect_unauthorized_merges([unlinked])

    assert [(d["pr"], d["issue"]) for d in detected] == [(_PR, None)]
    assert gh.issue_view_calls == [], "nothing to look up, so nothing to read"
    assert _recorded(paths)[0] == [str(_PR)]


def test_only_the_foreign_finding_is_dropped_from_a_mixed_pass(tmp_path: Path) -> None:
    gh = _SpyGitHub()
    app, paths = _app(
        tmp_path,
        _config("workstation"),
        gh,
        [_issue(_ISSUE, {"host:macbook"}), _issue(91, {"host:workstation"})],
    )

    detected = app._detect_unauthorized_merges(
        [_merged_worker_pr(_PR, _ISSUE, "sha-103"), _merged_worker_pr(101, 91, "sha-101")]
    )

    assert [(d["pr"], d["issue"]) for d in detected] == [(101, 91)]
    assert _recorded(paths)[0] == ["101"]
    assert set(gh.issue_view_calls) == {_ISSUE, 91}, "each linked issue is read once by number"
