"""Host routing in the unauthorized-merge tripwire (issue #2910, spec C2).

Two orchestrators on two hosts can serve one repository. The review decision
that covers a merged worker PR lives in the state directory of the host that
reviewed it, so the other host can only ever read ``missing``. The tripwire must
leave a merge whose linked issue the other host owns to that host, and must
still fail towards reporting whenever it cannot prove that.

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
from charlie_work.workflow import UNAUTHORIZED_MERGE_DETECTED_KEY, OrchestratorApp

_PR = 103
_ISSUE = 102
_KIND = "unauthorized_merge_detected"


class _SpyGitHub(FakeGitHub):
    """FakeGitHub that records every ``issue_list`` call and can fail it."""

    def __init__(self, *, fail_issue_list: bool = False) -> None:
        super().__init__()
        self.fail_issue_list = fail_issue_list
        self.issue_list_states: list[Any] = []

    def issue_list(self, labels=None, state=None):
        self.issue_list_states.append(state)
        if self.fail_issue_list:
            raise GitHubError("gh unavailable")
        return super().issue_list(labels, state)


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


@pytest.mark.parametrize(
    ("host", "default_host", "issue_labels", "case"),
    [
        ("workstation", True, {"host:macbook"}, "default host, issue labelled for the Mac"),
        ("macbook", False, {"host:workstation"}, "Mac, issue labelled for the workstation"),
        ("macbook", False, set(), "non-default host, unlabelled issue belongs to the default"),
        ("workstation", True, {"host:macbook", "host:workstation"}, "two hosts: owned by neither"),
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


def test_foreign_closed_issue_is_found_because_the_read_is_state_all(tmp_path: Path) -> None:
    """A CLOSED foreign issue (the normal post-merge shape) is visible only to a ``state="all"`` read."""
    gh = _SpyGitHub()
    app, _ = _app(
        tmp_path, _config("workstation"), gh, [_issue(_ISSUE, {"host:macbook"}, state="CLOSED")]
    )

    assert app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")]) == []
    assert gh.issue_list_states == ["all"]


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
    assert gh.issue_list_states == [], "routing off must not add an issue read"
    state_keys, ring, db_rows = _recorded(paths)
    assert state_keys == [str(_PR)] and len(ring) == 1 and len(db_rows) == 1


def test_unreadable_issue_list_fails_towards_reporting(tmp_path: Path) -> None:
    gh = _SpyGitHub(fail_issue_list=True)
    app, paths = _app(tmp_path, _config("workstation"), gh, [_issue(_ISSUE, {"host:macbook"})])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [d["pr"] for d in detected] == [_PR]
    assert gh.issue_list_states == ["all"], "the read was attempted, then failed"
    assert _recorded(paths)[0] == [str(_PR)]


def test_issue_missing_from_the_snapshot_is_not_proven_foreign(tmp_path: Path) -> None:
    gh = _SpyGitHub()
    app, _ = _app(tmp_path, _config("workstation"), gh, [_issue(7, {"host:macbook"})])

    detected = app._detect_unauthorized_merges([_merged_worker_pr(_PR, _ISSUE, "sha-103")])

    assert [d["pr"] for d in detected] == [_PR]


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
    assert gh.issue_list_states == [], "nothing to look up, so nothing to read"
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
    assert gh.issue_list_states == ["all"], "one read serves the whole pass"
