"""Host routing in reconcile and the review/merge lane (spec C2, interface I-7).

Reconcile, janitor routing and rescue leave an issue another host owns, and the
PRs bound to it, alone. Each assertion has a positive control: the same fixture
with routing off (or owned here) still produces today's drift.

"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub as AppFakeGitHub
from _reconcile_fixtures import FakeGitHub, _issue, _pr
from _review_fixtures import _approved_automerge, _fake_claude_worker_record
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.github import GitHubError
from charlie_work.host_routing import foreign_issue_numbers, local_issues, local_prs
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import detect_drift
from charlie_work.state import empty_state
from charlie_work.workflow import OrchestratorApp


def _config(host: str | None, *, default_host: bool = True) -> OrchestratorConfig:
    base = OrchestratorConfig()
    return replace(base, dispatch=replace(base.dispatch, host=host, default_host=default_host))


def _routed(config: OrchestratorConfig, host: str = "workstation") -> OrchestratorConfig:
    return replace(config, dispatch=replace(config.dispatch, host=host))


def _kinds(gh: FakeGitHub, config: OrchestratorConfig, number: int) -> set[str]:
    drift = detect_drift(gh, empty_state(), config)
    return {item.kind for item in drift if item.issue_number == number}


def test_foreign_issue_numbers() -> None:
    issues = [_issue(1, []), _issue(2, ["host:macbook"]), _issue(3, ["host:workstation"])]
    assert foreign_issue_numbers(issues, _config(None)) == frozenset()
    assert foreign_issue_numbers(issues, _config("workstation")) == frozenset({2})
    assert foreign_issue_numbers(issues, _config("macbook", default_host=False)) == frozenset(
        {1, 3}
    )


def test_local_issues_keeps_the_list_when_routing_is_off() -> None:
    issues = [_issue(1, []), _issue(2, ["host:macbook"])]
    assert local_issues(issues, _config(None)) is issues
    assert [i["number"] for i in local_issues(issues, _config("workstation"))] == [1]


def test_reconcile_skips_an_active_label_on_a_foreign_issue() -> None:
    labels = OrchestratorConfig().labels
    gh = FakeGitHub(prs=[], issues=[_issue(40, [labels.in_progress, "host:macbook"])])

    # Positive control: with routing off the stranded active label is drift.
    assert "issue_active_label_no_open_pr" in _kinds(gh, _config(None), 40)
    assert _kinds(gh, _config("workstation"), 40) == set()


def test_reconcile_still_heals_an_issue_owned_here() -> None:
    labels = OrchestratorConfig().labels
    gh = FakeGitHub(prs=[], issues=[_issue(40, [labels.in_progress, "host:macbook"])])
    assert "issue_active_label_no_open_pr" in _kinds(
        gh, _config("macbook", default_host=False), 40
    )


def test_reconcile_skips_a_merged_pr_bound_to_a_foreign_issue() -> None:
    # The window between the owner's merge and its finalize: the PR is merged
    # and the issue is still open with its active label. Only the owner closes it.
    labels = OrchestratorConfig().labels
    gh = FakeGitHub(
        prs=[_pr(50, "MERGED", head_ref="agent/issue-40-x")],
        issues=[_issue(40, [labels.in_progress, "host:macbook"])],
    )
    assert "merged_outside_orchestrator" in _kinds(gh, _config(None), 40)
    assert "merged_outside_orchestrator" not in _kinds(gh, _config("workstation"), 40)


class _App:
    def __init__(self, gh: Any, config: OrchestratorConfig) -> None:
        self.gh = gh
        self.config = config


def _gh_with_host(label: str | None) -> AppFakeGitHub:
    gh = AppFakeGitHub()
    if label is not None:
        gh.issues[0]["labels"].append({"name": label})
    return gh


def test_local_prs_is_a_no_op_with_routing_off() -> None:
    gh = _gh_with_host("host:macbook")
    prs = gh.pr_list()
    assert local_prs(_App(gh, _config(None)), prs) is prs


def test_local_prs_follows_the_linked_issue_owner() -> None:
    # FakeGitHub's PR #456 is bound to issue #123 (branch agent/issue-123-...).
    mac = _gh_with_host("host:macbook")
    assert [p["number"] for p in local_prs(_App(mac, _config("workstation")), mac.pr_list())] == []
    mac_config = _config("macbook", default_host=False)
    assert [p["number"] for p in local_prs(_App(mac, mac_config), mac.pr_list())] == [456]

    plain = _gh_with_host(None)
    assert [p["number"] for p in local_prs(_App(plain, mac_config), plain.pr_list())] == []
    workstation = _config("workstation")
    assert [p["number"] for p in local_prs(_App(plain, workstation), plain.pr_list())] == [456]


def test_an_unlinked_pr_belongs_to_the_default_host() -> None:
    gh = _gh_with_host(None)
    gh.prs[0] = {**gh.prs[0], "headRefName": "operator/tweak", "title": "tweak", "body": ""}
    assert len(local_prs(_App(gh, _config("workstation")), gh.pr_list())) == 1
    assert local_prs(_App(gh, _config("macbook", default_host=False)), gh.pr_list()) == []


def test_local_prs_rejects_a_stale_branch_binding_and_follows_the_closer() -> None:
    """The stale-branch guard: PR #456 sits on branch ``agent/issue-123-…`` but
    #123 is CLOSED; its body closes foreign issue #500 (issue #2822 review
    finding 1). Linkage must resolve identically to the reap loop's
    validator-threaded scan: the branch-name binding to closed #123 is
    rejected, the body's closing keyword binds #500, and ownership follows
    #500's host label."""
    gh = AppFakeGitHub()
    gh.issues[0]["state"] = "CLOSED"
    gh.issues.append(
        {
            "number": 500,
            "title": "Foreign work",
            "url": "https://example.test/issues/500",
            "body": "Foreign work",
            "labels": [{"name": "host:macbook"}],
            "state": "OPEN",
        }
    )
    gh.prs = [
        {
            **gh.prs[0],
            "title": "Unrelated change",
            "body": "Closes #500\n\nTests: regression coverage added.",
        }
    ]
    prs = gh.pr_list()
    # The default host drops the PR: it would close the macbook host's issue.
    assert [p["number"] for p in local_prs(_App(gh, _config("workstation")), prs)] == []
    # The owning host keeps it.
    assert [
        p["number"] for p in local_prs(_App(gh, _config("macbook", default_host=False)), prs)
    ] == [456]


class _FailingIssues(AppFakeGitHub):
    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        raise GitHubError("boom")


def test_issue_fetch_failure_keeps_todays_lane_on_the_default_host_only() -> None:
    gh = _FailingIssues()
    prs = gh.pr_list()
    assert local_prs(_App(gh, _config("workstation")), prs) == prs
    assert local_prs(_App(gh, _config("macbook", default_host=False)), prs) == []


def _finalize_app(root: Path, config: OrchestratorConfig) -> tuple[OrchestratorApp, AppFakeGitHub]:
    """The OrchestratorApp + _finalize_externally_merged_issues(app) harness.

    Issue #123 is CLOSED with the ready label and a ``host:macbook`` label
    (foreign, no merged PR binds it); issue #124 is OPEN and foreign; issue
    #125 is CLOSED, ready and owned here.
    """
    paths = runtime_paths(root, config.runtime.state_dir)
    fake_gh = AppFakeGitHub()
    fake_gh.issues[0]["state"] = "CLOSED"
    fake_gh.issues[0]["labels"] = [
        {"name": config.labels.ready},
        {"name": "host:macbook"},
    ]
    fake_gh.issues.append(
        {
            "number": 124,
            "title": "Foreign work",
            "url": "https://example.test/issues/124",
            "body": "Foreign work",
            "labels": [{"name": config.labels.ready}, {"name": "host:macbook"}],
            "state": "OPEN",
        }
    )
    fake_gh.issues.append(
        {
            "number": 125,
            "title": "Local work",
            "url": "https://example.test/issues/125",
            "body": "Local work",
            "labels": [{"name": config.labels.ready}],
            "state": "CLOSED",
        }
    )
    return OrchestratorApp(root, paths, config, fake_gh), fake_gh


def test_finalize_externally_merged_leaves_foreign_ready_issues_alone(
    tmp_path: Path,
) -> None:
    """Issue #2822 review finding 2: _finalize_externally_merged_issues filters
    its ready-issue list through host_routing.local_issues, so on a routed
    host a foreign issue is neither label-stripped nor returned as this
    host's ready backlog, while the owned issue still is. Positive control:
    with routing off, the same closed foreign-labelled ready issue is
    stripped."""
    from charlie_work.orchestration.state_merge_train import (
        _finalize_externally_merged_issues,
    )

    labels = OrchestratorConfig().labels
    routing_off = _config(None)

    app, gh = _finalize_app(tmp_path / "routing-off", routing_off)
    _finalized, remaining, _outcome = _finalize_externally_merged_issues(app)
    assert 123 not in [i["number"] for i in remaining]  # finalized and dropped
    assert (123, labels.ready) in gh.labels_removed  # ...and label-stripped

    routed = _routed(routing_off)
    app, gh = _finalize_app(tmp_path / "routed", routed)
    _finalized, remaining, _outcome = _finalize_externally_merged_issues(app)
    numbers = [i["number"] for i in remaining]
    # Foreign issues: closed #123 is absent and not label-stripped; open #124
    # is absent from the returned ready list.
    assert 123 not in numbers
    assert (123, labels.ready) not in gh.labels_removed
    assert 124 not in numbers
    # The owned closed ready issue #125 is still finalized locally.
    assert 125 not in numbers
    assert (125, labels.ready) in gh.labels_removed


def _loop_pr(number: int, issue: int) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{issue}: work",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue}-fix",
        "baseRefName": "main",
        "headRefOid": f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue}\n\nTests: regression coverage added.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


_LOOP_ISSUES = [
    {
        "number": 123,
        "title": "Local work",
        "url": "https://example.test/issues/123",
        "body": "Local work",
        "labels": [{"name": "automated-ready"}],
        "state": "OPEN",
    },
    {
        "number": 500,
        "title": "Foreign work",
        "url": "https://example.test/issues/500",
        "body": "Foreign work",
        "labels": [{"name": "host:macbook"}],
        "state": "OPEN",
    },
]


def _loop_app(root: Path, config: OrchestratorConfig) -> tuple[OrchestratorApp, AppFakeGitHub]:
    """The reap-loop harness from tests/test_charlie_work_loop_review.py:
    OrchestratorApp with two open PRs (456 -> local issue 123, 457 -> foreign
    issue 500)."""
    paths = runtime_paths(root, config.runtime.state_dir)
    fake_gh = AppFakeGitHub()
    fake_gh.issues = [dict(issue) for issue in _LOOP_ISSUES]
    fake_gh.prs = [_loop_pr(456, 123), _loop_pr(457, 500)]
    return OrchestratorApp(root, paths, config, fake_gh), fake_gh


def _approved_reviewer(app: OrchestratorApp) -> Any:
    """A launch_claude_worker stand-in that writes an approved review
    decision at the launched head, so the loop's review/merge lane treats
    the PR as approved."""

    def fake_launch(issue_number: int, branch: str, prompt_text: str, **kwargs: Any):
        pr_dir = app.paths.prs / f"pr-{issue_number}"
        pr_dir.mkdir(parents=True, exist_ok=True)
        (pr_dir / "review-decision.json").write_text(
            json.dumps(
                {
                    "decision": "approved",
                    "reviewed_head_sha": kwargs.get("head_sha"),
                }
            ),
            encoding="utf-8",
        )
        return _fake_claude_worker_record(issue_number, branch)

    return fake_launch


def test_reap_loop_leaves_a_foreign_pr_unprocessed(monkeypatch: Any, tmp_path: Path) -> None:
    """Issue #2822 review finding 3: in one reap-loop pass a PR bound to a
    foreign-owned issue receives no review, no merge evaluation and no
    janitor label activity, while an owned PR still flows to both."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        auto_merge=_approved_automerge(),
    )

    app, fake_gh = _loop_app(tmp_path / "routed", _routed(config))
    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        _approved_reviewer(app),
    )
    result = app.loop(merge=False)

    touched = {entry.get("pr") for entry in result.data["reviews"]} | {
        entry.get("pr") for entry in result.data["merges"]
    }
    assert touched == {456}
    assert result.data["open_tracked_prs"] == 1
    assert not any(number == 500 for number, _label in fake_gh.labels_added)
    assert not any(number == 500 for number, _label in fake_gh.labels_removed)

    # Positive control: routing off, both PRs are reviewed in the same pass.
    app, fake_gh = _loop_app(tmp_path / "routing-off", config)
    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        _approved_reviewer(app),
    )
    result = app.loop(merge=False)

    touched = {entry.get("pr") for entry in result.data["reviews"]} | {
        entry.get("pr") for entry in result.data["merges"]
    }
    assert 456 in touched and 457 in touched
