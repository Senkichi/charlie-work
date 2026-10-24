"""Host routing in reconcile and the review/merge lane (spec C2, interface I-7).

Reconcile, janitor routing and rescue leave an issue another host owns, and the
PRs bound to it, alone. Each assertion has a positive control: the same fixture
with routing off (or owned here) still produces today's drift.

"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub as AppFakeGitHub
from _reconcile_fixtures import FakeGitHub, _issue, _pr
from _src_ast import source_text
from charlie_work.config import OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.host_routing import foreign_issue_numbers, local_issues, local_prs
from charlie_work.reconcile import detect_drift
from charlie_work.state import empty_state


def _config(host: str | None, *, default_host: bool = True) -> OrchestratorConfig:
    base = OrchestratorConfig()
    return replace(base, dispatch=replace(base.dispatch, host=host, default_host=default_host))


_REAP_LOOP = Path(__file__).resolve().parent.parent / "src/charlie_work/orchestration/reap_loop.py"


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


class _FailingIssues(AppFakeGitHub):
    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        raise GitHubError("boom")


def test_issue_fetch_failure_keeps_todays_lane_on_the_default_host_only() -> None:
    gh = _FailingIssues()
    prs = gh.pr_list()
    assert local_prs(_App(gh, _config("workstation")), prs) == prs
    assert local_prs(_App(gh, _config("macbook", default_host=False)), prs) == []


def test_review_and_merge_lane_reads_prs_through_host_routing() -> None:
    # reap_loop feeds every PR to janitor routing, rescue and the merge path;
    # filtering at its one fetch keeps all three host-local.
    wired = "prs = _wf.host_local_prs(self, deadline.call(self.gh.pr_list, []))"
    assert wired in source_text(_REAP_LOOP)
