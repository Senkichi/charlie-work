"""Issue #2005: cross-repo native blockers must not alias onto same-numbered local issues.

Drives the real ``GitHub`` client with ``subprocess.run`` faked, so the whole
path (REST blocked_by parse -> ``are_issues_open`` -> ``_get_open_blockers_for_issue``)
is exercised. Payload shape follows GitHub's REST ``blocked_by`` response
(issue objects carrying ``repository_url``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module
from charlie_work.backlog_reachability import _get_open_blockers_for_issue
from charlie_work.blocker_cycles import declared_blockers_by_issue
from charlie_work.github import get_github_issue_dependencies
from charlie_work.github_capabilities.cross_repo_blockers import CrossRepoBlocker

LOCAL = "Senkichi/jobcannon"
FOREIGN = "Senkichi/fresh-eyes"
API = "https://api.github.com/repos"


def _install(monkeypatch, tmp_path: Path, *, blocked_by: list[dict], states: dict[str, dict]):
    """Fake git remote + gh api. ``states`` maps an api path to its JSON (or None=fail)."""
    calls: list[str] = []

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"https://github.com/{LOCAL}.git\n", stderr=""
            )
        path = command[2]
        calls.append(path)
        if path.endswith("/dependencies/blocked_by"):
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(blocked_by), stderr=""
            )
        if command[2] == "graphql":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        payload = states.get(path)
        if payload is None:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="HTTP 500")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    return github_module.GitHub(repo_root=tmp_path), calls


def _foreign_dep(number: int = 60) -> dict:
    return {"number": number, "repository_url": f"{API}/{FOREIGN}", "state": "open"}


def _local_states(**kw: str) -> dict[str, dict]:
    return {f"repos/{LOCAL}/issues/{n}": {"state": s} for n, s in kw.items()}


def test_open_foreign_blocker_blocks_even_if_same_number_local_is_closed(monkeypatch, tmp_path):
    # Only the foreign repo's #60 is open. Every other lookup (graphql, and the
    # per-issue ``issue view`` of local #60 -- the old bug's target) resolves
    # as failed/closed in this fake, so a bare-number resolution reads closed.
    states = {f"repos/{FOREIGN.lower()}/issues/60": {"state": "open"}}
    gh, _ = _install(monkeypatch, tmp_path, blocked_by=[_foreign_dep()], states=states)
    assert gh.are_issues_open([60]) == set()

    deps = get_github_issue_dependencies(gh, 459)
    assert deps == [CrossRepoBlocker(60, FOREIGN)]
    assert deps != [60]

    declared, open_blockers = _get_open_blockers_for_issue(gh, {"number": 459, "body": ""})
    assert len(open_blockers) == 1
    assert isinstance(open_blockers[0], CrossRepoBlocker)
    assert declared == open_blockers


def test_closed_foreign_blocker_unblocks(monkeypatch, tmp_path):
    states = {f"repos/{FOREIGN.lower()}/issues/60": {"state": "closed"}}
    gh, _ = _install(monkeypatch, tmp_path, blocked_by=[_foreign_dep()], states=states)

    declared, open_blockers = _get_open_blockers_for_issue(gh, {"number": 459, "body": ""})
    assert len(declared) == 1
    assert open_blockers == []


def test_foreign_blocker_lookup_failure_fails_closed(monkeypatch, tmp_path):
    gh, _ = _install(monkeypatch, tmp_path, blocked_by=[_foreign_dep()], states={})

    _declared, open_blockers = _get_open_blockers_for_issue(gh, {"number": 459, "body": ""})
    assert len(open_blockers) == 1


def test_foreign_state_lookup_targets_foreign_repo_and_is_cached(monkeypatch, tmp_path):
    states = {f"repos/{FOREIGN.lower()}/issues/60": {"state": "open"}}
    gh, calls = _install(monkeypatch, tmp_path, blocked_by=[_foreign_dep()], states=states)

    blocker = get_github_issue_dependencies(gh, 459)[0]
    assert gh.are_issues_open([blocker]) == {blocker}
    assert gh.are_issues_open([blocker]) == {blocker}
    assert calls.count(f"repos/{FOREIGN.lower()}/issues/60") == 1


def test_same_repo_dependency_unchanged(monkeypatch, tmp_path):
    same = {"number": 7, "repository_url": f"{API}/{LOCAL}"}
    gh, _ = _install(monkeypatch, tmp_path, blocked_by=[same, {"number": 8}], states={})

    deps = get_github_issue_dependencies(gh, 459)
    assert deps == [7, 8]
    assert not any(isinstance(d, CrossRepoBlocker) for d in deps)


def test_same_repo_repository_url_is_case_insensitive(monkeypatch, tmp_path):
    same = {"number": 7, "repository_url": f"{API}/{LOCAL.lower()}"}
    gh, _ = _install(monkeypatch, tmp_path, blocked_by=[same], states={})

    deps = get_github_issue_dependencies(gh, 459)
    assert deps == [7]
    assert not isinstance(deps[0], CrossRepoBlocker)


def test_cross_repo_blocker_does_not_collide_with_local_number():
    foreign = CrossRepoBlocker(60, FOREIGN)
    assert foreign != 60
    assert 60 != foreign
    assert len({foreign, 60}) == 2
    assert sorted([70, foreign, 5]) == [5, foreign, 70]


def test_blocker_cycles_excludes_cross_repo_blockers():
    class Gh:
        def issue_dependencies(self, numbers):
            return {459: [CrossRepoBlocker(459, FOREIGN), 12]}

    declared = declared_blockers_by_issue(Gh(), [{"number": 459, "body": ""}])
    # A foreign #459 must not become a self-loop on the local #459.
    assert declared == {459: {12}}


@pytest.mark.parametrize("url", [None, "", "https://x/y", 5])
def test_repo_from_repository_url_rejects_malformed(url):
    from charlie_work.github_capabilities.cross_repo_blockers import repo_from_repository_url

    assert repo_from_repository_url(url) is None
