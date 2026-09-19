"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``_decide``: the Bash and ``mcp__github__merge_pull_request``
decision paths -- fleet-repo merges gate on ``_run_merge_check``,
out-of-fleet merges pass through, and ambiguous PR numbers deny
without running the check.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

# ---------------------------------------------------------------------------
# _decide -- Bash
# ---------------------------------------------------------------------------


def test_decide_bash_fleet_pr_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 42 --squash"}, tmp_path)
    assert reason is not None
    assert "not_approved" in reason
    assert "#42" in reason


def test_decide_bash_fleet_pr_allowed_on_passing_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 42 --squash"}, tmp_path)
    assert reason is None


def test_decide_bash_out_of_fleet_repo_skips_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide("Bash", {"command": "gh pr merge -R other/repo 42 --squash"}, tmp_path)
    assert reason is None
    assert calls == []


def test_decide_bash_no_pr_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(
        hook, "_run_merge_check", lambda repo_root, pr: (True, "should not be called")
    )
    reason = hook._decide("Bash", {"command": "gh pr merge --squash"}, root)
    assert reason is not None
    assert "o/repo" in reason


def test_decide_bash_gh_repo_env_override_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer-mandated bypass: GH_REPO=owner/repo was invisible to the
    # old parser, so a merge into a fleet repo from a cwd outside every fleet
    # root sailed through undecided. cwd is deliberately outside the fleet
    # root to prove the target came from GH_REPO, not from cwd resolution.
    outside = tmp_path / "outside"
    outside.mkdir()
    fleet_root = tmp_path / "repo"
    fleet_root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"senkichi/charlie-work": fleet_root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide(
        "Bash",
        {"command": "GH_REPO=senkichi/charlie-work gh pr merge 5"},
        outside,
    )
    assert reason is not None
    assert "not_approved" in reason
    assert "#5" in reason


def test_decide_bash_ambiguous_pr_denies_without_running_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})

    def _must_not_run(repo_root: Path, pr: int) -> tuple[bool, str]:
        raise AssertionError("merge-check must not run when the PR number is ambiguous")

    monkeypatch.setattr(hook, "_run_merge_check", _must_not_run)
    reason = hook._decide("Bash", {"command": "gh pr merge -x 42 1195"}, root)
    assert reason is not None
    assert "o/repo" in reason


# ---------------------------------------------------------------------------
# _decide -- MCP merge_pull_request
# ---------------------------------------------------------------------------


def test_decide_mcp_fleet_repo_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "head_moved"))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is not None
    assert "head_moved" in reason
    assert "#7" in reason


def test_decide_mcp_fleet_repo_allowed_on_passing_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is None


def test_decide_mcp_non_fleet_owner_repo_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "other", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is None
    assert calls == []


def test_decide_mcp_missing_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo"},
        tmp_path,
    )
    assert reason is not None


def test_decide_mcp_non_int_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": "not-an-int"},
        tmp_path,
    )
    assert reason is not None


def test_decide_mcp_bool_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bool is a subclass of int; pullNumber=true must not preflight PR #1.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/r": tmp_path})
    called: list[Any] = []
    monkeypatch.setattr(hook, "_run_merge_check", lambda *a: called.append(a) or (True, ""))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "r", "pullNumber": True},
        tmp_path,
    )
    assert reason is not None
    assert "cannot determine PR number" in reason
    assert called == []
