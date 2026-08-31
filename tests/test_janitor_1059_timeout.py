"""Issue #1059 regression tests: unbounded subprocess.run -> run_captured.

These tests verify that the janitor's git-probe call sites route through
``run_captured`` (which enforces a bounded ``timeout_seconds``) rather than
calling ``subprocess.run`` directly.  The approach patches the global
``subprocess.run`` to raise ``TimeoutExpired``.  Against the fixed code,
``run_captured`` catches ``TimeoutExpired`` internally and returns a
``RunResult`` -- the calling function handles it as a value.  Against the
unfixed code (raw ``subprocess.run`` without timeout), ``TimeoutExpired``
is NOT in the calling function's ``except`` clause and propagates -- the
test fails, proving it exercises the fix.

``detect_cross_pr_revert`` was extracted to ``charlie_work.cross_pr_revert``
(issue #1068 rework, file-size ratchet #1442); its timeout test imports from
the owning module and asserts the fail-closed ``UNDETERMINED`` verdict rather
than the pre-#1068 ``None`` return.

Extracted from ``tests/test_janitor.py`` to satisfy the file-size ratchet
(issue #1442): the monolith was already over the 800-line cap, so the new
regression tests land here instead.  Helpers (``_init_repo``, ``_green_pr``)
are inlined rather than cross-imported per the #1284 self-containment rule
enforced by ``tests/test_zero_cross_test_import_guard.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from charlie_work.cross_pr_revert import (
    CrossPrRevertStatus,
    detect_cross_pr_revert,
)
from charlie_work.janitor import (
    _get_unpushed_commit_info,
    check_operator_containment,
)


# ---------------------------------------------------------------------------
# Inlined helpers (duplicated from test_janitor.py per #1284 self-containment)
# ---------------------------------------------------------------------------


def _init_repo(repo_root: Path) -> None:
    """Initialize a git repo with a single commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def _green_pr(**overrides) -> dict:
    base = {
        "number": 456,
        "title": "fix: search is broken",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": "abc123",
        "baseRefName": "main",
        "body": "Closes #123.\n\nTests: added unit tests for the search path.",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "additions": 10,
        "deletions": 5,
        "isCrossRepository": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def _timeout_subprocess_run(*_args: object, **_kwargs: object):
    """Raise ``TimeoutExpired`` to simulate a hung subprocess."""
    raise subprocess.TimeoutExpired(["git"], 60)


def test_check_no_op_rework_timeout_returns_false_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out git fetch/rev-list must return False with a warning, not hang."""
    from charlie_work import janitor as janitor_module

    repo = tmp_path / "repo"
    _init_repo(repo)

    pr = _green_pr(
        headRefOid="def456",
        headRefName="agent/issue-123-fix",
        baseRefName="main",
    )
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }

    # Patch AFTER _init_repo so the repo setup is not affected.
    monkeypatch.setattr("subprocess.run", _timeout_subprocess_run)

    failures: list[str] = []
    warnings: list[str] = []
    result = janitor_module._check_no_op_rework(
        pr,
        pr_state,
        failures,
        warnings,
        repo,
        pr_diff=None,
        review_decision={"decision": "request_changes"},
    )

    assert result is False
    assert any("Could not verify whether PR head advance" in w for w in warnings)


def test_detect_cross_pr_revert_timeout_returns_undetermined_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out git fetch must return UNDETERMINED (fail-closed), not hang.

    ``detect_cross_pr_revert`` was extracted to ``charlie_work.cross_pr_revert``
    (issue #1068) and returns a ``CrossPrRevertResult`` instead of ``str | None``.
    A timeout is a transient git-operation failure, so the gate fails closed
    with ``UNDETERMINED`` rather than folding into ``CLEAN`` (issue #1068).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    pr = _green_pr(headRefName="agent/issue-1-fix", baseRefName="main")

    monkeypatch.setattr("subprocess.run", _timeout_subprocess_run)

    result = detect_cross_pr_revert(pr, repo)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True


def test_check_operator_containment_timeout_returns_empty_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out git status must return an empty tuple, not hang."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    monkeypatch.setattr("subprocess.run", _timeout_subprocess_run)

    result = check_operator_containment(repo, "diff --git a/f b/f\n", 999)
    assert result == ()


def test_get_unpushed_commit_info_timeout_returns_none_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out git worktree list must return None, not hang."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    monkeypatch.setattr("subprocess.run", _timeout_subprocess_run)

    result = _get_unpushed_commit_info("agent/issue-1-fix", repo)
    assert result is None
