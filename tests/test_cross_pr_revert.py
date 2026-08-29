"""Unit tests for ``charlie_work.cross_pr_revert`` (issue #1068).

The cross-PR base-revert detection gate was verbatim-moved out of the
over-cap ``src/charlie_work/janitor.py`` monolith into
``src/charlie_work/cross_pr_revert.py`` so the fail-closed rework did not
land new code in an over-cap monolith (file-size ratchet, issue #1442).
These are the gate's behavioral unit tests, moved with the function so the
monkeypatch target (``cross_pr_revert.subprocess``) and the warning-log
logger name (``charlie_work.cross_pr_revert``) resolve against the module
that actually owns the code.

The merge-ready integration coverage for the ``UNDETERMINED`` fail-closed
path lives in ``tests/test_charlie_work.py`` alongside the other
``merge_ready`` integration tests (it depends on that file's
``FakeGitHub`` / ``OrchestratorApp`` / ``_init_cross_pr_revert_repo``
fixtures).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from charlie_work.cross_pr_revert import (
    CrossPrRevertStatus,
    detect_cross_pr_revert,
)


def _green_pr(**overrides) -> dict:
    """Minimal green-PR dict for the cross-PR revert gate.

    ``detect_cross_pr_revert`` only reads ``body``, ``headRefName`` and
    ``baseRefName``; the remaining fields mirror ``tests/test_janitor.py``'s
    ``_green_pr`` shape for consistency.
    """
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


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _make_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    return repo_root


# --------------------------------------------------------------------------
# ref validation (issue #659): invalid refs fail closed, never reach argv
# --------------------------------------------------------------------------


def test_detect_cross_pr_revert_skips_invalid_head_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag-like headRefName must not reach ``git fetch origin <head> <base>``.

    Ref validation fails before any subprocess call, so the gate is
    UNDETERMINED (fail-closed), not CLEAN — the unverifiable state must not
    fold into the same return as "verified clean" (issue #1068).
    """
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    pr = _green_pr(headRefName="--upload-pack=evil")

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called with invalid refs")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fail_if_called)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True


def test_detect_cross_pr_revert_skips_invalid_base_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag-like baseRefName must not reach ``git fetch origin <head> <base>``.

    Ref validation fails before any subprocess call, so the gate is
    UNDETERMINED (fail-closed), not CLEAN (issue #1068).
    """
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    pr = _green_pr(headRefName="agent/issue-1-fix", baseRefName="--upload-pack=evil")

    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.run should not be called with invalid refs")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fail_if_called)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True


def test_detect_cross_pr_revert_warns_on_invalid_ref(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ref validation failure must be logged and fail closed, not silently pass.

    The gate returns UNDETERMINED with a diagnostic reason and emits a warning
    log, so the unverifiable state is distinguishable from "verified clean"
    (issue #1068).
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    pr = _green_pr(headRefName="--upload-pack=evil")

    with caplog.at_level(logging.WARNING, logger="charlie_work.cross_pr_revert"):
        result = detect_cross_pr_revert(pr, repo_root)

    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "ref validation failed" in result.reason
    assert any(
        "detect_cross_pr_revert" in record.message and "not a valid git ref name" in record.message
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# detect_cross_pr_revert explicit verdict states (issue #1068)
#
# The gate used to return None for both "verified clean" and "could not
# verify", silently disabling the merge gate on any transient local-git
# failure. It now returns a CrossPrRevertResult with an explicit status so
# the caller can fail closed on UNDETERMINED instead of advancing toward
# merge on an unverified gate.
# --------------------------------------------------------------------------


def test_detect_cross_pr_revert_fetch_failure_is_undetermined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``git fetch`` non-zero exit is UNDETERMINED, not CLEAN (issue #1068)."""
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        if argv[1] == "fetch":
            return _FakeCompleted(returncode=1)
        raise AssertionError(f"unexpected git call after fetch failure: {argv}")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "git fetch" in result.reason


def test_detect_cross_pr_revert_rev_list_failure_is_undetermined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``git rev-list`` non-zero exit is UNDETERMINED, not CLEAN (issue #1068)."""
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        if argv[1] == "fetch":
            return _FakeCompleted(returncode=0)
        if argv[1] == "rev-list":
            return _FakeCompleted(returncode=1)
        raise AssertionError(f"unexpected git call: {argv}")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "rev-list" in result.reason


def test_detect_cross_pr_revert_per_commit_log_failure_is_undetermined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero ``git log`` for a branch commit makes the gate UNDETERMINED.

    A per-commit subject-fetch failure used to ``continue`` and fall through to
    the final ``return None`` (CLEAN-equivalent). Now the incompletely-inspected
    gate is UNDETERMINED so a revert hidden in the uninspected commit cannot
    merge unflagged (issue #1068).
    """
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        if argv[1] == "fetch":
            return _FakeCompleted(returncode=0)
        if argv[1] == "rev-list":
            return _FakeCompleted(returncode=0, stdout="deadbeef\n")
        if argv[1] == "log" and argv[2] == "-1":
            # Per-commit subject fetch fails.
            return _FakeCompleted(returncode=1)
        raise AssertionError(f"unexpected git call: {argv}")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "could not inspect" in result.reason


def test_detect_cross_pr_revert_oserror_is_undetermined_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An OSError during git ops is UNDETERMINED + logged, not a silent None.

    The OSError branch previously returned None with no logging at all, making
    a disk/IO failure indistinguishable from "verified clean" (issue #1068).
    """
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        raise OSError("disk hiccup")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    with caplog.at_level(logging.WARNING, logger="charlie_work.cross_pr_revert"):
        result = detect_cross_pr_revert(pr, repo_root)

    assert result.status is CrossPrRevertStatus.UNDETERMINED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "OS error" in result.reason
    assert any(
        "detect_cross_pr_revert" in record.message and "OS error" in record.message
        for record in caplog.records
    )


def test_detect_cross_pr_revert_no_repo_root_is_clean(tmp_path: Path) -> None:
    """No repo_root configured means the gate is structurally inapplicable — CLEAN.

    This is "gate not applicable", not "verified clean": in production
    ``repo_root`` is always a configured git checkout, so this branch is only
    reached in tests/edge configs and preserves the prior "gate skipped"
    behavior. The issue #1068 defect is transient git-operation failures, not
    structural inapplicability.
    """
    pr = _green_pr()
    result = detect_cross_pr_revert(pr, None)
    assert result.status is CrossPrRevertStatus.CLEAN
    assert result.blocks_merge is False


def test_detect_cross_pr_revert_not_a_git_repo_is_clean(tmp_path: Path) -> None:
    """A repo_root that is not a git repo is CLEAN (gate not applicable)."""
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    pr = _green_pr()
    result = detect_cross_pr_revert(pr, not_a_repo)
    assert result.status is CrossPrRevertStatus.CLEAN
    assert result.blocks_merge is False


def test_detect_cross_pr_revert_missing_refs_is_clean(tmp_path: Path) -> None:
    """Missing headRefName/baseRefName is CLEAN (gate not applicable)."""
    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr(headRefName=None, baseRefName=None)
    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.CLEAN
    assert result.blocks_merge is False


def test_detect_cross_pr_revert_clean_when_no_branch_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully-verified gate with no branch-only commits is CLEAN (issue #1068)."""
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        if argv[1] == "fetch":
            return _FakeCompleted(returncode=0)
        if argv[1] == "rev-list":
            return _FakeCompleted(returncode=0, stdout="")
        raise AssertionError(f"unexpected git call: {argv}")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.CLEAN
    assert result.blocks_merge is False
    assert result.reason is None


def test_detect_cross_pr_revert_clean_on_explicit_marker(tmp_path: Path) -> None:
    """An explicit ``allow-revert:`` marker line is CLEAN — the operator opt-out."""
    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr(body="Closes #123\n\nallow-revert: intentional revert of feature C")
    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.CLEAN
    assert result.blocks_merge is False
    assert result.reason is None


def test_detect_cross_pr_revert_detected_returns_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matched cross-PR revert yields REVERT_DETECTED with a blocking reason."""
    from charlie_work import cross_pr_revert as cross_pr_revert_module

    repo_root = _make_repo_root(tmp_path)
    pr = _green_pr()
    revert_sha = "cafebabe1234567890"
    base_sha = "feedface1234567890"

    def _fake_run(argv: list[str], **_kwargs: object) -> _FakeCompleted:
        if argv[1] == "fetch":
            return _FakeCompleted(returncode=0)
        if argv[1] == "rev-list":
            return _FakeCompleted(returncode=0, stdout=f"{revert_sha}\n")
        if (
            argv[1] == "log"
            and argv[2] == "-1"
            and argv[3] == "--format=%s"
            and argv[4] == revert_sha
        ):
            return _FakeCompleted(returncode=0, stdout='Revert "feature C"')
        if argv[1] == "log" and argv[3] == "--format=%H":
            # base-commit grep match
            return _FakeCompleted(returncode=0, stdout=f"{base_sha}\n")
        if argv[1] == "log" and argv[2] == "-1" and argv[4] == base_sha:
            return _FakeCompleted(returncode=0, stdout="feature C")
        raise AssertionError(f"unexpected git call: {argv}")

    monkeypatch.setattr(cross_pr_revert_module.subprocess, "run", _fake_run)

    result = detect_cross_pr_revert(pr, repo_root)
    assert result.status is CrossPrRevertStatus.REVERT_DETECTED
    assert result.blocks_merge is True
    assert result.reason is not None
    assert "feature C" in result.reason
    assert "allow-revert" in result.reason
