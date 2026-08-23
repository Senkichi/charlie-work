"""Regression coverage for W4 (rework-conflict.md Finding 5, HYPOTHESIS).

Finding 5 flagged an "unresolved contradiction": per adapters.py's result
construction, ``SessionDispatchResult.failure_kind`` should propagate a
deterministic launch failure (e.g. ``rework_branch_conflict``) all the way out
of ``dispatch_sessions()`` -- but four consecutive identical worktree-conflict
failures for issue #533 (03:14-03:35Z) were observed with no escalation event.

Verification (see PR #550, commit 7f980b4, landed ~05:22Z the same day --
AFTER the observed failures): the actual bug was that
``workflow.py:_dispatch_rework_impl``'s failure branch did not consult
``failure_kind`` at all before that commit; it only checked the redispatch
count. ``git show 7f980b4 -- src/charlie_work/workflow.py`` confirms the
``failed_result = next(...)`` / ``terminal_failure = failure_kind in
DETERMINISTIC_ESCALATION_FAILURE_KINDS`` check at workflow.py:9548-9557 was
*introduced* by that commit -- it did not exist when issue #533's failures
were observed. workflow.py is out of scope for this file (owned by other
implementers) and already has App-level coverage for its consumption of
``failure_kind`` (test_charlie_work.py::
test_dispatch_rework_deterministic_failure_kind_escalates_immediately), but
that test mocks ``dispatch_sessions`` entirely and so never exercises the
real adapters.py/claude_code.py/devin_shell.py plumbing this file covers.

These tests close that gap at the correct layer: they drive the *real*
``dispatch_sessions()`` in adapters.py against a real ``ReworkBranchConflictError``
/ ``WorktreeUnsafeError`` raised from a patched ``create_worktree``, and assert
the resulting ``SessionDispatchResult.failure_kind`` survives unchanged --
proving there is no drop/rename/reordering bug in adapters.py or the
claude-code/devin-shell adapters it dispatches to for any
``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` member reachable at launch time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work import claude_code, devin_shell
from charlie_work.adapters import AdapterSettings, SessionRequest, dispatch_sessions
from charlie_work.config import DETERMINISTIC_ESCALATION_FAILURE_KINDS
from charlie_work.worktree import ReworkBranchConflictError, WorktreeUnsafeError


def _make_request(tmp_path: Path, issue_number: int, *, rework: bool = True) -> SessionRequest:
    prompt_path = tmp_path / f"prompt-{issue_number}.md"
    prompt_path.write_text(f"Fix issue #{issue_number}", encoding="utf-8")
    return SessionRequest(
        issue_number=issue_number,
        issue_title=f"issue {issue_number}",
        prompt_path=prompt_path,
        branch_name=f"agent/issue-{issue_number}-conflict",
        rework=rework,
    )


def _fake_create_worktree_raising(exc_factory):
    """Build a fake create_worktree matching the real signature that always raises."""

    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        raise exc_factory()

    return fake_create_worktree


def _conflict_error() -> ReworkBranchConflictError:
    return ReworkBranchConflictError(
        worktree_path=Path("/worktrees/agent-issue-533-conflict"),
        branch="agent/issue-533-conflict",
        base_ref="origin/main",
        conflicted_paths=("src/charlie_work/workflow.py",),
    )


def test_dispatch_sessions_claude_code_propagates_rework_branch_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact shape from Finding 5: a ReworkBranchConflictError raised inside
    create_worktree during a claude-code rework launch must surface as
    failure_kind="rework_branch_conflict" on the SessionDispatchResult returned
    by dispatch_sessions(), unchanged."""
    monkeypatch.setattr(
        claude_code, "create_worktree", _fake_create_worktree_raising(_conflict_error)
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    request = _make_request(tmp_path, 533)
    settings = AdapterSettings(adapter="claude-code", sessions_dir=tmp_path / "sessions")

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        [request],
    )

    assert len(results) == 1
    result = results[0]
    assert result.ok is False
    assert result.issue_number == 533
    assert result.failure_kind == "rework_branch_conflict"
    assert result.failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS

    # The on-disk results.json (read by other callers/tooling) must carry the
    # same failure_kind -- write_session_results must not lose it either.
    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert on_disk["results"][0]["failure_kind"] == "rework_branch_conflict"


def test_dispatch_sessions_claude_code_propagates_worktree_unsafe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Generality check (per coordination note): any OTHER member of
    DETERMINISTIC_ESCALATION_FAILURE_KINDS must propagate the same way --
    the fix/verification must not be conflict-specific."""

    def _unsafe_error() -> WorktreeUnsafeError:
        return WorktreeUnsafeError("worktree has uncommitted modifications")

    monkeypatch.setattr(
        claude_code, "create_worktree", _fake_create_worktree_raising(_unsafe_error)
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    request = _make_request(tmp_path, 601, rework=False)
    settings = AdapterSettings(adapter="claude-code", sessions_dir=tmp_path / "sessions")

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        [request],
    )

    assert results[0].ok is False
    # Issue #807: ``WorktreeUnsafeError`` now carries a discriminator (shim
    # dirt vs local commits) derived from the reason string. "worktree has
    # uncommitted modifications" is the actual raise-site string for shim
    # dirt, so it classifies as the mechanical kind that stays in
    # DETERMINISTIC_ESCALATION_FAILURE_KINDS. (An unrecognized reason string
    # now fails closed to ``worktree_unsafe_local_commits`` instead — see
    # test_worktree_unsafe_kind_fallback_is_local_commits_not_shim_dirt.)
    assert results[0].failure_kind == "worktree_unsafe_shim_dirt"


def test_dispatch_sessions_devin_shell_propagates_rework_branch_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same shape, sibling adapter: devin-shell must not drop failure_kind either.

    This also confirms the adapters.py:_run_devin_shell_adapter outer
    `except Exception` (which builds a result with NO failure_kind) is never
    reached for this exception -- ReworkBranchConflictError is a RuntimeError
    subclass, caught by devin_shell.launch_devin_session's own internal
    (OSError, SubprocessError, ValueError, RuntimeError) handler first.
    """
    monkeypatch.setattr(
        devin_shell, "create_worktree", _fake_create_worktree_raising(_conflict_error)
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    request = _make_request(tmp_path, 533)
    settings = AdapterSettings(adapter="devin-shell", sessions_dir=tmp_path / "sessions")

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        [request],
    )

    assert results[0].ok is False
    assert results[0].failure_kind == "rework_branch_conflict"


def test_dispatch_sessions_preserves_per_request_failure_kind_in_mixed_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch with one failing and one succeeding request must not smear the
    failing request's failure_kind onto the successful one (or vice versa) --
    covers an ordering/matching bug as an alternative shape to a straight drop."""
    real_create_worktree = claude_code.create_worktree

    def selective_create_worktree(repo_root, branch, **kwargs):
        if kwargs.get("issue_number") == 533:
            raise _conflict_error()
        # Real worktree creation for the "healthy" issue keeps this test honest
        # about matching-by-issue_number rather than matching-by-position.
        return real_create_worktree(repo_root, branch, **kwargs)

    monkeypatch.setattr(claude_code, "create_worktree", selective_create_worktree)

    import subprocess

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t.com",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "x",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    failing_request = _make_request(tmp_path, 533)
    healthy_request = _make_request(tmp_path, 700, rework=False)
    settings = AdapterSettings(
        adapter="claude-code",
        sessions_dir=tmp_path / "sessions",
        worktrees_dir=tmp_path / "worktrees",
        base_ref="HEAD",
        claude_command=("python", "-c", "print('ok')"),
    )

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        [failing_request, healthy_request],
    )

    by_issue = {r.issue_number: r for r in results}
    assert by_issue[533].ok is False
    assert by_issue[533].failure_kind == "rework_branch_conflict"
    assert by_issue[700].failure_kind is None
