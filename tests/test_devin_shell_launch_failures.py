"""Failure-path tests for ``devin_shell.launch_devin_session``.

Split out of ``tests/test_devin_shell_launch.py`` (itself split out of
``tests/test_devin_shell.py``, issue #1542, Track-1 pilot): the
error-record paths -- missing binary, worktree-creation failure,
foreign-writer error, fetch failure, and render error -- plus
failure-then-retry branch cleanup and rework-launch failure branch
preservation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from _devin_shell_fixtures import (
    _FAKE_DEVIN_SLEEP,
    _fake_worktree,
    _install_fake_create_worktree,
    _write_fake_devin,
)

from charlie_work import devin_shell
from charlie_work.devin_shell import launch_devin_session
from charlie_work.worktree import WorktreeForeignWriterError


def test_launch_with_missing_binary_yields_error_record_not_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        7,
        "agent/issue-7-x",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(
            "definitely-not-a-real-devin-binary-xyz",
            "--prompt-file",
            "{prompt_path}",
        ),
    )

    assert record.pid is None
    assert record.error is not None
    assert record.issue_number == 7

    sidecar_path = sessions_dir / "issue-7.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None


def test_launch_worktree_creation_failure_yields_error_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If worktree creation fails, launch must return an error record, not raise."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    def failing_create_worktree(*args, **kwargs):
        raise RuntimeError("git worktree add failed: branch already exists")

    monkeypatch.setattr(devin_shell, "create_worktree", failing_create_worktree)

    record = launch_devin_session(
        8,
        "agent/issue-8-conflict",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert record.pid is None
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""

    payload = json.loads((sessions_dir / "issue-8.json").read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None


def test_launch_worktree_foreign_writer_error_serializes_path_and_writes_clean_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for issue #1184.

    ``WorktreeForeignWriterError.worktree_path`` is a ``pathlib.Path``. If the
    launch shim inserts it into the failure ``SessionRecord`` without
    ``str()`` coercion, ``json.dump`` raises ``TypeError`` mid-write inside
    ``_write_json``, stranding a ``.tmp`` file and never producing a readable
    sidecar or a returned record — the caller instead sees an uncaught
    exception, which the orchestrator logs as a generic launch failure and
    burns a rework-cap slot on.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    foreign_path = tmp_path / "foreign-wt"

    def foreign_writer_create_worktree(*args, **kwargs):
        raise WorktreeForeignWriterError(
            worktree_path=foreign_path,
            pid=1234,
            session_id="abc",
        )

    monkeypatch.setattr(devin_shell, "create_worktree", foreign_writer_create_worktree)

    # Must not raise: the whole point of the shim is to convert this
    # exception into a durable error record.
    record = launch_devin_session(
        9,
        "agent/issue-9-foreign",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert record.failure_kind == "worktree_foreign_writer"
    assert isinstance(record.worktree_path, str)
    assert record.worktree_path == str(foreign_path)

    sidecar_path = sessions_dir / "issue-9.json"
    tmp_sidecar_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")

    assert sidecar_path.exists()
    assert not tmp_sidecar_path.exists()

    # json.loads must succeed cleanly — this is the assertion that would
    # fail (TypeError during json.dump, no file or a stranded .tmp instead)
    # if the str() coercion were reverted.
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "worktree_foreign_writer"
    assert payload["worktree_path"] == str(foreign_path)


def test_launch_devin_session_fetch_failure_yields_error_record_not_exception(
    tmp_path: Path,
) -> None:
    """End-to-end test: real fetch failure inside create_worktree must return error record.

    This test forces a real git fetch failure by creating a repo with a broken origin URL,
    then calling launch_devin_session with base_ref that triggers a fetch. The adapter must
    catch the RuntimeError from create_worktree and return an error record, not raise.

    This is a mutation gate: if RuntimeError is removed from the except tuple in
    launch_devin_session, this test will fail with an uncaught exception.
    """
    # Create a real git repo with an origin remote
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )

    # Clone the remote repo to create a local repo with origin configured
    repo_root = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(repo_root)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
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

    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")

    # Break the origin remote to simulate a fetch failure
    subprocess.run(
        ["git", "remote", "set-url", "origin", "file:///nonexistent/path"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Call launch_devin_session with base_ref that triggers a fetch
    # Empty string resolves to origin/main, which will trigger the fetch in create_worktree
    record = launch_devin_session(
        142,
        "agent/issue-142-fetch-failure",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        base_ref="",  # Triggers fetch of origin/main
    )

    # The adapter must catch the RuntimeError and return an error record
    assert record.pid is None
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""

    # Verify the sidecar was written with the error
    sidecar_path = sessions_dir / "issue-142.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None
    assert "worktree creation failed" in payload["error"]


def test_launch_render_error_returns_error_record_and_tears_down_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)

    monkeypatch.setattr(
        devin_shell,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(devin_shell, "remove_worktree", tracking_remove_worktree)

    # Template with an unknown placeholder that bypasses load validation
    record = launch_devin_session(
        1,
        "agent/issue-1",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("echo", "{unknown_placeholder}"),
    )

    # Must return an error record, not raise
    assert record.error is not None
    assert "command template rendering failed" in record.error
    assert record.pid is None

    # Worktree must have been torn down
    assert len(worktree_removed) == 1

    # Sidecar must record the error
    sidecar_path = sessions_dir / "issue-1.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert payload["pid"] is None


def test_launch_failure_then_retry_succeeds(tmp_path: Path) -> None:
    """Launch failure should clean up branch and worktree, allowing retry to succeed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt text\n", encoding="utf-8")

    # Initialize a real git repo for the worktree to use
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
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
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    branch_name = "agent/issue-42-retry"

    # First launch fails (binary doesn't exist)
    record1 = launch_devin_session(
        42,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert record1.error is not None
    assert "failed to launch devin" in record1.error

    # Verify the branch is deleted after the failure
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout

    # Second launch should succeed (using fake devin script)
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)
    record2 = launch_devin_session(
        42,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )

    assert record2.error is None
    assert record2.branch == branch_name


def test_rework_launch_failure_preserves_branch(tmp_path: Path) -> None:
    """Rework-mode launch failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt text\n", encoding="utf-8")

    # Initialize a real git repo
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
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
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    branch_name = "agent/issue-44-rework"

    # Create the branch first (simulating a previous PR cycle)
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Rework-mode launch fails (binary doesn't exist)
    record = launch_devin_session(
        44,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert record.error is not None
    assert "failed to launch devin" in record.error

    # Verify the branch is preserved (not deleted)
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
