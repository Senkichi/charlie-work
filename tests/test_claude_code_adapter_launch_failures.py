"""Launch error paths: missing binary, worktree/prompt/fetch
failures, error-record teardown, and real-git retry/rework branch
delete-preserve semantics.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _fake_worktree,
    _install_fake_create_worktree,
)

from charlie_work import claude_code, layout
from charlie_work.claude_code import launch_claude_worker
from charlie_work.worktree import WorktreeForeignWriterError


def test_launch_claude_worker_missing_binary_returns_error_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        99,
        "agent/issue-99-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record.ok
    assert record.error is not None
    assert "failed to launch claude" in record.error
    assert record.pid is None

    # The worktree itself must not leak: remove_worktree was attempted (best
    # effort — a fake worktree isn't a real git worktree, so the git command
    # inside it fails, but that's covered separately below). The sidecar must
    # still be written with the error regardless.
    sidecar_path = sessions_dir / "issue-99.claude.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] == record.error


def test_launch_claude_worker_create_worktree_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    def failing_create_worktree(
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
        raise RuntimeError("git worktree add failed: branch already exists")

    monkeypatch.setattr(claude_code, "create_worktree", failing_create_worktree)

    record = launch_claude_worker(
        13,
        "agent/issue-13-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""
    assert record.pid is None

    sidecar_path = sessions_dir / "issue-13.claude.json"
    assert sidecar_path.exists()


def test_launch_claude_worker_foreign_writer_error_serializes_path_and_writes_clean_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for issue #1184.

    ``WorktreeForeignWriterError.worktree_path`` is a ``pathlib.Path``. If
    ``launch_claude_worker`` inserts it into the failure ``ClaudeWorkerRecord``
    without ``str()`` coercion, ``_write_json_atomic`` raises ``TypeError``
    mid-write, stranding a ``.tmp`` file and never producing a readable
    sidecar or a returned record.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    foreign_path = tmp_path / "foreign-wt"

    def foreign_writer_create_worktree(
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
        raise WorktreeForeignWriterError(
            worktree_path=foreign_path,
            pid=1234,
            session_id="abc",
        )

    monkeypatch.setattr(claude_code, "create_worktree", foreign_writer_create_worktree)

    # Must not raise: the shim converts this exception into a durable
    # error record.
    record = launch_claude_worker(
        14,
        "agent/issue-14-foreign",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.failure_kind == "worktree_foreign_writer"
    assert isinstance(record.worktree_path, str)
    assert record.worktree_path == str(foreign_path)

    sidecar_path = sessions_dir / "issue-14.claude.json"
    tmp_sidecar_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")

    assert sidecar_path.exists()
    assert not tmp_sidecar_path.exists()

    # json.loads must succeed cleanly — this is the assertion that would
    # fail (TypeError during json.dump, no file or a stranded .tmp instead)
    # if the str() coercion were reverted.
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "worktree_foreign_writer"
    assert payload["worktree_path"] == str(foreign_path)


def test_launch_claude_worker_fetch_failure_yields_error_record_not_exception(
    tmp_path: Path,
) -> None:
    """End-to-end test: real fetch failure inside create_worktree must return error record.

    This test forces a real git fetch failure by creating a repo with a broken origin URL,
    then calling launch_claude_worker with base_ref that triggers a fetch. The adapter must
    catch the RuntimeError from create_worktree and return an error record, not raise.

    This is a mutation gate: if RuntimeError is removed from the except tuple in
    launch_claude_worker, this test will fail with an uncaught exception.
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

    # Break the origin remote to simulate a fetch failure
    subprocess.run(
        ["git", "remote", "set-url", "origin", "file:///nonexistent/path"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Call launch_claude_worker with base_ref that triggers a fetch
    # Empty string resolves to origin/main, which will trigger the fetch in create_worktree
    record = launch_claude_worker(
        142,
        "agent/issue-142-fetch-failure",
        "do the thing",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        base_ref="",  # Triggers fetch of origin/main
    )

    # The adapter must catch the RuntimeError and return an error record
    assert not record.ok
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""
    assert record.pid is None

    # Verify the sidecar was written with the error
    sidecar_path = sessions_dir / "issue-142.claude.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert "worktree creation failed" in payload["error"]


def test_launch_claude_worker_remove_worktree_called_on_launch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    removed: list[Path] = []

    def fake_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        removed.append(worktree_path)
        return True

    monkeypatch.setattr(claude_code, "remove_worktree", fake_remove_worktree)

    record = launch_claude_worker(
        21,
        "agent/issue-21-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record.ok
    assert len(removed) == 1
    assert removed[0] == Path(record.worktree_path)


def test_launch_claude_worker_render_error_returns_error_record_and_tears_down_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)
        return True

    monkeypatch.setattr(
        claude_code,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(claude_code, "remove_worktree", tracking_remove_worktree)

    # Template with an unknown placeholder that bypasses load validation
    record = launch_claude_worker(
        1,
        "agent/issue-1",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("echo", "{unknown_placeholder}"),
    )

    # Must return an error record, not raise
    assert not record.ok
    assert record.error is not None
    assert "command template rendering failed" in record.error
    assert record.pid is None

    # Worktree must have been torn down
    assert len(worktree_removed) == 1

    # Sidecar must record the error
    sidecar_path = sessions_dir / "issue-1.claude.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert payload["pid"] is None


def test_launch_claude_worker_prompt_write_failure_tears_down_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If writing the prompt file fails, the worktree is torn down."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)
        return True

    monkeypatch.setattr(
        claude_code,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(claude_code, "remove_worktree", tracking_remove_worktree)

    # Monkeypatch Path.write_text to raise OSError on the prompt file
    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    record = launch_claude_worker(
        1,
        "agent/issue-1",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error
    assert len(worktree_removed) == 1


def test_launch_claude_worker_tmp_dir_creation_failure_tears_down_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the session-scoped temp dir cannot be created, launch_claude_worker
    must return an error record and tear down the worktree, never raise
    (issue #1767)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)
        return True

    monkeypatch.setattr(
        claude_code,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1767"),
    )
    monkeypatch.setattr(claude_code, "remove_worktree", tracking_remove_worktree)

    # Fail only the new session-tmp-dir mkdir -- this is a mutation gate: if
    # sanitize_env's new mkdir call is ever wrapped in a swallowing
    # try/except instead of letting it raise, this test fails because the
    # launch would then "succeed" with no worker environment isolation.
    original_mkdir = Path.mkdir

    def failing_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if self.name == layout.WORKER_TMP_DIRNAME:
            raise OSError("Mock tmp dir creation failure")
        return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    record = launch_claude_worker(
        1767,
        "agent/issue-1767",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.error is not None
    assert "failed to prepare worker environment" in record.error
    assert record.pid is None
    assert len(worktree_removed) == 1


# ---------------------------------------------------------------------------
# Real-git integration tests: branch delete/preserve semantics
# ---------------------------------------------------------------------------


def test_launch_failure_then_retry_succeeds(tmp_path: Path) -> None:
    """Launch failure should clean up branch and worktree, allowing retry to succeed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

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
    record1 = launch_claude_worker(
        42,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record1.ok
    assert "failed to launch claude" in record1.error

    # Verify the branch is deleted after the failure
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout

    # Second launch should succeed (using fake claude script)
    record2 = launch_claude_worker(
        42,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record2.ok
    assert record2.branch == branch_name


def test_rework_launch_failure_preserves_branch(tmp_path: Path) -> None:
    """Rework-mode launch failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

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

    branch_name = "agent/issue-43-rework"

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
    record = launch_claude_worker(
        43,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert not record.ok
    assert "failed to launch claude" in record.error

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


def test_rework_prompt_write_failure_preserves_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rework-mode prompt-write failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

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

    branch_name = "agent/issue-44-rework-prompt-fail"

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

    # Monkeypatch Path.write_text to raise OSError on the prompt file
    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    # Rework-mode launch fails on prompt write
    record = launch_claude_worker(
        44,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error

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
