"""Core ``launch_claude_worker`` mechanics: prompt/sidecar writes,
worktree handling, rework/recovery modes, and sidecar naming.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
import pytest

from _claude_adapter_fixtures import (
    _fake_claude_script,
    _fake_worktree,
    _install_fake_create_worktree,
)
from _worker_marker_wait import read_worker_marker

from charlie_work import claude_code
from charlie_work.claude_code import (
    launch_claude_worker,
    read_worker_records,
)


def test_launch_claude_worker_writes_prompt_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    calls: list[dict] = []
    _install_fake_create_worktree(monkeypatch, tmp_path, calls=calls)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert record.error is None
    assert record.issue_number == 42
    assert record.branch == "agent/issue-42-fix"
    assert record.pid is not None
    assert record.started_at.endswith("Z")

    worktree_path = Path(record.worktree_path)
    prompt_path = worktree_path / ".orchestrator-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "Do the thing."
    assert record.prompt_path == str(prompt_path)

    # create_worktree got the right args, including venv_source/worktrees_dir passthrough.
    assert calls[0]["branch"] == "agent/issue-42-fix"
    assert calls[0]["repo_root"] == repo_root

    # Sidecar JSON is present and matches the returned record.
    sidecar_path = sessions_dir / "issue-42.claude.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["issue_number"] == 42
    assert payload["branch"] == "agent/issue-42-fix"
    assert payload["error"] is None

    log_path = Path(record.log_path)
    assert log_path == sessions_dir / "issue-42.claude.log"


def test_launch_claude_worker_process_receives_prompt_via_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        7,
        "agent/issue-7-x",
        "prompt payload for stdin",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    worktree_path = Path(record.worktree_path)

    # Wait for the fake claude subprocess (very fast: reads stdin, writes, exits).
    marker_path = worktree_path / "worker-ran.txt"
    read_worker_marker(marker_path, expected="prompt payload for stdin")


def test_launch_claude_worker_records_default_xdist_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #646: with no ambient/config cap, the sidecar records the safe default."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("UV_NO_SYNC", raising=False)

    record = launch_claude_worker(
        141,
        "agent/issue-141-default-cap",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert record.xdist_cap == "2"
    assert record.uv_no_sync is None  # no .venv in this fake worktree

    payload = json.loads((sessions_dir / "issue-141.claude.json").read_text(encoding="utf-8"))
    assert payload["xdist_cap"] == "2"
    assert payload["uv_no_sync"] is None


def test_launch_claude_worker_prompt_path_placeholder_skips_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    script_path = tmp_path / "fake_claude_argv.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            prompt_path = Path(sys.argv[1])
            Path("worker-ran.txt").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
            print("ok")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        8,
        "agent/issue-8-x",
        "prompt payload for argv",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path), "{prompt_path}"),
    )

    assert record.ok
    assert record.prompt_path in record.command

    worktree_path = Path(record.worktree_path)
    marker_path = worktree_path / "worker-ran.txt"
    read_worker_marker(marker_path, expected="prompt payload for argv")


def test_launch_claude_worker_resolves_argv0_through_resolve_cli_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #487: a bare ``"claude"`` on Windows is an npm ``.CMD`` shim that
    ``Popen(shell=False)`` cannot find (WinError 2). ``launch_claude_worker``
    must resolve argv[0] through ``resolve_cli_binary`` before spawning —
    exercised here by stubbing the resolver to swap in a real Python script
    standing in for the resolved ``.exe``, and asserting the recorded
    ``command`` reflects the resolved binary, not the original template
    token."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    fake_script = _fake_claude_script(tmp_path)
    resolved_calls: list[str] = []

    def fake_resolve_cli_binary(name: str) -> str:
        resolved_calls.append(name)
        assert name == "claude"
        return fake_script[0]

    monkeypatch.setattr(claude_code, "resolve_cli_binary", fake_resolve_cli_binary)

    record = launch_claude_worker(
        55,
        "agent/issue-55-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("claude", str(fake_script[1])),
    )

    assert record.ok
    assert resolved_calls == ["claude"]
    assert record.command[0] == fake_script[0]
    assert record.command[0] != "claude"


def test_launch_claude_worker_with_venv_source_junction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When venv_source is configured, the junction is copied into the worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake venv source
    venv_source = tmp_path / "shared_venv"
    venv_source.mkdir()
    (venv_source / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    # Track the venv_source argument passed to create_worktree
    calls: list[dict] = []

    def tracking_create_worktree(
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
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        venv_source=venv_source,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["venv_source"] == venv_source


def test_launch_claude_worker_with_custom_worktrees_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When worktrees_dir is configured, it's passed to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    custom_worktrees_dir = tmp_path / "custom_worktrees"

    # Track the worktrees_dir argument passed to create_worktree
    calls: list[dict] = []

    def tracking_create_worktree(
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
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-custom-dir",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        worktrees_dir=custom_worktrees_dir,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["worktrees_dir"] == custom_worktrees_dir


def test_launch_claude_worker_rework_mode_reuses_existing_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In rework mode, create_worktree is called with rework=True."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    calls: list[dict] = []

    def tracking_create_worktree(
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
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-rework",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["rework"] is True


def test_launch_claude_worker_recovery_mode_passes_recovery_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In recovery mode, the recovery dict is passed to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    recovery_dict = {
        "worktree_path": "/tmp/wt/issue-42",
        "branch": "agent/issue-42",
        "has_commits": True,
    }

    calls: list[dict] = []

    def tracking_create_worktree(
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
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-recovery",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        recovery=recovery_dict,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["recovery"] == {
        **recovery_dict,
        "inconclusive_probe_deferred_count": 0,
    }


def test_launch_claude_worker_rework_log_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In rework mode, the log file uses the -rework suffix."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-rework",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert record.ok
    assert record.log_path.endswith("-rework.claude.log")


def test_launch_claude_worker_api_kind_sidecar_naming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An api-kind launch writes ``issue-<n>.api.json`` and records the kind/provider."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-api",
        "Do the api thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        adapter_kind="api",
        provider="openai",
    )

    assert record.ok
    assert record.adapter_kind == "api"
    assert record.provider == "openai"

    sidecar_path = sessions_dir / "issue-42.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["adapter_kind"] == "api"
    assert payload["provider"] == "openai"

    # Default read_worker_records filters to claude-code and must not return api records.
    assert read_worker_records(sessions_dir) == []

    # Reading with the api kind filter returns the record and preserves identity.
    api_records = read_worker_records(sessions_dir, adapter_kind="api")
    assert len(api_records) == 1
    assert api_records[0].adapter_kind == "api"
    assert api_records[0].provider == "openai"
