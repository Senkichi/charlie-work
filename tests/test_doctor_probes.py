"""Adapter probes and on-disk state artifacts surfaced by ``run_doctor``.

Covers the devin/claude adapter probes, post-mortem surfacing, and the
quarantine-file checks. Split out of ``tests/test_doctor.py`` (issue
#1563, Track 1 shoulder) -- bodies are verbatim relocations; shared
helpers live in ``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    DevinConfig,
    WorkerRoleConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from charlie_work.subprocess_runner import RunResult
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
    _write_sidecar,
)


def test_doctor_adapter_probe_runs_devin_probe_and_surfaces_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    # A failed launch (error set) and a launched-but-dead one (implausible PID
    # that OpenProcess/os.kill can never find -> is_session_alive False).
    _write_sidecar(
        sessions_dir,
        "issue-1.json",
        {
            "issue_number": 1,
            "branch": "agent/issue-1",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-1.log",
            "error": "devin not found",
        },
    )
    _write_sidecar(
        sessions_dir,
        "issue-2.json",
        {
            "issue_number": 2,
            "branch": "agent/issue-2",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": 999_999_999,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-2.log",
            "error": None,
        },
    )
    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["devin CLI probe"].ok is True
    assert "devin 1.2.3" in by_name["devin CLI probe"].detail
    sessions = by_name["launched sessions"]
    assert sessions.ok is False  # one failed record present
    assert "1 failed" in sessions.detail
    assert "1 exited" in sessions.detail
    assert sessions.severity == "warning"  # never blocks the run
    assert ok is True  # a warning-only sessions finding must not fail doctor


def test_doctor_surfaces_post_mortem_terminal_cause_and_attempt_ref(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #261 F6: a dead session with a written post-mortem sidecar and a
    preserved attempt ref must surface both in the "dead session post-mortems"
    doctor check — otherwise operators have no visibility into a
    push-gate-hook kill or salvaged unpushed commits without reading raw
    sidecar JSON by hand."""
    import subprocess

    from charlie_work.attempt_refs import snapshot_attempt_ref
    from charlie_work.post_mortem import PostMortemRecord

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_sidecar(
        sessions_dir,
        "issue-7.json",
        {
            "issue_number": 7,
            "branch": "agent/issue-7",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-7.log",
            "error": "devin not found",
        },
    )
    # A written post-mortem sidecar for the same dead session.
    record = PostMortemRecord(
        issue_number=7,
        generated_at="2026-01-01T00:05:00+00:00",
        db_path=str(tmp_path / "sessions.db"),
        matched=True,
        session_id="sess-7",
        failure_kind="worker_blocked",
        terminal_tool="bash",
        terminal_reason="push-gate hook rejected: rm -rf attempted",
    )
    (sessions_dir / "issue-7.post-mortem.json").write_text(
        json.dumps(record.to_dict()), encoding="utf-8"
    )

    # A real attempt ref preserved for issue 7, in a real git repo at repo_root
    # (list_attempt_refs shells out to `git for-each-ref`).
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    snapshot = snapshot_attempt_ref(tmp_path, "HEAD", issue_number=7)
    assert snapshot.ref_name is not None  # sanity: the fixture actually wrote a ref

    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    post_mortems = by_name["dead session post-mortems"]
    assert "issue #7" in post_mortems.detail
    assert "failure_kind=worker_blocked" in post_mortems.detail
    assert "terminal_tool=bash" in post_mortems.detail
    assert "push-gate hook rejected" in post_mortems.detail
    assert snapshot.ref_name in post_mortems.detail
    assert post_mortems.severity == "warning"  # never blocks the run


def test_doctor_surface_post_mortems_absent_degrades_silently(tmp_path: Path, monkeypatch) -> None:
    """Issue #261 F6: a dead session with NO post-mortem sidecar and no
    attempt refs must not surface a "dead session post-mortems" check at
    all — post-mortem extraction is best-effort/opportunistic (issue #261),
    never a doctor failure or a misleading empty finding."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_sidecar(
        sessions_dir,
        "issue-9.json",
        {
            "issue_number": 9,
            "branch": "agent/issue-9",
            "prompt_path": "p.md",
            "command": ["devin"],
            "pid": None,
            "started_at": "2026-01-01T00:00:00Z",
            "log_path": "issue-9.log",
            "error": "devin not found",
        },
    )
    # No .post-mortem.json sidecar written, and no repo_root git refs — the
    # extraction never ran / found nothing.

    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert "dead session post-mortems" not in by_name
    assert ok is True  # a failed launch alone is a warning, never a hard failure here


def test_doctor_adapter_probe_reports_failed_devin_binary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(
            returncode=None, stdout="", stderr="", error="devin: not found"
        ),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["devin CLI probe"].ok is False
    assert "not found" in by_name["devin CLI probe"].detail
    assert ok is False  # a broken adapter CLI is an error-severity block
    # No sessions dir was created -> surfaced as a benign warning, not a crash.
    assert by_name["launched sessions"].severity == "warning"


def test_doctor_adapter_probe_claude_code_probes_claude(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="claude 2.0", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        # Empty venv_source skips the venv-existence check so this test stays
        # scoped to the probe path.
        claude_code=ClaudeCodeConfig(venv_source=""),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {check.name: check for check in checks}
    assert by_name["claude CLI probe"].ok is True
    assert "claude 2.0" in by_name["claude CLI probe"].detail


def test_doctor_without_adapter_probe_omits_probe_checks(tmp_path: Path) -> None:
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "devin CLI probe" not in names
    assert "launched sessions" not in names


def test_doctor_corrupt_state_is_not_quarantined(tmp_path: Path) -> None:
    """doctor must report a failure on corrupt state WITHOUT renaming the file."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # Write a corrupt (non-JSON) state file.
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text("NOT JSON {{{", encoding="utf-8")

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    # The original file must still exist — doctor must not quarantine it.
    assert paths.state_file.exists(), "doctor must not rename/quarantine the state file"
    # Doctor should surface the corruption as a check failure.
    by_name = {check.name: check for check in checks}
    assert by_name["state file"].ok is False


def test_doctor_surfaces_existing_quarantine_files_as_warning(tmp_path: Path) -> None:
    """Pre-existing *.corrupt-* files must appear as a warning-severity check."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # Simulate a previously-quarantined corrupt state file.
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt = paths.state_file.parent / f"{paths.state_file.name}.corrupt-20260101T000000Z"
    corrupt.write_text("{}", encoding="utf-8")

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert "state file quarantine" in by_name
    quarantine_check = by_name["state file quarantine"]
    assert quarantine_check.ok is False
    assert quarantine_check.severity == "warning"
    assert "1 quarantined" in quarantine_check.detail
    # A warning-only finding must not block doctor.
    assert ok is True


def test_doctor_no_quarantine_check_when_none_exist(tmp_path: Path) -> None:
    """No quarantine check emitted when there are no corrupt-* files."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert "state file quarantine" not in names


def test_doctor_adapter_probe_uses_configured_devin_binary(tmp_path: Path, monkeypatch) -> None:
    """probe_devin must be called with the binary from devin.shell_command."""
    captured: list[tuple[str, ...]] = []

    def fake_probe_devin(repo_root, **kwargs):
        captured.append(kwargs.get("command", ()))
        return RunResult(returncode=0, stdout="custom-devin 9.9", stderr="")

    monkeypatch.setattr("charlie_work.devin_shell.probe_devin", fake_probe_devin)

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            shell_command=("my-devin-wrapper", "--prompt-file", "{prompt_path}", "--print"),
        ),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    assert len(captured) == 1
    assert captured[0][0] == "my-devin-wrapper", (
        "probe must use the configured binary, not the hardcoded default"
    )


def test_doctor_adapter_probe_uses_configured_claude_binary(tmp_path: Path, monkeypatch) -> None:
    """probe_claude must be called with the binary from claude_code.command."""
    captured: list[tuple[str, ...]] = []

    def fake_probe_claude(repo_root, **kwargs):
        captured.append(kwargs.get("command", ()))
        return RunResult(returncode=0, stdout="my-claude 5.0", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe_claude)

    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="claude-code"),
        claude_code=ClaudeCodeConfig(
            command=("my-claude-wrapper", "-p", "--permission-mode", "acceptEdits"),
            venv_source="",
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    assert len(captured) == 1
    assert captured[0][0] == "my-claude-wrapper", (
        "probe must use the configured binary, not the hardcoded default"
    )
