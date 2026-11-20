"""Doctor probes for the opencode harness: CLI/bin, argv drift, review posture.

Split out of ``tests/test_opencode_worker.py`` (800-line file-size ratchet,
same convention as ``test_opencode_log_parser.py`` / ``test_opencode_review_summary.py``).
These exercise ``doctor._probe_opencode*`` through ``run_doctor`` with a fake
``probe_claude`` seam, so no real binary is spawned here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from _doctor_fixtures import FakeDoctorGitHub, _config

from charlie_work.config import OrchestratorConfig, ReviewerRoleConfig, WorkerRoleConfig
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.subprocess_runner import RunResult

MODEL = "glm-5.3-flash"


def _doctor(tmp_path: Path, config: OrchestratorConfig):
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)
    return {check.name: check for check in checks}


def _fake_run_help(command: tuple[str, ...]) -> RunResult | None:
    """A realistic `opencode run --help` body for the argv probe."""
    if len(command) > 1 and command[1] == "run":
        return RunResult(
            returncode=0,
            stdout=(
                "USAGE\n  opencode run [flags] [<message...>]\n\nFLAGS\n"
                "  --auto                    Auto-approve permissions\n"
                "  --standalone              Run with a private server\n"
                "  --format choice           Output format (choices: default, json)\n"
                "  --log-level <all|trace|debug|info|warn|warning|error|fatal|none>\n"
                "                            Sets the minimum log level\n"
                "  --print-logs              Print logs to stderr\n"
            ),
            stderr="",
        )
    return None


def test_doctor_probes_opencode_as_primary_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        seen.append(command)
        help_result = _fake_run_help(command)
        if help_result is not None:
            return help_result
        return RunResult(returncode=0, stdout="opencode v2.3.0\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    monkeypatch.setattr("charlie_work.doctor._host_opencode_providers", lambda: (frozenset(), ""))
    by_name = _doctor(tmp_path, _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL)))
    assert ("opencode", "--version") in seen
    assert ("opencode", "run", "--help") in seen
    assert by_name["opencode CLI probe"].ok is True
    assert "opencode v2.3.0" in by_name["opencode CLI probe"].detail
    # The launch argv is validated against the installed binary's help, so a
    # flag the binary dropped (or a value it rejects) fails the doctor probe
    # instead of every launch.
    assert by_name["opencode run argv probe"].ok is True


def test_doctor_opencode_argv_probe_catches_drifted_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binary without --variant (2.x) and lowercase-only --log-level
    choices rejects the 1.x-era template: the argv probe fails loudly (the
    version-only probe cannot see this -- `opencode --version` exits 0)."""

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        if len(command) > 1 and command[1] == "run":
            return RunResult(
                returncode=0,
                stdout=(
                    "FLAGS\n"
                    "  --auto                    Auto-approve\n"
                    "  --format choice           (choices: default, json)\n"
                    "  --log-level <all|warn|error>  Log level\n"
                    "  --print-logs              Print logs\n"
                ),
                stderr="",
            )
        return RunResult(returncode=0, stdout="opencode v2.3.0\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    config = _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL))
    config = replace(
        config,
        opencode=replace(
            config.opencode,
            command=("opencode", "run", "--auto", "--variant", "high", "--log-level", "WARN"),
        ),
    )
    by_name = _doctor(tmp_path, config)
    probe = by_name["opencode run argv probe"]
    assert probe.ok is False
    assert "--variant" in probe.detail
    assert "--log-level WARN" in probe.detail


def test_doctor_opencode_argv_probe_accepts_equals_form_and_checks_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--log-level=<value>`` is the other spelling operators use; the value
    must be validated there too (an earlier probe silently skipped it)."""

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        if len(command) > 1 and command[1] == "run":
            return RunResult(
                returncode=0,
                stdout=(
                    "FLAGS\n"
                    "  --auto                    Auto-approve\n"
                    "  --standalone              Private server\n"
                    "  --log-level <all|warn|error>  Log level\n"
                ),
                stderr="",
            )
        return RunResult(returncode=0, stdout="opencode v2.3.0\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    config = _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL))
    bad = replace(
        config,
        opencode=replace(config.opencode, command=("opencode", "run", "--log-level=WARN")),
    )
    probe = _doctor(tmp_path, bad)["opencode run argv probe"]
    assert probe.ok is False
    assert "--log-level WARN" in probe.detail

    good = replace(
        config,
        opencode=replace(config.opencode, command=("opencode", "run", "--log-level=warn")),
    )
    assert _doctor(tmp_path, good)["opencode run argv probe"].ok is True


def test_doctor_opencode_unparseable_log_level_choices_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A help format change must degrade to "unverified", never to a red
    doctor: failing here would report a healthy install as broken."""

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        if len(command) > 1 and command[1] == "run":
            return RunResult(
                returncode=0,
                stdout=(
                    "FLAGS\n"
                    "  --auto    Auto-approve\n"
                    "  --standalone    Private server\n"
                    "  --format    Output format\n"
                    "  --print-logs    Print logs\n"
                    "  --log-level    Log level\n"
                ),
                stderr="",
            )
        return RunResult(returncode=0, stdout="opencode v2.3.0\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    config = _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL))
    by_name = _doctor(tmp_path, config)
    assert by_name["opencode run argv probe"].ok is True
    unverified = by_name["opencode --log-level value"]
    assert unverified.ok is True
    assert unverified.severity == "warning"


def test_doctor_reports_the_review_argv_posture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env-only review posture depends on argv shape: the probe asserts the
    effective review template carries --standalone and reports an operator
    --server (which the sanitizer strips for reviews)."""
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, *, command=None, **_kw: (
            _fake_run_help(tuple(command or ()))
            or RunResult(returncode=0, stdout="opencode v2.3.0\n", stderr="")
        ),
    )
    monkeypatch.setattr("charlie_work.doctor._host_opencode_providers", lambda: (frozenset(), ""))
    config = _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL))
    assert _doctor(tmp_path, config)["opencode review argv posture"].ok is True

    with_server = replace(
        config,
        opencode=replace(
            config.opencode,
            command=("opencode", "run", "--server", "http://127.0.0.1:4096"),
        ),
    )
    by_name = _doctor(tmp_path, with_server)
    # The sanitizer strips --server and pins --standalone, so the review
    # posture itself still holds; the operator is told their flag is ignored.
    assert by_name["opencode review argv posture"].ok is True
    server_note = by_name["opencode review --server override"]
    assert server_note.ok is True
    assert server_note.severity == "warning"
    assert "--server" in server_note.detail


def test_doctor_flags_a_reviewer_provider_declared_only_in_host_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review launch pins an empty OPENCODE_CONFIG_DIR, so a reviewer model
    whose provider is custom (declared only in the operator's global config)
    cannot route. Doctor surfaces the overlap -- as a warning, because a host
    entry may equally be an override of a BUILT-IN provider, which a reviewer
    reaches without it."""
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, *, command=None, **_kw: RunResult(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(
        "charlie_work.doctor._host_opencode_providers",
        lambda: (frozenset({"my-custom"}), "C:/home/.config/opencode/opencode.json"),
    )
    config = _config(worker=WorkerRoleConfig(harness="claude-code", model="claude-sonnet-5-5"))
    reviewer = ReviewerRoleConfig(harness="opencode", model="my-custom/some-model")
    config = replace(config, reviewer=reviewer)
    check = _doctor(tmp_path, config)["opencode review provider visibility"]
    assert check.ok is True
    assert check.severity == "warning"
    assert "my-custom" in check.detail
    assert "empty OPENCODE_CONFIG_DIR" in check.detail
    assert "provider.no-route" in check.detail

    # A host provider the reviewer chain does not use is still mentioned
    # (informational), never a failure.
    monkeypatch.setattr(
        "charlie_work.doctor._host_opencode_providers",
        lambda: (frozenset({"unrelated"}), "C:/home/.config/opencode/opencode.json"),
    )
    warning = _doctor(tmp_path, config)["opencode review provider visibility"]
    assert warning.ok is True
    assert warning.severity == "warning"

    # No host provider block at all: silent (nothing to say).
    monkeypatch.setattr("charlie_work.doctor._host_opencode_providers", lambda: (frozenset(), ""))
    assert "opencode review provider visibility" not in _doctor(tmp_path, config)


def test_doctor_probes_opencode_when_only_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        seen.append(command)
        if command[0] == "opencode":
            return RunResult(returncode=None, stdout="", stderr="", error="opencode: not found")
        return RunResult(returncode=0, stdout="claude 9\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    worker = WorkerRoleConfig(
        harness="claude-code",
        model="claude-sonnet-5-5",
        fallbacks=(RoleEntry("opencode", MODEL),),
    )
    by_name = _doctor(tmp_path, _config(worker=worker))
    assert ("opencode", "--version") in seen
    assert by_name["opencode CLI probe"].ok is False
    assert "not found" in by_name["opencode CLI probe"].detail


def test_doctor_probes_opencode_in_the_reviewer_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opencode REVIEW fallback launches the same CLI, so a missing binary
    must surface at doctor time even when the worker chain is claude-code."""
    seen: list[tuple[str, ...]] = []

    def fake_probe(repo_root, *, command=None, **_kw):
        command = tuple(command)
        seen.append(command)
        if command[0] == "opencode":
            return RunResult(returncode=None, stdout="", stderr="", error="opencode: not found")
        return RunResult(returncode=0, stdout="claude 9\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    config = _config(worker=WorkerRoleConfig(harness="claude-code", model="claude-sonnet-5-5"))
    config = replace(
        config,
        reviewer=ReviewerRoleConfig(
            harness="claude-code",
            model="claude-sonnet-5-5",
            fallbacks=(RoleEntry("opencode", MODEL),),
        ),
    )
    by_name = _doctor(tmp_path, config)
    assert ("opencode", "--version") in seen
    assert by_name["opencode CLI probe"].ok is False


def test_doctor_does_not_probe_opencode_when_not_in_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, **_kw: RunResult(returncode=0, stdout="ok", stderr=""),
    )
    worker = WorkerRoleConfig(harness="claude-code", model="claude-sonnet-5-5")
    by_name = _doctor(tmp_path, _config(worker=worker))
    assert "opencode CLI probe" not in by_name
