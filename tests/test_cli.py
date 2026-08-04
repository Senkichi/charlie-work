from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.config import (
    ConfigError,
    NotifyConfig,
    OrchestratorConfig,
    RunnerAllocationConfig,
    RunnerScalingConfig,
)
from charlie_work.cross_family import LEGACY_VACUOUS_SUMMARY
from charlie_work.fleet_dispatch import ApiWorkerFleetReport
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import log_event
from charlie_work.paths import runtime_paths
from charlie_work.quiesce import QuiesceReport
from charlie_work.state_migration import MigrationChild, MigrationOutcome, MigrationPlan
from charlie_work.supervise import SelfDeployResult
from charlie_work.workflow import CommandResult


class _FakeGitHub:
    """Stub GitHub client sufficient to drive cli.main through the verdict path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def name_with_owner(self) -> str:
        return "owner/repo"

    def pr_view(self, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "title": "Fix search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-1-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #1\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }

    def pr_diff(self, number: int) -> str:
        return "diff content"

    def add_issue_label(self, number: int, label: str) -> bool:
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        return True


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    return tmp_path


def test_cli_verdict_missing_summary_file_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #357: a missing --summary-file must fail the CLI command (non-zero rc)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    missing_summary = repo / "missing-summary.md"

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(missing_summary),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "OS error" in captured.out
    assert "No such file or directory" in captured.out
    assert not (repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json").exists()


def test_cli_verdict_success_records_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: a readable summary file records the verdict and exits 0."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    assert decision_path.exists()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "approved"
    assert decision["summary"] == "lgtm"


def test_cli_verdict_reviewed_head_flag_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #467: --reviewed-head is accepted and recorded with provenance."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
            "--reviewed-head",
            "sha-abc",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc"
    assert decision["reviewed_head_source"] == "live"
    assert "(head from live)" in captured.out


def test_cli_verdict_missing_summary_file_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --json, the missing-file failure is still a machine-parseable non-zero result."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    missing_summary = repo / "missing-summary.md"

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--json",
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(missing_summary),
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "OS error" in payload["message"]
    assert not (repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json").exists()


def test_cli_verdict_request_changes_derives_required_changes_from_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #792 AC-6, third producer path: `charlie verdict` has no
    --required-changes flag at all -- every request_changes/blocked verdict
    recorded through this CLI command arrives at record_review with
    required_changes=None, so real reviewer prose supplied via
    --summary-file must be derived into required_changes rather than
    persisted empty."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    prose = "The retry wrapper swallows the exception type; push a fix before merging."
    summary.write_text(prose, encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "request_changes",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "review recorded" in captured.out
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "request_changes"
    assert decision["required_changes"] == [prose]
    assert decision["findings_channel"] == "derived"


def test_cli_verdict_request_changes_persists_vacuous_marker_for_legacy_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same manual-CLI producer path, the nothing-derivable outcome: a
    --summary-file containing the historical content-free placeholder
    records ok=True with required_changes: [] and findings_channel:
    "vacuous" -- never a non-zero exit, matching record_review's
    never-reject invariant (issue #792 AC-3)."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text(LEGACY_VACUOUS_SUMMARY, encoding="utf-8")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "request_changes",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    decision_path = repo / ".var" / "charlie-work" / "prs" / "pr-1" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["required_changes"] == []
    assert decision["findings_channel"] == "vacuous"


def test_cli_verdict_isolates_fleet_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #362: cli.main must not write to the operator's real fleet registry.

    Even when no ``--fleet-dir`` is supplied, ``build_app`` calls ``touch_repo``,
    which would otherwise resolve to the global ``%LOCALAPPDATA%`` path. The
    suite-wide autouse fixture redirects writes to a per-test directory; this
    test proves the real default registry is untouched.
    """
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    # Capture the real default fleet path (no env override). The autouse conftest
    # fixture normally sets CHARLIE_WORK_FLEET_DIR, so temporarily clear it to
    # resolve the platform default, then restore the isolated override.
    monkeypatch.delenv("CHARLIE_WORK_FLEET_DIR")
    real_fleet_json = fleet_dir() / "fleet.json"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))

    real_before = real_fleet_json.read_text(encoding="utf-8") if real_fleet_json.exists() else None

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0

    # The write must land in the per-test fleet directory, not the real one.
    isolated_fleet_json = tmp_path / "fleet" / "fleet.json"
    assert isolated_fleet_json.exists()
    data = json.loads(isolated_fleet_json.read_text(encoding="utf-8"))
    assert data["repos"]["owner/repo"]["repo_root"] == str(repo)

    # The operator's real registry must be unchanged (or still absent).
    if real_before is None:
        assert not real_fleet_json.exists()
    else:
        assert real_fleet_json.read_text(encoding="utf-8") == real_before


def test_cli_spec_review_missing_file_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #363: a missing --file for spec_review exits 1 with an OS error message."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    missing_spec = repo / "missing-spec.md"

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "why-charlie-hate-spec",
            "--file",
            str(missing_spec),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "OS error" in captured.out
    assert "No such file or directory" in captured.out
    assert not (repo / ".var" / "charlie-work" / "cross-family").exists()


def test_cli_spec_review_unreadable_file_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --json, an unreadable --file failure is still a machine-parseable non-zero result."""
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    unreadable = repo / "unreadable.md"
    unreadable.write_text("secret", encoding="utf-8")

    orig_read_text = Path.read_text

    def _read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == unreadable:
            raise PermissionError(13, "Permission denied", str(self))
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_text)

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--json",
            "why-charlie-hate-spec",
            "--file",
            str(unreadable),
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "OS error" in payload["message"]
    assert not (repo / ".var" / "charlie-work" / "cross-family").exists()


def test_cli_fleet_supervise_parser() -> None:
    """``fleet supervise`` accepts the expected limit/repos/poll/max-runtime/merge flags."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "fleet",
            "supervise",
            "--limit",
            "2",
            "--repos",
            "a/b,c/d",
            "--poll-interval",
            "10",
            "--max-runtime",
            "60",
            "--merge",
        ]
    )
    assert args.command == "fleet"
    assert args.fleet_command == "supervise"
    assert args.limit == 2
    assert args.repos == "a/b,c/d"
    assert args.poll_interval == 10
    assert args.max_runtime == 60
    assert args.merge is True


def test_cli_fleet_supervise_command_runs_and_returns_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``charlie fleet supervise`` is wired through ``main`` and returns the loop result."""
    captured: dict[str, Any] = {}

    def _fake_run_fleet_supervise(**kwargs: Any) -> CommandResult:
        captured["kwargs"] = kwargs
        return CommandResult(True, "fleet supervisor complete", {})

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    rc = cli.main(["fleet", "supervise", "--limit", "3", "--max-runtime", "120"])

    assert rc == 0
    assert captured["kwargs"]["limit"] == 3
    assert captured["kwargs"]["max_runtime_override"] == 120


def test_cli_maps_restart_requested_to_a_distinct_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#862 AC1: a restart-requesting exit is distinguishable from a clean one.

    Both are ``ok=True``, which is exactly why the fleet used to sit
    unsupervised for a full watchdog interval after every self-deploy without
    anything in Task Scheduler looking wrong.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            True,
            "fleet supervisor complete",
            {"exit_reason": "self_deploy", "restart_requested": True},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == cli.EXIT_RESTART_REQUESTED


def test_cli_maps_a_deliberate_stop_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4's control: same command, same ok=True, opposite exit code.

    Paired with the test above so the distinct code is attributable to
    ``restart_requested`` rather than to the command being run at all.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            True,
            "fleet supervisor complete",
            {"exit_reason": "max_runtime", "restart_requested": False},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == 0


def test_cli_maps_restart_requested_even_on_a_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Replace me" is orthogonal to "I succeeded".

    A supervisor that self-deployed and then hit an error still has new code on
    disk and still needs replacing -- relaunching is the recovery, not a reward
    for a clean run. This was gated on ``result.ok``, which made the preserved
    restart reason on an aborted run inert: exit 1, no relaunch, fleet
    unsupervised for the interval. That is #862 reached through the error path
    instead of the happy path.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            False,
            "fleet supervisor aborted on pass 1: state file locked",
            {"exit_reason": "self_deploy", "restart_requested": True},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == cli.EXIT_RESTART_REQUESTED


def test_cli_maps_a_plain_failure_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the test above: ok=False without a restart reason stays 1.

    Ungating the restart code from ``ok`` must not turn every failure into a
    relaunch request -- only the ones that actually asked to be replaced.
    """

    def _fake_run_fleet_supervise(**_kwargs: Any) -> CommandResult:
        return CommandResult(
            False,
            "fleet supervisor aborted on pass 1: boom",
            {"exit_reason": "aborted", "restart_requested": False},
        )

    monkeypatch.setattr(cli, "run_fleet_supervise", _fake_run_fleet_supervise)

    assert cli.main(["fleet", "supervise"]) == 1


def test_cli_fleet_supervise_loop_forwards_args_after_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper passes supervise's flags through verbatim.

    Guards the passthrough contract the launcher depends on: `supervise-loop`
    deliberately does not re-declare supervise's flags, so a regression that
    swallowed them would start the supervisor with default settings instead of
    the launcher's `--max-runtime 0`.
    """
    captured: dict[str, Any] = {}

    def _fake_run_fleet_supervise_loop(**kwargs: Any) -> CommandResult:
        captured["kwargs"] = kwargs
        return CommandResult(True, "supervise-loop done", {})

    monkeypatch.setattr(cli, "run_fleet_supervise_loop", _fake_run_fleet_supervise_loop)

    rc = cli.main(["fleet", "supervise-loop", "--max-relaunches", "2", "--", "--max-runtime", "0"])

    assert rc == 0
    assert captured["kwargs"]["max_relaunches"] == 2
    assert captured["kwargs"]["supervise_args"] == ("--max-runtime", "0")


def test_run_fleet_bash_rats_self_deploys_before_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`charlie fleet bash-rats` calls self_deploy on the orchestrator root first."""
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            from_sha="abc123",
            to_sha="def456",
            message="updated: def456",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    args = cli.build_parser().parse_args(
        ["--fleet-dir", "custom-fleet", "fleet", "bash-rats", "--limit", "2"]
    )
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    deploy_mock.assert_called_once()
    orchestrator_root = deploy_mock.call_args[0][0]
    assert isinstance(orchestrator_root, Path)
    assert (orchestrator_root / "pyproject.toml").exists()
    assert deploy_mock.call_args.kwargs.get("fleet_dir_override") == "custom-fleet"
    assert fleet_loop_mock.called is True
    assert fleet_loop_mock.call_args.kwargs.get("fleet_dir_override") == "custom-fleet"

    out = capsys.readouterr().out
    assert "self-deploy: updated: def456" in out


def test_run_fleet_bash_rats_self_deploy_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed self_deploy does not abort a `charlie fleet bash-rats` pass."""
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="diverged or dirty tree",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    out = capsys.readouterr().out
    assert "self-deploy skipped: diverged or dirty tree" in out


def test_run_fleet_bash_rats_emits_attention_digest_on_repair_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A self_deploy repair failure emits an attention digest when notify is enabled."""
    digest_path = tmp_path / "digest.jsonl"
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="venv pth repair failed: Access is denied",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)

    # Provide a config whose notify sink writes to a temp file.
    from charlie_work.config import OrchestratorConfig

    notify_config = NotifyConfig(enabled=True, sink="file", file_path=str(digest_path))
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(notify=notify_config),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    assert digest_path.exists()
    digest_line = digest_path.read_text(encoding="utf-8").strip()
    digest = json.loads(digest_line)
    assert digest["repo"] == "fleet"
    assert len(digest["transitions"]) == 1
    assert digest["transitions"][0]["adapter_kind"] == "self-deploy"
    assert digest["transitions"][0]["health"] == "ERROR"
    assert "Access is denied" in digest["transitions"][0]["last_log_line"]


def test_run_fleet_bash_rats_emits_attention_digest_on_venv_repaired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successful self_deploy venv repair emits an attention digest when notify is enabled."""
    digest_path = tmp_path / "digest.jsonl"
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            venv_repaired=True,
            message="venv editable target repaired: shared venv editable .pth points to main checkout src",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)

    # Provide a config whose notify sink writes to a temp file.
    from charlie_work.config import OrchestratorConfig

    notify_config = NotifyConfig(enabled=True, sink="file", file_path=str(digest_path))
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(notify=notify_config),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    assert digest_path.exists()
    digest_line = digest_path.read_text(encoding="utf-8").strip()
    digest = json.loads(digest_line)
    assert digest["repo"] == "fleet"
    assert len(digest["transitions"]) == 1
    assert digest["transitions"][0]["adapter_kind"] == "self-deploy"
    assert digest["transitions"][0]["health"] == "REPAIRED"


def test_run_fleet_bash_rats_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud in `charlie fleet bash-rats`, not silent.

    Mirrors test_run_fleet_supervise_loud_on_absent_global_layer: drives the
    REAL ``load_layered_config`` (not a mock) against an empty fleet dir so the
    ``require_global=True`` wiring in ``run_fleet_bash_rats`` is exercised
    end-to-end. Every other ``run_fleet_bash_rats`` test mocks
    ``load_layered_config`` away (``lambda *a, **k: None``), so without this
    test a future silent drop of ``require_global=True`` would pass CI
    undetected -- fleet-wide knobs (notify, runner prologues) would revert to
    dataclass defaults with no error raised, the #623 failure shape.

    ``self_deploy`` and ``fleet_loop`` are mocked so the pass does not touch the
    network or the real fleet; only the config-load path is left real.
    """
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message="up to date",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)
    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(tmp_path), "fleet", "bash-rats", "--limit", "1"]
    )
    result = cli.run_fleet_bash_rats(args)

    # The require_global=True ConfigError is caught and printed loudly. If
    # require_global=True were silently dropped, load_layered_config would
    # return an OrchestratorConfig (not raise), "config load failed" would
    # never be printed.
    out = capsys.readouterr().out
    assert "config load failed" in out, (
        "an absent global layer must be printed, not silently defaulted"
    )
    assert str(tmp_path / "config.yaml") in out, (
        "the expected global config path must appear in the failure message"
    )
    assert "absent" in out, "an absent layer must read as absent in the message"

    # The pass continued (the command must not crash). The fallback reloads
    # with require_global=False so the per-repo config is NOT discarded with
    # the global layer -- regressing to global_config=None would reproduce the
    # #623 silent-disable failure (every per-repo knob reverting to defaults).
    # Here cwd has no per-repo config, so the reload yields pristine defaults,
    # but the point is it is a real OrchestratorConfig, not the None sentinel
    # that would skip the runner prologues and silence notify unconditionally.
    assert result.ok is True
    assert fleet_loop_mock.call_count == 1
    assert fleet_loop_mock.call_args.kwargs.get("global_config") is not None, (
        "fleet_loop must NOT receive global_config=None when the global layer "
        "is absent -- the per-repo config must survive the fallback, not be "
        "discarded with the global layer (#623 silent-disable regression)"
    )


# ---------------------------------------------------------------------------
# api-worker fleet report CLI wiring (issue #483)
#
# The standalone compute_api_worker_fleet_report() is tested in
# test_fleet_dispatch.py. These tests cover the CLI-visible wiring the issue
# actually asks for: _render_api_worker_report() and the api_worker_report key
# threaded through run_fleet_status / fleet_loop's CommandResult.data. A silent
# breakage in the rendering/key-lookup path would otherwise ship undetected.
# ---------------------------------------------------------------------------


def _sample_api_worker_report() -> ApiWorkerFleetReport:
    return ApiWorkerFleetReport(
        provider="kimi-k3",
        today_usd=1.50,
        lifetime_usd=7.25,
        cap_usd=15.00,
        live=2,
        enabled_k=1,
        enabled_m=4,
    )


def test_render_api_worker_report_prints_line_when_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_render_api_worker_report prints the line (indented) when a dict with a line is present."""
    report = _sample_api_worker_report().to_dict()
    cli._render_api_worker_report({"api_worker_report": report})
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out
    # Rendered with the two-space indent used in fleet status output.
    assert out.startswith("  ")


def test_render_api_worker_report_silent_when_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When api_worker_report is None (no repo configures the section), nothing is printed."""
    cli._render_api_worker_report({"api_worker_report": None})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_key_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the api_worker_report key is absent entirely, nothing is printed."""
    cli._render_api_worker_report({"repos": {}, "errors": []})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_empty_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dict with a falsy 'line' (e.g. empty string) is omitted, not a blank print."""
    cli._render_api_worker_report({"api_worker_report": {"line": "", "provider": "x"}})
    assert capsys.readouterr().out == ""


def test_render_api_worker_report_silent_when_not_dict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-dict api_worker_report value (defensive) is omitted, not raised."""
    cli._render_api_worker_report({"api_worker_report": "unexpected string"})
    assert capsys.readouterr().out == ""


def test_run_fleet_status_threads_api_worker_report_into_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_fleet_status places compute_api_worker_fleet_report's dict into CommandResult.data.

    The autouse conftest fixture points CHARLIE_WORK_FLEET_DIR at an empty
    per-test directory, so the per-repo status loop iterates an empty registry
    and only the api_worker_report wiring is exercised. Mocking the compute
    function isolates the threading from the standalone-function coverage.
    """
    report = _sample_api_worker_report()
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: report)

    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    assert result.ok is True
    assert result.data["api_worker_report"] == report.to_dict()
    assert "line" in result.data["api_worker_report"]
    assert result.data["api_worker_report"]["provider"] == "kimi-k3"


def test_run_fleet_status_api_worker_report_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no repo configures the section, api_worker_report is None (line omitted)."""
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: None)

    args = cli.build_parser().parse_args(["fleet", "status"])
    result = cli.run_fleet_status(args)

    assert result.ok is True
    assert result.data["api_worker_report"] is None


def test_cli_fleet_status_main_renders_api_worker_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet status` prints the api-worker line when configured.

    Mocks compute_api_worker_fleet_report so the rendering path is exercised
    without a full fleet setup; the empty registry (autouse fixture) means no
    per-repo status lines are printed, so the api-worker line is isolated.
    """
    report = _sample_api_worker_report()
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: report)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out
    assert "$1.50 today" in out


def test_cli_fleet_status_main_omits_line_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet status` prints no api-worker line when unconfigured."""
    monkeypatch.setattr(cli, "compute_api_worker_fleet_report", lambda **_k: None)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker:" not in out


def test_cli_fleet_work_main_renders_api_worker_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet work` prints the api-worker line from fleet_loop's data.

    fleet_loop is mocked to return a CommandResult carrying the api_worker_report
    dict, so the work/bash-rats rendering branch in main() is exercised directly.
    """
    report_dict = _sample_api_worker_report().to_dict()
    fleet_loop_mock = MagicMock(
        return_value=CommandResult(
            True,
            "fleet pass complete: 0 repo(s) processed",
            {
                "repos": {},
                "digest": {"count": 0, "orphan_sweep_calls": 0},
                "api_worker_report": report_dict,
            },
        )
    )
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    rc = cli.main(["fleet", "work", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker: kimi-k3" in out
    assert "enabled 1/4 repos" in out


def test_cli_fleet_work_main_omits_line_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: `charlie fleet work` prints no api-worker line when report is None."""
    fleet_loop_mock = MagicMock(
        return_value=CommandResult(
            True,
            "fleet pass complete: 0 repo(s) processed",
            {
                "repos": {},
                "digest": {"count": 0, "orphan_sweep_calls": 0},
                "api_worker_report": None,
            },
        )
    )
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    rc = cli.main(["fleet", "work", "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api-worker:" not in out


def test_run_fleet_work_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud in `charlie fleet work`, not silent.

    Mirrors test_run_fleet_supervise_loud_on_absent_global_layer: drives the
    REAL ``load_layered_config`` (not a mock) against an empty fleet dir so the
    ``require_global=True`` wiring in ``run_fleet_work`` is exercised
    end-to-end. Every other ``run_fleet_work`` / ``charlie fleet work`` test
    mocks ``load_layered_config`` away (``lambda *a, **k: None``), so without
    this test a future silent drop of ``require_global=True`` would pass CI
    undetected -- fleet-wide knobs (notify, runner prologues) would revert to
    dataclass defaults with no error raised, the #623 failure shape.

    ``fleet_loop`` is mocked so the pass does not touch the real fleet; only the
    config-load path is left real.
    """
    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(tmp_path), "fleet", "work", "--limit", "1"]
    )
    result = cli.run_fleet_work(args)

    # The require_global=True ConfigError is caught and printed loudly. If
    # require_global=True were silently dropped, load_layered_config would
    # return an OrchestratorConfig (not raise), "config load failed" would
    # never be printed.
    out = capsys.readouterr().out
    assert "config load failed" in out, (
        "an absent global layer must be printed, not silently defaulted"
    )
    assert str(tmp_path / "config.yaml") in out, (
        "the expected global config path must appear in the failure message"
    )
    assert "absent" in out, "an absent layer must read as absent in the message"

    # The pass continued (the command must not crash). The fallback reloads
    # with require_global=False so the per-repo config is NOT discarded with
    # the global layer -- regressing to global_config=None would reproduce the
    # #623 silent-disable failure (every per-repo knob reverting to defaults).
    # Here cwd has no per-repo config, so the reload yields pristine defaults,
    # but the point is it is a real OrchestratorConfig, not the None sentinel
    # that would skip the runner prologues and silence notify unconditionally.
    assert result.ok is True
    assert fleet_loop_mock.call_count == 1
    assert fleet_loop_mock.call_args.kwargs.get("global_config") is not None, (
        "fleet_loop must NOT receive global_config=None when the global layer "
        "is absent -- the per-repo config must survive the fallback, not be "
        "discarded with the global layer (#623 silent-disable regression)"
    )


def test_run_fleet_work_fallback_preserves_per_repo_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The require_global fallback must keep the per-repo config, not discard it.

    The review on PR #630 found that the fallback set ``global_config=None``
    when ``require_global=True`` raised, discarding the per-repo config too --
    not just the global layer. That regressed a previously-valid no-op state
    into the exact #623 silent-disable failure: every per-repo knob (notify,
    labels) reverted to its dataclass default while passes kept reporting
    success. The fallback now reloads with ``require_global=False`` so per-repo
    settings survive.

    cwd becomes a repo with a per-repo config that turns notify ON; the fleet
    dir is a separate empty dir so the global layer is absent. Drives the REAL
    ``load_layered_config`` (not a mock) so the fallback path is exercised
    end-to-end.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "orchestrator.config.yaml").write_text(
        "notify:\n  enabled: true\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo_dir)
    fleet_empty = tmp_path / "fleet"
    fleet_empty.mkdir()

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_empty), "fleet", "work", "--limit", "1"]
    )
    result = cli.run_fleet_work(args)

    # The absent global layer is still loud...
    out = capsys.readouterr().out
    assert "config load failed" in out, (
        "the absent global layer must still be printed even when per-repo config is preserved"
    )

    # ...but the per-repo notify setting survives the fallback. Before the fix
    # global_config was None and notify was effectively off; now it is the
    # per-repo OrchestratorConfig with notify.enabled=True.
    assert result.ok is True
    passed_config = fleet_loop_mock.call_args.kwargs.get("global_config")
    assert passed_config is not None, (
        "the per-repo config must survive the require_global fallback, not be "
        "discarded with the absent global layer"
    )
    assert getattr(passed_config.notify, "enabled", False) is True, (
        "the per-repo notify.enabled=True must survive the require_global "
        "fallback, not revert to the dataclass default False (#623 regression)"
    )


def test_run_runners_allocate_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An absent global layer must fail loudly in `charlie runners allocate`.

    ``runner_allocation`` is a fleet-wide knob declared in the global fleet
    config layer. An unreachable global layer silently flips it to its
    dataclass default (``enabled=False``), and ``run_runners_allocate`` would
    then report "runner_allocation feature is not enabled in config" -- the
    exact #623 silent-disable failure: the operator sees an opt-out message
    when the real cause is an unready volume. The command now loads with
    ``require_global=True`` and fails loudly instead, so an unready volume is
    distinguishable from a fleet that genuinely opted out of runner allocation.

    Drives the REAL ``load_layered_config`` (not a mock) against an empty fleet
    dir so the ``require_global=True`` wiring is exercised end-to-end.
    ``find_repo_root`` is stubbed so the command does not require a real git
    work tree at cwd; the config load fails before any network/runner work.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: tmp_path)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(["--fleet-dir", str(tmp_path), "runners", "allocate"])
    result = cli.run_runners_allocate(args)

    # The command fails loudly with the config-load cause, NOT the silent
    # "not enabled" message that a defaulted runner_allocation would produce.
    assert result.ok is False, (
        "an absent required global layer must fail the command, not silently "
        "default runner_allocation to disabled"
    )
    assert "config load failed" in result.message, (
        f"the failure must name the config load, not 'not enabled': {result.message!r}"
    )
    assert "cannot decide runner_allocation" in result.message, (
        f"the failure must name runner_allocation: {result.message!r}"
    )
    assert str(tmp_path / "config.yaml") in result.message, (
        "the expected global config path must appear in the failure message"
    )
    assert "absent" in result.message, "an absent layer must read as absent in the failure message"
    assert "not enabled" not in result.message, (
        "the silent-disable 'not enabled' message must NOT appear when the real "
        "cause is an unreachable global layer"
    )


def test_run_doctor_command_reports_structured_finding_on_unparseable_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6-G / G-AC3: doctor must not itself crash on the exact condition it
    exists to diagnose.

    Before this fix, ``run_doctor_command`` called ``load_layered_config``
    unguarded, so a config parse failure (e.g. the 2026-07-29 incident's
    unknown ``cross_family: auto_verdict`` key) propagated past this function
    to ``main()``'s generic ``except (ConfigError, ValueError)`` handler,
    which prints to stderr and exits 2 with no machine-readable finding --
    the operator gets nothing to act on. Now the failure is caught locally
    and rendered as a structured, blocking ``DoctorCheck`` finding instead.

    Drives the REAL ``load_layered_config`` (not mocked) against a real
    unparseable config file on disk, mirroring the exact incident shape used
    in test_fleet_dispatch.py's
    test_fleet_loop_real_unknown_config_key_reproduces_incident.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: tmp_path)
    # Deliberately NOT mocking cli.load_layered_config.
    (tmp_path / "orchestrator.config.yaml").write_text(
        "labels:\n  ready: automated-ready\ncross_family:\n  totally_unknown_key: true\n",
        encoding="utf-8",
    )

    args = cli.build_parser().parse_args(["--fleet-dir", str(tmp_path), "doctor"])
    result = cli.run_doctor_command(args)

    assert result.ok is False, "an unparseable config must fail the command, not crash it"
    assert "1 finding" in result.message
    checks = result.data["checks"]
    assert len(checks) == 1, f"expected exactly one synthetic finding, got: {checks!r}"
    check = checks[0]
    assert check["name"] == "config file"
    assert check["ok"] is False
    assert check["severity"] == "error", "a config parse failure must be blocking, not a warning"
    assert "totally_unknown_key" in check["detail"]
    assert "cross_family" in check["detail"]


# --------------------------------------------------------------------------
# runners ensure-started: single-controller guard (issue #598)
# --------------------------------------------------------------------------


def _ensure_started_config(
    *,
    scaling_enabled: bool = True,
    allocation_enabled: bool = False,
    managed_root: str = "",
) -> OrchestratorConfig:
    """Build an OrchestratorConfig with the runner_scaling/allocation knobs set.

    ``managed_root`` defaults to empty so the guard fires before any
    path-existence check; tests that need to reach ``ensure_runners_started``
    pass a real tmp_path.
    """
    return OrchestratorConfig(
        runner_scaling=RunnerScalingConfig(
            enabled=scaling_enabled,
            managed_root=managed_root,
        ),
        runner_allocation=RunnerAllocationConfig(enabled=allocation_enabled),
    )


def test_run_runners_ensure_started_refuses_when_allocation_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ensure-started must refuse when runner_allocation is enabled (#598).

    ``ensure_runner_running`` relaunches any runner where
    ``not is_runner_launched(...)`` -- exactly the state a deliberately parked
    slot is in. Running ensure-started while allocation is enabled therefore
    restarts every parked listener and silently undoes ``runners allocate``,
    burning a full ``demand_idle_samples`` hysteresis window reconverging.
    The guard refuses and points at the single controller.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(allocation_enabled=True),
    )
    ensure_mock = MagicMock(return_value=(0, []))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is False, (
        "ensure-started must refuse when runner_allocation is enabled, not "
        "silently relaunch parked slots"
    )
    assert "runner_allocation is enabled" in result.message, (
        f"refusal must name runner_allocation: {result.message!r}"
    )
    assert "runners allocate" in result.message, (
        f"refusal must point at the single controller: {result.message!r}"
    )
    assert "--force" in result.message, (
        f"refusal must mention the --force escape hatch: {result.message!r}"
    )
    (
        ensure_mock.assert_not_called(),
        (
            "ensure_runners_started must NOT be called when the guard refuses -- "
            "calling it would relaunch parked slots, which is the exact bug"
        ),
    )


def test_run_runners_ensure_started_force_bypasses_allocation_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--force is the explicit escape hatch past the allocation guard (#598).

    A deliberate manual recovery that ``runners allocate`` cannot do may need
    to relaunch every listener; --force records that intent explicitly so the
    operator owns the reconvergence cost rather than the command silently
    incurring it.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(
            allocation_enabled=True,
            managed_root=str(managed_root),
        ),
    )
    ensure_mock = MagicMock(return_value=(2, ["jc-1: launched", "jc-2: launched"]))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started", "--force"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is True, f"--force must bypass the guard and proceed: {result.message!r}"
    (
        ensure_mock.assert_called_once(),
        ("ensure_runners_started must be called when --force is passed"),
    )


def test_run_runners_ensure_started_proceeds_when_allocation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No guard regression: allocation disabled means ensure-started runs.

    The single-controller guard only applies when allocation is enabled. A
    fleet that has not opted into runner_allocation must keep using
    ensure-started as the recovery path for managed runners that die on
    reboot/logoff -- the guard must not turn into a blanket refusal.
    """
    managed_root = tmp_path / "runners"
    managed_root.mkdir()
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: tmp_path)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *a, **k: _ensure_started_config(
            allocation_enabled=False,
            managed_root=str(managed_root),
        ),
    )
    ensure_mock = MagicMock(return_value=(1, ["jc-1: launched"]))
    monkeypatch.setattr(cli, "ensure_runners_started", ensure_mock)

    args = cli.build_parser().parse_args(["runners", "ensure-started"])
    result = cli.run_runners_ensure_started(args)

    assert result.ok is True, (
        "ensure-started must proceed when runner_allocation is disabled -- "
        "the guard is not a blanket refusal"
    )
    (
        ensure_mock.assert_called_once(),
        ("ensure_runners_started must be called when allocation is disabled"),
    )


# --------------------------------------------------------------------------
# Global --dry-run must survive subcommand parsing
# --------------------------------------------------------------------------


def _iter_subparsers(parser: Any, prefix: tuple[str, ...] = ()) -> Any:
    """Yield (argv_prefix, parser) for every subparser, recursively."""
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield (*prefix, name), sub
                yield from _iter_subparsers(sub, (*prefix, name))


def _dry_run_action(parser: Any) -> Any:
    for action in parser._actions:
        if "--dry-run" in getattr(action, "option_strings", ()):
            return action
    return None


def test_every_subcommand_dry_run_flag_defers_to_the_global_one() -> None:
    """A subcommand-level --dry-run must not clobber the top-level one.

    ``--dry-run`` exists on the top-level parser, so an operator may write it
    before or after the subcommand and both must work. argparse applies a
    subparser's own default *after* the global flag was parsed, so a plain
    ``action="store_true"`` on a subparser silently overwrites True with False.
    Discovered on the live host: ``charlie --dry-run runners allocate`` launched a
    real runner listener and reported its PID.

    Derived from the parser itself rather than a hand-maintained list, so a new
    subcommand that repeats the plain idiom fails this test.
    """
    import argparse

    parser = cli.build_parser()
    offenders = []
    for path, sub in _iter_subparsers(parser):
        action = _dry_run_action(sub)
        if action is not None and action.default is not argparse.SUPPRESS:
            offenders.append(" ".join(path))
        elif "dry_run" in sub._defaults:
            # argparse seeds a namespace from two places -- action defaults and
            # the parser's own ``_defaults`` (what ``set_defaults`` writes) -- and
            # a subparser parses into a *fresh* namespace, then copies every key
            # onto the parent's. So ``set_defaults(dry_run=False)`` overwrites the
            # already-parsed global True exactly like an action default does,
            # while leaving ``action.default is SUPPRESS`` and this test green.
            # Checking both is what makes this guard complete rather than a patch
            # over the one mechanism that happened to bite us.
            offenders.append(" ".join(path) + " (via set_defaults)")

    assert offenders == [], (
        "these subcommands clobber the global --dry-run; route them through "
        f"cli._add_dry_run: {offenders}"
    )

    # The top-level flag must keep a real default so args.dry_run always exists.
    assert _dry_run_action(parser).default is False


def test_set_defaults_clobbers_a_global_flag_too() -> None:
    """Pins the argparse behaviour that the guard above's second check exists for.

    Stdlib-only, no repo coupling: a subparser that never declares ``--dry-run``
    but calls ``set_defaults(dry_run=False)`` still overwrites an already-parsed
    global ``--dry-run``. That is the blind spot in checking ``action.default``
    alone -- there is no action here to inspect.

    If a future Python stops copying subparser defaults over values the parent
    already parsed, this test fails and the guard's ``_defaults`` branch can go.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    leaf = parser.add_subparsers(dest="command").add_parser("leaf")
    leaf.set_defaults(dry_run=False)

    assert _dry_run_action(leaf) is None, "no action to inspect -- that is the point"
    assert parser.parse_args(["--dry-run", "leaf"]).dry_run is False


def test_global_dry_run_reaches_runners_allocate_in_either_position() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(["--dry-run", "runners", "allocate"]).dry_run is True
    assert parser.parse_args(["runners", "allocate", "--dry-run"]).dry_run is True
    assert parser.parse_args(["runners", "allocate"]).dry_run is False


# --------------------------------------------------------------------------
# migrate-state-dir
# --------------------------------------------------------------------------


def _refuse_to_call(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must not be called on this path")


def _fake_migration_plan(tmp_path: Path, *, blocked: bool = False) -> MigrationPlan:
    src_root = tmp_path / "src-state"
    dst_root = tmp_path / "dst-state"
    if blocked:
        child = MigrationChild(
            name="dispatches",
            src_path=src_root / "dispatches",
            dst_path=dst_root / "dispatches",
            disposition="blocked",
            reasons=("2 registered git worktrees nested inside this child: a, b",),
            remediation=(f'git -C "{tmp_path}" worktree move "a" "{dst_root / "a"}"',),
        )
        return MigrationPlan(src_root=src_root, dst_root=dst_root, children=(child,))
    child = MigrationChild(
        name="issues",
        src_path=src_root / "issues",
        dst_path=dst_root / "issues",
        disposition="move",
    )
    return MigrationPlan(src_root=src_root, dst_root=dst_root, children=(child,))


def _migrate_args(repo: Path, *extra: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(["--repo", str(repo), "migrate-state-dir", *extra])


def test_migrate_state_dir_parser_defaults_plan_only_and_apply_flips_it() -> None:
    """Requirement 1: acting requires the explicit ``--apply`` opt-in."""
    parser = cli.build_parser()

    plan_only = parser.parse_args(["migrate-state-dir"])
    assert plan_only.apply is False
    assert plan_only.src is None
    assert plan_only.dst is None
    assert plan_only.quiesce_patterns is None

    applied = parser.parse_args(["migrate-state-dir", "--apply"])
    assert applied.apply is True


def test_migrate_state_dir_src_equals_dst_reports_already_migrated_without_planning(
    tmp_path: Path,
) -> None:
    """A repo that never overrode ``runtime.state_dir`` has src == dst by
    default -- there is nothing to migrate, and the planner must never run.
    """
    repo = _make_repo(tmp_path)
    args = _migrate_args(repo)

    result = cli.run_migrate_state_dir_command(args, planner=_refuse_to_call)

    assert result.ok is True
    assert "already migrated" in result.message
    assert result.data["already_migrated"] is True

    # Both derived roots must be fully resolved, not just joined. If dst were a
    # bare join (as ``layout.default_state_root`` alone returns) while src goes
    # through ``runtime_paths`` (which calls ``.resolve()``), a repo whose
    # ``.var`` is a symlink/junction would make this comparison see two
    # different-looking paths for the same on-disk location -- the false
    # negative this short circuit exists to prevent.
    expected = (repo / ".var" / "charlie-work").resolve()
    assert Path(result.data["src_root"]) == expected
    assert Path(result.data["dst_root"]) == expected


def test_migrate_state_dir_overridden_runtime_state_dir_derives_distinct_roots(
    tmp_path: Path,
) -> None:
    """The actual scenario the command exists for: a repo whose
    ``orchestrator.config.yaml`` still points ``runtime.state_dir`` at a
    legacy location. With no ``--src``/``--dst`` given, derivation must
    produce two distinct, fully-resolved roots and hand them both to the
    planner -- exercising the real default derivation end-to-end rather than
    only the short-circuit-when-equal path covered above.
    """
    repo = tmp_path
    (repo / ".git").mkdir()
    legacy = repo / "legacy-orchestrator-state"
    legacy.mkdir()
    (legacy / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    (repo / "orchestrator.config.yaml").write_text(
        "runtime:\n  state_dir: legacy-orchestrator-state\n", encoding="utf-8"
    )
    args = _migrate_args(repo)
    captured: dict[str, Path] = {}

    def _capture_and_plan(*, repo_root: Path, src_root: Path, dst_root: Path) -> MigrationPlan:
        captured["src_root"] = src_root
        captured["dst_root"] = dst_root
        return _fake_migration_plan(tmp_path)

    result = cli.run_migrate_state_dir_command(args, planner=_capture_and_plan)

    # Assert against what was actually handed to the planner -- not
    # ``result.data``, which reports the *plan's own* ``src_root``/``dst_root``
    # (here the fake plan's fixture paths, unrelated to the derivation this
    # test exercises).
    expected_src = legacy.resolve()
    expected_dst = (repo / ".var" / "charlie-work").resolve()
    assert captured["src_root"] == expected_src
    assert captured["dst_root"] == expected_dst
    assert captured["src_root"] != captured["dst_root"]
    assert "already_migrated" not in result.data


def test_migrate_state_dir_apply_refuses_without_any_quiesce_pattern(tmp_path: Path) -> None:
    """Fail-closed: no built-in pattern list exists (CLAUDE.md rule 9), so
    ``--apply`` with none supplied cannot prove quiescence and must refuse.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(repo, "--src", str(src), "--dst", str(dst), "--apply")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "no --quiesce-pattern given" in result.message


def test_migrate_state_dir_apply_refuses_when_quiesce_not_ok(tmp_path: Path) -> None:
    """Requirement 3: ``--apply`` calls ``check_quiescence`` and refuses to
    act when the fleet is not quiescent, saying exactly why.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    not_quiescent = QuiesceReport(
        ok=False,
        matched=(),
        excluded_pids=frozenset(),
        summary="NOT quiescent: 1 matching process(es)",
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: not_quiescent,
        actuator=_refuse_to_call,
    )

    assert result.ok is False
    assert "fleet is not quiescent" in result.message
    assert "NOT quiescent: 1 matching process(es)" in result.message


def test_migrate_state_dir_apply_refuses_when_plan_has_blocked_children(tmp_path: Path) -> None:
    """Requirement 5: on refusal, the blocked child's name, reasons, and
    remediation command lines must appear VERBATIM -- the operator runs them
    by hand.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    plan = _fake_migration_plan(tmp_path, blocked=True)
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: plan,
        quiescence_checker=lambda **kwargs: quiescent,
    )

    assert result.ok is False
    (blocked_child,) = plan.blocked
    assert blocked_child.name in result.message
    for reason in blocked_child.reasons:
        assert reason in result.message
    for step in blocked_child.remediation:
        assert step in result.message


def test_migrate_state_dir_dry_run_succeeds_even_when_quiesce_not_ok(tmp_path: Path) -> None:
    """Requirement 3: a dry run must keep working on a live fleet -- quiesce
    is reported informationally only and never gates a non-``--apply`` run.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo, "--src", str(src), "--dst", str(dst), "--quiesce-pattern", "fleet supervise"
    )
    assert args.apply is False
    not_quiescent = QuiesceReport(
        ok=False,
        matched=(),
        excluded_pids=frozenset(),
        summary="NOT quiescent: 1 matching process(es)",
    )

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: not_quiescent,
        actuator=_refuse_to_call,
    )

    assert result.ok is True
    assert "NOT quiescent" in result.message


def test_migrate_state_dir_dry_run_flag_overrides_apply_and_never_actuates(tmp_path: Path) -> None:
    """The global/subcommand ``--dry-run`` overrides ``--apply``, and this
    path never touches quiescence gating either -- it is still a non-acting
    path.
    """
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(repo, "--src", str(src), "--dst", str(dst), "--apply", "--dry-run")

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        actuator=_refuse_to_call,
    )

    assert result.ok is True
    assert "--apply ignored" in result.message


def test_migrate_state_dir_apply_happy_path_actuates_when_quiescent(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    src, dst = tmp_path / "src-state", tmp_path / "dst-state"
    args = _migrate_args(
        repo,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--apply",
        "--quiesce-pattern",
        "fleet supervise",
    )
    quiescent = QuiesceReport(ok=True, matched=(), excluded_pids=frozenset(), summary="quiescent")
    outcome = MigrationOutcome(ok=True, moved=("issues",))

    result = cli.run_migrate_state_dir_command(
        args,
        planner=lambda **kwargs: _fake_migration_plan(tmp_path),
        quiescence_checker=lambda **kwargs: quiescent,
        actuator=lambda plan_arg: outcome,
    )

    assert result.ok is True
    assert "moved 1 children" in result.message


def _fake_repo(root: Path) -> Path:
    """A directory git's fallback resolution will treat as a work-tree root.

    `git rev-parse` fails inside it (no HEAD), which drives `find_repo_root`
    into its documented `.git`-walking fallback — deterministic and offline.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").mkdir(exist_ok=True)
    return root


def test_config_from_another_repo_is_refused(tmp_path: Path) -> None:
    """Issue #895: --config selects the config, never the state.

    The real incident: `charlie --config <job-cannon> tripwire ack 1392` run from
    a charlie-work cwd wrote job-cannon's ack into charlie-work's state, exit 0,
    while job-cannon's finding kept pinning ok=False.
    """
    repo_a = _fake_repo(tmp_path / "charlie-work")
    repo_b = _fake_repo(tmp_path / "job-cannon")
    foreign_config = repo_b / "orchestrator.config.yaml"
    foreign_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        cli._assert_config_repo_matches(foreign_config, repo_a)

    message = str(excinfo.value)
    assert str(repo_b) in message
    assert str(repo_a) in message
    # The error must carry the corrective invocation, not just the diagnosis.
    assert "--repo" in message


def test_config_from_the_same_repo_is_allowed(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path / "charlie-work")
    own_config = repo / "orchestrator.config.yaml"
    own_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    cli._assert_config_repo_matches(own_config, repo)  # must not raise


def test_config_outside_any_git_repo_is_allowed(tmp_path: Path) -> None:
    """A shared/layered config legitimately lives outside the repo it configures.

    The gate fires only when the config provably belongs to a *different* work
    tree — otherwise this would break exactly the deployment shape it must not
    touch.
    """
    repo = _fake_repo(tmp_path / "charlie-work")
    shared = tmp_path / "shared-config"
    shared.mkdir()
    loose_config = shared / "orchestrator.config.yaml"
    loose_config.write_text("runtime:\n  state_dir: .var/charlie-work\n", encoding="utf-8")

    cli._assert_config_repo_matches(loose_config, repo)  # must not raise


def test_no_config_flag_is_allowed(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path / "charlie-work")
    cli._assert_config_repo_matches(None, repo)  # must not raise


# --------------------------------------------------------------------------
# runners shadow-status (issue #909)
# --------------------------------------------------------------------------


def _shadow_status_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, argparse.Namespace]:
    """Wire find_repo_root/load_layered_config to a fresh tmp repo + fleet dir.

    Returns (repo_root, fleet_directory, args) so callers can write directly
    to the exact paths the command under test will read.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    fleet_directory = tmp_path / "fleet"
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False: repo_root)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_directory), "runners", "shadow-status"]
    )
    return repo_root, fleet_directory, args


def _allocation_state_file(repo_root: Path) -> Path:
    paths = runtime_paths(repo_root, OrchestratorConfig().runtime.state_dir)
    return paths.state_file


def _write_allocation_event(state_file: Path, *, source: str, actuating_planner: str) -> None:
    """Append one runner_allocation event, matching runner_allocation_pass.py's
    real payload shape (source/actuating_planner live inside the payload, not
    as columns -- issue #909 trap 1)."""
    log_event(
        state_file,
        "runner_allocation",
        {
            "source": source,
            "actuating_planner": actuating_planner,
            "budget": 8,
            "budget_reason": "test",
            "changes": [],
            "applied": [],
            "dry_run": False,
            "notes": [],
            "targets": [],
        },
    )


def _journal_record(
    pass_id: str, *, agreed: bool, changes: list[Any] | None = None
) -> dict[str, Any]:
    """One diff-journal record, matching ci_fleet.diff_journal's write shape."""
    return {
        "pass_id": pass_id,
        "inputs": {},
        "live_plan": {
            "budget": 1,
            "budget_reason": "test",
            "targets": [],
            "changes": changes or [],
            "notes": [],
        },
        "shadow_plan": {
            "budget": 1,
            "budget_reason": "test",
            "targets": [],
            "changes": [],
            "notes": [],
        },
        "agreed": agreed,
        "outcome": "ok",
        "differences": [],
        "shadow_ms": 1.0,
    }


def _write_journal(fleet_directory: Path, records: list[dict[str, Any]]) -> Path:
    fleet_directory.mkdir(parents=True, exist_ok=True)
    journal_file = fleet_directory / "shadow-planner-diff.jsonl"
    with journal_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return journal_file


def test_run_runners_shadow_status_reports_not_found_for_missing_stores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Neither store exists yet: both must be reported as explicitly missing.

    A silent empty result is exactly the failure #909 exists to prevent -- an
    operator must be able to tell "nothing recorded yet" apart from "the
    report is broken and came back empty". Also asserts the command does not
    create either store as a side effect of looking for it: sqlite3.connect()
    creates an empty file on open, which would make a "read-only reporter"
    a lie the first time it runs against a fresh repo.
    """
    repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)

    result = cli.run_runners_shadow_status(args)

    assert result.ok is True
    assert result.data["events_db"]["found"] is False
    assert result.data["journal"]["found"] is False
    assert result.data["actuating"] is None
    assert result.data["configured_not_yet_in_effect"] is None
    assert result.data["agreement_streak"] is None
    assert result.data["change_agreement_streak"] is None
    assert result.data["gate"] is None
    assert not _allocation_state_file(repo_root).parent.joinpath("events.db").exists(), (
        "looking for events.db must not create it"
    )
    assert not fleet_directory.exists(), "looking for the journal must not create the fleet dir"


def test_run_runners_shadow_status_actuating_from_latest_prologue_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """actuating must reflect the LATEST source=prologue row, not the first."""
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="legacy")
    _write_allocation_event(state_file, source="prologue", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["events_db"]["found"] is True
    assert result.data["actuating"]["planner"] == "new"
    assert result.data["configured_not_yet_in_effect"] is None


def test_run_runners_shadow_status_newer_cli_row_is_pending_not_actuating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trap 2 (#909): a newer source='cli' row must never read as confirmation.

    Exact scenario the issue names: an operator flips config and runs a
    dry-run ``charlie runners allocate`` (source='cli') naming the new
    planner, but the supervisor caches config at startup and is still
    actuating the old one until it respawns. 'actuating' must stay 'legacy'
    -- the cli row must appear only in the separate, explicitly-pending
    field. Discriminates against a naive "most recent runner_allocation
    event regardless of source" implementation, which would report 'new'
    here instead.
    """
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="prologue", actuating_planner="legacy")
    _write_allocation_event(state_file, source="cli", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["actuating"]["planner"] == "legacy", (
        "the supervisor has not respawned yet -- actuating must not jump to the cli row's planner"
    )
    assert result.data["configured_not_yet_in_effect"] is not None
    assert result.data["configured_not_yet_in_effect"]["planner"] == "new"
    assert (
        result.data["actuating"]["planner"]
        != result.data["configured_not_yet_in_effect"]["planner"]
    ), "the two fields must disagree here -- that disagreement is the whole point of trap 2"


def test_run_runners_shadow_status_older_cli_row_is_not_surfaced_as_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale cli pre-check must not resurface once the supervisor catches up.

    Inverse of the trap-2 test: the cli row is written FIRST (the dry-run
    pre-check) and a prologue row confirming the supervisor already picked up
    the same planner is written after. Once that has happened, the old cli
    row is history, not a pending change, and must not be shown as one.
    """
    repo_root, _fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    state_file = _allocation_state_file(repo_root)
    _write_allocation_event(state_file, source="cli", actuating_planner="new")
    _write_allocation_event(state_file, source="prologue", actuating_planner="new")

    result = cli.run_runners_shadow_status(args)

    assert result.data["actuating"]["planner"] == "new"
    assert result.data["configured_not_yet_in_effect"] is None


def test_run_runners_shadow_status_change_streak_excludes_noop_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The load-bearing streak counts only passes that emitted a real change.

    ``live_plan`` is a dict (keys budget/budget_reason/changes/notes/targets)
    that is always truthy -- a naive ``if record.get("live_plan")`` check
    would count every no-op pass as "real" and this test would see
    change_agreement_streak == {"streak": 5, "total": 5}, identical to the
    all-passes streak, which defeats the entire point of reporting it
    separately. Only ``live_plan["changes"]`` (a list) says whether a pass
    actually emitted a change.
    """
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    real_change = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=[]),
            _journal_record("p2", agreed=True, changes=[]),
            _journal_record("p3", agreed=True, changes=[]),
            _journal_record("p4", agreed=True, changes=real_change),
            _journal_record("p5", agreed=True, changes=[]),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["journal"]["found"] is True
    assert result.data["agreement_streak"] == {"streak": 5, "total": 5}
    assert result.data["change_agreement_streak"] == {"streak": 1, "total": 1}, (
        "must count only the one changed pass, not be inflated by the 4 surrounding no-op passes"
    )


def test_run_runners_shadow_status_streaks_break_on_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both streaks are the TRAILING agreed run, and a disagreement resets it."""
    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    change = [{"repo": "x/y", "runner": "r-1", "action": "start", "reason": "test"}]
    _write_journal(
        fleet_directory,
        [
            _journal_record("p1", agreed=True, changes=change),
            _journal_record("p2", agreed=False, changes=change),
            _journal_record("p3", agreed=True, changes=change),
        ],
    )

    result = cli.run_runners_shadow_status(args)

    assert result.data["agreement_streak"] == {"streak": 1, "total": 3}
    assert result.data["change_agreement_streak"] == {"streak": 1, "total": 3}


def test_run_runners_shadow_status_gate_uses_real_ci_fleet_evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate verdict is ci_fleet's real §6.3 evaluate(), not a fabricated field.

    Proven by using REQUIRED_STREAK's real value (200): 3 agreed passes must
    NOT open the gate. A fabricated/hardcoded ``ok: True`` would pass this
    test's shape but fail this specific assertion.
    """
    from ci_fleet.shadow_gate import REQUIRED_STREAK

    _repo_root, fleet_directory, args = _shadow_status_setup(monkeypatch, tmp_path)
    _write_journal(
        fleet_directory,
        [_journal_record(f"p{i}", agreed=True, changes=[]) for i in range(3)],
    )

    result = cli.run_runners_shadow_status(args)

    gate = result.data["gate"]
    assert gate is not None
    assert gate["streak"] == 3
    assert gate["streak_required"] == REQUIRED_STREAK
    assert REQUIRED_STREAK > 3, "sanity: the fixture must be short of the real threshold"
    assert gate["streak_ok"] is False
    assert gate["ok"] is False, "3/200 must not open the gate"
    assert "GATE CLOSED" in gate["report"]


def test_shadow_status_subcommand_is_registered() -> None:
    args = cli.build_parser().parse_args(["runners", "shadow-status"])
    assert args.command == "runners"
    assert args.runners_command == "shadow-status"


def test_main_dispatches_runners_shadow_status(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = CommandResult(ok=True, message="runners shadow-status complete", data={})
    mock = MagicMock(return_value=mock_result)
    monkeypatch.setattr(cli, "run_runners_shadow_status", mock)

    exit_code = cli.main(["runners", "shadow-status"])

    assert exit_code == 0
    mock.assert_called_once()
