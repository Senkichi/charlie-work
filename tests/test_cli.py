from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.config import NotifyConfig
from charlie_work.fleet_dispatch import ApiWorkerFleetReport
from charlie_work.fleet_paths import fleet_dir
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
