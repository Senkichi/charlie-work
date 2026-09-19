"""CLI surface and GitHub CLI invocation helpers: ``charlie`` argument parsing / exit-code mapping, ``github.run`` allow-failure handling, and ``merge_pr`` argv construction.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from _fakes_github import FakeGitHub
from charlie_work import cli, github as github_module
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_cli_accepts_json_after_subcommand(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_app", lambda args: object())
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda app, args: cli.CommandResult(True, "ok", {"json_output": args.json_output}),
    )

    assert cli.main(["roll-call", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["json_output"] is True


def test_cli_tripwire_ack_writes_state(monkeypatch, tmp_path: Path) -> None:
    """`charlie tripwire ack <pr> --reason ...` persists the ack through the CLI (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(
        ["tripwire", "ack", "1408", "--reason", "root cause fixed in #672", "--by", "operator"]
    )
    assert exit_code == 0

    acks = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]
    assert acks["1408"]["reason"] == "root cause fixed in #672"
    assert acks["1408"]["by"] == "operator"


def test_cli_tripwire_ack_requires_reason(monkeypatch, capsys, tmp_path: Path) -> None:
    """`charlie tripwire ack` without --reason exits non-zero and writes nothing (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    monkeypatch.setattr(cli, "build_app", lambda args: app)

    exit_code = cli.main(["tripwire", "ack", "1408"])
    assert exit_code == 1
    assert UNAUTHORIZED_MERGE_ACK_KEY not in load_state(paths.state_file)


def test_cli_routes_reconcile_fix_flag(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    class StubApp:
        def reconcile(self, *, fix: bool = False):
            seen["fix"] = fix
            return cli.CommandResult(True, "ok", {})

    monkeypatch.setattr(cli, "build_app", lambda args: StubApp())

    assert cli.main(["mop-up", "--fix"]) == 0
    assert seen["fix"] is True


# --- --repo path validation ----------------------------------------------------


def test_cli_repo_nonexistent_path_errors(tmp_path: Path, capsys) -> None:
    """charlie --repo <nonexistent> must error cleanly (exit 2), not create dirs."""
    ghost = tmp_path / "ghost-repo"
    assert not ghost.exists()

    exit_code = cli.main(["--repo", str(ghost), "roll-call"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "ghost-repo" in err or "--repo" in err
    # Must NOT have created the phantom directory.
    assert not ghost.exists()


def test_cli_main_maps_github_error_to_exit_2(monkeypatch, capsys) -> None:
    from charlie_work.github import GitHubError as _GitHubError

    def _boom(args):
        raise _GitHubError("boom")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "GitHub error: boom" in capsys.readouterr().err


def test_cli_main_maps_config_error_to_exit_2(tmp_path: Path, monkeypatch, capsys) -> None:
    """Issue #12: ConfigError (e.g., unknown top-level section) yields exit 2."""
    from charlie_work.config import ConfigError as _ConfigError

    def _boom(args):
        raise _ConfigError("unknown config section(s): auto-merge")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "config error: unknown config section(s): auto-merge" in capsys.readouterr().err


def test_cli_main_maps_yaml_error_to_exit_2(tmp_path: Path, monkeypatch, capsys) -> None:
    """Issue #12: YAMLError (malformed config) yields exit 2."""

    def _boom(args):
        raise yaml.YAMLError("malformed YAML")

    monkeypatch.setattr(cli, "build_app", _boom)

    assert cli.main(["roll-call"]) == 2
    assert "YAML error: malformed YAML" in capsys.readouterr().err


def test_cli_build_app_registers_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration test: cli.build_app registers repo in fleet.json."""
    from charlie_work.cli import build_app
    from charlie_work.github import GitHub

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir()  # Make it a git repo

    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return "owner/repo"

        def validate_field_lists(self) -> None:
            # build_app is an integration test for fleet.json registration; the
            # gh --json field-list probe needs no real GitHub CLI here.
            pass

    # Monkeypatch GitHub to use our fake
    def fake_github(
        repo_root: Path, dry_run: bool = False, runtime: object | None = None
    ) -> GitHub:
        return FakeGitHub(repo_root=repo_root, dry_run=dry_run)

    monkeypatch.setattr("charlie_work.cli.GitHub", fake_github)

    # Redirect fleet_dir resolution to tmp_path via the env var fleet_paths.fleet_dir()
    # itself supports. Patching the module-level name directly no longer works since
    # fleet_registry composes fleet paths through layout.py, which binds its own
    # reference to fleet_paths.fleet_dir at import time.
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))

    # Build args
    import argparse

    args = argparse.Namespace(repo=repo_root, config=None, dry_run=False, fleet_dir=None)

    # Call build_app
    build_app(args)

    # Verify fleet.json was created
    fleet_json_path = tmp_path / "fleet" / "fleet.json"
    assert fleet_json_path.exists()

    # Verify registry entry
    import json

    registry = json.loads(fleet_json_path.read_text(encoding="utf-8"))
    assert "owner/repo" in registry["repos"]
    entry = registry["repos"]["owner/repo"]
    assert entry["repo_root"] == str(repo_root)
    assert entry["name_with_owner"] == "owner/repo"


def test_github_run_parses_allow_failure_json_stdout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout='[{"name": "Tests passed", "state": "FAILURE"}]',
            stderr="checks failed",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(
        ["pr", "checks", "123"], json_output=True, allow_failure=True
    )

    # allow_failure=True now returns a structured result with an ok flag.
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.value == [{"name": "Tests passed", "state": "FAILURE"}]


def test_github_run_allow_failure_returns_result_for_success(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"number": 123}',
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(
        ["pr", "view", "123"], json_output=True, allow_failure=True
    )

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == {"number": 123}


def test_github_run_allow_failure_text_value_on_success(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="diff text",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(["pr", "diff", "123"], allow_failure=True)

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == "diff text"


def test_github_merge_pr_argv_with_merge_flags(monkeypatch, tmp_path: Path) -> None:
    """Test that merge_flags are correctly passed to gh pr merge."""
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merge_pr(123, "squash", admin=False, merge_flags=("--auto", "--subject"))

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--auto", "--subject", "--squash"]
    assert args[0] == "gh"
    assert args[1:4] == ["pr", "merge", "123"]
    assert "--auto" in args
    assert "--subject" in args
    assert "--squash" in args
    # Verify merge_flags come before strategy flag
    auto_idx = args.index("--auto")
    subject_idx = args.index("--subject")
    squash_idx = args.index("--squash")
    assert auto_idx < squash_idx
    assert subject_idx < squash_idx


def test_github_merge_pr_argv_with_admin_flag(monkeypatch, tmp_path: Path) -> None:
    """Test that legacy admin flag is passed when merge_flags is empty."""
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    gh.merge_pr(123, "squash", admin=True, merge_flags=())

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--admin", "--squash"]
    assert args[0] == "gh"
    assert args[1:4] == ["pr", "merge", "123"]
    assert "--admin" in args
    assert "--squash" in args


def test_github_merge_pr_argv_merge_flags_precedence(monkeypatch, tmp_path: Path) -> None:
    """Test that merge_flags takes precedence over admin flag.

    Uses a legal non-managed flag (--auto) with admin=True to ensure the
    precedence logic is observable (the argv differs depending on which wins).
    """
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    # Both admin=True and merge_flags set; merge_flags should win
    gh.merge_pr(123, "squash", admin=True, merge_flags=("--auto",))

    assert len(captured_args) == 1
    args = captured_args[0]
    # Expected: ["gh", "pr", "merge", "123", "--auto", "--squash"]
    # merge_flags wins, so --auto is present and --admin is NOT present
    assert "--auto" in args
    assert "--admin" not in args
    assert "--squash" in args
    # Verify exact order: merge_flags before strategy flag
    auto_idx = args.index("--auto")
    squash_idx = args.index("--squash")
    assert auto_idx < squash_idx


def test_github_merge_pr_flags_are_orchestrator_managed(monkeypatch, tmp_path: Path) -> None:
    """Invariant: every flag merge_pr appends is in ORCHESTRATOR_MANAGED_MERGE_FLAGS.

    This gate ensures that removing a flag from the constant derivation fails tests
    on BOTH the validation side (config.py) and the argv side (merge_pr), preventing
    the drift issue #107 where merge_pr could add flags without config validation
    rejecting them.
    """
    captured_args = []

    def fake_run(cmd, *args, **kwargs):
        captured_args.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    strategies = ["merge", "squash", "rebase"]

    for strategy in strategies:
        for admin in (False, True):
            captured_args.clear()
            gh.merge_pr(123, strategy, admin=admin, merge_flags=())

            assert len(captured_args) == 1
            args = captured_args[0]

            # Extract flags (skip "gh", "pr", "merge", and the PR number)
            flags = [arg for arg in args if arg.startswith("--")]

            # Every flag merge_pr appends must be in ORCHESTRATOR_MANAGED_MERGE_FLAGS
            for flag in flags:
                assert flag in github_module.ORCHESTRATOR_MANAGED_MERGE_FLAGS, (
                    f"Flag {flag} appended by merge_pr(strategy={strategy}, admin={admin}) "
                    f"is not in ORCHESTRATOR_MANAGED_MERGE_FLAGS"
                )
