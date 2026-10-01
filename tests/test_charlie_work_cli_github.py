"""CLI surface and GitHub CLI invocation helpers: ``charlie`` argument parsing / exit-code mapping, ``github.run`` allow-failure handling, and ``merge_pr`` argv construction.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from _fake_transport import (
    FakeAdapter,
    graphql_ok,
    graphql_variables,
    make_github,
    ok,
    sent,
)
from _fakes_github import FakeGitHub
from charlie_work import cli, github as github_module
from charlie_work.config_validation import ConfigError
from charlie_work.github_transport import GraphQLError, GraphQLRequest, Response, RestRequest
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


def test_github_run_parses_allow_failure_json_stdout(tmp_path: Path) -> None:
    # A GraphQL reply that carries both data and errors is a failure whose body
    # still reaches the caller (#1933): allow_failure returns it parsed.
    partial = Response(
        200,
        (),
        '{"data": {"checks": [{"name": "Tests passed", "state": "FAILURE"}]}}',
        "http",
        graphql_errors=(GraphQLError("checks failed", "FORBIDDEN"),),
    )
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [partial]))

    result = gh.run(
        ["api", "graphql", "-f", "query=query { viewer { login } }"],
        json_output=True,
        allow_failure=True,
    )

    # allow_failure=True now returns a structured result with an ok flag.
    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is False
    assert result.value == {"data": {"checks": [{"name": "Tests passed", "state": "FAILURE"}]}}
    (request,) = http.api_requests
    assert isinstance(request, GraphQLRequest)
    assert "viewer" in request.document


def test_github_run_allow_failure_returns_result_for_success(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"number": 123})]))

    result = gh.run(
        ["api", "repos/{owner}/{repo}/pulls/123"], json_output=True, allow_failure=True
    )

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == {"number": 123}
    assert sent(http) == [("GET", "repos/{owner}/{repo}/pulls/123", None)]


def test_github_run_allow_failure_text_value_on_success(tmp_path: Path) -> None:
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok("diff text")]))

    result = gh.run(
        ["api", "-H", "Accept: application/vnd.github.v3.diff", "repos/{owner}/{repo}/pulls/123"],
        allow_failure=True,
    )

    assert isinstance(result, github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value == "diff text"
    assert http.api_requests[0].accept == "application/vnd.github.v3.diff"  # type: ignore[union-attr]


def _auto_merge_transport() -> FakeAdapter:
    """Answers the node-id read, then the auto-merge mutation."""

    def handler(request):
        if request.document.lstrip().startswith("query"):
            return graphql_ok({"repository": {"pullRequest": {"id": "PR_node_123"}}})
        return graphql_ok({"enablePullRequestAutoMerge": {"pullRequest": {"number": 123}}})

    return FakeAdapter("http", handler=handler)


def test_github_merge_pr_argv_with_merge_flags(tmp_path: Path) -> None:
    """``--auto`` becomes the enable-auto-merge mutation (an id read, then the
    mutation carrying the merge method); a flag outside the closed set is a
    config error (B8)."""
    gh, http, _ = make_github(tmp_path, http=_auto_merge_transport())

    assert gh.merge_pr(123, "squash", admin=False, merge_flags=("--auto",)) == "merged #123"

    read, mutation = http.api_requests
    assert isinstance(read, GraphQLRequest) and isinstance(mutation, GraphQLRequest)
    assert graphql_variables(read)["number"] == 123
    assert "enablePullRequestAutoMerge" in mutation.document
    assert graphql_variables(mutation) == {"id": "PR_node_123", "method": "SQUASH"}

    with pytest.raises(ConfigError, match="--subject"):
        gh.merge_pr(123, "squash", admin=False, merge_flags=("--auto", "--subject"))
    assert len(http.api_requests) == 2  # rejected before anything was sent


def test_github_merge_pr_argv_with_admin_flag(tmp_path: Path) -> None:
    """The legacy admin flag selects the direct REST merge (no ``--admin`` argv)."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"message": "merged"})]))
    gh.merge_pr(123, "squash", admin=True, merge_flags=())

    assert sent(http) == [
        ("PUT", "repos/{owner}/{repo}/pulls/123/merge", {"merge_method": "squash"})
    ]


def test_github_merge_pr_argv_merge_flags_precedence(tmp_path: Path) -> None:
    """Test that merge_flags takes precedence over admin flag.

    Uses a legal non-managed flag (--auto) with admin=True to ensure the
    precedence logic is observable (the requests differ depending on which wins:
    ``--auto`` is the GraphQL mutation, ``--admin`` the direct REST merge).
    """
    gh, http, _ = make_github(tmp_path, http=_auto_merge_transport())

    # Both admin=True and merge_flags set; merge_flags should win
    gh.merge_pr(123, "squash", admin=True, merge_flags=("--auto",))

    assert not [r for r in http.api_requests if isinstance(r, RestRequest)]  # no REST PUT
    assert all(isinstance(r, GraphQLRequest) for r in http.api_requests)
    assert "enablePullRequestAutoMerge" in http.api_requests[-1].document  # type: ignore[union-attr]
    assert graphql_variables(http.api_requests[-1])["method"] == "SQUASH"


def test_github_merge_pr_flags_are_orchestrator_managed(monkeypatch, tmp_path: Path) -> None:
    """Invariant: every flag merge_pr maps is in ORCHESTRATOR_MANAGED_MERGE_FLAGS.

    This gate ensures that removing a flag from the constant derivation fails tests
    on BOTH the validation side (config.py) and the request side (merge_pr), preventing
    the drift issue #107 where merge_pr could add flags without config validation
    rejecting them. The REST merge carries the strategy as ``merge_method``; the
    strategy flag it stands for must stay in the managed set.
    """
    strategies = {"merge": "--merge", "squash": "--squash", "rebase": "--rebase"}

    for strategy, strategy_flag in strategies.items():
        for admin in (False, True):
            gh, http, _ = make_github(
                tmp_path, http=FakeAdapter("http", [ok({"message": "merged"})])
            )
            gh.merge_pr(123, strategy, admin=admin, merge_flags=())

            (call,) = sent(http)
            assert call == (
                "PUT",
                "repos/{owner}/{repo}/pulls/123/merge",
                {"merge_method": strategy},
            )
            # Every flag merge_pr stands for must be in ORCHESTRATOR_MANAGED_MERGE_FLAGS
            for flag in (strategy_flag, *(["--admin"] if admin else [])):
                assert flag in github_module.ORCHESTRATOR_MANAGED_MERGE_FLAGS, (
                    f"Flag {flag} used by merge_pr(strategy={strategy}, admin={admin}) "
                    f"is not in ORCHESTRATOR_MANAGED_MERGE_FLAGS"
                )
