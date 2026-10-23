"""First-class repo retirement tests (issue #2754).

Covers the per-repo ``fleet.retired`` key, the derived registry field, the
wind-down/skip behaviour in ``fleet_loop``, the fleet-wide enumeration
contract, and the ``charlie fleet retire`` / ``unretire`` commands.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from _fleet_dispatch_fixtures import (  # noqa: F401  (autouse hermeticity)
    _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests,
    _per_repo_runtime_paths,
)
from charlie_work import cli
from charlie_work.config import FleetConfig, LocalIssuesConfig, OrchestratorConfig
from charlie_work.fleet_dispatch import (
    _fleet_has_configured_repos,
    _take_fleet_snapshot,
    _touch_registry_last_seen,
    fleet_loop,
)
from charlie_work.fleet_lanes import _run_fleet_reap_sweep
from charlie_work.fleet_registry import (
    _select_repos,
    count_fleet_live_reviews,
    count_fleet_live_sessions,
    managed_repo_names,
    managed_repo_roots,
    touch_repo,
)
from charlie_work.fleet_live_counts import count_fleet_live_sessions_with_adapter_kinds
from charlie_work.paths import runtime_paths
from charlie_work.repo_retirement import (
    entry_is_retired,
    repo_live_work,
    repo_retirement_state,
)
from charlie_work.command_result import CommandResult


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_config(repo: Path, *, retired: bool) -> Path:
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        f"fleet:\n  retired: {str(retired).lower()}\n"
        "local_issues:\n  enabled: true\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )
    return config


def _repo_entry(repo: Path, config: Path, *, state_dir: Path | None = None) -> dict[str, object]:
    state = state_dir or repo / ".var" / "charlie-work"
    state.mkdir(parents=True, exist_ok=True)
    stamp = _now_iso()
    return {
        "repo_root": str(repo),
        "name_with_owner": "local/localrepo",
        "config_path": str(config),
        "state_dir": str(state),
        "first_seen": stamp,
        "last_seen": stamp,
    }


class _FakeGitHub:
    def __init__(self, key: str = "local/localrepo") -> None:
        self._key = key

    def name_with_owner(self) -> str:
        return self._key


def _write_worker_sidecar(repo: Path, *, issue: int = 1) -> Path:
    sessions = repo / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    sidecar = sessions / f"issue-{issue}.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": issue,
                "branch": f"agent/issue-{issue}",
                "worktree_path": str(repo / "worktrees" / f"issue-{issue}"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin"],
                "pid": 1234,
                "started_at": _now_iso(),
                "log_path": str(repo / "log.txt"),
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _write_review_sidecar(repo: Path, *, pr: int = 7) -> Path:
    reviews = repo / ".var" / "charlie-work" / "dispatches" / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    sidecar = reviews / f"issue-{pr}.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": pr,
                "branch": f"agent/issue-{pr}",
                "worktree_path": str(repo / "worktrees" / f"pr-{pr}"),
                "prompt_path": str(repo / "prompt.md"),
                "command": ["devin"],
                "pid": 4321,
                "started_at": _now_iso(),
                "log_path": str(repo / "review.log"),
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_config_retired_defaults_false_and_global_layer_cannot_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fleet-layer ``fleet.retired`` is stripped; the repo layer still wins."""
    from charlie_work.global_config import load_layered_config

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "config.yaml").write_text("fleet:\n  retired: true\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    config = load_layered_config(repo, None, fleet_dir_override=str(fleet_dir))
    assert config.fleet.retired is False

    _write_config(repo, retired=True)
    config = load_layered_config(repo, None, fleet_dir_override=str(fleet_dir))
    assert config.fleet.retired is True


def test_repo_retirement_state_reads_only_the_repo_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert repo_retirement_state(repo) is None
    _write_config(repo, retired=True)
    assert repo_retirement_state(repo) is True


def test_touch_repo_keeps_retirement_and_bumps_last_seen(tmp_path: Path) -> None:
    """``touch_repo`` never clears retirement; it refreshes it from config."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config = _write_config(repo, retired=True)
    paths = runtime_paths(repo, ".var/charlie-work")
    fleet_dir = tmp_path / "fleet"

    first = touch_repo(str(fleet_dir), repo, paths, _FakeGitHub())  # type: ignore[arg-type]
    entry = first["repos"]["local/localrepo"]
    assert entry["retired"] is True
    first_last_seen = entry["last_seen"]
    assert entry["config_path"] == str(config)

    second = touch_repo(str(fleet_dir), repo, paths, _FakeGitHub())  # type: ignore[arg-type]
    entry = second["repos"]["local/localrepo"]
    assert entry["retired"] is True
    assert entry["last_seen"] >= first_last_seen


def test_registry_contract_pins_existing_keys(tmp_path: Path) -> None:
    """The additive ``retired`` field never removes an existing registry key."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_config(repo, retired=False)
    paths = runtime_paths(repo, ".var/charlie-work")
    entry = touch_repo(str(tmp_path / "fleet"), repo, paths, _FakeGitHub())["repos"][  # type: ignore[arg-type]
        "local/localrepo"
    ]
    assert set(entry) == {
        "repo_root",
        "name_with_owner",
        "config_path",
        "state_dir",
        "first_seen",
        "last_seen",
        "retired",
    }


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_retired_quiescent_repo_launches_nothing(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_report: MagicMock,
    tmp_path: Path,
) -> None:
    """A quiescent retired repo is skipped entirely: no app, no lane, no event."""
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)

    mock_load_registry.return_value = {"repos": {"local/localrepo": entry}}
    mock_load_config.return_value = OrchestratorConfig(
        fleet=FleetConfig(retired=True),
        local_issues=LocalIssuesConfig(enabled=True),
    )
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()
    mock_report.return_value = None

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )

    assert result.data["retired_quiescent"] == ["local/localrepo"]
    assert "local/localrepo" not in result.data["repos"]
    mock_app_class.assert_not_called()


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_flipping_the_key_takes_effect_on_the_next_pass(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_registry: MagicMock,
    mock_report: MagicMock,
    tmp_path: Path,
) -> None:
    """No supervisor restart: the real layered loader sees the flipped key.

    The config is written to the repo's own file and read through the real
    ``load_layered_config`` both passes -- only the file is flipped, so the
    test cannot pass by patching the loader's return value.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=False)
    entry = _repo_entry(repo, config_path)
    mock_load_registry.return_value = {"repos": {"local/localrepo": entry}}
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_gh_class.return_value = MagicMock()
    mock_report.return_value = None
    app = MagicMock()
    app.loop.return_value = CommandResult(True, "loop complete", {})
    mock_app_class.return_value = app

    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )
    app.loop.assert_called_once_with(2, merge=None, deadline_exceeded=ANY)

    _write_config(repo, retired=True)
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        limit=2,
        dry_run=False,
        work_only=False,
    )
    assert result.data["retired_quiescent"] == ["local/localrepo"]
    assert app.loop.call_count == 1


def test_repo_live_work_counts_workers_and_gate_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_worker_sidecar(repo)
    state_dir = repo / ".var" / "charlie-work"
    config = OrchestratorConfig()
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)

    live = repo_live_work(repo, config, state_dir)
    assert live.workers == 1
    assert live.quiescent is False

    gate = state_dir / "dispatches" / "local-merge-gate" / "pr-7"
    gate.mkdir(parents=True)
    (gate / "suite-runner.json").write_text(
        json.dumps({"pid": 999, "process_start_time": 1.0}), encoding="utf-8"
    )
    fake_probe = type("_Probe", (), {"is_alive": lambda self, *_a: True})()
    fake_clock = type("_Clock", (), {"now": lambda self: datetime.now(UTC)})()
    monkeypatch.setattr(
        "charlie_work.repo_retirement._host.current",
        lambda: type("_Host", (), {"probe": fake_probe, "clock": fake_clock})(),
    )
    live = repo_live_work(repo, config, state_dir)
    assert live.gate_suites == 1


def test_fleet_enumerations_ignore_retired_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"local/localrepo": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    assert managed_repo_names(None) == frozenset()
    assert _fleet_has_configured_repos(None, None) is False
    assert _take_fleet_snapshot(fleet_dir_override=None).repo_snapshots == frozenset()


def test_fleet_status_reports_retired_not_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"local/localrepo": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    result = cli.run_fleet_status(cli.build_parser().parse_args(["fleet", "status"]))

    assert result.ok is True
    assert result.data["repos"] == {}
    assert result.data["stale"] == []
    assert result.data["errors"] == []
    assert [row["repo_key"] for row in result.data["retired"]] == ["local/localrepo"]


def test_cli_retire_and_unretire_edit_only_the_repo_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    retire_args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    result = cli.run_fleet_retire(retire_args)
    assert result.ok is True
    assert repo_retirement_state(repo) is True

    dry_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "fleet", "unretire", "--dry-run"]
    )
    result = cli.run_fleet_unretire(dry_args)
    assert result.ok is True
    assert result.data["changed"] is True
    assert repo_retirement_state(repo) is True  # dry-run wrote nothing

    unretire_args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "unretire"])
    result = cli.run_fleet_unretire(unretire_args)
    assert result.ok is True
    assert repo_retirement_state(repo) is False


def test_status_and_doctor_calls_leave_a_retired_repo_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-registration on every CLI call never clears retirement."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    _write_config(repo, retired=True)
    fleet_dir = tmp_path / "fleet"

    monkeypatch.setattr("charlie_work.cli.github_client_for", lambda *_a, **_k: _FakeGitHub())
    monkeypatch.setattr("charlie_work.cli.OrchestratorApp", lambda *_a, **_k: MagicMock())

    status_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "--fleet-dir", str(fleet_dir), "roll-call"]
    )
    cli.build_app(status_args)
    registry = json.loads((fleet_dir / "fleet.json").read_text(encoding="utf-8"))
    assert registry["repos"]["local/localrepo"]["retired"] is True

    monkeypatch.setattr("charlie_work.cli.run_doctor", lambda *_a, **_k: (True, []))
    doctor_args = cli.build_parser().parse_args(
        ["--repo", str(repo), "--fleet-dir", str(fleet_dir), "doctor"]
    )
    cli.run_doctor_command(doctor_args)
    registry = json.loads((fleet_dir / "fleet.json").read_text(encoding="utf-8"))
    assert registry["repos"]["local/localrepo"]["retired"] is True


def test_select_repos_exposes_the_derived_retired_field(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = _write_config(repo, retired=True)
    entry = _repo_entry(repo, config_path)
    registry = {"repos": {"local/localrepo": entry}}

    selected = _select_repos(registry, None)

    assert selected[0][1]["retired"] is True
    assert entry_is_retired(entry) is True


def test_repo_retirement_state_is_total_for_garbage_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "orchestrator.config.yaml").write_text("fleet: [unterminated\n", encoding="utf-8")

    assert repo_retirement_state(repo) is None
    assert entry_is_retired({"repo_root": str(repo), "retired": True}) is True


def test_enumerations_keep_going_when_one_repo_config_is_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A garbage config must not abort a read-only fleet walk (issue #2754)."""
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    bad_config = bad / "orchestrator.config.yaml"
    bad_config.write_text("fleet: [unterminated\n", encoding="utf-8")
    good_config = _write_config(good, retired=False)
    bad_entry = _repo_entry(bad, bad_config)
    good_entry = _repo_entry(good, good_config)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {"owner/bad": bad_entry, "owner/good": good_entry},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    selected = _select_repos({"repos": {"owner/bad": bad_entry, "owner/good": good_entry}}, None)

    assert {key for key, _entry in selected} == {"owner/bad", "owner/good"}
    assert managed_repo_names(None) == frozenset({"bad", "good"})
    assert set(managed_repo_roots(None)) == {"bad", "good"}


def test_managed_repo_roots_skips_retired_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = tmp_path / "active"
    retired = tmp_path / "retired"
    active.mkdir()
    retired.mkdir()
    active_entry = _repo_entry(active, _write_config(active, retired=False))
    retired_entry = _repo_entry(retired, _write_config(retired, retired=True))
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {"owner/active": active_entry, "owner/retired": retired_entry},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    assert managed_repo_names(None) == frozenset({"active"})
    assert dict(managed_repo_roots(None)) == {"active": active}


def test_live_counts_skip_retired_repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared registry walk skips retired repos for every live counter."""
    active = tmp_path / "active"
    retired = tmp_path / "retired"
    entries: dict[str, dict[str, object]] = {}
    for repo, flag in ((active, False), (retired, True)):
        repo.mkdir()
        (repo / ".git").mkdir()
        config = _write_config(repo, retired=flag)
        _write_worker_sidecar(repo)
        _write_review_sidecar(repo)
        entries[f"owner/{repo.name}"] = _repo_entry(repo, config)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": entries}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)

    assert count_fleet_live_sessions(None) == (1, [])
    assert count_fleet_live_sessions_with_adapter_kinds(None) == (1, {"devin": 1}, [])
    assert count_fleet_live_reviews(None) == (1, [])


def test_non_quiescent_retired_repo_is_excluded_from_fleet_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the intentional exclusion of winding-down live work (issue #2754).

    A retired repo with a live worker and reviewer is still winding down, but
    ``_iter_registered_repo_contexts`` skips it, so that work does not consume
    the fleet-wide session/review budgets. The plan mandates *some* behaviour
    here; this repository keeps the issue's exclusion and flags the
    alternative (counting winding-down work) for the operator in the PR body.
    The flipped-key half is the positive control: the same fixture counts when
    it is not retired, so the zeroes above are the skip, not an empty fixture.
    """
    from charlie_work.fleet_registry import _iter_registered_repo_contexts

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    config_path = _write_config(repo, retired=True)
    _write_worker_sidecar(repo)
    _write_review_sidecar(repo)
    entry = _repo_entry(repo, config_path)
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"owner/retired": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: True)

    state_dir = repo / ".var" / "charlie-work"
    assert repo_live_work(repo, OrchestratorConfig(), state_dir).total == 2
    assert list(_iter_registered_repo_contexts(None, [])) == []
    assert count_fleet_live_sessions(None) == (0, [])
    assert count_fleet_live_reviews(None) == (0, [])

    # Positive control: the same fixture is counted once it is not retired.
    _write_config(repo, retired=False)
    assert count_fleet_live_sessions(None) == (1, [])
    assert count_fleet_live_reviews(None) == (1, [])


def test_run_fleet_reap_sweep_skips_retired_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    entry = _repo_entry(repo, _write_config(repo, retired=True))
    monkeypatch.setattr(
        "charlie_work.fleet_lanes._load_registry",
        lambda _path: {"repos": {"owner/retired": entry}},
    )
    called: list[object] = []
    monkeypatch.setattr(
        "charlie_work.fleet_lanes.OrchestratorApp",
        lambda *args, **kwargs: called.append(args),
    )

    assert _run_fleet_reap_sweep(fleet_dir_override=str(tmp_path / "fleet")) == {}
    assert called == []


def test_config_retirement_sweep_skips_retired_repo_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.config_retirement_sweep import _enumerate_layer_slots

    repo = tmp_path / "repo"
    repo.mkdir()
    entry = _repo_entry(repo, _write_config(repo, retired=True))
    selected = _select_repos({"repos": {"owner/retired": entry}}, None)

    def _must_not_run(*_a: object, **_k: object) -> object:
        raise AssertionError("retired repo layers must not be enumerated")

    slots, blocked = _enumerate_layer_slots(
        selected,
        fleet_dir_override=str(tmp_path / "fleet"),
        run_git=_must_not_run,
    )

    assert [slot.layer for slot in slots] == ["user-global"]
    assert blocked == []


def test_fleet_queues_skip_retired_repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    entry = _repo_entry(repo, _write_config(repo, retired=True))
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {"owner/retired": entry}}), encoding="utf-8"
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    called: list[object] = []
    monkeypatch.setattr(
        "charlie_work.cli.OrchestratorApp",
        lambda *args, **kwargs: called.append(args),
    )

    for command in ("review-queue", "operator-queue"):
        args = cli.build_parser().parse_args(["fleet", command])
        result = (
            cli.run_fleet_review_queue(args)
            if command == "review-queue"
            else cli.run_fleet_operator_queue(args)
        )
        assert result.ok is True
        assert result.data == {"repos": {}, "errors": []}

    assert called == []


def test_touch_registry_last_seen_persists_the_retired_flag(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    entry = {
        "repo_root": str(tmp_path / "repo"),
        "last_seen": "2000-01-01T00:00:00Z",
    }
    fleet_json = fleet_dir / "fleet.json"
    fleet_json.write_text(
        json.dumps({"version": 1, "repos": {"owner/retired": entry}}), encoding="utf-8"
    )

    _touch_registry_last_seen(
        fleet_json,
        ["owner/retired"],
        datetime.now(UTC),
        {"owner/retired": True},
    )

    saved = json.loads(fleet_json.read_text(encoding="utf-8"))
    assert saved["repos"]["owner/retired"]["retired"] is True
    assert saved["repos"]["owner/retired"]["last_seen"] != "2000-01-01T00:00:00Z"


def test_fleet_retire_preserves_comments_and_unrelated_content(tmp_path: Path) -> None:
    """The command edits only the ``fleet.retired`` line, never a YAML dump."""
    import yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "# operator notes: do not reflow this file\n"
        "fleet:\n"
        "  retired: false  # flipped by hand during incidents\n"
        "  global_max_concurrent_sessions: 3\n"
        "\n"
        "# dispatch tuning\n"
        "dispatch:\n"
        "  default_limit: 2\n",
        encoding="utf-8",
    )
    before_text = config.read_text(encoding="utf-8")
    before = yaml.safe_load(before_text)

    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    result = cli.run_fleet_retire(args)

    assert result.ok is True
    after_text = config.read_text(encoding="utf-8")
    after = yaml.safe_load(after_text)
    assert after == {**before, "fleet": {**before["fleet"], "retired": True}}
    assert "# operator notes: do not reflow this file" in after_text
    assert "# dispatch tuning" in after_text
    assert "retired: true  # flipped by hand during incidents" in after_text
    changed = [
        (old, new)
        for old, new in zip(before_text.splitlines(), after_text.splitlines(), strict=True)
        if old != new
    ]
    assert changed == [
        (
            "  retired: false  # flipped by hand during incidents",
            "  retired: true  # flipped by hand during incidents",
        )
    ]


def test_fleet_retire_inserts_the_flag_into_an_existing_block(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "# keep me\nfleet:\n  global_max_concurrent_sessions: 3\n# below\n",
        encoding="utf-8",
    )

    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    assert cli.run_fleet_retire(args).ok is True

    lines = config.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "# keep me",
        "fleet:",
        "  global_max_concurrent_sessions: 3",
        "  retired: true",
        "# below",
    ]


def _retire_repo(repo: Path) -> CommandResult:
    args = cli.build_parser().parse_args(["--repo", str(repo), "fleet", "retire"])
    return cli.run_fleet_retire(args)


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_fleet_retire_rewrites_a_single_line_flow_mapping_keeping_siblings(
    tmp_path: Path,
) -> None:
    """Issue #2754 rework: flow style must not drop the sibling keys."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet: {global_max_concurrent_sessions: 3, retired: false}\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    after = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert after == {
        "fleet": {"global_max_concurrent_sessions": 3, "retired": True},
        "dispatch": {"default_limit": 2},
    }
    assert repo_retirement_state(repo) is True


def test_fleet_retire_rewrites_a_multiline_flow_mapping(tmp_path: Path) -> None:
    """Issue #2754 rework: a multi-line flow mapping must stay parseable."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet: {\n"
        "  global_max_concurrent_sessions: 3,\n"
        "  retired: false,\n"
        "}\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    assert yaml.safe_load(config.read_text(encoding="utf-8")) == {
        "fleet": {"global_max_concurrent_sessions": 3, "retired": True},
        "dispatch": {"default_limit": 2},
    }
    assert repo_retirement_state(repo) is True


def test_fleet_retire_handles_a_column_zero_comment_before_the_key(tmp_path: Path) -> None:
    """A comment at column 0 inside the fleet block must not split it.

    Before the rework the edit inserted a second ``retired:`` above the
    comment; YAML accepted the duplicate and the last value won, so the
    command reported success while ``fleet.retired`` stayed false.
    """
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text(
        "fleet:\n"
        "# operator note inside the fleet block\n"
        "  retired: false  # flipped during incidents\n"
        "  global_max_concurrent_sessions: 3\n"
        "dispatch:\n  default_limit: 2\n",
        encoding="utf-8",
    )

    result = _retire_repo(repo)

    assert result.ok is True
    after_text = config.read_text(encoding="utf-8")
    assert repo_retirement_state(repo) is True
    assert "# operator note inside the fleet block" in after_text
    assert "retired: true  # flipped during incidents" in after_text
    assert after_text.count("retired:") == 1


def test_fleet_retire_appends_when_the_file_has_no_fleet_key(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text("dispatch:\n  default_limit: 2\n", encoding="utf-8")

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_text(encoding="utf-8") == (
        "dispatch:\n  default_limit: 2\n\nfleet:\n  retired: true\n"
    )
    assert repo_retirement_state(repo) is True


def test_fleet_retire_writes_an_empty_config_file(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_text("", encoding="utf-8")

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_text(encoding="utf-8") == "fleet:\n  retired: true\n"
    assert repo_retirement_state(repo) is True


def test_fleet_retire_is_an_idempotent_no_op(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet:\n  retired: true\n  global_max_concurrent_sessions: 3\n"
    config.write_bytes(before)

    result = _retire_repo(repo)

    assert result.ok is True
    assert result.data["changed"] is False
    assert config.read_bytes() == before


def test_fleet_retire_preserves_crlf_newlines(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    config.write_bytes(
        b"fleet:\r\n  retired: false\r\n  global_max_concurrent_sessions: 3\r\n"
        b"dispatch:\r\n  default_limit: 2\r\n"
    )

    result = _retire_repo(repo)

    assert result.ok is True
    assert config.read_bytes() == (
        b"fleet:\r\n  retired: true\r\n  global_max_concurrent_sessions: 3\r\n"
        b"dispatch:\r\n  default_limit: 2\r\n"
    )
    assert b"\r\r\n" not in config.read_bytes()


def test_fleet_retire_rejects_a_non_mapping_fleet_section(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet: [1, 2]\n"
    config.write_bytes(before)

    result = _retire_repo(repo)

    assert result.ok is False
    assert "fleet must be a mapping" in result.message
    assert config.read_bytes() == before


def test_fleet_retire_rejects_an_unreadable_config_file(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    (repo / "orchestrator.config.yaml").mkdir()

    result = _retire_repo(repo)

    assert result.ok is False
    assert "unreadable config" in result.message


def test_fleet_retire_refuses_a_render_that_would_not_round_trip(tmp_path: Path) -> None:
    """The post-render verification is a write gate, not a decoration."""
    import yaml

    repo = _git_repo(tmp_path)
    config = repo / "orchestrator.config.yaml"
    before = b"fleet:\n  retired: false\n  retired: false\n"
    config.write_bytes(before)
    # PyYAML accepts the duplicate and keeps the last value, so editing only
    # the first line would leave the effective value false.
    assert yaml.safe_load(before.decode())["fleet"]["retired"] is False

    result = _retire_repo(repo)

    assert result.ok is False
    assert "would not preserve" in result.message
    assert config.read_bytes() == before
