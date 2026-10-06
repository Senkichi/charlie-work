"""Worker worktrees, uv cache and temp on the ci-fleet host I/O volume (host_io_worker)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ci_fleet.host_io.manifest import VolumeInfo

from charlie_work import cli, host_io_worker, layout
from charlie_work.config import ClaudeCodeConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.env_sanitize import sanitize_env
from charlie_work.fleet_dispatch import _repo_state_dirs
from charlie_work.paths import resolved_layout, runtime_paths, worktree_roots
from charlie_work.worktree import WorktreeCleanResult, merge_clean_results

GUID = "\\\\?\\Volume{0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0}\\"
MANIFEST = {
    "schema": 1,
    "volume_guid": GUID,
    "drive": "Y:",
    "ci_root": "Y:\\ci",
    "runner_root": "Y:\\ci\\runners",
    "ci_uv_cache": "Y:\\ci\\uv-cache",
    "worker_root": "Y:\\fleet",
    "worker_uv_cache": "Y:\\fleet\\uv-cache",
    "defender_exclusions": [],
    "script_version": "1.0.0",
    "provisioned_at": "2026-10-10T17:00:00Z",
}
REFS = VolumeInfo(guid_path=GUID, filesystem="ReFS")


def _write_manifest(data: dict | None = None) -> Path:
    path = Path(os.environ["CI_FLEET_HOST_IO_MANIFEST"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MANIFEST if data is None else data), encoding="utf-8")
    return path


@pytest.fixture
def mounted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_io_worker, "probe_volume", lambda drive: REFS)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo-a"
    (repo / ".git").mkdir(parents=True)
    return repo


def _volume_root(repo: Path) -> Path:
    return layout.worktrees_dir(Path("Y:\\fleet") / repo.name)


# -- resolve_worker_io ----------------------------------------------------------


def test_no_manifest_means_no_volume_and_no_note() -> None:
    status = host_io_worker.resolve_worker_io(enabled=True)
    assert status == host_io_worker.WorkerIoStatus(None, "")


def test_a_valid_manifest_on_a_mounted_volume_is_used(mounted: None) -> None:
    _write_manifest()
    status = host_io_worker.resolve_worker_io(enabled=True)
    assert status.io == host_io_worker.WorkerIo(
        Path("Y:\\fleet"), Path("Y:\\fleet\\uv-cache"), "Y:"
    )
    assert status.note == ""


def test_the_kill_switch_wins_over_a_valid_manifest(mounted: None) -> None:
    _write_manifest()
    status = host_io_worker.resolve_worker_io(enabled=False)
    assert status.io is None
    assert status.note == "disabled by claude_code.host_io_worktrees"


def test_an_invalid_manifest_falls_back_with_a_note(mounted: None) -> None:
    _write_manifest({**MANIFEST, "schema": 2})
    status = host_io_worker.resolve_worker_io(enabled=True)
    assert status.io is None
    assert "is invalid: unknown schema 2" in status.note


def test_an_unmounted_volume_falls_back_with_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_manifest()
    monkeypatch.setattr(host_io_worker, "probe_volume", lambda drive: None)
    status = host_io_worker.resolve_worker_io(enabled=True)
    assert status.io is None
    assert status.note == "host-io volume unavailable: Y: is not mounted"


def test_the_configured_manifest_path_wins_over_the_environment(
    tmp_path: Path, mounted: None
) -> None:
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps(MANIFEST), encoding="utf-8")
    assert host_io_worker.resolve_worker_io(enabled=True).io is None
    assert host_io_worker.resolve_worker_io(enabled=True, manifest=str(other)).io is not None


def test_the_suite_never_reads_the_hosts_manifest() -> None:
    """Control for the autouse ``_isolate_host_io_manifest`` fixture."""
    path = Path(os.environ["CI_FLEET_HOST_IO_MANIFEST"])
    assert path.name == "host-io.json"
    assert not path.exists()
    assert "ProgramData" not in str(path)


# -- the worktrees root ---------------------------------------------------------


def test_the_volume_moves_the_default_root_and_keeps_the_old_one_as_legacy(
    tmp_path: Path, mounted: None
) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    resolved = resolved_layout(OrchestratorConfig(), repo)
    old_root = layout.worktrees_dir(runtime_paths(repo, RuntimeConfig().state_dir).root)
    assert resolved.worktrees == _volume_root(repo)
    assert resolved.legacy_worktrees == old_root
    assert resolved.worker_io.io is not None


def test_the_volume_also_beats_an_overridden_state_dir(tmp_path: Path, mounted: None) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    resolved = resolved_layout(OrchestratorConfig(runtime=RuntimeConfig(state_dir="custom")), repo)
    assert resolved.worktrees == _volume_root(repo)
    assert resolved.legacy_worktrees == repo / "custom" / layout.WORKTREES_DIRNAME


def test_an_explicit_worktrees_dir_beats_the_volume(tmp_path: Path, mounted: None) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(worktrees_dir="alt"))
    resolved = resolved_layout(config, repo)
    assert resolved.worktrees == repo / "alt"
    assert resolved.legacy_worktrees is None
    assert resolved.worker_io.io is None
    assert resolved.worker_io.note == "claude_code.worktrees_dir is set"


def test_the_kill_switch_restores_the_default_root(tmp_path: Path, mounted: None) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(host_io_worktrees=False))
    resolved = resolved_layout(config, repo)
    assert resolved.worktrees == layout.worktrees_dir(
        runtime_paths(repo, RuntimeConfig().state_dir).root
    )
    assert resolved.legacy_worktrees is None


def test_the_fleet_snapshot_resolves_the_same_root(tmp_path: Path, mounted: None) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    config = OrchestratorConfig()
    state_dir = runtime_paths(repo, RuntimeConfig().state_dir).root
    _, _, snapshot_root = _repo_state_dirs(repo, state_dir, config)
    assert snapshot_root == resolved_layout(config, repo).worktrees
    assert snapshot_root == worktree_roots(config, repo, state_dir).active


def test_sweep_roots_keep_the_legacy_root_only_while_it_has_entries(
    tmp_path: Path, mounted: None
) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    resolved = resolved_layout(OrchestratorConfig(), repo)
    assert resolved.sweep_roots() == (resolved.worktrees,)
    legacy = resolved.legacy_worktrees
    assert legacy is not None
    legacy.mkdir(parents=True)
    assert resolved.sweep_roots() == (resolved.worktrees,)
    (legacy / "agent-issue-1").mkdir()
    assert resolved.sweep_roots() == (resolved.worktrees, legacy)


def test_worktree_clean_sweeps_both_roots(
    tmp_path: Path, mounted: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest()
    repo = _repo(tmp_path)
    config = OrchestratorConfig()
    legacy = resolved_layout(config, repo).legacy_worktrees
    assert legacy is not None
    (legacy / "agent-issue-1").mkdir(parents=True)
    swept: list[Path] = []

    def _fake_clean(repo_root, worktrees_dir, state, cfg, gh, *, dry_run=False):
        swept.append(worktrees_dir)
        return WorktreeCleanResult(ok=True, message="ok", data={"worktrees_registered": 1})

    monkeypatch.setattr(cli, "GitHub", lambda *a, **k: object())
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: config)
    monkeypatch.setattr(cli, "clean_worktrees", _fake_clean)
    import argparse

    args = argparse.Namespace(repo=repo, config=None, fleet_dir=None, dry_run=True)
    result = cli.run_worktree_clean_command(args)
    assert result.ok is True
    assert swept == [_volume_root(repo), legacy]


# -- merge_clean_results --------------------------------------------------------


def _result(ok: bool, *, removed: list, out_of_scope: int, orphan: list) -> WorktreeCleanResult:
    return WorktreeCleanResult(
        ok=ok,
        message="m",
        data={
            "planned": [],
            "removed": removed,
            "skipped": [],
            "failed": [],
            "orphans": {"planned": [], "removed": orphan, "failed": []},
            "venv_ok": True,
            "venv_message": "v",
            "attention_events": [],
            "worktrees_registered": 5,
            "worktrees_out_of_scope": out_of_scope,
        },
    )


def test_one_root_is_returned_unchanged() -> None:
    only = _result(True, removed=[], out_of_scope=1, orphan=[])
    assert merge_clean_results([(Path("a"), only)]) is only


def test_two_roots_merge_lists_counts_and_ok() -> None:
    a = _result(True, removed=[{"worktree": "a/1"}], out_of_scope=3, orphan=[])
    b = _result(False, removed=[{"worktree": "b/1"}], out_of_scope=4, orphan=[{"worktree": "b/x"}])
    merged = merge_clean_results([(Path("a"), a), (Path("b"), b)])
    assert merged.ok is False
    assert merged.data["removed"] == [{"worktree": "a/1"}, {"worktree": "b/1"}]
    assert merged.data["orphans"]["removed"] == [{"worktree": "b/x"}]
    # 5 registered: 2 in scope under a, 1 under b, so 2 are out of scope under both.
    assert merged.data["worktrees_out_of_scope"] == 2
    assert merged.data["worktrees_registered"] == 5
    assert merged.data["roots"] == ["a", "b"]
    assert merged.message == "a: m; b: m"


def test_no_roots_is_an_error() -> None:
    with pytest.raises(ValueError):
        merge_clean_results([])


# -- worker env, temp and the cutover marker ------------------------------------


def test_worker_env_sets_the_uv_cache_only_when_the_volume_is_used() -> None:
    io = host_io_worker.WorkerIo(Path("Y:\\fleet"), Path("Y:\\fleet\\uv-cache"), "Y:")
    assert host_io_worker.worker_env(io) == {"UV_CACHE_DIR": str(Path("Y:\\fleet\\uv-cache"))}
    assert host_io_worker.worker_env(None) == {}


def test_adapter_settings_layer_the_uv_cache_under_operator_worker_env(
    tmp_path: Path, mounted: None
) -> None:
    from charlie_work.workflow import OrchestratorApp

    _write_manifest()
    repo = _repo(tmp_path)
    plain = OrchestratorConfig(claude_code=ClaudeCodeConfig())
    pinned = OrchestratorConfig(claude_code=ClaudeCodeConfig(worker_env={"UV_CACHE_DIR": "D:\\c"}))

    def _settings(config: OrchestratorConfig):
        app = OrchestratorApp(
            repo, runtime_paths(repo, RuntimeConfig().state_dir), config, object(), dry_run=True
        )
        return app._adapter_settings(adapter="claude-code")

    assert _settings(plain).worker_env["UV_CACHE_DIR"] == str(Path("Y:\\fleet\\uv-cache"))
    assert _settings(plain).worktrees_dir == _volume_root(repo)
    assert _settings(pinned).worker_env["UV_CACHE_DIR"] == "D:\\c"


def test_no_manifest_leaves_the_worker_env_untouched(tmp_path: Path) -> None:
    from charlie_work.workflow import OrchestratorApp

    repo = _repo(tmp_path)
    app = OrchestratorApp(
        repo, runtime_paths(repo, RuntimeConfig().state_dir), OrchestratorConfig(), object()
    )
    assert "UV_CACHE_DIR" not in app._adapter_settings(adapter="claude-code").worker_env


def test_temp_follows_the_worktree_onto_the_volume(tmp_path: Path) -> None:
    """Issue #1767's per-session temp dir is inside the worktree, so it needs no setting."""
    worktree = tmp_path / "fleet" / "repo-a" / layout.WORKTREES_DIRNAME / "agent-issue-1"
    worktree.mkdir(parents=True)
    env = sanitize_env(worktree)
    assert Path(env["TMP"]) == layout.worker_tmp_dir(worktree)
    assert Path(env["TEMP"]).is_relative_to(worktree)


def test_the_cutover_marker_is_written_once(tmp_path: Path) -> None:
    io = host_io_worker.WorkerIo(Path("Y:\\fleet"), Path("Y:\\fleet\\uv-cache"), "Y:")
    first = datetime(2026, 10, 12, 1, 2, 3, tzinfo=UTC)
    assert host_io_worker.record_cutover(tmp_path, io, dry_run=False, now=first) is True
    assert host_io_worker.record_cutover(tmp_path, io, dry_run=False) is False
    marker = tmp_path / host_io_worker.CUTOVER_FILENAME
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "cutover_at": "2026-10-12T01:02:03Z",
        "schema": 1,
    }


def test_no_cutover_marker_without_the_volume_or_on_a_dry_run(tmp_path: Path) -> None:
    io = host_io_worker.WorkerIo(Path("Y:\\fleet"), Path("Y:\\fleet\\uv-cache"), "Y:")
    assert host_io_worker.record_cutover(tmp_path, None, dry_run=False) is False
    assert host_io_worker.record_cutover(tmp_path, io, dry_run=True) is False
    assert not (tmp_path / host_io_worker.CUTOVER_FILENAME).exists()


def test_a_live_dispatch_records_the_cutover(tmp_path: Path, mounted: None) -> None:
    from charlie_work.fleet_paths import fleet_dir
    from charlie_work.workflow import OrchestratorApp

    _write_manifest()
    repo = _repo(tmp_path)
    app = OrchestratorApp(
        repo, runtime_paths(repo, RuntimeConfig().state_dir), OrchestratorConfig(), object()
    )
    app._adapter_settings(adapter="claude-code")
    assert (fleet_dir() / host_io_worker.CUTOVER_FILENAME).exists()


def test_the_kill_switch_defaults_on() -> None:
    assert OrchestratorConfig().claude_code.host_io_worktrees is True
