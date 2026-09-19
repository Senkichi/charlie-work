"""Per-lane error isolation for ``fleet_loop``.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import layout
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
)
from charlie_work.fleet_dispatch import (
    _build_fleet_attention_digest,
    fleet_loop,
)
from charlie_work.instrumentation import query_events
from charlie_work.github import GitHubError
from charlie_work.workflow import CommandResult


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_github_error_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop isolates GitHubError from one repo and continues to others."""
    # Setup registry with two repos
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances - first one raises GitHubError
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.side_effect = GitHubError("API rate limit exceeded")
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify result includes both repos
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]

    # Verify first repo failed but second succeeded
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True

    # Verify overall result is False (one repo failed)
    assert result.ok is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_unclassified_exception_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """fleet_loop isolates an unclassified exception from one repo and continues to others."""
    # Setup registry with two repos
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    # Create temp repo dirs
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # Mock config and paths
    mock_config = OrchestratorConfig()
    mock_load_layered_config.return_value = mock_config
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    # Mock OrchestratorApp instances - first one raises an unclassified RuntimeError
    mock_app1 = MagicMock()
    mock_app2 = MagicMock()
    mock_app1.loop.side_effect = RuntimeError("provider response malformed")
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.side_effect = [mock_app1, mock_app2]

    # Mock GitHub
    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    # Run fleet_loop - should not propagate the RuntimeError
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # Verify both repos are present in the result
    assert "repos" in result.data
    assert "owner/repo1" in result.data["repos"]
    assert "owner/repo2" in result.data["repos"]

    # Verify repo1 failed, repo2 succeeded and was processed
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app2.loop.call_count == 1

    # Verify the failing repo's message is recorded
    assert "fleet pass error" in result.data["repos"]["owner/repo1"].get("message", "")
    # The exception type must be part of the surfaced message (diagnosability).
    assert "RuntimeError" in result.data["repos"]["owner/repo1"]["message"]

    # Issue #738: the genuine lane-crash path (this test's RuntimeError raised
    # inside app.loop()) must set ``errored: True`` on the per-repo result data
    # at its point of origin in fleet_loop's ``except Exception`` handler, so
    # the supervisor headline can split "errored" from "completed with
    # conditions". The downstream headline-split tests plant this flag via a
    # synthetic fixture; this assertion verifies the flag is actually set by
    # the real exception path, not just honored when present.
    assert result.data["repos"]["owner/repo1"].get("errored") is True
    # The successful repo must not carry the marker.
    assert "errored" not in result.data["repos"]["owner/repo2"]

    # Verify overall result is False (one repo failed)
    assert result.ok is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_config_load_error_isolated(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC4 (most important): a repo whose lane fails during startup —
    i.e. inside ``load_layered_config`` itself, before ``OrchestratorApp`` is
    ever constructed — must not prevent another repo's lane from running.

    This is distinct from ``test_fleet_loop_unclassified_exception_isolated``
    above, which raises inside ``app.loop()`` (config load succeeds for both
    repos there). The real 2026-07-29 incident (``ConfigError: unknown
    key(s) in config section 'cross_family': auto_verdict``) failed at
    config-load time, before any per-repo app object existed — this test
    pins isolation at that exact point. D-4 requires the per-repo ``except``
    to keep catching this; this test would fail loudly (as a fleet-wide
    exception) if a future change narrowed or removed it.

    G-AC6: the injected failure happens inside ``load_layered_config``,
    strictly before ``paths = runtime_paths(...)`` executes in the same try
    block, so ``paths`` is genuinely unbound in repo1's except handler (not
    merely untested) -- see the ``mock_runtime_paths.call_count`` assertion
    below.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
            "owner/repo2": {
                "repo_root": str(tmp_path / "repo2"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry

    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo2").mkdir()

    # repo1's config load raises during startup; repo2's succeeds. Only one
    # OrchestratorApp is ever constructed (for repo2) because repo1 never
    # reaches that line — mock_app_class.return_value (not side_effect list)
    # pins that.
    #
    # Keyed by repo_root rather than a fixed-length call-order list: the
    # fleet pass also calls load_layered_config a second time for repo1 from
    # compute_api_worker_fleet_report (it re-loads any repo missing from
    # preloaded_configs, which repo1 is, since its first load failed). A
    # positional side_effect list of length 2 would exhaust after the two
    # per-repo-loop calls and raise a spurious StopIteration on that third
    # call. Retrying the same broken config deterministically re-raises the
    # same ConfigError, matching real load_layered_config behavior.
    repo1_root = tmp_path / "repo1"

    def _load_layered_config_side_effect(
        repo_root: Path, *args: Any, **kwargs: Any
    ) -> OrchestratorConfig:
        if Path(repo_root) == repo1_root:
            raise ConfigError(
                "unknown key(s) in config section 'cross_family': auto_verdict "
                "(valid: enabled, model, command)"
            )
        return OrchestratorConfig()

    mock_load_layered_config.side_effect = _load_layered_config_side_effect
    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2

    mock_gh = MagicMock()
    mock_gh_class.return_value = mock_gh

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # repo1 failed at startup; repo2's lane actually ran. This is the
    # isolation proof: only one OrchestratorApp was ever built, and its
    # loop() was called exactly once, for the surviving repo.
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1

    # G-AC6: repo1's ConfigError is raised inside load_layered_config,
    # strictly before `paths = runtime_paths(...)` is reached in that same
    # try block. runtime_paths is therefore called exactly once (for repo2
    # only) -- proving `paths` is genuinely unbound in repo1's except
    # handler, not just untested. The handler itself never references
    # `paths` (it uses `repo_root`/`entry`, both bound before the try); if a
    # future change added a `paths.state_file` reference there, this would
    # raise UnboundLocalError *inside* the except block, which escapes the
    # per-repo isolation boundary entirely (D-4) instead of being caught by
    # it -- this assertion is what pins that it never happens.
    assert mock_runtime_paths.call_count == 1

    message = result.data["repos"]["owner/repo1"]["message"]
    assert "fleet pass error" in message
    assert "ConfigError" in message
    assert "cross_family" in message

    # G-AC2: the raw digest feed carries the failure even though app.loop()
    # never ran for repo1 — _extract_attention_events() (which only runs
    # after a successful loop()) never fires for repo1, so this event must
    # come from the except block itself.
    digest_events = result.data["digest"]["events"]
    error_events = [e for e in digest_events if e.get("repo_key") == "owner/repo1"]
    assert len(error_events) == 1
    assert error_events[0]["type"] == "error"

    # Confirm the reused "error" branch actually maps this to a real
    # AttentionEntry (health=ERROR, already desktop-toast-eligible via
    # _DESKTOP_SEVERITIES) rather than silently falling through.
    attention_digest = _build_fleet_attention_digest(digest_events)
    matching = [e for e in attention_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"
    assert "cross_family" in (matching[0].last_log_line or "")

    assert result.ok is False


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_real_unknown_config_key_reproduces_incident(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
) -> None:
    """#6-G / G-AC5: full reproduction of the 2026-07-29 incident.

    The original incident's exact shape was an unknown key inside
    ``cross_family:`` (``cross_family: auto_verdict`` under a version-skew
    binary that didn't yet recognize it). The role-config Phase 2 cleanup
    deleted the dual-accept section tolerance entirely, so a bare
    ``cross_family:`` section is no longer specially tolerated -- it is
    rejected as an unknown top-level section like any other bogus key, the
    same as the original incident's exact shape would raise again today.
    This reproduces the same class of failure (an unknown key inside a real
    config file) via a section that has always validated its own keys
    (``labels``), driving the real, unmocked ``load_layered_config`` to
    raise ``ConfigError``. This proves: (a) an events.db row is recorded
    for the failing repo (queryable via query_events(kind=
    "fleet_pass_config_error")), (b) the fleet digest carries a matching
    AttentionEntry, and (c) a second, healthy repo's lane still completes —
    while a doctor check run against the failing repo's own state directory
    surfaces the same event as a finding (see
    test_check_recent_lane_failures_surfaces_past_event in test_doctor.py,
    which covers the doctor half of this chain with the same event shape).

    Only load_layered_config is left unmocked; runtime_paths/GitHub/
    OrchestratorApp stay mocked exactly as in the other fleet_loop tests —
    this isolates "does the real config parser really raise ConfigError for
    an unknown key, and does fleet_loop's except really catch it" from the
    rest of the per-repo machinery.
    """
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    (repo1 / "orchestrator.config.yaml").write_text(
        "labels:\n"
        "  ready: automated-ready\n"
        "  totally_unknown_key: true\n"
        "runtime:\n"
        "  state_dir: .var/charlie-work\n",
        encoding="utf-8",
    )
    repo1_state_dir = repo1 / ".var" / "charlie-work"
    repo1_state_dir.mkdir(parents=True)

    repo2 = tmp_path / "repo2"
    repo2.mkdir()

    mock_load_registry.return_value = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(repo1),
                "state_dir": str(repo1_state_dir),
                # No config_path override: load_layered_config resolves the
                # real file above via find_config_path(repo_root, None).
            },
            "owner/repo2": {
                "repo_root": str(repo2),
            },
        }
    }

    mock_paths = MagicMock()
    mock_paths.root = tmp_path / ".var" / "charlie-work"
    mock_runtime_paths.return_value = mock_paths

    mock_app2 = MagicMock()
    mock_app2.loop.return_value = CommandResult(True, "repo2 loop complete", {})
    mock_app_class.return_value = mock_app2
    mock_gh_class.return_value = MagicMock()

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=3,
        merge=True,
        dry_run=False,
        work_only=False,
    )

    # (c) repo2's lane proceeded despite repo1's real ConfigError.
    assert result.data["repos"]["owner/repo1"]["ok"] is False
    assert result.data["repos"]["owner/repo2"]["ok"] is True
    assert mock_app_class.call_count == 1
    assert mock_app2.loop.call_count == 1

    # G-AC6: same pre-`paths`-binding failure point as
    # test_fleet_loop_config_load_error_isolated, this time via the real,
    # unmocked load_layered_config raising the real ConfigError rather than
    # a mock side_effect. runtime_paths is called exactly once (repo2 only).
    assert mock_runtime_paths.call_count == 1

    message = result.data["repos"]["owner/repo1"]["message"]
    assert "ConfigError" in message
    assert "labels" in message
    assert "totally_unknown_key" in message

    # (a) the failure is durably recorded to repo1's own events.db.
    state_path = layout.state_file_path(repo1_state_dir)
    recorded = query_events(state_path, kind="fleet_pass_config_error")
    assert len(recorded) == 1
    assert recorded[0]["level"] == "error"
    assert recorded[0]["payload"]["repo_key"] == "owner/repo1"
    assert "totally_unknown_key" in recorded[0]["payload"]["error"]

    # (b) the fleet digest carries a matching entry.
    attention_digest = _build_fleet_attention_digest(result.data["digest"]["events"])
    matching = [e for e in attention_digest.transitions if e.adapter_kind == "owner/repo1"]
    assert len(matching) == 1
    assert matching[0].health == "ERROR"
