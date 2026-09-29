"""Issue #1810: per-pass ``loop()`` lanes gated on ``publishes_pull_requests``.

On a ``local_issues`` repo (``LocalFileGitHub`` --
``publishes_pull_requests = False``), two per-pass steps assumed a
GitHub/PR-capable backend and fired a failure/warning event every single
pass:

* ``_maybe_reconcile_drift`` -> ``_reconcile_locked`` -> ``detect_drift`` ->
  ``_fetch_prs`` -> ``gh.run(...)`` raises ``GitHubError`` (the repo has no
  remote to query) -- recorded as ``reconcile_pass_failed`` (ERROR).
* ``_maybe_reclaim_superseded_main_ci`` ->
  ``reclaim_superseded_main_ci_runs`` -> ``git fetch origin`` fails (no
  origin remote exists) -- recorded as ``main_ci_reclaim_failed`` (WARNING).

Each step is now skipped on the existing
``local_work_park.publishes_pull_requests`` capability predicate -- the same
one ``dead_worker_reap`` consults -- rather than surviving its own failure
per call site. The two ``_maybe_*`` lanes live in
``orchestration/state_pr_capability_lanes.py`` (a delegate leaf extracted
from ``state_maintenance`` for the file-size ratchet). These tests run
one ``loop()`` pass per backend and assert on the patched callees: never
invoked on the non-publishing backend, and invoked on the publishing
control (proving the skip is conditional on the capability, not the lane
being deleted outright).

(A third lane used to live here: the #1001 worker-GitHub-token dispatch
probe. Issue #1853 retired it outright -- workers are credential-free by
design, so a missing ``worker_env`` token is not a defect on ANY backend.
The publishing-backend control below now asserts the event is never
emitted even with ``require_worker_github_token=True`` still set.)
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

import charlie_work.workflow as workflow_module
from _fakes_github import FakeGitHub
from charlie_work.config import (
    AutoMergeConfig,
    DispatchConfig,
    LocalIssuesConfig,
    LocalLaneConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    ReviewDispatchConfig,
    WorkerRoleConfig,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.main_ci_reclaim import MainCiReclaimResult
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote -- the exact
    shape of a ``local_issues`` repo (e.g. ``local/mdls``).

    ``core.longpaths`` is set because pytest's basetemp under this repo's
    ``.var/worker-tmp`` pushes ``.git/objects/<sha>`` paths past the Windows
    MAX_PATH limit (``error: unable to write file ... Filename too long``,
    commit exit 1); the knob is a no-op elsewhere. The child env is also
    scrubbed of ``GIT_*`` variables so an ambient ``GIT_DIR``/
    ``GIT_INDEX_FILE``/``GIT_CONFIG_*`` leaked by whatever shell spawned
    pytest cannot redirect these commands at the *outer* repo instead of
    the fresh one.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for command in (
        ["git", "init", "--initial-branch=main"],
        ["git", "config", "core.longpaths", "true"],
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "--allow-empty",
            "-m",
            "chore: seed",
        ],
    ):
        result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise AssertionError(
                f"{command[:2]} rc={result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )


def _config(*, local_enabled: bool) -> OrchestratorConfig:
    """Arms every lane this issue gates: ``reconcile_pass`` and
    ``main_ci_reclaim`` on their production-enabled settings. The retired
    ``require_worker_github_token`` flag is deliberately left set -- issue
    #1853 made it a no-op, so even a config that still carries it must
    dispatch normally with no ``worker_token_missing`` event."""
    return OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30),
        main_ci_reclaim=MainCiReclaimConfig(enabled=True, workflow_filename="ci.yml"),
        dispatch=DispatchConfig(require_worker_github_token=True),
        worker=WorkerRoleConfig(harness="devin-shell"),
        local_issues=LocalIssuesConfig(enabled=local_enabled, issues_dir="docs/issues"),
    )


def _patch_lane_callees(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Spy on the two callees each gate must skip or reach.

    ``_reconcile_locked`` is patched on the class (``_maybe_reconcile_drift``
    calls ``self._reconcile_locked``), ``reclaim_superseded_main_ci_runs`` on
    the ``workflow`` facade (the lane reaches it through ``_wf.``).
    """
    reconcile_locked = Mock(return_value=CommandResult(True, "reconciled", {}))
    reclaim = Mock(
        return_value=MainCiReclaimResult(
            ok=True, tip_sha="tip", candidates_checked=0, cancelled=()
        )
    )
    monkeypatch.setattr(OrchestratorApp, "_reconcile_locked", reconcile_locked)
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", reclaim)
    return reconcile_locked, reclaim


def test_loop_pass_skips_pr_shaped_lanes_on_non_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a ``LocalFileGitHub`` (``publishes_pull_requests = False``), one
    ``loop()`` pass must not reach ``_reconcile_locked`` or
    ``reclaim_superseded_main_ci_runs`` at all -- and must emit none of the
    noise events -- even with every lane's own enable knob armed as
    production runs them."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _config(local_enabled=True)
    issues_dir = repo_root / config.local_issues.issues_dir
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(
        repo_root=repo_root,
        issues_dir=issues_dir,
        state_dir=config.runtime.state_dir,
    )
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)
    reconcile_locked, reclaim = _patch_lane_callees(monkeypatch)

    result = app.loop()

    reconcile_locked.assert_not_called()
    reclaim.assert_not_called()

    assert result.ok is True
    state = load_state(paths.state_file)
    kinds = {e.get("kind") for e in state.get("events", [])}
    assert "worker_token_missing" not in kinds
    assert not [k for k in kinds if str(k).startswith("reconcile_pass")]
    assert not [k for k in kinds if str(k).startswith("main_ci_reclaim")]


def test_loop_pass_runs_pr_shaped_lanes_on_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: a backend that publishes PRs (``FakeGitHub`` -- no
    ``publishes_pull_requests`` attribute, so the predicate defaults True)
    must still run both lanes under the same armed config, or the skip
    would be unconditional. Empty issues/PRs keep the pass cheap; both
    gates sit ahead of any candidate fetch.

    Issue #1853 addendum: the retired ``require_worker_github_token=True``
    in ``_config`` must NOT produce a ``worker_token_missing`` event or a
    dispatch deferral -- the gate is gone on every backend, publishing or
    not."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _config(local_enabled=False)
    gh = FakeGitHub()
    gh.issues = []
    gh.prs = []
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)
    reconcile_locked, reclaim = _patch_lane_callees(monkeypatch)

    app.loop()

    reconcile_locked.assert_called_once()
    reclaim.assert_called_once()

    state = load_state(paths.state_file)
    kinds = {e.get("kind") for e in state.get("events", [])}
    assert "reconcile_pass_completed" in kinds
    assert "worker_token_missing" not in kinds


# ---------------------------------------------------------------------------
# Issue #1968: ``local_lane_kill_switch_stalled`` event gating matrix.
#
# An explicit ``review_dispatch.enabled: false`` or ``auto_merge.enabled:
# false`` on a ``local_issues`` repo is honored -- the gated sub-phase stays
# skipped -- but before #1968 it also silently dead-ended every finished
# ticket at ``agent:review-ready``. These tests pin the stall alarm's
# condition matrix: fires only on the local backend, only while a disabled
# switch coexists with a review-ready issue parked past
# ``local_lane.kill_switch_stall_hours``, and never at the cost of the
# switch itself (the gated launcher still never runs).
# ---------------------------------------------------------------------------


def _write_review_ready_issue(issues_dir: Path, number: int) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_issue.md"
    path.write_text(
        f'---\ntitle: "issue {number}"\nstate: open\nlabels: ["agent:review-ready"]\n---\nBody.\n',
        encoding="utf-8",
    )
    return path


def _kill_switch_config(
    *, review_dispatch: bool, auto_merge: bool, stall_hours: float = 12.0
) -> OrchestratorConfig:
    return OrchestratorConfig(
        local_issues=LocalIssuesConfig(enabled=True, issues_dir="docs/issues"),
        review_dispatch=ReviewDispatchConfig(enabled=review_dispatch),
        auto_merge=AutoMergeConfig(enabled=auto_merge),
        local_lane=LocalLaneConfig(kill_switch_stall_hours=stall_hours),
    )


def _local_app(repo_root: Path, config: OrchestratorConfig):
    issues_dir = repo_root / config.local_issues.issues_dir
    issues_dir.mkdir(parents=True, exist_ok=True)
    gh = LocalFileGitHub(
        repo_root=repo_root,
        issues_dir=issues_dir,
        state_dir=config.runtime.state_dir,
    )
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    return OrchestratorApp(repo_root, paths, config, gh), paths, issues_dir


def _seed_issue_age(paths, number: int, *, updated_at: str) -> None:
    """Give the issue a state entry timestamp -- the alarm's primary age source."""
    state = empty_state()
    state["issues"][str(number)] = {"updated_at": updated_at}
    save_state(paths.state_file, state)


def _stall_events(paths) -> list[dict]:
    return [
        e
        for e in load_state(paths.state_file).get("events", [])
        if e.get("kind") == "local_lane_kill_switch_stalled"
    ]


def _iso_hours_ago(hours: float, *, now: datetime) -> str:
    return (now - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def test_local_lane_kill_switch_stalled_when_review_dispatch_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit ``review_dispatch.enabled: false`` + a stale review-ready
    issue fires the warning event -- and the switch is still honored (no
    reviewer launch is attempted)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=True)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(48, now=now))
    dispatch_reviewers = Mock(return_value={"launched": [], "claimed": []})
    monkeypatch.setattr(OrchestratorApp, "_local_dispatch_reviewers", dispatch_reviewers)

    app._local_lane(now=now)

    dispatch_reviewers.assert_not_called()  # the kill switch stays honored
    events = _stall_events(paths)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["switch"] == "review_dispatch.enabled"
    assert payload["issue_numbers"] == [5]
    assert payload["oldest_age_hours"] > 12.0
    assert payload["threshold_hours"] == 12.0

    # And it repeats on every pass while the stall persists.
    app._local_lane(now=now)
    assert len(_stall_events(paths)) == 2


def test_local_lane_kill_switch_stalled_when_auto_merge_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same stall with ``auto_merge.enabled: false`` names that key -- and
    the merge phase is still skipped."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=True, auto_merge=False)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(48, now=now))
    merge_approved = Mock(return_value=[])
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", merge_approved)

    app._local_lane(now=now)

    merge_approved.assert_not_called()
    events = _stall_events(paths)
    assert len(events) == 1
    assert events[0]["payload"]["switch"] == "auto_merge.enabled"


def test_local_lane_kill_switch_stalled_emits_per_disabled_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both switches off produces one event per switch per pass."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=False)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(48, now=now))
    monkeypatch.setattr(OrchestratorApp, "_local_dispatch_reviewers", Mock())
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", Mock())

    app._local_lane(now=now)

    events = _stall_events(paths)
    assert len(events) == 2
    assert {e["payload"]["switch"] for e in events} == {
        "review_dispatch.enabled",
        "auto_merge.enabled",
    }


def test_local_lane_kill_switch_quiet_under_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review-ready issue parked less than ``kill_switch_stall_hours`` is
    not yet a stall -- no event."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=True)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(1, now=now))
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", Mock(return_value=[]))

    app._local_lane(now=now)

    assert _stall_events(paths) == []


def test_local_lane_kill_switch_quiet_when_switches_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: a stale review-ready issue with the lane armed emits
    nothing -- the alarm measures config stranding, not parking itself."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=True, auto_merge=True)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(48, now=now))
    monkeypatch.setattr(
        OrchestratorApp,
        "_local_dispatch_reviewers",
        Mock(return_value={"launched": [], "claimed": []}),
    )
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", Mock(return_value=[]))

    app._local_lane(now=now)

    assert _stall_events(paths) == []


def test_local_lane_kill_switch_quiet_on_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote-backend control: the capability gate returns before the alarm
    is evaluated, so a PR-publishing backend with the same config and a
    review-ready issue emits nothing."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=False)
    gh = FakeGitHub()
    gh.issues = []
    gh.prs = []
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)

    result = app._local_lane(now=datetime.now(UTC))

    assert result.ok is True
    assert _stall_events(paths) == []


def test_local_lane_kill_switch_ages_from_issue_file_mtime_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no state entry, the alarm falls back to the issue file's mtime
    (what ``updatedAt`` carries on this backend) instead of skipping the
    issue."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=True)
    app, paths, issues_dir = _local_app(repo_root, config)
    issue_path = _write_review_ready_issue(issues_dir, 5)
    old = (datetime.now(UTC) - timedelta(hours=48)).timestamp()
    os.utime(issue_path, (old, old))
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", Mock(return_value=[]))

    app._local_lane(now=datetime.now(UTC))

    events = _stall_events(paths)
    assert len(events) == 1
    assert events[0]["payload"]["issue_numbers"] == [5]


def test_local_lane_kill_switch_muted_when_stall_hours_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``kill_switch_stall_hours: 0`` mutes the alarm only -- the disabled
    switch is still honored (no reviewer launch)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _kill_switch_config(review_dispatch=False, auto_merge=True, stall_hours=0)
    app, paths, issues_dir = _local_app(repo_root, config)
    _write_review_ready_issue(issues_dir, 5)
    now = datetime.now(UTC)
    _seed_issue_age(paths, 5, updated_at=_iso_hours_ago(48, now=now))
    dispatch_reviewers = Mock(return_value={"launched": [], "claimed": []})
    monkeypatch.setattr(OrchestratorApp, "_local_dispatch_reviewers", dispatch_reviewers)
    monkeypatch.setattr(OrchestratorApp, "_local_merge_approved", Mock(return_value=[]))

    app._local_lane(now=now)

    dispatch_reviewers.assert_not_called()
    assert _stall_events(paths) == []
