"""Per-pass ``loop()`` lanes gated on ``publishes_pull_requests``.

On a ``local_issues`` repo (``LocalFileGitHub`` --
``publishes_pull_requests = False``), ``_maybe_reclaim_superseded_main_ci``
-> ``reclaim_superseded_main_ci_runs`` -> ``git fetch origin`` fails (no
origin remote exists) -- recorded as ``main_ci_reclaim_failed`` (WARNING).
Issue #1810 skipped it on the existing
``local_work_park.publishes_pull_requests`` capability predicate -- the same
one ``dead_worker_reap`` consults -- rather than surviving its own failure
per call site. The two ``_maybe_*`` lanes live in
``orchestration/state_pr_capability_lanes.py`` (a delegate leaf extracted
from ``state_maintenance`` for the file-size ratchet). These tests run
one ``loop()`` pass per backend and assert on the patched callees: never
invoked on the non-publishing backend, and invoked on the publishing
control (proving the skip is conditional on the capability, not the lane
being deleted outright).

The reconcile lane used to share #1810's skip (``detect_drift`` ->
``_fetch_prs`` -> ``gh.run(...)`` raising ``GitHubError``), but issue #1969
removed it: ``detect_drift`` grew a no-remote issue branch under #1844, and
``_fetch_prs`` now answers the empty PR snapshot on a backend that cannot
host pull requests, so ``_reconcile_locked`` is local-safe end to end.
Running the pass matters -- it is the only automatic path that finalizes a
closed-while-``active`` state entry (``state_active_status_issue_closed``),
so the old skip left a closed local issue parked at ``escalated`` forever,
counting as a ``sink_census`` root for ``operator_queue_impact`` forever.

(A third lane used to live here: the #1001 worker-GitHub-token dispatch
probe. Issue #1853 retired it outright -- workers are credential-free by
design, so a missing ``worker_env`` token is not a defect on ANY backend --
and issue #1977 deleted its config flag. The publishing-backend control
below now asserts the event is never emitted.)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

import charlie_work.workflow as workflow_module
from _fakes_github import FakeGitHub
from _worktree_fixtures import _git, _init_repo as _init_repo_templated
from charlie_work.config import (
    AutoMergeConfig,
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
from charlie_work.command_result import CommandResult
from charlie_work.workflow import OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _reconcile_fixtures import _write_local_issue


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo on ``main`` with one commit and NO origin remote --
    the exact shape of a ``local_issues`` repo (e.g. ``local/mdls``).

    Delegates to ``_worktree_fixtures._init_repo`` so the repo materializes
    from the per-process ``plain`` git template (``shutil.copytree``, no
    subprocesses -- HS-CW-4, ``tests/_git_templates.py``) instead of
    re-running the ``init``/``config``/``commit`` spawn sequence per test:
    the spawn cost, amplified on the Windows CI host, is what the ledger
    flagged at ~2.8x baseline (issue #2425, same fix class as #2387).

    The produced repo carries a README commit rather than the old body's
    ``--allow-empty`` one -- no test here reads the tree -- and gets
    ``core.longpaths`` through ``conftest._isolate_git_env``'s
    ``GIT_CONFIG_*`` injection on every git child instead of an in-repo
    ``git config`` write. The old body's ``GIT_*`` env scrub only mattered
    while it spawned git itself.
    """
    _init_repo_templated(repo_root)


def test_init_repo_goes_through_the_git_template(tmp_path: Path) -> None:
    """Issue #2425 regression pin: ``_init_repo`` must materialize through the
    per-process ``plain`` git template (``_git_templates``) instead of
    re-running the ``init``/``config``/``commit`` spawn sequence per call --
    the ledger measured
    ``test_local_lane_kill_switch_stalled_when_auto_merge_disabled`` at ~2.8x
    baseline because each of this module's tests paid the full spawn cost on
    the Windows CI host. Mirrors ``test_git_templates``'s #2387 pin for
    ``_helpers._init_git_repo``."""
    import _git_templates
    from _worktree_fixtures import _git

    before = _git_templates._REGISTRY.materialized["plain"]
    repo = tmp_path / "repo"
    _init_repo(repo)
    assert _git_templates._REGISTRY.materialized["plain"] == before + 1
    assert _git(repo, "rev-parse", "--verify", "main").stdout.strip()


def _config(*, local_enabled: bool) -> OrchestratorConfig:
    """Arms every lane this issue gates: ``reconcile_pass`` and
    ``main_ci_reclaim`` on their production-enabled settings."""
    return OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30),
        main_ci_reclaim=MainCiReclaimConfig(enabled=True, workflow_filename="ci.yml"),
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
    ``loop()`` pass must not reach ``reclaim_superseded_main_ci_runs`` at
    all -- but issue #1969 un-skipped the reconcile lane, which is
    local-safe end to end and is the only automatic finalizer for a
    closed-while-active state entry. ``_reconcile_locked`` is therefore
    invoked (and mocked, keeping this a gating test), the pass records
    ``reconcile_pass_completed``, and the pass must emit none of the
    noise events -- even with every lane's own enable knob armed as
    production runs them.

    The leaf name predates #1969: ``pr_shaped_lanes`` now means the lanes
    still gated on ``publishes_pull_requests`` -- after #1969 that set is
    exactly ``main_ci_reclaim``. The name is kept verbatim because the
    collect-only gate (#1538) fails a required check on any leaf-name
    removal, rename included, absent an operator exemption label."""
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

    reconcile_locked.assert_called_once_with(fix=True, skip_dead_session_sweep=True, dry_run=False)
    reclaim.assert_not_called()

    assert result.ok is True
    state = load_state(paths.state_file)
    kinds = {e.get("kind") for e in state.get("events", [])}
    assert "worker_token_missing" not in kinds
    assert "reconcile_pass_completed" in kinds
    assert "reconcile_pass_failed" not in kinds
    assert not [k for k in kinds if str(k).startswith("main_ci_reclaim")]


def test_loop_pass_runs_pr_shaped_lanes_on_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: a backend that publishes PRs (``FakeGitHub`` -- no
    ``publishes_pull_requests`` attribute, so the predicate defaults True)
    must still run both lanes under the same armed config, or the skip
    would be unconditional. Empty issues/PRs keep the pass cheap; both
    gates sit ahead of any candidate fetch.

    Issue #1977 addendum: the retired token gate's config flag is deleted
    outright, so no config can produce a ``worker_token_missing`` event or
    a dispatch deferral -- the gate is gone on every backend, publishing
    or not."""
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


@pytest.fixture
def _committed_tracker_repo(tmp_path: Path) -> tuple[Path, OrchestratorConfig, Path]:
    """A ``local_issues`` repo whose seed issue files are already committed --
    the shape the tracker has in production, where ``issues_dir`` is tracked.

    Issue #2567: left uncommitted, the seed files reach the pass-end tracker
    flush (``local_issue_commits.flush_tracker_writes``, issue #2434) as one
    collapsed ``?? docs/issues/`` entry, and the flush sweeps them as two
    ``chore(issues): update`` commit groups inside the test's measured
    ``call`` phase -- nine git spawns (status, the detached/merge safety
    probes, and add+commit per issue) that a real ``loop()`` pass never
    pays on files it did not write. Committing them here -- fixture setup,
    which the ledger deliberately excludes from per-test alerting --
    confines the measured ``call`` phase to the work the pass itself
    produces, which is what the #1969 regression being pinned is about.
    The flush still runs end to end and would still commit any pass-time
    ``_mutate``/``issue_comment`` write; its own sweep/commit machinery is
    covered by ``test_local_issue_commits.py``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _config(local_enabled=True)
    issues_dir = repo_root / config.local_issues.issues_dir
    _write_local_issue(
        issues_dir,
        131,
        "closed-escalated",
        "closed escalated",
        "closed",
        f"[{config.labels.done}]",
        "merged already.",
    )
    _write_local_issue(
        issues_dir,
        132,
        "open-escalated",
        "open escalated",
        "open",
        f"[{config.labels.human_needed}]",
        "waiting on human.",
    )
    _git(repo_root, "add", "--", config.local_issues.issues_dir)
    _git(repo_root, "commit", "-m", "seed tracker issues")
    return repo_root, config, issues_dir


def test_loop_pass_reconcile_finalizes_closed_escalated_issue_on_local_backend(
    _committed_tracker_repo: tuple[Path, OrchestratorConfig, Path],
) -> None:
    """Issue #1969 regression, end to end through ``app.loop()``: a local
    issue closed out from under an ``escalated`` state entry (the repo's own
    #131/#132 were stuck that way since 2026-09-23) is finalized to
    ``closed`` by the periodic reconcile pass inside the loop -- no operator
    action, no mocked lane -- and stops counting as a ``sink_census`` root
    for ``operator_queue_impact``. The still-open ``escalated`` control
    issue is untouched (``escalated_labels_converged`` D-2)."""
    repo_root, config, issues_dir = _committed_tracker_repo
    gh = LocalFileGitHub(
        repo_root=repo_root,
        issues_dir=issues_dir,
        state_dir=config.runtime.state_dir,
    )
    paths = runtime_paths(repo_root, config.runtime.state_dir)

    state = empty_state()
    state["issues"]["131"] = {"number": 131, "status": "escalated"}
    state["issues"]["132"] = {
        "number": 132,
        "status": "escalated",
        # Stamped now so terminal_state_stale's >=2-day alert does not add an
        # unrelated drift item for the control issue.
        "terminal_since": datetime.now(UTC).isoformat(),
    }
    save_state(paths.state_file, state)

    app = OrchestratorApp(repo_root, paths, config, gh)

    result = app.loop()
    assert result.ok is True

    state = load_state(paths.state_file)
    assert state["issues"]["131"]["status"] == "closed"
    assert state["issues"]["132"]["status"] == "escalated"

    events = state.get("events", [])
    kinds = {e.get("kind") for e in events}
    assert "reconcile_pass_completed" in kinds
    assert "reconcile_pass_failed" not in kinds
    assert any(
        e.get("kind") == "reconcile"
        and e.get("payload", {}).get("kind") == "state_active_status_issue_closed"
        and e.get("payload", {}).get("issue_number") == 131
        for e in events
    )
    # The closed issue no longer feeds operator_queue_impact's root census;
    # the still-open escalated one still does.
    assert workflow_module.sink_census(state) == {132}


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
