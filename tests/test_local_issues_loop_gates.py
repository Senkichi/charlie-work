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
design, so a missing ``worker_env`` token is not a defect on ANY backend.
The publishing-backend control below now asserts the event is never
emitted even with ``require_worker_github_token=True`` still set.)
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

import charlie_work.workflow as workflow_module
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DispatchConfig,
    LocalIssuesConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    WorkerRoleConfig,
)
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.main_ci_reclaim import MainCiReclaimResult
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _reconcile_fixtures import _write_local_issue


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


def test_loop_pass_reconcile_finalizes_closed_escalated_issue_on_local_backend(
    tmp_path: Path,
) -> None:
    """Issue #1969 regression, end to end through ``app.loop()``: a local
    issue closed out from under an ``escalated`` state entry (the repo's own
    #131/#132 were stuck that way since 2026-09-23) is finalized to
    ``closed`` by the periodic reconcile pass inside the loop -- no operator
    action, no mocked lane -- and stops counting as a ``sink_census`` root
    for ``operator_queue_impact``. The still-open ``escalated`` control
    issue is untouched (``escalated_labels_converged`` D-2)."""
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
