"""Issue #1810: per-pass ``loop()`` lanes gated on ``publishes_pull_requests``.

On a ``local_issues`` repo (``LocalFileGitHub`` --
``publishes_pull_requests = False``), three per-pass steps assumed a
GitHub/PR-capable backend and fired a failure/warning event every single
pass:

* ``_maybe_reconcile_drift`` -> ``_reconcile_locked`` -> ``detect_drift`` ->
  ``_fetch_prs`` -> ``gh.run(...)`` raises ``GitHubError`` (the repo has no
  remote to query) -- recorded as ``reconcile_pass_failed`` (ERROR).
* ``_maybe_reclaim_superseded_main_ci`` ->
  ``reclaim_superseded_main_ci_runs`` -> ``git fetch origin`` fails (no
  origin remote exists) -- recorded as ``main_ci_reclaim_failed`` (WARNING).
* ``_dispatch_impl``'s worker-GitHub-token gate ->
  ``worker_github_token_findings`` emits ``worker_token_missing`` (WARNING,
  once) -- meaningless on a backend where no worker can ever push or open a
  PR.

Each step is now skipped on the existing
``local_work_park.publishes_pull_requests`` capability predicate -- the same
one ``dead_worker_reap`` consults -- rather than surviving its own failure
per call site. These tests run one ``loop()`` pass per backend and assert
on the patched callees: never invoked on the non-publishing backend, and
invoked on the publishing control (proving the skip is conditional on the
capability, not the lane being deleted outright).
"""

from __future__ import annotations

import subprocess
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
from charlie_work.env_sanitize import worker_github_token_findings
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.main_ci_reclaim import MainCiReclaimResult
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import CommandResult, OrchestratorApp

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401

# Imported after ``charlie_work.workflow`` deliberately: ``dispatch_state``
# does ``import charlie_work.workflow as _wf`` at module level, so importing
# it directly *before* ``charlie_work.workflow`` has been fully imported
# triggers Python's circular-import partial-module hazard -- ``workflow``'s
# own module-level ``discover_delegate_modules`` call would then reimport
# this already-in-progress module and see only the portion defined above
# the ``import charlie_work.workflow as _wf`` line, silently dropping every
# delegate defined below it (including ``_dispatch_impl``) from
# ``OrchestratorApp``. Importing ``charlie_work.workflow`` first (above)
# guarantees delegate installation has completed by the time this line
# runs. (Same hazard documented in
# tests/test_issue_1314_operator_queue_followups.py.)
import charlie_work.orchestration.dispatch_state as dispatch_state_module  # noqa: E402


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote -- the exact
    shape of a ``local_issues`` repo (e.g. ``local/mdls``)."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
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
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _config(*, local_enabled: bool) -> OrchestratorConfig:
    """Arms every lane this issue gates: ``reconcile_pass`` and
    ``main_ci_reclaim`` on their production-enabled settings, and a live
    worker-token gate (``devin-shell`` harness with no ``worker_env`` token,
    hard-refusal on) so a publishing backend WOULD escalate and defer."""
    return OrchestratorConfig(
        reconcile_pass=ReconcilePassConfig(enabled=True, interval_minutes=30),
        main_ci_reclaim=MainCiReclaimConfig(enabled=True, workflow_filename="ci.yml"),
        dispatch=DispatchConfig(require_worker_github_token=True),
        worker=WorkerRoleConfig(harness="devin-shell"),
        local_issues=LocalIssuesConfig(enabled=local_enabled, issues_dir="docs/issues"),
    )


def _patch_lane_callees(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    """Spy on the three callees each gate must skip or reach.

    ``_reconcile_locked`` is patched on the class (``_maybe_reconcile_drift``
    calls ``self._reconcile_locked``), ``reclaim_superseded_main_ci_runs`` on
    the ``workflow`` facade (the lane reaches it through ``_wf.``), and
    ``worker_github_token_findings`` on ``dispatch_state``'s own import
    binding. The findings spy wraps the real predicate so the positive
    control still exercises the real escalation path.
    """
    reconcile_locked = Mock(return_value=CommandResult(True, "reconciled", {}))
    reclaim = Mock(
        return_value=MainCiReclaimResult(
            ok=True, tip_sha="tip", candidates_checked=0, cancelled=()
        )
    )
    token_findings = Mock(wraps=worker_github_token_findings)
    monkeypatch.setattr(OrchestratorApp, "_reconcile_locked", reconcile_locked)
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", reclaim)
    monkeypatch.setattr(dispatch_state_module, "worker_github_token_findings", token_findings)
    return reconcile_locked, reclaim, token_findings


def test_loop_pass_skips_pr_shaped_lanes_on_non_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a ``LocalFileGitHub`` (``publishes_pull_requests = False``), one
    ``loop()`` pass must not reach ``_reconcile_locked``,
    ``reclaim_superseded_main_ci_runs``, or ``worker_github_token_findings``
    at all -- and must emit none of the three noise events -- even with
    every lane's own enable knob armed as production runs them."""
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
    reconcile_locked, reclaim, token_findings = _patch_lane_callees(monkeypatch)

    result = app.loop()

    reconcile_locked.assert_not_called()
    reclaim.assert_not_called()
    token_findings.assert_not_called()

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
    must still run all three lanes under the same armed config, or the skip
    would be unconditional. Empty issues/PRs keep the pass cheap; all three
    gates sit ahead of any candidate fetch."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _config(local_enabled=False)
    gh = FakeGitHub()
    gh.issues = []
    gh.prs = []
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    app = OrchestratorApp(repo_root, paths, config, gh)
    reconcile_locked, reclaim, token_findings = _patch_lane_callees(monkeypatch)

    app.loop()

    reconcile_locked.assert_called_once()
    reclaim.assert_called_once()
    token_findings.assert_called_once()

    state = load_state(paths.state_file)
    kinds = {e.get("kind") for e in state.get("events", [])}
    assert "reconcile_pass_completed" in kinds
    # The wrapped real predicate still fires the escalation on a backend
    # where workers do push/open PRs.
    assert "worker_token_missing" in kinds
