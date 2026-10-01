"""Shared fixtures for the stale-checks / terminally-absent-check ``review()``
tests (extracted from ``test_stale_checks_retrigger.py`` so no test module
imports another)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.config import AutoMergeConfig, OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHubWithMissingRequiredAndRuns

_STALE_HEAD = "abc123abc123"


def _stale_checks_config(
    *,
    ci_run_never_created_grace_minutes: int = 5,
    stale_checks_grace_minutes: int = 15,
    stale_checks_max_retriggers: int = 3,
) -> OrchestratorConfig:
    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
        enabled=True,
        ci_run_never_created_grace_minutes=ci_run_never_created_grace_minutes,
    )
    review = ReviewConfig(
        stale_checks_grace_minutes=stale_checks_grace_minutes,
        stale_checks_max_retriggers=stale_checks_max_retriggers,
    )
    return OrchestratorConfig(auto_merge=auto_merge, review=review)


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _stale_updated_at() -> str:
    return (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")


def _app_with_conflict_and_missing_checks(
    tmp_path: Path,
    *,
    runs: list[dict[str, Any]] | None = None,
    issue_status: str = "dispatched",
    head_sha: str = _STALE_HEAD,
    mergeable: str = "MERGEABLE",
    **config_kwargs: Any,
) -> OrchestratorApp:
    """A PR shaped like #1186/#1192/#1214: missing required checks (so
    ``_detect_ci_run_never_created`` can fire). Verified against all three
    PRs on 2026-08-16: each has an empty ``statusCheckRollup`` -- confirming
    the missing-checks condition this fixture models is real, not
    hypothetical.

    ``mergeable`` defaults to ``"MERGEABLE"`` so the retrigger-mechanism
    tests (AC5/AC6/AC7/AC11) exercise the close/reopen path. Issue #1451
    reversed #1274's binding comment item 5 for the CONFLICTING case: a
    conflicted branch can never get a ``pull_request`` workflow run, so
    close/reopen is now skipped and the PR is routed to merge-conflict
    rework instead. The CONFLICTING-skip test passes
    ``mergeable="CONFLICTING"`` to exercise that discrimination.

    ``issue_status`` defaults to "dispatched" purely as a reachability
    choice for the fixture: it is the status value that makes
    ``_route_janitor_gate_failure_to_rework`` return None (rework already
    pending) and fall through to the main janitor-gate path this test
    exercises. It is NOT a claim about the linked issues' real status.

    As of 2026-08-16, all three linked issues (#807/#763/#1068) actually
    carry ``agent:human-needed`` on GitHub -- i.e. they are escalated, so
    ``review()`` currently takes the *escalated* early-return branch for
    these PRs, not the main gate wired here (AC4 forbids this fixture from
    touching that branch, and item 7/part 3 owns exhaustion->escalation
    routing). Why they're escalated is not determinable from here: the
    escalation reason lives in the live ``.var/`` state.json this agent is
    fenced out of, and `gh` doesn't expose it. Flag for part 3: decide
    whether exhaustion->escalation alone is sufficient, or whether the
    stale-checks retrigger lane also needs an entry point on the escalated
    path to reach this population.
    """
    config = _stale_checks_config(**config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequiredAndRuns(runs=[] if runs is None else runs)
    fake_gh.prs[0]["headRefOid"] = head_sha
    fake_gh.prs[0]["updatedAt"] = _stale_updated_at()
    fake_gh.prs[0]["mergeable"] = mergeable
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": issue_status}
        save_state(app.paths.state_file, state)
    return app
