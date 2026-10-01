"""Regression tests for issue #1808: a dead reviewer whose terminal result
event is a provider-side ``api_error`` (429/500/502/503/529) must not consume
the per-PR dispatch attempt budget -- a provider outage otherwise walks every
in-flight PR to ``max_review_dispatch_attempts_exceeded`` in parallel.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.unescalate_reset_fields import UNESCALATE_PR_RESET_FIELDS
from charlie_work.workflow import _detect_and_handle_stalled_reviews
from charlie_work.write_gate import WriteGate

from _helpers import _init_git_repo
from _local_lane_fixtures import (
    _app,
    _init_repo,
    _make_branch,
    _parked_issue,
    new_repo_root,
)
from _review_fixtures import _PR_NUMBER, _round_archive_app

PR = 100


def _wg(state_file: Path) -> WriteGate:
    return WriteGate(dry_run=False, state_path=state_file, repo="charlie-work")


def _result_event(status: int | None, terminal_reason: str = "api_error") -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "api_error_status": status,
            "terminal_reason": terminal_reason,
            "stop_reason": "stop_sequence",
        }
    )


class _Env:
    def __init__(self, tmp_path: Path, max_api_errors: int = 3) -> None:
        self.tmp_path = tmp_path
        self.repo_root = tmp_path / "repo"
        _init_git_repo(self.repo_root)
        self.reviews_dir = tmp_path / "reviews"
        self.reviews_dir.mkdir()
        self.config = OrchestratorConfig(
            review_dispatch=ReviewDispatchConfig(
                enabled=True, max_consecutive_review_api_errors=max_api_errors
            )
        )
        self.state_file = tmp_path / "state.json"
        self.state_file.write_text(
            json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
        )
        self.started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def plant_dead_reviewer(self, log_text: str, pr: int = PR) -> None:
        """Simulate one claim (attempt +1) and a dead reviewer with ``log_text``."""
        with state_lock(self.state_file):
            st = load_state(self.state_file)
            prior = st["prs"].get(str(pr), {})
            st["prs"][str(pr)] = {
                **prior,
                "number": pr,
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": self.started,
                "reviewer_pid": 999999999,
                "reviewer_process_start_time": 1.0,
                "review_dispatch_attempt_count": int(prior.get("review_dispatch_attempt_count", 0))
                + 1,
            }
            save_state(self.state_file, st)
        log_path = self.reviews_dir / f"issue-{pr}-review.claude.log"
        log_path.write_text(log_text, encoding="utf-8")
        sidecar = {
            "issue_number": pr,
            "branch": f"agent/issue-{pr}-fix",
            "worktree_path": str(self.tmp_path / "wt"),
            "prompt_path": str(self.tmp_path / "prompt.md"),
            "command": ["claude", "-p"],
            "pid": 999999999,
            "started_at": self.started,
            "log_path": str(log_path),
            "error": None,
            "process_start_time": 1.0,
        }
        (self.reviews_dir / f"issue-{pr}.claude.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )

    def sweep(self) -> dict:
        _detect_and_handle_stalled_reviews(
            self.reviews_dir,
            self.state_file,
            self.config,
            self.repo_root,
            write_gate=_wg(self.state_file),
        )
        return load_state(self.state_file)

    def dispatch_and_die(self, log_text: str) -> dict:
        """One claim, one dead reviewer, one sweep; return the state."""
        self.plant_dead_reviewer(log_text)
        return self.sweep()


def test_three_consecutive_500s_leave_attempt_count_zero_and_arm_backoff(
    tmp_path: Path,
) -> None:
    env = _Env(tmp_path)
    for n in range(1, 4):
        state = env.dispatch_and_die(_result_event(500))
        pr = state["prs"][str(PR)]
        assert pr["review_dispatch_attempt_count"] == 0
        assert pr["review_dispatch_status"] is None  # rolled back, redispatchable
        assert pr["review_api_error_streak"] == n
    assert state["reviewer_quota"]["probe_after"]  # fleet-wide backoff armed
    outage = [e for e in state["events"] if e["kind"] == "review_provider_outage"]
    assert outage and outage[0]["payload"]["api_error_status"] == 500


def test_api_error_beyond_bound_starts_counting(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    for _ in range(3):
        env.dispatch_and_die(_result_event(529))
    state = env.dispatch_and_die(_result_event(529))  # the (N+1)th
    pr = state["prs"][str(PR)]
    assert pr["review_dispatch_attempt_count"] == 1
    assert pr["review_dispatch_status"] == "review_dispatch_failed"
    reasons = [
        e["payload"]["reason"] for e in state["events"] if e["kind"] == "review_dispatch_stalled"
    ]
    assert reasons[-1] == "provider_api_error_counted"


@pytest.mark.parametrize("status", [400, 401, 404, None])
def test_non_provider_api_error_is_still_counted(tmp_path: Path, status: int | None) -> None:
    env = _Env(tmp_path)
    state = env.dispatch_and_die(_result_event(status))
    pr = state["prs"][str(PR)]
    assert pr["review_dispatch_attempt_count"] == 1
    assert pr["review_dispatch_status"] == "review_dispatch_failed"
    assert not [e for e in state["events"] if e["kind"] == "review_provider_outage"]


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_every_provider_status_rolls_back(tmp_path: Path, status: int) -> None:
    state = _Env(tmp_path).dispatch_and_die(_result_event(status))
    assert state["prs"][str(PR)]["review_dispatch_attempt_count"] == 0


def test_streak_resets_on_a_definitive_non_api_death(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    env.dispatch_and_die(_result_event(500))
    env.dispatch_and_die(_result_event(500))
    state = env.dispatch_and_die(_result_event(None, terminal_reason="completed"))
    assert state["prs"][str(PR)]["review_api_error_streak"] == 0


def test_streak_is_cleared_by_unescalate() -> None:
    assert "review_api_error_streak" in UNESCALATE_PR_RESET_FIELDS


def _seed_streak(state_file: Path, pr_number: int, streak: int) -> None:
    with state_lock(state_file):
        state = load_state(state_file)
        prior = state["prs"].get(str(pr_number), {})
        state["prs"][str(pr_number)] = {
            **prior,
            "number": pr_number,
            "review_api_error_streak": streak,
        }
        save_state(state_file, state)


def test_record_review_clears_api_error_streak(tmp_path: Path) -> None:
    """A recorded remote-lane verdict proves the provider is healthy."""
    app, paths = _round_archive_app(tmp_path)
    _seed_streak(paths.state_file, _PR_NUMBER, 2)
    assert load_state(paths.state_file)["prs"][str(_PR_NUMBER)]["review_api_error_streak"] == 2
    result = app.record_review(
        _PR_NUMBER, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
    )
    assert result.ok, result.message
    assert load_state(paths.state_file)["prs"][str(_PR_NUMBER)]["review_api_error_streak"] == 0


def test_record_local_review_clears_api_error_streak() -> None:
    """Same invariant for the local lane (``record_local_review``)."""
    repo = new_repo_root()
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
    app._local_review_packets()
    _seed_streak(app.paths.state_file, 7, 2)
    assert load_state(app.paths.state_file)["prs"]["7"]["review_api_error_streak"] == 2
    verdict = app.record_local_review(
        7,
        "approved",
        summary="lgtm",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert verdict.ok, verdict.message
    assert load_state(app.paths.state_file)["prs"]["7"]["review_api_error_streak"] == 0


def test_zero_disables_rollback(tmp_path: Path) -> None:
    state = _Env(tmp_path, max_api_errors=0).dispatch_and_die(_result_event(500))
    assert state["prs"][str(PR)]["review_dispatch_attempt_count"] == 1


_THROTTLE_LOG = "You've hit your session limit resets 4:40pm (America/Los_Angeles)\n"


def test_throttle_death_between_api_errors_restarts_the_streak(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    env.dispatch_and_die(_result_event(500))
    state = env.dispatch_and_die(_result_event(500))
    assert state["prs"][str(PR)]["review_api_error_streak"] == 2
    state = env.dispatch_and_die(_THROTTLE_LOG)  # definitive throttled death
    assert state["prs"][str(PR)]["review_api_error_streak"] == 0
    state = env.dispatch_and_die(_result_event(500))
    assert state["prs"][str(PR)]["review_api_error_streak"] == 1
    assert state["prs"][str(PR)]["review_dispatch_attempt_count"] == 0


def test_turn_limit_counted_throttle_death_restarts_the_streak(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    state = env.dispatch_and_die(_result_event(500))
    assert state["prs"][str(PR)]["review_api_error_streak"] == 1
    with state_lock(env.state_file):
        st = load_state(env.state_file)
        st["prs"][str(PR)]["review_turn_limit_summary_posted"] = True
        save_state(env.state_file, st)
    state = env.dispatch_and_die(_THROTTLE_LOG)
    assert state["prs"][str(PR)]["review_dispatch_status"] == "review_dispatch_failed"
    assert state["prs"][str(PR)]["review_api_error_streak"] == 0


def test_probe_cleared_after_death_suppresses_backoff_but_still_rolls_back(
    tmp_path: Path,
) -> None:
    env = _Env(tmp_path)
    cleared = (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    with state_lock(env.state_file):
        st = load_state(env.state_file)
        st["reviewer_quota"] = {"last_probe_cleared_at": cleared}
        save_state(env.state_file, st)
    state = env.dispatch_and_die(_result_event(503))
    pr = state["prs"][str(PR)]
    assert pr["review_dispatch_status"] is None  # claim still rolled back
    assert pr["review_dispatch_attempt_count"] == 0
    assert not [e for e in state["events"] if e["kind"] == "review_provider_outage"]
    assert not state["reviewer_quota"].get("consecutive_probe_failures")


def test_two_dead_reviewers_in_one_sweep_arm_backoff_once(tmp_path: Path) -> None:
    env = _Env(tmp_path)
    env.plant_dead_reviewer(_result_event(529), pr=PR)
    env.plant_dead_reviewer(_result_event(529), pr=PR + 1)
    state = env.sweep()
    for pr_no in (PR, PR + 1):
        assert state["prs"][str(pr_no)]["review_dispatch_attempt_count"] == 0
        assert state["prs"][str(pr_no)]["review_api_error_streak"] == 1
    outage = [e for e in state["events"] if e["kind"] == "review_provider_outage"]
    assert len(outage) == 1
    assert state["reviewer_quota"]["consecutive_probe_failures"] == 1
