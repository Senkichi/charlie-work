"""Issue #3025: every lane that ends a dead rework session applies the throttle exemption.

Two approved PRs in a served repo were operator-queued because their merge-conflict
rework sessions died on a Devin free-model rate limit. The dead-session lane's
pre-review route (``_route_dead_worker_to_pre_review_rework``) resolved the death as
``rate_limited`` and used that only to skip adding a stamp. It never called
``exempt_provider_throttle_rework_death``, so the dead session's own dispatch stamp
stayed in ``redispatch_at``. The route's cap check read that stamp as one attempt too
many and escalated through the ``redispatch_escalated`` edge. That edge emits no
``session_failed_escalated`` event, which is why the death left no trace.

Every test runs the real lanes: ``OrchestratorApp.dispatch_rework`` stamps
``redispatch_at`` and ``classify_dead_sessions`` / the orphan sweep recover the dead
session from a sidecar plus the verbatim log from the issue. Only the process launch
and the provider window expiring between waves are faked. The static counterpart
that stops a new lane from skipping the exemption is
``tests/test_dead_rework_exemption_guard.py``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _dead_session_fixtures import _write_flat_review_decision
from _fakes_github import FakeGitHub
from _orphan_sweep_fixtures import _run_orphan_sweep
from _rework_dispatch_fixtures import _wg

from charlie_work import devin_shell
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    ReviewConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.dead_worker_sweep.dead_sessions import classify_dead_sessions
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.rework_attempt_exemption import EXEMPTION_EVENT_KIND
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

CAP = 2
ISSUE, PR, HEAD = 123, 456, "sha-abc123"
BRANCH = "agent/issue-123-fix-search"
# Verbatim from the two escalated PRs' rework logs (issue #3025). The structured
# trailer says ``unavailable``, not ``resource_exhausted``, so this is a rate-limit
# death by the prose marker only.
DEVIN_FREE_MODEL_RATE_LIMIT_LOG = (
    "Error: Agent error: Reached free model rate limit. Upgrade to Max for higher "
    "limits, or switch to a different model. Your limit will reset in 5 minutes "
    "(at 01:51 UTC).: {\n"
    '  "cognition.ai/errorKind": "unavailable",\n'
    '  "cognition.ai/retryable": true\n'
    "}\n"
)
# What a lane escalating a dead rework session writes: the label edge and the
# ``lifecycle_transition`` cause it records.
ESCALATION_CAUSES = frozenset({"redispatch_escalated", "redispatch_operator_queued"})


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _pr_row(*, conflicting: bool) -> dict[str, Any]:
    return {
        "number": PR,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": BRANCH,
        "baseRefName": "main",
        "headRefOid": HEAD,
        "mergeStateStatus": "DIRTY" if conflicting else "CLEAN",
        "mergeable": "CONFLICTING" if conflicting else "MERGEABLE",
        "body": "Closes #123\n\nTests: regression coverage added.",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


class _Bed:
    """Issue #123 in rework, its PR #456 either conflicting (approved) or ``request_changes``."""

    def __init__(self, tmp_path: Path, *, conflicting: bool) -> None:
        self.tmp_path = tmp_path
        self.config = OrchestratorConfig(
            devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
            worker=WorkerRoleConfig(harness="command"),
            watchdog=WatchdogConfig(
                enabled=True,
                stall_minutes=20,
                max_auto_redispatch=CAP,
                redispatch_window_minutes=240,
            ),
            review=ReviewConfig(max_conflict_rework_attempts=CAP, max_no_op_rework_attempts=CAP),
            auto_merge=AutoMergeConfig(
                required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
                update_open_prs="next",
                failed_attempt_alarm=1,
            ),
            # Keep the dead-session lane's post-mortem off the host's real devin store.
            post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        )
        self.paths = runtime_paths(tmp_path, self.config.runtime.state_dir)
        self.gh = FakeGitHub(repo_root=tmp_path)
        self.gh.prs = [_pr_row(conflicting=conflicting)]
        self.gh.issues[0]["labels"] = [{"name": self.config.labels.needs_rework}]
        self.sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        decision = "approved" if conflicting else "request_changes"
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state["issues"][str(ISSUE)] = {
                "number": ISSUE,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "status": "rework_requested",
                "branch_name": BRANCH,
            }
            state["prs"][str(PR)] = {
                "number": PR,
                "issue_number": ISSUE,
                "decision": decision,
                "reviewed_head_sha": HEAD,
            }
            save_state(self.paths.state_file, state)
        _write_flat_review_decision(self.paths, PR, decision, HEAD)
        (self.paths.prs / f"pr-{PR}" / "rework-prompt.md").write_text(
            "Resolve the merge conflict", encoding="utf-8"
        )
        self.app = OrchestratorApp(tmp_path, self.paths, self.config, self.gh)

    def state(self) -> dict[str, Any]:
        return load_state(self.paths.state_file)

    def entry(self) -> dict[str, Any]:
        return self.state()["issues"][str(ISSUE)]

    def events(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.state()["events"] if e.get("kind") == kind]

    def escalation_transitions(self) -> list[dict[str, Any]]:
        """Recorded ``lifecycle_transition`` rows that came from an escalation edge."""
        rows = query_events(self.paths.state_file, kind="lifecycle_transition", issue_number=ISSUE)
        return [r for r in rows if r["payload"].get("cause") in ESCALATION_CAUSES]

    def dispatch(self) -> Any:
        """One real ``dispatch_rework`` pass; only the process launch is faked."""
        import charlie_work.host as host_pkg
        from charlie_work.adapters import SessionDispatchResult
        from charlie_work.host.fakes import FakeWorkerLauncher

        def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
            return [
                SessionDispatchResult(
                    issue_number=request.issue_number,
                    issue_title=request.issue_title,
                    prompt_path=str(request.prompt_path),
                    branch_name=request.branch_name,
                    adapter="command",
                    ok=True,
                    pid=99999,
                    process_start_time=datetime.now(UTC).isoformat(),
                )
                for request in requests
            ]

        ports = dataclasses.replace(
            host_pkg.current(), worker_launch=FakeWorkerLauncher([fake_dispatch_sessions])
        )
        with (
            patch.object(host_pkg, "_ACTIVE", ports),
            patch("charlie_work.process_utils._enumerate_child_pids", lambda _pid: []),
        ):
            return self.app.dispatch_rework()

    def _window_expires(self) -> None:
        """The provider window the death armed expires before the next wave."""
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state.pop("throttled_until", None)
            save_state(self.paths.state_file, state)

    def die_in_dead_session_lane(self, log_text: str) -> list[dict[str, Any]]:
        """End the dispatched session and run the real dead-session lane over its sidecar."""
        log_path = self.sessions_dir / f"issue-{ISSUE}.log"
        log_path.write_text(f"applying rework\n{log_text}", encoding="utf-8")
        record = devin_shell.SessionRecord(
            issue_number=ISSUE,
            branch=BRANCH,
            # Absent on purpose: the inspection reads ``UNKNOWN`` without spawning git,
            # so the lane falls through to the log-tail classification under test.
            worktree_path=str(self.tmp_path / "wt" / f"issue-{ISSUE}"),
            prompt_path=str(self.tmp_path / "prompt.md"),
            command=("devin", "--prompt-file", "prompt.md"),
            pid=None,
            started_at=_now(),
            log_path=str(log_path),
            error=None,
        )
        (self.sessions_dir / f"issue-{ISSUE}.json").write_text(
            json.dumps(record.to_dict()), encoding="utf-8"
        )
        reaped = classify_dead_sessions(
            self.sessions_dir, self.paths.state_file, self.gh, self.config,
            write_gate=_wg(self.paths.state_file),
        )  # fmt: skip
        self._window_expires()
        return reaped

    def die_in_orphan_sweep(self, log_text: str, monkeypatch) -> None:
        """End the dispatched session and run the real state.json-keyed orphan sweep."""
        for stale in self.sessions_dir.glob(f"issue-{ISSUE}*"):
            stale.unlink()
        log_path = self.sessions_dir / f"issue-{ISSUE}-rework.log"
        log_path.write_text(f"applying rework\n{log_text}", encoding="utf-8")
        sidecar = {
            "issue_number": ISSUE,
            "branch": BRANCH,
            "worktree_path": "",
            "prompt_path": "",
            "command": [],
            "pid": 99999,
            "started_at": _now(),
            "log_path": str(log_path),
        }
        (self.sessions_dir / f"issue-{ISSUE}.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        _run_orphan_sweep(self.tmp_path, self.paths, self.config, self.gh, monkeypatch=monkeypatch)
        self._window_expires()

    def assert_not_escalated(self, wave: int) -> None:
        entry = self.entry()
        assert entry.get("status") != "escalated", (wave, entry.get("escalation_reason"))
        assert self.escalation_transitions() == [], wave
        assert self.events("session_failed_escalated") == [], wave


def test_conflict_rework_throttle_deaths_replay_never_escalate(tmp_path: Path) -> None:
    """Replay of #3025: an approved PR with a merge conflict, a rework session that dies
    on the free-model rate limit, three waves.

    Each death emits the exemption event and refunds the dead session's own dispatch
    stamp, the PR flag keeps the janitor's conflict counter still, and neither
    ``no_op_rework_cap_exceeded`` nor ``conflict_rework_attempts_cap_exceeded`` fires.
    """
    bed = _Bed(tmp_path, conflicting=True)
    for wave in range(1, 4):
        result = bed.dispatch()
        assert bed.entry()["status"] == "dispatched", (wave, result.data)
        assert len(bed.entry()["redispatch_at"]) == 1, wave

        reaped = bed.die_in_dead_session_lane(DEVIN_FREE_MODEL_RATE_LIMIT_LOG)

        assert [r["failure_kind"] for r in reaped] == ["rate_limited"], wave
        bed.assert_not_escalated(wave)
        entry = bed.entry()
        assert entry["status"] == "rework_requested", wave
        assert entry["redispatch_at"] == [], wave
        assert not entry.get("worker_death_at"), wave
        assert len(bed.events(EXEMPTION_EVENT_KIND)) == wave
        pr_state = bed.state()["prs"][str(PR)]
        assert pr_state["last_rework_was_startup_death"] is True, wave
        assert pr_state["last_rework_exemption"] == "provider_throttle", wave

        # The janitor re-detects the conflict every pass; it must not count this attempt.
        bed.app.merge_ready(PR, merge=False)
        pr_state = bed.state()["prs"][str(PR)]
        assert pr_state.get("conflict_rework_attempts", 0) == 0, wave
        assert pr_state.get("no_op_rework_attempts", 0) == 0, wave
        assert bed.events("janitor_rework_escalated") == [], wave

    final = bed.dispatch()
    assert bed.entry()["status"] == "dispatched"
    assert bed.escalation_transitions() == []
    assert not final.data.get("no_op_rework_escalated")
    reasons = {e["payload"].get("reason") for e in bed.events("session_failed_escalated")}
    assert "no_op_rework_cap_exceeded" not in reasons
    assert "conflict_rework_attempts_cap_exceeded" not in bed.entry().get(
        "escalation_reasons_seen", []
    )


def test_conflict_rework_ordinary_deaths_still_escalate_at_the_cap(tmp_path: Path) -> None:
    """Positive control for the replay: the same bed with an ordinary crash is counted and
    escalates at the cap, so the replay's silence is the exemption and not a dead harness."""
    bed = _Bed(tmp_path, conflicting=True)
    escalated_at: int | None = None
    for wave in range(1, CAP + 3):
        bed.dispatch()
        if bed.entry()["status"] == "escalated":
            escalated_at = wave
            break
        bed.die_in_dead_session_lane("Traceback (most recent call last): boom\n")
        if bed.entry()["status"] == "escalated":
            escalated_at = wave
            break

    assert escalated_at is not None and escalated_at <= CAP + 1
    assert bed.events(EXEMPTION_EVENT_KIND) == []


@pytest.mark.parametrize(
    "lane", ["dead_session_restore", "dead_session_pre_review", "orphan_sweep"]
)
def test_throttle_death_takes_no_escalation_edge_in_any_lane(
    tmp_path: Path, lane: str, monkeypatch
) -> None:
    """The same rate-limit death, ended by each lane that can end a dead rework session,
    leaves the issue in rework with no ``redispatch_escalated`` /
    ``redispatch_operator_queued`` edge, even when the history is one stamp short of the
    cap (so a death that kept its dispatch stamp would cross it).

    ``dead_session_restore`` is the ``request_changes`` PR route,
    ``dead_session_pre_review`` the conflicting-PR route, and ``orphan_sweep`` the
    state.json-keyed sweep that runs after the sidecar is reaped.
    """
    bed = _Bed(tmp_path, conflicting=lane == "dead_session_pre_review")
    # One earlier counted attempt: the cap is CAP, so this dispatch makes CAP.
    with state_lock(bed.paths.state_file):
        state = load_state(bed.paths.state_file)
        state["issues"][str(ISSUE)]["redispatch_at"] = [_now()] * (CAP - 1)
        state["issues"][str(ISSUE)]["worker_death_at"] = [_now()] * (CAP - 1)
        save_state(bed.paths.state_file, state)
    bed.dispatch()
    assert bed.entry()["status"] == "dispatched"
    assert len(bed.entry()["redispatch_at"]) == CAP

    if lane == "orphan_sweep":
        bed.die_in_orphan_sweep(DEVIN_FREE_MODEL_RATE_LIMIT_LOG, monkeypatch)
    else:
        bed.die_in_dead_session_lane(DEVIN_FREE_MODEL_RATE_LIMIT_LOG)

    bed.assert_not_escalated(1)
    entry = bed.entry()
    assert entry["status"] == "rework_requested"
    assert len(entry["redispatch_at"]) == CAP - 1
    assert len(bed.events(EXEMPTION_EVENT_KIND)) == 1
    assert bed.state()["prs"][str(PR)]["last_rework_was_startup_death"] is True


def _seed_dispatched(bed: _Bed, *, prior_counted: int) -> None:
    """A dispatched rework session whose own stamp is the newest of ``prior_counted + 1``.

    ``dispatched_at`` precedes the session's stamp, which precedes now; the earlier
    stamps are older attempts inside the redispatch window. This is the shape a real
    dispatch leaves behind (``dispatched_at`` written first, then the stamp).
    """
    now = datetime.now(UTC)
    older = [(now - timedelta(minutes=30 + i)).isoformat() for i in range(prior_counted)]
    dispatched_at = (now - timedelta(minutes=5)).isoformat()
    own_stamp = (now - timedelta(minutes=5) + timedelta(seconds=1)).isoformat()
    with state_lock(bed.paths.state_file):
        state = load_state(bed.paths.state_file)
        state["issues"][str(ISSUE)].update(
            status="dispatched",
            dispatched_at=dispatched_at,
            redispatch_at=[*older, own_stamp],
        )
        save_state(bed.paths.state_file, state)


def test_pre_review_route_refunds_the_stamp_that_would_cross_the_cap(tmp_path: Path) -> None:
    """The incident's boundary: the dead session's own dispatch stamp is the one that takes
    the history to ``cap + 1``. Before #3025 the route skipped adding a stamp but still
    counted that one and escalated through ``redispatch_operator_queued``, leaving no
    ``session_failed_escalated`` event. With the refund it is back to ``cap`` and the
    issue is restored to rework."""
    bed = _Bed(tmp_path, conflicting=True)
    _seed_dispatched(bed, prior_counted=CAP)
    assert len(bed.entry()["redispatch_at"]) == CAP + 1

    bed.die_in_dead_session_lane(DEVIN_FREE_MODEL_RATE_LIMIT_LOG)

    bed.assert_not_escalated(1)
    entry = bed.entry()
    assert entry["status"] == "rework_requested"
    assert len(entry["redispatch_at"]) == CAP
    (exempted,) = bed.events(EXEMPTION_EVENT_KIND)
    assert exempted["payload"]["source"] == "dead_pre_review_rework_route"
    assert exempted["payload"]["failure_kind"] == "rate_limited"


def test_pre_review_route_genuine_no_op_still_counts_toward_the_cap(tmp_path: Path) -> None:
    """Positive control for the boundary test, same bed and same history: a non-throttle death
    keeps its stamp (nothing is refunded, no exemption event) and escalates on the cap."""
    bed = _Bed(tmp_path, conflicting=True)
    _seed_dispatched(bed, prior_counted=CAP)

    bed.die_in_dead_session_lane("worker stalled with no output\n")

    entry = bed.entry()
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"
    assert [r["payload"]["cause"] for r in bed.escalation_transitions()] == [
        "redispatch_operator_queued"
    ]
    assert bed.events(EXEMPTION_EVENT_KIND) == []
