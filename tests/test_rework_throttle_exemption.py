"""Issue #2282: provider-throttle rework deaths never use up a rework attempt.

Every test here runs the real lanes. ``OrchestratorApp.dispatch_rework`` stamps
``redispatch_at`` and applies the caps. The orphan sweep
(``_detect_and_handle_orphaned_workers``) recovers the dead rework session and
classifies its death. Nothing calls a credit or record helper directly. The
only things faked are the process launch (``dispatch_sessions``), PID liveness,
and the provider window expiring between waves (``throttled_until`` is cleared
so the next dispatch is not held by the cooldown the death armed).

Before the fix, a throttle death was not credited to ``worker_death_at``, but
its dispatch stamp stayed in ``redispatch_at``. Every redispatch cap then read
it as a no-op, and the third wave escalated a healthy PR (#2254 / PR #2264).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _dead_session_fixtures import _write_flat_review_decision
from _fakes_github import FakeGitHub
from _orphan_sweep_fixtures import _run_orphan_sweep

from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.rework_attempt_exemption import (
    EXEMPTION_EVENT_KIND,
    PROVIDER_THROTTLE_EXEMPT_KINDS,
)
from charlie_work.role_quota_ledger import RESTRICTING_FAILURE_KINDS, session_stamp
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.throttle_signatures import PROVIDER_THROTTLE_FAILURE_KINDS
from charlie_work.worker_fate import profile_for
from charlie_work.workflow import OrchestratorApp

CAP = 2
ISSUE, PR, HEAD = 123, 456, "sha-abc123"
FREE_MODEL_RATE_LIMIT_LINE = (
    "Error: Agent error: Reached free model rate limit. Upgrade to Max for higher "
    "limits, or switch to a different model. Your limit will reset in 45 minutes."
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class _Bed:
    """One rework issue (#123) whose PR (#456) has a live ``request_changes`` verdict."""

    def __init__(self, tmp_path: Path) -> None:
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
        )
        self.paths = runtime_paths(tmp_path, self.config.runtime.state_dir)
        self.gh = FakeGitHub(repo_root=tmp_path)
        self.gh.issues[0]["labels"] = [{"name": self.config.labels.needs_rework}]
        self.sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state["issues"][str(ISSUE)] = {
                "number": ISSUE,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "status": "rework_requested",
                "branch_name": "agent/issue-123-fix-search",
            }
            state["prs"][str(PR)] = {
                "number": PR,
                "issue_number": ISSUE,
                "decision": "request_changes",
                "reviewed_head_sha": HEAD,
            }
            save_state(self.paths.state_file, state)
        _write_flat_review_decision(self.paths, PR, "request_changes", HEAD)
        (self.paths.prs / f"pr-{PR}" / "rework-prompt.md").write_text(
            "Fix the issues", encoding="utf-8"
        )
        self.app = OrchestratorApp(tmp_path, self.paths, self.config, self.gh)

    def entry(self) -> dict[str, Any]:
        return load_state(self.paths.state_file)["issues"][str(ISSUE)]

    def events(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in load_state(self.paths.state_file)["events"] if e.get("kind") == kind]

    def dispatch(self) -> Any:
        """One real ``dispatch_rework`` pass; only the process launch is faked."""
        import dataclasses

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
        with patch.object(host_pkg, "_ACTIVE", ports):
            return self.app.dispatch_rework()

    def die(
        self,
        *,
        stamped_kind: str | None = None,
        log_text: str | None = None,
        exit_code: int | None = None,
        worker_outcome: dict[str, Any] | None = None,
        role_stamp: dict[str, Any] | None = None,
    ) -> None:
        """End the dispatched session and run the real orphan sweep over it.

        ``stamped_kind`` is the classification an earlier lane (the stalled or
        dead-session classifier) already stamped. ``log_text`` is the captured
        worker log, left unclassified so the sweep has to classify it.
        ``role_stamp`` writes the session's ``role_entry`` (#2279) into the
        sidecar, as a launch-time stamp would. ``exit_code`` writes the
        worker's terminal record. ``worker_outcome``
        embeds an outcome in that record, making
        ``terminal_record_proves_completion`` hold for this pid -- the record's
        filename then carries the devin suffix, matching the sidecar's adapter,
        as the real watcher writes it.
        """
        for stale in self.sessions_dir.glob(f"issue-{ISSUE}*"):
            stale.unlink()
        if stamped_kind is not None:
            with state_lock(self.paths.state_file):
                state = load_state(self.paths.state_file)
                state["issues"][str(ISSUE)]["dead_worker_failure_kind"] = stamped_kind
                save_state(self.paths.state_file, state)
        if log_text is not None:
            log_path = self.sessions_dir / f"issue-{ISSUE}-rework.log"
            log_path.write_text(f"applying rework\n{log_text}\n", encoding="utf-8")
            sidecar = {
                "issue_number": ISSUE,
                "branch": "agent/issue-123-fix-search",
                "worktree_path": "",
                "prompt_path": "",
                "command": [],
                "pid": 99999,
                "started_at": _now(),
                "log_path": str(log_path),
            }
            if role_stamp is not None:
                sidecar["role_entry"] = role_stamp
            (self.sessions_dir / f"issue-{ISSUE}.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
        if exit_code is not None:
            record = {
                "pid": 99999,
                "exit_code": exit_code,
                "started_at": _now(),
                "ended_at": _now(),
                "duration_seconds": 300.0,
            }
            if worker_outcome is not None:
                record["worker_outcome"] = worker_outcome
            suffix = "devin" if worker_outcome is not None else "command"
            (self.sessions_dir / f"issue-{ISSUE}.{suffix}.terminal.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
        _run_orphan_sweep(self.tmp_path, self.paths, self.config, self.gh)
        # The provider window expires before the next wave.
        with state_lock(self.paths.state_file):
            state = load_state(self.paths.state_file)
            state.pop("throttled_until", None)
            save_state(self.paths.state_file, state)

    def assert_dispatched_not_escalated(self, result: Any, wave: int) -> None:
        entry = self.entry()
        assert entry.get("status") == "dispatched", (wave, entry.get("escalation_reason"))
        assert ISSUE not in result.data.get("no_op_rework_escalated", []), wave
        assert ISSUE not in result.data.get("worker_death_escalated", []), wave


def test_exempt_kinds_derive_from_the_restricting_ledger_kinds() -> None:
    """The exemption covers every kind the ledger restricts on and every kind the
    ``worker_death_at`` credit gate already skips. If either set grows, the
    refund grows with it."""
    assert RESTRICTING_FAILURE_KINDS <= PROVIDER_THROTTLE_EXEMPT_KINDS
    assert PROVIDER_THROTTLE_FAILURE_KINDS <= PROVIDER_THROTTLE_EXEMPT_KINDS


@pytest.mark.parametrize("kind", sorted(RESTRICTING_FAILURE_KINDS))
def test_three_consecutive_throttle_rework_deaths_do_not_escalate(
    tmp_path: Path, kind: str
) -> None:
    """(a) Three provider-throttle rework deaths in a row, with a cap of 2, never
    escalate. Each death refunds its own dispatch stamp, so the redispatch
    history never grows past the one live dispatch."""
    bed = _Bed(tmp_path)
    for wave in range(1, 4):
        bed.assert_dispatched_not_escalated(bed.dispatch(), wave)
        assert len(bed.entry().get("redispatch_at", [])) == 1, wave
        bed.die(stamped_kind=kind)
        entry = bed.entry()
        assert entry.get("status") == "rework_requested", wave
        assert entry.get("redispatch_at") == [], wave
        assert not entry.get("worker_death_at"), wave

    bed.assert_dispatched_not_escalated(bed.dispatch(), 4)
    exempted = bed.events(EXEMPTION_EVENT_KIND)
    assert len(exempted) == 3
    assert {e["payload"]["failure_kind"] for e in exempted} == {kind}
    assert all(e["payload"]["pr_number"] == PR for e in exempted)
    assert all(e["payload"]["refunded_redispatch_at"] for e in exempted)
    pr_state = load_state(bed.paths.state_file)["prs"][str(PR)]
    # The next dispatch cleared the PR flag; that dispatch is the one the janitor
    # gate will judge.
    assert pr_state.get("last_rework_was_startup_death") is False


def test_unclassified_death_with_free_model_rate_limit_log_is_exempted(
    tmp_path: Path,
) -> None:
    """(b) No lane stamped a kind. The captured log ends in the verbatim Devin
    free-model notice. The sweep classifies the log before the cap decision, and
    three such waves never escalate."""
    bed = _Bed(tmp_path)
    for wave in range(1, 4):
        bed.assert_dispatched_not_escalated(bed.dispatch(), wave)
        assert "dead_worker_failure_kind" not in bed.entry()
        bed.die(log_text=FREE_MODEL_RATE_LIMIT_LINE)
        entry = bed.entry()
        assert entry.get("status") == "rework_requested", wave
        assert entry.get("dead_worker_failure_kind") == "rate_limited", wave
        assert entry.get("redispatch_at") == [], wave

    bed.assert_dispatched_not_escalated(bed.dispatch(), 4)
    exempted = bed.events(EXEMPTION_EVENT_KIND)
    assert len(exempted) == 3
    assert exempted[0]["payload"] == {
        "issue_number": ISSUE,
        "pr_number": PR,
        "failure_kind": "rate_limited",
        "source": "orphan_sweep_credit",
        "refunded_redispatch_at": exempted[0]["payload"]["refunded_redispatch_at"],
        "redispatch_count": 0,
    }


def test_genuine_no_op_rework_still_escalates(tmp_path: Path) -> None:
    """(c) Positive control: the session ran, exited 0, and pushed nothing. That
    is a real no-op. It is not exempted, its dispatch stamp stays counted, and
    the issue escalates."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    bed.die(log_text="worked on it, nothing to change", exit_code=0)

    entry = bed.entry()
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "rework_no_op"
    assert len(entry.get("redispatch_at", [])) == 1
    assert bed.events(EXEMPTION_EVENT_KIND) == []


def test_clean_exit_rate_limit_death_is_exempted_not_a_no_op(tmp_path: Path) -> None:
    """(d) Issue #2286: an exit-0 rework session whose captured log ENDS in the
    verbatim provider free-model rate-limit error is a throttle death, not a
    no-op attempt. Three such waves restore ``rework_requested`` every time,
    refund each dispatch stamp, stamp the failure kind, and emit the exemption
    from the clean-exit lane -- ``no_op_rework_attempts`` never advances."""
    bed = _Bed(tmp_path)
    for wave in range(1, 4):
        bed.assert_dispatched_not_escalated(bed.dispatch(), wave)
        assert "dead_worker_failure_kind" not in bed.entry()
        bed.die(log_text=FREE_MODEL_RATE_LIMIT_LINE, exit_code=0)
        entry = bed.entry()
        assert entry.get("status") == "rework_requested", wave
        assert entry.get("dead_worker_failure_kind") == "rate_limited", wave
        assert entry.get("redispatch_at") == [], wave
        assert not entry.get("worker_death_at"), wave
        pr_state = load_state(bed.paths.state_file)["prs"][str(PR)]
        assert pr_state.get("last_rework_was_startup_death") is True, wave
        assert pr_state.get("last_rework_exemption") == "provider_throttle", wave
        assert pr_state.get("last_rework_failure_kind") == "rate_limited", wave

    bed.assert_dispatched_not_escalated(bed.dispatch(), 4)
    exempted = bed.events(EXEMPTION_EVENT_KIND)
    assert len(exempted) == 3
    assert {e["payload"]["failure_kind"] for e in exempted} == {"rate_limited"}
    assert {e["payload"]["source"] for e in exempted} == {"orphan_sweep_clean_exit"}
    assert all(e["payload"]["refunded_redispatch_at"] for e in exempted)
    recovered = bed.events("orphaned_worker_recovered")
    assert {e["payload"]["reason"] for e in recovered} == {"dead_worker_clean_exit_throttle"}


def test_clean_exit_throttle_arm_stamps_the_window_with_the_dead_role(
    tmp_path: Path,
) -> None:
    """Issue #2279: the #2286 clean-exit exemption lane arms the per-repo
    throttle window through ``persist_failure`` like every other lane, so the
    evidence must carry the dead session's ``role_entry`` -- otherwise a
    fallback's window would block the recovered primary entry."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    bed.die(
        log_text=FREE_MODEL_RATE_LIMIT_LINE,
        exit_code=0,
        role_stamp=session_stamp("worker", "devin-shell", "swe-2-high", 0),
    )
    state = load_state(bed.paths.state_file)
    assert state["throttle_harness"] == "devin-shell"
    assert state["throttle_model"] == "swe-2-high"


def test_clean_exit_proven_complete_with_rate_limit_log_is_not_exempted(
    tmp_path: Path,
) -> None:
    """(e) Issue #656 control: this pid's terminal record proves completion
    (exit 0 WITH a worker outcome), so a transcript ending in the rate-limit
    error is completion-era quoting -- the exemption never fires and the
    attempt is still counted as a no-op."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    bed.die(
        log_text=FREE_MODEL_RATE_LIMIT_LINE,
        exit_code=0,
        worker_outcome={"status": "completed", "head_sha": HEAD},
    )

    entry = bed.entry()
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "rework_no_op"
    assert len(entry.get("redispatch_at", [])) == 1
    assert bed.events(EXEMPTION_EVENT_KIND) == []


def test_clean_exit_with_earlier_rate_limit_mention_is_not_exempted(
    tmp_path: Path,
) -> None:
    """(f) Only the TERMINAL lines may carry the signature. A transcript that
    mentions the rate limit earlier but ends in ordinary output is a genuine
    no-op -- the whole-tail classifier would fire on it, the anchored check
    does not -- so the attempt counts and the issue escalates."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    mid_log = (
        "Error: Reached free model rate limit. Retrying.\n"
        "the retry succeeded\nfinished reviewing the diff\nran the tests\n"
        "wrote the summary\nall done, nothing left to change"
    )
    bed.die(log_text=mid_log, exit_code=0)

    entry = bed.entry()
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "rework_no_op"
    assert len(entry.get("redispatch_at", [])) == 1
    assert bed.events(EXEMPTION_EVENT_KIND) == []


def test_non_throttle_rework_deaths_still_escalate_at_the_cap(tmp_path: Path) -> None:
    """(c) Positive control at the cap: the same three waves with an ordinary
    crash are counted and escalate. This proves the harness can escalate, and
    that only throttle kinds are exempted."""
    bed = _Bed(tmp_path)
    escalated_at: int | None = None
    for wave in range(1, 5):
        result = bed.dispatch()
        if bed.entry().get("status") == "escalated" or ISSUE in result.data.get(
            "worker_death_escalated", []
        ):
            escalated_at = wave
            break
        bed.die(log_text="Traceback (most recent call last): boom")
        assert bed.entry().get("worker_death_at"), wave

    assert escalated_at is not None and escalated_at <= CAP + 1
    assert bed.entry().get("status") == "escalated"
    assert bed.events(EXEMPTION_EVENT_KIND) == []


def test_dead_session_restore_lane_refunds_and_flags_the_pr(tmp_path: Path) -> None:
    """The dead-session lane's restore (``_reap_restore_rework_requested``) uses
    the same enforcement point. It refunds the stamp the real dispatch wrote and
    sets the #1106 PR flag, which ``_route_janitor_gate_failure_to_rework``
    honors, so ``no_op_rework_attempts`` / ``conflict_rework_attempts`` are not
    advanced either."""
    from _rework_dispatch_fixtures import _wg

    from charlie_work.dead_worker_sweep.effects_rework import _reap_restore_rework_requested
    from charlie_work.worker import WorkerView

    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    assert len(bed.entry()["redispatch_at"]) == 1
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=ISSUE,
        repo_key="",
        pid=None,
        started_at=_now(),
        process_start_time=None,
        log_path=str(tmp_path / "issue-123.log"),
        worktree_path=str(tmp_path / "wt" / "issue-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
    )

    _reap_restore_rework_requested(
        bed.paths.state_file,
        bed.gh,
        bed.config,
        {ISSUE: list(bed.gh.prs)},
        worker,
        failure_kind="rate_limited",
        repo_root=None,
        write_gate=_wg(bed.paths.state_file),
    )

    state = load_state(bed.paths.state_file)
    entry = state["issues"][str(ISSUE)]
    assert entry["status"] == "rework_requested"
    assert entry["redispatch_at"] == []
    assert not entry.get("worker_death_at")
    pr_state = state["prs"][str(PR)]
    assert pr_state["last_rework_was_startup_death"] is True
    assert pr_state["last_rework_exemption"] == "provider_throttle"
    assert pr_state["last_rework_failure_kind"] == "rate_limited"
    (exempted,) = bed.events(EXEMPTION_EVENT_KIND)
    assert exempted["payload"]["source"] == "dead_rework_session_restore"
    (requeued,) = bed.events("rework_requeued")
    assert requeued["payload"]["provider_throttle_exempt"] is True
    assert requeued["payload"]["startup_death"] is False


@pytest.mark.parametrize(
    "log_text",
    [None, "worked on it, nothing to change"],
    ids=["no_log", "non_throttle_log"],
)
def test_clean_exit_stamped_throttle_is_exempted_without_terminal_signature(
    tmp_path: Path, log_text: str | None
) -> None:
    """(g) Issue #2286, stamp-first resolution: the dead-session lane often
    classifies and stamps ``dead_worker_failure_kind`` before the orphan sweep
    reaches the issue. A clean exit already stamped ``rate_limited`` exempts on
    the stamp alone -- the terminal-line check exists for UNCLASSIFIED deaths,
    so an absent log or a non-throttle one cannot veto the stamp. The dead
    session's dispatch stamp is refunded, the PR is flagged (so
    ``no_op_rework_attempts`` never advances), and the exemption emits from the
    clean-exit lane."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    assert len(bed.entry()["redispatch_at"]) == 1
    bed.die(stamped_kind="rate_limited", log_text=log_text, exit_code=0)

    entry = bed.entry()
    assert entry.get("status") == "rework_requested"
    assert entry.get("dead_worker_failure_kind") == "rate_limited"
    assert entry.get("redispatch_at") == []
    assert not entry.get("worker_death_at")
    pr_state = load_state(bed.paths.state_file)["prs"][str(PR)]
    assert pr_state.get("last_rework_was_startup_death") is True
    assert pr_state.get("last_rework_exemption") == "provider_throttle"
    assert pr_state.get("last_rework_failure_kind") == "rate_limited"
    assert pr_state.get("no_op_rework_attempts", 0) == 0
    (exempted,) = bed.events(EXEMPTION_EVENT_KIND)
    assert exempted["payload"]["issue_number"] == ISSUE
    assert exempted["payload"]["pr_number"] == PR
    assert exempted["payload"]["failure_kind"] == "rate_limited"
    assert exempted["payload"]["source"] == "orphan_sweep_clean_exit"
    assert exempted["payload"]["refunded_redispatch_at"]
    (recovered,) = bed.events("orphaned_worker_recovered")
    assert recovered["payload"]["reason"] == "dead_worker_clean_exit_throttle"
    assert recovered["payload"]["failure_kind"] == "rate_limited"


@pytest.mark.parametrize("kind", ["provider_suspended", "permission_denied"])
def test_clean_exit_stamped_non_throttle_still_counts_as_no_op(tmp_path: Path, kind: str) -> None:
    """(h) Negative companion to (g): the stamp-first branch resolves the kind,
    but the exemption itself still gates on ``PROVIDER_THROTTLE_EXEMPT_KINDS``.
    A clean exit stamped with a non-throttle kind -- ``provider_suspended``
    sits just outside the exempt set by design (terminal, no cooldown) and
    ``permission_denied`` is a config defect, not a provider death -- is NOT
    exempted: the dispatch stamp stays counted and the attempt routes to a
    real no-op."""
    bed = _Bed(tmp_path)
    bed.assert_dispatched_not_escalated(bed.dispatch(), 1)
    bed.die(stamped_kind=kind, exit_code=0)

    entry = bed.entry()
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "rework_no_op"
    assert entry.get("dead_worker_failure_kind") == kind
    assert len(entry.get("redispatch_at", [])) == 1
    pr_state = load_state(bed.paths.state_file)["prs"][str(PR)]
    assert not pr_state.get("last_rework_was_startup_death")
    assert bed.events(EXEMPTION_EVENT_KIND) == []


@pytest.mark.parametrize(
    ("adapter_kind", "terminal", "expected"),
    [
        # The quota signature fires for every adapter -- the structured
        # resource_exhausted trailer and the config prose fallback alike.
        ("devin", '{"cognition.ai/errorKind": "resource_exhausted"}', True),
        ("claude-code", "Error: weekly usage quota has been exhausted", True),
        ("api", '{"cognition.ai/errorKind": "resource_exhausted"}', True),
        # The generic throttle markers fire for every adapter, with or
        # without a resolved profile.
        ("devin", FREE_MODEL_RATE_LIMIT_LINE, True),
        (None, FREE_MODEL_RATE_LIMIT_LINE, True),
        # Provider-auth detection belongs to the api profile alone (#484):
        # every other adapter (and an unresolved one) must ignore an auth
        # failure sitting in the terminal lines.
        ("api", "API Error: 401 Unauthorized - invalid api key", True),
        ("devin", "API Error: 401 Unauthorized - invalid api key", False),
        ("claude-code", "API Error: 401 Unauthorized - invalid api key", False),
        (None, "API Error: 401 Unauthorized - invalid api key", False),
        # No provider signature at all.
        ("devin", "ran the tests\nall done, nothing left to change", False),
        ("api", "ran the tests\nall done, nothing left to change", False),
    ],
)
def test_terminal_is_throttle_error_signature_gating(
    adapter_kind: str | None, terminal: str, expected: bool
) -> None:
    """(i) Table-driven pin of ``_terminal_is_throttle_error``: the quota
    signature and generic throttle markers admit every adapter, while
    provider-auth detection is gated on ``profile.account_error_detection``
    (api only) -- a devin or claude-code worker quoting a 401 never
    classifies as a provider death."""
    from charlie_work.dead_worker_classification import _terminal_is_throttle_error

    profile = profile_for(adapter_kind) if adapter_kind is not None else None
    assert _terminal_is_throttle_error(terminal, profile, OrchestratorConfig()) is expected
