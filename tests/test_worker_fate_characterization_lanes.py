"""Characterization tests, part 2: flip 4/7/8/9 and the preserved paths.

Continues ``test_worker_fate_characterization.py`` (split for the file-size
ratchet); the shared builders live in ``_worker_fate_characterization_fixtures``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep, _write_outcome
from _worktree_fixtures import _git, _init_bare_remote_and_clone

from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import resolved_layout
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state
from charlie_work.worktree import worktree_path_for_branch
from _worker_fate_characterization_fixtures import (
    _no_pr_fake_gh,
    _seed_no_pr_dispatched_issue,
    _wg,
)
from _host_fixtures import host_probe


# ---------------------------------------------------------------------------
# Rule 4: a fresh outcome file should beat the exit code; the exit code
# decides only without one. Today `exit_code == 0` short-circuits BEFORE
# the fresh-outcome check ever runs, in both PR-linked branches.
# ---------------------------------------------------------------------------


def _seed_dead_worker_fresh_outcome_and_clean_exit(
    tmp_path: Path, paths, sessions_dir: Path
) -> None:
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude.terminal.json",
        pid=99999,
        exit_code=0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=300.0,
    )
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "pr_created": False, "head_sha": "abc123"},
        mtime=datetime.now(UTC),
    )


def test_flip4_request_changes_exit0_discards_fresh_outcome(tmp_path: Path) -> None:
    """FLIP 4: was a clean exit code short-circuiting before the fresh
    outcome file was ever consulted; now the with-PR lane consults the
    fresh outcome first and only falls back to the exit code when there is
    none (rule 4).

    `orphaned_worker_sweep.handle_dead_worker_with_pr`'s `request_changes`
    branch now calls `handle_dead_worker_completed_outcome` (which consults
    `fresh_completed_worker_outcome`) ahead of the `terminal_exit_code == 0`
    check. In this test environment there is no real git remote to confirm
    the declared push against, so the outcome cannot be *applied* -- but it
    is no longer *ignored* either: the drift reason changes from
    `dead_worker_clean_exit_no_op` (exit code never questioned) to
    `dead_worker_completed_outcome` (fresh outcome seen, push unconfirmed),
    paired with an explicit `rework_outcome_skipped` /
    `remote_head_unavailable` event recording exactly why it could not be
    applied. The issue is left `dispatched` either way -- rule 4 changes
    *what the worker's claim is checked against*, not the fallback status
    when it can't be verified.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _seed_dead_worker_fresh_outcome_and_clean_exit(tmp_path, paths, sessions_dir)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "dispatched"
    events = state.get("events", [])
    # New behaviour: the fresh outcome is consulted (and its head_sha
    # surfaced) instead of the exit code short-circuiting first.
    assert any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_completed_outcome"
        and e["payload"].get("worker_outcome_head_sha") == "abc123"
        for e in events
    )
    assert any(
        e.get("kind") == "rework_outcome_skipped"
        and e["payload"].get("reason") == "remote_head_unavailable"
        for e in events
    )
    assert not any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


def test_flip4_approved_rework_exit0_discards_fresh_outcome(tmp_path: Path) -> None:
    """FLIP 4: was a clean exit code short-circuiting before the fresh
    outcome file was ever consulted; now the with-PR lane consults the
    fresh outcome first (rule 4). Contrast/companion to the test above.

    The `approved` + `rework_requested`-PR-state branch of
    `handle_dead_worker_with_pr` has its own, separately-written call into
    `handle_dead_worker_completed_outcome` ahead of the exit-code check --
    the same fix, duplicated at this second call site rather than shared.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _seed_dead_worker_fresh_outcome_and_clean_exit(tmp_path, paths, sessions_dir)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "dispatched"
    events = state.get("events", [])
    assert any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_completed_outcome"
        and e["payload"].get("worker_outcome_head_sha") == "abc123"
        for e in events
    )
    assert any(
        e.get("kind") == "rework_outcome_skipped"
        and e["payload"].get("reason") == "remote_head_unavailable"
        for e in events
    )
    assert not any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


def test_flip4_nonzero_exit_still_consults_fresh_confirmed_push_outcome(
    tmp_path: Path,
) -> None:
    """B3 (wf-review-opus.md): rule 4 ("a fresh outcome file beats the exit
    code; the exit code decides only without one") is unconditional --
    ``resolve_fate``'s rows 2/3 credit a confirmed push regardless of exit
    code (a worker can push, then crash during teardown). Before this fix,
    ``handle_dead_worker_completed_outcome`` early-returned ``False`` for
    ANY confirmed non-zero exit code without even reading the outcome file,
    so a worker that pushed and then crashed was silently reset to
    ``rework_requested`` and credited a worker death instead of being
    recovered.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude.terminal.json",
        pid=99999,
        exit_code=1,  # a genuine crash, e.g. during teardown after pushing
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=300.0,
    )
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "pr_created": False, "head_sha": "abc123"},
        mtime=datetime.now(UTC),
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "dispatched"
    events = state.get("events", [])
    # The fresh, confirmed-push outcome is consulted (and surfaced) instead
    # of the non-zero exit code short-circuiting before it is ever read.
    assert any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_completed_outcome"
        and e["payload"].get("worker_outcome_head_sha") == "abc123"
        for e in events
    )
    # The generic worker-death reset must NOT also fire -- crediting a
    # death here for confirmed-pushed work is exactly what rule 4 forbids.
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


# ---------------------------------------------------------------------------
# Rule 7: terminal-record precedence and the `terminal or worktree`
# empty-dict fall-through are replaced by rule 1. Two sites read the same
# two sources with opposite fall-through semantics for a present-but-empty
# terminal outcome.
# ---------------------------------------------------------------------------


def test_flip7_workflow_empty_terminal_outcome_falls_through_to_worktree(tmp_path: Path) -> None:
    """FLIP 7 / N1 (wf-review-opus.md), design doc §3 step 0: current
    (fixed) behaviour.

    `workflow.py`'s no-PR orphan lane builds real
    `worker_fate.TerminalEvidence`/`OutcomeEvidence` and arbitrates them
    through `resolve_fate`. A terminal record that IS present but reports
    an empty `worker_outcome` dict (`{}`) builds a content-empty
    `OutcomeEvidence` (all claim fields `None`) -- `worker_fate.
    _carries_no_claim` is what stops that candidate from out-ranking the
    worktree file's real `blocked` declaration even though the terminal
    record's own timestamp is FRESH (postdates `dispatched_at`): without
    it, an empty-but-fresh terminal candidate would win rule 7's
    terminal-over-worktree precedence purely by timestamp, discarding the
    worktree's real content. Both candidates are deliberately fresh here
    (unlike a plain rule-1 staleness rejection) so this exercises the
    `{}`-is-no-claim path specifically, not staleness.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    issue_number = 9107
    branch = "agent/issue-9107-test"
    state_file, sessions_dir = _seed_no_pr_dispatched_issue(tmp_path, config, issue_number, branch)

    # Fresh (post-`dispatched_at`) on purpose -- see docstring. A stale
    # terminal timestamp would make rule 1 reject it regardless of whether
    # `{}`-handling works, proving nothing about this specific defect.
    fresh_ts = (datetime.now(UTC) + timedelta(seconds=5)).isoformat().replace("+00:00", "Z")
    write_worker_terminal_status(
        sessions_dir / f"issue-{issue_number}.claude.terminal.json",
        pid=99999,
        exit_code=1,
        started_at=fresh_ts,
        ended_at=fresh_ts,
        duration_seconds=300.0,
        worker_outcome={},
        worker_outcome_written_at=fresh_ts,
    )

    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps(
            {
                "outcome": "blocked",
                "reason_kind": "cross_repo_scope",
                "detail": "targets a different repo",
            }
        ),
        encoding="utf-8",
    )

    fake_gh = _no_pr_fake_gh(tmp_path, config, issue_number)

    with (
        host_probe(alive=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

    st = load_state(state_file)
    entry = st["issues"][str(issue_number)]
    # Fixed (N1) behaviour: `_carries_no_claim` keeps the fresh-but-empty
    # terminal candidate from out-ranking the worktree file's real (blocked)
    # content, so `resolve_fate` picks the worktree candidate and the sweep
    # escalates on it. Pre-fix, the empty-but-fresh terminal candidate would
    # win rule 7's precedence instead, `resolved_outcome` would carry no
    # `outcome` field, and neither assertion below would hold.
    assert entry["status"] == "escalated"
    assert entry.get("escalation_reason") == "worker_declared_blocked"


def test_flip7_read_rework_outcome_returns_empty_terminal_dict_unlike_workflow(
    tmp_path: Path,
) -> None:
    """FLIP 7 / N1 (wf-review-opus.md), design doc §3 step 0: current
    (fixed) behaviour -- same polarity as the workflow.py site above, not
    the opposite one this test's name still describes (kept to preserve
    the leaf name for the collect-only CI gate; see CLAUDE.md).

    `rework_outcome._read_rework_outcome` reads the SAME two sources
    (durable terminal record, worktree fallback) through the same
    `worker_fate.resolve_fate` freshness step `workflow.py`'s no-PR lane
    uses. This call site passes no `dispatched_at` (legacy mode: every
    candidate's timestamp freshness check passes unconditionally), so
    before the N1 fix a present-but-EMPTY terminal dict (`{}`) still beat
    "legacy mode"'s unconditional freshness and was returned as-is,
    discarding the worktree file's real content -- genuinely the opposite
    polarity from `workflow.py`'s (then-buggy) `or`-based fallthrough.
    `worker_fate._carries_no_claim` now makes an empty terminal candidate
    non-decisive here too, so this site now falls through to the
    worktree's real content exactly like `workflow.py`'s site does --
    the two sites agree instead of disagreeing.
    """
    from charlie_work.rework_outcome import _read_rework_outcome

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    worktrees_dir = tmp_path / "worktrees"
    branch = "agent/issue-9108"
    write_worker_terminal_status(
        sessions_dir / "issue-9108.claude.terminal.json",
        pid=1,
        exit_code=1,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=1.0,
        worker_outcome={},
    )
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "realcontent"}),
        encoding="utf-8",
    )

    result = _read_rework_outcome(sessions_dir, tmp_path, worktrees_dir, 9108, branch)

    assert result == {"push_succeeded": True, "pr_created": False, "head_sha": "realcontent"}


# ---------------------------------------------------------------------------
# Rule 8: `pr_created` should distinguish completed (PR exists) from
# pushed-without-PR consistently. Today `fresh_completed_worker_outcome`
# never inspects it at all.
# ---------------------------------------------------------------------------


def test_flip8_fresh_completed_outcome_ignores_pr_created_true(tmp_path: Path) -> None:
    """FLIP 8: current behaviour; rule 8 will make `pr_created` a distinguisher.

    `fresh_completed_worker_outcome` (rework_outcome.py) checks
    `push_succeeded`, `head_sha`, mtime freshness and the `blocked` sentinel
    -- but never reads `pr_created` at all. An outcome reporting
    `pr_created: True` (structurally unexpected -- workers carry no `gh`
    credential -- but not rejected) is still accepted as valid completed-
    outcome evidence exactly like `pr_created: False`.
    """
    from charlie_work.rework_outcome import fresh_completed_worker_outcome

    worktrees_dir = tmp_path / "worktrees"
    branch = "agent/issue-9109"
    dispatched_at = datetime.now(UTC) - timedelta(hours=1)
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": True, "head_sha": "abc999"}),
        encoding="utf-8",
    )
    fresh_ts = datetime.now(UTC).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))

    result = fresh_completed_worker_outcome(
        worktree_path, issue_number=1, live_head_sha="abc999", dispatched_at=dispatched_at
    )

    assert result is not None
    assert result["pr_created"] is True


# ---------------------------------------------------------------------------
# Rule 9: remote vs local ahead-count split per rule 3. The same-named
# "ahead count" means two different things at two different sites.
# ---------------------------------------------------------------------------


def _repo_with_pushed_and_local_commits(tmp_path: Path, branch: str) -> Path:
    _remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _git(repo_root, "checkout", "-b", branch)
    for i in range(2):
        (repo_root / f"pushed_{i}.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(repo_root, "add", f"pushed_{i}.txt")
        _git(repo_root, "commit", "-m", f"pushed commit {i}")
    _git(repo_root, "push", "-u", "origin", branch)
    (repo_root / "local_only.txt").write_text("local\n", encoding="utf-8")
    _git(repo_root, "add", "local_only.txt")
    _git(repo_root, "commit", "-m", "local-only commit, never pushed")
    return repo_root


def test_flip9_remote_branch_ahead_count_counts_only_pushed_commits(tmp_path: Path) -> None:
    """FLIP 9: current behaviour; rule 9 makes the remote/local split explicit.

    `remote_branch_ahead_count` counts commits on `origin/{branch}` only --
    the local-only, never-pushed commit is invisible to it.
    """
    from charlie_work.worktree import remote_branch_ahead_count

    branch = "agent/issue-9110"
    repo_root = _repo_with_pushed_and_local_commits(tmp_path, branch)

    ahead, error = remote_branch_ahead_count(repo_root, branch, "main")

    assert error is None
    assert ahead == 2


def test_flip9_local_worktree_ahead_count_counts_unpushed_commits_too(tmp_path: Path) -> None:
    """FLIP 9: current behaviour; contrast with the remote-only test above.

    `inspect_worktree_state`'s `ahead_count` is computed purely locally
    (`git merge-base main HEAD` + `git rev-list --count`), so it counts the
    local-only commit the remote-based function above cannot see -- the
    same branch reads as "2 ahead" and "3 ahead" depending on which site is
    asked.
    """
    from charlie_work.worktree import inspect_worktree_state

    branch = "agent/issue-9110"
    repo_root = _repo_with_pushed_and_local_commits(tmp_path, branch)

    inspection = inspect_worktree_state(repo_root, "main")

    assert inspection.ahead_count == 3


# ---------------------------------------------------------------------------
# Rule 6 (preserve): throttle kind is classified fresh at fate time and
# persisted as evidence; persisted dead_worker_failure_kind is read only
# through the fate. No two sites are documented as disagreeing on this
# today, but the exact mechanism -- a direct read of the persisted field --
# is what the new module's single fate accessor must keep behaving like.
# ---------------------------------------------------------------------------


def test_preserve_rule6_throttle_exemption_reads_failure_kind_directly(tmp_path: Path) -> None:
    """Preserve: the #654 timed backstop's throttle exemption reads
    `entry["dead_worker_failure_kind"]` directly (via
    `is_provider_throttle_failure`) and, while `state["throttled_until"]`
    is still in the future, returns immediately with NO side effects --
    it does not touch the rearm counter, `orphan_drift_at`, or emit any
    event. The new fate module must keep this exemption reachable the same
    way even once the field is read only through a fate accessor.
    """
    from _dws_facts import run_reap

    now = datetime.now(UTC)
    orphan_drift_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    entry: dict = {
        "dead_worker_failure_kind": "rate_limited",
        "orphan_drift_at": orphan_drift_at,
    }

    run = run_reap(
        entry,
        issue=9106,
        reap_minutes=60,
        max_rearms=3,
        throttled_until=(now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
        now=now,
    )

    assert run.reaped is False
    # No side effects: the throttled_until-in-future branch returns before
    # ever touching the rearm counter or re-arming orphan_drift_at.
    assert run.commits == ()
    assert "throttle_reap_rearm_count" not in run.entry
    assert run.entry["orphan_drift_at"] == orphan_drift_at
    assert "throttle_reap_rearm_count" not in entry
    assert entry["orphan_drift_at"] == orphan_drift_at


# ---------------------------------------------------------------------------
# A2 (preserve): WorkerView.is_alive collapsed onto the single liveness seam,
# worker_fate.is_alive; the per-adapter wrappers (claude_code.is_worker_alive,
# devin_shell.is_session_alive) were deleted in wf-r2-s2.
# ---------------------------------------------------------------------------


def test_preserve_a2_is_worker_alive_and_is_session_alive_agree(tmp_path: Path) -> None:
    """Preserve: every adapter's ``WorkerView.is_alive`` equals ``worker_fate.is_alive``.

    The two adapter-specific wrappers were byte-identical (``pid is None or
    pid <= 0`` short-circuits to ``False``, otherwise delegate to
    ``process_utils.is_pid_alive``). They are deleted; ``WorkerView.is_alive``
    now asks the one seam, for devin, claude-code and api alike. Pinned on the
    same inputs the wrappers were pinned on, plus an unknown kind (dead).
    """
    from charlie_work import worker_fate
    from charlie_work.worker import WorkerView

    def view(kind: str, pid: int | None) -> WorkerView:
        return WorkerView(
            adapter_kind=kind,
            issue_number=1,
            repo_key="",
            pid=pid,
            started_at="2026-01-01T00:00:00Z",
            process_start_time=None,
            log_path=str(tmp_path / "log.txt"),
            worktree_path=str(tmp_path),
            error=None,
            failure_kind=None,
            reclaimed=None,
        )

    dead_pid = 999_999_937
    for pid in (None, 0, -1, dead_pid):
        expected = worker_fate.is_alive(pid, None)
        for kind in ("devin", "claude-code", "api", "opencode"):
            assert view(kind, pid).is_alive() is expected, (kind, pid)
    # None / non-positive pids are dead by definition, independent of the host.
    assert worker_fate.is_alive(None, None) is False
    assert worker_fate.is_alive(0, None) is False
    assert worker_fate.is_alive(-1, None) is False
    # An unrecognised adapter kind has no fate profile -> conservatively dead.
    assert view("no-such-adapter", os.getpid()).is_alive() is False
    # A live pid reads alive through every adapter's view.
    for kind in ("devin", "claude-code", "api", "opencode"):
        assert view(kind, os.getpid()).is_alive() is True, kind
