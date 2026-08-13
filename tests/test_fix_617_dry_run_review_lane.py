"""Tests for issue #617: dry-run review lane escalates and resets rework budgets.

Two sites in the review lane mutated state.json (or fired real notifications)
under ``--dry-run`` because their dry-run gates sat strictly *after* the
destructive writes:

A. ``dispatch_reviews`` escalated PRs at the attempt cap and wrote a
   quota-alert marker *before* its dry-run early-return. The fix moves the
   gate above the rescue partition and the escalation block, mirroring
   ``_dispatch_impl``'s top-of-function short-circuit.

B. ``review()`` had no top-level ``self.dry_run`` gate at all -- the flag
   only reached the nested ``_cross_family_for_pr`` call. A preview was
   byte-identical to a real call: it wrote packet files, overwrote a prior
   terminal decision with ``"pending"``, and reset
   ``review_dispatch_attempt_count`` / ``no_op_rework_attempts`` to zero.
   The fix adds a top-of-function short-circuit that returns the plan and
   touches nothing.

These tests reuse ``FakeGitHub`` from ``test_charlie_work.py``
(PR #456 <-> issue #123).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import (
    NotifyConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_review_packet(paths, pr_number: int, head_sha: str) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        f'{{"number": {pr_number}, "headRefOid": "{head_sha}"}}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


def _seed_reviewing_issue(paths, pr_number: int, issue_number: int) -> None:
    """Seed state so the PR looks like it's in ``agent:reviewing`` with a
    current packet but no recorded verdict."""
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "reviewing",
        }
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
        }
        save_state(paths.state_file, state)


# ---------------------------------------------------------------------------
# Fix A: dispatch_reviews must not escalate or write quota-alert under dry-run
# ---------------------------------------------------------------------------


def test_dispatch_reviews_dry_run_does_not_escalate_at_attempt_cap(
    tmp_path: Path,
) -> None:
    """A PR at the review-dispatch attempt cap must NOT be escalated in
    dry-run mode. Pre-fix, the escalation block ran before the dry-run gate
    and flipped both the PR and issue status to ``"escalated"``."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    # Seed the PR at the attempt cap (3 == max_review_dispatch_attempts).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "reviewing"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "review_dispatch_attempt_count": 3,
            "review_dispatch_status": "review_dispatch_failed",
        }
        save_state(paths.state_file, state)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    # The PR must NOT have been escalated.
    assert state["prs"]["456"].get("status") != "escalated"
    assert state["issues"]["123"]["status"] == "reviewing"
    # The attempt counter must be unchanged.
    assert state["prs"]["456"]["review_dispatch_attempt_count"] == 3
    # No human-needed label applied.
    assert (123, config.labels.human_needed) not in fake_gh.labels_added


def test_dispatch_reviews_dry_run_does_not_write_quota_alert_marker(
    tmp_path: Path,
) -> None:
    """When the reviewer quota is exhausted and a probe is not ready,
    dry-run must report the deferral WITHOUT writing the one-shot
    ``alerted_at`` quota-alert marker or firing a real ``emit_digest``.
    Pre-fix, the quota-alert marker write sat before the dry-run gate."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        notify=NotifyConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_reviewing_issue(paths, 456, 123)
    # Seed an exhausted quota with a far-future probe_after.
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    throttled_until = (datetime.now(UTC) + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["reviewer_quota"] = {
            "throttled_until": throttled_until,
            "probe_after": future,
        }
        save_state(paths.state_file, state)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data.get("deferred_reason") == "reviewer_quota_probe_backoff"
    state = load_state(paths.state_file)
    # The alerted_at marker must NOT have been written.
    assert not (state.get("reviewer_quota") or {}).get("alerted_at")


def test_dispatch_reviews_dry_run_excludes_rescue_marked_pr(
    tmp_path: Path,
) -> None:
    """A rescue-marked PR must NOT appear in the dry-run preview as a normal
    dispatch/escalation candidate. Pre-fix, the dry-run branch did not
    exclude rescue-marked PRs before computing the selection, so a
    rescue-marked PR could be reported as a normal dispatch candidate —
    a misreport, since the real branch routes it through
    ``_process_rescue_review`` instead. The fix excludes rescue-marked PRs
    via a read-only state check (not ``_partition_rescue_candidates``,
    which has real side effects)."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    # Seed the PR as rescue-marked (the durable marker that _partition_rescue_candidates
    # and _process_rescue_review route on).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "reviewing"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "rescue_attempted": True,
            "rescue_cause": "request_changes",
        }
        save_state(paths.state_file, state)

    state_before = load_state(paths.state_file)
    result = app.dispatch_reviews()
    state_after = load_state(paths.state_file)

    assert result.ok is True
    # The rescue-marked PR must be reported as excluded, not as a dispatch
    # or escalation candidate.
    assert 456 in result.data["rescue_marked_excluded"]
    assert 456 not in result.data.get("escalated_skipped", [])
    assert 456 not in result.data.get("merge_conflict_routed", [])
    assert result.data["selected_count"] == 0
    # No state mutation: the dry-run must not have run a real rescue review.
    assert state_after == state_before


def test_dispatch_reviews_dry_run_reports_merge_conflict_routed(
    tmp_path: Path,
) -> None:
    """Dry-run must still report CONFLICTING PRs in ``merge_conflict_routed``
    (backward compat with issue #1497's dry-run test), but must NOT route
    them to rework."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_reviewing_issue(paths, 456, 123)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert 456 in result.data["merge_conflict_routed"]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "reviewing"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


# ---------------------------------------------------------------------------
# Fix B: review() must not persist packet files or reset counters under dry-run
# ---------------------------------------------------------------------------


def test_review_dry_run_does_not_write_packet_files(tmp_path: Path) -> None:
    """In dry-run mode, review() must not create or overwrite any packet
    files (pr.json, checks.json, diff.patch, review-prompt.md,
    review-decision.json). Pre-fix, all of these were written
    byte-identically to a real call."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    result = app.review(456)

    assert result.ok is True
    assert result.data.get("dry_run") is True
    pr_dir = paths.prs / "pr-456"
    # No packet files should exist.
    assert not (pr_dir / "pr.json").exists()
    assert not (pr_dir / "checks.json").exists()
    assert not (pr_dir / "diff.patch").exists()
    assert not (pr_dir / "review-prompt.md").exists()
    assert not (pr_dir / "review-decision.json").exists()


def test_review_dry_run_does_not_reset_rework_counters(tmp_path: Path) -> None:
    """In dry-run mode, review() must not reset
    ``review_dispatch_attempt_count`` or ``no_op_rework_attempts`` to zero,
    and must not flip the PR status to ``"reviewing"``. Pre-fix, the counter
    resets were the destructive part and were not recoverable by re-running."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    # Seed state with non-zero counters and a non-reviewing status.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "review_dispatch_attempt_count": 2,
            "no_op_rework_attempts": 1,
        }
        save_state(paths.state_file, state)

    result = app.review(456)

    assert result.ok is True
    state = load_state(paths.state_file)
    # Counters must be unchanged.
    assert state["prs"]["456"]["review_dispatch_attempt_count"] == 2
    assert state["prs"]["456"]["no_op_rework_attempts"] == 1
    # Status must not have been flipped to "reviewing".
    assert state["prs"]["456"]["status"] == "janitor_blocked"


def test_review_dry_run_does_not_overwrite_terminal_verdict(tmp_path: Path) -> None:
    """In dry-run mode, review() must not overwrite a prior terminal decision
    (e.g. ``"approved"``) with ``"pending"``. Pre-fix, when the head had
    advanced, review-decision.json was overwritten with the pending template,
    destroying the recorded verdict."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Advance the head so the non-dry-run path would reset the decision.
    fake_gh.pr_head_shas[456] = "sha-new-head"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    # Seed a terminal verdict on an old head.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    import json

    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "pr_number": 456,
                "issue_number": 123,
                "decision": "approved",
                "summary": "LGTM",
                "required_changes": [],
                "reviewed_head_sha": "sha-abc123",
                "reviewed_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    result = app.review(456)

    assert result.ok is True
    # The terminal verdict must still be on disk, untouched.
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "approved"
    assert decision["reviewed_head_sha"] == "sha-abc123"


def test_review_dry_run_does_not_mutate_state_at_all(tmp_path: Path) -> None:
    """In dry-run mode, review() must not write state.json at all -- no
    events, no status changes, no counter resets. This is the broad
    invariant: touch nothing."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "reviewing",
            "review_dispatch_attempt_count": 3,
        }
        save_state(paths.state_file, state)

    state_before = load_state(paths.state_file)
    result = app.review(456)
    state_after = load_state(paths.state_file)

    assert result.ok is True
    # State must be byte-identical (no events appended, no fields changed).
    assert state_after == state_before


def test_review_real_run_still_writes_packet(tmp_path: Path) -> None:
    """The other direction: a real (non-dry-run) review() must still write
    the packet and reset counters. Without this, the dry-run gate could
    silently disable review() entirely and the suite would still be green."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    result = app.review(456)

    assert result.ok is True
    pr_dir = paths.prs / "pr-456"
    assert (pr_dir / "pr.json").exists()
    assert (pr_dir / "review-prompt.md").exists()
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "reviewing"
    assert state["prs"]["456"]["review_dispatch_attempt_count"] == 0
