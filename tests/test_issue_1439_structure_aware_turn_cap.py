"""Issue #1439: structure-aware reviewer turn cap.

The flat ``review_max_turns`` budget ignores the size of the files a diff
touches, so a PR threading a monolith (e.g. workflow.py at ~25k lines) burns
the whole turn budget on grep -> Read-window navigation without ever reaching
a verdict, then retries the identical flat budget on the next dispatch.

These tests cover the three acceptance criteria:
1. A packet for a diff touching a >5k-line file gets the raised cap; a
   small-file diff gets the base cap.
2. The second dispatch after a turn-limit miss carries a higher cap than the
   first (the miss streak escalates the cap one step).
3. The Nth consecutive turn-limit miss escalates to ``agent:human-needed``
   instead of redispatching.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _write_review_packet,
)
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import (
    OrchestratorApp,
    _max_touched_file_line_count,
    resolve_review_turn_cap,
    structure_turn_cap_multiplier,
)

_PR = 456
_ISSUE = 123


def _review_app(tmp_path: Path, *, diff: str | None = None) -> OrchestratorApp:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    if diff is not None:
        fake_gh.diffs[_PR] = diff
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app


def _write_lines(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"line {i}" for i in range(n)) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_structure_multiplier_large_file_vs_small_file() -> None:
    """A touched file exceeding the threshold yields the raised multiplier;
    a small file yields 1 (the base cap)."""
    config = ReviewDispatchConfig()
    assert structure_turn_cap_multiplier(6000, config) == 2
    # "exceeds" the threshold is strict (>), so threshold+1 triggers.
    assert structure_turn_cap_multiplier(config.turn_cap_large_file_threshold + 1, config) == 2
    assert structure_turn_cap_multiplier(config.turn_cap_large_file_threshold, config) == 1
    assert structure_turn_cap_multiplier(100, config) == 1
    assert structure_turn_cap_multiplier(0, config) == 1


def test_structure_multiplier_clamped_to_max() -> None:
    """The structure multiplier never exceeds ``turn_cap_max_multiplier``."""
    config = ReviewDispatchConfig(
        turn_cap_large_file_multiplier=5,
        turn_cap_max_multiplier=3,
    )
    assert structure_turn_cap_multiplier(99999, config) == 3


def test_structure_multiplier_threshold_zero_disables() -> None:
    config = ReviewDispatchConfig(turn_cap_large_file_threshold=0)
    assert structure_turn_cap_multiplier(99999, config) == 1


def test_resolve_review_turn_cap_structure_and_miss_escalation() -> None:
    """``effective_multiplier = min(structure + streak, max_multiplier)``."""
    config = ReviewDispatchConfig()  # base 40, large-file x2, max x3
    base = config.review_max_turns
    # Large file, no misses -> x2.
    assert resolve_review_turn_cap(base, 2, 0, config) == base * 2
    # Large file, one miss -> x3.
    assert resolve_review_turn_cap(base, 2, 1, config) == base * 3
    # Large file, two misses -> clamped at x3.
    assert resolve_review_turn_cap(base, 2, 2, config) == base * 3
    # Small file, no misses -> x1 (base cap).
    assert resolve_review_turn_cap(base, 1, 0, config) == base
    # Small file, one miss -> x2 (one escalation step).
    assert resolve_review_turn_cap(base, 1, 1, config) == base * 2
    # Small file, many misses -> clamped at x3.
    assert resolve_review_turn_cap(base, 1, 5, config) == base * 3
    # Base 0 (unlimited) stays 0.
    assert resolve_review_turn_cap(0, 2, 3, config) == 0


def test_max_touched_file_line_count_reads_repo_root(tmp_path: Path) -> None:
    """File sizes are read from ``repo_root``; a new file falls back to its
    added-line count."""
    _write_lines(tmp_path / "big.py", 6000)
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    big_diff = (
        "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n"
        "@@ -1,1 +1,2 @@\n line 0\n+new\n"
    )
    assert _max_touched_file_line_count(big_diff, tmp_path) == 6000
    small_diff = (
        "diff --git a/small.py b/small.py\n--- a/small.py\n+++ b/small.py\n"
        "@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n"
    )
    assert _max_touched_file_line_count(small_diff, tmp_path) == 1
    # New file (not present at repo_root): size is the added-line count.
    new_diff = (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n"
        "@@ -0,0 +1,42 @@\n" + "".join("+line\n" for _ in range(42))
    )
    assert _max_touched_file_line_count(new_diff, tmp_path) == 42


# --------------------------------------------------------------------------
# Acceptance criterion 1: packet build stamps the structure-aware multiplier
# --------------------------------------------------------------------------


def test_packet_large_file_gets_raised_multiplier(tmp_path: Path) -> None:
    """A packet for a diff touching a >5k-line file stamps multiplier 2."""
    _write_lines(tmp_path / "big.py", 6000)
    diff = (
        "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n"
        "@@ -1,1 +1,2 @@\n line 0\n+new\n"
    )
    app = _review_app(tmp_path, diff=diff)
    app.review(_PR)
    pr_json = json.loads((app.paths.prs / f"pr-{_PR}" / "pr.json").read_text(encoding="utf-8"))
    assert pr_json["review_turn_cap_structure_multiplier"] == 2
    assert pr_json["review_turn_cap_max_touched_lines"] == 6000


def test_packet_small_file_gets_base_multiplier(tmp_path: Path) -> None:
    """A packet for a small-file diff stamps multiplier 1 (base cap)."""
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    diff = (
        "diff --git a/small.py b/small.py\n--- a/small.py\n+++ b/small.py\n"
        "@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n"
    )
    app = _review_app(tmp_path, diff=diff)
    app.review(_PR)
    pr_json = json.loads((app.paths.prs / f"pr-{_PR}" / "pr.json").read_text(encoding="utf-8"))
    assert pr_json["review_turn_cap_structure_multiplier"] == 1


# --------------------------------------------------------------------------
# Acceptance criterion 2: second dispatch after a turn-limit miss carries a
# higher cap than the first.
# --------------------------------------------------------------------------


def _seed_dispatchable_claim(app: OrchestratorApp, *, streak: int) -> None:
    """Seed state so ``dispatch_reviews`` sees PR #456 as a fresh dispatchable
    candidate with the given turn-limit miss streak and a completed (non-live)
    prior claim."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "issue_number": _ISSUE,
            "review_dispatch_status": "review_dispatch_completed",
            "review_dispatch_attempt_count": 0,
            "review_turn_limit_miss_streak": streak,
        }
        save_state(app.paths.state_file, state)


def test_dispatch_cap_escalates_after_turn_limit_miss(monkeypatch, tmp_path: Path) -> None:
    """The second dispatch after a turn-limit miss carries a higher cap than
    the first (small-file packet, base cap 40 -> 80 after one miss)."""
    prs = [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    # Build a small-file packet (multiplier 1) so the base cap is the floor.
    _write_review_packet(tmp_path, _PR, "sha-456")

    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    base = app.config.review_dispatch.review_max_turns

    # First dispatch: streak 0 -> base cap.
    _seed_dispatchable_claim(app, streak=0)
    app.dispatch_reviews()
    assert len(captured) == 1
    assert captured[0]["max_turns_override"] == base

    # Reset the completed claim so the PR is dispatchable again.
    captured.clear()
    _seed_dispatchable_claim(app, streak=1)
    app.dispatch_reviews()
    assert len(captured) == 1
    # Second dispatch after one turn-limit miss: one escalation step (x2).
    assert captured[0]["max_turns_override"] == base * 2
    assert captured[0]["max_turns_override"] > base


def test_dispatch_cap_escalates_with_large_file_packet(monkeypatch, tmp_path: Path) -> None:
    """A large-file packet (structure multiplier 2) starts at x2 and escalates
    to x3 after one turn-limit miss."""
    prs = [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    # Build a packet, then overwrite pr.json with a large-file multiplier.
    _write_review_packet(tmp_path, _PR, "sha-456")
    pr_json_path = app.paths.prs / f"pr-{_PR}" / "pr.json"
    pr_json = json.loads(pr_json_path.read_text(encoding="utf-8"))
    pr_json["review_turn_cap_structure_multiplier"] = 2
    pr_json_path.write_text(json.dumps(pr_json), encoding="utf-8")

    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    base = app.config.review_dispatch.review_max_turns

    _seed_dispatchable_claim(app, streak=0)
    app.dispatch_reviews()
    assert captured[0]["max_turns_override"] == base * 2

    captured.clear()
    _seed_dispatchable_claim(app, streak=1)
    app.dispatch_reviews()
    assert captured[0]["max_turns_override"] == base * 3


# --------------------------------------------------------------------------
# Acceptance criterion 3: Nth consecutive miss escalates instead of
# redispatching.
# --------------------------------------------------------------------------


def test_nth_turn_limit_miss_escalates(monkeypatch, tmp_path: Path) -> None:
    """After ``max_consecutive_turn_limit_misses`` (default 3) consecutive
    turn-limit misses, the PR escalates to agent:human-needed instead of being
    redispatched."""
    prs = [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, _PR, "sha-456")

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        raise AssertionError("a PR at the turn-limit miss backstop must NOT be redispatched")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    cap = app.config.review_dispatch.max_consecutive_turn_limit_misses
    # Seed the PR at the backstop: streak == cap, attempt_count below the
    # attempt cap so the turn-limit backstop is the escalation driver.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "issue_number": _ISSUE,
            "review_dispatch_status": "review_dispatch_completed",
            "review_dispatch_attempt_count": 0,
            "review_turn_limit_miss_streak": cap,
        }
        save_state(app.paths.state_file, state)

    result = app.dispatch_reviews()

    # Not launched.
    assert result.data["launched_count"] == 0
    # Escalated with the turn-limit-miss reason.
    escalated = query_events(app.paths.state_file, kind="review_dispatch_escalated")
    assert any(
        e["payload"].get("reason") == "max_consecutive_turn_limit_misses_exceeded"
        and e["payload"].get("pr_number") == _PR
        for e in escalated
    )
    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)].get("status") == "escalated"


def test_below_backstop_does_not_escalate(monkeypatch, tmp_path: Path) -> None:
    """A streak below the backstop cap does NOT escalate -- the PR is still
    dispatchable (with a raised cap)."""
    prs = [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, _PR, "sha-456")

    captured: list[dict[str, Any]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        captured.append(kwargs)
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    cap = app.config.review_dispatch.max_consecutive_turn_limit_misses
    _seed_dispatchable_claim(app, streak=cap - 1)
    result = app.dispatch_reviews()
    assert result.data["launched_count"] == 1
    assert len(captured) == 1
    escalated = query_events(app.paths.state_file, kind="review_dispatch_escalated")
    assert escalated == []


# --------------------------------------------------------------------------
# Turn-limit miss increments the streak (the wiring that drives criterion 2).
# --------------------------------------------------------------------------


def _seed_dead_reviewer(app: OrchestratorApp, *, turns: int) -> None:
    """Seed a dead reviewer sidecar + events.jsonl padded to ``turns`` so the
    reap path classifies the death as a turn-limit miss."""
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / f"issue-{_PR}-review.claude.log"
    log_path.write_text("I reviewed the diff but need more turns to verify.\n", encoding="utf-8")
    events_path = reviews_dir / f"issue-{_PR}-review.events.jsonl"
    lines: list[str] = []
    for index in range(turns):
        content = [{"type": "text", "text": f"Analysis step {index + 1}."}]
        if index < turns - 1:
            content.append({"type": "tool_use", "id": f"t{index}", "name": "Read", "input": {}})
        lines.append(json.dumps({"type": "assistant", "message": {"content": content}}))
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecar = {
        "issue_number": _PR,
        "branch": "agent/issue-123-fix",
        "worktree_path": str(reviews_dir / f"issue-{_PR}"),
        "prompt_path": str(reviews_dir / f"issue-{_PR}-review-prompt.md"),
        "command": ["claude", "-p", "--permission-mode", "plan"],
        "pid": 0,
        "started_at": (datetime.now(UTC) - timedelta(minutes=10))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / f"issue-{_PR}.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")


def test_turn_limit_miss_increments_streak(tmp_path: Path) -> None:
    """A turn-limit death increments ``review_turn_limit_miss_streak`` and
    mirrors the post-increment streak into the review_verdict_missed event."""
    prs = [
        {
            "number": _PR,
            "title": "Fix #123",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix",
            "baseRefName": "main",
            "headRefOid": "sha-456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, _PR, "sha-456")

    max_turns = app.config.review_dispatch.review_max_turns
    # Seed a prior streak of 1 so the post-increment value is unambiguous.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "issue_number": _ISSUE,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
            "review_turn_limit_miss_streak": 1,
        }
        save_state(app.paths.state_file, state)

    _seed_dead_reviewer(app, turns=max_turns)

    # Suppress the PR comment (network) -- the reap path tolerates failure.
    app.gh = FakeGitHub()

    app.dispatch_reviews()

    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)]["review_turn_limit_miss_streak"] == 2
    missed = query_events(app.paths.state_file, kind="review_verdict_missed")
    assert any(
        e["payload"].get("reason") == "turn_limit_summary_posted"
        and e["payload"].get("turn_limit_miss_streak") == 2
        for e in missed
    )
