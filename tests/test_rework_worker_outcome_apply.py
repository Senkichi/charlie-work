"""Issue #1853: the orchestrator applies a rework worker's
``.worker-outcome.json`` to the PR after verifying the remote head.

Workers carry no ``gh`` credential by design, so a rework session cannot run
``gh pr view``/``gh pr edit``/``gh pr comment`` itself. Instead the worker
reports the pushed head and drafts the PR updates in the outcome file; the
authenticated orchestrator verifies the remote head matches what the worker
reported and then applies the body edit and optional comment itself.

Contract under test:

- ``pr_body`` leads to exactly one PR body update (``gh.pr_edit``).
- ``pr_comment`` leads to exactly one comment (``gh.pr_comment``).
- A reported head that differs from the remote head produces no edit and an
  event.
- A missing or invalid outcome file produces no edit, no error, and an event.
- An outcome already applied at the same head is not re-applied (a blocked
  review route retries the pass without double-posting).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _dispatch_rework_config, _wg
from charlie_work import rework_outcome
from charlie_work.config import WORKER_OUTCOME_FILENAME
from charlie_work.github import GitHubError
from charlie_work.paths import runtime_paths
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.rework_outcome import (
    apply_rework_worker_outcome,
    fresh_completed_worker_outcome,
)
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch


BRANCH = "agent/issue-123-fix-search"
ISSUE_NUMBER = 123
PR_NUMBER = 456
HEAD_SHA = "sha-rework-head"


def _write_worktree_outcome(
    repo_root: Path, worktrees_dir: Path, branch: str, payload: dict[str, Any]
) -> Path:
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return worktree_path


def _write_terminal_outcome(
    sessions_dir: Path, issue_number: int, outcome: dict[str, Any]
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_worker_terminal_status(
        sessions_dir / f"issue-{issue_number}.test.terminal.json",
        pid=12345,
        exit_code=0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome=outcome,
    )


def _outcome(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "push_succeeded": True,
        "pr_created": False,
        "head_sha": HEAD_SHA,
        "pr_body": f"Closes #{ISSUE_NUMBER}\n\nReworked per review; suite green.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def remote_head(monkeypatch: pytest.MonkeyPatch):
    """Point the remote-head probe at a controlled SHA."""

    def _set(sha: str | None) -> None:
        monkeypatch.setattr(rework_outcome, "remote_branch_head_sha", lambda *_a: sha)

    _set(HEAD_SHA)
    return _set


def _apply(
    gh: FakeGitHub,
    tmp_path: Path,
    *,
    worktrees_dir: Path,
    sessions_dir: Path,
    issue_number: int = ISSUE_NUMBER,
    pr_number: int = PR_NUMBER,
) -> None:
    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    apply_rework_worker_outcome(
        gh,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        sessions_dir=sessions_dir,
        state_file=paths.state_file,
        write_gate=_wg(paths.state_file),
        issue_number=issue_number,
        pr_number=pr_number,
    )


def _state(tmp_path: Path) -> dict[str, Any]:
    config = _dispatch_rework_config()
    return load_state(runtime_paths(tmp_path, config.runtime.state_dir).state_file)


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Return the payloads of every ``kind`` event (events store fields under
    ``payload``)."""
    return [e.get("payload", {}) for e in state.get("events", []) if e.get("kind") == kind]


def _seed_issue_entry(tmp_path: Path) -> None:
    config = _dispatch_rework_config()
    state_file = runtime_paths(tmp_path, config.runtime.state_dir).state_file
    state = load_state(state_file)
    state["issues"][str(ISSUE_NUMBER)] = {
        "status": "rework_requested",
        "branch_name": BRANCH,
    }
    save_state(state_file, state)


# ---------------------------------------------------------------------------
# Apply path
# ---------------------------------------------------------------------------


def test_pr_body_produces_exactly_one_pr_edit(tmp_path: Path, remote_head) -> None:
    """Acceptance: a ``pr_body`` outcome leads to exactly one PR body update."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert len(gh.pr_edits) == 1
    assert gh.pr_edits[0][0] == PR_NUMBER
    assert "Closes #123" in gh.pr_edits[0][1]
    assert gh.pr_comments_posted == []
    state = _state(tmp_path)
    applied = _events(state, "rework_outcome_applied")
    assert len(applied) == 1
    assert applied[0]["issue_number"] == ISSUE_NUMBER
    assert applied[0]["pr_number"] == PR_NUMBER
    assert applied[0]["body_updated"] is True


def test_pr_comment_produces_exactly_one_comment(tmp_path: Path, remote_head) -> None:
    """Acceptance: a ``pr_comment`` outcome leads to exactly one comment."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(
        tmp_path,
        worktrees_dir,
        BRANCH,
        _outcome(pr_body=None, pr_comment="Disagree with finding 2: see commit abc."),
    )

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert len(gh.pr_comments_posted) == 1
    assert gh.pr_comments_posted[0][0] == PR_NUMBER
    assert "Disagree with finding 2" in gh.pr_comments_posted[0][1]
    assert gh.pr_edits == []
    state = _state(tmp_path)
    applied = _events(state, "rework_outcome_applied")
    assert len(applied) == 1
    assert applied[0]["comment_posted"] is True


def test_outcome_read_from_terminal_status_when_worktree_gone(tmp_path: Path, remote_head) -> None:
    """The durable terminal-status copy is authoritative: the worktree may be
    removed before the orchestrator applies the outcome."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_terminal_outcome(sessions_dir, ISSUE_NUMBER, _outcome())

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert len(gh.pr_edits) == 1


# ---------------------------------------------------------------------------
# Head verification
# ---------------------------------------------------------------------------


def test_head_mismatch_produces_no_edit_and_an_event(tmp_path: Path, remote_head) -> None:
    """Acceptance: reported head != remote head -> no edit, recorded event."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())
    remote_head("sha-someone-else-pushed")

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    skipped = _events(_state(tmp_path), "rework_outcome_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "head_mismatch"


# ---------------------------------------------------------------------------
# Missing / invalid outcome
# ---------------------------------------------------------------------------


def test_missing_outcome_produces_no_edit_and_an_event(tmp_path: Path, remote_head) -> None:
    """Acceptance: no outcome file -> no edit, no error, recorded event."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    skipped = _events(_state(tmp_path), "rework_outcome_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "no_outcome"


def test_outcome_without_head_sha_is_skipped(tmp_path: Path, remote_head) -> None:
    """An outcome that cannot report its pushed head is not verifiable —
    the orchestrator has nothing to check against, so it must not edit."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    outcome = _outcome()
    del outcome["head_sha"]
    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, outcome)

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    skipped = _events(_state(tmp_path), "rework_outcome_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "missing_head_sha"


def test_blocked_outcome_is_not_applied(tmp_path: Path, remote_head) -> None:
    """The blocked-outcome shape is a different channel — never a PR edit."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(
        tmp_path,
        worktrees_dir,
        BRANCH,
        {"outcome": "blocked", "reason_kind": "ambiguous_scope", "detail": "needs a human"},
    )

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert gh.pr_edits == []
    assert gh.pr_comments_posted == []
    skipped = _events(_state(tmp_path), "rework_outcome_skipped")
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "blocked_outcome"


# ---------------------------------------------------------------------------
# Dedup: apply once per reported head
# ---------------------------------------------------------------------------


def test_same_head_is_not_applied_twice(tmp_path: Path, remote_head) -> None:
    """A retried routing pass must not re-edit the body or double-post the
    comment — the applied head is recorded in state."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(
        tmp_path, worktrees_dir, BRANCH, _outcome(pr_comment="verification notes")
    )

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)
    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert len(gh.pr_edits) == 1
    assert len(gh.pr_comments_posted) == 1
    state = _state(tmp_path)
    assert len(_events(state, "rework_outcome_applied")) == 1
    assert state.get("rework_outcome_applied_heads", {}).get(str(ISSUE_NUMBER)) == HEAD_SHA


def test_new_head_reapplies(tmp_path: Path, remote_head) -> None:
    """A later rework round (new reported head) applies again — the marker
    keys off the head, not the issue."""
    gh = FakeGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome(head_sha="sha-round-2"))
    remote_head("sha-round-2")
    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    assert len(gh.pr_edits) == 2


def test_apply_failure_is_not_marked_applied(tmp_path: Path, remote_head) -> None:
    """A gh failure records an event but no applied marker, so the next pass
    can retry."""

    class FailingEditGitHub(FakeGitHub):
        def pr_edit(self, number: int, body_file: Path) -> None:
            raise GitHubError("gh: boom")

    gh = FailingEditGitHub(repo_root=tmp_path)
    worktrees_dir = tmp_path / "worktrees"
    sessions_dir = tmp_path / "sessions"
    _seed_issue_entry(tmp_path)
    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())

    _apply(gh, tmp_path, worktrees_dir=worktrees_dir, sessions_dir=sessions_dir)

    state = _state(tmp_path)
    assert _events(state, "rework_outcome_applied") == []
    failed = _events(state, "rework_outcome_apply_failed")
    assert len(failed) == 1
    assert "rework_outcome_applied_heads" not in state


# ---------------------------------------------------------------------------
# Wiring: dispatch_rework's route-to-review seam applies the outcome
# ---------------------------------------------------------------------------


def test_route_rework_candidate_applies_outcome_before_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_route_rework_candidate_to_review`` is the seam where a pushed
    rework head is routed back to review — the outcome must be applied there
    so a rerouted-but-blocked review still updates the PR."""
    from charlie_work.workflow import CommandResult, OrchestratorApp

    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub(repo_root=tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    monkeypatch.setattr(rework_outcome, "remote_branch_head_sha", lambda *_a: HEAD_SHA)

    state = load_state(paths.state_file)
    state["issues"][str(ISSUE_NUMBER)] = {
        "status": "rework_requested",
        "branch_name": BRANCH,
    }
    save_state(paths.state_file, state)

    _write_worktree_outcome(
        tmp_path,
        app._layout.worktrees,
        BRANCH,
        _outcome(pr_comment="notes"),
    )

    # review() is stubbed to a janitor-blocked result (ok=False, nothing
    # written): the apply must still have run.
    monkeypatch.setattr(
        app,
        "review",
        lambda pr_number: CommandResult(False, "blocked", {"pr": pr_number}),
    )

    routed, _ = app._route_rework_candidate_to_review(ISSUE_NUMBER, PR_NUMBER, "sha-reviewed")

    assert routed is False
    assert len(gh.pr_edits) == 1
    assert len(gh.pr_comments_posted) == 1


# ---------------------------------------------------------------------------
# Prompt contract: no gh commands in the rework brief
# ---------------------------------------------------------------------------


def test_rendered_rework_prompt_has_no_gh_command(tmp_path: Path) -> None:
    """Acceptance: the rendered rework brief's FINAL STEP contains no ``gh ``
    command — it directs the worker to git verification and
    ``.worker-outcome.json`` only.

    Shared sections elsewhere in the prompt legitimately *prohibit* gh
    subcommands (the no-merge contract, the scratch-dir cautionary example),
    so the no-``gh`` assertion is scoped to the operational tail the worker
    is told to execute; the template source itself must be gh-free too.
    """
    from charlie_work.workflow import OrchestratorApp

    config = _dispatch_rework_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)
    pr = {
        "number": 8,
        "title": "fix x",
        "url": "https://example.test/pull/8",
        "headRefName": "agent/issue-7-x",
    }
    rendered = app._write_rework_prompt(pr, 7, "note").read_text(encoding="utf-8")

    import re

    final_step = rendered.split("## FINAL STEP", 1)[-1]
    # Word-boundary "gh" followed by whitespace = a real gh invocation;
    # plain "gh " would false-positive inside "through ".
    assert not re.search(r"\bgh\s", final_step)
    assert "gh pr" not in final_step
    assert ".worker-outcome.json" in final_step
    assert "pr_body" in final_step
    assert "pr_comment" in final_step
    assert "head_sha" in final_step

    # The required-behavior bullets must also route review-result updates
    # through the outcome file, not "update the PR body or comment".
    bullets = rendered.split("## Required behavior", 1)[-1].split("$section", 1)[0]
    assert ".worker-outcome.json" in bullets
    assert "update the PR body" not in bullets


def test_rework_md_template_source_has_no_gh_command() -> None:
    """The rework template itself must not reference any ``gh`` command —
    the orchestrator (not the worker) owns every PR mutation."""
    template = Path(__file__).parent.parent / "src" / "charlie_work" / "prompts" / "rework.md"
    import re

    text = template.read_text(encoding="utf-8")
    assert not re.search(r"\bgh\s", text), "rework.md must not reference a gh command"
    assert "gh pr" not in text


# ---------------------------------------------------------------------------
# fresh_completed_worker_outcome (issue #1911): the orphan sweep's "did this
# dead worker actually finish?" probe for sessions with no terminal record
# ---------------------------------------------------------------------------


def _fresh_outcome_bed(tmp_path: Path) -> tuple[Path, datetime]:
    worktrees_dir = tmp_path / "worktrees"
    dispatched_at = datetime.now(UTC) - timedelta(hours=1)
    return worktrees_dir, dispatched_at


def _set_mtime(path: Path, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def test_fresh_completed_outcome_is_returned(tmp_path: Path) -> None:
    """A well-formed outcome written after dispatch started, pinned to the
    live head, is proof of a completed handoff."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())

    result = fresh_completed_worker_outcome(
        worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
    )

    assert result is not None
    assert result["head_sha"] == HEAD_SHA
    assert result["push_succeeded"] is True


def test_stale_outcome_is_not_completed(tmp_path: Path) -> None:
    """An outcome older than the dispatch belongs to a previous session."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())
    _set_mtime(worktree_path / WORKER_OUTCOME_FILENAME, dispatched_at - timedelta(minutes=5))

    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
        )
        is None
    )


def test_outcome_at_other_head_is_not_completed(tmp_path: Path) -> None:
    """A head_sha that does not match the live head describes a different
    remote state."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = _write_worktree_outcome(
        tmp_path, worktrees_dir, BRANCH, _outcome(head_sha="sha-other")
    )

    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
        )
        is None
    )


def test_unpushed_outcome_is_not_completed(tmp_path: Path) -> None:
    """``push_succeeded`` false/missing means the handoff never happened."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = _write_worktree_outcome(
        tmp_path, worktrees_dir, BRANCH, _outcome(push_succeeded=False)
    )

    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
        )
        is None
    )


def test_blocked_outcome_is_not_completed(tmp_path: Path) -> None:
    """The ``blocked`` declaration shape is a scope-escalation channel, never
    proof of a completed handoff — even if malformed enough to also carry
    push fields."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = _write_worktree_outcome(
        tmp_path,
        worktrees_dir,
        BRANCH,
        {
            "outcome": "blocked",
            "reason_kind": "ambiguous_scope",
            "push_succeeded": True,
            "head_sha": HEAD_SHA,
        },
    )

    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
        )
        is None
    )


def test_missing_outcome_or_missing_anchor_is_not_completed(tmp_path: Path) -> None:
    """No file, no worktree, no live head, or no dispatch timestamp all fail
    safe to ``None`` — the caller falls back to the worker-death path."""
    worktrees_dir, dispatched_at = _fresh_outcome_bed(tmp_path)
    worktree_path = worktree_path_for_branch(tmp_path, BRANCH, worktrees_dir)

    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at
        )
        is None
    )
    assert (
        fresh_completed_worker_outcome(None, live_head_sha=HEAD_SHA, dispatched_at=dispatched_at)
        is None
    )

    _write_worktree_outcome(tmp_path, worktrees_dir, BRANCH, _outcome())
    assert (
        fresh_completed_worker_outcome(
            worktree_path, live_head_sha=None, dispatched_at=dispatched_at
        )
        is None
    )
    assert (
        fresh_completed_worker_outcome(worktree_path, live_head_sha=HEAD_SHA, dispatched_at=None)
        is None
    )
