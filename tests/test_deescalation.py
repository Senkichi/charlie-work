"""Tests for issue #783's automated de-escalation sweep.

Escalation to ``agent:human-needed`` used to be a one-way door: four
``labels.py`` edges (``escalated``, ``blocked``, ``redispatch_escalated``,
``merged_pr_mention_flagged``) add the label; nothing in the automated loop
ever removed it, so PRs whose underlying artifact was already fine (pushed,
open, CI green, ``janitor_ok``) sat stuck behind a worker's process failure
(e.g. ``session_failed_escalated``) until an operator ran
``charlie unescalate`` by hand.

``OrchestratorApp._maybe_deescalate_mechanical`` /
``OrchestratorApp._deescalate_mechanical_issue`` are the automated re-entry
point for exactly that process-failure class, and only that class:

- Only ``reason_class == "mechanical"`` entries are ever candidates (written
  atomically alongside every ``status -> escalated/blocked`` transition at
  its call site). ``judgment`` escalations, and any pre-existing escalation
  with no recorded ``reason_class`` at all, fail closed and are never
  auto-cleared -- this module's two required regression tests
  (``test_deescalation_sweep_clears_mechanical_and_leaves_judgment_untouched``
  and ``test_deescalation_sweep_leaves_missing_reason_class_untouched``)
  prove exactly that.
- Clearing additionally requires a live, freshly-fetched PR that is OPEN,
  not ``mergeable == "CONFLICTING"`` (mirrors ``janitor._check_mergeable``'s
  own permissive definition -- a transient ``"UNKNOWN"`` mergeability value,
  the normal state for a few minutes after any push, is not treated as a
  conflict), and passes a freshly-computed ``run_janitor().ok``.
- ``auto_deescalation_count`` bounds how many times the sweep may clear the
  same issue (the oscillation guard, hazard (a) in issue #783): once it
  reaches ``config.deescalation.max_auto_deescalations``, the sweep stops
  clearing and instead emits ``deescalation_cap_exhausted`` exactly once,
  guarded by a ``deescalation_cap_notified_at`` marker that only a manual
  ``charlie unescalate`` resets.

These tests reuse ``FakeGitHub`` and the ``_second_mergequeue_pr`` two-issue
fixture helper from test_charlie_work.py, and the ``_app`` isolation helper
from test_fix_unescalate.py (pointing ``post_mortem.db_path`` at a
nonexistent path so ``issue_worker_liveness`` never picks up a real
self-hosted-runner ``sessions.db`` for the test PID).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock

from test_charlie_work import _second_mergequeue_pr
from test_fix_unescalate import _app, _events


def test_deescalation_sweep_clears_mechanical_and_leaves_judgment_untouched(
    tmp_path: Path,
) -> None:
    """AC5 + AC2: one sweep pass over two escalated issues clears the
    ``mechanical`` one and leaves the ``judgment`` one on the same-shaped PR
    completely untouched -- proving the sweep is a loop-level scan (not
    PR-number-driven) that discriminates purely on ``reason_class``.

    Issue 123/PR 456 also pins the mergeable-permissiveness fix: ``mergeable``
    is explicitly ``"UNKNOWN"`` (the common transient value in the minutes
    after any push), which must still clear -- only literal ``"CONFLICTING"``
    may block, mirroring ``janitor._check_mergeable`` exactly.
    """
    app = _app(tmp_path)
    _second_mergequeue_pr(app.gh)  # adds issue 124 / PR 789
    app.gh.prs[0] = {**app.gh.prs[0], "mergeable": "UNKNOWN"}

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            "reason_class": "mechanical",
        }
        state["prs"]["789"] = {
            "number": 789,
            "issue_number": 124,
            "status": "escalated",
        }
        state["issues"]["124"] = {
            "number": 124,
            "status": "escalated",
            "escalation_reason": "merged_pr_mention_flagged",
            "reason_class": "judgment",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)

    # The mechanical issue cleared: back to the passive open status, every
    # escalation-specific field dropped, the oscillation-guard counter bumped.
    issue_123 = state["issues"]["123"]
    assert issue_123["status"] == PASSIVE_OPEN_STATUS
    assert "reason_class" not in issue_123
    assert "escalation_reason" not in issue_123
    assert issue_123["auto_deescalation_count"] == 1
    assert (123, app.config.labels.pr_open) in app.gh.labels_added
    assert (123, app.config.labels.human_needed) in app.gh.labels_removed

    # The judgment issue on an identically-shaped PR is completely untouched.
    issue_124 = state["issues"]["124"]
    assert issue_124["status"] == "escalated"
    assert issue_124["reason_class"] == "judgment"
    assert issue_124["escalation_reason"] == "merged_pr_mention_flagged"
    assert "auto_deescalation_count" not in issue_124
    assert all(num != 124 for (num, _label) in app.gh.labels_added)
    assert all(num != 124 for (num, _label) in app.gh.labels_removed)

    cleared = _events(state, "deescalation_cleared")
    assert len(cleared) == 1
    assert cleared[0]["payload"]["issue_number"] == 123
    assert cleared[0]["payload"]["pr_number"] == 456
    assert cleared[0]["payload"]["reason_class"] == "mechanical"
    assert cleared[0]["payload"]["cleared_condition"] == "session_failed_escalated"
    assert cleared[0]["payload"]["pr_mergeable"] == "UNKNOWN"
    assert cleared[0]["payload"]["janitor_ok"] is True

    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    # Only the mechanical issue was ever a candidate -- the judgment issue's
    # entry never enters the candidate query at all (AC2).
    assert passes[0]["payload"]["candidates"] == 1


def test_deescalation_sweep_leaves_missing_reason_class_untouched(tmp_path: Path) -> None:
    """AC6: an escalation recorded before this field existed -- no
    ``reason_class`` key at all -- must fail closed and stay exactly as
    terminal as it was, never retroactively guessed at, even though the PR
    itself is green and mergeable.
    """
    app = _app(tmp_path)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            # Deliberately no "reason_class" key: every escalation written
            # before issue #783 shipped looks exactly like this.
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue_123 = state["issues"]["123"]
    assert issue_123["status"] == "escalated"
    assert "reason_class" not in issue_123
    assert "auto_deescalation_count" not in issue_123

    assert _events(state, "deescalation_cleared") == []
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []

    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    # The candidate query itself requires reason_class == "mechanical", so an
    # entry with the key absent never even enters the candidate list.
    assert passes[0]["payload"]["candidates"] == 0


def test_deescalation_cap_exhausted_stops_clearing_and_notifies_once(tmp_path: Path) -> None:
    """Hazard (a) oscillation guard: once ``auto_deescalation_count`` reaches
    ``max_auto_deescalations`` (default 2), the sweep must stop clearing the
    issue -- even though it is still ``reason_class == "mechanical"`` on a
    green PR -- and must emit ``deescalation_cap_exhausted`` exactly once,
    not on every subsequent pass, via the ``deescalation_cap_notified_at``
    dedup marker.
    """
    app = _app(tmp_path)
    assert app.config.deescalation.max_auto_deescalations == 2

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            "reason_class": "mechanical",
            "auto_deescalation_count": 2,
        }
        save_state(app.paths.state_file, state)

    # First evaluation: cap already reached -> must not clear, must notify.
    outcome = app._deescalate_mechanical_issue(123)
    assert outcome == {"cap_exhausted": True, "issue_number": 123}

    state = load_state(app.paths.state_file)
    issue_123 = state["issues"]["123"]
    assert issue_123["status"] == "escalated"
    assert issue_123["reason_class"] == "mechanical"
    assert issue_123["deescalation_cap_notified_at"]
    assert app.gh.labels_added == []
    assert app.gh.labels_removed == []

    exhausted = _events(state, "deescalation_cap_exhausted")
    assert len(exhausted) == 1
    assert exhausted[0]["payload"] == {
        "issue_number": 123,
        "auto_deescalation_count": 2,
        "max_auto_deescalations": 2,
    }

    # Second evaluation (simulating the next periodic pass): the issue is
    # still capped, but the dedup marker must suppress a second event -- the
    # terminal state is diagnosable from the first event, not re-announced
    # forever.
    outcome_again = app._deescalate_mechanical_issue(123)
    assert outcome_again is None

    state = load_state(app.paths.state_file)
    assert len(_events(state, "deescalation_cap_exhausted")) == 1
    assert state["issues"]["123"]["status"] == "escalated"
