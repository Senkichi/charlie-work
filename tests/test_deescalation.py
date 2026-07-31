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

import ast
import importlib.util
from pathlib import Path

from charlie_work.instrumentation import log_event
from charlie_work.state import (
    DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS,
    ESCALATION_REASON_CLASS_BY_EVENT_KIND,
    PASSIVE_OPEN_STATUS,
    load_state,
    save_state,
    state_lock,
)

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


# --- issue #797 legacy reason_class backfill tests ---


def _log_escalation_event(state_path: Path, issue_number: int, kind: str) -> None:
    """Write a single escalation event directly to ``events.db``."""
    payload = {"issue_number": issue_number, "reason": "legacy"}
    if kind == "janitor_rework_escalated":
        payload["pr_number"] = 456
        payload["reason"] = "merge_conflict"
        payload["attempts"] = 3
    log_event(state_path, kind, payload, repo="test-repo")


def test_backfill_reason_class_makes_legacy_session_failed_visible_but_stays_terminal(
    tmp_path: Path,
) -> None:
    """Issue 645/673 scenario: a legacy ``session_failed_escalated`` issue
    acquires ``reason_class == "mechanical"`` and becomes a sweep candidate,
    but without a live, green, OPEN PR it still cannot clear."""
    app = _app(tmp_path)
    _log_escalation_event(app.paths.state_file, 123, "session_failed_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            # Deliberately no reason_class: legacy state.
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["reason_class"] == "mechanical"
    assert issue["escalation_reason"] == "session_failed_escalated"

    backfilled = _events(state, "deescalation_reason_class_backfilled")
    assert len(backfilled) == 1
    assert backfilled[0]["payload"]["issue_number"] == 123
    assert backfilled[0]["payload"]["from_event_kind"] == "session_failed_escalated"
    assert backfilled[0]["payload"]["reason_class"] == "mechanical"

    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    assert passes[0]["payload"]["candidates"] == 1
    assert passes[0]["payload"]["cleared"] == []


def test_backfill_reason_class_leaves_janitor_rework_escalated_untouched(
    tmp_path: Path,
) -> None:
    """Issue 662 scenario: ``janitor_rework_escalated`` is a deliberately
    preserved forensic record and must stay fail-closed."""
    app = _app(tmp_path)
    _log_escalation_event(app.paths.state_file, 662, "janitor_rework_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["662"] = {
            "number": 662,
            "status": "escalated",
            "escalation_reason": "janitor_rework_escalated",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["662"]
    assert issue["status"] == "escalated"
    assert "reason_class" not in issue
    assert _events(state, "deescalation_reason_class_backfilled") == []
    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    assert passes[0]["payload"]["candidates"] == 0


def test_backfill_reason_class_leaves_missing_event_untouched(
    tmp_path: Path,
) -> None:
    """Issues 602/627 scenario: no escalation-transition event was recorded,
    so there is no evidence to backfill from."""
    app = _app(tmp_path)

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["602"] = {
            "number": 602,
            "status": "escalated",
            "escalation_reason": "dispatch_failed_cap_exceeded",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["602"]
    assert issue["status"] == "escalated"
    assert "reason_class" not in issue
    assert _events(state, "deescalation_reason_class_backfilled") == []
    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    assert passes[0]["payload"]["candidates"] == 0


def test_backfill_reason_class_is_idempotent(tmp_path: Path) -> None:
    """A second sweep on an already-backfilled issue must not re-emit the
    migration event or rewrite the field."""
    app = _app(tmp_path)
    _log_escalation_event(app.paths.state_file, 123, "session_failed_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert len(_events(state, "deescalation_reason_class_backfilled")) == 1

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        # Force the sweep to run again.
        state["deescalation_pass"] = {"next_deescalation_at": "2000-01-01T00:00:00Z"}
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert len(_events(state, "deescalation_reason_class_backfilled")) == 1


def test_backfill_reason_class_preserves_attempt_counters(
    tmp_path: Path,
) -> None:
    """Backfilling only writes ``reason_class``; it must not reset or bypass
    any existing per-mechanism or oscillation-guard counter."""
    app = _app(tmp_path)
    _log_escalation_event(app.paths.state_file, 123, "session_failed_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            "auto_deescalation_count": 1,
            "redispatch_at": ["2026-07-30T00:00:00Z"],
            "conflict_rework_attempts": 2,
            "request_changes_count": 3,
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["reason_class"] == "mechanical"
    assert issue["auto_deescalation_count"] == 1
    assert issue["redispatch_at"] == ["2026-07-30T00:00:00Z"]
    assert issue["conflict_rework_attempts"] == 2
    assert issue["request_changes_count"] == 3


def test_backfill_reason_class_then_clears_green_pr(tmp_path: Path) -> None:
    """A legacy mechanical issue with a green, OPEN, transiently-unknown PR
    is first backfilled, then de-escalated and relabeled in the same sweep."""
    app = _app(tmp_path)
    _log_escalation_event(app.paths.state_file, 123, "session_failed_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "headRefOid": "sha-abc123",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "request_changes_count": 0,
            "consecutive_failed_merge_attempts": 0,
            "janitor_ok": True,
            "janitor_failures": [],
            "review_dispatch_status": "review_dispatch_completed",
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
            "reviewed_head_sha": None,
            "reviewed_patch_id": None,
        }
        save_state(app.paths.state_file, state)

    # mergeable UNKNOWN is the permissive stale-head case the existing sweep
    # tests use; the default FakeGitHub is otherwise green and OPEN.
    app.gh.prs[0] = {**app.gh.prs[0], "mergeable": "UNKNOWN"}

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == PASSIVE_OPEN_STATUS
    assert issue["auto_deescalation_count"] == 1

    cleared = _events(state, "deescalation_cleared")
    assert len(cleared) == 1
    assert cleared[0]["payload"]["issue_number"] == 123
    assert cleared[0]["payload"]["cleared_condition"] == "session_failed_escalated"

    backfilled = _events(state, "deescalation_reason_class_backfilled")
    assert len(backfilled) == 1
    assert backfilled[0]["payload"]["issue_number"] == 123


def test_backfill_reason_class_positive_control(tmp_path: Path) -> None:
    """Positive control from issue #797: 6 of 9 legacy escalations acquire
    ``reason_class == "mechanical"`` and become visible; 3 named exclusions
    remain fail-closed."""
    app = _app(tmp_path)

    # Six session_failed escalations, three no-event, one janitor_rework.
    session_failed = [592, 593, 606, 645, 648, 673]
    no_event = [602, 627]
    janitor = [662]

    for issue_number in session_failed:
        _log_escalation_event(app.paths.state_file, issue_number, "session_failed_escalated")
    _log_escalation_event(app.paths.state_file, 662, "janitor_rework_escalated")

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        for issue_number in session_failed + no_event + janitor:
            state["issues"][str(issue_number)] = {
                "number": issue_number,
                "status": "escalated",
                "escalation_reason": "session_failed_escalated",
            }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    for issue_number in session_failed:
        assert state["issues"][str(issue_number)]["reason_class"] == "mechanical", issue_number
    for issue_number in no_event + janitor:
        assert "reason_class" not in state["issues"][str(issue_number)], issue_number

    backfilled = _events(state, "deescalation_reason_class_backfilled")
    assert len(backfilled) == 6
    assert {e["payload"]["issue_number"] for e in backfilled} == set(session_failed)

    passes = _events(state, "deescalation_pass_completed")
    assert len(passes) == 1
    assert passes[0]["payload"]["candidates"] == 6
    assert passes[0]["payload"]["cleared"] == []


# --- mapping completeness machine check ---


def _is_event_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in ("append_event", "_record_event")
    if isinstance(func, ast.Attribute):
        return func.attr == "_record_event"
    return False


def _call_arg(call: ast.Call, pos: int, kw: str) -> ast.expr | None:
    if pos < len(call.args):
        return call.args[pos]
    for k in call.keywords:
        if k.arg == kw:
            return k.value
    return None


def _dict_literal_keys(node: ast.expr) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _escalation_event_kinds_from_workflow() -> set[str]:
    """Discover escalation-transition event kinds by inspecting workflow.py.

    An escalation event is an ``append_event`` / ``_record_event`` call whose
    ``kind`` ends with ``_escalated``, or a ``_record_event`` call in a function
    that also assigns a ``reason_class`` and whose payload carries an
    ``escalated`` key.

    The test is deliberately conservative: only kinds that unambiguously mark
    an ``escalated``/``blocked`` status transition with a ``reason_class`` are
    discovered. Any new escalation kind that does not match these patterns must
    be explicitly added to the mapping/unclassified sets, so it fails CI
    instead of silently degrading to fail-closed-forever.
    """
    spec = importlib.util.find_spec("charlie_work.workflow")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Build parent map so we can find the enclosing function for any node.
    parents: dict[ast.AST, ast.AST] = {}

    class _ParentVisitor(ast.NodeVisitor):
        def visit(self, node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                parents[child] = node
                self.visit(child)

    _ParentVisitor().visit(tree)

    def _enclosing_function(node: ast.AST) -> ast.AST | None:
        while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = parents.get(node)
        return node

    # Identify functions that assign a reason_class. This filters out
    # diagnostics-only events like ``janitor_gate`` that merely observe an
    # already-escalated PR without performing an escalation transition.
    def _uses_reason_class(func: ast.AST) -> bool:
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                name: str | None = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name == "escalation_reason_class":
                    return True
        return False

    # First pass: collect local dict-literal assignments per function.
    func_dicts: dict[ast.AST, dict[str, set[str]]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_dicts[func] = {}
        for assign in ast.walk(func):
            if isinstance(assign, ast.Assign):
                for target in assign.targets:
                    if isinstance(target, ast.Name) and isinstance(assign.value, ast.Dict):
                        func_dicts[func][target.id] = _dict_literal_keys(assign.value)
            elif isinstance(assign, ast.AnnAssign) and isinstance(assign.target, ast.Name):
                if isinstance(assign.value, ast.Dict):
                    func_dicts[func][assign.target.id] = _dict_literal_keys(assign.value)

    kinds: set[str] = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not _is_event_call(call.func):
            continue
        kind_node = _call_arg(call, 1, "kind")
        if not isinstance(kind_node, ast.Constant) or not isinstance(kind_node.value, str):
            continue
        kind = kind_node.value
        if kind.endswith("_escalated"):
            kinds.add(kind)
            continue

        func = _enclosing_function(call)
        if func is None or not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _uses_reason_class(func):
            continue

        payload_node = _call_arg(call, 2, "payload")
        if payload_node is None:
            continue
        payload_keys = _dict_literal_keys(payload_node)
        if not payload_keys and isinstance(payload_node, ast.Name):
            payload_keys = func_dicts.get(func, {}).get(payload_node.id, set())
        if "escalated" in payload_keys:
            kinds.add(kind)

    return kinds


def test_escalation_event_kind_mapping_is_complete() -> None:
    """Every escalation-transition event kind the code can emit is either
    classified in ``ESCALATION_REASON_CLASS_BY_EVENT_KIND`` or explicitly
    listed as deliberately unclassified. A new kind that is not accounted for
    fails CI instead of silently degrading to fail-closed-forever."""
    discovered = _escalation_event_kinds_from_workflow()
    expected = set(ESCALATION_REASON_CLASS_BY_EVENT_KIND) | set(
        DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS
    )
    assert discovered == expected, (
        f"escalation event kinds not covered by mapping: {discovered - expected}\n"
        f"mapping/unclassified kinds not discovered in source: {expected - discovered}"
    )
