"""Regression tests for issue #1683: the auto-deescalation sweep cleared
mechanical review-dispatch escalations without resetting the counters that
gate them, so every automated clear was inert -- the very next
``dispatch_reviews`` pass re-escalated the PR (counter still at cap) without
a single new ``review_dispatch_claim``, burning ``auto_deescalation_count``
until ``deescalation_cap_exhausted`` parked the issue permanently
(live evidence: PR #1623 / issue #1614).

``REWORK_BUDGET_RESET_BY_ESCALATION_REASON`` mapped only the rework lanes'
reasons; both review-dispatch escalation reasons were absent:

- ``max_review_dispatch_attempts_exceeded`` gates on
  ``review_dispatch_attempt_count`` (workflow.py ``dispatch_reviews``).
- ``max_consecutive_turn_limit_misses_exceeded`` gates on
  ``review_turn_limit_miss_streak``.

Both counters are same-head-sensitive: ``review()``'s packet write
preserves them whenever ``review_dispatch_attempt_last_head`` matches the
packet head, and resets them only on a fresh dispatch cycle.  The reset
must therefore also pop ``review_dispatch_attempt_last_head`` so the next
packet write re-baselines the whole dispatch epoch.

The baseline pop alone does NOT re-arm the streak lane, though: the
re-baseline only fires when ``review()`` regenerates the packet, and the
normal post-clear state is a packet that is already current -- so the
sweep's clear must also zero ``review_dispatch_attempt_count`` directly
(every turn-limit miss is also a spent dispatch attempt, leaving it at
cap).  ``dispatch_reviews()``'s escalation check reads that counter
directly and never consults the baseline, so a clear that left it at cap
re-escalated under the OTHER reason before any new claim.  The
sweep-to-dispatch regression test below exercises exactly that path.

The structural guard at the bottom of this file derives the full set of
``reason_class="mechanical"`` escalation reasons reachable through
``_escalate_issue`` call sites -- including the dynamic ``failure_kind``
domain -- and asserts every one is either a map key or on a small,
commented "no per-mechanism counter exists" allowlist, so a future
mechanical lane cannot repeat this omission silently.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import charlie_work
from charlie_work.config import DETERMINISTIC_ESCALATION_FAILURE_KINDS
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.unescalate_reset_fields import REWORK_BUDGET_RESET_BY_ESCALATION_REASON

from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _write_review_packet,
)
from _unescalate_fixtures import _app, _events


def test_sweep_resets_review_dispatch_attempt_budget_on_first_clear(
    tmp_path: Path,
) -> None:
    """A clear of ``max_review_dispatch_attempts_exceeded`` must zero
    ``review_dispatch_attempt_count`` -- the counter that gates the cleared
    reason -- and pop ``review_dispatch_attempt_last_head`` so the next
    ``review()`` packet write opens a fresh dispatch cycle instead of
    preserving the exhausted epoch's counter under the same head.

    Without this, a PR escalated after infrastructure-killed review
    attempts (issue #1614 / PR #1623) re-escalates on the next dispatch
    pass at the cap check with no intervening ``review_dispatch_claim``,
    and each inert clear consumes one ``auto_deescalation_count`` slot
    until the sweep's own retry cap is exhausted.
    """
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": "max_review_dispatch_attempts_exceeded",
            # The gating counter at cap and its same-head baseline.
            "review_dispatch_attempt_count": 3,
            "review_dispatch_attempt_last_head": "sha-escalated-head",
            # Unrelated counters -- must NOT be reset by this lane's clear:
            # the sibling review-lane streak (it has its own reason) and the
            # rework lanes.
            "review_turn_limit_miss_streak": 2,
            "request_changes_count": 3,
            "no_op_rework_attempts": 2,
            "conflict_rework_attempts": 1,
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "max_review_dispatch_attempts_exceeded",
            "reason_class": "mechanical",
            "terminal_since": "2026-08-14T12:00:00Z",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    pr_456 = state["prs"]["456"]
    # The gating counter is reset -- the actual #1683 fix.
    assert pr_456["review_dispatch_attempt_count"] == 0
    # The same-head baseline is gone so the next packet write re-baselines
    # (a kept baseline would preserve the exhausted epoch's counters).
    assert "review_dispatch_attempt_last_head" not in pr_456
    # Unrelated lanes untouched -- the reset is scoped to the cleared reason.
    assert pr_456["review_turn_limit_miss_streak"] == 2
    assert pr_456["request_changes_count"] == 3
    assert pr_456["no_op_rework_attempts"] == 2
    assert pr_456["conflict_rework_attempts"] == 1
    # Escalation cleared on both sides.
    assert "escalation_reason" not in pr_456
    assert "escalation_reason" not in state["issues"]["123"]
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    # The per-episode marker is stamped and the event reports the reset
    # truthfully: a counter was actually zeroed.
    issue_123 = state["issues"]["123"]
    assert issue_123["rework_budget_reset_for_terminal_since"] == "2026-08-14T12:00:00Z"
    cleared = _events(state, "deescalation_cleared")
    assert cleared[0]["payload"]["rework_budget_reset"] is True
    assert cleared[0]["payload"]["rework_budget_reset_needed"] is True


def test_sweep_resets_turn_limit_miss_streak_and_rebaselines_head(
    tmp_path: Path,
) -> None:
    """A clear of ``max_consecutive_turn_limit_misses_exceeded`` must zero
    ``review_turn_limit_miss_streak`` -- the counter that gates the cleared
    reason -- AND ``review_dispatch_attempt_count`` -- the sibling-lane
    counter the next dispatch pass reads directly -- plus pop
    ``review_dispatch_attempt_last_head``.

    The direct attempt-count zero is load-bearing, not cosmetic: every
    turn-limit miss also consumed a ``review_dispatch_attempt_count``
    slot, so the attempt counter is at cap too, and
    ``dispatch_reviews()``'s escalation check reads it without consulting
    the baseline.  Relying on the popped baseline to re-zero it via
    ``review()``'s fresh-dispatch-cycle path leaves it at cap whenever
    the packet is already current -- the normal post-clear state -- so
    the next dispatch pass re-escalates under
    ``max_review_dispatch_attempts_exceeded`` before any new claim.
    """
    app = _app(tmp_path)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": "max_consecutive_turn_limit_misses_exceeded",
            # The gating counter at cap, plus the shared dispatch-epoch
            # baseline it is preserved under.
            "review_turn_limit_miss_streak": 3,
            "review_dispatch_attempt_last_head": "sha-escalated-head",
            # The sibling lane's counter at cap -- every miss spent an
            # attempt, so the sweep must zero it directly.
            "review_dispatch_attempt_count": 3,
            # Unrelated rework-lane counters -- must NOT be reset.
            "request_changes_count": 4,
            "no_op_rework_attempts": 2,
            "conflict_rework_attempts": 1,
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "max_consecutive_turn_limit_misses_exceeded",
            "reason_class": "mechanical",
            "terminal_since": "2026-08-14T12:00:00Z",
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    pr_456 = state["prs"]["456"]
    # The gating counter is reset -- the actual #1683 fix.
    assert pr_456["review_turn_limit_miss_streak"] == 0
    # The shared epoch baseline is popped so both counters re-baseline on
    # the next packet write.
    assert "review_dispatch_attempt_last_head" not in pr_456
    # The sibling lane's counter is zeroed synchronously -- the deferred
    # packet-write re-baseline cannot be relied on because the packet is
    # typically already current after a clear.
    assert pr_456["review_dispatch_attempt_count"] == 0
    # Unrelated rework lanes untouched.
    assert pr_456["request_changes_count"] == 4
    assert pr_456["no_op_rework_attempts"] == 2
    assert pr_456["conflict_rework_attempts"] == 1
    # Escalation cleared on both sides.
    assert "escalation_reason" not in pr_456
    assert "escalation_reason" not in state["issues"]["123"]
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    cleared = _events(state, "deescalation_cleared")
    assert cleared[0]["payload"]["rework_budget_reset"] is True
    assert cleared[0]["payload"]["rework_budget_reset_needed"] is True


def test_streak_clear_then_dispatch_pass_redispatches_instead_of_reescalating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end regression for the round-2 finding: after the sweep clears
    ``max_consecutive_turn_limit_misses_exceeded``, the very next
    ``dispatch_reviews()`` pass must see a PR with a genuinely fresh
    dispatch budget -- claimed and launched as ONE new attempt -- not
    re-escalated under ``max_review_dispatch_attempts_exceeded`` with zero
    new ``review_dispatch_claim``s.

    This is the exact hole a baseline-pop-only re-arm leaves: the cleared
    PR's packet is already current (``review()``'s fresh-cycle re-baseline
    never fires), while ``dispatch_reviews()``'s escalation check reads
    ``review_dispatch_attempt_count`` directly and never consults
    ``review_dispatch_attempt_last_head`` -- so a still-at-cap count
    re-escalated under the sibling reason before any new dispatch attempt.
    """
    app = _dispatch_reviews_app(tmp_path)
    # Packet already current at the live head -- the normal post-clear
    # state, and the condition that makes the deferred review()
    # re-baseline unreachable.
    _write_review_packet(tmp_path, 456, "sha-abc123")
    # _escalate_issue stamps these claim fields via pr_extra when the
    # streak cap trips; age failed_at past the stale-claim timeout so the
    # cleared PR is immediately re-dispatchable (the escalation sat far
    # longer than the 5-minute claim timeout before the sweep cleared it).
    stale_failed_at = (
        (datetime.now(UTC) - timedelta(minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES + 5))
        .isoformat()
        .replace("+00:00", "Z")
    )
    terminal_since = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": "max_consecutive_turn_limit_misses_exceeded",
            "review_turn_limit_miss_streak": 3,
            "review_dispatch_attempt_count": 3,
            "review_dispatch_attempt_last_head": "sha-abc123",
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_failed_at": stale_failed_at,
            "review_dispatch_pending_at": None,
            "review_dispatched_at": None,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "max_consecutive_turn_limit_misses_exceeded",
            "reason_class": "mechanical",
            "terminal_since": terminal_since,
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    # Sanity: the sweep actually cleared the escalation -- otherwise the
    # dispatch pass below proves nothing about the re-arm.
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    assert "escalation_reason" not in state["issues"]["123"]

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> Any:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(456, "agent/issue-123-fix-search")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(app.paths.state_file)
    # The pre-fix shape: an immediate second escalation under the SIBLING
    # reason on the still-at-cap attempt counter.
    assert _events(state, "review_dispatch_escalated") == []
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    pr_456 = state["prs"]["456"]
    assert pr_456.get("status") != "escalated"
    # Exactly one new dispatch attempt since the clear -- the fresh claim
    # the re-armed budget makes possible.
    assert int(pr_456["review_dispatch_attempt_count"]) == 1
    assert pr_456["review_dispatch_status"] == "review_dispatch_dispatched"
    assert len(launched) == 1


# --- structural guard -------------------------------------------------
#
# The defect class this issue fixes is "a mechanical escalation reason
# whose lane gates on a per-mechanism counter had no map entry".  A
# hand-maintained copy of the reason list in a test would silently drift
# the moment a new lane is added, so the set below is DERIVED from the
# source: every ``_escalate_issue`` call in ``charlie_work`` whose
# ``reason_class`` argument can be ``"mechanical"``, with the ``reason``
# argument resolved through local assignments, if-expression branches,
# f-strings, and function-parameter call-site propagation.
#
# Reasons that cannot be reduced to a finite literal set are represented
# by dynamic descriptors:
#
# - ``failure_kind`` (bare name or ``*.failure_kind`` attribute): every
#   mechanical-classed call site membership-gates it on
#   ``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` (deterministic-judgment
#   kinds select ``reason_class="judgment"`` instead), so the live
#   frozenset is substituted -- a new kind added to that set is checked
#   automatically.
# - ``fstring:<skeleton>`` and ``expr:<source>``: genuinely unbounded
#   domains, allowlisted per-shape below with a justification.


# Mechanical escalation reasons with no per-mechanism PR-record counter
# for the sweep to reset.  Each entry documents why a clear needs no
# counter reset -- a NEW reason landing here must carry the same kind of
# justification, and this set must stay exactly in step with what the AST
# scan discovers (asserted both directions).
_NO_COUNTER_ALLOWLIST: dict[str, str] = {
    # --- event/condition-driven reasons; nothing counted to re-arm ---
    "dead_dispatched_worker_reap": (
        "Fires when a dispatched worker's sidecar ages past "
        "watchdog.dead_dispatched_reap_minutes; the guard is wall-clock "
        "drift on a live claim, not a resettable counter."
    ),
    "worker_declared_blocked": ("Worker self-reported a blocked state; event-driven, no counter."),
    "zero_artifact_dispatch_loop": (
        "Detected from sessions-dir artifacts "
        "(_is_zero_artifact_dispatch_loop), not a state.json counter."
    ),
    "cross_repo_hop": (
        "Deterministic failure kind: the worker's issue scope targets "
        "another managed repo; re-dispatch repeats the hop regardless of "
        "any counter."
    ),
    "orphan_sweep_redispatch_cap_exceeded": (
        "Gated by the issue-level ``orphan_redispatch_*`` windowed "
        "bookkeeping (UNESCALATE_ISSUE_RESET_FIELDS domain, not PR "
        "fields); the sweep additionally never reaches it -- a PR-less "
        "issue skips as ``no_open_pr``."
    ),
    "infra_rerun_cap_exceeded": (
        "Gated by live infra-failed required checks -- themselves a "
        "janitor blocker, so the sweep cannot clear while the condition "
        "persists -- and by the per-head/per-run-id ``infra_rerun_"
        "attempts`` budget, which re-baselines on head change and per "
        "new run id; no flat counter to reset."
    ),
    "stale_checks_retrigger_exhausted": (
        "Gated by ``stale_checks_retrigger_attempts``, a per-PR monotonic "
        "retrigger budget that even the operator door deliberately does "
        "not reset (absent from UNESCALATE_PR_RESET_FIELDS) -- and the "
        "triggering condition (required checks missing) is a janitor "
        "blocker, so the sweep cannot clear while it persists."
    ),
    # --- issue-level windowed caps: the sweep's map resets PR-record ---
    # --- fields only; these live on the issue record and are the      ---
    # --- operator door's UNESCALATE_ISSUE_RESET_FIELDS domain. They   ---
    # --- re-gate only on a fresh dispatch attempt (the next genuine   ---
    # --- rework cycle), not on the very next pass.                    ---
    "redispatch_cap_exceeded": (
        "Gated by the issue-level ``redispatch_at`` windowed-timestamp "
        "list, not a PR counter; the windowed list is operator-door "
        "domain (UNESCALATE_ISSUE_RESET_FIELDS)."
    ),
    "worker_death_loop": (
        "Gated by the issue-level ``worker_death_at`` windowed-timestamp "
        "list; operator-door domain."
    ),
    "dispatch_blocked_environment": (
        "Gated by the issue-level ``blocked_environment_at`` "
        "windowed-timestamp list; operator-door domain."
    ),
    "dispatch_failed_cap_exceeded": (
        "Gated by the issue-level ``dispatch_failed_at`` "
        "windowed-timestamp list; operator-door domain."
    ),
    # --- DETERMINISTIC_ESCALATION_FAILURE_KINDS members: deterministic ---
    # --- failure classifications, escalated on first occurrence --    ---
    # --- there is no retry counter to re-arm.                         ---
    "worker_blocked": "Deterministic failure kind; escalates on first occurrence.",
    "worktree_unsafe_shim_dirt": "Deterministic failure kind; escalates on first occurrence.",
    "rework_branch_conflict": "Deterministic failure kind; escalates on first occurrence.",
    "provider_suspended": "Deterministic failure kind; escalates on first occurrence.",
    # --- local (no-remote) lane, issue #1844: condition-driven, no counter ---
    "local_branch_missing": (
        "Fires when the branch a local lane record points at no longer "
        "resolves to a commit; the guard is the branch ref itself, not a "
        "resettable counter."
    ),
    "local_merge_error": (
        "Fires on a local merge-gate infrastructure failure (worktree "
        "attach, base-sync exception, unresolvable suite command, merge "
        "refusal); event-driven, no counter."
    ),
}

# Dynamic ``reason`` domains that cannot be reduced to a finite literal
# set.  Each entry is keyed by the resolver's descriptor and justified in
# its value; a NEW dynamic reason expression at any mechanical call site
# fails the guard until it is justified here or made enumerable.
_DYNAMIC_REASON_DOMAINS: dict[str, str] = {
    "fstring:rescue_review_{*}": (
        "rescue_review_<cause> is formatted from the stored ``rescue_cause`` "
        "marker; the lane's re-entry guard is the durable one-shot "
        "``rescue_attempted`` marker, not a per-mechanism counter."
    ),
    "expr:gate_result.reason": (
        "The pre-flight cross-repo gate emits free-form prose reasons "
        "(``cross_repo_target: ...`` / ``cross_repo_scope: ...`` with "
        "embedded paths); the lane has no per-mechanism counter."
    ),
}


class _SourceIndex:
    """AST index over every ``charlie_work`` module: function-local name
    assignments, function signatures, module-level assignments, and every
    call site with its enclosing scope.

    Purpose-built for one question: "what literal strings can flow into
    the ``reason`` and ``reason_class`` kwargs of an ``_escalate_issue``
    call?" -- which requires resolving names through local assignments and
    (for parameters like ``attempts_key``) through the enclosing
    function's own call sites.
    """

    def __init__(self) -> None:
        pkg_dir = Path(charlie_work.__file__).resolve().parent
        # scope node -> {name: [assigned exprs]}; module trees and
        # function defs are both scope nodes (keyed by identity).
        self.assigns: dict[int, dict[str, list[ast.expr]]] = {}
        # func node -> ordered parameter names.
        self.param_names: dict[int, list[str]] = {}
        # func node -> {param name: default expr}.
        self.param_defaults: dict[int, dict[str, ast.expr]] = {}
        # func node -> enclosing module tree (for module-scope fallback).
        self.parent_scope: dict[int, ast.Module] = {}
        # every call in the package, with its innermost enclosing scope.
        self.calls: list[tuple[ast.AST, ast.Call]] = []
        for path in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self._index_module_scope(tree)
            self._walk(tree, tree)

    def _index_module_scope(self, tree: ast.Module) -> None:
        assigns: dict[str, list[ast.expr]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assigns.setdefault(node.target.id, []).append(node.value)
        self.assigns[id(tree)] = assigns

    def _walk(self, node: ast.AST, scope: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._index_func(node)
            self.parent_scope[id(node)] = (
                scope if isinstance(scope, ast.Module) else self.parent_scope[id(scope)]
            )
            scope = node
        if isinstance(node, ast.Call):
            self.calls.append((scope, node))
        for child in ast.iter_child_nodes(node):
            self._walk(child, scope)

    def _index_func(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        assigns: dict[str, list[ast.expr]] = {}

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                # Nested scopes bind their own names; do not descend.
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
                ):
                    continue
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigns.setdefault(target.id, []).append(child.value)
                elif (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.value is not None
                ):
                    assigns.setdefault(child.target.id, []).append(child.value)
                elif isinstance(child, (ast.For, ast.AsyncFor)) and isinstance(
                    child.target, ast.Name
                ):
                    # Loop targets are bound but not literal-resolvable.
                    assigns.setdefault(child.target.id, []).append(child.iter)
                elif isinstance(child, (ast.With, ast.AsyncWith)):
                    for item in child.items:
                        if isinstance(item.optional_vars, ast.Name):
                            assigns.setdefault(item.optional_vars.id, []).append(item.context_expr)
                visit(child)

        visit(func)
        self.assigns[id(func)] = assigns
        args = func.args
        self.param_names[id(func)] = [
            p.arg for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        ]
        defaults: dict[str, ast.expr] = {}
        positional = list(args.posonlyargs) + list(args.args)
        for param, default in zip(
            positional[len(positional) - len(args.defaults) :], args.defaults
        ):
            defaults[param.arg] = default
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                defaults[param.arg] = default
        self.param_defaults[id(func)] = defaults

    # -- expression resolution --

    def resolve(
        self, node: ast.expr, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        """Resolve ``node`` to (literal strings, dynamic descriptors).

        Over-approximates literals (union of every branch/assignment) so a
        missed string is never silently dropped; anything not reducible to
        a literal becomes a descriptor the caller must allowlist.
        """
        literals: set[str] = set()
        dynamics: set[str] = set()

        def merge(res: tuple[set[str], set[str]]) -> None:
            literals.update(res[0])
            dynamics.update(res[1])

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                literals.add(node.value)
        elif isinstance(node, ast.Name):
            merge(self._resolve_name(node.id, scope, in_progress))
        elif isinstance(node, ast.IfExp):
            merge(self.resolve(node.body, scope, in_progress))
            merge(self.resolve(node.orelse, scope, in_progress))
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                merge(self.resolve(elt, scope, in_progress))
        elif isinstance(node, ast.JoinedStr):
            merge(self._resolve_joined(node, scope, in_progress))
        elif isinstance(node, ast.Attribute) and node.attr == "failure_kind":
            # ``foo.failure_kind`` reaching a mechanical escalation site is
            # membership-gated to DETERMINISTIC_ESCALATION_FAILURE_KINDS at
            # every call site (judgment kinds take reason_class="judgment").
            literals.update(DETERMINISTIC_ESCALATION_FAILURE_KINDS)
        else:
            dynamics.add(f"expr:{ast.unparse(node)}")
        return literals, dynamics

    def _resolve_joined(
        self, node: ast.JoinedStr, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        segments: list[set[str]] = []
        dynamic = False
        for value in node.values:
            if isinstance(value, ast.Constant):
                segments.append({str(value.value)})
                continue
            lits, dyn = self.resolve(value.value, scope, in_progress)
            if dyn:
                dynamic = True
            segments.append(lits)
        if dynamic:
            skeleton = "".join(
                value.value if isinstance(value, ast.Constant) else "{*}" for value in node.values
            )
            return set(), {f"fstring:{skeleton}"}
        combos = [""]
        for segment in segments:
            combos = [c + s for c in combos for s in segment]
            if len(combos) > 64:
                skeleton = "".join(
                    value.value if isinstance(value, ast.Constant) else "{*}"
                    for value in node.values
                )
                return set(), {f"fstring:{skeleton}"}
        return set(combos), set()

    def _resolve_name(
        self, name: str, scope: ast.AST, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        # ``failure_kind`` is the one name that must never be chased
        # through the call graph: every mechanical-classed escalation site
        # membership-gates it on DETERMINISTIC_ESCALATION_FAILURE_KINDS
        # before it can reach ``_escalate_issue``, so the live frozenset
        # IS the mechanical domain (a new kind is picked up automatically).
        if name == "failure_kind":
            return set(DETERMINISTIC_ESCALATION_FAILURE_KINDS), set()
        seen_scope = scope
        while seen_scope is not None:
            scope_assigns = self.assigns.get(id(seen_scope), {}).get(name)
            if scope_assigns:
                literals: set[str] = set()
                dynamics: set[str] = set()
                for expr in scope_assigns:
                    lits, dyn = self.resolve(expr, seen_scope, in_progress)
                    literals |= lits
                    dynamics |= dyn
                # A name that is both assigned and a parameter can still
                # carry the caller's value into reads that precede the
                # assignment; union the parameter domain (over-approximate).
                if name in self.param_names.get(id(seen_scope), []):
                    lits, dyn = self._param_domain(seen_scope, name, in_progress)
                    literals |= lits
                    dynamics |= dyn
                return literals, dynamics
            if name in self.param_names.get(id(seen_scope), []):
                return self._param_domain(seen_scope, name, in_progress)
            seen_scope = self.parent_scope.get(id(seen_scope))
        return set(), {f"name:{name}"}

    def _param_domain(
        self, func: ast.AST, param: str, in_progress: frozenset
    ) -> tuple[set[str], set[str]]:
        """Resolve a function parameter from every call site of ``func``."""
        key = (id(func), param)
        if key in in_progress:
            return set(), set()
        in_progress = in_progress | {key}
        func_name = getattr(func, "name", "")
        positional = [
            p.arg
            for p in (*func.args.posonlyargs, *func.args.args)  # type: ignore[attr-defined]
        ]
        literals: set[str] = set()
        dynamics: set[str] = set()
        bound_somewhere = False
        for caller_scope, call in self.calls:
            if _callee_name(call) != func_name:
                continue
            arg = None
            for keyword in call.keywords:
                if keyword.arg == param:
                    arg = keyword.value
            if arg is None and param in positional:
                idx = positional.index(param)
                if idx < len(call.args):
                    arg = call.args[idx]
            if arg is None:
                # ``**kwargs`` could smuggle a binding we cannot see.
                if any(keyword.arg is None for keyword in call.keywords):
                    dynamics.add(f"param:{func_name}:{param}:**kwargs")
                continue
            bound_somewhere = True
            lits, dyn = self.resolve(arg, caller_scope, in_progress)
            literals |= lits
            dynamics |= dyn
        default = self.param_defaults.get(id(func), {}).get(param)
        if default is not None:
            lits, dyn = self.resolve(default, func, in_progress)
            literals |= lits
            dynamics |= dyn
        if not bound_somewhere and default is None:
            dynamics.add(f"param:{func_name}:{param}")
        return literals, dynamics


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _mechanical_escalation_reasons() -> tuple[set[str], set[str]]:
    """Derive every ``escalation_reason`` reachable with
    ``reason_class="mechanical"`` from ``_escalate_issue`` call sites.

    Returns ``(literal_reasons, dynamic_descriptors)``.  A call site is
    mechanical-capable when its ``reason_class`` resolves to a set
    containing ``"mechanical"`` or to anything unresolvable (conservative:
    an unresolvable class could be mechanical at runtime).
    """
    index = _SourceIndex()
    literal_reasons: set[str] = set()
    dynamic_sites: set[str] = set()
    for scope, call in index.calls:
        if _callee_name(call) != "_escalate_issue":
            continue
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        class_lits, class_dyn = (
            index.resolve(kwargs["reason_class"], scope, frozenset())
            if "reason_class" in kwargs
            else (set(), {"missing:reason_class"})
        )
        if "mechanical" not in class_lits and not class_dyn:
            continue
        if "reason" not in kwargs:
            dynamic_sites.add("missing:reason")
            continue
        reason_lits, reason_dyn = index.resolve(kwargs["reason"], scope, frozenset())
        literal_reasons |= reason_lits
        dynamic_sites |= reason_dyn
    return literal_reasons, dynamic_sites


def test_every_mechanical_escalation_reason_is_mapped_or_allowlisted() -> None:
    """Structural guard for issue #1683: every ``reason_class="mechanical"``
    escalation reason reachable via ``_escalate_issue`` must either be a
    key in ``REWORK_BUDGET_RESET_BY_ESCALATION_REASON`` (its lane's gating
    counter gets re-armed on clear) or appear on the commented
    ``_NO_COUNTER_ALLOWLIST``.

    Fails on the pre-fix codebase: ``max_review_dispatch_attempts_exceeded``
    and ``max_consecutive_turn_limit_misses_exceeded`` were in neither set.
    A future mechanical lane that adds a new ``_escalate_issue`` call (or a
    new ``attempts_key``/``failure_kind`` domain member) lands here as an
    unaccounted reason instead of repeating this omission silently.
    """
    literal_reasons, dynamic_sites = _mechanical_escalation_reasons()

    mapped = set(REWORK_BUDGET_RESET_BY_ESCALATION_REASON)
    allowlisted = set(_NO_COUNTER_ALLOWLIST)
    unaccounted = literal_reasons - mapped - allowlisted
    assert not unaccounted, (
        "mechanical escalation reason(s) with no reset-map entry and no "
        f"allowlist justification: {sorted(unaccounted)} -- if the lane "
        "gates on a per-mechanism counter, add a "
        "REWORK_BUDGET_RESET_BY_ESCALATION_REASON entry; otherwise add it "
        "to _NO_COUNTER_ALLOWLIST with a comment explaining why no "
        "counter reset applies"
    )
    # The allowlist must not rot: every entry must name a genuinely
    # reachable mechanical reason.
    stale_allowlist = allowlisted - literal_reasons
    assert not stale_allowlist, (
        f"allowlist entries no longer reachable as mechanical escalation "
        f"reasons (stale or renamed): {sorted(stale_allowlist)}"
    )
    # Dynamic reason domains must be exactly the justified set -- a new
    # unbounded domain fails until justified, and a removed one drops its
    # justification with it.
    assert dynamic_sites == set(_DYNAMIC_REASON_DOMAINS), (
        f"unexpected dynamic reason descriptors: "
        f"{sorted(dynamic_sites - set(_DYNAMIC_REASON_DOMAINS))}; "
        f"stale: {sorted(set(_DYNAMIC_REASON_DOMAINS) - dynamic_sites)}"
    )
