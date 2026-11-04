"""Issue #3025: a new lane cannot end a dead rework session without the throttle exemption.

``rework_attempt_exemption.exempt_provider_throttle_rework_death`` is the single point of
enforcement for "a provider-throttle death consumes no rework attempt", but a single point
of enforcement only helps if every caller reaches it. Two approved PRs were operator-queued
because one lane (``_route_dead_worker_to_pre_review_rework``) resolved the death as
``rate_limited``, used that only to skip adding a stamp, and never called it.

The shared thing every escalation of a dead rework session does is take the
``redispatch_escalated`` label edge (or its operator-queue twin,
``redispatch_operator_queued``). So this guard walks every function in ``src/`` that passes
either name to a call and requires each one to be classified:

* a **dead-rework-session lane** must call the exemption, and call it *before* its first
  edge (source order: the refund changes the history the cap reads, so a call placed after
  the edge is a call that came too late);
* anything else must be registered with the reason it is not a dead rework session.

A new caller that is in neither list fails the first test, naming the function, so the
author has to decide which it is. A registry entry whose function was renamed or removed
fails the second, so the registry cannot rot into a list that silently matches nothing.

This is the static half. The runtime half, a throttle death handled by each lane taking no
escalation edge, is ``tests/test_rework_throttle_exemption_lanes.py``. The scan is lexical:
an edge name built dynamically (an f-string, a variable) is invisible to it, which is why
the runtime tests exist alongside it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from _src_ast import parsed, parsed_source, source_files

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
EDGE_NAMES = frozenset({"redispatch_escalated", "redispatch_operator_queued"})
EXEMPTION_FN = "exempt_provider_throttle_rework_death"

# Functions that can end a dead rework session with an escalation edge. Each must apply
# the exemption before it chooses the edge.
DEAD_REWORK_SESSION_LANES: frozenset[str] = frozenset(
    {
        "dead_worker_sweep/effects_rework.py::_reap_restore_rework_requested",
        "dead_worker_sweep/effects_rework.py::_route_dead_worker_to_pre_review_rework",
    }
)

# Functions that take the edge but cannot be ending a dead rework session, with why.
NOT_A_DEAD_REWORK_SESSION: Mapping[str, str] = {
    "dead_worker_sweep/dead_sessions.py::_escalate_launch_failure": (
        "escalates a deterministic launch failure only when no PR is open "
        "(decide_dead_sessions.launch_failure_escalates); a rework session always has an "
        "open PR, and a launch that failed never ran a model, so there is no log to classify"
    ),
    "dead_worker_sweep/dead_sessions_reclaim.py::_reclaim_no_open_pr": (
        "the no-open-PR reclaim (reclaim_route -> Reclaim.NO_OPEN_PR); a rework session "
        "always has an open PR and goes to _route_open_pr instead"
    ),
    "reconcile.py::apply_fixes": (
        "applies the session_failed_escalated drift item, which detect_drift raises only "
        "for a worker with no open PR and a DETERMINISTIC_* failure kind; a provider-throttle "
        "kind is in neither set"
    ),
    "orchestration/dispatch_state.py::_dispatch_impl": (
        "fresh implementer dispatch: the launch-result cap and the cross-repo pre-flight "
        "gate. No rework session exists yet"
    ),
    "orchestration/state_dispatch_rework.py::_dispatch_rework_impl": (
        "rework DISPATCH, not death handling: the launch-result and blocked-environment "
        "caps run before any session has started, and they only read the redispatch history "
        "the exemption keeps clean"
    ),
    "orchestration/state_dispatch_rework.py::_apply_redispatch_escalated_edges": (
        "applies the label edge for issues dispatch_rework already escalated on the "
        "no-op / worker-death caps; it consumes the counters the exemption keeps clean"
    ),
}


@dataclass(frozen=True)
class EdgeUnit:
    """A function that passes an escalation-edge name to a call."""

    key: str
    first_edge_line: int
    first_exemption_line: int | None


def _units(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Outermost functions: module-level ones and class methods. Nested defs roll up."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    yield f"{node.name}.{sub.name}", sub


def _callee(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def edge_units(tree: ast.Module, rel: str) -> list[EdgeUnit]:
    """Every function in ``tree`` that passes an escalation-edge name to a call.

    The name may be the first argument of ``_escalation_edge(...)`` or a literal handed
    straight to ``transition(...)``: any ``EDGE_NAMES`` string constant used as a call
    argument counts. Dict keys (the edge tables in ``labels.py`` / ``escalation.py``) are
    definitions, not uses, and are not arguments.
    """
    found: list[EdgeUnit] = []
    for name, fn in _units(tree):
        edge_lines: list[int] = []
        exemption_lines: list[int] = []
        for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
            if _callee(call) == EXEMPTION_FN:
                exemption_lines.append(call.lineno)
            values = [*call.args, *(kw.value for kw in call.keywords)]
            edge_lines.extend(
                v.lineno for v in values if isinstance(v, ast.Constant) and v.value in EDGE_NAMES
            )
        if edge_lines:
            found.append(
                EdgeUnit(
                    key=f"{rel}::{name}",
                    first_edge_line=min(edge_lines),
                    first_exemption_line=min(exemption_lines) if exemption_lines else None,
                )
            )
    return found


def lane_violation(unit: EdgeUnit) -> str | None:
    """Why a dead-rework-session lane does not apply the exemption in time, else None."""
    if unit.first_exemption_line is None:
        return f"{unit.key} never calls {EXEMPTION_FN}"
    if unit.first_exemption_line >= unit.first_edge_line:
        return (
            f"{unit.key} calls {EXEMPTION_FN} at line {unit.first_exemption_line}, not before "
            f"its first escalation edge at line {unit.first_edge_line}"
        )
    return None


def _scan_src() -> dict[str, EdgeUnit]:
    units: dict[str, EdgeUnit] = {}
    for path in source_files(SRC_ROOT):
        rel = path.relative_to(SRC_ROOT).as_posix()
        for unit in edge_units(parsed(path), rel):
            units[unit.key] = unit
    return units


def test_every_escalation_edge_caller_is_classified() -> None:
    """A function that takes ``redispatch_escalated`` / ``redispatch_operator_queued`` must be
    either a registered dead-rework-session lane or a registered non-rework-death context."""
    unclassified = sorted(
        set(_scan_src()) - DEAD_REWORK_SESSION_LANES - set(NOT_A_DEAD_REWORK_SESSION)
    )
    assert not unclassified, (
        "New caller(s) of the redispatch escalation edge: "
        f"{unclassified}. Decide whether each can end a dead rework session. If it can, "
        f"it must call {EXEMPTION_FN} before choosing the edge and be added to "
        "DEAD_REWORK_SESSION_LANES. If it cannot, add it to NOT_A_DEAD_REWORK_SESSION with "
        "the reason (#3025)."
    )


def test_registry_entries_all_resolve_to_real_callers() -> None:
    """A renamed or deleted function must not leave a registry entry matching nothing."""
    found = set(_scan_src())
    registered = DEAD_REWORK_SESSION_LANES | set(NOT_A_DEAD_REWORK_SESSION)
    assert not (registered - found), (
        f"registry entries with no escalation-edge caller behind them: "
        f"{sorted(registered - found)}"
    )
    assert not (DEAD_REWORK_SESSION_LANES & set(NOT_A_DEAD_REWORK_SESSION)), (
        "a function cannot be both a dead-rework-session lane and a non-rework-death context"
    )
    assert all(len(reason) > 40 for reason in NOT_A_DEAD_REWORK_SESSION.values()), (
        "every NOT_A_DEAD_REWORK_SESSION entry needs a real reason"
    )


def test_restore_lane_predicate_covers_exactly_the_exempt_kinds() -> None:
    """``_reap_restore_rework_requested`` decides "throttle death, never escalate" with
    ``is_provider_throttle_failure`` and only afterwards applies the exemption (which uses
    ``PROVIDER_THROTTLE_EXEMPT_KINDS``). The two agree today. If a kind were exempt but not
    in the predicate, that lane would count the death and could escalate before reaching the
    exemption, which is the #3025 hole. This pin makes the divergence fail here instead."""
    from charlie_work.rework_attempt_exemption import PROVIDER_THROTTLE_EXEMPT_KINDS
    from charlie_work.throttle_signatures import PROVIDER_THROTTLE_FAILURE_KINDS

    assert PROVIDER_THROTTLE_EXEMPT_KINDS == PROVIDER_THROTTLE_FAILURE_KINDS, (
        "the exemption now covers kinds the restore lane's predicate does not "
        f"({sorted(PROVIDER_THROTTLE_EXEMPT_KINDS ^ PROVIDER_THROTTLE_FAILURE_KINDS)}); "
        "teach _reap_restore_rework_requested to decide from the exemption's result "
        "(as _route_dead_worker_to_pre_review_rework does) before extending the set"
    )


def test_dead_rework_session_lanes_apply_the_exemption_before_their_edge() -> None:
    units = _scan_src()
    violations = [
        v for key in sorted(DEAD_REWORK_SESSION_LANES) if (v := lane_violation(units[key]))
    ]
    assert not violations, "; ".join(violations)


# --- detector controls: the guard above must be able to fail -------------------------------

_LANE_OK = """
def lane(state, entry, failure_kind):
    exempt_provider_throttle_rework_death(state, 1, entry, failure_kind)
    edge = _escalation_edge("redispatch_escalated", "mechanical")
"""
_LANE_BYPASS = """
def lane(state, entry, failure_kind):
    edge = _escalation_edge("redispatch_escalated", "mechanical")
"""
_LANE_TOO_LATE = """
def lane(state, entry, failure_kind):
    edge = _escalation_edge("redispatch_escalated", "mechanical")
    exempt_provider_throttle_rework_death(state, 1, entry, failure_kind)
"""
_LANE_LITERAL_EDGE = """
class Lane:
    def end(self, gh, labels):
        write_gate.transition(gh, labels, 1, "redispatch_operator_queued")
"""


def _only(source: str, rel: str = "synthetic.py") -> EdgeUnit:
    (unit,) = edge_units(parsed_source(source), rel)
    return unit


def test_detector_accepts_a_lane_that_exempts_first() -> None:
    assert lane_violation(_only(_LANE_OK)) is None


def test_detector_flags_a_lane_that_bypasses_the_exemption() -> None:
    assert "never calls" in (lane_violation(_only(_LANE_BYPASS)) or "")


def test_detector_flags_a_lane_that_exempts_after_choosing_the_edge() -> None:
    assert "not before" in (lane_violation(_only(_LANE_TOO_LATE)) or "")


def test_detector_sees_a_literal_operator_queue_edge_in_a_method() -> None:
    """A caller that skips ``_escalation_edge`` and hands the edge literal to ``transition``
    is still found, keyed by its class-qualified name."""
    assert _only(_LANE_LITERAL_EDGE).key == "synthetic.py::Lane.end"


def test_neutering_the_real_lanes_trips_the_guard() -> None:
    """Strip the exemption call from the real ``effects_rework`` source: both registered
    lanes must then be flagged. If this passes while the guard stays green on a source with
    the call removed, the guard is not measuring the call."""
    rel = "dead_worker_sweep/effects_rework.py"
    source = (SRC_ROOT / rel).read_text(encoding="utf-8")
    assert f"{EXEMPTION_FN}(" in source
    neutered = source.replace(f"{EXEMPTION_FN}(", "_neutered(")
    by_key = {u.key: u for u in edge_units(parsed_source(neutered), rel)}
    flagged = {key for key in DEAD_REWORK_SESSION_LANES if lane_violation(by_key[key])}
    assert flagged == {k for k in DEAD_REWORK_SESSION_LANES if k.startswith(rel)}
