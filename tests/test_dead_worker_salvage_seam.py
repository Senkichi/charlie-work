"""Issue #2262: pin every dead-worker requeue locus to the shared
"dead worker with commits" salvage seam.

A dead-worker requeue locus is a production module that hands a dead
worker's issue back to dispatch — i.e. one that emits the
``session_failed_relabeled`` event. The locus set is DERIVED by AST scan
of ``src/charlie_work/**``, never hardcoded: any module that

- contains an ``ast.Constant`` equal to ``"session_failed_relabeled"``
  (``emit(...)``, ``append_event(...)``, ``DriftItem(kind=...)``), or
- calls one of the relabeled-event emitters
  (``_emit_session_failed_relabeled``, ``session_failed_relabeled_payload``,
  ``_session_failed_relabeled_payload``)

is a locus. Modules that DEFINE the emitters are the seam plumbing itself
and are exempt (derivable exemption, not an allow-list).

Every locus must reference the shared salvage seam — a name from the
``_attempt_salvage`` → ``park_unpublishable_work`` / local-park family
(including the ``ParkOrReclaim`` sweep request the decide phase uses to
reach ``park_or_reclaim_local_orphan``). A fourth locus that requeues a
dead worker without consulting the seam fails this test — that is exactly
the regression this issue fixes, so the pin must not be softened by
hardcoding today's module set.

Derived locus set at the time this test was added (for the reader only —
the test does not use it):
``orchestration/misc_worker_dispatch.py`` (phantom-live-worker route),
``dead_worker_sweep/dead_sessions_reclaim.py`` (sidecar reaper),
``dead_worker_sweep/decide_no_pr.py`` (state/PID orphan sweep), and
``reconcile.py`` (operator mop-up lane).
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "charlie_work"

_REQUEUE_EVENT = "session_failed_relabeled"

# Function-level emitters of the requeue event. A module DEFINING one of
# these is the seam plumbing, not a locus.
_EMITTER_DEFS = frozenset(
    {
        "_emit_session_failed_relabeled",
        "_session_failed_relabeled_payload",
        "session_failed_relabeled_payload",
    }
)

# The shared "dead worker with commits" seam family: the probe+publish
# helper, the publish tail it calls, the local-park lane members, and the
# sweep's ParkOrReclaim request that reaches them. A locus referencing ANY
# of these consults the seam before requeueing.
_SALVAGE_SEAM_NAMES = frozenset(
    {
        "salvage_dead_worker_commits",
        "park_salvageable_local_orphan",
        "park_or_reclaim_local_orphan",
        "park_unpublishable_work",
        "park_labelless_dead_local_session",
        "park_backstop_due_local_orphans",
        "_attempt_salvage",
        "ParkOrReclaim",
        "publishes_pull_requests",
    }
)


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _module_facts(tree: ast.Module) -> tuple[bool, bool, set[str]]:
    """Return (is_producer, is_definer, referenced_names) for one module."""
    is_producer = False
    is_definer = False
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == _REQUEUE_EVENT:
            is_producer = True
        elif isinstance(node, ast.Call):
            if _call_name(node.func) in _EMITTER_DEFS:
                is_producer = True
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _EMITTER_DEFS:
                is_definer = True
        elif isinstance(node, ast.alias):
            names.add(node.name)
    return is_producer, is_definer, names


def _requeue_loci() -> dict[str, set[str]]:
    """Derive {module_relpath: referenced_names} for every requeue locus."""
    loci: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        is_producer, is_definer, names = _module_facts(tree)
        if is_producer and not is_definer:
            loci[str(path.relative_to(_SRC.parent.parent))] = names
    return loci


def test_derived_locus_set_is_nonempty() -> None:
    """The derivation must not silently produce an empty set — that would
    make the pin vacuous the day the emitters get renamed."""
    loci = _requeue_loci()
    assert loci, "no session_failed_relabeled loci found — the derivation is broken"


def test_every_dead_worker_requeue_locus_routes_through_salvage_seam() -> None:
    """Every module that can hand a dead worker's issue back to dispatch
    must consult the shared committed-work salvage seam."""
    offenders = {
        path for path, names in _requeue_loci().items() if not names & _SALVAGE_SEAM_NAMES
    }
    assert not offenders, (
        "dead-worker requeue loci that do not reference the shared salvage "
        f"seam ({sorted(_SALVAGE_SEAM_NAMES)}): {sorted(offenders)}. A locus "
        "that requeues without probing for committed work reintroduces the "
        "issue #2262 archive-the-work gap — route it through "
        "local_work_park.salvage_dead_worker_commits (or an equivalent member "
        "of the seam family) instead."
    )


def test_park_salvageable_local_orphan_delegates_to_shared_helper() -> None:
    """The no-PR-backend gate must stay a thin gate over the shared helper —
    the probe+publish tail must live in exactly one place."""
    path = _SRC / "local_work_park.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "park_salvageable_local_orphan"
    )
    calls_salvage = any(
        isinstance(node, ast.Call) and _call_name(node.func) == "salvage_dead_worker_commits"
        for node in ast.walk(fn)
    )
    probes_worktree = any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"inspect_worktree_state", "branch_diff_result"}
        for node in ast.walk(fn)
    )
    assert calls_salvage, (
        "park_salvageable_local_orphan must delegate to "
        "salvage_dead_worker_commits — the single 'dead worker with commits' "
        "seam (issue #2262)"
    )
    assert not probes_worktree, (
        "park_salvageable_local_orphan re-grew its own worktree probe — the "
        "probe lives in salvage_dead_worker_commits; duplicate probing is "
        "how the loci diverged in the first place"
    )
