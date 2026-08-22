"""W6 PR4 keystone AST enforcement test (issue #1264).

## What this enforces

Every ``OrchestratorApp``/module-level write in ``src/charlie_work`` goes
through one of the six gated primitives: ``state.save_state``,
``state.append_event``, the ``_record_event``-shaped forwarding wrapper
(``record_event`` on ``WriteGate``, ``_record_event`` pre-migration),
``instrumentation.log_event``, ``labels.transition``, and (issue #1264,
W6 PR3, R6a) ``process_utils.kill_orphan_pid`` via ``WriteGate.kill_process``.
``write_gate.py`` (W6 PR1) wraps all six behind ``WriteGate``, whose
``dry_run=True`` path performs zero writes, zero process kills, and zero
event emissions. The wave's PR2/PR3 migrated the stalled-review and
dead-worker clusters onto ``self.write_gate.<method>`` (Convention A, bound
``OrchestratorApp`` methods) or an explicit ``write_gate: WriteGate``
parameter validated by ``require_write_gate()`` (Convention B, free/module-
level functions) -- see ``write_gate.py``'s own docstring and issue #1264's
R1 ratification.

This scanner is the keystone that closes the wave. It walks every module
under ``src/charlie_work`` (dynamically discovered via ``rglob`` -- no
hardcoded module list) and classifies every call to one of the six gated
primitives into exactly one of four buckets:

  (a) a WriteGate method call (``self.write_gate.<m>(...)`` or the
      Convention B local ``write_gate.<m>(...)``) -- covered by construction;
  (b) inside ``write_gate.py`` itself -- the primitive's own forwarding
      wrapper body, not a mutator-layer call site;
  (c) inside the primitive's own defining module (``state.py``,
      ``instrumentation.py``, ``labels.py``, ``process_utils.py``) -- the
      primitive's own internals, not an external call site;
  (d) on the explicit, comment-justified allow-list below (the four
      deliberately-unconditional ``log_event`` observability sites in
      ``stalled_review_reap.py`` that issue #708/#734 require to stay raw --
      issue #1264's R4 disposition -- plus the permanent legacy-forwarding-
      wrapper exemption for ``_record_event``'s own body, see the comment on
      ``_LEGACY_FORWARDING_WRAPPER_SCOPES`` below).

Anything else in buckets (a)-(d) is accounted for; everything that remains
is scoped by issue #1264's R9 ruling (comment 5324595347 on #1264), which
supersedes the flat "zero raw calls anywhere" idea an earlier prototype used:

## R9: function-granular predicate + per-module shrink-only ratchet

A keystone run at base ``3c935ad`` inventoried 346 raw primitive calls: 4
R4-allowlisted, 25 in PR2/PR3's converted clusters, and the rest (285 in
workflow.py functions no wave PR converts, plus 32 across 8 other modules)
belonging to code this wave never touches. A flat "zero raw calls in the
tree" assertion is therefore unsatisfiable by construction -- the wave
converts two clusters, not the whole tree.

R9's derived, fail-closed predicate operates at FUNCTION granularity, not
module granularity (``workflow.py`` has imported ``write_gate`` since PR1's
wiring, so a module-level predicate collapses to whole-tree and
reintroduces the 285-site problem):

  **Any function that uses WriteGate at all -- a ``write_gate`` parameter
  (Convention B) or a DIRECT ``self.write_gate.<m>(...)``/``write_gate.<m>(...)``
  call in its own body (Convention A) -- must use it exclusively: zero
  raw (non-gate-covered) primitive calls inside that function, outside the
  R4 allow-list.** Mixed usage is precisely the defect class the wave's
  mutation probes catch. The predicate auto-tightens as every future
  conversion lands -- no hardcoded module or function list.

  Convention A requires WriteGate to be the *receiver of a call*, not
  merely referenced as a value -- see
  ``_function_own_body_uses_write_gate``'s docstring for the concrete case
  (``dispatch_reviews``, ``_loop_body``) that makes this distinction
  load-bearing rather than pedantic: both methods forward
  ``write_gate=self.write_gate`` to exactly one Convention-B helper call
  inside a much larger, pre-wave body, and a reference-based predicate
  would wrongly sweep that body's unrelated raw calls into the zero-
  tolerance bucket.

For everything OUTSIDE the predicate (a function that never directly calls
``write_gate`` at all), the keystone additionally asserts a **per-module
shrink-only count ratchet** on raw primitive calls: counts recorded per
module at this PR's base (``_RATCHET_BASELINE`` below, re-derived live
against the actual post-PR3 tree, not merely copied from the prototype's
pre-PR3 numbers) may only decrease or hold; any increase fails, naming the
module. This is issue #619's ratchet concept, narrowed to counts so it
never needs a site-by-site allow-list for the ~300 out-of-wave sites and
only trips when someone adds a NEW raw write to unconverted territory --
exactly the event it exists to catch. The full site inventory (workflow.py's
remaining functions; ``reconcile.py``; ``fleet_registry.py``; the #1324
(``_record_event`` raw body + ``_maybe_reconcile_drift``/
``_maybe_reclaim_superseded_main_ci`` site inventory), #1325
(``_detect_and_handle_stalled_sessions`` kill/write leak), #1326 (salvage-
path raw ``git push`` -- outside WriteGate's primitive set entirely), and
#1327 (``_deescalate_mechanical_issue`` state/label divergence) territories)
ships as the post-wave conversion backlog in the commit body, not as test
assertions.

**Closure scoping**: each function/method (``FunctionDef``/
``AsyncFunctionDef``) is its own predicate scope. A nested closure defined
inside a WriteGate-using function is judged independently -- if the closure
itself never references ``write_gate`` in its own body (not counting any
further-nested closures), a raw call inside it falls under the ratchet, not
the parent's exclusivity check. This mirrors ``walk()``'s existing
scope-rebinding-on-``FunctionDef`` traversal and prevents a closure's raw
call from being laundered into (or a real violation from being hidden by)
its parent's predicate membership.

## R10: ``_record_event``'s forwarding body

``OrchestratorApp._record_event`` (workflow.py) is the pre-migration
forwarding wrapper ``WriteGate.record_event`` replaces. Its own body does
exactly what ``WriteGate.record_event``'s body does -- a bare
``append_event`` call for the same forwarding reason bucket (b) already
exempts inside ``write_gate.py``. Per R10, this exemption is PERMANENT, not
a mid-wave convenience: PR4 does not delete or convert ``_record_event`` --
that is issue #1324's post-wave, single-point-of-enforcement-at-the-sink
work. ``test_legacy_forwarding_wrapper_scopes_are_not_stale`` below keeps
the exemption honest: it must always cover a real, presently-raw call site,
or the entry (and/or the #1324 follow-up) has gone stale.

## R11: kill-primitive vocabulary and its limits

The primitive vocabulary includes ``kill_orphan_pid`` and ``kill_process``
(``WriteGate.kill_process`` wraps ``process_utils.kill_orphan_pid``, added
by W6 PR3 per R6a); ``process_utils.py`` is a primitive-defining module
(bucket c), so its internal ``taskkill``/``os.kill`` spellings are exempt as
implementation. Raw ``os.kill``/``taskkill``/``kill_process_tree`` calls
ELSEWHERE in the tree are structurally invisible to a name-matching
scanner -- documented as a stated limit, not silently: the known instance
(``workflow.py``'s ``_detect_and_handle_stalled_sessions``) lives in issue
#1325's territory, which the R9 predicate excludes (that function never
references ``write_gate``) and the ratchet covers. Widening the vocabulary
to syscall-level spellings is #619-residual scope, not PR4.

## Closed gap: ``dispatch_reviews`` (issue #1329)

Deriving R9's predicate against the real post-PR3 tree surfaced 10 raw
primitive calls inside ``OrchestratorApp.dispatch_reviews``
(``workflow.py:10937,10958,11134,11143,11199,11213,11237,11262,11363,11373``)
that R9's exclusive-use rule correctly flagged: the function also makes real
``self.write_gate.append_event(...)``/``self.write_gate.save_state(...)``
calls at ``workflow.py:11582,11596,11605``, converted by W6 PR2 with an
inline comment explaining exactly why -- that block sits under the same
``if self.dry_run:`` early return at ``workflow.py:10826`` (issue #617) as
all ten raw sites above it, and PR2's own stated rationale for converting it
("Gate-internal checking makes that protection structural instead of
relying on the distant guard staying correct (R7)") applies identically to
all ten. PR2 converted the one block and left the other ten raw -- a
genuine partial conversion, not a scanner artifact.

This keystone deliberately did **not** add those ten sites to
``_ALLOWED_RAW_PRIMITIVE_SITES``. The R4 allow-list is for call sites that
are correct as designed (the stalled-review sites must always fire
regardless of ``dry_run`` for observability reasons issue #708/#734
require); these ten were not that -- they were an incomplete conversion, and
exempting them would have laundered a live gap through the same mechanism
meant for deliberate design choices, defeating the exact "mixed usage"
defect class R9 exists to catch. Filed as issue #1329 with the full
guard-chain trace, and flagged in the PR4 handoff report as a
needs-adjudication finding.

Operator ruling R12 (issue #1264 comment 5325295148) amended PR4's scope to
include the fix: convert all ten sites to ``self.write_gate.*``, mirroring
PR2's R7 conversion shape exactly (drop the ``state_path=``/``repo=``
arguments the gate auto-binds; every other argument and all state-threading
reassignments preserved verbatim). That conversion is included in this PR.
``test_write_gate_no_unaccounted_raw_primitive_calls`` is green against the
current tree as a result -- this section is kept as the record of how the
gap was found, why it was refused an allow-list exemption, and how it was
closed, per issue #1329 (left open for the operator's post-merge completion
comment and manual close, not closed by this commit).

## Structural anchors, not line numbers

Every site (and every allow-list entry) is keyed by ``(module path,
enclosing scope, primitive name, unparsed call source, occurrence index)``
-- never a bare line number, which rots as the surrounding file edits. The
first four fields mirror ``tests/test_instrumentation.py``'s own
``_UnresolvedKindSite.key`` idiom exactly. The fifth, ``occurrence``, is
required because calling the same primitive with byte-identical arguments
multiple times in one function is the dominant real shape (e.g.
``save_state(self.paths.state_file, state)``, repeated verbatim many times
per function) -- without it, one allow-list entry would silently exempt
every site with that shape, not the single site it names.

``test_write_gate_allowlist_entries_are_not_stale`` enforces the same
symmetric direction ``test_event_kind_registry_exhaustive`` does for its own
allow-list: an allow-list entry (or the legacy-forwarding-wrapper exemption)
that no longer matches a real site is a silent hole and must fail the build
too, not just accumulate as dead weight.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).parents[1] / "src" / "charlie_work"

# The gated-mutator-layer's fixed primitive vocabulary. Includes BOTH
# `_record_event` (the pre-migration `OrchestratorApp` private forwarding
# method) and `record_event` (WriteGate's replacement method name) --
# mirroring `tests/test_instrumentation.py`'s own
# `_EMIT_FUNCS = {"log_event", "append_event", "_record_event", "record_event"}`,
# which already treats these two names as the same underlying "record an
# event" shape for its own (different) purpose. Matching is receiver-
# agnostic: a bare `Name` or an `Attribute.attr`, same as that scanner's
# `_emit_func_name`.
#
# `kill_orphan_pid` + `kill_process` (issue #1264 R11): the 6th WriteGate
# method, added by W6 PR3. `kill_orphan_pid` is the raw primitive
# (process_utils.py); `kill_process` is WriteGate's wrapper method name --
# both are matched the same receiver-agnostic way the other primitive/
# wrapper name pairs already are.
_GATED_PRIMITIVE_NAMES = {
    "save_state",
    "append_event",
    "_record_event",
    "record_event",
    "log_event",
    "transition",
    "kill_orphan_pid",
    "kill_process",
}

# Bucket (c): modules that ARE a gated primitive's own home. A call made
# from inside one of these files is the primitive's own internals (e.g.
# `state.append_event`'s own body calling `instrumentation.log_event`), not
# a mutator-layer call site that needs to route through WriteGate.
# `process_utils.py` added per R11: it defines `kill_orphan_pid`, whose body
# makes the real `taskkill`/`os.kill` calls this scanner's name vocabulary
# cannot see (documented limit, see module docstring's R11 section).
_PRIMITIVE_DEFINING_MODULES = {"state.py", "instrumentation.py", "labels.py", "process_utils.py"}

# Bucket (b): write_gate.py's own forwarding-wrapper bodies. The whole file
# is exempt rather than only the six wrapper methods individually --
# write_gate.py contains nothing else (the `WriteGate` dataclass's six
# methods plus `require_write_gate`), so a module-level exemption and a
# method-level one are equivalent in practice and the module-level check is
# simpler to state and verify.
_WRITE_GATE_MODULE = "write_gate.py"

# Extra bucket-(b)-shaped exemption, kept separate from bucket (d)'s
# explicit allow-list below because it exempts a *scope's own body*
# (structurally identical to why write_gate.py's wrappers are exempt), not
# an individual deliberately-raw call site.
#
# `OrchestratorApp._record_event` (workflow.py) is the PRE-migration
# forwarding wrapper `WriteGate.record_event` replaces -- see write_gate.py's
# own docstring: "the `_record_event`-shaped forwarding wrapper
# (`record_event` here)". Its own body does exactly what
# `WriteGate.record_event`'s body does: `return append_event(state, kind,
# payload, state_path=self.paths.state_file, repo=self.repo_root.name,
# level=level)` -- a bare `append_event` call for the same forwarding
# reason bucket (b) already exempts inside write_gate.py.
#
# PERMANENT per issue #1264 R10 (comment 5324595347): PR4 does NOT delete or
# convert `_record_event` -- that is issue #1324's post-wave,
# single-point-of-enforcement-at-the-sink work, and folding it into PR4
# would breach the wave's scope discipline. This is not a mid-wave
# convenience with an expiration; `test_legacy_forwarding_wrapper_scopes_are_not_stale`
# below is what keeps it honest for as long as `_record_event` exists.
_LEGACY_FORWARDING_WRAPPER_SCOPES = {("workflow.py", "_record_event")}


def _is_exempt_module(rel_path: str) -> bool:
    """Bucket (b) + (c): whole-module exemptions."""
    return rel_path == _WRITE_GATE_MODULE or rel_path in _PRIMITIVE_DEFINING_MODULES


def _primitive_name(node: ast.Call) -> str | None:
    """Receiver-agnostic name match, mirroring `_emit_func_name` in
    tests/test_instrumentation.py: a bare `Name` or an `Attribute.attr`."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in _GATED_PRIMITIVE_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _GATED_PRIMITIVE_NAMES:
        return func.attr
    return None


def _bare_primitive_name(expr: ast.expr) -> str | None:
    """If `expr` is a bare `Name` or `Attribute` whose id/attr is one of the
    six gated primitives, return that primitive name. Otherwise None.

    This is the alias-source detector for issue #1374: a parameter DEFAULT
    or an Assign VALUE that is a bare primitive reference (e.g.
    `log_event_fn=log_event` or `_LOG = instrumentation.log_event`) makes
    the parameter/variable name an alias for that primitive. Receiver-
    agnostic, same shape as `_primitive_name` but applied to default/value
    expressions rather than call func expressions."""
    if isinstance(expr, ast.Name) and expr.id in _GATED_PRIMITIVE_NAMES:
        return expr.id
    if isinstance(expr, ast.Attribute) and expr.attr in _GATED_PRIMITIVE_NAMES:
        return expr.attr
    return None


def _collect_function_param_aliases(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Issue #1374: return a mapping of parameter name -> underlying
    primitive name for parameters whose DEFAULT is a bare Name/Attribute
    matching a gated primitive.

    The injectable-default-for-testability pattern
    (`def f(emit=log_event): emit(...)`) is GOOD style this repo actively
    uses, but it is also the pattern that silently exits the R9 ratchet:
    the scanner's `_primitive_name` matches by literal name only, so a call
    through the parameter name is invisible. This mapping lets the walk
    resolve such calls to the underlying primitive and classify them into
    the same four buckets.

    Convention B `write_gate` parameters are NOT aliases (they have their
    own lane) and are excluded by construction: 'write_gate' is not in
    `_GATED_PRIMITIVE_NAMES`, so `_bare_primitive_name` returns None for it
    regardless of its default.

    Alias resolution stays local and static -- one function scope, default
    expression only, no dataflow analysis. A parameter reassigned inside
    the body to something else is still treated as an alias (fail-closed:
    a false positive is caught by the ratchet and dispositioned, a false
    negative is the silent hole this scanner exists to prevent)."""
    aliases: dict[str, str] = {}
    args = func_node.args
    # Positional defaults align with the TAIL of posonlyargs+args.
    posargs = [*args.posonlyargs, *args.args]
    paired = zip(posargs[-len(args.defaults) :], args.defaults)
    for arg, default in paired:
        prim = _bare_primitive_name(default)
        if prim is not None:
            aliases[arg.arg] = prim
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is None:
            continue
        prim = _bare_primitive_name(default)
        if prim is not None:
            aliases[arg.arg] = prim
    return aliases


def _collect_module_aliases(tree: ast.Module) -> dict[str, str]:
    """Issue #1374: return a mapping of variable name -> underlying
    primitive name for module-level `Assign` statements whose value is a
    bare Name/Attribute matching a gated primitive.

    The module-level alias pattern (`_LOG = log_event` followed by
    `_LOG(...)`) is the other obvious spelling of the injectable-default
    escape, and gets the same treatment. Only simple `Name` targets are
    considered (no tuple unpacking, no attribute targets) -- a complex
    target is not the "obvious spelling" the issue names, and adding it
    would be speculative scope. Only top-level statements are scanned: an
    alias assigned inside a function body is a local variable, not a
    module alias, and is invisible to this pass by design (no dataflow
    analysis)."""
    aliases: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            prim = _bare_primitive_name(stmt.value)
            if prim is None:
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = prim
    return aliases


def _is_write_gate_receiver(value: ast.expr) -> bool:
    """True if `value` statically looks like a WriteGate instance.

    Two real shapes in this codebase: Convention A `self.write_gate.<m>(...)`
    (an `Attribute` chain whose value is itself an `Attribute` named
    `write_gate`), and Convention B `write_gate.<m>(...)` (a bare `Name`
    called `write_gate`, the local parameter after
    `write_gate = require_write_gate(write_gate)`).

    Purely structural (name-based), not real type resolution -- a variable
    named `write_gate` that isn't actually a `WriteGate` instance would
    false-negative; no such site exists in this codebase today.
    """
    if isinstance(value, ast.Name) and value.id == "write_gate":
        return True
    if isinstance(value, ast.Attribute) and value.attr == "write_gate":
        return True
    return False


def _is_gate_covered_call(node: ast.Call) -> bool:
    """Bucket (a): the call is `<write-gate-shaped-receiver>.<primitive>(...)`."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return _is_write_gate_receiver(func.value)


def _calls_write_gate_directly(node: ast.AST) -> bool:
    """True if `node` is a call whose receiver is write-gate-shaped --
    `self.write_gate.<m>(...)` or `write_gate.<m>(...)` -- for ANY method
    name, not just the six gated primitives (future-proof against WriteGate
    growing new methods).

    Deliberately narrower than "any reference to write_gate anywhere in the
    body": a function that merely forwards `write_gate=self.write_gate` as
    an argument to another call is not itself making a gate-routed write --
    it is plumbing. See `_function_own_body_uses_write_gate`'s docstring for
    why that distinction matters.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_write_gate_receiver(node.func.value)
    )


def _function_own_body_uses_write_gate(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """R9 predicate: does this function use WriteGate at all, in its OWN
    body -- a `write_gate` parameter (Convention B) or a DIRECT
    `self.write_gate.<m>(...)`/`write_gate.<m>(...)` call (Convention A)?

    Convention A deliberately requires WriteGate to be the *receiver of a
    call*, not merely referenced as a value. The first implementation of
    this predicate matched any `self.write_gate`/`write_gate.*` reference
    anywhere in the body, including as a keyword-argument VALUE. Live-
    deriving the ratchet baseline against the post-PR3 tree surfaced why
    that is wrong: `dispatch_reviews` and `_loop_body` are large,
    multi-purpose `OrchestratorApp` methods that each forward
    `write_gate=self.write_gate` to exactly ONE Convention-B helper call
    buried inside a much larger body that also contains long-standing,
    pre-wave raw calls (quota-alert bookkeeping, pass-level telemetry)
    entirely unrelated to that one forwarded call. Treating the forward
    itself as Convention A usage would sweep ~13 of those unrelated,
    pre-existing raw calls into the in-predicate zero-tolerance bucket --
    not the "mixed usage" defect class R9 describes (a function whose OWN
    gate-routed writes sit next to raw ones it should also have converted),
    but a false-positive-shaped result that would force either a forbidden
    new conversion of both functions in full, or a pile of allow-list
    entries with no real justification. Requiring WriteGate to be the call
    receiver -- the same shape `_is_gate_covered_call` already recognizes
    for bucket (a) -- fixes this without weakening detection of real mixed
    usage: `test_scanner_flags_mixed_usage_inside_a_write_gate_function_as_a_violation`
    (a direct `self.write_gate.append_event(...)` call) and the Convention
    B parameter tests still pass unchanged, and
    `test_scanner_does_not_treat_write_gate_argument_forwarding_as_convention_a`
    pins the corrected forwarding behavior.

    Deliberately does NOT descend into nested `FunctionDef`/
    `AsyncFunctionDef` nodes: a nested closure is its own predicate scope
    (see module docstring, "Closure scoping"). A closure that itself
    references `write_gate` is judged independently by a separate call to
    this same function against the closure's own node.
    """
    all_args = [
        *func_node.args.posonlyargs,
        *func_node.args.args,
        *func_node.args.kwonlyargs,
    ]
    if any(a.arg == "write_gate" for a in all_args):
        return True

    def scan(node: ast.AST) -> bool:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested closures are their own scope; do not descend
            if _calls_write_gate_directly(child):
                return True
            if scan(child):
                return True
        return False

    return scan(func_node)


@dataclass(frozen=True)
class _RawPrimitiveSite:
    """One call to a gated primitive not routed through WriteGate.

    Structural anchor -- (path, scope, primitive, unparsed call source,
    occurrence) -- never a bare line number. `lineno` is retained for
    human-readable reporting only and deliberately excluded from `key`,
    mirroring `test_instrumentation.py`'s `_UnresolvedKindSite`.

    `occurrence` (0-indexed, in source/traversal order) is REQUIRED for real
    exact-siteness: the same primitive is routinely invoked with
    byte-identical arguments multiple times in one function (the dominant
    real shape: `save_state(self.paths.state_file, state)`, where `state` is
    reassigned between calls but always referenced by the same name, so the
    call expression's own text never changes). Without `occurrence`, one
    allow-list entry would silently cover every call with that shape in that
    scope, not the single site it was written to justify. It survives
    incidental drift elsewhere in the file; it does NOT survive a
    same-shaped call being added, removed, or reordered earlier in the same
    scope, which shifts every later occurrence's index -- a residual
    limitation, not a solved problem.

    `in_predicate` (issue #1264 R9, NOT part of `key`): True when this
    site's enclosing scope itself uses WriteGate (a `write_gate` parameter
    or `self.write_gate`/`write_gate.*` reference anywhere in that scope's
    own body, not counting nested closures) -- i.e. this site is subject to
    the function-granular exclusive-use predicate and must be accounted for
    on the R4 allow-list, not merely shrink-ratcheted. Excluded from `key`
    because it is a derived classification of the site, not part of its
    structural identity; two scans of the same tree always agree on it for
    a given site, so allow-list matching (which uses `key`) doesn't need it.
    """

    path: str  # POSIX-relative to src/charlie_work
    scope: str  # enclosing function/method name, or "<module>"
    primitive: str
    call_source: str  # ast.unparse() of the full Call node
    occurrence: int = 0  # 0-indexed rank among identical (path, scope, primitive, call_source)
    lineno: int = 0
    in_predicate: bool = False

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (self.path, self.scope, self.primitive, self.call_source, self.occurrence)


def _scan_module_for_raw_primitive_calls(
    tree: ast.Module | ast.AST, rel_path: str, *, apply_legacy_exemption: bool = True
) -> list[_RawPrimitiveSite]:
    """Full-tree walk (not just direct children) for one already-parsed
    module. Exposed separately from the tree-wide scan so the seeded-
    violation self-tests below can feed it a synthetic module without
    writing a file to disk. The full-tree-walk technique itself mirrors
    `test_rework_prompts_split.py`'s AC9 scanner (a write can be several
    frames of nesting below the function's top level, e.g. inside an
    `if`/`for`/`with` block; only a full walk, not direct children, finds
    it -- issue #619's limit 3).

    `apply_legacy_exemption` defaults on (matches `_scan_gated_mutator_layer`'s
    real usage). It exists so `test_legacy_forwarding_wrapper_scopes_are_not_stale`
    can re-scan a module WITHOUT the exemption applied and check that a real
    site still exists underneath it -- the same "would this entry ever fail"
    discipline `test_write_gate_allowlist_entries_are_not_stale` applies to
    `_ALLOWED_RAW_PRIMITIVE_SITES`, applied here to the other exemption set.

    Each site's `in_predicate` (R9) is resolved from its innermost enclosing
    FunctionDef/AsyncFunctionDef's own body (see
    `_function_own_body_uses_write_gate`); a site directly at module level
    (`scope == "<module>"`) is never in-predicate -- there is no enclosing
    function to carry a `write_gate` parameter or reference.

    Issue #1374 alias resolution: a call through a parameter name whose
    DEFAULT is a bare primitive (`def f(emit=log_event): emit(...)`) or
    through a module-level alias (`_LOG = log_event; _LOG(...)`) is resolved
    to the underlying primitive and classified into the same four buckets.
    Alias resolution is local and static -- one function scope for param
    aliases (a nested closure does NOT inherit the parent's param aliases,
    mirroring the closure-scoping rule for `in_predicate`), default
    expression only, no dataflow analysis. Module-level aliases apply in
    every scope (a local param alias shadows a module alias of the same
    name, matching Python's own scoping). Fail-closed: an alias call that
    cannot be dispositioned lands in the unaccounted bucket and trips the
    ratchet, same as a directly-named raw call.
    """
    sites: list[_RawPrimitiveSite] = []
    occurrence_counts: dict[tuple[str, str, str], int] = {}
    # Per-module cache: unqualified scope name -> does that scope's own body
    # use WriteGate. A collision between two functions sharing a name within
    # the SAME module (e.g. two methods of different classes) is a known,
    # documented residual limit of the unqualified-scope-name design this
    # scanner inherits from test_instrumentation.py's own idiom -- not
    # something R9 asked PR4 to solve.
    scope_uses_write_gate: dict[str, bool] = {}
    # Issue #1374: per-scope param-alias map (param name -> primitive name).
    # Keyed by the same unqualified scope name as scope_uses_write_gate and
    # subject to the same collision limit.
    scope_param_aliases: dict[str, dict[str, str]] = {}
    # Issue #1374: module-level alias map (var name -> primitive name),
    # computed once from top-level Assign statements. Applies in every
    # scope; a local param alias of the same name takes precedence.
    module_aliases: dict[str, str] = (
        _collect_module_aliases(tree) if isinstance(tree, ast.Module) else {}
    )

    def _resolve_call_primitive(node: ast.Call, scope_name: str) -> str | None:
        """Direct name match first (the existing scanner path); if that
        misses, alias resolution for issue #1374 -- a bare-Name call whose
        id is a param alias for this scope or a module-level alias."""
        prim = _primitive_name(node)
        if prim is not None:
            return prim
        func = node.func
        if not isinstance(func, ast.Name):
            return None
        # Scope param aliases take precedence (a local param shadows a
        # module-level alias of the same name, matching Python scoping).
        param_aliases = scope_param_aliases.get(scope_name)
        if param_aliases is not None and func.id in param_aliases:
            return param_aliases[func.id]
        if func.id in module_aliases:
            return module_aliases[func.id]
        return None

    def walk(node: ast.AST, scope_name: str) -> None:
        if isinstance(node, ast.Call):
            primitive = _resolve_call_primitive(node, scope_name)
            if primitive is not None and not _is_gate_covered_call(node):
                exempt = (
                    apply_legacy_exemption
                    and (
                        rel_path,
                        scope_name,
                    )
                    in _LEGACY_FORWARDING_WRAPPER_SCOPES
                )
                if not exempt:
                    call_source = ast.unparse(node)
                    shape_key = (scope_name, primitive, call_source)
                    occurrence = occurrence_counts.get(shape_key, 0)
                    occurrence_counts[shape_key] = occurrence + 1
                    sites.append(
                        _RawPrimitiveSite(
                            path=rel_path,
                            scope=scope_name,
                            primitive=primitive,
                            call_source=call_source,
                            occurrence=occurrence,
                            lineno=node.lineno,
                            in_predicate=scope_uses_write_gate.get(scope_name, False),
                        )
                    )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Entering a new function scope: unqualified enclosing
            # function/method name, no class-qualification (two methods on
            # different classes sharing a name produce the same scope
            # string -- documented limit, see above).
            scope_uses_write_gate[node.name] = _function_own_body_uses_write_gate(node)
            scope_param_aliases[node.name] = _collect_function_param_aliases(node)
            for child in ast.iter_child_nodes(node):
                walk(child, node.name)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, scope_name)

    walk(tree, "<module>")
    return sites


def _scan_gated_mutator_layer(root: Path) -> list[_RawPrimitiveSite]:
    """Walk every `.py` module under `root` (dynamically discovered, no
    hardcoded module list); return every raw (un-gated) call to a gated
    primitive outside write_gate.py and the primitives' own defining
    modules."""
    sites: list[_RawPrimitiveSite] = []
    for path in sorted(root.rglob("*.py")):
        rel_path = path.relative_to(root).as_posix()
        if _is_exempt_module(rel_path):
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        sites.extend(_scan_module_for_raw_primitive_calls(tree, rel_path))
    return sites


def _ratchet_violations(actual_counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """R9's per-module shrink-only ratchet, as a pure function of two count
    dicts -- factored out so the mechanics (hold/shrink pass, increase
    fails) can be unit-tested with synthetic data, independent of the real
    tree. A module present in `actual_counts` but absent from `baseline` is
    compared against an implicit baseline of 0 (a brand-new raw site in
    previously-clean territory is exactly the event this ratchet exists to
    catch). Returns the sorted list of module names whose count increased,
    empty when the ratchet holds everywhere."""
    violations = []
    for module, actual in actual_counts.items():
        if actual > baseline.get(module, 0):
            violations.append(module)
    return sorted(violations)


# ---------------------------------------------------------------------------
# Bucket (d): explicit, comment-justified allow-list of raw sites.
#
# Every entry must be independently justified and traceable to a binding
# decision -- this is not a shrink-only ratchet or a place to quietly
# accumulate exceptions. `test_write_gate_allowlist_entries_are_not_stale`
# enforces the symmetric direction: an entry that stops matching a real
# site fails the build too.
# ---------------------------------------------------------------------------
_ALLOWED_RAW_PRIMITIVE_SITES: tuple[_RawPrimitiveSite, ...] = (
    # Issue #1264 R4 / #708: recovery declined to act because the prompt
    # path is missing from state. Deliberately unconditional -- routing
    # through WriteGate would newly suppress this observability event under
    # dry_run=True, losing a signal the issue's own text says must survive
    # every skip path.
    _RawPrimitiveSite(
        path="stalled_review_reap.py",
        scope="_detect_and_handle_stalled_reviews",
        primitive="log_event",
        call_source=(
            "log_event(state_file, 'review_stale_claim_recovery_skipped', "
            "{'pr_number': int(pr_key) if pr_key.isdigit() else None, "
            "'reason': 'prompt_path missing from state'}, level='warning')"
        ),
        in_predicate=True,
    ),
    # Issue #1264 R4 / #708: same pattern, prompt file missing on disk.
    _RawPrimitiveSite(
        path="stalled_review_reap.py",
        scope="_detect_and_handle_stalled_reviews",
        primitive="log_event",
        call_source=(
            "log_event(state_file, 'review_stale_claim_recovery_skipped', "
            "{'pr_number': int(pr_key) if pr_key.isdigit() else None, "
            "'reason': 'prompt_path file does not exist on disk', "
            "'prompt_path': prompt_path_str}, level='warning')"
        ),
        in_predicate=True,
    ),
    # Issue #1264 R4 / #734: decision already recorded (second of three
    # silent skip paths the issue requires to stay observable).
    _RawPrimitiveSite(
        path="stalled_review_reap.py",
        scope="_detect_and_handle_stalled_reviews",
        primitive="log_event",
        call_source=(
            "log_event(state_file, 'review_stale_claim_recovery_skipped', "
            "{'pr_number': int(pr_key) if pr_key.isdigit() else None, "
            "'reason': 'decision_already_recorded', 'decision': decision_value}, "
            "level='warning')"
        ),
        in_predicate=True,
    ),
    # Issue #1264 R4 / #734: packet not stale yet (third of three silent
    # skip paths).
    _RawPrimitiveSite(
        path="stalled_review_reap.py",
        scope="_detect_and_handle_stalled_reviews",
        primitive="log_event",
        call_source=(
            "log_event(state_file, 'review_stale_claim_recovery_skipped', "
            "{'pr_number': int(pr_key) if pr_key.isdigit() else None, "
            "'reason': 'packet_not_stale', 'packet_age': packet_age}, level='info')"
        ),
        in_predicate=True,
    ),
    # Issue #1363: preflight non-fatal-failure warning emission, added
    # inside `_loop_impl` (workflow.py) alongside that same function's two
    # long-standing raw `log_event` calls for `loop_started`/`loop_completed`.
    # `tests/test_write_gate_dry_run_loop.py`'s module docstring documents
    # those two as deliberate, orthogonal pass-level telemetry that fires
    # "on every pass, dry-run or not" and states plainly that "none of the
    # wave's clusters ever targeted this wrapper -- it is not a 'caller
    # migrated onto WriteGate' in C1.2's sense." This third `log_event` call
    # is the same kind of telemetry for the same function, by the same
    # design: a preflight warning (config gone stale, clock skew) must
    # survive `dry_run=True` exactly like the sibling calls it sits next to,
    # so routing it through `self.write_gate.log_event(...)` would be wrong
    # twice over -- it would newly suppress the warning under dry-run
    # (contradicting the documented invariant), and it would flip
    # `_loop_impl` into R9's in-predicate exclusive-use bucket, which would
    # then flag its two untouched sibling `log_event` calls as violations
    # too, forcing an out-of-scope conversion of telemetry this wave never
    # targeted. `in_predicate=False` here matches the real scan (the
    # function makes no direct `self.write_gate.*`/`write_gate.*` call), so
    # this entry is not an R9-exclusivity exemption -- it keeps this
    # genuinely-new, by-design raw site off the per-module shrink-only
    # ratchet the same way the ratchet's own ~1160-1162 sentence intends for
    # sites that are correct as designed rather than an unconverted debt.
    _RawPrimitiveSite(
        path="workflow.py",
        scope="_loop_impl",
        primitive="log_event",
        call_source=(
            "log_event(self.paths.state_file, kind, {'check': check.name, "
            "'detail': check.detail}, repo=self.repo_root.name, level='warning')"
        ),
        in_predicate=False,
    ),
)


# ---------------------------------------------------------------------------
# R9's per-module shrink-only ratchet baseline.
#
# Derived LIVE against this PR's actual base (post-PR3,
# fd1fae6e1e468d92b6c59c0255ba0db26fa60512, re-verified after the rebase onto
# the merged main SHA) by running _scan_gated_mutator_layer(_SRC_ROOT) and
# counting sites where in_predicate is False and the site is not on the R4
# allow-list, grouped by module. This is NOT copied from the prototype's
# pre-PR3 346-site inventory -- that count included the 25 sites PR3's
# conversion removed from this ratchet's scope entirely (they are now
# in-predicate, gate-routed, or on the allow-list). See this PR's commit
# body for the full per-module breakdown and the follow-up issue that owns
# each module's remaining backlog (#1324/#1325/#1326/#1327 for the named
# workflow.py lanes; reconcile.py and fleet_registry.py are pre-existing,
# out-of-wave raw sites with no #1264 sub-issue yet).
# ---------------------------------------------------------------------------
_RATCHET_BASELINE: dict[str, int] = {
    # Issue #1372: +3 log_event calls in fleet_loop's stale-entry handling
    # (stale detection warning, prune warning, and the existing lane-completed
    # events). These are out-of-wave raw sites in unconverted territory, same
    # class as the pre-existing 8 — the ratchet holds at the new count.
    "fleet_dispatch.py": 11,
    "fleet_registry.py": 1,
    "reconcile.py": 5,
    "state_migration.py": 1,
    "supervise.py": 11,
    "supervisor_lifecycle.py": 3,
    "workflow.py": 270,
    "worktree.py": 1,
}


def _format_inventory(sites: list[_RawPrimitiveSite]) -> str:
    by_path: dict[str, list[_RawPrimitiveSite]] = {}
    for site in sites:
        by_path.setdefault(site.path, []).append(site)
    lines = [f"{len(sites)} unaccounted raw primitive call(s) across {len(by_path)} module(s):"]
    for path in sorted(by_path):
        module_sites = sorted(by_path[path], key=lambda s: s.lineno)
        lines.append(f"  {path} ({len(module_sites)}):")
        for site in module_sites:
            lines.append(f"    :{site.lineno} {site.scope}() -- {site.primitive}(...)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-tests: prove the scanner DETECTS, not merely runs. Mirrors
# test_instrumentation.py's own regression-control idiom: a seeded-violation
# positive control paired with a gate-routed negative control, so a scanner
# that flagged *everything* (or *nothing*) would be caught by one half or
# the other.
# ---------------------------------------------------------------------------


def test_scanner_module_and_primitive_definitions_are_exempt() -> None:
    assert _is_exempt_module("write_gate.py")
    assert _is_exempt_module("state.py")
    assert _is_exempt_module("instrumentation.py")
    assert _is_exempt_module("labels.py")
    assert _is_exempt_module("process_utils.py")
    assert not _is_exempt_module("workflow.py")
    assert not _is_exempt_module("stalled_review_reap.py")


def test_scanner_detects_a_seeded_bare_primitive_call() -> None:
    """Seed a synthetic module with exactly one bare, un-gated call to a
    gated primitive and assert the scanner reports exactly that site --
    the task's explicit seeded-violation AC. A scanner that ran clean
    because it silently matched nothing would pass every other test in
    this file too; only this one proves detection actually fires."""
    source = textwrap.dedent(
        """
        def _do_something(state):
            state = append_event(state, "some_kind", {"x": 1})
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected exactly one seeded violation, found: {sites}"
    (site,) = sites
    assert site.primitive == "append_event"
    assert site.scope == "_do_something"
    assert site.path == "synthetic_fixture.py"
    assert not site.in_predicate, "a function with no write_gate reference is never in-predicate"
    assert site.key == (
        "synthetic_fixture.py",
        "_do_something",
        "append_event",
        "append_event(state, 'some_kind', {'x': 1})",
        0,  # first (and only) occurrence of this exact call shape in this scope
    )


def test_scanner_detects_a_seeded_violation_nested_several_frames_deep() -> None:
    """Issue #619 limit 3 ('can't see a write several frames below the
    guarded call'): the scanner must be a full-tree walk, not direct
    children only. Seed the bare call inside nested `if`/`for` blocks."""
    source = textwrap.dedent(
        """
        def _sweep(items, flag):
            for item in items:
                if flag:
                    with open("x") as _f:
                        save_state(path, {"item": item})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"nested seeded violation not detected: {sites}"
    assert sites[0].primitive == "save_state"
    assert sites[0].scope == "_sweep"


def test_scanner_assigns_distinct_occurrence_indices_to_identical_call_shapes() -> None:
    """The same primitive invoked with byte-identical arguments multiple
    times in one function (`save_state(self.paths.state_file, state)`
    repeated verbatim) is the dominant real shape, not an edge case.
    Without `occurrence` in the key, two genuinely different call sites
    collapse into one key and an allow-list entry silently exempts both --
    this proves they don't."""
    source = textwrap.dedent(
        """
        def _do_something(state):
            state = save_state(path, state)
            state["a"] = 1
            state = save_state(path, state)
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 2, f"expected two distinct call sites, found: {sites}"
    assert sites[0].call_source == sites[1].call_source, (
        "test premise requires byte-identical call text between the two sites"
    )
    assert {s.occurrence for s in sites} == {0, 1}, (
        f"identical-shaped calls must get distinct occurrence indices, got: {sites}"
    )
    assert len({s.key for s in sites}) == 2, (
        f"identical-shaped calls collapsed onto the same key: {sites}"
    )


@pytest.mark.parametrize(
    "source,description",
    [
        pytest.param(
            'def m(self, state):\n    self.write_gate.append_event(state, "k", {})\n',
            "convention-a-bound-method",
            id="convention-a",
        ),
        pytest.param(
            "def f(state, write_gate):\n"
            "    write_gate = require_write_gate(write_gate)\n"
            '    write_gate.append_event(state, "k", {})\n',
            "convention-b-free-function",
            id="convention-b",
        ),
        pytest.param(
            'def m(self):\n    self.write_gate.transition(gh, labels, 1, "e")\n',
            "convention-a-transition",
            id="convention-a-transition",
        ),
        pytest.param(
            "def f(pid, write_gate):\n"
            "    write_gate = require_write_gate(write_gate)\n"
            "    write_gate.kill_process(pid)\n",
            "convention-b-kill-process",
            id="convention-b-kill-process",
        ),
    ],
)
def test_scanner_does_not_flag_gate_routed_calls(source: str, description: str) -> None:
    """Negative control paired with the seeded-violation tests above: a call
    that IS correctly routed through WriteGate (either calling convention,
    including the R11 kill_process addition) must not be flagged. Without
    this, a scanner that flags every call by name regardless of receiver
    would still pass the seeded-violation tests (it flags real violations
    too) while producing false positives on every already-migrated site --
    this is the half that catches that failure mode."""
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")
    assert sites == [], f"gate-routed call incorrectly flagged ({description}): {sites}"


def test_scanner_flags_mixed_usage_inside_a_write_gate_function_as_a_violation() -> None:
    """R9's core function-granular predicate: a function that uses
    WriteGate (here, Convention A -- `self.write_gate.append_event`) but
    ALSO makes one raw, non-gate-covered call in the same body must have
    that raw call flagged as an in-predicate violation. This is exactly the
    'mixed usage' defect class R9 says the wave's mutation probes catch --
    a function is not exempt from the exclusive-use rule just because it
    also does some real gate-routed writes."""
    source = textwrap.dedent(
        """
        class OrchestratorApp:
            def m(self, state):
                self.write_gate.append_event(state, "k", {})
                state = save_state(path, state)
                return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected exactly the raw save_state call, found: {sites}"
    (site,) = sites
    assert site.primitive == "save_state"
    assert site.scope == "m"
    assert site.in_predicate, (
        "a raw call inside a function that also references self.write_gate "
        "must be classified in-predicate -- it is subject to the exclusive-use rule"
    )


def test_scanner_does_not_treat_write_gate_argument_forwarding_as_convention_a() -> None:
    """Real-tree-derived regression: `dispatch_reviews` and `_loop_body`
    each forward `write_gate=self.write_gate` to exactly one Convention-B
    helper call, inside a much larger body that also contains long-
    standing, pre-wave raw calls entirely unrelated to that one forwarded
    call. Forwarding a reference as an argument VALUE is plumbing, not a
    function directly making a gate-routed write -- it must not trip
    Convention A. Without this distinction, live-deriving the ratchet
    baseline against the post-PR3 tree found 13 in-predicate 'violations'
    that were really just pre-existing code sharing a body with one
    unrelated forwarded call -- see
    `_function_own_body_uses_write_gate`'s docstring."""
    source = textwrap.dedent(
        """
        class OrchestratorApp:
            def m(self, state):
                helper(state, write_gate=self.write_gate)
                state = save_state(path, state)
                return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected exactly the raw save_state call, found: {sites}"
    (site,) = sites
    assert site.primitive == "save_state"
    assert site.scope == "m"
    assert not site.in_predicate, (
        "forwarding self.write_gate as an argument to another call is not Convention A "
        "usage -- the raw call must be ratchet-only, not swept into the zero-tolerance bucket"
    )


def test_scanner_marks_convention_b_function_in_predicate_even_before_reassignment() -> None:
    """A Convention B function's `write_gate` PARAMETER (not just the later
    `require_write_gate(write_gate)`-reassigned local) is enough to put the
    whole function in-predicate -- the parameter itself is a reference to
    WriteGate usage, so a raw call appearing textually before the
    reassignment line is still correctly judged, not accidentally exempted
    by scanning order."""
    source = textwrap.dedent(
        """
        def f(state, write_gate):
            state = append_event(state, "k", {})
            write_gate = require_write_gate(write_gate)
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1
    assert sites[0].in_predicate, (
        "the write_gate parameter alone marks the whole function in-predicate, "
        "regardless of where inside the body the raw call textually appears"
    )


def test_scanner_does_not_launder_a_nested_closures_raw_call_into_the_parent_predicate() -> None:
    """Closure scoping: a nested function defined inside a WriteGate-using
    function is its OWN predicate scope. If the nested closure itself never
    references `write_gate`, a raw call inside it must NOT be judged
    against the parent's exclusive-use rule (that would either wrongly fail
    a parent that is otherwise perfectly gate-exclusive, or -- worse --
    silently launder a real violation into the parent's already-True
    in_predicate status without a corresponding gate-covered call excusing
    it). The raw call must instead be ratchet-only (in_predicate=False),
    attributed to the closure's own scope name."""
    source = textwrap.dedent(
        """
        class OrchestratorApp:
            def m(self, items):
                self.write_gate.append_event({}, "k", {})

                def _inner(item):
                    return save_state(path, item)

                return [_inner(i) for i in items]
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected exactly the nested raw save_state call: {sites}"
    (site,) = sites
    assert site.scope == "_inner", "the raw call must be attributed to the closure's own scope"
    assert not site.in_predicate, (
        "a closure that never itself references write_gate is out-of-predicate, "
        "even though its enclosing function uses write_gate"
    )


def test_scanner_recognizes_a_write_gate_referencing_closure_as_its_own_in_predicate_scope() -> (
    None
):
    """Symmetric case: a nested closure that DOES itself reference
    `write_gate` (captured from the enclosing Convention B function) is
    independently in-predicate -- a raw call inside it is judged against
    the CLOSURE's own exclusive-use rule, not exempted just because it is
    nested."""
    source = textwrap.dedent(
        """
        def f(items, write_gate):
            write_gate = require_write_gate(write_gate)

            def _inner(item):
                write_gate.append_event({}, "k", {})
                return save_state(path, item)

            return [_inner(i) for i in items]
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected exactly the raw save_state call inside _inner: {sites}"
    (site,) = sites
    assert site.scope == "_inner"
    assert site.in_predicate, (
        "a closure that itself references write_gate is in-predicate on its own terms"
    )


# ---------------------------------------------------------------------------
# Issue #1374: injectable-default-alias and module-level-alias resolution.
#
# The scanner's `_primitive_name` matches by literal name only, so a call
# through a parameter whose DEFAULT is a bare primitive
# (`def f(emit=log_event): emit(...)`) or through a module-level alias
# (`_LOG = log_event; _LOG(...)`) is invisible -- silently exiting the R9
# ratchet. These tests prove the alias-resolution extension actually fires.
# ---------------------------------------------------------------------------


def test_scanner_detects_injectable_default_alias_call() -> None:
    """Issue #1374 acceptance criterion 1: a function with a parameter
    whose DEFAULT is a bare primitive name, and a call through that
    parameter name, is counted as a raw call to the underlying primitive.
    This is the exact pattern `preflight.py`'s `emit_preflight_refusal`
    uses (`log_event_fn: Callable = log_event`), which the pre-fix scanner
    reported 0 sites for in the same pass it reported 271 for workflow.py.
    """
    source = textwrap.dedent(
        """
        def emit_preflight_refusal(state_path, check, *, log_event_fn=log_event):
            log_event_fn(state_path, "loop_refused_preflight", {"check": check.name})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"expected the alias call to be visible, found: {sites}"
    (site,) = sites
    assert site.primitive == "log_event", (
        "the alias call must be classified as the underlying primitive, not the alias name"
    )
    assert site.scope == "emit_preflight_refusal"
    assert site.call_source.startswith("log_event_fn("), (
        "call_source must be the actual call text (alias name), not the resolved primitive"
    )
    assert not site.in_predicate, (
        "a function whose only WriteGate-shaped reference is a primitive alias "
        "(not a write_gate param or self.write_gate call) is NOT in-predicate -- "
        "the alias is a raw primitive reference, not WriteGate usage"
    )


def test_scanner_detects_injectable_default_alias_keyword_only_param() -> None:
    """The alias pattern with a keyword-only parameter (the real
    `emit_preflight_refusal` shape uses `*, log_event_fn=log_event`)."""
    source = textwrap.dedent(
        """
        def f(state, *, emit=append_event):
            state = emit(state, "k", {})
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"kw-only alias not detected: {sites}"
    assert sites[0].primitive == "append_event"


def test_scanner_detects_injectable_default_alias_attribute_default() -> None:
    """A parameter whose default is a bare Attribute matching a primitive
    (e.g. `fn=instrumentation.log_event`) is also resolved -- the issue
    names 'Name/Attribute' for parameter defaults."""
    source = textwrap.dedent(
        """
        def f(state, fn=instrumentation.log_event):
            fn(state, "k", {})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"Attribute-default alias not detected: {sites}"
    assert sites[0].primitive == "log_event"


def test_scanner_detects_module_level_alias_call() -> None:
    """Issue #1374 acceptance criterion 2: a module-level
    `_ALIAS = log_event` followed by `_ALIAS(...)` is counted as a raw
    call to the underlying primitive. This is the other obvious spelling
    of the injectable-default escape."""
    source = textwrap.dedent(
        """
        _LOG = log_event

        def f(state_path):
            _LOG(state_path, "kind", {"x": 1})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"module-level alias call not detected: {sites}"
    (site,) = sites
    assert site.primitive == "log_event"
    assert site.scope == "f", "the call inside f() is attributed to f, not <module>"
    assert site.call_source.startswith("_LOG(")


def test_scanner_detects_module_level_alias_at_module_scope() -> None:
    """A module-level alias called at module scope (not inside a function)
    is attributed to `<module>` and is never in-predicate."""
    source = textwrap.dedent(
        """
        _SAVE = save_state
        _SAVE(path, {"init": True})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"module-scope alias call not detected: {sites}"
    (site,) = sites
    assert site.primitive == "save_state"
    assert site.scope == "<module>"
    assert not site.in_predicate


def test_scanner_detects_module_level_alias_with_attribute_value() -> None:
    """A module-level alias whose value is a bare Attribute matching a
    primitive (`_LOG = instrumentation.log_event`) is also resolved."""
    source = textwrap.dedent(
        """
        _LOG = instrumentation.log_event

        def f(state_path):
            _LOG(state_path, "kind", {})
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"Attribute-value module alias not detected: {sites}"
    assert sites[0].primitive == "log_event"


def test_scanner_does_not_flag_a_non_primitive_default_alias() -> None:
    """Negative control: a parameter whose default is NOT a gated
    primitive (e.g. `emit=print`) must not be resolved -- the scanner
    should report zero sites for a call through it."""
    source = textwrap.dedent(
        """
        def f(msg, emit=print):
            emit(msg)
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")
    assert sites == [], f"non-primitive default alias incorrectly flagged: {sites}"


def test_scanner_does_not_flag_a_module_level_alias_to_non_primitive() -> None:
    """Negative control: a module-level alias whose value is not a gated
    primitive must not be resolved."""
    source = textwrap.dedent(
        """
        _PRINT = print

        def f(msg):
            _PRINT(msg)
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")
    assert sites == [], f"non-primitive module alias incorrectly flagged: {sites}"


def test_scanner_param_alias_does_not_leak_into_nested_closure() -> None:
    """Closure scoping for aliases mirrors the closure-scoping rule for
    `in_predicate`: a nested closure does NOT inherit the parent's param
    aliases. If the closure's own params don't alias a primitive, a call
    through the parent's alias name inside the closure is NOT resolved --
    it is a free variable capture, which is outside the static, one-scope
    alias resolution the issue specifies (no dataflow analysis). This
    prevents a parent's `emit=log_event` from silently making a closure's
    `emit(...)` visible when the closure captured a differently-typed
    variable of the same name."""
    source = textwrap.dedent(
        """
        def outer(state, emit=log_event):
            def inner(emit=print):
                emit(state, "k", {})
            return inner
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")
    # inner's own `emit=print` default shadows outer's `emit=log_event`;
    # the call inside inner is through inner's `emit` (print), not a
    # primitive alias -- zero sites.
    assert sites == [], (
        f"parent param alias leaked into closure whose own param shadows it: {sites}"
    )


def test_scanner_module_alias_applies_inside_nested_closure() -> None:
    """Unlike param aliases (which are scope-local), a module-level alias
    applies in every scope, including nested closures -- the closure
    captures the module global, and the alias is a module-level fact, not
    a scope-local one."""
    source = textwrap.dedent(
        """
        _LOG = log_event

        def outer():
            def inner(state_path):
                _LOG(state_path, "k", {})
            return inner
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"module alias not detected inside nested closure: {sites}"
    (site,) = sites
    assert site.scope == "inner"
    assert site.primitive == "log_event"


def test_scanner_param_alias_shadows_module_alias() -> None:
    """A local param alias of the same name as a module alias takes
    precedence (matching Python's own scoping: a local variable shadows a
    module global). If the param aliases a DIFFERENT primitive than the
    module alias, the call is resolved to the param's primitive."""
    source = textwrap.dedent(
        """
        _FN = save_state

        def f(state, fn=append_event):
            state = fn(state, "k", {})
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 1, f"param-alias-shadow not resolved: {sites}"
    assert sites[0].primitive == "append_event", (
        "the local param alias (append_event) must shadow the module alias (save_state)"
    )


def test_scanner_alias_call_in_predicate_function_is_in_predicate() -> None:
    """If a function that uses WriteGate (has a `write_gate` param) ALSO
    has a primitive-alias param and calls through it, the alias call is
    in-predicate -- it is a raw call inside a WriteGate-using function,
    subject to the exclusive-use rule, not merely ratcheted. This is the
    'mixed usage' defect class applied to aliases."""
    source = textwrap.dedent(
        """
        def f(state, write_gate, emit=log_event):
            write_gate = require_write_gate(write_gate)
            write_gate.append_event(state, "k", {})
            emit(state, "other", {})
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    alias_sites = [s for s in sites if s.primitive == "log_event"]
    assert len(alias_sites) == 1, f"alias call not detected in predicate function: {sites}"
    assert alias_sites[0].in_predicate, (
        "an alias call inside a WriteGate-using function is in-predicate -- "
        "it is subject to the exclusive-use rule, not merely ratcheted"
    )


def test_scanner_assigns_distinct_occurrences_to_identical_alias_calls() -> None:
    """The occurrence-index discipline (issue #619 / the existing
    `test_scanner_assigns_distinct_occurrence_indices_to_identical_call_shapes`)
    applies to alias-resolved calls too: two byte-identical `emit(...)`
    calls in the same scope get distinct occurrence indices so an allow-
    list entry covers exactly one, not both."""
    source = textwrap.dedent(
        """
        def f(state, emit=append_event):
            state = emit(state, "k", {})
            state["a"] = 1
            state = emit(state, "k", {})
            return state
        """
    )
    tree = ast.parse(source)
    sites = _scan_module_for_raw_primitive_calls(tree, "synthetic_fixture.py")

    assert len(sites) == 2, f"expected two distinct alias call sites, found: {sites}"
    assert sites[0].call_source == sites[1].call_source
    assert {s.occurrence for s in sites} == {0, 1}
    assert len({s.key for s in sites}) == 2


def test_real_pr2_pr3_converted_sites_are_not_flagged() -> None:
    """Real-code confirmation that bucket (a) fires against actual repo
    formatting, not just the synthetic fixtures above. W6 PR2 (merged as
    #1323) converted every write path in stalled_review_reap.py except
    `_append_sweep_events` (deferred to PR3 per R5); PR3 then converted
    `_append_sweep_events` and both its callers, closing the R5 boundary.
    At this PR's base, `_detect_and_handle_stalled_reviews` is the ONLY
    scope in this module with any raw (non-gate-covered) call, and every
    one of those is on the R4 allow-list -- if bucket (a)'s receiver-shape
    check were subtly wrong against real repo formatting, or if R5's
    completion had left a residual raw call, this module's unaccounted
    count would be nonzero."""
    module_path = _SRC_ROOT / "stalled_review_reap.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    sites = _scan_module_for_raw_primitive_calls(tree, "stalled_review_reap.py")

    scopes_found = {s.scope for s in sites}
    assert scopes_found <= {"_detect_and_handle_stalled_reviews"}, (
        f"unexpected raw call site outside the R4-allowlisted scope "
        f"(R5 completion should have closed _append_sweep_events): {sites}"
    )
    allowed_keys = {e.key for e in _ALLOWED_RAW_PRIMITIVE_SITES}
    non_allowlisted = [s for s in sites if s.key not in allowed_keys]
    assert non_allowlisted == [], (
        f"expected zero unaccounted raw sites in stalled_review_reap.py post-PR3, "
        f"found: {non_allowlisted}"
    )


def test_legacy_forwarding_wrapper_exemption_toggle_is_reversible() -> None:
    """Positive control for the staleness check below: with the exemption
    OFF, `_record_event`'s own bare `append_event` call must be visible;
    with it ON (the default `_scan_gated_mutator_layer` uses), it must be
    exempt. Proves the `apply_legacy_exemption` toggle -- and by extension
    the staleness check that relies on it -- can actually distinguish
    'still matches' from 'stopped matching', using a synthetic module so it
    doesn't depend on workflow.py's real current shape staying byte-for-byte
    stable."""
    source = textwrap.dedent(
        """
        class OrchestratorApp:
            def _record_event(self, state, kind, payload):
                return append_event(state, kind, payload, state_path=self.paths.state_file)
        """
    )
    tree = ast.parse(source)

    exempted = [
        s
        for s in _scan_module_for_raw_primitive_calls(tree, "workflow.py")
        if s.scope == "_record_event"
    ]
    assert exempted == [], f"legacy wrapper exemption did not apply by default: {exempted}"

    unexempted = [
        s
        for s in _scan_module_for_raw_primitive_calls(
            tree, "workflow.py", apply_legacy_exemption=False
        )
        if s.scope == "_record_event"
    ]
    assert len(unexempted) == 1, f"expected exactly one un-exempted site, got: {unexempted}"
    assert unexempted[0].primitive == "append_event"


def test_legacy_forwarding_wrapper_scopes_are_not_stale() -> None:
    """Same discipline as `test_write_gate_allowlist_entries_are_not_stale`,
    applied to `_LEGACY_FORWARDING_WRAPPER_SCOPES`: an unmaintained
    exemption is a silent hole, and per R10 this exemption is permanent for
    as long as `_record_event` exists -- so it must always cover a real
    site. Re-scan each named module WITHOUT the exemption (relies on
    `test_legacy_forwarding_wrapper_exemption_toggle_is_reversible` above
    having already proven the toggle can distinguish match from no-match)
    and assert at least one real, currently-un-gated call remains inside
    that exact (path, scope) -- proving the entry is presently covering a
    real site, not a mistyped or already-deleted one."""
    for rel_path, scope in _LEGACY_FORWARDING_WRAPPER_SCOPES:
        module_path = _SRC_ROOT / rel_path
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        sites = _scan_module_for_raw_primitive_calls(tree, rel_path, apply_legacy_exemption=False)
        matching = [s for s in sites if s.scope == scope]
        assert matching, (
            f"_LEGACY_FORWARDING_WRAPPER_SCOPES entry ({rel_path!r}, {scope!r}) no longer "
            "matches any raw call site -- remove it (and/or force _record_event's deletion, "
            "issue #1324) rather than leaving a stale exemption"
        )


def test_write_gate_allowlist_entries_are_not_stale() -> None:
    """Symmetric direction of the main enforcement check, mirroring
    test_instrumentation.py's `test_event_kind_registry_exhaustive` 'stale'
    assertion: every _ALLOWED_RAW_PRIMITIVE_SITES entry must match a real
    site found by today's scan. An entry that matches nothing is a silent
    hole -- the code it was written to justify moved or changed, and the
    allow-list is now lying about what it covers."""
    sites = _scan_gated_mutator_layer(_SRC_ROOT)
    found_keys = {s.key for s in sites}

    stale = [entry for entry in _ALLOWED_RAW_PRIMITIVE_SITES if entry.key not in found_keys]
    assert not stale, (
        "_ALLOWED_RAW_PRIMITIVE_SITES entry no longer matches any real raw call site -- "
        "remove it or update it to match the current source:\n"
        + "\n".join(f"  {e.path} in {e.scope}(): {e.primitive}(...)" for e in stale)
    )


def test_write_gate_allowlist_staleness_check_detects_a_truly_dead_entry() -> None:
    """Task requirement: 'Dead allow-list/exemption entries must FAIL the
    test.' `test_write_gate_allowlist_entries_are_not_stale` above proves
    the CURRENT real entries all still match (the correct-state case) --
    this test is the positive control proving the staleness check itself
    would actually catch a dead one, using the exact same logic against a
    synthetic entry that cannot possibly match any real site."""
    sites = _scan_gated_mutator_layer(_SRC_ROOT)
    found_keys = {s.key for s in sites}

    dead_entry = _RawPrimitiveSite(
        path="stalled_review_reap.py",
        scope="_detect_and_handle_stalled_reviews",
        primitive="log_event",
        call_source="log_event(state_file, 'this_call_source_does_not_exist_anywhere', {})",
    )
    assert dead_entry.key not in found_keys, (
        "test premise violated: the synthetic dead entry unexpectedly matched a real site"
    )

    stale = [
        entry
        for entry in (*_ALLOWED_RAW_PRIMITIVE_SITES, dead_entry)
        if entry.key not in found_keys
    ]
    assert dead_entry in stale, "the staleness check did not flag the seeded dead entry"


def test_ratchet_passes_when_module_count_holds_or_shrinks() -> None:
    """R9 ratchet mechanics, positive case, tested as a pure function of
    synthetic count dicts (independent of the real tree)."""
    baseline = {"workflow.py": 5, "reconcile.py": 3}
    holds = {"workflow.py": 5, "reconcile.py": 3}
    shrinks = {"workflow.py": 4, "reconcile.py": 0}

    assert _ratchet_violations(holds, baseline) == []
    assert _ratchet_violations(shrinks, baseline) == []


def test_ratchet_fails_when_module_count_increases() -> None:
    """R9 ratchet mechanics, negative case: an increase in ANY module fails,
    naming that module (and only that module -- an unrelated module holding
    steady must not be swept in)."""
    baseline = {"workflow.py": 5, "reconcile.py": 3}
    one_regressed = {"workflow.py": 6, "reconcile.py": 3}

    assert _ratchet_violations(one_regressed, baseline) == ["workflow.py"]


def test_ratchet_treats_a_brand_new_module_as_baseline_zero() -> None:
    """A module with ANY out-of-predicate raw site that has no baseline
    entry at all is compared against an implicit baseline of 0 -- new raw
    writes appearing in previously-untouched (i.e. previously fully clean)
    territory must fail, not silently pass because the module was never
    enumerated."""
    baseline: dict[str, int] = {}
    actual = {"brand_new_module.py": 1}

    assert _ratchet_violations(actual, baseline) == ["brand_new_module.py"]


# ---------------------------------------------------------------------------
# The keystone assertion itself (issue #1264 R9).
#
# Two independent checks, both fail-closed, together covering every raw
# call to a gated primitive in src/charlie_work:
#   1. Every IN-PREDICATE site (its enclosing function uses WriteGate) must
#      be gate-routed or on the R4 allow-list -- zero tolerance, no ratchet.
#   2. Every OUT-OF-PREDICATE module's raw-call count must not exceed its
#      recorded _RATCHET_BASELINE -- shrink or hold only.
# ---------------------------------------------------------------------------
def test_write_gate_no_unaccounted_raw_primitive_calls() -> None:
    """The PR4 keystone. Fails loudly, naming every offending site or
    module, otherwise."""
    sites = _scan_gated_mutator_layer(_SRC_ROOT)
    allowed_keys = {entry.key for entry in _ALLOWED_RAW_PRIMITIVE_SITES}

    in_predicate_unaccounted = [
        site for site in sites if site.in_predicate and site.key not in allowed_keys
    ]
    assert not in_predicate_unaccounted, (
        "Raw (un-gated) call(s) to a WriteGate primitive found inside a function that "
        "itself uses WriteGate (issue #1264 R9's exclusive-use predicate). Route through "
        "self.write_gate.<method>(...) (Convention A) or an explicit write_gate: WriteGate "
        "parameter validated by require_write_gate() (Convention B), or add a reasoned "
        "_ALLOWED_RAW_PRIMITIVE_SITES entry if this site is deliberately raw by design:\n"
        + _format_inventory(in_predicate_unaccounted)
    )

    out_of_predicate_counts: dict[str, int] = {}
    for site in sites:
        if site.in_predicate or site.key in allowed_keys:
            continue
        out_of_predicate_counts[site.path] = out_of_predicate_counts.get(site.path, 0) + 1

    regressed = _ratchet_violations(out_of_predicate_counts, _RATCHET_BASELINE)
    assert not regressed, (
        "Raw primitive call count increased in module(s) outside issue #1264's wave scope "
        "(R9's per-module shrink-only ratchet). A NEW raw write appeared in territory this "
        "wave does not convert -- either route it through WriteGate, or if it is genuinely "
        "out of scope, this ratchet is the wrong place to relax; file/expand the owning "
        "follow-up issue instead:\n"
        + "\n".join(
            f"  {module}: baseline={_RATCHET_BASELINE.get(module, 0)} "
            f"actual={out_of_predicate_counts[module]}"
            for module in regressed
        )
    )


# ---------------------------------------------------------------------------
# Companion dry-run integration test -- see test_write_gate_dry_run_loop.py
# (issue #1264 W6 PR4) for the full OrchestratorApp loop-pass assertion.
# ---------------------------------------------------------------------------
