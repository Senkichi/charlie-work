# OrchestratorApp: Mikado Graph and Delegation Plan (Track 2 Phase B design)

Status: DESIGN ONLY. This document moves no code. It is the single planning
artifact for Track 2 Phase B of the god-object paydown umbrella (charlie-work
#1582), produced under the Phase B prep issue (#1628). Every leaf named here is
drafted as its own `needs-design` issue (see the leaf-issue drafts referenced in
Provenance); this document does not file them.

Author: synthesis/adjudication pass over three analysis artifacts generated in
this pass (member inventory, four-tier patch census, Mikado projection), plus a
second-author rework after adversarial review (Section 11). Every number below
cites the script that produced it and the base commit
`eb634c9b319462955984cce9452a3660497c901d` it was measured at. `workflow.py` is
byte-identical (git blob `ae92e6e0`) between that base and the design-PR head
`20b934d0`, so every census/fence/mikado figure holds at head; the second-author
synthetic verifications (Section 8.4) were run in the worktree venv at head
`20b934d0`.

Target: `src/charlie_work/workflow.py`, class `OrchestratorApp` (lines
4006-23450). Interpreter: CPython 3.13.5. Metric authority:
`.attachment-budgets.json` OrchestratorApp entry (`kind: class`) and the live
`class`-archetype Tukey fence recomputed at HEAD (Section 2.4).

This plan follows the merged GitHub-class plan
(`docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md`,
Phase A leaf L06b) but deliberately **departs from its collaborator-class shape**
(Section 2.5): Phase B moves bodies into **module-level free functions**, not
collaborator classes, because a new collaborator class is itself a new `class`
attachment point that the saturation gate hard-blocks, while a free-function
module is not an attachment point at all (Section 2.4, 2.5, verified 8.4).


## 1. Summary

`OrchestratorApp` is an APC `class`-archetype god object: **133 lexical method
definitions** (`analyze_orchestratorapp.py`; AST over `workflow.py` at base,
re-confirmed by the live APC scanner at head -- `reconcile.py`), by a wide margin
the largest class in the tree. The live `class` fence is **8.5** (Section 2.4),
so the goal is member_count <= 8.

Unlike `GitHub` (a frozen `@dataclass` implementing the `GitHubLike` Protocol),
`OrchestratorApp` is a **plain mutable class** (`class_bases: []`,
`class_decorators: []`; `analyze_orchestratorapp.py`). There is **no Protocol,
no conformance test, and no `isinstance` site** to satisfy. That removes the
entire `functools.wraps`/`__signature__` conformance burden that dominated the
GitHub plan -- but it does **not** remove the delegation requirement, because the
reason `OrchestratorApp` cannot shed members by moving bodies alone is a
different, sharper constraint: its dominant test-interception seam is
**module-level free-function patching**, not class-member patching (Section
2.3). The census (`patch_census.py`) finds two distinct patch populations, not
one: **42 member-patch sites** (Tier A class-level 6 + Tier B instance-level 36)
across 16 distinct `OrchestratorApp` members, and -- a different kind of
interception entirely -- **140 module-attribute patch sites** (Tier D) across 21
names in the `charlie_work.workflow` module namespace that `OrchestratorApp`
bodies call. These are different populations summing to 182 sites, not 182 sites
of one kind; the dominant seam is the module-level population, 140 to 42. Any
relocated body must keep resolving those 21 names through the
`charlie_work.workflow` module, or those 140 sites stop intercepting.

**The delegation shape (reworked -- free functions, not collaborator classes).**
Each member body moves to a **module-level `def <name>(self, ...)`** in a #1283
domain module, keeping `self` as its first parameter so the byte-identical body
still reads `self._record_event(...)`, `self.gh`, `self.config`.
`OrchestratorApp` then installs `<name>` as a **class-level assignment**
(`OrchestratorApp.<name> = <module>.<name>`). A plain function assigned on a
class binds as a method through the ordinary descriptor protocol, so
`app.<name>()` passes the instance as `self` with no adapter, no collaborator
`__getattr__`, and no `_owner` back-reference (verified 8.4b). A class-level
`Assign` is not a `FunctionDef` child, so it contributes **zero** to the `class`
metric while every instance patch (Tier B) and class patch (Tier A) keeps
intercepting (verified 8.4a-b).

**Why free functions and not collaborator classes (the load-bearing change from
the first draft; Section 2.5).** A collaborator class holding the moved bodies is
itself a new `class` attachment point. The `state` cluster alone is 38 members;
a `State` collaborator would be a 38-member `class`, 4.5x the fence of 8.5, and
`baseline.compare()` (`baseline.py:331-363`) emits a hard `block` for a brand-new
attachment point that appears already saturated. So the collaborator-class shape
does not remove a saturated point; it trades one saturated `class` for several
new ones, each needing its own baseline row and its own paydown. A free-function
module is **not** a `class` attachment point -- and there is no generic `module`
archetype in the scanner at all (Section 2.4) -- so it creates **zero** new
attachment points (verified 8.4a: a module of 15 realistic free functions scans
to zero points; the same 15 names as a class scan to one saturated `class`
point). This is exactly the umbrella #1582 rule: remove lexical members without
relocating them into a new point that becomes its own saturated class.

End state (Section 8): the Mikado projection (`mikado.py`) lands
`OrchestratorApp` at **member_count = 7** -- `__init__` plus the six
`@_guard_state_lock`-guarded public command entrypoints
(`dispatch_reviews`, `intake`, `loop`, `merge_ready`, `review`, `status`) --
strictly under the fence of 8.5, across roughly 19 sequenced move-PRs.

**Honest verdict, stated up front (full detail in Section 8):** this reaches the
*metric* exit with a *named residual*. It does **not** dismantle the god object.
126 of the 133 names survive as forwarding assignment shims. The bodies genuinely
leave the class and become independently testable free functions, and the
133 -> 7 count drop is real and mechanically ratchet-verifiable. But true Phase B
exit (umbrella #1582's "no facade shim survives" condition, enforced by a
vulture-class sweep) additionally requires repointing the Tier A/B call sites and
the src-tree consumers of the 22 externally-referenced members onto the
free functions and then deleting the shims -- the expensive part, calibrated
against #1449 (27 names over 11 days) and deferred as a follow-on. Two further
residuals the operator accepts: **zero new attachment points, but ~23 destination
modules of which three exceed the repo's 800-line file convention** (Section 8.2,
residual 4 -- there is no CI gate on file length, so this is a convention debt,
disclosed not hidden), and the standard attribute-surface / hot-path / legibility
residuals. This document recommends the metric-exit-with-named-residual as
Phase B's landing point and scopes the full-exit as an explicit follow-on, rather
than presenting 7 as "god object dismantled."


## 2. Counting definition and census

### 2.1 What the metric counts

The `class` archetype counts **lexical** `FunctionDef`/`AsyncFunctionDef` nodes
that are direct children of the `ClassDef.body`
(`src/charlie_work/attachment_contracts/archetypes.py`: `_is_def`, and
`members = tuple(child.name for child in node.body if _is_def(child))`).
Consequences that drive the whole design:

- The 3 class-level `Assign` statements in the body (`_UNESCALATE_PR_RESET_FIELDS`,
  `_UNESCALATE_ISSUE_RESET_FIELDS`, `_REWORK_BUDGET_RESET_BY_ESCALATION_REASON`;
  constant aliases at lines 12061-12063) do NOT count -- they are `Assign`, not
  `FunctionDef`.
- `OrchestratorApp` has **0 `AnnAssign` fields**: all 11 instance attributes
  (`gh`, `config`, `paths`, `dry_run`, `write_gate`, `repo_root`, `prompt_dirs`,
  `_layout`, `_preflight_config_mtimes`, `_worker_token_escalated`,
  `fleet_dir_override`) are set via `self.x = ...` inside `__init__`
  (`analyze_orchestratorapp.py`). So there is no field surface to convert; every
  reduction comes from a `def`.
- A class-level **assignment** of a module-level free function
  (`OrchestratorApp.foo = domain_mod.foo`, or an install loop) is not a
  `FunctionDef` child and does NOT count. This is the metric fact the facade
  exploits (verified 8.4a).
- A `def` shim -- an explicit thin `def foo(self, ...): return foo_impl(self, ...)`
  -- WOULD count. So thin `def` delegates cannot reduce the metric; only
  class-level assignment delegates can. This is the pivot of the whole plan.

Authoritative current count: **133 lexical defs**, 0 async
(`analyze_orchestratorapp.py`; enumerated with per-member line spans in the
inventory JSON; live APC scanner agrees at head -- `reconcile.py`). Class span:
lines 4006-23450.

### 2.2 How OrchestratorApp differs from GitHub (three plan-changing facts)

| dimension | GitHub (Phase A) | OrchestratorApp (Phase B) | plan consequence |
|-----------|------------------|---------------------------|------------------|
| class shape | frozen `@dataclass`, `GitHubLike` Protocol | plain mutable class, no bases | no conformance test / `isinstance` to satisfy; instance (Tier B) patches work directly |
| dominant seam | `run` member (134 class/instance patch sites) | module-level free functions (140 Tier D sites) | preserve **module-namespace resolution**, not one member |
| shared state | `_list_cache` (owner-held, by reference) | one attr write outside `__init__` (`_worker_token_escalated`) | near-zero state coupling; call graph is a DAG (no SCCs) |

Because there is no Protocol, the delegate does **not** need
`functools.wraps` + `__signature__` for a conformance comparison. The moved body
is a plain module-level function; installing it as a class attribute is all that
is required for `app.<name>()` to work (verified 8.4b). Nothing in the test suite
requires signature fidelity the way `tests/test_githublike_protocol.py` did for
`GitHub`.

### 2.3 The four-tier patch census (adopt as the planning number)

`patch_census.py`, AST over `tests/` at HEAD. **Counting definitions (stated
before the numbers, per the negative-results rule):**

- **site** = a distinct `(file, line)` pair where a name is rebound.
- **member set** = the 133 `FunctionDef` members of `OrchestratorApp`, derived
  from the live AST (never hardcoded).
- **Tier A** = class-level rebind of `OrchestratorApp.<member>`:
  `patch.object(OrchestratorApp, 'm')`, `patch('...workflow.OrchestratorApp.m')`,
  or `OrchestratorApp.m = ...`.
- **Tier B** = instance-level rebind on an `OrchestratorApp` instance
  (`monkeypatch.setattr` / `patch.object` / attribute assignment where the object
  resolves to an OA instance, via signal in {construct, factory, fixture-param,
  name-heuristic}).
- **Tier C** = subclass override: `class X(OrchestratorApp)` lexically
  (re)defining a member.
- **Tier D** = module-level rebind of a `charlie_work.workflow` name that an OA
  body references: `patch('charlie_work.workflow.n')` /
  `monkeypatch.setattr(workflow, 'n', ...)` -- the patch path is the **module**,
  not `...OrchestratorApp.n`.
- **B_unresolved** (loud bucket) = a `setattr`/`patch.object`/attr-assign whose
  member IS in the OA set but whose object could NOT be tied to an OA instance.
  Reported separately, never counted in any tier.

| tier | meaning | distinct sites | members touched |
|------|---------|----------------|-----------------|
| A | class-level member patch | **6** | 5 |
| B | instance-level member patch | **36** | 13 |
| C | subclass override | **0** | 0 |
| D | module-level free-function patch | **140** | 21 |
| B_unresolved | loud bucket (uncounted) | **0** | -- |

**Positive controls (the zero/low claims are only meaningful because the same
machinery finds these):** Tier A top = `_process_rescue_review` (2 sites); Tier B
top = `review` (11 sites). The B_unresolved bucket is empty *and* the controls
are non-empty, so "0 unresolved" is a finding, not a broken query. 117 of the 133
members carry no patch site of any tier.

**Tier A is 6, verified against a raw grep.** A raw
`grep -rEn "(monkeypatch.setattr\(|patch.object\(|patch\()[^)]*OrchestratorApp"`
over `tests/` returns 31 lines, but they split cleanly: 6 genuine class-member
patches (`monkeypatch.setattr(OrchestratorApp, "m", ...)` for
`_maybe_probe_quota_recovery`, `_maybe_reconcile_drift`, `_maybe_reclaim_worktrees`,
`_maybe_reclaim_superseded_main_ci`, and `_process_rescue_review` x2), plus ~25
`@patch("charlie_work.fleet_dispatch.OrchestratorApp")` /
`patch("charlie_work.cli.OrchestratorApp")` whole-class-symbol patches. The
latter replace the entire class object at its import site in another module (they
mock the constructor); they are a separate category and are **unaffected** by
member -> delegate conversion, which is why the census correctly excludes them
from Tier A.

### 2.4 The fence, the exit target, and the absence of a module fence

`saturate()` (`src/charlie_work/attachment_contracts/outliers.py`) computes the
boundary per-kind as `Q3 + 1.5*IQR` over the eligible same-kind population
(excluding `is_linear_ledger`, `is_structurally_trivial`, and `member_count < 1`),
with nearest-rank quartiles and a statistical FLOOR of 4.

Recomputed live at HEAD over the actual scanned tree (`fence_probe.py`):
eligible `class` population **46**, Q1 **1.0**, Q3 **4.0**, IQR **3.0**, fence
**8.5**. So the exit target is **member_count <= 8** (strictly under 8.5;
`member_count > boundary` is the saturation test, so 8 is not saturated).

**There is no `module` archetype, so there is no module fence.** The scanner's
`Kind` (`model.py:13-20`) is exactly six literals -- `typer_app`, `click_group`,
`blueprint`, `class`, `migration_runner`, `test_module` -- none of which is a
generic module. A module-shaped attachment point is emitted **only** by
`_module_ledger_points` (which requires a `<prefix><int>` linear-ledger family --
`ledger.py`) or `_test_module_point` (which requires a `test_*`/`*_test`
filename). A #1283 domain module holding relocated free functions with ordinary
names is neither, so it produces **no attachment point at all** and is never
tested against any fence. Verified with a positive control (`probe_module2.py`,
8.4a): a module of 15 realistically-named free functions scans to **0** points;
the same 15 names wrapped in a class scan to **1** `class` point saturated at 8.5.
This is why the free-function shape sizes against the repo's **800-line file
convention** (a line count), not against a fence -- and APC deliberately never
measures lines: `model.py:5-6` states "No line count is ever read anywhere in
this package (binding operator constraint)." There is no file-length gate in
charlie-work (`pyproject.toml`'s `line-length = 99` is ruff's per-line character
width, not a file cap; no hook or CI check enforces file length). Destination
module sizing (Section 5) is therefore a **repo convention with no CI backstop**;
Section 8.2 residual 4 discloses where the plan exceeds it.

**Fence stability under OA's own reduction (`fence_probe.py`):** as OA's count
falls 133 -> 1 the fence is **invariant at 8.5** at every step (OA is a single
high outlier; dropping it does not move Q3 of the other 45 points). So the
projection's per-leaf target never moves as OA shrinks. **The first draft's
"fence under population growth" counterfactual is now moot and has been deleted:**
the free-function shape appends **no** new eligible `class` points, so the
eligible `class` population is unchanged except for OA's own descending count,
which is the invariant direction just measured. The exit criterion remains the
**committed ratchet baseline value** (which only descends), not a live-recomputed
fence, so no later population shift can un-satisfy a landed leaf (Section 9).

### 2.5 Why the collaborator-class shape was rejected

The merged GitHub Phase A plan used collaborator classes. Phase B cannot: a
collaborator class holding a cluster's moved bodies is a **new `class` attachment
point**. The clusters are large (Section 3: `state` 38, unrouted-helpers 32,
misc-singleton fold 25), so a single collaborator per cluster would be a
38/32/25-member `class` -- 4.5x/3.8x/2.9x the fence of 8.5. `baseline.compare()`
(`baseline.py:331-363`) emits `Finding(severity="block")` for a
currently-saturated point that has **no baseline entry** ("a brand-new AP
appearing already saturated, not adoption"), so each such collaborator is a hard
CI block that must be granted its own baseline row (or sub-split below 8.5) before
CI passes. Splitting each cluster into many sub-fence collaborator classes is
possible but multiplies attachment points and baseline rows for no benefit the
free-function shape does not already provide at zero new points. Prior art does
not rescue the class shape at this magnitude: the Transport collaborator (PR
#1619) entered the baseline saturated at 12 vs fence 8.5 (1.4x), an acknowledged
marginal case; `state`@38 is 4.5x the fence -- a new god object, not a marginal
collaborator. **Free functions create zero attachment points (2.4, 8.4a), so the
saturation gate is satisfied structurally rather than by granting new baseline
rows.** This is the single reason the shape changed from the first draft, and it
is why the first draft's Section 2.4 "extracted classes span ~4-12" claim is
**deleted**: under this shape there are no extracted classes at all.


## 3. Domain segmentation and the free-function delegation shape

`OrchestratorApp` already imports from 53 domain modules
(`analyze_orchestratorapp.py`, `imported_domain_modules`). Each member is
assigned to the domain module it most references (by resolved free-function
calls), deterministically from the inventory -- no member list is hand-typed.
The clustering (`analyze_orchestratorapp.py`, `domain_clusters`):

| domain module (#1283 target) | members | notes |
|------------------------------|---------|-------|
| `charlie_work.state` | 42 | dominant cluster; state read/transition helpers |
| (unrouted) | 36 | orchestration-glue helpers with no single dominant domain import |
| `charlie_work.github` | 13 | PR/issue interaction wrappers over `self.gh` |
| `charlie_work.instrumentation` | 8 | event/digest emission |
| `charlie_work.dead_worker_reap` | 4 | orphan/stall reaping |
| `charlie_work.prompts` | 3 | prompt rendering |
| `charlie_work.backlog_reachability` | 3 | |
| 17 further singleton/pair domains | 1-2 each | escalation, worktree, janitor, checks, reconcile, citation_check, ... |

(The per-leaf member counts in Section 5 differ from these raw cluster sizes
because `__init__`, the six guarded commands, the property/staticmethod members,
and the single state-writer are assigned to the residual / adapter / extract
leaves first; the domain leaves get what remains -- see `mikado.py`.)

Structural facts that make this clustering safe to cut in almost any order
(`analyze_orchestratorapp.py`):

- **No non-trivial SCCs** in the member call graph (`nontrivial_sccs: []`). The
  `self.<method>()` call graph is a DAG, so there is no mutual-recursion cluster
  that must move atomically.
- **One mutated instance attribute outside `__init__`:**
  `_worker_token_escalated`, written only inside `_dispatch_impl` (at two branch
  sites, `workflow.py:5156` and `:5230`) -- `instance_attr_writes_outside_init`,
  independently confirmed by an stdlib-ast scan over Assign/AugAssign/AnnAssign/
  setattr/tuple-unpack targets. Every other member is read-only against instance
  state, so delegation carries read-only dependencies and the move is clean. That
  single writer is isolated into its own extract leaf (Section 5, L08).
- **62 pure leaves** (members that call no other OA method) -- these are the
  lowest-risk conversions.

### 3.1 The free-function move and its two mechanical rules

Each converted member moves like this, in a verbatim (byte-identical) body move:

```
# before, in workflow.py, inside class OrchestratorApp:
    def _record_review_outcome(self, pr, verdict):
        emit_digest(...)                       # bare module-level free fn
        return self._route_to_rework(pr)       # another OA member

# after:
#   src/charlie_work/orchestration/state_review.py  (a #1283 domain module)
import charlie_work.workflow as _wf
def _record_review_outcome(self, pr, verdict):
    _wf.emit_digest(...)                       # SAME body, module-namespaced call
    return self._route_to_rework(pr)           # unchanged: self resolves the shim

#   in workflow.py, after the class body (or via the installer, Section 3.2):
    OrchestratorApp._record_review_outcome = state_review._record_review_outcome
```

Two rules, each mechanical and each gated by the audit (Section 4):

1. **`self` stays the first parameter and the body is byte-identical.** The
   function is installed as a class attribute, so `app._record_review_outcome(...)`
   binds `app` as `self` through the descriptor protocol (verified 8.4b). No
   collaborator object, no `__getattr__`, no `_owner`. `self._route_to_rework(pr)`
   inside a moved body resolves the *installed shim* on `OrchestratorApp` exactly
   as before -- so intra-class call chains keep working with no back-resolution
   machinery.

2. **Module-level free functions are referenced through the `charlie_work.workflow`
   namespace (the #1627 rule -- this is what Tier D forces).** The 140 Tier D
   sites patch module-level free functions in `charlie_work.workflow` -- e.g.
   `patch("charlie_work.workflow._worker_pid_alive", ...)` (a `_dispatch_impl`
   dependency) and `patch("charlie_work.workflow.emit_digest", ...)`. They
   intercept because an OA body calls the bare name, resolved through the
   `charlie_work.workflow` module global the test patched. A moved body in a
   domain module must therefore reference these names **through the
   `charlie_work.workflow` module object** (`import charlie_work.workflow as _wf;
   _wf.emit_digest(...)`), NOT via a fresh
   `from charlie_work.workflow import emit_digest` re-imported into the domain
   module -- a fresh import binds a *new* name in the domain module's namespace
   that `patch("charlie_work.workflow.emit_digest")` does not reach, silently
   breaking those tests. This is the #1627 module-level-patch-binding concern, and
   it is the single sharpest design rule for Phase B (verified 8.4c against live
   `unittest.mock`; the reviewer independently reproduced the intercept-vs-miss
   split).

**Circular-import hazard, and why the prescribed form is cycle-safe (review
Finding 4).** A #1283 domain module that `workflow.py` itself imports at import
time, and which then does `import charlie_work.workflow as _wf`, forms an import
cycle. The module-object form is cycle-safe **by construction**: `import
charlie_work.workflow as _wf` binds the *module object* -- which Python has
already registered in `sys.modules`, partially initialized, before `workflow.py`
finishes executing -- and `_wf.emit_digest` is not resolved until call time, long
after both modules finish importing. A `from charlie_work.workflow import
emit_digest` at module top would instead try to read the attribute *during* the
cyclic import, when it may not yet be defined -- an `ImportError` -- which is a
second, independent reason the from-import form is banned here. Verified (8.4c):
a synthetic `workflow` module that imports a domain module which does `import
...workflow as _wf` imports cleanly, and `patch("...workflow.target")` reaches the
moved body's `_wf.target()` call.

### 3.2 The install machinery (rule-9 decision) and its effect on workflow.py

The names must be re-attached to `OrchestratorApp`. Two options, and this plan
picks one and defends it:

- **(A) 126 literal `OrchestratorApp.<name> = <mod>.<name>` lines** in/after the
  class body. Statically greppable, but 126 hand-maintained assignment lines that
  must be edited by hand for every future move -- the "toothpick-brittle manual
  list" CLAUDE.md rule 9 explicitly forbids.
- **(B) a derived installer** `_install_delegates(OrchestratorApp)` in a new
  module `src/charlie_work/workflow_delegation.py` (the analog of the merged
  `github_delegation.py`) that **introspects each destination domain module's
  public top-level `def`s**, builds a `name -> module` route table, raises on any
  cross-module name collision, and installs each function as a class attribute
  (wrapping the property/staticmethod adapters, Section 3.3). `workflow.py` calls
  `_install_delegates(OrchestratorApp)` once after the class body.

**Decision: (B), the derived installer, kept in its own module.** Rationale:
the route table is *derived from the destination modules' own contents* (rule 9's
"lists derive from robust state declaration and collection"), and the collision
check is the robustness rule 9 asks for -- a hand-list has neither. The cost is a
real legibility hit: the class's member surface is no longer visible to a static
reader of `workflow.py` (they must read `_install_delegates` and the domain
modules to see what `OrchestratorApp` exposes). That cost is disclosed as part of
the legibility residual (Section 8.2, residual 3); it does not change the metric.

**Effect on `workflow.py`'s own size.** The first draft worried about
"`workflow.py`'s size ratchet." There is no such ratchet: `workflow.py` is a
module, not an attachment point (Section 2.4), and APC reads no line counts, so
`workflow.py`'s line count is governed only by the same 800-line *convention*,
which it already vastly exceeds and which has no CI gate. Under this plan
`workflow.py` **shrinks** substantially: ~14.6k lines of moved bodies leave (the
19,288 total member lines minus the 4,635 residual lines that stay), replaced by
one `_install_delegates(...)` call. The installer living in its own module keeps
that one call, not 126 assignment lines, in `workflow.py`.

### 3.3 Adapters (property / staticmethod)

Nine members are decorated (`analyze_orchestratorapp.py`): six with
`@_guard_state_lock` (the guarded commands -- Section 5 residual), one `@property`
(`layout`), two `@staticmethod` (`_is_dead_blocker`, `_write_json`). The
property and the two staticmethods move to free functions and are installed as
**adapter-wrapped class-level assignments**
(`OrchestratorApp.layout = property(orchestration_adapters.layout)`,
`OrchestratorApp._write_json = staticmethod(orchestration_adapters._write_json)`)
-- still `Assign`, not `FunctionDef`, so they drop from the count while keeping
`app.layout` and `OrchestratorApp._write_json(...)` working (verified 8.4d). This
is the adapter leaf.


## 4. Test-tier impact and the four-round monkeypatch audit

Because `OrchestratorApp` is a plain mutable class, an **instance** patch
(`monkeypatch.setattr(app, "m", fn)`, Tier B, 36 sites) sets an attribute on the
instance that shadows the class-level assignment -- it keeps working unchanged. A
**class** patch (Tier A, 6 sites) rebinds `OrchestratorApp.m`; since the installed
delegate is itself a class attribute, `patch.object(OrchestratorApp, "m", ...)`
replaces it identically to replacing a `def`. Both member-patch tiers survive the
conversion untouched (verified 8.4b: class patch, instance `patch.object`, and
`monkeypatch.setattr` all intercept a class-level function assignment). The tier
that would break under a careless move is Tier D, per Section 3.1 rule 2.

**Four-round monkeypatch audit checklist (per move-PR, the #1449 precedent).**
Each round is a distinct gate (verification-ladder taxonomy in parentheses):

- **Round 1 -- Tier A/B member patches (pre-flight).** For every member the PR
  converts, grep `tests/` for `patch.object(OrchestratorApp, "<m>"`,
  `patch("...OrchestratorApp.<m>"`, and instance `setattr(<app>, "<m>"`. Confirm
  each still resolves to the installed delegate (class patch) or shadows it
  (instance patch). Expected: unaffected. Any miss escalates. The exact Tier A/B
  sites for each leaf are enumerated in Section 5 (from `patch-census.json`).
- **Round 2 -- Tier D module-namespace binding (revision).** For every
  module-level free function the moved body calls (the leaf's Tier D name set,
  Section 5), confirm the relocated body references it via the
  `charlie_work.workflow` namespace, and that `patch("charlie_work.workflow.<fn>")`
  still intercepts the moved body. This is the round that catches the Section 3.1
  rule-2 hazard.
- **Round 3 -- #1627 binding-style sweep (revision).** Confirm no relocated body
  introduced a `from charlie_work.workflow import <fn>` that rebinds a patched
  name into a domain-module namespace. AST-grep the moved module for
  `ImportFrom(module="charlie_work.workflow")` against the Tier D name set.
- **Round 4 -- doubles and subclasses (escalation).** Tier C is 0 today; confirm
  the PR introduces no `class X(OrchestratorApp)` override, and that no test
  double asserts on class identity rather than behavior. Escalate any Tier C
  appearance (it would couple a leaf to a subclass).

The audit's outstanding-issue count is taken from each round's structured grep
result, not re-derived from a report file (verification-ladder stall rule).


## 5. Ordered leaf list (the Mikado leaves)

`mikado.py`, projecting member_count after each leaf. **10 leaves.** L00 is
pure infrastructure and moves zero members (umbrella rule: the first PR moves no
methods). The domain leaves are ordered by descending size so the ratchet
visibly and quickly descends; the DAG call graph and single-writer state
(Section 3) leave the order otherwise free.

A single reviewable move-PR converts at most **BATCH_MAX = 10** members
(GitHub's largest leaf moved 12; 10 keeps a batch inside one opus48 review). A
leaf exceeding that lands as `ceil(n/10)` sequential **batch-PRs**, each
ratcheting the committed baseline by its own delta. Four leaves are oversized;
the campaign is ~19 move-PRs total.

**Destination-module sizing (`sizing.py`, `reseam.py`).** Each leaf's converted
bodies move to one or more new `#1283` domain modules under
`src/charlie_work/orchestration/`. "Modules (>=800)" is the *minimum* module
count to keep each file under the 800-line convention (`ceil(sum-of-body-loc /
800)`); the actual split follows the leaf's sub-domain / hub structure (per-leaf
notes below), so real files will be smaller and more numerous. The convention has
no CI gate (Section 2.4); modules that a single oversized member forces over 800
are named as residual 4 (Section 8.2).

| # | leaf | conv | batches | body loc | dest modules (>=800) | mc after | under fence? |
|---|------|------|---------|----------|----------------------|----------|--------------|
| L00 | `workflow_delegation.py` machinery (moves 0) | 0 | -- | 0 | 1 (machinery) | 133 | no |
| L01 | domain: `state` | 38 | 4 | 6953 | >=9 (`orchestration/state_*`) | 95 | no |
| L02 | orchestration-helper delegates (unrouted) | 32 | 4 | 985 | >=2 (`orchestration/helpers_*`) | 63 | no |
| L03 | misc singleton-domain delegates | 25 | 3 | 1663 | >=3 (`orchestration/misc_*`) | 38 | no |
| L04 | domain: `github` | 12 | 2 | 1369 | >=2 (`orchestration/github_ops_*`) | 26 | no |
| L05 | domain: `instrumentation` | 8 | 1 | 707 | 1 (`orchestration/instrumentation_ops`) | 18 | no |
| L06 | domain: `dead_worker_reap` | 4 | 1 | 1025 | >=2 (`orchestration/reap_*`) | 14 | no |
| L07 | domain: `prompts` | 3 | 1 | 91 | 1 (`orchestration/prompt_ops`) | 11 | no |
| L08 | extract: dispatch-state (`_dispatch_impl`) | 1 | 1 | 1795 | 1 (over cap; residual 4) | 10 | no |
| L09 | adapter: property/staticmethod delegates | 3 | 1 | 65 | 1 (`orchestration/adapters`) | **7** | **yes** |

Total new destination modules: **~23** (all under `src/charlie_work/orchestration/`),
of which **three exceed the 800-line convention** because a single relocated body
does (see per-leaf notes and Section 8.2 residual 4). None is a `class` or any
other attachment point, so none appears in any fence population (Section 2.4).

**Per-leaf re-seam data** (Tier A/B member-patch sites the leaf must re-verify,
from `patch-census.json`; Tier D module-namespace names its bodies reference,
from the inventory `import_refs`/`module_level_refs` intersected with the 21 Tier
D names). Members with no Tier A/B site and leaves with no Tier D name are the
zero-risk conversions.

- **L00 -- machinery (moves 0).** Adds `src/charlie_work/workflow_delegation.py`
  (`_make_delegate` adapter helpers, the introspective `_build_routes` with
  collision detection, `_install_delegates(OrchestratorApp)`), an empty
  `src/charlie_work/orchestration/` package skeleton, and the post-class install
  call in `workflow.py`. Ratchet delta 0 (still 133). Blast radius: import surface
  only. Revert: delete the module + the install call.
  - Tier A/B sites: none. Tier D names: none. Destination module: 1 (machinery,
    well under cap).
- **L01 -- state (38, 4 batches; 6953 loc, >=9 modules).** The dominant cluster;
  split by sub-domain / hub into `orchestration/state_review.py`,
  `state_routing.py`, `state_merge.py`, `state_reconcile.py`, ... (7 hubs:
  `record_review`, `_route_to_rework`, `_request_merge_conflict_rework`,
  `_route_janitor_gate_failure_to_rework`, `_merge_train_candidates`,
  `_update_approval_head`, `_is_dispatchable`). Read-only against instance state.
  - Tier A sites (5 members, 6 sites): `_process_rescue_review`
    (`tests/test_charlie_work.py`, 2), `_maybe_probe_quota_recovery` (1),
    `_maybe_reclaim_worktrees` (1), `_maybe_reconcile_drift` (1),
    `_maybe_reclaim_superseded_main_ci` (1).
  - Tier B sites (4 members, 9 sites): `record_review` (4),
    `_route_rework_candidate_to_review` (3), `_maybe_reconcile_drift` (1),
    `_maybe_reclaim_superseded_main_ci` (1).
  - Tier D names (12) bound through `charlie_work.workflow`:
    `_build_rework_issue_fetch_skip_payload`, `_calculate_patch_id`,
    `_detect_and_handle_stalled_sessions`, `_next_round_number`, `clean_worktrees`,
    `dispatch_sessions`, `emit_digest`, `linked_issue_number`,
    `reclaim_superseded_main_ci_runs`, `run_quota_probe`,
    `salvage_push_stranded_commits`, `state_lock`.
  - Over-cap: `_dispatch_rework_impl` (1648 loc) and `record_review` (923 loc)
    each exceed 800; the module housing each is a residual-4 convention exceedance.
- **L02 -- helpers/unrouted (32, 4 batches; 985 loc, >=2 modules).**
  Orchestration glue with no single dominant domain import ->
  `orchestration/helpers_*`. Highest hub density; the installed shims are reached
  by other OA bodies via `self._x()` -> installed shim, which works with no
  back-resolution (Section 3.1 rule 1). Land after L01 so the state shims those
  helpers call already exist.
  - Tier A: none. Tier B (2 members, 5 sites): `_comment_pr` (4), `_is_base_current`
    (1). Tier D names: none.
- **L03 -- misc singletons (25, 3 batches; 1663 loc, >=3 modules).** The 17
  singleton/pair domains -> `orchestration/misc_*`; each member routes to its own
  domain module's free functions through `_wf`.
  - Tier A: none. Tier B (2 members, 3 sites): `reconcile` (2), `_reconcile_locked`
    (1). Tier D names (4): `_calculate_patch_id`, `_collect_escalated_label_subjects`,
    `remove_review_checkout`, `state_lock`.
- **L04 -- github (12, 2 batches; 1369 loc, >=2 modules).** Wrappers over
  `self.gh` -> `orchestration/github_ops_*`. `self.gh` stays an owner attribute
  on the instance; a moved body reaches it as `self.gh` because it runs with the
  OA instance bound as `self` (Section 3.1 rule 1) -- no `__getattr__` needed.
  - Tier A: none. Tier B (1 member, 3 sites): `review_queue` (3). Tier D names (2):
    `detect_cross_pr_revert`, `linked_issue_number`.
- **L05 -- instrumentation (8, 1 PR; 707 loc, 1 module).**
  - Tier A: none. Tier B (1 member, 1 site): `ensure_labels` (1). Tier D names:
    none in the intersection; still run Round 2 for any `emit_digest`-adjacent
    body.
- **L06 -- dead_worker_reap (4, 1 PR; 1025 loc, >=2 modules).**
  - Tier A: none. Tier B (1 member, 3 sites): `_loop_body` (3). Tier D names (6):
    `_classify_dead_sessions_and_update_throttle_state`,
    `_detect_and_handle_orphaned_workers`, `_detect_and_handle_stalled_sessions`,
    `_sweep_orphan_processes_for_dead_sessions`, `emit_digest`,
    `linked_issue_number`. Round 2 is the gate here.
- **L07 -- prompts (3, 1 PR; 91 loc, 1 module).**
  - Tier A/B: none. Tier D names (1): `render_prompt`.
- **L08 -- extract dispatch-state (1: `_dispatch_impl`; 1795 loc, over cap).** The
  one member that writes `_worker_token_escalated` outside `__init__` (at
  `workflow.py:5156` and `:5230`) and the one with a `self`-closure hazard. Two
  landing options, operator's call:
  - *(default in the projection)* move it as a free function like the rest; the
    flag stays `self._worker_token_escalated` (an instance attribute set on the OA
    instance the body runs against), so a byte-identical move keeps the two writes
    working with no state relocation -- the "extract class" framing of the first
    draft is unnecessary under free functions because there is no collaborator to
    own the flag. Delta -1 -> member_count 10 at this step.
  - *(simpler)* leave `_dispatch_impl` as a `def` and it joins the residual
    (member_count 8, still under fence). Recommended if the verbatim move trips the
    AST gate on the `self`-closure.
  - Tier A/B: none. Tier D names (6): `_detect_and_handle_stalled_sessions`,
    `_worker_pid_alive`, `dispatch_sessions`, `emit_digest`, `linked_issue_number`,
    `state_lock`. Over-cap: 1795 loc -> its module is a residual-4 exceedance.
- **L09 -- adapter (3: `layout`, `_is_dead_blocker`, `_write_json`; 65 loc, 1
  module).** Property/staticmethod adapter-wrapped assignments (Section 3.3).
  Lands member_count at **7 -- strictly under the fence.**
  - Tier A/B: none. Tier D names: none.

**Residual kept as `def` (7):** `__init__` plus the six `@_guard_state_lock`
commands (`dispatch_reviews`, `intake`, `loop`, `merge_ready`, `review`,
`status`). These are the class's genuine orchestration role -- the guarded public
command entrypoints -- and are the *named residual set* the umbrella exit permits.
They stay in `workflow.py` (4,635 body loc), so they create no new module. A
guarded `def` still counts as a member (verified 8.4e), which is why the residual
is 7 not lower; `_guard_state_lock` preserves `__name__` and the `__wrapped__`
chain (verified 8.4e), so each command could *optionally* be adapter-wrapped as
`name = _guard_state_lock(_delegate(...))` (a class-level `Assign`) to push the
count toward 1 while preserving the StateLockBusy guard -- the plan keeps them as
`def`s because they are the legible command surface, and 7 is already under fence.

**Recommendation on `dispatch` (review Finding 7).** `dispatch` is the most
externally-referenced member name in the src tree (`external_references_in_src`
attributes **up to 98 call sites across 10 files** to it -- an upper bound: the
resolver matches by bare attribute name, so the 98 conflates real `app.dispatch`
calls with `self.config.dispatch`/`ctx.config.dispatch` references to a different
object and `workflow.py`'s own same-module `self.dispatch` calls; the resolver
limitation is in Provenance). **Recommended: convert `dispatch` as a delegate,
not the residual.** Reason: the deterministic residual criterion is the
`@_guard_state_lock` decorator, and `dispatch` does not carry it; keeping the
criterion uniform (rather than hand-adding `dispatch` as an exception) is the
rule-9 discipline the whole plan follows, and conversion is free of external-caller
cost -- a class-level assignment resolves every `.dispatch` access transparently
(Section 8.1), so it needs **zero external repoints regardless of the exact
count**. It therefore sits in L03/L04's delegate set. The operator may instead
elect to keep `dispatch` a legible command in the residual (member_count 8, still
under fence); that is a legibility preference, not a correctness need.


## 6. Mikado graph

Goal node: **OrchestratorApp member_count <= 8 (strictly under fence 8.5).** The
graph is a star: L00 is the shared prerequisite for all move leaves; the DAG call
graph and single-writer state (Section 3) remove hard inter-leaf edges, so
L01-L09 depend only on L00. The descending order is a blast-radius/ratchet-visibility
preference, not a data edge, with two soft edges noted below.

ASCII tree (goal at root; leaves are the work; `(-N -> M)` = members converted
-> member_count after):

```
GOAL: OrchestratorApp member_count <= 8 (fence 8.5); start = 133
|
+-- L00 machinery (workflow_delegation.py + orchestration/ pkg; moves 0)  [prereq of all]
    |
    +-- L01 state              (-38 -> 95)   [4 batch-PRs ~10]
    +-- L02 helpers/unrouted   (-32 -> 63)   [4 batch-PRs ~8]  ..soft.. L01
    +-- L03 misc singletons    (-25 -> 38)   [3 batch-PRs ~9]
    +-- L04 github             (-12 -> 26)   [2 batch-PRs ~6]
    +-- L05 instrumentation    (-8  -> 18)
    +-- L06 dead_worker_reap   (-4  -> 14)
    +-- L07 prompts            (-3  -> 11)
    +-- L08 dispatch state     (-1  -> 10)   [the one state writer; free-fn or residual]
    +-- L09 adapter            (-3  -> 7)    [UNDER FENCE]
```

Adjacency (prerequisite -> dependent; "soft" = advisory ordering only):

| edge | type | reason |
|------|------|--------|
| L00 -> L01..L09 | hard | delegate machinery must exist first |
| L01 -> L02 | soft | helper bodies call state shims; both resolve on the owner regardless of order |
| (none) among L01..L07 | -- | DAG call graph + single-writer state remove hard edges |

Machine-readable graph, Mermaid, and indented-text renderings:
`mikado-graph.json`, `mikado-graph.mermaid.txt`, `mikado-graph.indented.txt`
(archived in Provenance). Each move-PR = exactly one batch: verbatim moves only;
if a body must be edited to move (Section 3.1 rule-2 namespace rebind), that batch
splits into a move-PR then an edit-PR.


## 7. Ordering rationale

The leaf order is by descending member count so the committed ratchet descends
fast and visibly (133 -> 95 -> 63 -> 38 after the first three leaves), front-loading
the biggest count-drops. The DAG call graph and the single instance-state writer
(Section 3) mean L01-L07 have no hard data edges among themselves, so any
permutation is technically valid. Two deliberate deviations from pure size order:
L08 (`_dispatch_impl`, the one state-writing member) lands late because it is the
one body with a `self`-closure hazard and benefits from all its callees already
being shims; L09 (adapter) lands last so the final, smallest conversion is the one
that crosses the fence, making the fence-crossing PR trivially reviewable.


## 8. Honest end-state verdict

**Question:** can `OrchestratorApp` reach <= 8 lexical members while every test
(all four patch tiers) keeps passing and the public command/attribute surface
keeps working?

**Answer: YES -- member_count = 7 -- via module-level free functions installed as
class-level assignment shims, plus a named 7-member residual. But this is the
metric exit with a disclosed residual, not the god object dismantled.** Both
halves are stated below without spin.

### 8.1 What the free-function facade buys, verified

- **Metric drop is real and mechanical.** 133 -> 7 by converting 126 `def`s to
  module-level free functions installed as class-level assignments across ~19
  ratcheted PRs (`mikado.py`).
- **Zero new attachment points.** The moved bodies land in free-function modules,
  which are not `class` (or any) attachment points (Section 2.4, verified 8.4a).
  Nothing new enters any fence population; `baseline.compare()` never sees a new
  saturated point. This is the whole reason the shape is free functions and not
  collaborator classes (Section 2.5).
- **All four patch tiers keep intercepting.** Tier A (6) and Tier B (36) member
  patches are unaffected (Section 4, verified 8.4b). Tier D (140) is preserved by
  the module-namespace rule (Section 3.1 rule 2, verified 8.4c) and gated by the
  Round-2 audit. Tier C is 0.
- **Zero external-caller repoints for the metric exit.** The externally-referenced
  members (`analyze_orchestratorapp.py`, `external_references_in_src` flags 23
  names, of which `__init__` is a `super().__init__()` name-collision false
  positive, leaving 22 real) -- including `dispatch` (up to 98 sites / 10 files;
  upper bound, see Provenance), `review`, `status`, `operator_queue` -- keep
  working as class-level assignments, because `app.dispatch(...)` resolves the
  installed function identically to a method (verified 8.4b). This holds
  regardless of the exact per-member site counts, and is a strict advantage over
  the GitHub plan, whose full exit required retyping 183 DI/attribute call sites.
- **The bodies genuinely leave.** The god-object logic moves to independently
  importable and testable free functions; that is the real architectural gain,
  independent of the metric.

### 8.2 What survives (the disclosed residuals -- do not read as a win)

Umbrella #1582's Phase B exit condition is "no facade shim survives Phase B exit
(vulture-class sweep)." **This plan produces 126 surviving facade shims** (the
class-level assignment delegates) plus the 7-member `def` residual. The class
still advertises 133 reachable names; only their implementations moved. The
metric correctly reports 7 (it measures lexical `def`s), but a static reader of
`OrchestratorApp` sees 126 forwarding shims, not a 7-method class. That is the
opposite of the stated vulture-sweep exit, and this document does not claim
otherwise.

The operator is accepting four residuals:

1. **Attribute-surface residual.** 126 names still resolve on `OrchestratorApp`
   via installed assignments. Exit path in 8.3.
2. **Hot-path indirection.** Every delegated call crosses one installed-shim hop
   until the shims are deleted. (Note: this is *one* hop, not two -- the
   free-function shape has no collaborator `__getattr__` back-resolution, so it is
   strictly cheaper than the first draft's collaborator shape.)
3. **Legibility residual.** The class file is smaller but the class's public
   surface is unchanged, and -- because the installer is derived, not literal
   (Section 3.2 decision B) -- the surface is not statically visible in
   `workflow.py`; a reader must follow `_install_delegates` and the
   `orchestration/` modules to find an implementation.
4. **Module-size-convention residual (no CI gate).** The move creates ~23 new
   `orchestration/` modules. There is no `module` fence and no file-length gate
   (Section 2.4), so none is an enforcement violation -- but **three modules
   exceed the repo's 800-line file convention** because a single relocated body
   does: the `_dispatch_impl` module (1795 loc), the `state_*` module housing
   `_dispatch_rework_impl` (1648 loc), and the `state_*` module housing
   `record_review` (923 loc). A verbatim move cannot split a body, so these cannot
   be brought under 800 at move time without a separate refactor of the method
   itself (out of scope for a delegation PR). Disclosed as convention debt, not
   hidden; the alternative (editing the body to split it during the move) would
   violate the AST verbatim-move gate.

### 8.3 The full-exit path (recommended as a scoped follow-on, not Phase B)

To satisfy the vulture-sweep exit -- delete the shims -- Phase B would
additionally need to:

1. Repoint the **6 Tier A** + **36 Tier B** patch sites onto the free functions
   (a test-side change per moved member), and
2. Repoint the **src-tree consumers** of the 22 externally-referenced members
   (up to ~98 `dispatch` sites, etc.; counts are upper bounds -- see Provenance)
   onto the free functions or a narrowed command surface, and
3. Delete the installed assignments and run the vulture-class sweep.

Step 2 is the expensive part -- the same shape as the deferred 27-site
`linked_issue_number` repoint carried over from Phase A, and calibrated against
#1449 (27 names moved over 11 days). **Recommendation:** land Phase B as the
metric-exit-with-named-residual (member_count 7, all tiers green, zero external
repoints, zero new attachment points), ratchet the baseline to 7, and file the
full-exit (shim deletion + call-site repoint) as an explicit follow-on umbrella
child. Presenting 7 as "dismantled" would be false; presenting it as "bodies
extracted to free functions, count ratcheted, shims scheduled for deletion" is
accurate.

If the operator rejects the facade entirely (no surviving shims *at all* in
Phase B), the honest answer flips: `OrchestratorApp` **cannot** reach <= 8
lexical members in one phase while keeping all 133 names callable, because
presenting 133 names at <= 8 `def`s is only possible by installing the rest as
non-`def` attributes. The choice is facade-with-scheduled-deletion (recommended)
or a multi-month call-site migration before any count drops.

### 8.4 Verified sub-claims (second-author synthetic verification)

Run in the worktree venv at head `20b934d0` (`verify.py`, `probe_module2.py`,
`reconcile.py`; scratch cyclic package `pkgcyc/` for the cycle case). Each is a
positive result with the control noted:

- **(a) class-level assignment counts 0; a free-function module is not a point.**
  A synthetic class with 2 `def`s + 3 class-level assignments (a plain assign, a
  `property(...)`, a `staticmethod(...)`) scans to `member_count == 2`
  (`probe_module2.py`/`verify.py`). Positive control: a module of 15
  realistically-named free functions scans to **0** attachment points; the same 15
  names as class methods scan to **1** `class` point saturated at fence 8.5. So
  the metric fact and the no-module-fence fact are both empirically grounded.
- **(b) all three member-patch forms intercept a class-level function assignment.**
  With `OrchestratorApp.m = mod.m` installed, `patch("...OrchestratorApp.m")`,
  `patch.object(app, "m")`, and `monkeypatch.setattr(app, "m", ...)` all
  intercept, and an unpatched `app.m()` binds the instance as `self` (descriptor
  protocol) -- confirming Tier A and Tier B survive with no adapter.
- **(c) Tier D module-namespace binding is cycle-safe.** A synthetic `workflow`
  module imports a domain module at import time; the domain module does `import
  ...workflow as _wf` and a moved body calls `_wf.target()`. The cyclic import
  completes with no `ImportError`, and `patch("...workflow.target")` reaches the
  moved body (returns the patched value); the unpatched call returns the real
  value. This is the empirical basis for Section 3.1 rule 2 and the Finding-4
  cycle-safety claim.
- **(d) property/staticmethod adapters work.**
  `layout = property(mod.layout_body)` yields `app.layout` == the body's return;
  `_write_json = staticmethod(mod.write_json_body)` yields
  `OrchestratorApp._write_json("x")` == the body's return.
- **(e) the guarded residual is unchanged.** A guarded `def review` still counts
  as a `class` member (so the residual stays at 7), and `_guard_state_lock`
  (a `functools.wraps` guard) preserves `__name__` and the call semantics.
- **Baseline vs live reconciliation (`reconcile.py`).** The live APC scanner
  reports `OrchestratorApp` member_count **133**; the committed baseline entry is
  **130** with a bump `to: 131` (effective ceiling **131**). See Section 9 for
  what this +2 gap means for the first ratchet PR.
- **Fence numbers and OA-reduction stability:** `fence_probe.py` (Section 2.4).
- **Tier counts and controls:** `patch_census.py` (Section 2.3), Tier A
  cross-checked against raw grep.
- **Single state writer, no SCCs, clusters, external refs, per-member line spans:**
  `analyze_orchestratorapp.py`; per-leaf sizing and re-seam: `sizing.py`,
  `reseam.py` (Section 5).


## 9. Ratchet, gate, and stop-conditions

- **Baseline-vs-live gap the first PR must reconcile.** The committed
  `.attachment-budgets.json` OrchestratorApp entry is member_count **130** with a
  bump to **131** (effective ceiling 131), while the live scanner reads **133**
  (`reconcile.py`). The tree is thus already **2 over** its committed ceiling --
  a pre-existing drift, not created by this plan. L00 (machinery, moves 0) must
  therefore first **re-baseline the committed value to the true live 133** (a
  `baseline --ratchet` that records reality) before any leaf ratchets it *down*;
  otherwise the first move-PR's "lower to the post-move value" starts from a stale
  130/131 that does not match live. After L00 re-baselines to 133, every leaf
  ratchets down from there (133 -> 95 -> ... -> 7). The operator should read the
  umbrella text as: the exit criterion is the **committed ratchet baseline value**
  (which only descends after L00's one-time reality-sync), not the umbrella's
  originally-written "strictly under 6.0 live" -- see the fence-target note below.
- **Fence-target reconciliation (review Finding 5).** Umbrella #1582's Phase B
  exit was written as "strictly under 6.0 live." The live `class` fence is now
  **8.5** (`fence_probe.py`, Section 2.4), because the eligible `class` population
  changed since #1582 was written; the fence is recomputed per scan. **8.5
  supersedes the umbrella's stale 6.0 for the live target**, and in any case the
  enforced exit criterion is the committed baseline ratchet value (which the plan
  drives to 7), not a live-recomputed fence. The operator should update the
  umbrella text to read 8.5 (or, more robustly, "the current live `class` fence")
  so no one compares against 6.0.
- **Ratchet as definition of done.** Each move-PR (each batch) lowers the
  committed OrchestratorApp `member_count` to its post-move value in the same PR
  and runs `uv run python -m charlie_work.attachment_contracts baseline --ratchet`
  so the budget can only descend. Final committed value: 7 (or 8 if L08 is
  deferred to the residual).
- **AST-equivalence gate (review Finding 2, the #1607 call shape).** Every move-PR
  must prove the moved `FunctionDef` is byte-identical between removal site and
  destination. The CLI is broken (#1600), so invoke the gate's **library entry
  points** directly:
  `charlie_work.ast_equivalence_gate.extract_symbols` on the **base** tree and on
  the **head** tree, then `derive_moved_symbols` over the two symbol dicts. Per
  #1607, the destination module MUST already be present in the **base** symbol
  dict (extract it from the base tree *including* the not-yet-populated
  destination module) before calling `derive_moved_symbols`, so the gate's
  first-non-stub preference (`ast_equivalence_gate.py:235-242`) correctly pairs
  each removed symbol with its destination rather than treating the destination as
  a brand-new stub. Assert the moved-symbol set matches and each body hashes equal.
  A body that cannot move verbatim (a Section 3.1 rule-2 namespace rebind, or
  L08's `self`-closure) splits into a move-PR + edit-PR.
- **#1621 count check.** Each move-PR runs the collected-vs-recorded test-count
  reconciliation (#1621): the number of tests pytest *collects* for the touched
  modules must equal the number the run *records*, so a silently-dropped test
  (e.g. a Tier D patch that stopped intercepting and left a test uncollected)
  cannot hide behind a green run.
- **Four-round monkeypatch audit** (Section 4) on every move-PR.
- **workflow.py review lane.** Every PR touching `workflow.py` is opus48-reviewed
  (highest blast radius in the campaign -- it is the fleet's own orchestrator).

**Stop conditions (escalate, do not push through):**
- The AST gate reports a non-verbatim move that is not a deliberate edit-split.
- A Tier D test goes red -- indicates a module-namespace rebind (Section 3.1 rule
  2); the fix is to reference the free function via `charlie_work.workflow`, never
  to edit the test.
- A Tier A/B member patch fails to resolve to its installed shim -- indicates the
  installer missed the name (check `_build_routes` collision handling).
- Any Tier C subclass appears (couples a leaf to a subclass).
- A domain module raises `ImportError` at import (indicates a `from
  charlie_work.workflow import ...` slipped in against the Section 3.1 rule-2
  module-object form).

**Merge-lane finding (operator action required).** The live charlie-work
`pyproject.toml` has **no** `dispatch.human_merge_labels` configured. In this
repo's dispatch logic, a `needs-design` PR whose `agent:merge-hold` label is
removed will **auto-merge** -- there is no human-merge gate keyed on
`needs-design`. Therefore the `agent:merge-hold` label on this design PR (and on
every Phase B leaf PR) MUST stay until the operator merges by hand; removing it
does not route to human review, it releases the PR to the merge queue. This is a
lane property, not a per-PR choice, and every leaf issue repeats it.


## 10. Provenance

Analysis scripts and outputs produced in this pass (archived under llibrary
`raw/analyses/2026-09-god-object-paydown/track2-phaseb/`), all measured at base
`eb634c9b319462955984cce9452a3660497c901d` (`workflow.py` blob `ae92e6e0`,
byte-identical at design-PR head `20b934d0`):

- `analyze_orchestratorapp.py` -> `orchestratorapp-inventory.json` / `.md` --
  133 members, 0 async, 0 fields, 3 class Assigns, per-member line spans, self
  attr reads/writes/calls (separated), `self.gh` usage, resolved domain-module
  calls, decorators, domain clusters, in-degree/hub ranking, SCCs (none),
  single instance-state writer, 62 pure leaves, external src references (23
  member names).
  - **Resolver limitation (external_references_in_src).** The external-reference
    counts are matched by **bare attribute name** and are therefore **upper
    bounds**, not exact external call counts: (a) the same method name on a
    different object is counted -- `dispatch`'s 98 sites carry receivers
    `{app, config, ctx.config, self.config}`; (b) same-module `self.<name>` calls
    inside `workflow.py` are counted; (c) dunder collisions -- `__init__`'s 14
    hits are all `super().__init__()`, so the 23-name list is 22 real
    externally-referenced members. None of this changes the load-bearing claim:
    attribute access resolves a class-level assignment transparently, so metric
    exit needs zero external repoints **regardless of the exact counts**.
- `patch_census.py` -> `patch-census.json` / `.md` -- four-tier patch census
  (A 6/5, B 36/13, C 0/0, D 140/21), B_unresolved 0 (loud bucket), positive
  controls (`_process_rescue_review` 2, `review` 11), 117 no-patch members.
- `mikado.py` -> `mikado-graph.json` + `.mermaid.txt` + `.indented.txt` --
  10-leaf projection, 133 -> 7, ~19 batch-PRs, 7-member residual.
- `fence_probe.py` -- live fence 8.5 (pop 46, Q1 1.0, Q3 4.0, IQR 3.0),
  OA-reduction stability, guarded-command adaptability.
- `sizing.py` -- per-leaf body-loc sums and >=800-line module counts; the three
  over-cap single members (`_dispatch_impl` 1795, `_dispatch_rework_impl` 1648,
  `record_review` 923).
- `reseam.py` -- per-leaf Tier A/B member-patch sites and Tier D referenced names
  (Section 5).
- `probe_module2.py` -- positive control: free-function module 0 points vs
  same-names class 1 saturated point.
- `reconcile.py` -- live scanner 133 vs committed baseline 130 / bump 131.
- `verify.py` + scratch package `pkgcyc/` -- synthetic verification (a)-(e),
  Section 8.4.

Primary-source verification performed in this pass:
- Metric definition: `src/charlie_work/attachment_contracts/archetypes.py`
  (`_is_def`, members comprehension); no `module` archetype -- `model.py:13-20`
  `Kind`, and `model.py:5-6` "no line count is ever read".
- Fence machinery: `src/charlie_work/attachment_contracts/outliers.py`
  (`saturate`, `_quartiles`, FLOOR).
- New-saturated-point block: `baseline.py:331-363` (compare() blocks a brand-new
  saturated AP with no baseline entry) -- the reason collaborator classes were
  rejected (Section 2.5).
- Class shape: `workflow.py` `OrchestratorApp` bases/decorators (both empty),
  `_guard_state_lock` at line 909.
- Tier A cross-check: raw grep over `tests/` (Section 2.3).
- Tier D mechanism: `patch("charlie_work.workflow._worker_pid_alive")` sites and
  the single OA referrer `_dispatch_impl` (Section 3.1).

Leaf issues (one `needs-design` issue each, blocked in sequence under #1582 /
#1628): see the leaf-issue drafts produced alongside this design. This document
files none of them.


## 11. Rework after adversarial review (PR #1629)

The first draft used **collaborator classes**. The adversarial review's Finding 1
(HIGH) showed that shape trades one saturated `class` for three new saturated
`class` points (`State`@38, `OrchestrationHelpers`@32, `MiscDomainDelegates`@25),
each a `baseline.compare()` hard block, and that the "honest verdict" neither
disclosed this nor was consistent with the Section 2.4 "~4-12" claim. The rework
adopts the umbrella #1582 shape -- **module-level free functions installed as
class-level assignments** -- which the second author verified creates **zero new
attachment points** (there is no `module` archetype/fence; a free-function module
scans to zero points; positive control in 8.4a). This dissolves Finding 1 at the
root rather than mitigating it. Per-finding resolution:

- **F1 (HIGH):** shape changed to free functions -> zero new saturated points
  (Sections 1, 2.5, 8.1). The "~4-12 extracted classes" claim is **deleted** (no
  classes exist); the population-growth fence caveat is **deleted** (no classes
  appended). New residual 4 discloses the three over-800-line modules a verbatim
  move forces (8.2).
- **F2 (LOW):** #1607 gate call shape spelled out in Section 9 (destination module
  in the base symbol dict before `derive_moved_symbols`, library entry points, CLI
  unused per #1600).
- **F3 (LOW):** per-leaf Tier A/B re-seam sites and Tier D referenced names
  enumerated in Section 5, with a named destination module per leaf.
- **F4 (LOW):** circular-import hazard named and shown cycle-safe by the
  module-object import form (Section 3.1, verified 8.4c).
- **F5 (NIT):** 8.5 supersedes the umbrella's stale 6.0; Section 9 says what the
  operator should read.
- **F6 (NIT):** "one mutated attribute (`_worker_token_escalated`) in one method
  (`_dispatch_impl`), at two branch sites (`workflow.py:5156`, `:5230`)" -- fixed
  in Sections 3 and 5.
- **F7 (NIT):** `dispatch` recommendation given (delegate, not residual), with the
  uniform-criterion + zero-repoint rationale, operator's final call (Section 5).
