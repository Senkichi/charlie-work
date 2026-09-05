# OrchestratorApp: Mikado Graph and Delegation Plan (Track 2 Phase B design)

Status: DESIGN ONLY. This document moves no code. It is the single planning
artifact for Track 2 Phase B of the god-object paydown umbrella (charlie-work
#1582), produced under the Phase B prep issue (#1628). Every leaf named here is
drafted as its own `needs-design` issue (see the leaf-issue drafts referenced in
Provenance); this document does not file them.

Author: synthesis/adjudication pass over three analysis artifacts generated in
this pass (member inventory, four-tier patch census, Mikado projection). Every
number below cites the script that produced it and the base commit
`eb634c9b319462955984cce9452a3660497c901d` it was measured at. Where a figure
could be read two ways, the reconciled number and its file:line / script
evidence are stated inline.

Target: `src/charlie_work/workflow.py`, class `OrchestratorApp` (lines
4006-23450). Interpreter: CPython 3.13.5. Metric authority:
`.attachment-budgets.json` OrchestratorApp entry (`kind: class`) and the live
`class`-archetype Tukey fence recomputed at HEAD (Section 2.4).

This plan follows the merged GitHub-class plan
(`docs/design/2026-09-03-github-class-mikado-graph-and-protocol-segmentation.md`,
Phase A leaf L06b) and reuses its generated-delegate machinery pattern. The
structural differences between `GitHub` and `OrchestratorApp` -- which change the
plan in three load-bearing ways -- are stated in Section 2.2 and Section 8.


## 1. Summary

`OrchestratorApp` is an APC `class`-archetype god object: **133 lexical method
definitions** (`analyze_orchestratorapp.py`; AST over `workflow.py` at HEAD), by
a wide margin the largest class in the tree. The live `class` fence is **8.5**
(Section 2.4), so the goal is member_count <= 8.

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

The plan decomposes `OrchestratorApp` by domain into the #1283 collaborator
modules its methods already import from. The member *bodies* move to collaborator
classes; the member *names* remain resolvable on `OrchestratorApp` through
generated class-level delegate attributes. A generated delegate is an
**assignment**, not a `def`, so it contributes zero to the `class` metric while
keeping every instance patch (Tier B) and class patch (Tier A) intercepting.

End state (Section 8): the Mikado projection (`mikado.py`) lands
`OrchestratorApp` at **member_count = 7** -- `__init__` plus the six
`@_guard_state_lock`-guarded public command entrypoints
(`dispatch_reviews`, `intake`, `loop`, `merge_ready`, `review`, `status`) --
strictly under the fence of 8.5, across roughly 19 sequenced move-PRs.

**Honest verdict, stated up front (full detail in Section 8):** this reaches the
*metric* exit with a *named residual*. It does **not** dismantle the god object.
125 of the 133 names survive as forwarding delegate shims. The bodies genuinely
leave the class and become independently testable collaborators, and the
133 -> 7 count drop is real and mechanically ratchet-verifiable. But true Phase B
exit (umbrella #1582's "no facade shim survives" condition, enforced by a
vulture-class sweep) additionally requires repointing the Tier A/B call sites and
the src-tree consumers of the 22 externally-referenced members onto the
collaborators and then deleting the shims -- the expensive part, calibrated
against #1449 (27 names over 11 days) and deferred as a follow-on. This document
recommends the metric-exit-with-named-residual as Phase B's landing point and
scopes the full-exit as an explicit follow-on, rather than presenting 7 as "god
object dismantled."


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
- A class-level **assignment** installed after the class body
  (`OrchestratorApp.foo = <delegate>`, or an install loop) is not a
  `FunctionDef` child and does NOT count. This is the metric fact the facade
  exploits. It is a disclosed judgment call (Section 8.3), the same one taken and
  merged for `GitHub`.
- A `def` shim -- an explicit thin `def foo(self, ...): return self._collab.foo(...)`
  -- WOULD count. So thin `def` delegates cannot reduce the metric; only
  class-level assignment delegates can. This is the pivot of the whole plan.

Authoritative current count: **133 lexical defs**, 0 async
(`analyze_orchestratorapp.py`; enumerated with per-member line spans in the
inventory JSON). Class span: lines 4006-23450.

### 2.2 How OrchestratorApp differs from GitHub (three plan-changing facts)

| dimension | GitHub (Phase A) | OrchestratorApp (Phase B) | plan consequence |
|-----------|------------------|---------------------------|------------------|
| class shape | frozen `@dataclass`, `GitHubLike` Protocol | plain mutable class, no bases | no conformance test / `isinstance` to satisfy; instance (Tier B) patches work directly |
| dominant seam | `run` member (134 class/instance patch sites) | module-level free functions (140 Tier D sites) | preserve **module-namespace resolution**, not one member |
| shared state | `_list_cache` (owner-held, by reference) | one attr write outside `__init__` (`_worker_token_escalated`) | near-zero state coupling; call graph is a DAG (no SCCs) |

Because there is no Protocol, the delegate does **not** need
`functools.wraps` + `__signature__` for a conformance comparison. It is still
useful to `functools.wraps` the delegate so tracebacks and `inspect`-based test
assertions keep the original name (verified: `_guard_state_lock(_delegate)`
preserves `__name__` and the `__wrapped__` chain -- Section 8.4), but nothing in
the test suite *requires* signature fidelity the way
`tests/test_githublike_protocol.py` did for `GitHub`.

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

### 2.4 The fence and the exit target

`saturate()` (`src/charlie_work/attachment_contracts/outliers.py`) computes the
boundary per-kind as `Q3 + 1.5*IQR` over the eligible same-kind population
(excluding `is_linear_ledger`, `is_structurally_trivial`, and `member_count < 1`),
with nearest-rank quartiles and a statistical FLOOR of 4.

Recomputed live at HEAD over the actual scanned tree (`fence_probe.py`):
eligible `class` population **46**, Q1 **1.0**, Q3 **4.0**, IQR **3.0**, fence
**8.5**. So the exit target is **member_count <= 8** (strictly under 8.5;
`member_count > boundary` is the saturation test, so 8 is not saturated).

**Fence stability -- two directions, both measured (`fence_probe.py`):**

- *Under reduction* (OA's own count falls 133 -> 1): the fence is **invariant at
  8.5** at every step (OA is a single high outlier; dropping it does not move Q3
  of the other 45 points). So the projection's per-leaf target never moves as OA
  shrinks.
- *Under population growth* (Phase B adds collaborator classes): appending 3, 6,
  10, 15, 20 new eligible classes of 4-8 members each **raises** the fence
  (8.5 -> 11.0 -> 13.5), making 7 more comfortably under. **Caveat (the one case
  that bites):** appending 20+ classes of *uniformly exactly 4* members
  compresses the IQR (Q1 rises toward Q3) and drops the fence to 7.0; at that
  point member_count 7 is at the boundary (7 > 7.0 is False, so still not
  *saturated*, but the margin is gone). Realistic collaborators are
  size-varied (the domain clusters are 3-42 members, so their extracted classes
  span ~4-12), which is the fence-*raising* case -- but this shows the live fence
  is **not** an invariant under arbitrary population growth. Mitigation
  (Section 9): the exit criterion is the **committed ratchet baseline value**
  (which only descends), not a live-recomputed fence, so a later fence shift
  cannot silently un-satisfy a landed leaf.


## 3. Domain segmentation (to the #1283 collaborator modules)

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

Structural facts that make this clustering safe to cut in almost any order
(`analyze_orchestratorapp.py`):

- **No non-trivial SCCs** in the member call graph (`nontrivial_sccs: []`). The
  `self.<method>()` call graph is a DAG, so there is no mutual-recursion cluster
  that must move atomically.
- **Exactly one instance-state write outside `__init__`:**
  `_worker_token_escalated`, written only by `_dispatch_impl`
  (`instance_attr_writes_outside_init`). Every other member is read-only against
  instance state, so delegation carries read-only dependencies and the extract is
  clean. That single write is isolated into its own extract leaf (Section 5).
- **62 pure leaves** (members that call no other OA method) -- these are the
  lowest-risk conversions.

### 3.1 The two-step delegation shape

The seam mirrors GitHub's merged pattern, adapted for a non-Protocol class. The
machinery is a new module `src/charlie_work/workflow_delegation.py` (the analog
of the merged `github_delegation.py`), holding `_make_delegate`,
`_build_routes`, and `_install_delegates(owner_cls)`; `workflow.py` calls
`_install_delegates(OrchestratorApp)` once after the class body. Housing it in a
separate module (not inline) keeps `workflow.py`'s own size ratchet clear, the
same constraint that put `github_delegation.py` in its own file.

Two directions of resolution, each bounded so there is no recursion cycle:

1. **owner -> collaborator (explicit routing table, no `__getattr__`).**
   `OrchestratorApp` installs a class-level delegate for every moved name from an
   explicit `_ROUTES` dict (name -> collaborator attribute), built by iterating
   each collaborator class's non-dunder members at import time (CLAUDE.md rule 9
   -- no hardcoded lists; `_build_routes` raises on a cross-collaborator name
   collision instead of letting the last writer win). `OrchestratorApp` gets
   **no** `__getattr__`; a name not in `_ROUTES` and not a real attribute raises
   `AttributeError` normally. This termination prevents the cycle.

2. **collaborator -> owner (`__getattr__`, bounded).** A moved body is
   byte-identical, so it still says `self._record_event(...)`, `self.gh.pr_view(...)`,
   `self.config`. On a collaborator instance, `self.<name>` for a name the
   collaborator does not define triggers the collaborator's `__getattr__`, which
   forwards to `self._owner.<name>`. That resolves to a real owner attribute
   (`gh`, `config`, `paths`) or an owner delegate (another collaborator's method).
   Because the owner side is explicit and terminates, the chain always ends.

**The load-bearing constraint (this is what Tier D forces).** The 140 Tier D
sites patch module-level free functions in `charlie_work.workflow` -- e.g.
`patch("charlie_work.workflow._worker_pid_alive", ...)` (66 sites) and
`patch("charlie_work.workflow.emit_digest", ...)` (20 sites). These intercept
because an OA body calls the bare name `_worker_pid_alive(...)`, which resolves
through the `charlie_work.workflow` module global that the test patched. A moved
body relocated into `workflow_delegation.py` or a collaborator module must
therefore reference these free functions **through the `charlie_work.workflow`
module namespace** (e.g. `import charlie_work.workflow as _wf; _wf._worker_pid_alive(...)`,
or keep the import bound to that module), NOT via a fresh
`from charlie_work.workflow import _worker_pid_alive` re-imported into the new
module -- a fresh import binds a *new* name in the collaborator's namespace that
`patch("charlie_work.workflow._worker_pid_alive")` does not reach, silently
breaking 66 tests. This is exactly the #1627 module-level-patch-binding concern,
and it is the single sharpest design rule for Phase B. Note `_worker_pid_alive`
is referenced by only ONE OA method (`_dispatch_impl`); the 66 sites protect that
call plus the module-level callers of the same free function -- the delegation
must preserve the module binding regardless of which class the caller lives in.

### 3.2 Adapters (property / staticmethod)

Nine members are decorated (`analyze_orchestratorapp.py`): six with
`@_guard_state_lock` (the guarded commands -- Section 5 residual), one `@property`
(`layout`), two `@staticmethod` (`_is_dead_blocker`, `_write_json`). The
property and the two staticmethods convert to **adapter-wrapped class-level
assignments** (`layout = property(_delegate_layout)`,
`_write_json = staticmethod(_delegate_write_json)`) -- still `Assign`, not
`FunctionDef`, so they drop from the count while keeping `app.layout` and
`OrchestratorApp._write_json(...)` working. This is the adapter leaf.


## 4. Test-tier impact and the four-round monkeypatch audit

Because `OrchestratorApp` is a plain mutable class, an **instance** patch
(`monkeypatch.setattr(app, "m", fn)`, Tier B, 36 sites) sets an attribute on the
instance that shadows the class-level delegate -- it keeps working unchanged. A
**class** patch (Tier A, 6 sites) rebinds `OrchestratorApp.m`; since the delegate
is itself a class attribute, `patch.object(OrchestratorApp, "m", ...)` replaces
it identically to replacing a `def`. So both member-patch tiers survive the
conversion untouched. The tier that would break under a careless move is Tier D,
per Section 3.1.

**Four-round monkeypatch audit checklist (per move-PR, the #1449 precedent).**
Each round is a distinct gate (verification-ladder taxonomy in parentheses):

- **Round 1 -- Tier A/B member patches (pre-flight).** For every member the PR
  converts, grep `tests/` for `patch.object(OrchestratorApp, "<m>"`,
  `patch("...OrchestratorApp.<m>"`, and instance `setattr(<app>, "<m>"`. Confirm
  each still resolves to the installed delegate (class patch) or shadows it
  (instance patch). Expected: unaffected. Any miss escalates.
- **Round 2 -- Tier D module-namespace binding (revision).** For every
  module-level free function the moved body calls, confirm the relocated body
  references it via the `charlie_work.workflow` namespace, and that
  `patch("charlie_work.workflow.<fn>")` still intercepts the moved body (run the
  specific Tier D tests for that function; e.g. any move touching a
  `_worker_pid_alive` caller runs the 66-site `_worker_pid_alive` suite). This is
  the round that catches the Section 3.1 hazard.
- **Round 3 -- #1627 binding-style sweep (revision).** Confirm no relocated body
  introduced a `from charlie_work.workflow import <fn>` that rebinds a patched
  name into a new module namespace. AST-grep the moved module for
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

| # | leaf | converts | batches | member_count after | under fence? |
|---|------|----------|---------|--------------------|--------------|
| L00 | `workflow_delegation.py` machinery (moves 0) | 0 | -- | 133 | no |
| L01 | domain: `state` | 38 | 4 (~10/PR) | 95 | no |
| L02 | orchestration-helper delegates (unrouted) | 32 | 4 (~8/PR) | 63 | no |
| L03 | misc singleton-domain delegates | 25 | 3 (~9/PR) | 38 | no |
| L04 | domain: `github` | 12 | 2 (~6/PR) | 26 | no |
| L05 | domain: `instrumentation` | 8 | 1 | 18 | no |
| L06 | domain: `dead_worker_reap` | 4 | 1 | 14 | no |
| L07 | domain: `prompts` | 3 | 1 | 11 | no |
| L08 | extract: escalation/dispatch-state collaborator (`_dispatch_impl`) | 1 | 1 | 10 | no |
| L09 | adapter: property/staticmethod delegates | 3 | 1 | **7** | **yes** |

Per-leaf notes:

- **L00 -- machinery (moves 0).** Adds `workflow_delegation.py`
  (`_make_delegate`, `_ROUTES` builder, `_install_delegates`), an empty
  collaborator-package skeleton, and the post-class install call. Ratchet delta
  0 (still 133). Blast radius: import surface only. Revert: delete the module +
  the install call.
- **L01 -- state (38, 4 batches).** The dominant cluster. Read-only against
  instance state (none of the 42 writes `_worker_token_escalated`). Each batch
  converts ~10 members to `State`-collaborator delegates. 7 of these are hub
  members (in-degree >= 2), so their delegates are reached by other OA bodies via
  `self._x()` -> delegate -> collaborator; the Round-2 audit confirms Tier D
  bindings inside them.
- **L02 -- helpers/unrouted (32, 4 batches).** Orchestration glue with no single
  dominant domain import; extracted to an `OrchestrationHelpers` collaborator.
  16 of 32 are hubs (the highest hub density) -- these are the internal
  coordination points, so the collaborator `__getattr__` back-resolution
  (Section 3.1) is exercised most here; land after L01 so the state delegates
  those helpers call already exist.
- **L03 -- misc singletons (25, 3 batches).** The 17 singleton/pair domains
  folded into one `MiscDomainDelegates` collaborator to keep the leaf count
  reviewable; each member still routes to its own domain module's free functions.
- **L04 -- github (12, 2 batches).** Wrappers over `self.gh`. `self.gh` stays an
  owner attribute; moved bodies reach it via collaborator `__getattr__` ->
  `self._owner.gh`. No Tier D exposure (these call `self.gh.*`, a real attribute,
  not module free functions).
- **L05-L07 -- instrumentation (8), dead_worker_reap (4), prompts (3).**
  Single-PR leaves. `instrumentation` and `dead_worker_reap` carry Tier D
  exposure (`emit_digest`, `_worker_pid_alive` families) -- Round 2 is the gate.
- **L08 -- extract escalation/dispatch state (1: `_dispatch_impl`).** The one
  member that writes `_worker_token_escalated` outside `__init__`, and the one
  with a `self`-closure hazard. It is NOT a mechanical assignment-delegate: it
  moves as an Extract Class where the collaborator owns the
  `_worker_token_escalated` flag, or (simpler) `_dispatch_impl` stays a `def` and
  is deferred out of the count reduction. Projection keeps it as a delegate for
  the -1; if the operator prefers, it joins the residual (member_count 8, still
  under fence). Flagged as the one non-trivial extract.
- **L09 -- adapter (3: `layout`, `_is_dead_blocker`, `_write_json`).**
  Property/staticmethod adapter-wrapped assignments (Section 3.2). Lands
  member_count at **7 -- strictly under the fence.**

**Residual kept as `def` (7):** `__init__` plus the six `@_guard_state_lock`
commands (`dispatch_reviews`, `intake`, `loop`, `merge_ready`, `review`,
`status`). These are the class's genuine orchestration role -- the guarded public
command entrypoints -- and are the *named residual set* the umbrella exit permits.
`_guard_state_lock` uses `functools.wraps`, so each can *optionally* be
adapter-wrapped as `name = _guard_state_lock(_delegate(...))` (a class-level
`Assign`) to push the count toward 1 while preserving the StateLockBusy guard and
the member name (verified, Section 8.4); the plan keeps them as `def`s because
they are the legible command surface, and 7 is already under fence.

Note on `dispatch`: `dispatch` is the most externally-referenced member name in
the src tree. The analyzer's `external_references_in_src` attributes **up to 98
call sites across 10 files** to it -- an upper bound, not an exact external count:
the resolver matches by bare attribute name, and the receiver set for these sites
is `{app, config, ctx.config, self.config}`, so the 98 conflates real
`app.dispatch(...)` calls with `self.config.dispatch`/`ctx.config.dispatch`
references to a *different* object and with `workflow.py`'s own same-module
`self.dispatch(...)` calls (the file list includes `workflow.py` itself). The
resolver's name-collision limitation is stated in Provenance. `dispatch` is NOT
`@_guard_state_lock`-decorated, so the deterministic criterion places it in
L03/L04's delegate set, not the residual. Converting it to a class-level delegate
needs **zero external-caller repoints regardless of the exact count** -- every one
of those `.dispatch` accesses resolves the class-level delegate transparently
(Section 8.1). Whether `dispatch` should instead join the residual as a legible
command is an operator call; the mikado criterion is the guard decorator, applied
uniformly.


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
+-- L00 machinery (workflow_delegation.py; moves 0)        [prereq of all]
    |
    +-- L01 state              (-38 -> 95)   [4 batch-PRs ~10]
    +-- L02 helpers/unrouted   (-32 -> 63)   [4 batch-PRs ~8]  ..soft.. L01
    +-- L03 misc singletons    (-25 -> 38)   [3 batch-PRs ~9]
    +-- L04 github             (-12 -> 26)   [2 batch-PRs ~6]
    +-- L05 instrumentation    (-8  -> 18)
    +-- L06 dead_worker_reap   (-4  -> 14)
    +-- L07 prompts            (-3  -> 11)
    +-- L08 extract dispatch   (-1  -> 10)   [Extract Class; the one state writer]
    +-- L09 adapter            (-3  -> 7)    [UNDER FENCE]
```

Adjacency (prerequisite -> dependent; "soft" = advisory ordering only):

| edge | type | reason |
|------|------|--------|
| L00 -> L01..L09 | hard | delegate machinery must exist first |
| L01 -> L02 | soft | helper delegates call state delegates; both resolve via owner regardless of order |
| (none) among L01..L07 | -- | DAG call graph + single-writer state remove hard edges |

Machine-readable graph, Mermaid, and indented-text renderings:
`mikado-graph.json`, `mikado-graph.mermaid.txt`, `mikado-graph.indented.txt`
(archived in Provenance). Each move-PR = exactly one batch: verbatim moves only;
if a body must be edited to move (Section 3.1 namespace rebind, or L08's state
extract), that batch splits into a move-PR then an edit-PR.


## 7. Ordering rationale

The leaf order is by descending member count so the committed ratchet descends
fast and visibly (133 -> 95 -> 63 -> 38 after the first three leaves), front-loading
the biggest count-drops. The DAG call graph and the single instance-state writer
(Section 3) mean L01-L07 have no hard data edges among themselves, so any
permutation is technically valid. Two deliberate deviations from pure size order:
L08 (the `_dispatch_impl` state extract) lands late because it is the one
non-mechanical move and benefits from all its callees already being delegates;
L09 (adapter) lands last so the final, smallest conversion is the one that
crosses the fence, making the fence-crossing PR trivially reviewable.


## 8. Honest end-state verdict

**Question:** can `OrchestratorApp` reach <= 8 lexical members while every test
(all four patch tiers) keeps passing and the public command/attribute surface
keeps working?

**Answer: YES -- member_count = 7 -- via generated class-level delegate
attributes plus a named 7-member residual. But this is the metric exit with a
disclosed residual, not the god object dismantled.** Both halves are stated
below without spin.

### 8.1 What the delegate facade buys, verified

- **Metric drop is real and mechanical.** 133 -> 7 by converting 126 `def`s to
  class-level assignment delegates across ~19 ratcheted PRs (`mikado.py`).
- **All four patch tiers keep intercepting.** Tier A (6) and Tier B (36) member
  patches are unaffected (Section 4). Tier D (140) is preserved by the
  module-namespace rule (Section 3.1) and gated by the Round-2 audit (Section 4).
  Tier C is 0.
- **Zero external-caller repoints for the metric exit.** The externally-referenced
  members (`analyze_orchestratorapp.py`, `external_references_in_src` flags 23
  names, of which `__init__` is a `super().__init__()` name-collision false
  positive, leaving 22 real) -- including `dispatch` (up to 98 sites / 10 files;
  upper bound, see Provenance), `review`, `status`, `operator_queue` -- keep
  working as class-level delegates, because `app.dispatch(...)` resolves the
  delegate identically to a method. This conclusion holds regardless of the exact
  per-member site counts, and is a strict advantage over the GitHub plan, whose
  full exit required retyping 183 DI/attribute call sites.
- **The bodies genuinely leave.** The god-object logic moves to collaborator
  classes that are independently constructable and testable; that is the real
  architectural gain, independent of the metric.

### 8.2 What survives (the disclosed residual -- do not read as a win)

Umbrella #1582's Phase B exit condition is "no facade shim survives Phase B exit
(vulture-class sweep)." **This plan produces 125 surviving facade shims** (the
class-level assignment delegates) plus the 7-member `def` residual. The class
still advertises 133 reachable names; only their implementations moved. The
metric correctly reports 7 (it measures lexical `def`s), but a static reader of
`OrchestratorApp` sees 125 forwarding shims, not a 7-method class. That is the
opposite of the stated vulture-sweep exit, and this document does not claim
otherwise.

The operator is accepting three residuals (mirroring the merged GitHub decision):

1. **Attribute-surface residual.** 125 names still resolve on `OrchestratorApp`
   via delegates. Exit path in 8.3.
2. **Hot-path indirection.** Every delegated call crosses one delegate hop + one
   collaborator `__getattr__` hop until the shims are deleted.
3. **Legibility residual.** The class file is smaller but the class's public
   surface is unchanged; readers must follow `_ROUTES` to find an implementation.

### 8.3 The full-exit path (recommended as a scoped follow-on, not Phase B)

To satisfy the vulture-sweep exit -- delete the shims -- Phase B would
additionally need to:

1. Repoint the **6 Tier A** + **36 Tier B** patch sites onto the collaborators
   (a test-side change per moved member), and
2. Repoint the **src-tree consumers** of the 22 externally-referenced members
   (up to ~98 `dispatch` sites, etc.; counts are upper bounds -- see Provenance)
   onto the collaborators or a narrowed command surface, and
3. Delete the delegates and run the vulture-class sweep.

Step 2 is the expensive part -- the same shape as the deferred 27-site
`linked_issue_number` repoint carried over from Phase A, and calibrated against
#1449 (27 names moved over 11 days). **Recommendation:** land Phase B as the
metric-exit-with-named-residual (member_count 7, all tiers green, zero external
repoints), ratchet the baseline to 7, and file the full-exit (shim deletion +
call-site repoint) as an explicit follow-on umbrella child. Presenting 7 as
"dismantled" would be false; presenting it as "bodies extracted, count ratcheted,
shims scheduled for deletion" is accurate.

If the operator rejects the facade entirely (no surviving shims *at all* in
Phase B), the honest answer flips: `OrchestratorApp` **cannot** reach <= 8
lexical members in one phase while keeping all 133 names callable, because
presenting 133 names at <= 8 `def`s is only possible by installing the rest as
non-`def` attributes. The choice is facade-with-scheduled-deletion (recommended)
or a multi-month call-site migration before any count drops.

### 8.4 Verified sub-claims (evidence)

- **Metric fact:** class-level `Assign` is not a `FunctionDef` child ->
  `archetypes.py` `_is_def` + the members comprehension. `OrchestratorApp` has 3
  such constant Assigns today that already do not count.
- **Guarded-command adaptability:** `fence_probe.py` builds
  `_guard_state_lock(_delegate("review"))` and reports
  `__name__ == "review"` with the `__wrapped__` chain intact -- so the 6 guarded
  commands can be adapter-wrapped without losing name or the StateLockBusy guard.
- **Fence numbers and stability:** `fence_probe.py` (Section 2.4).
- **Tier counts and controls:** `patch_census.py` (Section 2.3), Tier A
  cross-checked against raw grep.
- **Single state writer, no SCCs, clusters, external refs:**
  `analyze_orchestratorapp.py` (Section 3).


## 9. Ratchet, gate, and stop-conditions

- **Ratchet as definition of done.** Each move-PR (each batch) lowers the
  committed `.attachment-budgets.json` OrchestratorApp `member_count` to its
  post-move value in the same PR and runs
  `uv run python -m charlie_work.attachment_contracts baseline --ratchet` so the
  budget can only descend. L00 lands the machinery at 133 (delta 0). The exit
  criterion is the **committed baseline value**, not a live-recomputed fence
  (Section 2.4 caveat), so a later population shift cannot un-satisfy a landed
  leaf. Final committed value: 7 (or 8 if L08 is deferred to the residual).
- **AST-equivalence gate.** Every move-PR must prove the moved `FunctionDef` is
  byte-identical between removal site and destination. The CLI is broken
  (#1600), so invoke the gate's library entry points directly:
  `charlie_work.ast_equivalence_gate.extract_symbols` /
  `derive_moved_symbols` over the PR diff, asserting the moved-symbol set matches
  and each body hashes equal. A body that cannot move verbatim (L08's state
  extract, or a Section 3.1 namespace rebind) splits into a move-PR + edit-PR.
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
- A Tier D test goes red -- indicates a module-namespace rebind (Section 3.1);
  the fix is to reference the free function via `charlie_work.workflow`, never to
  edit the test.
- A Tier A/B member patch fails to resolve to its delegate -- indicates the
  install loop missed the name (check `_ROUTES` collision handling).
- Any Tier C subclass appears (couples a leaf to a subclass).
- The live fence drops toward the committed baseline under collaborator growth
  (Section 2.4 caveat) -- the ratchet baseline still governs, but re-run
  `fence_probe.py` to confirm the margin.

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
`eb634c9b319462955984cce9452a3660497c901d`:

- `analyze_orchestratorapp.py` -> `orchestratorapp-inventory.json` / `.md` --
  133 members, 0 async, 0 fields, 3 class Assigns, per-member line spans, self
  attr reads/writes/calls (separated), `self.gh` usage, resolved domain-module
  calls, decorators, domain clusters, in-degree/hub ranking, SCCs (none),
  single instance-state writer, 62 pure leaves, external src references (23
  member names).
  - **Resolver limitation (external_references_in_src).** The external-reference
    counts are matched by **bare attribute name** and are therefore **upper
    bounds**, not exact external call counts. Three artifacts follow: (a) the same
    method name on a different object is counted -- `dispatch`'s 98 sites carry
    receivers `{app, config, ctx.config, self.config}`, so `self.config.dispatch`
    is folded in with `app.dispatch`; (b) same-module `self.<name>` calls inside
    `workflow.py` are counted (the per-member file lists include `workflow.py`
    itself); (c) dunder collisions -- `__init__`'s 14 hits are all
    `super().__init__()` in unrelated classes (receiver set `{super()}`), a pure
    false positive, so the 23-name list is 22 real externally-referenced members.
    None of this changes the design's load-bearing claim: attribute access
    resolves a class-level delegate transparently, so metric exit needs zero
    external repoints **regardless of the exact counts**. The counts are used only
    as scale texture and as the (over-)estimate of the full-exit repoint cost.
- `patch_census.py` -> `patch-census.json` / `.md` -- four-tier patch census
  (A 6/5, B 36/13, C 0/0, D 140/21), B_unresolved 0 (loud bucket), positive
  controls (`_process_rescue_review` 2, `review` 11), 117 no-patch members.
- `mikado.py` -> `mikado-graph.json` + `.mermaid.txt` + `.indented.txt` --
  10-leaf projection, 133 -> 7, ~19 batch-PRs, 7-member residual.
- `fence_probe.py` -- live fence 8.5 (pop 46, Q1 1.0, Q3 4.0, IQR 3.0),
  stability counterfactuals (reduction + append), guarded-command adaptability.

Primary-source verification performed in this pass:
- Metric definition: `src/charlie_work/attachment_contracts/archetypes.py`
  (`_is_def`, members comprehension).
- Fence machinery: `src/charlie_work/attachment_contracts/outliers.py`
  (`saturate`, `_quartiles`, FLOOR).
- Class shape: `workflow.py` `OrchestratorApp` bases/decorators (both empty),
  `_guard_state_lock` at line 909.
- Tier A cross-check: raw grep over `tests/` (Section 2.3).
- Tier D mechanism: `patch("charlie_work.workflow._worker_pid_alive")` sites and
  the single OA referrer `_dispatch_impl` (Section 3.1).

Leaf issues (one `needs-design` issue each, blocked in sequence under #1582 /
#1628): see the leaf-issue drafts produced alongside this design. This document
files none of them.
