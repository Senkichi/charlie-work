# GitHub Class: Mikado Graph and Protocol Segmentation (Track 2 design)

Status: DESIGN ONLY. This document moves no code. It is the single planning
artifact for the Track 2 execution umbrella (charlie-work #1582), produced under
the Track 2 prep issue (#1543). Every leaf named here is filed as its own
`needs-design` issue (see the leaf-issue drafts referenced in Provenance).

Author: synthesis/adjudication pass over three prior analysis artifacts
(inventory, caller census, toolchain/oracle). Where those artifacts disagreed,
this document states the reconciled number and the file:line evidence for it.

Target: `src/charlie_work/github.py`, class `GitHub` (lines 310-2015) and the
`GitHubLike` protocol (lines 2019-2127). Interpreter: CPython 3.13.5 (load-bearing;
see Section 8). Metric authority: `.attachment-budgets.json` GitHub entry
(`kind: class`, `boundary: 6.0`, `member_count: 53`).


## 1. Summary

`GitHub` is an APC `class`-archetype god object: 53 lexical method definitions,
boundary 6.0 (goal: member_count <= 5). It is also the fleet's most heavily
mocked collaborator: `run` alone carries 134 monkeypatch sites, and 117 test
doubles define at least one `GitHubLike` member.

The plan decomposes `GitHub` by capability into seven sub-protocols plus a
transport collaborator, moving 51 of the 53 methods out of the class body
verbatim (AST-equivalence-gated). Two definitions stay: `__post_init__` (dataclass
hook) and `run` (the mandated monkeypatch interception seam -- 134 patch sites
depend on it). The moved method *bodies* live on collaborator classes; the moved
method *names* remain resolvable on `GitHub` through generated, signature-carrying
class-level delegate attributes installed from an explicit routing table. A
generated delegate is an assignment, not a `def`, so it contributes zero to the
`class`-archetype metric while keeping `GitHub` a drop-in `GitHubLike`
implementation for the two `isinstance` sites and the class-level conformance
test.

End state (Section 8): `GitHub` reaches **member_count = 2**, well under the 6.0
boundary, while `isinstance(gh, GitHubLike)` and the full conformance test at
`tests/test_githublike_protocol.py` stay green. This is achievable **only** via
the generated-delegate facade; the two alternatives the brief named -- a
`__getattr__`-only facade and explicit thin delegates -- are both rejected with
empirical evidence.


## 2. Counting definition and census

### 2.1 What the metric counts

The `class` archetype counts **lexical** `FunctionDef`/`AsyncFunctionDef` nodes
that are direct children of the `ClassDef.body`
(`src/charlie_work/attachment_contracts/archetypes.py:64-65`, `_is_def`; and
`:276`, `members = tuple(child.name for child in node.body if _is_def(child))`).
Consequences that drive the whole design:

- Dataclass fields (`AnnAssign`) do NOT count. `GitHub`'s `repo_root`, `dry_run`,
  `runtime` fields are free.
- A class-level **assignment** (`GitHub.foo = <delegate>`, or a post-class
  install loop) is NOT a `FunctionDef` child and does NOT count. This is the
  metric fact the facade exploits, and it is a disclosed judgment call, not a
  loophole discovered by accident (Section 8.3).
- `_is_protocol_base` (`archetypes.py:265`) excludes `Protocol` subclasses from
  the class population, so the new sub-protocols and the redeclared union add
  zero to any budget.

Authoritative current count: **53 lexical defs** (verified by AST over
`github.py` at the Track 2 base; enumerated with line spans in Section 6). The
budget file's `member_count: 53` agrees.

### 2.2 Reconciling the member-count numbers

- **53** (this doc, `.attachment-budgets.json`): lexical defs. Correct.
- **56** (an inventory pass): 53 defs + 3 dataclass fields (`repo_root`,
  `dry_run`, `runtime`) counted for caller-search convenience. The extra 3 are
  fields, not defs; they do not count toward the metric.
- The brief's "12 non-Protocol internals" is off by one: there are **13**
  (53 = 40 `GitHubLike` methods + 13 internals). `dry_run` is a frozen field /
  read-only-property surface, not a `def`, which is where the miscount came from.

### 2.3 Caller census (adopt C1+C2 as the planning number)

Tiers, AST-verified at the base sha, source tree only unless noted:

| Tier | Meaning                                   | src sites | notes |
|------|-------------------------------------------|-----------|-------|
| C1   | `self.gh.<m>(...)` (attribute on gh)      | 108       | planning |
| C2   | DI param annotated `GitHubLike`, `.<m>()` | 75        | planning; 39 distinct annotated signatures |
| C3   | heuristic name match (`gh`-shaped local)  | 6         | verify-at-touch |
| C4   | everything else (src)                     | 4         | verify-at-touch |
| C4t  | tests (C4-shape)                          | 169       | doubles + fixtures |

**Planning number = C1 + C2 = 183 source call sites** to retype or verify as
capabilities move. C3/C4/tests are reported separately and confirmed at the
moment each leaf is cut (line numbers shift as earlier leaves land).

Patch sites: **146 total**, of which **`run` = 134** (`patch_site_totals_per_member`
in the census). No other member exceeds 5 (`pr_update_branch` = 5). Two
`isinstance(_, GitHubLike/GitHub)` sites exist and both must keep passing:
`src/charlie_work/workflow.py:4075` (`isinstance(self.gh, GitHub)`) and
`tests/test_githublike_protocol.py:118` (`isinstance(gh, GitHubLike)`).

Doubles: **117** test doubles define >= 1 `GitHubLike` member; `FakeGitHub`
(15 method defs / 40 union members / ~808 instantiations) spans every cluster.

Reconciliation of the stale figures:

- #1543's **220 / ~680**: loose full-text grep counts (every textual occurrence
  of a member name and of `gh.`), not call-site resolution. Superseded.
- The decision doc's **~114 / ~165**: an intermediate estimate; ~114 tracks
  C1+C3 (108+6), ~165 an early C1+C2 before AST de-duplication.
- This doc's **108 / 75** are AST-resolved call sites (#1543's raw 111 for C1
  dropped to 108 after removing 3 non-call attribute reads). Adopt 108/75.


## 3. Capability segmentation

Every one of the 40 `GitHubLike` methods, plus the 13 internals, is assigned to
exactly one destination. `run` is `GitHubLike` but stays on the owner as the seam
(so the union still declares it). Ambiguous members (noted) are pinned to one
cluster to keep leaves independently revertible.

### 3.1 Sub-protocols (GitHubLike methods -> destination class)

Destination classes live in a new `src/charlie_work/github_capabilities/`
package, one module per cluster. Each holds a collaborator class constructed with
a back-reference to the owner `GitHub` instance (`self._owner`), reached for
transport and shared state (Section 3.3).

| # | Sub-protocol / class | GitHubLike members | count |
|---|----------------------|--------------------|-------|
| A | `CommentsLike` / `Comments` | issue_comment, pr_comment | 2 |
| B | `LabelsLike` / `Labels` | add_issue_label, remove_issue_label, add_pr_label, remove_pr_label, label_list, label_create | 6 |
| C | `ChecksLike` / `Checks` | pr_checks, check_run_annotations, commit_check_runs, actions_job, workflow_runs_for_head, check_graphql_rate_limit | 6 |
| D | `RepoMetaLike` / `RepoMeta` | name_with_owner, compare, compare_diff, commit, invalidate_list_cache | 5 |
| E | `PullRequestsLike` / `PullRequests` | pr_create, pr_view, pr_list, pr_diff, pr_commits, pr_ready, merged_pr_list, merged_prs_for_issue | 8 |
| F | `IssuesLike` / `Issues` | close_issue, issue_view, issue_list, issue_dependencies, are_issues_open | 5 |
| G | `MergeBranchLike` / `MergeBranch` | merge_pr, delete_branch, pr_update_branch, pr_close, pr_reopen, push_empty_commit, branch_protection | 7 |
|   | (owner seam) | run | 1 (stays) |

Total GitHubLike methods: 2+6+6+5+8+5+7+1 = 40.

**Destination invariant (MANDATORY, per-module).** Every capability module's
first statement MUST be `from __future__ import annotations`. `github.py` has it,
which is why `inspect.signature` returns annotations as *strings* and the
conformance test's string comparison at `tests/test_githublike_protocol.py:106`
holds. A verbatim-moved method in a destination that lacks the future import has
its return annotation *evaluated* at def time, so `inspect.signature` returns a
`types.UnionType` (e.g. `dict[str, typing.Any] | None`) that is `!=` the protocol's
string annotation and the conformance test fails. The trap: the AST-equivalence
gate stays GREEN because the future import is a module-level statement, not part of
the moved `FunctionDef` node -- so this failure is invisible to the gate and only
the conformance test catches it. Verified in `scratchpad/track2/probe_wraps.py`
(WITH-future PASS, NO-future FAIL). Every move leaf's Checks list this invariant.

**Decorator invariant.** No `GitHub` member is decorated (AST-verified: zero
members carry a decorator list). In particular `_max_retries`, `_retry_base_seconds`,
`_timeout_seconds` (315-328) are plain methods, not `@property`, and `run` calls
them as `self._max_retries()` etc. So a plain-function generated delegate is correct
for every member; none needs `property(...)` wrapping and none is a `cached_property`
with owner-pinned storage. This is what keeps the final `member_count` at 2.

Ambiguity calls (pinned): `check_graphql_rate_limit` -> Checks (it wraps a
GraphQL rate probe used by the checks path, though it is transport-adjacent);
`merged_prs_for_issue` and `merged_pr_list` -> PullRequests (they return PR data
even though `merged_prs_for_issue` is issue-keyed); `invalidate_list_cache` ->
RepoMeta (it is cache lifecycle, and RepoMeta is the smallest write-side cluster).

### 3.2 Non-protocol internals (13) -> Transport collaborator / owner

Transport core (in-degree >= 2) is its own collaborator
(`src/charlie_work/github_capabilities/transport.py`, class `Transport`), NOT a
sub-protocol (these members are not `GitHubLike`). `__post_init__` stays on the
owner; `run` stays on the owner as the seam.

| member | line span | destination |
|--------|-----------|-------------|
| `run` | 379-563 | **owner (seam, stays)** |
| `__post_init__` | 330-337 | **owner (dataclass hook, stays)** |
| `_run_bool` | 628-639 | Transport |
| `_list_json` | 680-693 | Transport |
| `_repo_owner_name` | 1782-1825 | Transport |
| `_graphql_query` | 1827-1872 | Transport |
| `_graphql_issue_states` | 1874-1913 | Transport |
| `_graphql_issue_dependencies` | 1915-1986 | Transport |
| `_normalize_rest_pr` | 347-377 | Transport |
| `_pr_checks_fallback` | 959-1042 | Transport |
| `_max_retries` | 315-318 | Transport |
| `_retry_base_seconds` | 320-323 | Transport |
| `_timeout_seconds` | 325-328 | Transport |
| `validate_field_lists` | 1267-1360 | Transport (validation/config) |

Note: `issue_dependencies` (1988-2015, GitHubLike) is in cluster F; its private
GraphQL helpers `_graphql_issue_states`/`_graphql_issue_dependencies` are
transport. The moved `Issues.issue_dependencies` body calls
`self._graphql_issue_dependencies`, which resolves through the owner routing
table to `Transport` (Section 3.3). This split is intentional: keep the public
surface in its capability cluster, the wire mechanics in transport.

### 3.3 The delegation seam (transport core is a shared collaborator)

Every destination routes through the same `run`. The invariant the 134 patch
sites depend on:

**`GitHub.run` is the sole HTTP/CLI interception point. No collaborator may
cache a bound `run` (or any owner attribute). Every call late-resolves through
the owner instance so that `monkeypatch.setattr(GitHub, "run", ...)` and
`isinstance`-time class lookups still intercept.**

Two directions of resolution, each bounded so there is no recursion cycle:

1. **owner -> collaborator (explicit table, no `__getattr__`).** The owner
   installs a class-level delegate for every moved name from an explicit routing
   dict `_ROUTES: dict[str, str]` (name -> collaborator attribute). The owner has
   **no** `__getattr__`; a name not in `_ROUTES` and not a real owner attribute
   raises `AttributeError` normally. This termination is what prevents the cycle.

2. **collaborator -> owner (`__getattr__`, bounded).** A moved body is byte
   -identical, so it still says `self.run(...)`, `self._list_cache`,
   `self._graphql_query(...)`. On a collaborator instance, `self.<name>` for a
   name the collaborator does not define triggers the collaborator's
   `__getattr__`, which forwards to `self._owner.<name>`. That resolves to a real
   owner attribute (`run`, `_list_cache`) or an owner delegate (another
   collaborator's method / a transport internal). Because the owner side is
   explicit and terminates, the chain always ends.

`self.run` inside a moved body is **instance** access, so the collaborator's
`__getattr__` fires (unlike the class-level conformance lookup, Section 8). The
seam is preserved: `self._owner.run` is looked up fresh each call and picks up
the patched class attribute.

Delegate construction (net-new infra, introduced in L01):

```
def _make_delegate(name, collab_attr):
    def _delegate(self, *args, **kwargs):
        collab = getattr(self, collab_attr)          # owner-held collaborator
        return getattr(collab, name)(*args, **kwargs)
    src_fn = getattr(_SIGNATURE_SOURCE[name], "__func__", _SIGNATURE_SOURCE[name])
    functools.wraps(src_fn)(_delegate)               # carries __wrapped__/__name__
    _delegate.__signature__ = inspect.signature(src_fn)
    return _delegate
```

`functools.wraps` + explicit `__signature__` make `inspect.signature(GitHub.foo)`
return the source method's signature, including the string return annotation
(`from __future__ import annotations` is in force in the source method's module
-- the destination module's future import, per the mandatory invariant of
Section 3.1, not `github.py`'s own), which is what
the conformance test compares (Section 8.1). The routing table is derived, not
hand-maintained per member: `_ROUTES` is built by iterating each collaborator
class's **non-dunder** members at import time (CLAUDE.md rule 9 -- no hardcoded
lists) -- protocol methods and underscore-prefixed internals alike, across every
collaborator class including `Transport` (which is not a sub-protocol, Section
3.2, so its internals would be missed by a public-members-only or
sub-protocol-authority rule). `_ROUTES` covers **all** moved members -- both the
public sub-protocol methods and the underscore-prefixed internals (`_run_bool`,
`_max_retries`, `_timeout_seconds`, etc.), because `run` and the moved bodies call
internals by name and the owner has no `__getattr__` fallthrough to catch them.

Where the machinery lives (L01 finding, PR #1596): not inline in `github.py`. That file
sat at 3173 lines against the file-size ratchet's 3200-line high-water mark (#1442,
`tests/test_file_size_ratchet.py`, `file_size_ratchet_baseline.json`), so adding the
~150 lines of `_make_delegate` / `_ROUTES` / `_install_delegates` there would have
tripped a second, unrelated ratchet. The machinery is `src/charlie_work/github_delegation.py`,
with `_install_delegates(owner_cls)` taking the owner class as a parameter (no circular
import); `github.py` calls it once after the class body. Each sub-protocol is colocated
with its collaborator module under `src/charlie_work/github_capabilities/`, and
`_build_routes` raises `ValueError` on a cross-collaborator name collision instead of
letting the last writer win. Later leaves shrink `github.py`, so this constraint only
bites L01.

### 3.4 `_list_cache` strategy

`_list_cache` is shared mutable state touched by 8 members across clusters
(inventory). Decision: **`_list_cache` stays on the owner** as the single frozen
-dataclass field, initialized in `__post_init__` via
`object.__setattr__(self, "_list_cache", {})` (the existing escape hatch).
Collaborators read and mutate it by reference through `self._list_cache` ->
collaborator `__getattr__` -> `self._owner._list_cache`, which returns the one
shared dict; in-place mutation (the cache's only write pattern -- no
reassignment) is visible to all collaborators and to the owner.

Consequence for leaf ordering: because the cache is a single owner-held object
reached by reference, **it does not couple the cluster leaves**. A cache-touching
method in Labels and one in Issues can move in either order; neither needs the
other to have moved first. `invalidate_list_cache` (the cache's public
clear-method) is pinned to RepoMeta (L05) but its ordering is free for the same
reason. This decoupling is what makes the leaf order in Section 5 a free choice
driven by blast radius, not by data dependency.


## 4. Protocol plan

### 4.1 Sub-protocols and the union

Seven `@runtime_checkable` sub-protocols (Section 3.1, A-G), each declaring only
its cluster's methods with the exact current signatures (param names, kinds,
string return annotations copied verbatim from `github.py`). `GitHubLike` is
redeclared as the **union** of the seven sub-protocols plus `run` and the
`dry_run` read-only property:

```
@runtime_checkable
class GitHubLike(CommentsLike, LabelsLike, ChecksLike, RepoMetaLike,
                 PullRequestsLike, IssuesLike, MergeBranchLike, Protocol):
    @property
    def dry_run(self) -> bool: ...
    def run(self, ...) -> GitHubRunResult: ...
```

Five members are asserted present in `GitHubLike.__dict__` by name-specific
tests (`branch_protection`, `pr_ready`, `pr_close`, `pr_reopen`,
`push_empty_commit` -- `tests/test_githublike_protocol.py:33-84`). Inheriting them
from sub-protocols puts them in the sub-protocol's `__dict__`, not
`GitHubLike.__dict__`, so those five `in GitHubLike.__dict__` assertions break.
**Fix: redeclare those five members directly on the `GitHubLike` union body**
(in addition to inheriting them), so both `GitHubLike.__dict__` membership and
the sub-protocol membership hold. Redeclaration costs nothing on the metric
(`_is_protocol_base` excludes protocols).

### 4.2 Conformance test additions

`test_githublike_protocol.py` is extended (in L01) with, for each of the seven
sub-protocols: an `isinstance(gh, <SubLike>)` assertion and the same
`_compatible_signature` loop over `<SubLike>.__protocol_attrs__` that the union
test already runs (`:112-128`). This makes each capability boundary independently
regression-guarded, so a later cluster move that drifts a signature fails its own
sub-protocol test, not just the union test.

### 4.3 Doubles quantification

117 doubles define >= 1 union member. Per cluster (defining-doubles / lead
instantiation counts, from the census):

| cluster | doubles defining >=1 member | notable doubles (instantiations) |
|---------|------------------------------|----------------------------------|
| comments | 4 | FakeGitHub[808], _FakeGitHub[9], _CapturingGitHub[4] |
| checks | 7 | FakeGitHubWithChecks[31], FakeGitHub[808] |
| merge-branch | 18 | FakeGitHub[808], FakeGitHubForOrphan[30] |
| repo-meta | 12 | _NWOGitHub[6], FakeGitHub[808] |
| labels | 23 | LabelFailGitHub[7], FakeDoctorGitHub[63], FakeGitHub[808] |
| issues | 43 | FakeGitHub[808], FakeDoctorGitHub[63], FakeGitHubForOrphan[30] |
| pull-requests | 27 | FakeGitHub[808] |
| transport | 19 | FakeGitHub[808] |

Doubles are unaffected by a capability move as long as they continue to satisfy
the union at runtime: a double sets its own methods on itself, so it keeps
matching whichever sub-protocol declares them. No double needs editing to
preserve `isinstance` (presence-by-name). Doubles that assert on `run`
interception are untouched because `run` stays put. This is why the moves are
low-risk for the test tier despite the large double count.


## 5. Ordered leaf list (the Mikado leaves)

Nine leaves. L01 is pure infrastructure and moves zero methods (umbrella #1582
Phase A requires the first PR to move no methods). L02-L09 each move exactly one
cluster and are independently revertible (facade delegate + collaborator class
revert as a unit). Order is by ascending blast radius, which the `_list_cache`
decoupling (Section 3.4) leaves us free to choose.

For each leaf: members / destination / callers by tier / patch sites / doubles /
checks / ratchet delta / blast radius / revert.

**L01 -- Facade + sub-protocol infrastructure (moves 0 methods).**
- Adds: `github_capabilities/` package skeleton (empty collaborator classes with
  `__init__(self, owner)` + `__getattr__` -> owner); `_make_delegate`, `_ROUTES`
  builder, and the post-class install loop on `GitHub`; the seven sub-protocols;
  `GitHubLike` redeclared as their union with the five redeclared members; the
  extended conformance test (Section 4.2); `.attachment-budgets.json` unchanged
  (still 53 -- no method moved yet).
- Callers: none retyped. Patch sites: none. Doubles: none edited.
- Checks: full suite green; conformance test (union + seven new sub-protocol
  assertions) green; AST gate not applicable (net-new code, not a move) --
  reviewed as ordinary infra by opus48 (touches protocol surface).
- Ratchet delta: **0** (still 53). Blast radius: whole test surface imports
  `GitHubLike`, so a bad union break shows immediately. Revert: delete the
  package + revert the union redeclaration.

**L02 -- Comments (pilot).** members: issue_comment (1400-1401),
pr_comment (1403-1404) -> `Comments`. Narrowest surface (2 members) and fewest
defining doubles (4); chosen as the end-to-end pilot that also retypes its DI
consumers to `CommentsLike`. Callers: the comment call sites among C1/C2 (small;
confirm at cut). Patch sites: 0 direct (neither is in the 134/`run` set). Doubles:
4, none edited. Checks: AST gate `uv run charlie ast-equivalence-check --base
origin/main` proves both bodies moved verbatim; conformance + sub-protocol
CommentsLike test green; full suite; ratchet. Ratchet delta: **-2** (53 -> 51).
Blast radius: minimal. Revert: revert the two delegate entries + `Comments`.

**L03 -- Labels.** 6 members: add_issue_label (1362-1363), remove_issue_label
(1365-1366), add_pr_label (1368-1376), remove_pr_label (1378-1385), label_list
(1406-1410), label_create (1412-1420) -> `Labels`. Doubles: 23 (incl.
LabelFailGitHub[7]). Patch sites: 0 in the run-set. Ratchet delta: **-6**
(51 -> 45). Blast radius: moderate (label doubles many but presence-only).

**L04 -- Checks/CI.** 6 members: pr_checks (905-957), check_run_annotations
(1062-1078), commit_check_runs (1118-1140), actions_job (1044-1060),
workflow_runs_for_head (1142-1163), check_graphql_rate_limit (641-678) ->
`Checks`. Note: `pr_checks` body calls `self._pr_checks_fallback` (transport,
moved in L09) -- until L09, `_pr_checks_fallback` is still an owner attribute, so
the collaborator `__getattr__` resolves it either way; order L04-before-L09 is
safe. Doubles: 7. Ratchet delta: **-6** (45 -> 39).

**L05 -- RepoMeta.** 5 members: name_with_owner (1765-1780), compare (1191-1206),
compare_diff (1243-1265), commit (1080-1116), invalidate_list_cache (339-345) ->
`RepoMeta`. `invalidate_list_cache` clears the owner-held `_list_cache` by
reference (Section 3.4). Doubles: 12. Ratchet delta: **-5** (39 -> 34).

**L06 -- PullRequests.** 8 members: pr_create (565-626), pr_view (876-897),
pr_list (744-764), pr_diff (899-903), pr_commits (1165-1189), pr_ready
(1474-1492), merged_pr_list (766-814), merged_prs_for_issue (816-874) ->
`PullRequests`. `pr_ready` is in the five `__dict__`-asserted names, redeclared
on the union (Section 4.1) -- its move does not touch that assertion. Doubles: 27.
Patch sites: `pr_update_branch` is NOT here (it is L07/merge... see L08).
Ratchet delta: **-8** (34 -> 26).

**L07 -- Issues.** 5 members: close_issue (1387-1398), issue_view (731-742),
issue_list (695-729), issue_dependencies (1988-2015), are_issues_open
(1703-1763) -> `Issues`. **Constraint (must not split):** `are_issues_open`
(1703-1763) builds a closure that captures `self` and calls `self.issue_view`
inside a thread pool. Moving `are_issues_open` without `issue_view` in the same
leaf would make the closure reach `issue_view` back through the owner delegate
across a thread boundary -- correct but an avoidable hazard. Keep both in L07.
Doubles: 43 (largest). Ratchet delta: **-5** (26 -> 21).

**L08 -- MergeBranch.** 7 members: merge_pr (1422-1443), delete_branch
(1445-1459), pr_update_branch (1461-1472), pr_close (1494-1511), pr_reopen
(1513-1537), push_empty_commit (1539-1701), branch_protection (1208-1241) ->
`MergeBranch`. Four of these (branch_protection, pr_close, pr_reopen,
push_empty_commit) are `__dict__`-asserted names, redeclared on the union.
`pr_update_branch` has 5 patch sites (the second-highest) -- confirm each still
targets a working delegate. Doubles: 18. Ratchet delta: **-7** (21 -> 14).

**L09 -- Transport internals.** 12 members: _run_bool (628-639), _list_json
(680-693), _repo_owner_name (1782-1825), _graphql_query (1827-1872),
_graphql_issue_states (1874-1913), _graphql_issue_dependencies (1915-1986),
_normalize_rest_pr (347-377), _pr_checks_fallback (959-1042), _max_retries
(315-318), _retry_base_seconds (320-323), _timeout_seconds (325-328),
validate_field_lists (1267-1360) -> `Transport`. `run` (379-563) STAYS on the
owner and calls into `Transport` for `_run_bool`/retry/timeout via the owner
routing (owner delegates `_run_bool` etc. to `Transport`; `run`'s body says
`self._run_bool(...)` which, on the owner, must resolve). **Owner subtlety:** the
owner has no `__getattr__`, so `run`'s references to moved internals resolve
through the installed owner delegates for those internal names (the `_ROUTES`
table covers internals too, not just protocol methods). Doubles: 19. Ratchet
delta: **-12** (14 -> **2**). Blast radius: highest -- every collaborator's
transport calls funnel here; land last so all callers exist. Revert: revert the
Transport class + its delegate entries; `run` never moved so the seam is intact
throughout.

End: member_count = **2** (`__post_init__`, `run`). Budget lowered to 2 in L09
(or ratcheted per-leaf; see Section 9).


## 6. Mikado graph

Goal node: **GitHub strictly under boundary 6.0** (member_count <= 5). The graph
is nearly a star: L01 is the shared prerequisite; L02-L09 each depend only on
L01 (the `_list_cache` decoupling removes inter-cluster edges). L09 is drawn
after the others only by blast-radius preference, not by a hard data edge -- the
one soft edge is L04's `pr_checks` calling `_pr_checks_fallback`, which resolves
through the owner whether or not L09 has landed, so it is advisory.

ASCII tree (goal at root, leaves are the work):

```
GOAL: GitHub member_count <= 5
|
+-- L01 infra (sub-protocols + facade + conformance ext)   [prereq of all]
    |
    +-- L02 Comments        (-2 -> 51)
    +-- L03 Labels          (-6 -> 45)
    +-- L04 Checks          (-6 -> 39)   ..soft.. L09 (_pr_checks_fallback)
    +-- L05 RepoMeta        (-5 -> 34)
    +-- L06 PullRequests    (-8 -> 26)
    +-- L07 Issues          (-5 -> 21)   [are_issues_open+issue_view together]
    +-- L08 MergeBranch     (-7 -> 14)
    +-- L09 Transport       (-12 -> 2)   [land last: highest fan-in]
```

Adjacency table (prerequisite -> dependent; "soft" = advisory ordering only):

| edge | type | reason |
|------|------|--------|
| L01 -> L02..L09 | hard | delegate machinery + sub-protocols must exist first |
| L04 -> L09 | soft | `pr_checks` uses `_pr_checks_fallback`; resolves via owner regardless |
| (none) L02..L08 among themselves | -- | `_list_cache` shared-by-reference removes coupling |

Each leaf = exactly one PR (umbrella rule: verbatim moves only; if a body must be
edited, split that leaf into a move-PR then an edit-PR).


## 7. Ordering rationale

The leaf order (Section 5) is chosen by ascending blast radius, not by data
dependency: because `_list_cache` is a single owner-held object reached by
reference (Section 3.4) and every cross-collaborator call resolves through the
owner routing table, L02-L08 have no hard edges among themselves (Section 6
adjacency). L01 is the only hard prerequisite; L09 lands last only because its
transport members have the highest fan-in. Any permutation of L02-L08 is
technically valid; the chosen order front-loads the smallest, lowest-double
clusters (comments, labels) as ratchet-and-gate rehearsals before the
highest-double cluster (issues, 43 doubles) and the highest-fan-in cluster
(transport).


## 8. Honest end-state verdict

**Question:** can `GitHub` reach < 6 lexical members while remaining
`GitHubLike`-conformant, usable by the two `isinstance` sites, and passing the
signature conformance test?

**Answer: YES -- member_count = 2 -- but only via generated class-level delegate
attributes. The two facade alternatives the brief named are both empirically
rejected.**

### 8.1 What the conformance test actually requires (evidence)

`tests/test_githublike_protocol.py`:
- `:118` `isinstance(gh, GitHubLike)` -- **instance** runtime-checkable check.
- `:120-128` for each protocol attr: `getattr(gh, name)` callable, then
  `proto_sig = inspect.signature(getattr(GitHubLike, name))` and
  `concrete_sig = inspect.signature(getattr(GitHub, name))` -- a **class-level**
  `getattr(GitHub, name)` for all 40 methods.
- `:87-109` `_compatible_signature`: equal param names (minus self), equal param
  kinds, and equal **return-annotation string** (annotations are strings under
  `from __future__ import annotations`).

So the facade must make, for all 40 methods: (a) `isinstance` pass, and (b)
`getattr(GitHub, name)` return a callable whose `inspect.signature` matches the
protocol's, return annotation included.

### 8.2 Why `__getattr__`-only is rejected (empirical, CPython 3.13.5)

Probe run against the project interpreter (`scratchpad/track2/probe_d_facade.py`):

```
python: 3.13.5
== FacadeGetattr (__getattr__ only) ==
  isinstance: False
  getattr_static: MISSING
  signature(CLASS) ERR: AttributeError ... has no attribute 'foo'
```

On 3.12+, `_ProtocolMeta.__instancecheck__` uses `inspect.getattr_static`, which
does **not** invoke `__getattr__`. So a `__getattr__`-only facade returns
`isinstance == False` and fails `:118`. Independently, `getattr(GitHub, name)` at
`:127` is class-level access, for which `__getattr__` (an instance hook) never
fires -> `AttributeError` -> the test errors. The `__getattr__`-only facade fails
twice. (An earlier draft of this plan rested on instance-`__getattr__` isinstance
passing; the probe falsified it. This is the correction.)

### 8.3 Why generated class-level delegates work (and what the residual is)

Same probe, class-level assigned delegate:

```
== FacadeClassAttr (class-level assigned delegate) ==
  isinstance: True
  getattr_static: <function _delegate...>
  signature(CLASS FacadeClassAttr.foo): (self, *a, **k)
== AST proof ==
  ClassDef.body FunctionDef children of K: ['m']   # the assigned attr is NOT counted
```

A class-level assignment satisfies `getattr_static` (so `isinstance` passes and
`getattr(GitHub, name)` succeeds) and is **not** an `ast.FunctionDef` child (so it
adds nothing to member_count). The generic `(*a, **k)` signature in the raw probe
would fail `_compatible_signature`; the real delegate fixes that with
`functools.wraps(src_fn)` + `__signature__ = inspect.signature(src_fn)`, which
makes `inspect.signature(GitHub.name)` return the source method's exact signature
and string return annotation, comparing equal at `:106`. This is verified, not
assumed: `scratchpad/track2/probe_wraps.py` runs the actual `_compatible_signature`
body (`:87-109`) against a `functools.wraps`-signed delegate whose source lives in a
future-import module and reports PASS (param names, kinds, and the string
`'dict[str, Any] | None'` all match); the same probe reports FAIL when the source
module omits the future import (Blocker 1 above).

Explicit hand-written thin delegates are rejected for the opposite reason: each
`def` would be a lexical member and count toward the metric -- 40 of them keeps
`GitHub` a god object.

**Disclosed judgment call.** This is a metric-boundary decision, not a free win.
The method *bodies* -- the actual god-object logic, and the 134-site transport
surface -- genuinely leave `GitHub` and become independently testable
collaborators. But the 40 method *names* still resolve on `GitHub`'s attribute
surface via generated delegates, because the conformance test and the two
`isinstance` sites require class-level, correctly-signed access. The operator is
accepting three residuals:

1. **Attribute-surface residual.** `GitHub` still advertises 40 names; only their
   implementations moved. The metric correctly reports 2 (it measures lexical
   defs), but static readers of the class see forwarding shims, not a 2-method
   class. The exit path: retype the 183 C1+C2 consumers and the two `isinstance`
   sites to the narrow sub-protocols, retire/reduce the union conformance test,
   then delete the delegates (umbrella #1582 Phase A exit, after a vulture sweep).
2. **Signature indirection.** `inspect.signature(GitHub.name)` returns a
   `functools.wraps`-copied signature, not one from a natively-defined method.
   Verified compatible with the conformance test's string comparison; a consumer
   doing deep introspection (e.g. reading `__code__` argcount off `GitHub.name`)
   would see the delegate. None found in the census.
3. **Hot-path indirection.** Every call crosses one delegate + one collaborator
   `__getattr__` hop until the delegates are deleted at Phase A exit.

If the operator rejects residual 1 (the facade), the honest answer flips: `GitHub`
**cannot** reach < 6 lexical members while keeping the current class-level
conformance test and the two `isinstance` sites, because presenting 40
correctly-signed methods at class level with <= 5 `def`s is only possible by
installing them as non-`def` attributes. The choice is: facade (metric passes,
disclosed residual) or keep `GitHub` fat. There is no third architecture that
satisfies both the metric and the unchanged conformance test.


## 9. Ratchet, gate, and stop-conditions

- **Ratchet as definition of done.** Each of L02-L09 lowers the committed
  `.attachment-budgets.json` GitHub `member_count` to the post-move value in the
  same PR and runs `uv run python -m charlie_work.attachment_contracts baseline --ratchet` so the budget
  can only descend. L01 lands the machinery at 53 (delta 0); the net across L01+L02
  is already negative. Final committed value: 2.
- **AST gate.** Every move PR runs `uv run charlie ast-equivalence-check --base
  origin/main`; the moved `FunctionDef` must be byte-identical between removal and
  destination (the gate exits 0 as evidence, and the reviewer -- opus48 for any PR
  touching protocol surface -- reads the diff-derived symbol set). A body that
  cannot move verbatim (e.g. an import path that must change) splits into a
  move-PR + an edit-PR.
- **Conformance gate.** The union test plus the seven sub-protocol tests must be
  green on every PR.
- **workflow.py.** `:4075` `isinstance(self.gh, GitHub)` must keep passing; any PR
  touching `workflow.py` is opus48-reviewed and human-merged (umbrella lane rule).

**Stop conditions (escalate, do not push through):**
- The AST gate reports a non-verbatim move that is not a deliberate edit-split.
- `isinstance` or a conformance assertion goes red and the fix is not a delegate
  signature copy (indicates a real signature drift, not a mechanical gap).
- A double breaks in a way that presence-by-name cannot explain (indicates a
  double asserting on class identity, not the protocol).
- The 3.13.x `getattr_static` behavior changes under a Python upgrade (re-run the
  probe; the whole facade rests on it).
- After delegates are deleted at Phase A exit, a consumer still imports
  `GitHubLike` for a method the sub-protocols do not cover.


## 10. Provenance

Inputs adjudicated (archived under llibrary
`raw/analyses/2026-09-god-object-paydown/track2/`):
- `github-class-inventory.md` / `.json` -- class span, 53 members, capability
  tags, in-degree ranking, transport core, non-verbatim hazards.
- `caller-census.md` / `.json` -- tiers C1-C4, 146 patch sites (`run` 134), 117
  doubles, 39 DI signatures (all annotated `GitHubLike`), 2 isinstance sites.
- `toolchain-and-oracle.md` -- AST gate CLI, Protocol conformance scenarios, PR
  exemplar 92d27321, mergequeue/Aviator, `needs-design` semantics.

Primary-source verification performed in this pass:
- Metric definition: `src/charlie_work/attachment_contracts/archetypes.py:64-65,
  265, 276`.
- Budget authority: `.attachment-budgets.json` GitHub entry (member_count 53).
- Method line spans: AST over `src/charlie_work/github.py` at the Track 2 base
  (Section 6).
- Conformance test: `tests/test_githublike_protocol.py:87-128` (class-level
  signature requirement) and `:33-84` (five `__dict__`-asserted names).
- Facade viability: `scratchpad/track2/probe_d_facade.py` on CPython 3.13.5
  (Section 8.2-8.3).

Leaf issues (one `needs-design` issue each, blocked in sequence under #1582):
see the leaf-issue drafts and `manifest.json` produced alongside this design.
```
