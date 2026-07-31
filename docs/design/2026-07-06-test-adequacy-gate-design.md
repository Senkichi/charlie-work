# Test-Adequacy Gate — Design

**Status**: Draft v2 (design approved; revised after adversarial spec review, pending final operator read → implementation plan)
**Date**: 2026-07-06
**Author**: brainstormed with operator
**Scope**: `charlie-work` review pipeline (`janitor.py`, `workflow.review`, `prompts/review.md`, `config.py`, `labels.py`)
**Review**: revised against a 5-lens adversarial spec review (factual-accuracy, invariant-adherence, completeness, implementability, risk). Must-fix items MF1–MF4 and should-fix items folded in; see §13 for the revision log.

---

## 1. Problem

Worker agents (Devin by default, Claude Code first-class) fail at test-writing **reliably**, in two
observed modes:

- **"Skips them"** — the worker abuses the `unless not applicable` escape hatch in the worker prompt
  (`prompts/worker.md` step 6, `prompts/worker_claude_code.md` step 5) or simply omits tests, shipping
  a feature-only PR.
- **"Green but hollow"** — the worker adds tests that pass but assert nothing real (over-mocked,
  tautological, never exercise the changed behavior). CI goes green on worthless coverage.

Both are **gate-acceptance failures, not capability failures**: nothing downstream rejects a PR for
inadequate tests, so the worker is *permitted* to ship one. The two checks that nominally guard this
today verify **narration, not reality**:

1. **Janitor tier** — `_check_body` (`janitor.py:169`) fails a PR only if its *body text* fails the
   regex `\b(tests?|verified?|rationale|…)\b` (`janitor.py:33`). A worker writes "added tests" in the
   PR body and passes, regardless of whether one test file changed. It greps the *description*, not the
   *diff*.
2. **Review tier** — `prompts/review.md:40` instructs the adversarial LLM reviewer to approve only if
   "Tests or a strong no-test rationale are present," but hands it **no structural signal** and **no
   rubric for hollowness**. A confident PR body walks it past over-mocked tests.

The invariant we want, in the language of the project's own design rules (*"enforce invariants at
boundaries, not scattered defensive checks — make invalid states unrepresentable"*):

> **A PR whose diff adds/changes non-trivial product code but adds no test coverage, and carries no
> explicit auditable exemption, is unmergeable.**

## 2. Chosen direction (and rejected alternatives)

The operator chose a **root-cause gate** over the originally-floated "spawn haiku Claude Code workers to
write the tests" stage, for these reasons:

- **Single point of enforcement.** The failure is that the gate *accepts* bad tests. Fix the gate, and
  every worker family is covered — Devin *and* Claude Code — with no new worker lifecycle, no
  same-branch sequencing between two workers, and no per-PR worker spend.
- **Capability mismatch of the rejected idea.** "Green but hollow" is a *reasoning* failure (knowing
  what is worth asserting). Handing that to Haiku — weaker than Devin, and asked to test code it did
  not author — risks producing hollow tests from a cheaper model. Whatever writes the tests, *something*
  must still judge whether they are real; that judge is the actual missing piece.

**Rejected / deferred alternatives:**

- *Dedicated test-writer worker stage* — deferred. A gate that *rejects* inadequate tests is the
  invariant; a test-writer is at most a remediation strategy layered on later (and the existing rework
  loop already re-dispatches a worker with corrective instructions, see §6).
- *Prompt-only hardening* (remove the "not applicable" escape, demand a coverage claim) — insufficient
  alone; it still relies on the same agent grading its own homework, and produces no enforceable signal.
  **Note:** the `Test-exempt:` escape hatch (§5.1/D3) is a *narrow, auditable* exemption, not a return
  to self-grading — its reason is treated as an adversarially-verified claim by Tier 2 (§5.2), not an
  accepted fact.

## 3. Goals / non-goals

**Goals**

- Make a **pure "skip"** (product code changed, no test file touched) a deterministic, near-zero-cost
  hard failure that works on **any** consumer repo.
- Materially raise the cost of shipping **"green but hollow"** tests by giving the existing adversarial
  reviewer a structural signal and an explicit rubric.
- Close the loop automatically: a structural failure re-dispatches the worker with corrective
  instructions and escalates to a human after the existing rework cap — **once per real head advance**,
  not once per polling pass (§6; the existing `_check_no_op_rework` gate blocks unchanged-head re-review).
- Preserve every project invariant (frozen config dataclasses, atomic JSON writes, error-as-values,
  label/state authority incl. a **valid label-edge sequence**, adapters non-blocking).

**Non-goals (v1)**

- Fully-deterministic detection of *semantic* hollowness. A hollow test still executes the changed
  line, so even line-coverage cannot catch it; only mutation testing can, and that is deferred (§9).
- Deterministic detection of *assertionless* test additions on repos whose assertion style is not
  configured — that case downgrades to a Tier-2 warning by default (§5.1 step 5; resolves R1↔step-5).
- Requiring coverage tooling in consumer repos. `charlie-work` is repo-agnostic (`--repo` targets
  job-cannon, empericus, itself); its own `pyproject.toml` ships no coverage tooling
  (`dev = [pytest, ruff]`, verified). The gate must not hard-require `pytest-cov`.
- Changing worker prompts or adding a new worker type.

## 4. Design decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Enforcement locus | **Root-cause gate**, worker-agnostic |
| D2 | Coverage assumption | **Assume none** — Tiers 1+2 are the repo-agnostic floor; diff-coverage is an opt-in, gracefully-skipped extension (§9) |
| D3 | Exemption mechanism | **Explicit structured marker** `<exempt_marker> <reason>` (default marker `Test-exempt:`; regex built from `config.exempt_marker`, non-empty reason) in the PR body, replacing the fuzzy body-grep as the authoritative signal; the reason is Tier-2-verified (§5.2) |
| D4 | Structural-failure routing | **Auto `request_changes` → rework** via the existing `record_review` path; idempotency is enforced upstream by `_check_no_op_rework` (no new guard), and the failure path traverses a valid label-edge sequence (§6) |
| D5 | Default state | **`enabled = False`** — opt-in, mirroring `CrossFamilyConfig` ("absent block = no-op", `config.py:236`, verified); flip to default-on only after per-repo override guidance (§12, NTH1) |

## 5. Architecture — three tiers

### 5.1 Tier 1 — structural check (deterministic, repo-agnostic, the hard gate)

A **self-contained** function added to `janitor.py`:

```
check_test_adequacy(diff: str, pr: dict, config: TestAdequacyConfig) -> TestAdequacyVerdict
```

**On `janitor.py` "purity":** the module's docstring claims "no I/O, no `gh` calls," but that is
**stale** — the module already shells out to git in 7 places (`_check_no_op_rework`,
`_get_unpushed_commit_info`, `check_operator_containment`; subprocess sites at lines 295, 313, 358, 383,
426, 441, 561). `check_test_adequacy` should nonetheless be written **pure** (takes the diff *string*;
no `gh`, no I/O, no subprocess) to avoid growing that surface. The implementation PR should also fix the
stale module docstring in one line (SF1).

**Placement.** In `workflow.review`, the check runs only when `config.test_adequacy.enabled` (short-
circuit — no diff parse at all when disabled; SF3), *after* `run_janitor` passes and the diff is fetched
for the packet (`workflow.py:1146`), and *before* the expensive containment/cross-family/packet work. A
draft/closed/conflicting/red-CI PR still exits at the existing `run_janitor` short-circuit
(`workflow.py:1118`) and never reaches the parse. Label sequencing on failure is specified in §6 (MF2).

**Algorithm** (all thresholds/markers/prefixes config-driven — no hardcoded lists in business logic):

1. **Parse the diff.** Factor the file/hunk splitting out of `check_operator_containment`
   (`janitor.py:404`) into a shared helper that yields `(filename, is_new_file, hunk_lines)` per file.
   **Note (MF3c):** that existing code only extracts `+++ b/` paths and `@@` headers into raw text
   blocks for byte-comparison — it does **not** tally added/removed lines. The `+`/`-` counting below is
   **net-new code**, not reused. Wrap the entire parse+classify in `try/except`: any exception →
   `ok=True` with a `warnings` entry ("diff unparseable — test-adequacy skipped"). It must **never**
   raise into `review()` (SF6; do not rely on the missing-key precedent for exception safety).
2. **Partition** changed files into `test` / `product` / `exempt` using `TestAdequacyConfig` globs
   (`test_path_globs`, `exempt_path_globs`), matched against the file path. Exempt = docs, examples,
   lockfiles, config, etc. `test_path_globs` wins over product classification (a file matching a test glob is a
   test file). **Accepted limitation (SF8):** product-grade logic hidden in `conftest.py` (blanket-
   globbed as test) is invisible to `added_product_loc`; documented, not defended in v1.
3. **Count added lines (MF3a/b).** An **added line** = a line in a file's hunk body beginning with `+`
   and not `+++`. A line is **blank/comment** iff, after `str.strip()`, it is empty **or** starts with
   any `comment_prefixes` entry (default `("#",)`). Multi-line/docstring comments are **not** detected
   in v1 (accepted false negative). Then:
   - `added_product_loc` = count of added, non-blank/comment lines across **product** files. A pure
     rename with no hunk body (100% similarity) contributes 0.
   - `added_test_loc` = the same across **test** files.
4. **Detect assertions (SF4).** For each added line in a **test** file, it "contains an assertion" iff
   any `assertion_markers` entry occurs as a **plain, case-sensitive substring** of the line (first
   match short-circuits). Markers appearing inside string literals/comments are accepted (v1 does not
   parse them out). `assertion_count` = number of added test lines containing an assertion.
5. **Verdict (resolves R1↔step-5):**
   - **Exempt** (D3): build the marker regex from config —
     `exempt_re = re.compile(rf"^{re.escape(config.exempt_marker)}\s*(?P<reason>.+)$", re.M)` (case-
     sensitive, non-empty reason) — and if the PR body matches → `ok=True`, record
     `exempt=(True, reason)` for Tier 2 to scrutinize. Skip the rest. The literal `Test-exempt:` is only
     the default; `config.exempt_marker` is authoritative (no hardcoded marker in business logic).
   - **Hard fail** iff `added_product_loc ≥ min_product_lines` **AND** no test file changed at all
     (`test_files_changed == 0`). This is the unambiguous "pure skip."
   - **Warn (→ Tier 2)** iff `added_product_loc ≥ min_product_lines` **AND** test files changed **AND**
     `assertion_count == 0`. Possibly hollow, possibly a bespoke assertion helper not in
     `assertion_markers`; the judge decides. **Unless** `require_assertions=True` (config knob for repos
     confident in their markers), in which case this is a **hard fail** instead.
   - Otherwise **pass** (optionally warn on a very low `added_test_loc / added_product_loc` ratio).

This makes a **pure skip** structurally unrepresentable at near-zero cost on any repo, and routes the
fuzzier "tests present but maybe hollow" signal to Tier 2 rather than hard-failing bespoke-assertion
repos.

**Return type** — a new frozen dataclass parallel to `JanitorVerdict`:

```python
@dataclass(frozen=True)
class TestAdequacyFacts:
    added_product_loc: int
    added_test_loc: int
    assertion_count: int
    test_files_changed: int
    untested_product_files: tuple[str, ...]
    exempt: bool
    exempt_reason: str            # "" when not exempt

@dataclass(frozen=True)
class TestAdequacyVerdict:
    ok: bool                       # False only on a hard fail
    failures: tuple[str, ...]      # names the untested product files + LOC
    warnings: tuple[str, ...]      # low-ratio / zero-marker / unparseable signals for Tier 2
    facts: TestAdequacyFacts
```

`facts` is consumed verbatim by Tier 2 so the LLM starts from hard numbers rather than re-deriving them
(the "scout inline before fanning out" principle).

### 5.2 Tier 2 — adversarial rubric (LLM, sharpens the existing reviewer)

Upgrade `prompts/review.md`:

- New **"## Test adequacy"** review step that forces a **behavior-coverage table**: for each behavior
  the diff adds or changes, name the specific test that would fail if that behavior regressed. Behaviors
  with no such test become findings.
- Explicit **hollow-test rejection heuristics**: reject tests that only assert a mock was called;
  re-assert constants; contain assertions that cannot fail (`assert True`, `assert x == x`); or never
  import/exercise the changed symbol.
- **Exemption scrutiny (SF2):** when `facts.exempt`, instruct the reviewer to treat the exemption
  **reason as a claim to verify against the diff**, not a fact to accept — a bogus "Test-exempt: n/a"
  should draw a `request_changes`.
- Add *"every non-exempt changed behavior has a genuine regression test"* to the approval criteria list
  (`prompts/review.md:35`).
- Inject a new **`$test_adequacy_section`** (rendered from `TestAdequacyFacts` + warnings) using the
  same section-injection pattern as `$janitor_section` / `$cross_family_section` (`workflow.py:1180`,
  `prompts/review.md:22`). When `enabled=False` the section renders as a fixed empty string (SF3).

**Honest limit.** Tier 2 is LLM-based, so per the project's own philosophy (deterministic checks *gate*;
LLM findings *inform verdicts* — cf. cross-family "leads, never merge gates") it is **not** a new hard
deterministic block. It is a sharper rubric feeding the normal `request_changes` verdict. The
"deterministic FAILURE" from D1 applies fully to a **pure skip** (Tier 1 hard fail); *hollow* and
*assertionless-on-unconfigured-repos* are raised by Tier 2 and, optionally later, by mutation testing
(§9).

### 5.3 Tier 3 — diff-coverage (opt-in, deferred, documented extension point)

Not in v1. When a consumer repo sets `coverage_command` in config, a future module runs it in the PR's
worktree, maps covered lines onto the diff's added product lines, and hard-fails uncovered added lines
above `min_diff_coverage`. Absent/erroring tooling → warn-and-skip (error-as-value). Config fields are
reserved now (§7) so enabling it later is additive.

## 6. Integration & auto-rework routing (D4)

`workflow.review` flow is unchanged through the janitor gate and diff fetch. **New logic**, immediately
after the diff is fetched (`workflow.py:1146`) and before the containment/cross-family/packet work:

```
if not config.test_adequacy.enabled:
    test_adequacy_section = ""              # SF3: skip the check entirely (the diff is still
                                            # fetched at workflow.py:1146 for the packet either way)
else:
    verdict = check_test_adequacy(diff, pr, config.test_adequacy)   # never raises (§5.1 step 1)
    if not verdict.ok:
        # Idempotency is ALREADY enforced UPSTREAM — no dedup guard here (corrected after
        # review; the earlier draft's per-poll-thrash rationale was unreachable). run_janitor
        # runs _check_no_op_rework (janitor.py:227) at the janitor gate (workflow.py:1117),
        # which hard-fails when pr_state.decision == "request_changes" AND headRefOid ==
        # reviewed_head_sha, returning at workflow.py:1118 BEFORE the diff fetch (1146). So
        # once pass 1 records request_changes at HEAD, any later pass on the UNCHANGED head
        # exits at the janitor gate and Tier 1 never re-runs — the counter is never
        # re-incremented. Tier 1 is reached ONLY when the head advanced since the last
        # request_changes, so the counter increments once per REAL head advance, escalating
        # after max_rework_cycles genuine reworks. Adding a second head-SHA check here would
        # be a scattered defensive check for an invariant already enforced at the janitor
        # boundary (CLAUDE.md single-point-of-enforcement). Edge: this relies on
        # reviewed_head_sha being recorded — it is whenever gh returns headRefOid; if
        # headRefOid is ever unavailable the upstream guard degrades, an accepted edge shared
        # with the LLM verdict path.
        #
        # MF2 label-edge validity — traverse the SAME edges as an LLM request_changes.
        # review()'s review_started transition (add pr_open+reviewing, remove needs_rework)
        # normally runs at the END of review() (workflow.py:1258); here we must apply it
        # BEFORE recording the deterministic verdict. in_progress persists until merge
        # (nothing removes it earlier — labels.py:40-53), so the sequence is:
        #   {in_progress} --review_started--> {in_progress,pr_open,reviewing}
        #                 --rework_requested--> {in_progress,pr_open,needs_rework}
        # — the SAME terminal label set as an LLM request_changes. Reusing the
        # rework_requested edge alone on {in_progress} would add needs_rework and try to
        # remove an absent reviewing → {in_progress,needs_rework}: pr_open is never added —
        # the WRONG terminal set (transition() records a partial-failure on the absent-reviewing
        # removal rather than raising, labels.py:67-73). Two cheap label calls; still before any
        # cross-family/packet LLM spend.
        transition(self.gh, self.config.labels, issue_number, "review_started")
        summary = render_test_adequacy_summary(verdict)      # templated, non-empty
        return self.record_review(pr_number, "request_changes", summary=summary)
    test_adequacy_section = render_test_adequacy_section(verdict.facts, verdict.warnings)
# continue: containment, cross_family, packet render (now with $test_adequacy_section)
```

**Widened `review()` contract (SF7 — document + audit).** Today `review()` (`workflow.py:1092-1285`)
never records a verdict or mutates `request_changes_count`; it produces a packet or short-circuits at
the janitor gate. D4 lets `review()` itself issue a `request_changes` verdict and advance/terminate the
rework loop. This widening must be **stated in the `review()` docstring**, and the implementation must
**audit existing callers and tests** (`bash-rats`/`loop`, `test_*`) for any assumption that `review()`
is verdict-neutral before landing.

**Why `record_review` and not a janitor-style silent block:** a "no tests" PR does not self-heal the
way a red-CI PR does (the worker already finished; nothing re-triggers it). Routing through
`record_review` (`workflow.py:1287`) reuses the **entire existing rework machinery**, verified
end-to-end against source:

- It increments the **durable per-PR `request_changes_count`** and, at `max_rework_cycles`
  (default 2, `config.py:119`), **escalates to `agent:human-needed`** (`workflow.py:1344-1351`). Because
  `_check_no_op_rework` blocks unchanged-head re-reviews upstream (see the routing note above), that is
  **2 real head-advancing reworks**, not 2 polling passes.
- It writes the **rework prompt** (`_write_rework_prompt` → `prompts/rework.md`, `workflow.py:2457`),
  feeding our templated summary (untested files + LOC + the `Test-exempt:` instruction) as
  `review_summary`, so the re-dispatched worker gets concrete, actionable guidance.
- It sets issue status `rework_requested` and moves labels via the `rework_requested` edge
  (`labels.py:44` = add `needs_rework`, remove `reviewing`; `workflow.py:1364`).
- **`dispatch_rework` (`workflow.py:1984`) is state-driven** — it selects any issue whose
  `status == "rework_requested"` with an open PR, filters out escalated PRs (`workflow.py:2102`), and
  runs in the standard `bash-rats` pass (`workflow.py:1903`). The loop closes with no new state machine.

**Manual adapter (SF8-adjacent):** `dispatch_rework` early-returns for the manual adapter
(`workflow.py:1996`). Under `manual`, the deterministic verdict still records correctly and writes the
rework prompt; the human operator picks it up. No auto-re-dispatch, consistent with manual semantics.

**Concurrency & fetch race.** At the injection point `review()` holds no state lock; `record_review`
takes its own `state_lock` (not re-entrant here). `record_review` re-fetches the PR
(`workflow.py:1299`) rather than reusing the `pr` dict `review()` already holds; between the two fetches
`headRefOid` could change, pinning `reviewed_head_sha` to a head Tier 1 never judged (NTH2). In v1 this
race is **accepted** (millisecond window; `_check_no_op_rework` already blocks re-recording on an
unchanged head, and the race is shared with the normal LLM verdict path); §9 keeps the shared
verdict-core refactor as the durable fix.

**Interaction with `_check_no_op_rework`** (`janitor.py:227`): if a re-dispatched worker pushes nothing,
that janitor check blocks it; if it pushes but still adds no tests, the advanced head passes
`_check_no_op_rework`, so Tier 1 re-runs on the *new* head and fails again, counter++. Both paths
converge on escalation. No conflict.

## 7. Config surface

A new frozen dataclass (mirrors `LabelConfig` / `CrossFamilyConfig` shape and the additive-only config
rule — no existing fields removed):

```python
@dataclass(frozen=True)
class TestAdequacyConfig:
    enabled: bool = False                       # D5 — opt-in; absent block = no-op
    min_product_lines: int = 10                 # below this, skip (small fixes may ride existing tests)
    test_path_globs: tuple[str, ...] = ("tests/**", "test_*.py", "*_test.py", "conftest.py")
    exempt_path_globs: tuple[str, ...] = ("*.md", "docs/**", "examples/**", "*.lock", "*.toml", "*.cfg", "*.ini")
    assertion_markers: tuple[str, ...] = (
        "assert ", "pytest.raises", "raises(", "assert_called", "self.assert",
    )
    comment_prefixes: tuple[str, ...] = ("#",)   # MF3 — blank/comment exclusion, per-repo
    require_assertions: bool = False             # R1 — hard-fail zero-marker test additions when True
    exempt_marker: str = "Test-exempt:"          # D3 — structured PR-body escape hatch
    # Tier 3 (reserved, deferred — §5.3)
    coverage_enabled: bool = False
    coverage_command: tuple[str, ...] = ()
    min_diff_coverage: float = 0.0
```

**Loading & validation (MF4 — explicit, not `cls(**data)`).** In `load_config`, `test_adequacy` must get
its own validation block, not a bare `_build_section` / `cls(**data)`:

- Each of the **five** tuple-of-str fields (`test_path_globs`, `exempt_path_globs`, `assertion_markers`,
  `comment_prefixes`, `coverage_command`) gets an explicit list→tuple coercion **and** type check
  mirroring the existing `required_checks` pattern (`config.py:344`); a non-list or non-str element →
  `ConfigError`.
- **Scalar** fields get `isinstance` rejection blocks (mirroring `base_ref`'s `isinstance(str)` check,
  `config.py:327`), each raising a readable `ConfigError` rather than a confusing `TypeError`:
  `min_product_lines` (int), `min_diff_coverage` (float), `exempt_marker` (str, non-empty), and the
  bools `enabled` / `coverage_enabled` / `require_assertions`.
- `config.exempt_marker` is the single source for the exemption regex (§5.1 step 5) — no literal marker
  string appears in business logic.
- Unknown-key rejection via the existing `_build_section` valid-key diff (`config.py:291`).
- Added to `OrchestratorConfig` (`config.py:263`) as `test_adequacy: TestAdequacyConfig` with a
  `field(default_factory=TestAdequacyConfig)`.

All defaults are Python-shaped but **overridable**, so a non-Python consumer repo can retarget globs,
comment prefixes, and assertion markers without code changes.

## 8. Invariant adherence & error handling

- **Frozen dataclasses** — `TestAdequacyConfig`, `TestAdequacyVerdict`, `TestAdequacyFacts` all
  `@dataclass(frozen=True)`.
- **Self-contained `check_test_adequacy`** — takes the diff *string*; no `gh`, no I/O, no subprocess.
  (Does **not** rely on a false "janitor is pure" claim — see §5.1; the module already does git I/O.)
- **Exception safety (SF6)** — the diff parse/classify is wrapped in `try/except`; any exception →
  `ok=True` + warning. A malformed or binary diff never raises into `review()`. Proven by a fixture
  test (§10).
- **Error-as-values** — the check returns a verdict object; it never raises. Tier-3 (future) coverage
  tooling absent/erroring → warn-and-skip.
- **Atomic writes** — no new state files; decision/rework artifacts go through the existing
  `record_review` / `_write_json` atomic paths.
- **Label/state authority incl. valid edge sequence (MF2)** — the failure path traverses
  `review_started` then `rework_requested`, an existing valid path through `labels.py`'s edge table;
  no new edge, no orphaned/duplicated active labels.
- **Zero behavior change when disabled** — `enabled=False` short-circuits before any parse (SF3), so an
  existing deployment sees no difference *and no added cost* until it opts in.

## 9. Deferred extensions (documented, not built)

- **Tier 3 diff-coverage** — opt-in per repo (config reserved in §7); graceful skip when tooling absent.
- **Mutation-of-diff** — the only fully-deterministic detector of *semantic* hollowness (mutate changed
  lines, assert the new tests catch it). Expensive (suite runs ×N), needs `mutmut`/`cosmic-ray`; a
  diff-scoped opt-in at most.
- **Shared verdict-core refactor (NTH2)** — extract the `request_changes` recording path so `review()`
  passes its already-fetched `pr` dict in, eliminating the re-fetch race (§6). Deferred;
  `_check_no_op_rework` already blocks unchanged-head re-recording, so the race is narrow in v1.
- **Default-on flip** — once the structural gate is proven and per-repo override guidance exists (§12),
  change D5 to `enabled=True`.

## 10. Testing plan

- **Tier 1** is a self-contained function → **table-driven** unit tests against diff fixtures, in the
  style of `tests/test_janitor.py`:
  - feature + real assertions → **pass**
  - feature + no test files (pure skip) → **hard fail**
  - feature + test file with zero recognized markers → **warn/pass** (default) / **hard fail** when
    `require_assertions=True`
  - docs-only / config-only diff → **pass** (exempt globs)
  - rename-only / 100%-similarity move → **pass** (0 added product LOC)
  - rename+modify diff (near-zero added lines on a changed product file) → boundary-documented behavior
  - binary-file diff → **pass with warning**, and a **malformed diff → pass, never raises** (SF6)
  - feature + valid `Test-exempt: <reason>` → **pass** (exempt recorded); `Test-exempt:` with empty
    reason → **not exempt** (regex requires non-empty); a custom `exempt_marker` override is honored
    (regex built from `config.exempt_marker`, not the literal)
  - product diff below `min_product_lines` → **pass**
  - test-only diff (no product change) → **pass**
  - bugfix that only *modifies* existing tests (with assertions) → **pass**
  - `conftest.py` carrying non-trivial logic → documents the accepted-evasion limitation (SF8)
- **Routing** — unit test (mock `gh`) that a hard-fail verdict: (a) applies `review_started` then
  `record_review("request_changes", …)` with a non-empty summary; (b) is **not re-recorded on an
  unchanged head** — assert the upstream `_check_no_op_rework` janitor gate blocks the re-review so
  Tier 1 is unreached and `request_changes_count` is unchanged on the second pass; (c) increments once
  per new head and escalates at the cap; (d) leaves the valid label set `{in_progress, pr_open,
  needs_rework}` (never `{in_progress, needs_rework}` skipping `pr_open`).
- **Tier 2** — prompt-render test that `$test_adequacy_section` substitutes (populated and empty-when-
  disabled) and the new criteria render; golden-file check on `review.md`. LLM judgment is not
  unit-testable.
- **Config** — load/validation tests: defaults, unknown-key `ConfigError`, list→tuple coercion for all
  tuple fields, non-mapping/non-numeric rejection for scalars.

## 11. Risks & open questions

- **R1 — assertion-marker gap (resolved).** A repo using bespoke assert helpers (`expect(x)`,
  `should.equal`) with no configured marker hits `assertion_count==0`. Per §5.1 step 5 this is a
  **warning → Tier 2** by default (not a hard fail), so no false-positive blocking; repos confident in
  their markers set `require_assertions=True`. Enabling on a non-pytest-style repo therefore requires
  setting `assertion_markers` first — a documented prerequisite (§12), not just tuning advice.
- **R2 — `min_product_lines` tuning.** Too low → false positives on tiny features; too high → misses
  small untested changes. Default 10; revisit with production data.
- **R3 — rework-budget sharing.** A Tier-1 hard fail consumes **one rework cycle per real head
  advance** — the existing `_check_no_op_rework` gate (janitor.py:227) ensures an unchanged head never
  re-enters Tier 1 (§6) — sharing the 2-cycle budget with LLM review rounds. This is desirable: total
  rework is capped regardless of reason, and escalation is the correct terminal state for a worker that
  will not add tests.
- **R4 — body-grep coexistence.** `_check_body`'s existing `require_tests_or_rationale` grep stays
  (additive rule). It becomes redundant-but-harmless once Tier 1 is authoritative; a later cleanup may
  downgrade it to a warning. Flagged, not changed, in v1.
- **R5 — `conftest.py` evasion (accepted).** Product logic placed in `conftest.py` is globbed as test
  and escapes `added_product_loc`. Adversarial and unlikely from a good-faith worker; documented as an
  accepted v1 false-negative, revisitable by refining the glob.

## 12. Rollout

1. Land Tiers 1+2 + routing (MF2 label sequencing; idempotency via the existing `_check_no_op_rework`
   gate) + config, behind `enabled=False`.
2. Enable on one Python repo (charlie-work itself), observe rework/escalation rates and false positives.
3. Tune `min_product_lines` / `assertion_markers` / `require_assertions` from data.
4. **Before any D5 default-on flip (NTH1):** document that non-Python repos MUST set language-appropriate
   `test_path_globs`, `comment_prefixes`, and `assertion_markers` first, or every PR fails/warns; a
   default-on flip without per-repo overrides would break non-Python repos permanently. Surface this as
   a `charlie doctor` check (not `load_config`, which has no repo file access) that warns when
   `enabled=True` but no repo file matches any configured `test_path_glob`.
5. Consider Tier 3 for repos with coverage tooling.

## 13. Revision log

- **v3 (2026-07-06)** — revised after a verification pass (2 blocking findings). **Removed the MF1
  head-SHA dedup guard as redundant:** `_check_no_op_rework` (janitor.py:227) already blocks
  unchanged-head re-review at the janitor gate *before* Tier 1 (workflow.py:1117-1118 precede 1146), so
  the rework counter increments once per real head advance without a second check — the
  single-point-of-enforcement the earlier draft violated. Made `exempt_marker` authoritative (regex
  derived from `config.exempt_marker`, no hardcoded literal). Added scalar/bool config validation,
  corrected the tuple-field count (five) and `test_files_changed` identifier, clarified the diff is
  still fetched for the packet when disabled, and moved the non-Python guard to a `doctor` check.
- **v2 (2026-07-06)** — revised after a 5-lens adversarial spec review (499k tokens, 19 raw findings).
  Incorporated must-fix MF1 (head-SHA idempotency — **superseded in v3**), MF2 (valid label-edge sequence), MF3 (precise
  LOC/comment/assertion definitions + `comment_prefixes`; corrected "no reusable counting logic exists"),
  MF4 (explicit config coercion); should-fix SF1–SF8 (stale purity claim, gameable exemption, disabled
  short-circuit, assertion match semantics, R1↔step-5 contradiction via `require_assertions`, diff-parse
  exception safety, widened-`review()`-contract documentation, test-fixture + `conftest.py` gaps); and
  nice-to-have NTH1 (non-Python override prerequisite) / NTH2 (fetch-race note).
- **v1 (2026-07-06)** — initial design from operator brainstorming.
