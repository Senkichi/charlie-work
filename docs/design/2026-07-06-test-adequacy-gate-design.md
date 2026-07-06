# Test-Adequacy Gate — Design

**Status**: Draft (design approved, pending spec review + implementation plan)
**Date**: 2026-07-06
**Author**: brainstormed with operator
**Scope**: `charlie-work` review pipeline (`janitor.py`, `workflow.review`, `prompts/review.md`, `config.py`)

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

> **A PR whose diff adds/changes non-trivial product code but adds no real test assertions, and carries
> no explicit auditable exemption, is unmergeable.**

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

## 3. Goals / non-goals

**Goals**

- Make **"skips them"** a deterministic, zero-LLM-cost hard failure that works on **any** consumer repo.
- Materially raise the cost of shipping **"green but hollow"** tests by giving the existing adversarial
  reviewer a structural signal and an explicit rubric.
- Close the loop automatically: a structural failure re-dispatches the worker with corrective
  instructions and escalates to a human after the existing rework cap, rather than stalling.
- Preserve every project invariant (frozen config dataclasses, atomic JSON writes, error-as-values,
  label/state as the source of truth, adapters non-blocking).

**Non-goals (v1)**

- Fully-deterministic detection of *semantic* hollowness. A hollow test still executes the changed
  line, so even line-coverage cannot catch it; only mutation testing can, and that is deferred (§9).
- Requiring coverage tooling in consumer repos. `charlie-work` is repo-agnostic (`--repo` targets
  job-cannon, empericus, itself); its own `pyproject.toml` ships no coverage tooling
  (`dev = [pytest, ruff]`). The gate must not hard-require `pytest-cov`.
- Changing worker prompts or adding a new worker type.

## 4. Design decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Enforcement locus | **Root-cause gate**, worker-agnostic |
| D2 | Coverage assumption | **Assume none** — Tiers 1+2 are the repo-agnostic floor; diff-coverage is an opt-in, gracefully-skipped extension (§9) |
| D3 | Exemption mechanism | **Explicit structured marker** `Test-exempt: <reason>` in the PR body, replacing the fuzzy body-grep as the authoritative signal |
| D4 | Structural-failure routing | **Auto `request_changes` → rework** via the existing `record_review` path (§6) |
| D5 | Default state | **`enabled = False`** — opt-in, mirroring `CrossFamilyConfig` ("absent block = no-op", `config.py:236`); flip to default-on once proven in production |

## 5. Architecture — three tiers

### 5.1 Tier 1 — structural check (deterministic, repo-agnostic, the hard gate)

A **pure** function added to `janitor.py` (keeping that module's "no I/O, no `gh` calls — the caller
feeds data in" contract):

```
check_test_adequacy(diff: str, pr: dict, config: TestAdequacyConfig) -> TestAdequacyVerdict
```

It runs as **"janitor phase 2"**: in `workflow.review`, *after* the cheap `run_janitor` gate passes and
the diff has already been fetched for the packet (`workflow.py:1146`). This ordering is deliberate — a
draft / closed / conflicting / red-CI PR still exits at the existing `run_janitor` short-circuit
(`workflow.py:1118`) and never reaches the diff parse.

**Algorithm (all thresholds/markers config-driven — no hardcoded lists in business logic):**

1. Parse the unified diff (reuse the diff-parsing machinery already in
   `check_operator_containment`, `janitor.py:404`, factored into a shared helper).
2. **Partition** changed files into `test` / `product` / `exempt` using `TestAdequacyConfig` globs
   (`test_path_globs`, `exempt_path_globs`). Exempt = docs, lockfiles, config, etc.
3. Compute **added product LOC** (added lines in product files, excluding blank/comment) and
   **added test LOC**.
4. Detect **assertions** among added test lines via `assertion_markers`
   (default: `assert `, `pytest.raises`, `raises(`, `assert_called`, `self.assert`).
5. **Fail** iff:
   `added_product_LOC ≥ min_product_lines`
   **AND** (`no test file changed` **OR** `assertions_in_added_test_lines == 0`)
   **AND** the PR body contains no `exempt_marker` (`Test-exempt:`) line.
6. Otherwise pass (optionally emitting warnings — e.g. a very low test/product LOC ratio — that get
   surfaced to Tier 2).

This makes **"skips them"** and the crudest **"hollow"** (a test file with zero assertions)
structurally unrepresentable, at near-zero cost, on any repo.

**Return type** — a new frozen dataclass parallel to `JanitorVerdict`:

```python
@dataclass(frozen=True)
class TestAdequacyVerdict:
    ok: bool
    failures: tuple[str, ...]        # human-readable, name the untested product files + LOC
    warnings: tuple[str, ...]        # low-ratio / near-miss signals for the reviewer
    facts: TestAdequacyFacts         # structured numbers for the Tier-2 injected section
```

`TestAdequacyFacts` carries `added_product_loc`, `added_test_loc`, `assertion_count`,
`untested_product_files`, `exempt` (bool + reason) — consumed verbatim by Tier 2 so the LLM starts from
hard numbers rather than re-deriving them (the "scout inline before fanning out" principle).

### 5.2 Tier 2 — adversarial rubric (LLM, sharpens the existing reviewer)

Upgrade `prompts/review.md`:

- New **"## Test adequacy"** review step that forces a **behavior-coverage table**: for each behavior
  the diff adds or changes, name the specific test that would fail if that behavior regressed. Behaviors
  with no such test become findings.
- Explicit **hollow-test rejection heuristics**: reject tests that only assert a mock was called;
  re-assert constants; contain assertions that cannot fail (`assert True`, `assert x == x`); or never
  import/exercise the changed symbol.
- Add *"every non-exempt changed behavior has a genuine regression test"* to the approval criteria list
  (`prompts/review.md:35`).
- Inject a new **`$test_adequacy_section`** (rendered from `TestAdequacyFacts`) using the same
  section-injection pattern as `$janitor_section` / `$cross_family_section`
  (`workflow.py:1180`, `prompts/review.md:22`).

**Honest limit.** Tier 2 is LLM-based, so per the project's own philosophy (deterministic checks *gate*;
LLM findings *inform verdicts* — cf. cross-family "leads, never merge gates") it is **not** a new hard
deterministic block. It is a sharper rubric feeding the normal `request_changes` verdict. The
"deterministic FAILURE" from D1 applies fully to **skips** and to the **structural** slice of hollow
(Tier 1); *semantic* hollow is raised by Tier 2 and, optionally later, by mutation testing (§9).

### 5.3 Tier 3 — diff-coverage (opt-in, deferred, documented extension point)

Not in v1. When a consumer repo sets `coverage_command` in config, a future module runs it in the PR's
worktree, maps covered lines onto the diff's added product lines, and hard-fails uncovered added lines
above `min_diff_coverage`. Absent/erroring tooling → warn-and-skip (error-as-value). Config fields are
reserved now (§7) so enabling it later is additive.

## 6. Integration & auto-rework routing (D4)

`workflow.review` flow, unchanged through the janitor gate and diff fetch. **New step**, immediately
after the diff is fetched (`workflow.py:1146`) and before the containment/cross-family/packet work:

```
verdict = check_test_adequacy(diff, pr, config.test_adequacy)
if config.test_adequacy.enabled and not verdict.ok:
    # deterministic reviewer issues a verdict — no LLM spend
    summary = render_test_adequacy_summary(verdict)          # templated, non-empty
    return self.record_review(pr_number, "request_changes", summary=summary)
# pass: stash verdict.facts for the $test_adequacy_section, continue
```

**Why `record_review` and not a janitor-style silent block:** a "no tests" PR does not self-heal the
way a red-CI PR does (the worker already finished; nothing re-triggers it). Routing through
`record_review` (`workflow.py:1287`) reuses the **entire existing rework machinery**, verified
end-to-end:

- It increments the **durable per-PR `request_changes_count`** and, at `max_rework_cycles`
  (default 2, `config.py:119`), **escalates to `agent:human-needed`** instead of looping forever
  (`workflow.py:1349`). A worker that keeps shipping testless PRs escalates after 2 bounces.
- It writes the **rework prompt** (`_write_rework_prompt` → `prompts/rework.md`, `workflow.py:2457`),
  feeding our templated summary (the untested files + LOC + the `Test-exempt:` instruction) as
  `review_summary`, so the re-dispatched worker gets concrete, actionable guidance.
- It sets issue status `rework_requested` and moves labels (`workflow.py:1364`).
- **`dispatch_rework` (`workflow.py:1984`) is state-driven** — it selects any issue whose
  `status == "rework_requested"` with an open PR, filters out escalated PRs (`workflow.py:2102`), and
  runs in the standard `bash-rats` pass (`workflow.py:1903`). The loop closes with no new state machine.

**Concurrency safety.** At the injection point `review()` holds no state lock (the janitor gate's lock
sections are already closed); `record_review` acquires its own `state_lock` and is therefore not
re-entrant here. The redundant `gh pr view` inside `record_review` is harmless; an optional refactor can
extract a shared verdict-recording core that both `review` and `record_review` call, to avoid the second
fetch.

**Interaction with `_check_no_op_rework`** (`janitor.py:227`): if a re-dispatched worker pushes nothing,
the existing no-op-rework janitor check blocks it; if it pushes but still adds no tests, Tier 1 fails
again (counter++). Both paths converge on escalation. No conflict.

## 7. Config surface

A new frozen dataclass (mirrors `LabelConfig` / `CrossFamilyConfig` shape and the additive-only config
rule — no existing fields removed):

```python
@dataclass(frozen=True)
class TestAdequacyConfig:
    enabled: bool = False                       # D5 — opt-in; absent block = no-op
    min_product_lines: int = 10                 # below this, skip (small fixes may ride existing tests)
    test_path_globs: tuple[str, ...] = ("tests/**", "test_*.py", "*_test.py", "conftest.py")
    exempt_path_globs: tuple[str, ...] = ("*.md", "docs/**", "*.lock", "*.toml", "*.cfg", "*.ini")
    assertion_markers: tuple[str, ...] = (
        "assert ", "pytest.raises", "raises(", "assert_called", "self.assert",
    )
    exempt_marker: str = "Test-exempt:"         # D3 — structured PR-body escape hatch
    # Tier 3 (reserved, deferred — §5.3)
    coverage_enabled: bool = False
    coverage_command: tuple[str, ...] = ()
    min_diff_coverage: float = 0.0
```

- Added to `OrchestratorConfig` (`config.py:263`) as `test_adequacy: TestAdequacyConfig`.
- Loaded/validated in `load_config` with the same list→tuple coercion and unknown-key rejection as the
  other sections (`config.py:305`); a non-list glob/marker or non-numeric threshold → `ConfigError`.
- All defaults are Python-shaped but **overridable**, so a non-Python consumer repo can retarget globs
  and assertion markers without code changes.

## 8. Invariant adherence & error handling

- **Frozen dataclasses** — `TestAdequacyConfig`, `TestAdequacyVerdict`, `TestAdequacyFacts` all
  `@dataclass(frozen=True)`.
- **Purity of `janitor.py`** — `check_test_adequacy` takes the diff *string* as input; no `gh`, no I/O.
- **Error-as-values** — a diff that fails to parse is treated as "unknown" → warn, do not block (matches
  janitor's "missing key → skip the check" stance, `janitor.py:88`). Never raise into `review()`.
- **Atomic writes** — no new state files; the decision/rework artifacts are written by the existing
  `record_review` / `_write_json` atomic paths.
- **Label/state authority** — routing goes through `record_review`, so labels + `state.json` remain the
  single source of truth; nothing is inferred from chat.
- **Zero behavior change when disabled** — `enabled=False` default means an existing deployment sees no
  difference until it opts in.

## 9. Deferred extensions (documented, not built)

- **Tier 3 diff-coverage** — opt-in per repo (config reserved in §7); graceful skip when tooling absent.
- **Mutation-of-diff** — the only fully-deterministic detector of *semantic* hollowness (mutate changed
  lines, assert the new tests catch it). Expensive (suite runs ×N), needs `mutmut`/`cosmic-ray`; a
  diff-scoped opt-in at most.
- **Verdict-core refactor** — extract the shared request_changes recording path so `review` needn't
  re-fetch the PR (§6).
- **Default-on flip** — once the structural gate is proven in production, change D5 to `enabled=True`.

## 10. Testing plan

- **Tier 1** is a pure function → **table-driven** unit tests against diff fixtures, in the style of
  `tests/test_janitor.py`:
  - feature + real assertions → **pass**
  - feature + no test files → **fail**
  - feature + test file with zero assertions → **fail**
  - docs-only / rename-only diff → **pass** (exempt globs)
  - feature + `Test-exempt:` marker → **pass**
  - product diff below `min_product_lines` → **pass**
  - bugfix that only *modifies* existing tests (with assertions) → **pass**
  - diff-parse failure → **pass with warning** (never raises)
- **Routing** — unit test that a failing verdict calls `record_review("request_changes", …)` with a
  non-empty summary, increments the counter, and escalates at the cap (mock `gh`).
- **Tier 2** — prompt-render test that `$test_adequacy_section` substitutes and the new criteria render;
  golden-file check on `review.md`. LLM judgment itself is not unit-testable.
- **Config** — load/validation tests: defaults, unknown-key `ConfigError`, list→tuple coercion,
  non-mapping/non-numeric rejection.

## 11. Risks & open questions

- **R1 — assertion heuristic false-negatives.** A repo using a bespoke assertion helper (`expect(x)`,
  `should.equal`) with no default marker would read as "no assertions." *Mitigation:* markers are
  config-driven (§7); document the override. Warn-not-block on the ratio signal reduces blast radius.
- **R2 — `min_product_lines` tuning.** Too low → false positives on tiny features; too high → misses
  small untested changes. Default 10; revisit with production data.
- **R3 — rework-budget sharing.** A Tier-1 `request_changes` consumes one of the 2 rework cycles shared
  with substantive LLM review rounds. *Assessment:* desirable — total rework is capped regardless of
  reason, and escalation is the correct terminal state for a worker that will not write tests.
- **R4 — body-grep coexistence.** `_check_body`'s existing `require_tests_or_rationale` grep stays
  (additive rule). It becomes redundant-but-harmless once Tier 1 is authoritative; a later cleanup may
  downgrade it to a warning. Flagged, not changed, in v1.

## 12. Rollout

1. Land Tiers 1+2 + routing + config behind `enabled=False`.
2. Enable on one repo (charlie-work itself), observe rework/escalation rates.
3. Tune `min_product_lines` / markers from data.
4. Flip D5 default-on; consider Tier 3 for repos with coverage tooling.
