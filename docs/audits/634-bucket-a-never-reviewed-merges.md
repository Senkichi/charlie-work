# Audit #634: Bucket-A — five never-really-reviewed merges

## Scope

Issue #634 narrowed the audit to **bucket A only**: the five PRs that merged
with a hollow verdict (#597 signature — the reviewer echoed the prompt's
placeholder or recorded nothing, the fallback parser accepted it, and Aviator
auto-merged). The root cause is fixed on main (`b7ebec1`, #597/#611), so this
cannot recur. What remained was verifying the *code* those five merged, since
nothing ever reviewed it.

Operator update (2026-08-13): "approved as a fan-out over bucket-A only (the 5
never-really-reviewed PRs). #805 runs first; its method was validated on #627."

## Method

Each PR's merge commit was examined via `git show <merge-sha>` against the
current `origin/main` head. The diff and the current state of each changed file
were reviewed for:

1. Security issues (credential exposure, injection, path traversal, unsafe
   deserialization)
2. Correctness bugs (logic errors, race conditions, missing error handling)
3. Violations of project invariants (CLAUDE.md)
4. Resource leaks, unbounded operations, performance issues
5. Test quality (do tests actually test the behavior?)

Critical findings from the initial parallel review were **independently
verified** against the live code and, where applicable, empirically tested
(e.g. the `git patch-id --stable` whitespace claim was tested with real diffs).

## PRs audited

| PR | Merge SHA | Title | Files changed |
|----|-----------|-------|---------------|
| #578 | `7622a0d` | feat(observability): doctor probes and fleet-report budget line for api worker | cli.py, doctor.py, fleet_dispatch.py + 3 test files |
| #466 | `59f8d40` | fix(worktree): reparse-point-safe cleanup and orphan sweep | worktree.py + 1 test file |
| #387 | `4be4b45` | fix(janitor): route definitive required-check failures to rework | janitor.py, workflow.py + 2 test files |
| #386 | `52d2d38` | perf(auto-merge): carry forward approved verdict on clean rebase via patch-id | workflow.py + 1 test file |
| #364 | `1c109b2` | fix(github): scope merged_pr_list() query and guard unconditional call | github.py, workflow.py + 1 test file |

## Findings

### #578 — doctor probes and fleet-report budget line for api worker

**Verdict: CLEAN.** No security issues, no invariant violations, excellent test
coverage.

- **Security**: `doctor.py` reports only the environment variable *name*
  (`provider.api_key_env`), never its value. Tests verify the secret value does
  not leak into output. `base_url` is validated as HTTPS.
- **Invariants**: `ApiWorkerFleetReport` uses `@dataclass(frozen=True)`. All
  operations are read-only (no JSON state writes). Errors are contained with
  broad `except Exception` matching the established fleet-report pattern.
- **Tests**: Comprehensive — covers not-configured, disabled, all-checks-pass,
  missing-env-var, bad-URL, ledger corruption, budget headroom, partial
  enablement, spend from ledger, CLI wiring. Non-vacuous.

**LOW (info, not actionable)**: `fleet_dispatch.py:398` uses
`assert representative is not None` with a comment "configured_m > 0 guarantees
this". Assertions are stripped by `python -O`, but the surrounding logic
(lines 395–396 return `None` when `configured_m == 0`) makes this unreachable
in practice. Not a blocker; an explicit `if representative is None: return None`
would be more robust but is not required.

---

### #466 — reparse-point-safe worktree cleanup and orphan sweep

**Verdict: CLEAN.** The reparse-point safety mechanisms are correctly
implemented with multiple defense layers.

- **`_unlink_reparse_point`** (`worktree.py:1898`): Correctly unlinks
  directory symlinks/junctions without following into the target. On Windows,
  prefers `os.unlink` for directory symlinks, falls back to `os.rmdir` for
  junctions. Target is left untouched in all cases.
- **`_unlink_worktree_reparse_points`** (`worktree.py:1920`): Walks the tree
  with `os.walk(followlinks=False)`, unlinks all reparse points, and prunes
  them from `dirnames` so the walk never descends into them.
- **`_robust_rmtree`** (`worktree.py:1956`): Calls
  `_unlink_worktree_reparse_points` *first*, then `shutil.rmtree`. This is the
  correct order — all junctions/symlinks are removed before `rmtree` runs, so
  `rmtree` cannot follow them into a shared venv target.
- **Orphan sweep** (`worktree.py:4383–4407`): Skips junction children via
  `is_junction(child)`, spares live foreign worktrees via
  `is_live_foreign_worktree`, and uses `_robust_rmtree` for deletion. A
  `git worktree list` failure skips the sweep entirely and emits a
  `worktree_list_failed` attention event (the rework added in this PR).
- **Tests**: Windows-only tests verify junctions are not followed during
  deletion and shared venv contents are preserved. The
  `test_clean_worktrees_skips_orphan_sweep_when_worktree_list_fails` test
  covers the reviewed failure mode.

**Defense-in-depth observation (LOW, not actionable in this PR's scope)**:
The orphan sweep does not check whether `worktrees_dir` *itself* is a junction
before calling `iterdir()`. If `worktrees_dir` were a junction, `iterdir()`
would enumerate the target's contents. However:
- `worktrees_dir` is config-derived (`layout.worktrees_dir`), not
  attacker-controlled at runtime.
- The CLAUDE.md containment invariant is specifically about `ci_fleet`'s
  `discover_runner_instances` and `managed_root`, not the worktree cleanup
  subsystem.
- Each child is checked with `is_junction(child)` before deletion, and
  `_robust_rmtree` unlinks internal reparse points, so the actual deletion
  path is safe even if a child directory contains junctions.

The pre-existing `_materialize_directory` function (`worktree.py:2047+`) has
containment gaps (rglob can follow symlinks; `.resolve()` on target path).
These predate this PR and are out of scope for the bucket-A audit.

---

### #387 — janitor: route required-check failures to rework

**Verdict: CLEAN.** The routing logic is correct, loop prevention is sound, and
all invariants are respected.

- **Routing** (`workflow.py:10233`): When `verdict.is_check_failure_block` is
  True (failed required checks and no other janitor failures), the PR is routed
  to `request_changes` via `record_review`, producing a rework prompt naming
  the failing checks. This is correct — a PR whose CI settles on a red required
  check would otherwise dead-end in `janitor_blocked` (which has zero
  readers).
- **Loop prevention**: The initial review raised a concern about "no global
  cap on check-failure rework attempts." This was **verified as incorrect**.
  `record_review` (`workflow.py:13032`) caps all rework cycles via
  `max_rework_cycles`: when `request_changes_count >= max_rework_cycles`, the PR
  is escalated to a human instead of dispatching another cycle. This applies to
  ALL `request_changes` verdicts, including those routed from check failures.
  Additionally, `_check_no_op_rework` (`janitor.py:406`) prevents unchanged
  re-pushes from re-entering the cycle.
- **Rerun debounce** (`checks.py`): Each failing check run ID gets exactly one
  rerun attempt per head SHA. A new head SHA clears attempt markers. This gives
  first failures one retry chance before routing to rework.
- **Invariants**: `JanitorVerdict` uses `@dataclass(frozen=True)`. All label
  transitions use `config.labels.*`. State mutations use `state_lock` +
  `save_state`.
- **Tests**: Cover first-failure routing, second-failure (definitive) routing,
  and rerun API error fallback. Non-vacuous.

**Terminology note (LOW, documentation only)**: The PR title says "definitive
required-check failures" but the code routes ALL required-check failures (first
failure and definitive failure alike) to rework. The rerun debounce is a
separate layer that gives first failures one retry chance. The routing behavior
is correct; the title is imprecise. This is a documentation issue, not a code
bug.

---

### #386 — carry forward approved verdict on clean rebase via patch-id

**Verdict: CLEAN.** The carry-forward extension is correct and safe. Two
critical findings from the initial review were independently verified as **false
positives**.

- **Patch-id whitespace collision (CRITICAL → FALSE POSITIVE)**: The initial
  review claimed `git patch-id --stable` ignores whitespace, allowing two diffs
  with different Python indentation to produce the same patch-id. This was
  **empirically tested and refuted**:

  ```
  # Diff A: unindent return True (changes Python semantics)
  -        return True
  +    return True
  → patch-id: 3e3514963406ac6eade28f2b08f8407b9c4b502b

  # Diff B: change return True to return False (different content)
  -        return True
  +        return False
  → patch-id: a868cb9a81345f4eacf028cd10017c595af1ea7d
  ```

  `git patch-id --stable` hashes the `+`/`-` content lines **including
  whitespace**. It strips only commit metadata and hunk line-number offsets.
  Different indentation produces different patch-ids. The attack scenario
  described in the initial review is not possible.

- **Empty patch-id collision (HIGH → FALSE POSITIVE)**: The initial review
  claimed two empty patch-ids could match and carry forward a verdict. The code
  at `workflow.py:17165` guards with
  `if live_patch_id and live_patch_id == reviewed_patch_id` — the
  `if live_patch_id` check means empty/falsy patch-ids short-circuit and never
  match. Additionally, line 17160 returns early when `reviewed_patch_id` is
  empty. Empty-patch-id collision is not possible.

- **Stale CI check for approved verdicts (MEDIUM → by design)**: The
  carry-forward code has a stale-CI check specific to `request_changes`
  verdicts (`workflow.py:11022–11026`) that prevents carrying forward a
  request_changes verdict when the only findings cite required checks that are
  now green. This check is not applied to `approved` verdicts. This is by
  design: for `request_changes`, a verdict with only CI findings would block
  the PR even after CI recovers; for `approved`, the review verdict is just
  preserved, and the CI gate runs separately before merging. The carry-forward
  does not bypass the CI gate.

- **Invariants**: `_update_approval_head` uses `self._write_json` (atomic
  temp-file + replace). State updates use `state_lock`. `_calculate_patch_id`
  uses `run_captured` (returns `RunResult`, never raises).
- **Tests**: Cover basic approved carry-forward, request_changes carry-forward,
  blocked carry-forward, binary content safety, mixed binary/text. Non-vacuous.

---

### #364 — scope merged_pr_list() query and guard unconditional call

**Verdict: CLEAN.** No security issues, no invariant violations, comprehensive
test coverage.

- **Field scoping**: `MERGED_PR_LIST_FIELDS` contains exactly the fields needed
  by consumers (`number`, `title`, `body`, `headRefName`, `isCrossRepository`,
  `state`, `headRefOid`). Verified against all consumers:
  `linked_issue_number()` uses `headRefName`, `isCrossRepository`, `title`,
  `body`; `issue_numbers_mentioned_by_pr()` uses `title`, `body`; post-merge
  audit paths use `headRefOid`. All required fields are present.
- **Skip optimization**: `merged_pr_list()` is skipped entirely when there are
  no ready-labeled issues in the pass (`workflow.py:8276, 8291, 8481`). Applied
  in both dry-run and real dispatch paths.
- **Retry/backoff**: Uses the existing `_is_transient_gh_error()` function
  which includes 502/503/504 in its allowlist. Exponential backoff with jitter.
  Respects the "errors as values" invariant — raises `GitHubError` on terminal
  failures, which is the correct behavior for this codebase.
- **Security**: REST endpoint uses an f-string for the URL, but interpolated
  values are controlled (integer page numbers, no user input). No injection
  vector.
- **Tests**: Cover retry on transient 502, max retries exhaustion, non-transient
  errors not retried, skip when no ready issues, still queries when ready
  issues exist, REST pagination, field normalization. Non-vacuous.

**Minor documentation note (INFO)**: The PR title says "5 fields" but
`MERGED_PR_LIST_FIELDS` contains 7 fields. This is a title imprecision, not a
code issue.

## Summary

| PR | Verdict | Critical | High | Medium | Low | False positives refuted |
|----|---------|----------|------|--------|-----|------------------------|
| #578 | CLEAN | 0 | 0 | 0 | 1 | — |
| #466 | CLEAN | 0 | 0 | 0 | 1 | — |
| #387 | CLEAN | 0 | 0 | 0 | 1 | 1 (no global cap) |
| #386 | CLEAN | 0 | 0 | 0 | 0 | 2 (patch-id ws, empty patch-id) |
| #364 | CLEAN | 0 | 0 | 0 | 0 | — |

**No actionable security or correctness issues found.** All five bucket-A PRs
are sound. The code that was never reviewed by anything is safe.

Three LOW-severity observations (none requiring action):
1. #578: `assert` in production code (unreachable in practice)
2. #466: `worktrees_dir` itself is not checked for being a junction
   (defense-in-depth, config-derived path)
3. #387: PR title says "definitive" but code routes all check failures
   (documentation imprecision)

Two critical findings from the initial parallel review were independently
verified as false positives:
1. #386: `git patch-id --stable` does NOT ignore whitespace in `+`/`-` content
   lines (empirically tested)
2. #386: Empty patch-id collision is prevented by the `if live_patch_id` guard

## Buckets B, C, D — no action required

Per the issue body and the operator's narrowing comments:
- **Bucket B** (6 PRs, `request_changes` at merge time, merged by human): The
  operator's 2026-07-26 comment established that the carry-forward mechanism
  (`workflow.py:8741–8800`) carries the verdict forward itself before merging,
  so these are era artifacts that predate #412/#414 landing, not an ongoing
  leak.
- **Bucket C** (11 PRs, approved with head drift): Expected artifact of the
  carry-forward mechanism, not a gate failure.
- **Bucket D** (26 PRs, era artifacts): Predates the review gate or the
  `reviewed_head_sha` field. Benign.

The four late missing records in D (#418, #501, #505, #542) are noted in the
issue body as "worth a glance" but are outside the narrowed scope (bucket A
only).
