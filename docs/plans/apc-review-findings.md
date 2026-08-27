# Attachment-Point Contracts — Adversarial Review Findings (Round 3)

**Reviewer:** delegate (adversarial, round 3)
**Date:** 2026-08-24
**Scope:** `src/charlie_work/attachment_contracts/`, `tests/attachment_contracts/`,
`.attachment-budgets.json`, `.github/workflows/attachment-contracts.yml`,
`docs/specs/attachment-point-contracts-spec.md`,
`docs/plans/attachment-contracts-backtest-report.md`,
decision doc `llibrary/docs/plans/2026-08-24-god-object-mitigation-DECISION.md` §1.1–1.4.

## Verdict

- **Outstanding (correctness / spec-violation / safety):** 1 — #14
- **Blocking (gates the Week-2 enforce flip):** 0
- **Verified fixed since round 2:** #9 (core regression), #10, #13
- Week-1 shadow (`--report-only`) is **not** blocked by any finding — the CI job cannot
  fail in report-only mode. #14 is a latent Week-2 false-positive risk, not a Week-1 gate.
- **140 tests pass** under Windows/uv (`uv run python -m pytest tests/attachment_contracts`).

---

## OUTSTANDING

### #14 — MEDIUM — Structural-triviality detector's test-double arm is prefix-only; test-support fixtures leak into the `class` population and get frozen as false positives

**Files:** `src/charlie_work/attachment_contracts/archetypes.py:33` (`_TEST_DOUBLE_NAME_RE`),
`:224-244` (`_is_test_double_name` / `_is_structurally_trivial`); committed
`.attachment-budgets.json` (frozen entry `_SalvageTestGitHub`).

**Status — this is the narrowed residual of round-2 #9.** The core round-2 regression is
**fixed**: the class fence rose from the regressed 3.5 back to **6.0**, and 465
structurally-trivial classes (Protocols, Exception subclasses, empty `@dataclass` shells,
function-nested fixtures, `Fake*`/`Test*`-prefixed doubles) are now excluded from the
saturation population. Independently recomputed from a live scan, the class fence and the
frozen set match the committed baseline exactly:

- class pop n=55, q1=1.0, q3=3.0, iqr=2.0, **boundary=6.0**; frozen = `GitHub`(53),
  `OrchestratorApp`(134), `WedgeWatchdog`(9), `_PrStateWriteVisitor`(10),
  `_SalvageTestGitHub`(7). The round-2 4-member test doubles (`FakeClock`, `FakeApp`, …)
  are gone (now structurally trivial).

**The residual defect:** `_TEST_DOUBLE_NAME_RE = ^_?(Fake|Test)[A-Za-z0-9_]*$` anchors the
double marker at the **start** of the name only. This repo's actual test doubles use
**infix / compound** names, which the regex misses:

- `_SalvageTestGitHub` (7 members, `tests/_salvage_fixtures.py`) — a fake GitHub client
  (instantiated with `repo_root=`, `closing_issue_numbers=`, `pr_view_raises=` in
  `tests/test_closing_reference.py`). It is **frozen as a saturated `class` right now**.
  At the Week-2 enforce flip, adding an 8th method to this fake hard-fails CI.
- `_NoOpGitHub` (5 members, `tests/test_worker.py`) — a test double, exactly **one method
  below the fence**. One added method makes it a new saturated AP (blocked via the #13
  new-AP path at Week-2).
- `CachingFakeGitHub` (3), `_RecordingFakeRun` (2) — fakes currently well under the fence
  but sitting in the population as if they were real service classes.

So 2 of the 5 frozen `class` entries are test-side, one (`_SalvageTestGitHub`) is an
unambiguous false positive, and the exposure grows as the test suite grows. (The other
test-side entry, `_PrStateWriteVisitor`, an `ast.NodeVisitor` with `visit_*` methods, is a
genuine multi-method class — freezing it is defensible, not counted.)

**Why prefix-matching is the wrong layer:** it is exactly the brittle name-list pattern
CLAUDE.md and the spec warn against — every new double-naming convention
(`Mock*`/`Stub*`/`Spy*`/`Dummy*`/`NoOp*`/`Recording*`, or any infix `*Fake*`) needs the
regex widened by hand. Widening the regex would chase the symptom.

**Fix (single point of enforcement, structural, no name list):** exclude from the `class`
saturation population any `class` AP whose `file` is under `tests/` and that is not the
`Test*` method-holder the `test_module` archetype already counts — i.e. test-support /
fixture classes are not production god-object risk and belong out of the distribution the
same way ledgers and Protocols are. The scan already walks `src/` and `tests/` separately
(`iter_source_files`), so the `tests/`-prefix split is derived from the tree layout, not a
hand-maintained list. This removes `_SalvageTestGitHub`, `_NoOpGitHub`, `CachingFakeGitHub`,
and `_RecordingFakeRun` from the population in one structural rule, and the test-side
saturation signal remains covered by the `test_module` archetype. Then re-freeze the
baseline and confirm the frozen `class` set is production-only (`GitHub`, `OrchestratorApp`,
`WedgeWatchdog`). Add a regression test: a multi-method fixture class under `tests/` with a
non-`Fake`/`Test` prefix is not saturated.

**Not blocking Week-1** (report-only cannot fail) and not a mis-freeze of any *production*
class; recorded as MEDIUM because it is a real false positive today and a growing one, and
the decision doc's go/expand gate requires an acceptable false-positive rate.

---

## VERIFIED FIXED (round-2 findings, re-checked this pass)

### #9 — FIXED (core regression) — class fence no longer pulled onto legitimate classes

Round 2's blocking defect was the member-count filter compressing the fence to 3.5 and
freezing many 4-member test doubles. The fix added `AttachmentPoint.is_structurally_trivial`
(`model.py:40`), computed structurally in `archetypes.py:_is_structurally_trivial`
(`:228-244`) — Protocol bases, `Exception` subclasses, empty `@dataclass` shells,
**function-nested classes** (`_iter_classdefs` tracks `nested_in_function`, `:144-178`,
catching inline test doubles a name check misses), and `Fake*`/`Test*`-prefixed names —
and excludes them from the population in `outliers.py:82-86`, the same way ledgers are.
Verified independently: fence is now **6.0** (above the original round-1 5.0), 465 classes
excluded as trivial, and the round-2 4-member doubles no longer appear. Narrowed residual
tracked as **#14** above.

### #10 — FIXED — G4 actor-split no longer forgeable

`baseline.py:validate_bump` (`:181-216`) now requires a shape-checked, non-empty `ack` on
**every** bump regardless of `actor` (`:205-215`), with `_ACK_SHAPE` (`:178`) accepting only
an http(s) URL, a `#123` / `owner/repo#123` issue ref, or a `source:id` handle. The round-2
discriminating vector — `Bump(actor="interactive", ack="")` from a worker that mislabels
its own actor — is now rejected (empty ack fails for both actors). The docstring correctly
records that binding `actor` to real execution context is out of scope for a
comparison-only validator (backstopped by CODEOWNERS on the baseline diff). Gap closed.

### #13 — FIXED — new already-saturated AP now blocked when a baseline exists

`baseline.py:compare()` `baseline_entry is None` branch (`:276-305`) now emits a
`Finding(block)` for a currently-saturated point with no baseline entry, and still snapshots
it into the ratcheted document. The docstring (`:253-264`) correctly establishes that
`compare()` is only reached once a baseline file exists (both call sites guard on it; true
freeze-on-adopt is handled by `generate()`), so a new saturated AP here is a new god-object,
not adoption. Two regression tests present:
`test_baseline.py:339 test_compare_new_saturated_point_with_no_baseline_entry_blocks` and
`test_check.py:153 test_check_tree_new_saturated_ap_with_existing_baseline_blocks`.

---

## SEVEN REQUIRED CHECKS

| # | Check | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | No line-count metric anywhere | **PASS** | Grep over `src/`, `tests/`, CI yaml: no member metric derives from lines. The two `.splitlines()` hits are `excludes.py:77` (parsing `.git-blame-ignore-revs`) and `backtest.py:475` (parsing `git log` output); `changed_file_count` is a **file** count for G3 codemod detection, not lines. Member count = `len(members)` from AST binding (`model.py:43`). |
| 2 | Grafts G1–G6 implemented | **PASS** | G1 backtest positive control (`backtest.py`, report Overall PASS, orchestrator+test_charlie_work saturated at all 4 samples); G2 scaffold redirect (`redirect.py`); G3 exclude-set (`excludes.py`, one sanctioned `exclude_globs` + structural dirs + codemod-shape); G4 ack-on-every-bump (`baseline.py:181-216`, #10 closed); G5 pinned-set KPI/churn overlay (informational, present); G6 parse-failure → `error` Finding, never dropped: **fail-open at hook** (blanket `except Exception: return 0`, `hook_entry.py:210-214`) yet **fail-toward-stop at CI** and interactively in enforce (empirically: `check_tree` emits an error Finding → `__main__` exit 1). |
| 3 | Saturation math | **PASS** | Independently recomputed from a live scan: nearest-rank ceil quartiles, class fence q3=3.0 iqr=2.0 **boundary=6.0**, test_module **41.0** — both match the committed baseline exactly. Strict `>` (exact-tie-not-saturated, `test_outliers.py:63-76`); FLOOR=4 honored; ledgers and structurally-trivial points excluded from the population and given no verdict. |
| 4 | Baseline determinism + tamper guard | **PASS** | Deterministic serialization (sorted entries, indent=1, sort_keys). Two-layer tamper guard: single-snapshot `check_tamper` (hand-raised `member_count=999` detected, `test_check.py:180`) and diff-based `check_ratchet_tamper` (raise-to-match). Empirically confirmed: a `53→60` hand-raise vs the previous baseline yields an `error` Finding. |
| 5 | Hook safety | **PASS** | Empirically: malformed stdin → 0; unattended (`CHARLIE_FLEET_WORKER=1` **or** `CLAUDE_CODE_UNATTENDED=1`) + `mode=enforce` + real block finding → **0** (never exit 2); interactive + enforce + block → 2; interactive + advise → 0; no baseline above target → 0; `except Exception → 0`. `_resolve_mode` env override cannot defeat the unattended guard (`if unattended or mode != "enforce"`). |
| 6 | Windows correctness | **PASS** | Repo-relative POSIX identities via `as_posix()`; `target.resolve().relative_to(root)` with a `ValueError → return 0` fallback for cross-drive/outside-root; subprocess calls pass arg **lists** (no `shell=True`, no POSIX-only assumptions); worktrees torn down via `git worktree remove --force`, never `rm -rf`. 140 tests pass under Windows/uv. |
| 7 | Tests behavioral, not self-consistency | **PASS** | `test_outliers.py` hand-computes quartiles/IQR/boundary and the strict-`>` tie case. `test_baseline.py`/`test_check.py` assert concrete severities/counts for growth, tamper, ratchet, G4 ack, #13 new-AP, and G6 parse-failure. `test_check.py` uses code-under-test only to *freeze* the baseline fixture; the load-bearing assertions (block/error/redirect) are behavioral. |

---

## NON-COUNTED NOTES (design tradeoffs / defense-in-depth — not outstanding)

- **Ledger-shaped method names exempt a class from saturation.** A `class` whose methods
  form a dominant `<prefix><int>` contiguous sequence (≥3 members, ≥80% dominance, gaps ≤2)
  is reclassified `migration_runner`/`is_linear_ledger` and excluded — observed while
  fuzzing a 200-method `m0..m199` payload. This is the spec-sanctioned structural ledger
  exemption (`ledger.py`), and a real service class with numerically-suffixed methods is
  rare and a self-defeating evasion (unusable API). Design tradeoff, not a defect.
- `backtest.py:_worktree_remove` shells `git worktree remove --force` via `_run_git`
  (`check=True`) inside `run_backtest`'s `finally`; if teardown fails it masks the original
  scan exception. Offline tooling only; prefer swallowing in the cleanup path. (Carried from
  round 2, still non-counted.)
- A linear-ledger AP that loses its `<prefix><int>` shape flips `kind` and silently enters
  the class distribution; low likelihood, worth a comment near `ledger.py`. (Carried.)
