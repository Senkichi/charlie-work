# APC pilot — adversarial review, round 1 (this file)

Reviewer: automated adversarial pass. Date: 2026-08-24.

**Numbering note.** The source already cites prior-round numbers in comments
(`finding #1`, `#3a/b/c`, `#4`, `#5`, `#7`) and the spec cites a "round-1 review
(finding #6)". Those are a *different, earlier* review. To avoid collision this
round numbers findings **independently, starting at #8**. When a code comment says
`finding #N` with N<=7 it refers to the earlier round, not this file.

Scope of the 7 requested checks and their verdicts are in the PASS section at the
bottom. Outstanding count = correctness / spec-violation / safety only; stylistic
notes are listed separately and are NOT counted.

---

## Outstanding findings

### #8 — [HIGH · BLOCKING] Deliverable-0 gate is in FAIL, and `counterexamples_clean` is structurally near-unsatisfiable for the module set it names

Evidence:
- `docs/plans/attachment-contracts-backtest-report.md:3` — **`Overall: FAIL`**.
- Same report line 14: `counterexamples_clean` FAIL — "only 3/13 counterexample
  module(s) actually produced an AP ... below the 50% coverage floor", listing
  10 modules that "emitted no AP in any sample": `event_kinds.py, fleet_paths.py,
  git_pull_blockers.py, logging_setup.py, markdown_fence.py, prompt_sections.py,
  rescue.py, safe_path.py, safe_ref.py, throttle_signatures.py`.
- Decision doc §1.1 (G1 / Deliverable 0): "The rest of the pilot is gated on
  Deliverable 0 passing."

Why it is structural, not transient: `archetypes.scan_tree` emits an
`AttachmentPoint` only for class / Typer-app / blueprint / ledger archetypes. The
10 modules above are bare-function modules (no class, no app object), so they
*cannot* produce an AP by construction. `_criterion_counterexamples_clean`
(`backtest.py`, `_COUNTEREXAMPLE_MIN_COVERAGE = 0.5`) requires >=50% of the 13
named counterexamples to have produced an AP before "zero saturations" is allowed
to count as a pass; the ceiling is 3/13 = 23%, so the criterion is pinned at FAIL
regardless of the actual (correct) zero-false-positive result. The positive
control therefore validates *nothing* about the archetypes it names — and in
particular cannot detect class-level over-firing (see #9), because none of its
counterexamples is a legitimately multi-method class.

Impact: the pilot's own hard gate is unmet, yet the committed baseline
(`.attachment-budgets.json`, generated 2026-08-25) and the CI workflow are already
in the tree. Either the gate passed under different criteria than what is committed,
or the pilot advanced past a FAIL.

Fix: replace the bare-function modules in `COUNTEREXAMPLE_MODULES` with
counterexamples that actually mint an AP under the current archetypes (small
real classes / a small Typer app that must remain unsaturated), OR redefine the
criterion so "module scanned, produced no AP, therefore trivially not a false
positive" counts toward coverage instead of being excluded from it. Do not ship
the baseline/CI until Deliverable 0 is green under the committed criteria.

### #9 — [MEDIUM] `class` outlier boundary is corrupted by zero-member classes; 6-method fixtures are frozen as "god objects"

Evidence (live scan, `python -m charlie_work.attachment_contracts scan --root .`):
- `class`: n=531, Q1=0, Q3=2, IQR=2, **boundary=5.0**, and **148 of 531 class APs
  have member_count == 0** (`class Foo: pass`, Protocols, empty dataclasses).
- `.attachment-budgets.json` consequently freezes as saturated: `WorkerView` (6),
  `WriteGate` (6), `TestRequireValidRev` (6), `TestResolveCliBinary` (6), two
  6-member `FakeGitHub` test fixtures, `_SalvageTestGitHub` (7), etc.

Root cause: the 148 zero-member classes anchor Q1 at 0, so IQR collapses to 2 and
the Tukey fence lands at Q3 + 1.5·IQR = 5. A 6-method class is not a god object;
it is being flagged because ~28% of the population is empty-bodied. This is a
structural defect, not a Week-1 tuning knob — no boundary tuning removes the
downward pull of the empty classes. Related: nothing guards `IQR == 0`
(`outliers.py`), where the fence degenerates to strict `> Q3` with zero tolerance;
`FLOOR` only guards `n < 4`.

Impact: an inflated baseline full of false-positive freezes, and — once Week-2
enforce lands — false blocks on any 5→6-method class growth. The G1 control (see
#8) structurally cannot catch this because it names no multi-method class.

Fix: require `member_count >= 1` to enter the outlier population for the `class`
kind (an AP with no bound members is not evidence about the god-object
distribution), and add an explicit `IQR == 0` degenerate-fence guard.

### #10 — [MEDIUM · SAFETY] G4 actor-split is self-attested and forgeable

Evidence: `baseline.py:157-173` (`validate_bump`):
```
if bump.actor == "worker" and not bump.ack.strip():
    return "G4: worker bump requires a non-empty external ack ..."
if bump.actor not in ("interactive", "worker"):
    return ...
```
- `actor` is a plain field read from the baseline JSON (`_bump_from_dict`,
  `baseline.py:34-40`: `actor=str(raw["actor"])`). Nothing binds it to the real
  execution context (no `CHARLIE_FLEET_WORKER` / `CLAUDE_CODE_UNATTENDED` check at
  validate time).
- No code path anywhere *generates* a bump — `grep` shows `Bump(...)` is
  constructed only in `_bump_from_dict` (deserialization) and tests. So the actor
  field is always hand-typed into the committed JSON.
- `ack` is validated for non-emptiness only. `ack: "x"` passes; there is no
  format/reference validation despite the message demanding an "issue URL /
  dispatch-prompt id / human handle".

Impact: the exact laundering G4 claims to close (decision doc §1.1 G4: "a worker
that can write code can trivially write its own token") is still open. A worker
authoring a bump sets `actor: "interactive"` (ack then not required) or supplies a
junk ack, and the gate passes. The only real control is out-of-band PR review of
the baseline diff — which is a human process, not the enforced-invariant G4 was
sold as.

Fix: G4 cannot be enforced from a self-declared field. Either (a) drop the actor
distinction and require *every* bump to carry an externally-verifiable ack token
that is format-checked (URL/issue-id regex), or (b) sign/attest bumps out of band
(e.g. a required approver in CODEOWNERS on the baseline file) and stop claiming the
JSON field enforces it. At minimum, validate ack shape rather than non-emptiness.

### #11 — [MEDIUM] `compare()` silently drops the baseline `mode` key → a routine `--ratchet` disables the hook's interactive-enforce block

Evidence:
- `baseline.py:273-279` — `compare()` builds `ratcheted` from a fixed key set
  (`version`, `generated_by`, `generated_at`, `floor`, `entries`); a `mode` key
  present in the input document is not carried forward.
- `__main__.py:103-105` — `_cmd_baseline --ratchet` does
  `document = load(...)` → `compare(...)` → `dump(ratcheted, baseline_path)`,
  writing the stripped document back over the file.
- `hook_entry.py:60-69` — `_resolve_mode` reads `document.get("mode", "advise")`;
  absence ⇒ `"advise"`.

Impact: if an operator sets `mode: enforce` in the committed baseline to arm the
PreToolUse interactive block, the next `baseline --ratchet` (the encouraged
happy-path maintenance action) silently reverts it to `advise`, with no finding and
a diff that reads as a normal ratchet. A safety control turns itself off during
routine maintenance. (CI enforcement is controlled by deleting `--report-only`
from the workflow yaml, per `__main__.py:8-9`, so CI is unaffected — the blast
radius is the hook-side interactive block only. Still a silent downgrade of an
enforcement surface.)

Fix: preserve unknown top-level keys in `compare()` —
`ratcheted = {**baseline_document, "entries": [...]}` — or make `mode` env-only
and delete the baseline-key path from `_resolve_mode` and the spec so the two
enforcement surfaces don't disagree.

### #12 — [MEDIUM] `loads()` lets `KeyError`/`ValueError` escape as a non-`TamperError`, breaking the Week-1 "step can never fail the job" guarantee for exactly the tamper vector it targets

Evidence:
- `baseline.py:54-65` / `34-40` — `_entry_from_dict` / `_bump_from_dict` extract
  fields with bare `str(raw["kind"])`, `int(raw["member_count"])`,
  `float(raw["boundary"])`, `int(raw["to"])`, `str(raw["actor"])`. A missing key
  raises `KeyError`; a non-numeric `member_count`/`boundary` raises `ValueError`.
  `loads()` guards *structural* shape (wrong version, non-list entries, non-dict
  entry, duplicate key, non-list bumps → `TamperError`) but NOT field extraction.
- `check.py:88-101` — `check_tree` wraps `baseline_mod.load(...)` in
  `except baseline_mod.TamperError` only. A `KeyError`/`ValueError` is not caught,
  so it propagates out of `check_tree`.
- `__main__.py:159` vs `164-165` — `_cmd_check_tree` calls `check_tree(...)` at
  159 and only returns 0 for `--report-only` at 164. The crash happens *before*
  the report-only short-circuit.
- `attachment-contracts.yml` comment claims `--report-only` "can never fail the
  job". That is false for a hand-tampered baseline with a missing/non-numeric
  field — the workflow step dies with an uncaught traceback and nonzero exit,
  bypassing the annotation/Finding pipeline entirely, for precisely the tamper the
  guard exists to catch. Same unguarded escape in `_load_previous_baseline_document`
  (`__main__.py:148-151` catches only `TamperError`).

Note: at the PreToolUse hook this is fail-open (`hook_entry.main`'s
`except Exception: return 0`), so hook safety is preserved; the impact is CI-only.

Fix: wrap the field extraction in `_entry_from_dict` / `_bump_from_dict` in
`try/except (KeyError, ValueError, TypeError)` and re-raise as `TamperError`, so
every malformed baseline surfaces as a structured Finding rather than a crash and
honors the report-only contract.

---

## Non-outstanding notes (NOT counted — stylistic / robustness / informational)

- **Interactive enforce hook exits 2 on an AST parse failure of the edited file.**
  `check_file` → `_parse_failure_finding` severity `"error"` → exit 2 in enforce.
  This is *stronger* than G6's "parse failures must never silently pass" (decision
  doc §1.1 G6: fail toward the CI hard-stop), not a divergence — blocking at the
  hook is a superset of fail-closed. UX cost only: transiently-unparseable
  intermediate edits get blocked for interactive humans (workers are advisory).
  Note, not a defect.
- **Test quality is mixed, not uniformly behavioral.** `test_outliers.py` and
  `test_baseline.py` hand-compute expected values (genuinely behavioral). But
  `test_check.py::_freeze_baseline` and `test_hook_entry.py::_freeze_baseline_at`
  derive the baseline by running `scan_tree`→`saturate_all`→`generate` — the code
  under test — so their "clean when file matches baseline" assertions are
  self-consistency for the generation half; the *growth* assertions layered on top
  are behavioral. `test_check.py` states an expected `boundary == 2` in a comment
  (lines ~43-47) but nothing asserts it.
- **FLOOR=4 docstring justification is mathematically loose** (n=2,3 also yield
  distinct order statistics); the value/behavior is fine, only the rationale text
  overstates the case.
- **APC hook is not wired in `cw-apc/.claude/settings.json`** (only
  merge_preflight / git_push_lint / worker_stop_gate are). This is by design — the
  PreToolUse hook is operator-gated in `~/.claude` — but it means the interactive
  block path is not exercised end-to-end inside this repo.
- **`_STRUCTURAL_DIR_SUFFIXES` is a misnomer** — matched by exact directory-name
  membership, not suffix. Behavior matches the spec ("any dir named generated /
  vendor"); the name misleads.
- **`run_backtest` temp cleanup** relies on `TemporaryDirectory` → `shutil.rmtree`
  after per-sample `git worktree remove --force`; if a remove ever failed, the
  rmtree would run over the leftover worktree. Theoretical junction-follow risk
  only — backtest worktrees carry no `.venv` junction — but worth a guard if the
  fixture ever grows one.

---

## Verified PASSES (the 7 requested checks)

1. **No line-count metric anywhere.** `grep` across the package finds no LOC/line
   metric feeding a decision; the only line-ish reads are prose and
   `changed_file_count` (codemod-shape heuristic, a file count, not lines).
   Saturation is purely member-count per AP.
2. **Grafts G1-G6 implemented, not just named** — G1 backtest exists (but see #8);
   G2 redirect scaffold present; G3 exclude-set + codemod skip wired
   (`select_samples` + `Excludes`); G4 present but forgeable (#10);
   G5 pinned-baseline compare/ratchet present; G6 AST-parse-failure →
   `Finding(error)` at both hook (advisory) and CI (blocking), verified via
   `archetypes.scan_tree` routing SyntaxError/UnicodeDecodeError/OSError to
   `parse_failures`. **G4 self-ack:** yes, a worker CAN self-ack (#10).
   **G6 parse failure:** advisory at hook (exit 0 with G6 context), blocking at CI
   (`error` → nonzero), which is the intended fail-toward-CI asymmetry.
3. **Saturation math correct.** Nearest-rank quartiles with ceil convention
   (`_nearest_rank_quartile`: `rank = max(1, min(n, ceil(q*n)))`); strict `>`
   (`outliers.py:89`, `saturated = p.member_count > boundary`); FLOOR=4 honored
   (`population < FLOOR` → not saturated); ledgers excluded from the population
   (`outliers.py:65`, `is_linear_ledger` filter). Boundary values in the committed
   baseline reproduce from the live scan (class 5.0, test_module 41.0).
4. **Baseline determinism + tamper guards work.** `dumps` sorts keys and entries,
   fixed indent, trailing newline (deterministic). `check_tamper`
   (`baseline.py:335-388`) detects a single-snapshot baseline>actual with no
   covering bump; `check_ratchet_tamper` (`283-332`) detects a member_count rise
   vs the previous committed baseline (the raise-to-match laundering vector). Both
   verified to fire on a hand-raised entry. `loads()` rejects duplicate
   `(kind,file,identity)` and wrong version. (Escape gap for missing/non-numeric
   fields is #12.)
5. **Hook safety.** No baseline found upward → `return 0` (no-op outside piloted
   repos). Unattended (`CHARLIE_FLEET_WORKER=1` / `CLAUDE_CODE_UNATTENDED=1`) can
   NEVER reach `return 2` — the unattended/non-enforce branch short-circuits to
   `return 0` before the interactive-enforce exit. Malformed stdin → `return 0`.
   `except Exception: return 0` (fail-open). Fail-open at hook, fail-closed at CI
   confirmed (G6).
6. **Windows correctness.** Subprocess calls are list-form with explicit `cwd=`
   (no shell, no POSIX-only assumptions); AP identities use `PurePosixPath` /
   `as_posix()` for stable cross-platform keys; worktrees removed via
   `git worktree remove --force` (never `rm -rf`, honoring the junction hazard).
7. **Tests** — see the non-outstanding note on test quality; core numeric logic is
   hand-computed and behavioral, hook/CI parity is exercised, with the two
   self-consistency caveats noted.

---

### Counts
- Outstanding (correctness / spec / safety): **5** (#8-#12)
- Blocking: **1** (#8 — pilot's own Deliverable-0 gate is FAIL)
