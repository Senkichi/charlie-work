# Attachment-Point Contracts — Round-1 Adversarial Review Findings

Reviewer: delegate (round 1). Date: 2026-08-24.
Scope: `src/charlie_work/attachment_contracts/`, `tests/attachment_contracts/`,
`.github/workflows/attachment-contracts.yml`, `.attachment-budgets.json`,
`docs/plans/attachment-contracts-backtest-report.md`, decision doc §1.1–1.4 / grafts
G1–G6, spec `docs/specs/attachment-point-contracts-spec.md`.

Method: full codepath trace + empirical repro. 86/86 package tests pass.

**Counts: 7 outstanding (correctness/spec/safety), 3 blocking.**
Stylistic preferences are excluded from the counts and marked NON-COUNTING below.

---

## PASSING VERIFICATION POINTS (no finding)

- **Item 1 — NO line-count metric anywhere.** Grepped `src/`, `tests/`, the CI yaml,
  `.attachment-budgets.json`, and the backtest report for `line_count|readlines|splitlines|
  n_lines|num_lines|max_lines|nloc|len(...split)`. Only hit outside source is
  `tests/attachment_contracts/test_hook_entry.py:121` — `.splitlines()` reading back a
  logged JSON audit file, not counting source lines. The gate metric is `member_count`
  (bound members per attachment point) throughout. The spec's "200–400 lines / 800 max"
  under Conventions is an authoring convention for this package, NOT a gate metric — no
  code reads it. **PASS.**
- **Item 2 — G5 (pinned-baseline KPI + churn overlay).** Not in cw-apc; correctly lives
  in llibrary `scripts/god_object_scan.py` per decision doc §4 row 9 (measurement harness
  routed there). Confirmed implemented: `pinned_paths` frozenset, `added_pinned`/
  `share_pinned` accretion, `share_top3` overlay (lines 325–386, 455, 555–583, 600–631,
  702–763). **PASS** (present, right location).
- **Item 3 — saturation math.** `outliers.saturate()` filters `non_ledger` before
  computing the distribution (ledgers are out-of-distribution, receive no verdict);
  nearest-rank quartiles `rank = max(1, min(n, math.ceil(quartile*n)))`; boundary
  `Q3 + 1.5*IQR`; strict `p.member_count > boundary`; `FLOOR = 4`. Hand-checked against
  `test_outliers.py` expected values. **PASS.**
- **Item 7 — behavioral tests.** `test_outliers.py`, `test_ledger.py`, `test_archetypes.py`
  assert hand-computed numeric expectations, not self-consistency. **PASS** (with the
  coverage gaps noted in Findings 1 and 3, whose missing tests are the reason those
  defects shipped).

---

## FINDING 1 — [HIGH / BLOCKING] Tamper guard misses raise-to-match laundering; G4 external-ack is fully bypassable

**Files:** `src/charlie_work/attachment_contracts/baseline.py` — `compare()` (≈183–267),
`check_tamper()` (≈270–323).

**Claim:** A worker can grow a saturated attachment point past its frozen ceiling and pass
CI clean — with no bump and no external ack — by hand-editing the matching `member_count`
in `.attachment-budgets.json` up to the new value. This defeats decision-doc §1.2 row 3
("CI fails on baseline tamper") and the entire G4 actor-split (worker may not self-ack).

**Mechanism (both guards blind to the same move):**
- `compare()` blocks only when `point.member_count > effective_ceiling`, where
  `effective_ceiling = max(entry.member_count, max(bump.to))`. Raising the entry's
  `member_count` raises the ceiling in lockstep → `135 > 135` is false → no finding.
- `check_tamper()` fires only when `entry.member_count > actual_count` (baseline claims
  MORE than reality — the "lower the bar" direction). The laundering move raises the
  baseline to *equal* reality, so `135 > 135` is false → no finding.
- `validate_bump()` correctly requires a non-empty ack, but is never reached because no
  bump is added — the attacker edits `member_count` directly, not `bumps`.

**Empirical proof:** Against the real module, growing `OrchestratorApp` 134→135 while
hand-editing its `member_count` to 135 (no bump entry) yields
`compare block findings: []`, `tamper findings: []` → **LAUNDERED CLEAN.**

**Compounding factor (prerequisite for the fix, not a separate finding):** `generated_at`
is stamped from `datetime.now()` (`__main__.py` baseline writer ≈line 106), so the baseline
file is NOT byte-reproducible end-to-end despite `baseline.py`'s module docstring implying a
deterministic artifact. A tamper check therefore cannot rely on a file-bytes git-diff — it
must compare the parsed `entries` array (identity → member_count/bumps), ignoring
`generated_at`.

**Fix guidance:** The tamper guard must detect *any* mutation of a frozen entry's
`member_count` that is not accompanied by a validly-acked bump — both directions, not just
downward. In CI, recompute the baseline from HEAD and diff the `entries` array against the
committed file: any `member_count` delta on an existing identity is a finding unless a new
`bumps[]` entry with a non-empty external ack (G4) justifies exactly that delta. Equivalently:
make the ceiling a function of `frozen_member_count + acked_bumps` only, and treat the
entry's own `member_count` field as immutable-once-frozen (raising it is itself the tamper
event). Add a regression test for the raise-to-match case — the existing
`test_tamper_detects_hand_raised_member_count` only exercises baseline(50) > actual(10),
which is why this shipped.

---

## FINDING 2 — [HIGH / BLOCKING] Uncaught decode/OS errors in scan crash the hook (breaks fail-open) and the CI report-only step (breaks shadow exit-0)

**Files:** `src/charlie_work/attachment_contracts/archetypes.py:220` (`scan_tree` read);
`src/charlie_work/attachment_contracts/hook_entry.py:133` (`main` → `check_file`, no guard);
`.github/workflows/attachment-contracts.yml:91–101` (report-only "always exit 0" contract).

**Claim:** `scan_tree` reads source with
`text = file_path.read_text(encoding="utf-8")` (line 220) OUTSIDE the try/except that wraps
`ast.parse` (221–224, catches only `SyntaxError`). A file that is not valid UTF-8, or is
unreadable (`OSError` — permission, race with an editor, dangling path), raises
`UnicodeDecodeError` (a `ValueError` subclass, NOT a `SyntaxError`) or `OSError`, which
propagates uncaught through `check_tree` → `check_file`.

This breaks TWO safety contracts simultaneously:
- **G6 / hook fail-open:** `hook_entry.main()` guards only `json.JSONDecodeError` around
  `json.load` (114–117). There is no try/except around the `check_file` call (133), so a
  decode/OS error becomes an unhandled traceback and a non-zero exit — the hook's
  advisory-only, never-block contract is violated, and in an unattended worker this is
  exactly the "hook must NEVER exit 2 / never crash into a blocking state" invariant.
- **CI shadow report-only:** the yaml step (100–101) relies on `check.py`/`__main__.py`
  "always exit 0 under `--report-only`" (documented 91–99). An exception raised inside the
  scan escapes *before* the exit-code logic runs, so the shadow-mode job can fail the build
  on an undecodable file — the opposite of shadow mode.

Note this is distinct from the null-byte case, which I verified is caught: `ast.parse('\x00')`
raises `SyntaxError` on modern CPython. The live gap is the *read*, not the parse.

**Fix guidance (single point of enforcement, matches the repo's own stance):**
1. At `archetypes.py:220`, wrap the read (not just the parse) so a decode/OS failure is
   recorded as a `parse_failure` for that file and the scan continues. This routes the
   failure into the existing G6 fail-toward-hard-stop-at-CI machinery instead of crashing.
2. In `hook_entry.main()`, add one top-level `except Exception: <log>; return 0` around the
   check call so the hook is fail-open against *any* unforeseen scan error, not only the
   two currently enumerated. Do not scatter per-call try/excepts.

---

## FINDING 3 — [HIGH / BLOCKING] G1 backtest reports PASS on a vacuous counterexample control and a structurally-dead Cluster-B probe (Deliverable-0 validity)

**Files:** `src/charlie_work/attachment_contracts/backtest.py` — `_criterion_counterexamples_clean`
(229–262, esp. `passed = not hits` at 251), `_cluster_b_informational` (265–293, counter at
283–285); `archetypes.py` `_module_ledger_points` (≈152–165); report
`docs/plans/attachment-contracts-backtest-report.md`.

The decision doc grafts G1 in specifically to prevent "0 flags" being read as evidence when
it is really an untested query (§1.5.1: "if the backtest shows the bare-function case is
missed, that is a pilot finding, not a silent hole"). The implementation reproduces the very
hole G1 exists to close.

**(a) Vacuous counterexample control — 10 of 13.** `passed = not hits` (251), and `hits`
only fills when a counterexample module *both* appears in the AP inventory *and* saturates.
Report line 12: "3/13 counterexample module(s) actually produced an AP (queried), 10 emitted
no AP in any sample." The docstring (232–238) is admirably honest that a module emitting no
AP is "an untested query, not evidence" — then the function returns `passed=True` regardless.
The gate's green is carried by 3 real negatives and 10 vacuous ones.

**(b) Cluster-B probe can never fire.** `_cluster_b_informational` counts only points with
`kind == "migration_runner"` and `identity.startswith("module:")` (283). A bare-function
non-ledger module produces NO point at all — `_module_ledger_points` returns `[]` when the
top-level functions are not a ledger — so it can never reach that counter. `cluster_b_score: 0`
(report line 16) is structurally guaranteed, not measured. (a) and (b) are blind to the same
class: the 10 unqueried counterexamples are largely bare-function modules — the exact case
the pilot claims to have checked.

**(c) "6-month" replay is ~2 month-buckets (scoping caveat, NON-BLOCKING sub-point).**
`select_samples` (125) takes `sorted(by_month)[-months:]`; the target repo's history has
only 2 distinct month keys (`git log --date=short` → earliest 2026-07-01, 2 unique months).
Report samples: 2026-07-01, 2026-08-01, plus three August anchors. The three anchors do
cover the known regrowth, so the positive-control criteria (orchestrator / test_charlie_work /
test_worktree saturated) are meaningful — but the PASS should be read as ~7 weeks + anchors,
not a 6-month control.

**Why blocking:** decision §3 makes the G1 backtest a hard gate for the go/expand decision
("the G1 backtest passed"). As implemented, PASS does not establish the property it gates on
(low false-positive rate on non-god-object modules, and detection of the bare-function case).

**Fix guidance:**
1. `_criterion_counterexamples_clean` must return **inconclusive/fail** when queried coverage
   is below a derived threshold (e.g. require a minimum fraction of the 13 counterexamples to
   actually emit an AP before "zero false positives" can count as a pass). A control that
   could not have failed is not a pass.
2. The Cluster-B probe needs a detector that can *see* non-ledger bare-function modules at all
   (emit a point, or a distinct "module scanned, no AP archetype matched" signal), otherwise
   its score is a constant and communicates nothing. Wire that signal so §1.5.1's promised
   "pilot finding" can actually surface.
3. Report the sample window honestly (available-months vs the 6-month intent).

---

## FINDING 4 — [MEDIUM] Hook scans stale on-disk state, ignoring the pending edit in tool_input

**File:** `src/charlie_work/attachment_contracts/hook_entry.py:133` (and the `check_file`
contract in `check.py`).

**Claim:** `main()` extracts `file_path` from the PreToolUse `tool_input` payload but then
calls `check_file` which reads the file's *current on-disk* content. In PreToolUse the edit
has not been applied yet, so the hook evaluates the pre-edit file — the growth that the edit
introduces is invisible until the write already happened. The hook advises on the wrong
revision.

**Evidence it is masked in tests:** `test_hook_entry.py::_over_budget_repo` pre-writes the
grown file to disk and *then* runs the hook, so the on-disk state already reflects the edit —
the stale-read path is never exercised.

**Fix guidance:** In PreToolUse, evaluate the *proposed* content from `tool_input`
(`content` / `new_string` applied to the target), not the on-disk file. If evaluating the
proposed edit is out of scope for the pilot, document explicitly that the hook is a
post-write advisory (and rely on CI as the real gate) so the "sub-second pre-edit feedback"
framing is not overclaimed.

---

## FINDING 5 — [MEDIUM] G3 backtest excludes are loaded but unwired (dead safety machinery)

**Files:** `src/charlie_work/attachment_contracts/excludes.py:54` (`is_codemod_commit`),
`:37`/`:72–82` (`blame_ignore_shas` / `_load_blame_ignore_revs`);
`src/charlie_work/attachment_contracts/backtest.py` (consumer).

**Claim:** `is_codemod_commit(changed_file_count)` and the resolved `blame_ignore_shas`
frozenset are computed by `load_excludes` but have **zero callers** (verified by grep across
the package). The backtest replay — the intended consumer per the excludes module docstring
("for backtest use … git-blame-ignore-revs / codemod-shape detection") — never consults
either. This is L3-unwired: the code exists and is tested at unit level but is not invoked by
any real consumer, so bulk-reformat / codemod commits are NOT actually excluded from the
backtest distribution, contrary to G3's intent.

**Fix guidance:** Wire `blame_ignore_shas` and `is_codemod_commit` into the backtest sample /
distribution path (skip or down-weight codemod-shaped commits and blame-ignored SHAs), or, if
G3 exclusion is deferred, delete the dead surface and note the deferral so it is not mistaken
for active protection.

---

## FINDING 6 — [MEDIUM] `check_file` runs a full-tree scan, contradicting the spec's sub-second single-file contract

**File:** `src/charlie_work/attachment_contracts/check.py:99–106` (`check_file`).

**Claim:** The spec describes the single-file / hook path as a "single-file scan
(sub-second)". `check_file` instead delegates to the full `check_tree`, scanning the entire
repo on every hook invocation. The parity rationale (a single file's saturation verdict
depends on the whole-distribution boundary, so you must scan the tree to classify one file)
is defensible and documented — but as written it is a real performance/spec deviation: the
hook is O(repo), not O(file), on every keystroke-adjacent edit.

**Fix guidance:** Either (a) cache the distribution/boundary per (kind) so a single-file
re-check reuses a recent tree scan and only re-derives the one point — restoring sub-second
behavior — or (b) amend the spec to state that single-file checks require a full tree scan
for boundary parity and drop the "sub-second" claim. Do not leave the spec and behavior in
conflict.

---

## FINDING 7 — [LOW-MED] Duplicate-identity entries collide in the baseline dict → lossy ratchet writeback

**Files:** `.attachment-budgets.json` (two identical `tests/test_worker.py::FakeGitHub`
`class` entries, lines 131–146; also two `FakeGitHub` classes in different files);
`src/charlie_work/attachment_contracts/baseline.py` (`compare()` / ratchet keying by
identity).

**Claim:** The baseline keys entries by identity when comparing/ratcheting, but the file can
contain two entries with the same `(kind, identity)` — visible directly:
`tests/test_worker.py::FakeGitHub` appears twice (lines 134–138 and 142–146). When
`compare()` builds an identity→entry map, one silently overwrites the other. Consequence is
worse than a missed verdict: `baseline --ratchet` writes back a document that **drops** one
of the duplicates — a lossy mutation of the frozen artifact, not merely a gating gap. Two
distinct real classes both named `FakeGitHub` (in `tests/_fakes_github.py` and
`tests/_reconcile_fixtures.py`) show the collision is reachable across files too if identity
is not file-qualified.

**Fix guidance:** Make the baseline key the full `(file, kind, identity)` tuple (file-qualified),
and reject duplicate keys at load time with a hard error rather than last-write-wins. Emit the
identity de-dup as a validation error so a duplicate can never silently erase a frozen entry
on ratchet.

---

## NON-COUNTING NOTES (context / minor, not in the 7)

- **`_run_git` encoding (`backtest.py:376–384`):** `subprocess.run(..., text=True)` with no
  explicit `encoding=` depends on ambient `PYTHONUTF8` (set in the CI yaml and the operator
  env, so covered *here*). Fragile as a library default; prefer explicit `encoding="utf-8"`.
- **Live `class` boundary = 5.0 saturating ~18 ordinary classes:** `.attachment-budgets.json`
  freezes many small legit classes (member_count 6–12) at/above a boundary of 5.0. Against
  §3's "false positives must stay ~0" Week-1 exit metric, this predicts a high false-positive
  rate on adoption — context for the pilot operator, not a code defect (the boundary is a
  derived Tukey fence over a small, skewed class population).
- **`backtest.py` uses `git worktree remove --force`, never `rm -rf`** — Windows
  junction-follow hazard correctly avoided (confirmed, positive note).
