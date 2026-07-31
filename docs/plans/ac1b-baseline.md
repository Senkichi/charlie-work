# AC-1b findings-actionability baseline

Measured output of `scripts/ac1b_findings_actionability.py` against the real
`.var/charlie-work/prs` corpus, run through the REAL production renderer
(`charlie_work.workflow._render_required_changes_section`) at the pinned
pre-F1 baseline commit. This is durable evidence, not a plan — see
`docs/plans/rework-findings-channel.md` (untracked, main checkout) for the
fix design this measures against.

Re-run the same script, unmodified, after F1/F5 land to get the "after"
numbers:

```
VIRTUAL_ENV= PYTHONPATH="$PWD/src" uv run --no-sync python \
    scripts/ac1b_findings_actionability.py --repo <path-to-checkout>
```

## Pinned baseline

- **Code SHA:** `b3b827de7cc157d60992b138d3d789f08d8f1d22`
  (`test(worktree): close COMPLETED-misclassification gap left by PR #695's
  regression test (#764)`) — `origin/main` at the time this measurement was
  taken, pinned via a detached-HEAD worktree
  (`.claude/worktrees/ac1b-baseline`) so PR #766 (F1) merging mid-measurement
  could not change the "before" numbers.
- **Corpus:** `C:\Users\senki\repos\charlie-work\.var\charlie-work\prs`
  (main checkout; gitignored runtime state, not part of any commit).
- **Corpus size found:** 20 `request_changes` verdicts — matches the plan's
  claimed 20.

## Results by category (never a single aggregate)

| Category | Count | AC-1 (non-empty) | AC-1b (actionable) | Projected post-F1 AC-1b* |
|---|---:|---:|---:|---:|
| Cross-family generic collapse | 7 | 0 | 0 | 0 |
| Synthetic CI-failure | 3 | 0 | 0 | 0 |
| Real reviewer prose | 10 | 0 | 0 | 10 |
| **TOTAL** | **20** | **0** | **0** | **10** |

\* Diagnostic only — a local stand-in for F1's contract (fence the `summary`
as fallback body when `required_changes` is empty), **not** the real F1
code, since F1 had not merged at the pinned SHA above. Re-run the script
against the real post-F1 checkout for the authoritative number.

Baseline AC-1 = 0/20 confirms the plan's §2.2 claim exactly. Baseline AC-1b
= 0/20 follows trivially from AC-1 = 0/20 (nothing renders, so nothing can
be actionable) — this baseline table alone does not exercise the
actionability *definition*; the mutation checks below do that.

## Self-test (detector controls, run before the main table)

| Input | Referents found | Actionable | Expected |
|---|---|---|---|
| `"See src/charlie_work/workflow.py:3678 in \`_render_required_changes_section\`."` | file_path, code_symbol, line_number (3 kinds) | True | True |
| `"Cross-family review found BLOCKER/MAJOR findings"` (literal collapse sentinel) | none | False | False |

**PASSED.** The detector can produce both True and False — the all-zero
baseline table above is not an artifact of a broken detector.

## Mutation checks (prove the harness is sensitive, using the real renderer)

1. **Monkeypatch `summary` only**, real unmodified renderer: rendered output
   is `''` before and after (**no movement**). This is the *expected* result
   on this pre-F1 baseline — `_render_required_changes_section` (workflow.py
   :3678) early-returns `""` whenever `required_changes` is empty, without
   ever consulting `summary`. This null result is itself confirmation of the
   defect in plan §2.1, not evidence the harness is broken.
2. **Mutate `required_changes`** (the field the pre-F1 renderer *does*
   read) to `["Fix the null check in src/charlie_work/workflow.py:3700"]`,
   real unmodified renderer: rendered section is non-empty and scores
   actionable (`file_path`, `line_number` both found). **PASSED** — proves
   the detector + real-renderer pipeline can and does move when the input
   changes.

## Findings that contradict `docs/plans/rework-findings-channel.md` §2.3 / §8

**1. Category counts: measured 7 / 3 / 10, not the plan's claimed 8 / 3 / 9.**

§2.3's table claims `8` cross-family-generic-collapse verdicts (40% of the
corpus) and `9` real-reviewer-prose verdicts. Exact-match classification
against the real `cross_family.parse_cross_family_verdict`-derived sentinel
(`"Cross-family review found BLOCKER/MAJOR findings"`, 48 chars) found only
**7**: pr-690, pr-692, pr-695, pr-696, pr-699, pr-700, pr-724. §2.3's own
citation list names only 6 of these (pr-690, 692, 695, 699, 700, 724) and
omits pr-696 — so even the plan's own supporting citation undercounts its
stated total of 8.

Two verdicts are plausible candidates for why the plan's total came out
higher: pr-693 (309-char summary) and pr-698 (461-char summary) both carry
a cross-family review artifact on disk and both sit comfortably under the
parser's 500-char truncation — i.e. they look like cases where cross-family
review ran and its `_VERDICT_RE` *did* match (a real parse, not a collapse),
producing genuine reviewer prose rather than the fallback constant. If the
plan's "8" was counting *cross-family-sourced* verdicts rather than
strictly *cross-family-collapsed-to-the-constant* verdicts, that would
explain the discrepancy — but this is inference from the data available,
not confirmed provenance (`review-decision.json` does not record which
reviewer produced a given verdict). Presence of a `cross-family-review.md`
artifact file is not reliable provenance either: pr-529 has one but its
recorded summary is the synthetic CI-failure string, not cross-family
content.

Total still sums to 20 either way; only the internal 3-way split moves.

**2. The synthetic-CI-failure category does not yield an actionable brief
under AC-1b's own definition — contradicting §2.3's "Yes" and §8's ~12/20
projection.**

§2.3's table states synthetic CI-failure verdicts satisfy "Does F1's
fallback yield an actionable brief? **Yes**", and §8 projects "~12/20 after
F1 alone (the 9 prose + 3 CI verdicts)". Measured: the synthetic CI-failure
summary text is `"CI failed on <check-name>; push a fix"` (e.g. `"CI failed
on Lint; push a fix"`, all 3 on-disk instances) — this contains **zero**
concrete referents under AC-1b's own stated definition (file path, symbol
name, or line reference; `docs/plans/rework-findings-channel.md` §8). It
names a check category, not a file, symbol, or line.

The projected-post-F1 column above shows this precisely: cross-family = 0/7
(expected, unchanged), synthetic-CI = **0/3** (contradicts §2.3's "Yes"),
real-prose = 10/10. Projected total is **10/20**, not the ~12/20 the plan
states. This is not a stricter reading on this harness's part — §8 supplies
the exact same three-referent-kind definition this script implements; the
plan's own §2.3 table and §8 projection are inconsistent with the plan's
own §8 definition. Whether the resolution is to loosen AC-1b's definition
(e.g. treat a named CI check as sufficient) or to accept that F1 alone
leaves the synthetic-CI-failure category unactionable (matching the
cross-family category) is a decision for a human, not this measurement —
this harness deliberately does not loosen the detector to make the
category pass.

## Detector definition (for auditability)

- **File path:** `dir/dir/file.ext` shape, or a bare `file.ext` filename
  carrying one of a fixed set of known source/config extensions (`.py`,
  `.md`, `.json`, `.yml`, `.yaml`, `.toml`, `.ps1`, `.sh`, `.js`, `.ts`,
  `.tsx`, `.jsx`, `.cfg`, `.ini`, `.txt`, `.html`, `.css`, `.sql`).
- **Code symbol:** a backtick-quoted identifier (`` `foo_bar` ``), or an
  `identifier(` call/def-shaped token.
- **Line reference:** `:123`, `line 123`, or `L123`.

Scoring is applied to the reviewer-authored body only (the fenced
` ```...``` ` block, or bullet-list items), not the renderer's own heading
or lead-in prose — F1's heading wording is explicit implementer discretion
(plan §6/F1), so scoring the full rendered string would let template
boilerplate (e.g. a backticked field name in the lead-in sentence) produce
a false-actionable verdict, independent of what the reviewer actually
wrote.
