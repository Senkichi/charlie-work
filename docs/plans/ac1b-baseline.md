# AC-1b findings-actionability baseline

Measured output of `scripts/ac1b_findings_actionability.py` against the real
`.var/charlie-work/prs` corpus, run through the REAL production renderer
(`charlie_work.workflow._render_required_changes_section`) at the pinned
post-F1/F5 commit. See `docs/plans/rework-findings-channel.md` (untracked,
main checkout) for the fix design this measures against.

Re-run the same script, unmodified, to update the measurement after further
fixes land:

```
VIRTUAL_ENV= PYTHONPATH="$PWD/src" uv run --no-sync python \
    scripts/ac1b_findings_actionability.py --repo <path-to-checkout>
```

## Pinned post-fix run

- **Code SHA:** `8d28a83fc60bbb3ac0ee17f176ce58dcaac36931` (`fix: preserve
  per-worktree skip reasons in the worktrees_reclaimed event (#1012)`)
- **F1 / F5 landings:**
  - F1 = `35c072d` (PR #766), landed 2026-07-31T02:29:42Z
  - F5 = `9b1f637` (PR #768), refined by `395aab1` and `bbcc132`.
  All three are confirmed ancestors of the pinned SHA.
- **Corpus:** `C:\Users\senki\repos\charlie-work\.var\charlie-work\prs`
- **Corpus size found:** 19 `request_changes` verdicts

## Results by category (never a single aggregate)

| Category | Count | AC-1 (non-empty) | AC-1b (actionable) |
|---|---|---:|---:|
| cross_family_generic_collapse | 2 | 2 | 0 |
| synthetic_ci_failure | 3 | 3 | 2 |
| real_reviewer_prose | 14 | 14 | 8 |
| **TOTAL** | **19** | **19** | **10** |

The `proj. post-F1 AC-1b` column from the pre-F1 report has been replaced: the
re-run uses the real post-F1 renderer, so there is no projection left to show.

## The aggregate is not an "after" number

The 10/19 headline is **58% pre-F1 residue** and should not be quoted as a
post-fix result. F1 changed what gets captured at verdict *generation* time;
re-rendering a pre-F1 record through the post-F1 renderer cannot add referents
that were never captured.

Splitting on each verdict's own `reviewed_at` field against F1's landing time
(`2026-07-31T02:29:42Z`):

| Period | Count | AC-1b (actionable) | Actionable % |
|---|---|---:|---:|
| pre-F1 (`reviewed_at < 2026-07-31T02:29:42Z`) | 11 | 3 | 27% |
| post-F1 (`reviewed_at >= 2026-07-31T02:29:42Z`) | 8 | 7 | 88% |

All 8 post-F1 verdicts have `carried_forward_from=False`, so none of them
carry a post-F1 `reviewed_at` over pre-F1 content. F1's real effect on the
records it could influence is therefore **27% -> 88%**.

## Cross-family post-fix actionability is unmeasured, not zero

`docs/plans/rework-findings-channel.md` section 8 names cross-family generic
collapse as the FALSE GREEN that AC-1b exists to discriminate. That is the
category the work targeted. Both cross-family verdicts in this corpus are
pre-F1:

- `pr-695` — `reviewed_at` 2026-07-31T01:08:02Z (~81 min before F1)
- `pr-724` — `reviewed_at` 2026-07-30T05:20:18Z (~21 h before F1)

**There is no post-F1 cross-family verdict in the corpus at all.** The `0/2`
in the table above is therefore a pre-F1 artifact, not a post-fix result.

`classify_verdict` assigns this category by a pure content test —
`stripped == sentinel`, where the sentinel is the content-free string
`Cross-family review found BLOCKER/MAJOR findings`. It does not look at
provenance. A cross-family review that now carries real content does not land
in this bucket; it lands in `real_reviewer_prose`. So an empty
`cross_family_generic_collapse` post-F1 is ambiguous between:

- **(a) F1 worked** — no verdict collapses to the sentinel anymore. This is
  the designed outcome, and an empty bucket is what success looks like.
- **(b) No cross-family verdict reached the corpus at all.**

Looking at `events.db` for the cross-family path: there were
`cross_family_verdict_unparseable` (16) and `cross_family_verdict_abandoned` (8)
events on 2026-07-31 between 06:44Z and 12:55Z, all with
`reason=blocker_or_major_with_no_extractable_summary`. Two of the abandoned
PRs (`pr-692` and `pr-802`) do appear in the post-F1 corpus, but their stored
verdicts are dated after their abandonment, so those stored verdicts came from
a different review path, not from the cross-family attempt that was thrown
away. The burst ended before `bbcc132` landed at 2026-07-31T14:30:25Z, and
there have been zero events of either kind since.

Therefore **cross-family post-fix behaviour is still unverified**: the path
has not been exercised since the parse repair landed. The AC-1b corpus is the
wrong instrument for the cross-family question; the right check is prospective
and narrow — when the next cross-family review produces BLOCKER/MAJOR
findings, does it parse, and does the stored verdict carry concrete referents?

## Self-test (detector controls, run before the main table)

| Input | Referents found | Actionable | Expected |
|---|---|---|---|
| `"See src/charlie_work/workflow.py:3678 in \`_render_required_changes_section\`."` | file_path, code_symbol, line_number (3 kinds) | True | True |
| `"Cross-family review found BLOCKER/MAJOR findings"` (literal collapse sentinel) | none | False | False |

**PASSED.** The detector can produce both `True` and `False` — the results
above are not an artifact of a broken detector.

## Mutation checks (prove the harness is sensitive, using the real renderer)

1. **Monkeypatch `summary` only**, real unmodified renderer: the harness
   reports `count moved` explicitly. A `False` result is expected when the
   sample verdict has a non-empty `required_changes` list (the renderer reads
   that field, not `summary`), or on a pre-F1 checkout when the renderer
   early-returns `''` for empty `required_changes`. It is never a silent
   failure.
2. **Mutate `required_changes`** (the field the real renderer reads) to
   `["Fix the null check in src/charlie_work/workflow.py:3700"]`, real
   unmodified renderer: rendered section is non-empty and scores actionable
   (`file_path`, `line_number` both found). **PASSED** — proves the detector +
   real-renderer pipeline can and does move when the input changes.

## Pre-F1 baseline (retained for comparison)

The original pre-F1 measurement pinned the renderer at `b3b827de7cc157d60992b138d3d789f08d8f1d22`
(`test(worktree): close COMPLETED-misclassification gap left by PR #695's
regression test (#764)`). It found 20 `request_changes` verdicts and reported:

| Category | Count | AC-1 (non-empty) | AC-1b (actionable) | Projected post-F1 AC-1b* |
|---|---|---:|---:|---:|---:|
| Cross-family generic collapse | 7 | 0 | 0 | 0 |
| Synthetic CI-failure | 3 | 0 | 0 | 0 |
| Real reviewer prose | 10 | 0 | 0 | 10 |
| **TOTAL** | **20** | **0** | **0** | **10** |

\* Diagnostic only — a local stand-in for F1's contract, before the real F1
renderer was available.

The pre-F1 table showed AC-1 = 0/20 and AC-1b = 0/20 because the pre-F1
renderer early-returns an empty string whenever `required_changes` is empty.
The `0/7` cross-family and `0/3` synthetic-CI numbers are pre-F1 artifacts and
should not be read as post-Fix results. The plan's §2.2 claim (pre-F1 AC-1 =
0/20) is confirmed by this baseline. The current post-Fix measurement above
replaces the pre-F1 report as the authoritative baseline.

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
boilerplate (e.g. a backticked field name in the lead-in sentence) produce a
false-actionable verdict, independent of what the reviewer actually wrote.
