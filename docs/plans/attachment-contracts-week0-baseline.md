# Attachment-Point Contracts — Week-0 measurement baseline

Generated: 2026-08-24. Source: `.attachment-budgets.json` (repo root, generated via
`uv run python -m charlie_work.attachment_contracts baseline`, mode defaults to `advise`)
and the llibrary god-object-scan harness run against `charlie-work` main.

## Acceptance checks

| Check | Result |
|---|---|
| `.attachment-budgets.json` mode resolves to `advise` by default | PASS — no `mode` key on-disk; `hook_entry._resolve_mode` falls back to `env ATTACHMENT_CONTRACTS_MODE` → baseline `mode` key → `"advise"` default (`src/charlie_work/attachment_contracts/hook_entry.py:50-59`) |
| `OrchestratorApp` (`src/charlie_work/workflow.py`) saturated | PASS — `member_count: 134`, `boundary: 5.0` |
| `tests/test_charlie_work.py` `test_module` AP saturated | PASS — `member_count: 1059`, `boundary: 41.0` |
| `tests/test_worktree.py` saturated | PASS — `member_count: 222`, `boundary: 41.0` |
| Zero of the 13 counterexample modules saturated | PASS — none of the 13 appear as baseline entries |
| Each of the 13 counterexamples appears in the raw `scan` inventory as positive control | PARTIAL, expected per spec — see note below |
| Any migration_runner/ledger AP present is exempt | N/A — zero `migration_runner`/`is_linear_ledger=true` APs exist in the current tree (`scan` output has 0 matches for either) |

### Counterexample positive-control note (not a bug)

`archetypes.py` only emits an `AttachmentPoint` for `typer_app` / `click_group` /
`blueprint` / `class` / `migration_runner` / `test_module` constructs (spec
"Archetype detection"). Of the 13 counterexample modules named in the spec:

- **3/13** contain a `class` and do surface in the `scan` inventory, all far below
  the `class` boundary (5.0): `file_lock.py` (`ByteRangeFileLock`, 3 members),
  `dirty_tree.py` (`DirtyTreeReport`, 1 member), `closing_keyword_gate.py`
  (`UnexpectedClosingReference`, 0 members).
- **10/13** are pure bare-function modules with no class/typer/click/blueprint
  construct at all — `event_kinds.py`, `fleet_paths.py`, `git_pull_blockers.py`,
  `logging_setup.py`, `markdown_fence.py`, `prompt_sections.py`, `rescue.py`,
  `safe_path.py`, `safe_ref.py`, `throttle_signatures.py` — and therefore emit
  **zero** `AttachmentPoint`s. This is not a scan-exclusion bug: they were walked
  (confirmed present on disk, not in `excludes.py`'s structural excludes, not in
  `parse_failures`), they simply have no archetype-matching construct to attach an
  AP to. This is exactly the documented "Cluster-B" caveat in the spec
  (`docs/specs/attachment-point-contracts-spec.md:207-208`: "whether any
  bare-function module registers via the `class`/module archetypes — expected
  partial miss, report it") and matches `attachment-contracts-backtest-report.md`'s
  own finding verbatim (3/13 produced an AP, 10/13 did not, zero false-positive
  saturations). No threshold or archetype-detection change was made — widening
  archetype detection to catch bare-function modules is a scope decision for a
  later iteration (module-level "class" archetype), not a bug in this pilot.

## `members_per_saturated_AP` (from `.attachment-budgets.json`, n=43)

Sorted: 6, 6, 6, 6, 6, 6, 7, 7, 9, 9, 10, 10, 12, 26, 41, 42, 45, 47, 47, 48, 49,
53, 54, 55, 55, 55, 58, 59, 59, 60, 75, 82, 99, 105, 118, 130, 134, 135, 152, 159,
166, 222, 1059

**p90 (nearest-rank, ceil(0.9 * 43) = 39th value): 152**

(`tests/test_worktree.py`, `test_module`, boundary 41.0 — the module immediately
below `tests/test_charlie_work.py` at 1059, which is the extreme outlier driving
the distribution's tail.)

## llibrary god-object-scan harness (charlie-work, main, window since 2026-08-24)

Command run:
```
uv run --active python scripts/god_object_scan.py --repo C:/Users/senki/repos/charlie-work \
  --out-dir C:/Users/senki/repos/llibrary/raw/analyses/2026-08-god-object/week0 \
  --window-since 2026-08-24 \
  --pin-files "tests/test_charlie_work.py,src/charlie_work/workflow.py,tests/test_worktree.py"
```
Output: `raw/analyses/2026-08-god-object/week0/charlie-work.metrics.json`,
`raw/analyses/2026-08-god-object/week0/charlie-work.per_commit.csv` (new files under
`raw/`, `raw/` itself untouched/append-only).

Headline metrics (from `charlie-work.metrics.json`):
- `commit_count`: 734, `end_state_file_count`: 287
- `gini.coefficient`: 0.6817 (file-size concentration across 287 files)
- `overall.total_added_source_lines`: 302,331
- `overall.share_ge800` (share of added lines landing in files that are/become
  >=800 lines): 0.5495
- `overall.share_p90`: 0.4253
- `overall.share_top3` (workflow.py + 2 other largest files): 0.3275
- `overall.share_pinned` (share attributable to the 3 pinned files —
  `tests/test_charlie_work.py`, `src/charlie_work/workflow.py`,
  `tests/test_worktree.py`): 0.3297
- 2 `regrowth_episodes` detected (a file dropping in size then regaining it),
  both on files outside the pinned set: `src/charlie_work/supervise.py` and
  `tests/test_config.py`

These are the Week-0 reference points; Week-1/Week-2 measurements should be diffed
against this file, not re-derived, per the idempotency contract in
`llibrary/docs/specs/2026-05-11-plan-b-safety-rails.md`.

## Backtest (G1, Deliverable 0 gate)

See `docs/plans/attachment-contracts-backtest-report.md` /
`.json` (already generated in this worktree): **Overall PASS**, 5 samples
(2026-07-01, 2026-08-01, and the 3 anchor SHAs `1ead858`/`7373d47`/`9de0b9f`).
All 4 hard-gate criteria pass; the counterexample positive-control finding there
(3/13 produced an AP) is the same result independently reproduced against the
live worktree scan above.
