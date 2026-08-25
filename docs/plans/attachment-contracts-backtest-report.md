# Attachment-Point Contracts backtest report

**Overall: PASS**

Samples: 5

## Criteria

- [PASS] `orchestrator_saturated` — saturated at all 4 samples where present
- [PASS] `test_charlie_work_saturated` — saturated at all 4 samples where present
- [PASS] `test_worktree_saturated_at_anchors` — saturated at all 3 anchor samples
- [PASS] `counterexamples_clean` — zero false-positive saturations; positive control: 3/13 counterexample module(s) actually produced an AP (queried), 10 emitted no AP in any sample (untested by this gate, not a validated pass): event_kinds.py, fleet_paths.py, git_pull_blockers.py, logging_setup.py, markdown_fence.py, prompt_sections.py, rescue.py, safe_path.py, safe_ref.py, throttle_signatures.py

## Cluster-B score (informational, not gated)

0 bare-function module(s) registering only via module-ledger archetype (expected partial miss)

## Samples

- `d28ce28f0eae42ce1da0eba742f41cf24966f24d` (2026-07, 2026-07-01): 22 points, 5 saturated, 0 parse failures
- `197a85b49a812715a4a872551d0fec538bd1e9ea` (2026-08, 2026-08-01): 442 points, 20 saturated, 0 parse failures
- `1ead8585f58f832c5442dc3d7173a662348e150a` (anchor, 2026-08-16): 589 points, 50 saturated, 0 parse failures
- `7373d47b2d0cf4612dd5600de9f66c320000ac1f` (anchor, 2026-08-17): 628 points, 38 saturated, 0 parse failures
- `9de0b9fc14279ef3578424bfb8fa6687d6eb12a9` (anchor, 2026-08-23): 697 points, 42 saturated, 0 parse failures
