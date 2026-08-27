# Attachment-Point Contracts backtest report

**Overall: PASS**

Samples: 5

Sample window (honest, finding #3c): 2 distinct calendar month(s) available in this history (2026-07, 2026-08), requested 6; plus 3 explicit anchor(s). This is NOT necessarily a 6-month control -- read it as coverage over whatever history the repo actually has, plus the named anchors.

## Criteria

- [PASS] `orchestrator_saturated` — saturated at all 4 samples where present
- [PASS] `test_charlie_work_saturated` — saturated at all 4 samples where present
- [PASS] `test_worktree_saturated_at_anchors` — saturated at all 3 anchor samples
- [PASS] `counterexamples_clean` — zero false-positive saturations; positive control: 13/13 counterexample module(s) were present in the tree at some sample (queried)

## Cluster-B score (informational, not gated)

128 module(s) scanned with no AP archetype matched at all: d28ce28f0eae42ce1da0eba742f41cf24966f24d:src/devin_orchestrator/__init__.py, d28ce28f0eae42ce1da0eba742f41cf24966f24d:src/devin_orchestrator/__main__.py, d28ce28f0eae42ce1da0eba742f41cf24966f24d:src/devin_orchestrator/cli.py, d28ce28f0eae42ce1da0eba742f41cf24966f24d:src/devin_orchestrator/prompts.py, d28ce28f0eae42ce1da0eba742f41cf24966f24d:src/devin_orchestrator/state.py, 197a85b49a812715a4a872551d0fec538bd1e9ea:src/charlie_work/__init__.py, 197a85b49a812715a4a872551d0fec538bd1e9ea:src/charlie_work/__main__.py, 197a85b49a812715a4a872551d0fec538bd1e9ea:src/charlie_work/api_worker.py, 197a85b49a812715a4a872551d0fec538bd1e9ea:src/charlie_work/cli.py, 197a85b49a812715a4a872551d0fec538bd1e9ea:src/charlie_work/env_sanitize.py

## Samples

- `d28ce28f0eae42ce1da0eba742f41cf24966f24d` (2026-07, 2026-07-01): 22 points, 0 saturated, 0 parse failures
- `197a85b49a812715a4a872551d0fec538bd1e9ea` (2026-08, 2026-08-01): 442 points, 9 saturated, 0 parse failures
- `1ead8585f58f832c5442dc3d7173a662348e150a` (anchor, 2026-08-16): 589 points, 23 saturated, 0 parse failures
- `7373d47b2d0cf4612dd5600de9f66c320000ac1f` (anchor, 2026-08-17): 628 points, 26 saturated, 0 parse failures
- `9de0b9fc14279ef3578424bfb8fa6687d6eb12a9` (anchor, 2026-08-23): 697 points, 28 saturated, 0 parse failures
