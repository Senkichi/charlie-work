# Approved PRs hand off to the Aviator merge queue

When `auto_merge.mergequeue_label` is set, an approved and green PR gets that label (a merge queue handoff) instead of being self-merged, and Aviator's queue (`.aviator/config.yml`, trigger label `mergequeue`) merges it asynchronously. The live config has done this since 2026-07-17, after PR #419 added the handoff as an optional, default-off setting (`orchestrator.config.yaml`, `config.py` `AutoMergeConfig.mergequeue_label`). The self-merge path remains for repos that leave the label unset.

## Considered Options

GitHub's native merge queue was tried first. Issue #410 asked to adopt it, and the worker's preflight found it unavailable: `Senkichi/charlie-work` is a private repo owned by a personal (User) account with branch protection off. The issue's stated fallback was the existing front-of-train branch sync. The live config comment records that the native queue returns 422 on this repo and that Aviator works on it.

## Consequences

- A handed-off PR is not merged. `merge_ready` records `status="mergequeue"`, never `"merged"`, so its idempotency short-circuit does not fire and the handoff can be safely re-applied. Reconcile's `merged_outside_orchestrator` path moves the state to `"merged"` and runs the merged label transition once GitHub reports the merge.
- The recorded verdict, not the queue, authorizes the merge. Aviator is configured with zero required approvals (`number_of_approvals: 0`), and `orchestrator.config.yaml` notes the out-of-band `review-decision.json` remains the real gate.
- Aviator can accept the label and then strip it. `merge_ready` in `workflow.py` detects this (`mergequeue_label_reverted`) and re-drives the handoff. #1873 (PR #1888) fixed a deadlock in which a stale `mergequeue` status suppressed the base sync that the recovery path needed.
- Queue behavior needs its own watchdogs: a time-in-queue bound (`mergequeue_wedge_hours`, #1401) and an Aviator-failure trigger.
- Aviator's queue sync-merges would trip the #502 unauthorized-merge tripwire, so `auto_merge.queue_bot_login` (#1194, PR #1242) lets it recognize them. Aviator's parallel-mode draft PRs (`mq-bot-*` branches) are hidden from the fleet by `queue_bot.is_queue_bot_pr`. Parallel mode was switched on 2026-09-27 (`5898297e`).
