# The fleet dashboard is read-only and reads only local derived state

Accepted 2026-10-01; implementation pending.

The fleet dashboard (`charlie dashboard`, local web app on 127.0.0.1) never changes fleet state and never calls GitHub. It reads only what the fleet already writes: each repo's `status-snapshot.json`, the per-repo and global `events.db`, the fleet registry, and `ci_fleet`'s allocation files. It keeps derived facts in `dashboard.db`, a rollup it builds incrementally and that can be deleted and rebuilt at any time. When a problem it shows needs action, the page shows the `charlie …` command to run instead of a button.

## Considered Options

- **Action buttons (requeue, unescalate, park or start a runner).** Rejected. The dashboard would become a second controller. `charlie runners allocate` must be the only thing that decides which listeners run, and lifecycle changes have to go through the commands that own them, because a raw label edit gets reverted by the next pass.
- **Filling gaps in the history with GitHub API calls** (merge timestamps, label history). Rejected. Those calls use the same token and rate limit as the fleet, and they would give the dashboard a second source of truth that can disagree with `events.db`. A metric that `events.db` cannot produce is a defect in what the orchestrator records. It gets filed against the orchestrator, and the chart says "not instrumented yet" until the event exists.
- **Querying every repo's `events.db` directly for each chart.** Rejected. Metrics such as lead time and stage time have to be reconstructed from sequences of events, and redoing that across every repo on each request is slow.

## Consequences

- `dashboard.db` is a cache, not state. Deleting it is always safe, and nothing else may read it as authoritative.
- Every chart shows the time window its sources cover, so a repo that joined the fleet late does not look like it had zero throughput before it joined.
- The "needs me" zone uses the same alarm definitions as the heartbeat, taken from a shared module. The page and the notifications therefore cannot disagree about what counts as wrong.
