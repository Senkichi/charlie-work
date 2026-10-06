# Reviewer launch exceptions are per-PR failures, never pass crashes

Accepted 2026-10-06; records the decision left open by wave D (#2222), review finding for issue #2237.

`RealReviewLauncher.launch` catches every exception the selected harness launcher raises and returns it through `host.launch.error_record` as a failure value. Before the port existed, `dispatch_reviews` called the launcher directly inside a per-PR `try` whose handler covered only `OSError`, `GitHubError` and `ValueError` — so a programming error (TypeError, AttributeError, …) escaped and crashed the whole review-dispatch pass.

Two consequences of the broad catch, both deliberate:

- The exception text lands in `record.error`, so it runs through the same `match_throttle_tail` / `match_quota_tail` check a returned error record gets. A raised exception carrying provider-throttle or quota text still rolls the claim back and arms the reviewer-quota backoff instead of wedging the PR as `review_dispatch_failed`.
- Any other raised exception becomes a per-PR `review_dispatch_failed` with a `launch_failed` event (`error_class` internal): the failure is bounded to one PR, retried on later passes, and escalated by the review-dispatch attempt cap — the same path as any launch failure.

## Considered Options

- **Narrow the catch: log and re-raise non-environmental exceptions.** Rejected. A raised exception means the launcher broke its own contract — launchers return failure values rather than raising ("Errors from external processes come back as values"). Re-raising would abort `dispatch_reviews` mid-pass and deny review dispatch to every other selected PR that pass, and the loop would hit the same bug again on the next pass. Converting to `review_dispatch_failed` keeps the blast radius at one PR while the `launch_failed` event still makes the defect visible.

## Consequences

- The launch port's contract stays "never raises" (the `host/launch.py` module docstring), with `error_record` as the single failure-value constructor on this seam.
- A deterministic launcher bug surfaces per PR each pass until the attempt cap escalates it, instead of crashing every pass for the whole batch.
- Tests that exercise a raising launcher should assert on the recorded `record.error` and emitted `launch_failed` event, not on a raised exception.
