## Linked issue

Closes #1383

## What changed

A fleet-wide Actions budget/runner outage causes required checks to fail with a `FAILURE` conclusion (not `CANCELLED`/`INFRA_FAILURE`) within seconds and with zero executed steps. The existing janitor gate treated these as ordinary code failures, routing healthy PRs through repeated no-op rework cycles that burned rework/no-op caps and eventually escalated them with `no_op_rework_cap_exceeded`.

This PR adds a distinct `infra_blocked` classification at the check-ingestion data boundary, before the rework-routing decision:

- **`InfraBlockedConfig`** (nested under `auto_merge`) holds the annotation patterns, instant-fail threshold, and persistence/window escalation knobs — all config-driven, not hardcoded in business logic.
- **`is_infra_blocked_check`** in `checks.py` is the single classifier: structural signals (zero non-setup steps, instant-fail) preferred over string matching, with config-listed annotation patterns as an independent signal.
- **`_enrich_checks_infra_blocked`** in `workflow.py` reclassifies `FAILURE` required checks to `INFRA_BLOCKED` before `summarize_checks`, used by both `review()` (rework routing) and `merge_ready()` (merge execution) — replacing the old inline `merge_ready`-only enrichment that missed this failure class.
- **`CheckSummary.infra_blocked`** is a new bucket distinct from `infra_failed` (per-PR #841); `JanitorVerdict.is_infra_blocked_block` mirrors `is_infra_failure_block`.
- **`review()`** holds the PR without dispatching rework, without incrementing attempt counters, and emits a `check_infra_blocked` warning event. Cross-pass persistence emits exactly one `infra_blocked_escalated` error event per configured window (not per PR per pass), tracked via a module-level dict that survives app instance rebuilds.
- **`heartbeat_check.py`**'s new `check_infra_blocked_events` is the consumer (AC4).
- **`github.is_infrastructure_failure`** delegates to the new classifier with a default config, preserving backward compatibility.

### Acceptance criteria

- **AC1**: A simulated check run failing in under 10 seconds with zero steps and a budget annotation is classified `infra_blocked` and dispatches no rework. ✓ (`test_infra_blocked_budget_failure_no_rework`)
- **AC2**: Attempt counters remain unchanged after an `infra_blocked` pass; a later genuine test failure on the same PR still routes to rework normally. ✓ (`test_infra_blocked_no_rework_counter_incremented`, `test_infra_blocked_then_genuine_failure_routes_to_rework`)
- **AC3**: Persistence across N passes emits exactly one operator escalation event per window, not one event per PR per pass. ✓ (`test_infra_blocked_persistence_one_escalation_per_window`)
- **AC4**: The new event kind has a consumer. ✓ (`test_infra_blocked_check_infra_blocked_event_emitted`, `check_infra_blocked_events` in heartbeat_check.py)

## Verification

```
uv run --extra dev pytest tests/test_checks.py tests/test_infrastructure_failure.py tests/test_janitor.py tests/test_event_kind_consumers.py tests/test_instrumentation.py tests/test_config.py -q --tb=short
........................................................................ [ 16%]
........................................................................ [ 32%]
........................................................................ [ 48%]
........................................................................ [ 65%]
........................................................................ [ 81%]
........................................................................ [ 97%]
..........                                                               [100%]
442 passed
```

```
uv run --extra dev pytest tests/test_charlie_work.py -q --tb=short -k "infra or rerun or rework or cancelled or merge_ready or merge_attempt or janitor"
..............................                                           [100%]
246 passed
```

```
uv run ruff check src/ scripts/ tests/test_checks.py tests/test_charlie_work.py
All checks passed!
```

## Risks / uncertain areas

- **`merge_ready()` enrichment change**: The old inline enrichment in `merge_ready()` set `INFRA_FAILURE` (routing to `infra_failed`); the shared helper sets `INFRA_BLOCKED` (routing to `infra_blocked`). This is the intended behavior change — checks that were previously `infra_failed` are now correctly `infra_blocked` — but it means `merge_ready` no longer attempts per-PR infra reruns for these checks. This is correct: fleet-wide budget failures have no per-PR rerun path. The existing `infra_failed` bucket (CANCELLED/TIMED_OUT) is unchanged.
- **Module-level `_infra_blocked_window` dict**: Cross-pass state is tracked in a module-level dict keyed by repo_root string, surviving app instance rebuilds within the same supervisor process. On supervisor restart, the counter resets — acceptable since the escalation window is time-bounded and a restart is itself a signal the operator is present.
- **`is_infra_blocked_check` `enabled` check**: The `config.enabled` gate is checked in the classifier itself (not just the caller), so `is_infrastructure_failure` (which passes a default config) still classifies when called directly. This is intentional: the legacy wrapper should preserve its pre-#1383 behavior of detecting zero-step/billing-annotation failures.

Generated with [Devin](https://devin.ai)
