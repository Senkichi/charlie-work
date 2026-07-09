# 04 — Session-level admission control / quota-learning (trigger-conditioned)

## What it is

The one capability the 2026-07-09 evaluation found in freecode that **no gateway ships**
(checked: LiteLLM, Portkey, Bifrost, Kong): deciding whether you can afford to *start a
whole worker session* against today's remaining provider windows — as opposed to
per-request rate limiting, which everything has. Plus its companion: **learning** real
limits from observed 429s instead of trusting static config.

Frozen seeds in `reference/`:

- `capacity.py` (155 LOC, pure, no I/O): `usable_limit()` (limit × (1 − safety_margin)),
  `tighten_after_429()` (+0.05 margin per 429, capped 0.5, original limit preserved as the
  audit anchor), `window_for_timestamp()` (calendar-aligned per-minute/hour/day windows),
  `seed_capacity_from_quota_snapshot()` (runtime observations override seeds by
  `(provider, model, dimension, window)` 4-tuple).
- `planner_models.py`: the frozen `CapacityModel` row type with an honest `confidence`
  tier (`low` for prose-sourced limits) and `citation` field.
- `limits.py`: YAML seed loader.

## Trigger — do NOT build this yet

Build only when **both** hold:

1. Multiple concurrent workers/reviewers share one rate-limited credential (same API key or
   subscription window), and
2. `fleet.global_max_concurrent_sessions` (a blunt concurrency cap) demonstrably stops
   working — sessions failing mid-flight on 429/exhaustion rather than being throttled at
   dispatch.

Until then this is speculative machinery — freecode's whole history is the cautionary tale
for building capacity management ahead of demonstrated contention. Note the trigger is
about *shared paid/subscription* capacity; the original free-tier use case is dead
(aggregate free capacity ≈ $20/month equivalent — see freecode `docs/evaluation-2026-07/`).

## Design sketch when triggered (days, not weeks)

1. `src/charlie_work/admission.py`: port `capacity.py` + `planner_models.py` nearly verbatim
   (pure arithmetic; keep the no-I/O property and choke-guard it per spec 03).
2. Persistence: a small table/JSON ledger under `.var/charlie-work/` tracking
   `(provider, model, dimension, window)` consumption per calendar window — atomic-write
   rules apply. Do **not** port freecode's SQLite store (2,360 LOC) for this.
3. Dispatch integration: one check in the dispatch path (where the concurrency budget is
   consulted today) — estimate per-session cost from recent watchdog/cost-tripwire telemetry
   (`worker.py` already parses stream-json usage), admit/defer, decrement window on launch,
   reconcile actuals at reap.
4. Learning loop: adapter throttle classifications (the 429/throttle regexes in
   `claude_code.py`) call `tighten_after_429()`; observed limits enter the ledger as
   `source="runtime"` rows overriding seeds, exactly freecode's D-12/D-13 semantics.
5. Fleet layer: the global budget becomes min(concurrency cap, admission verdict) — additive,
   not a replacement, so the blunt cap stays as the fail-safe.

## Why not "just use LiteLLM"

2026 LiteLLM covers per-request RPM/TPM, cooldowns, affinity, and $-budget caps — if
charlie-work ever routes secondary roles through a proxy (spec 05), use those natively. The
session-admission decision ("don't start what you can't finish today") sits **above** any
proxy, at dispatch time, and belongs in the orchestrator.
