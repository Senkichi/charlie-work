# 01 — Decision-event audit trail

## What it is

A ~277-LOC, zero-coupling observability core from freecode
(`reference/events.py`, `reference/sinks.py`, `reference/redaction.py`):

- `DecisionEvent` — frozen dataclass; every record carries `schema: "<ns>/decision/<N>"`,
  UTC ISO timestamp, stable `event_type` string, and a payload dict.
- `scrub()` (redaction.py) — recursive payload scrubbing by key-pattern before any sink sees
  the record; the call-site contract is "never put content in the payload in the first
  place; redaction is defense-in-depth at the sink."
- `JsonlFileSink` / `HmacChainSink` (sinks.py) — append-only JSONL; the HMAC variant chains
  each record over the previous record's MAC, making post-hoc tampering detectable with a
  locally held seed.

## Why charlie-work wants it

charlie-work's `state.json` + GitHub labels record **outcomes** (issue is `pr-open`, PR was
merged). They do not record **decisions with reasons**: why the janitor gate short-circuited
a PR, why a verdict was `request_changes` on cycle 3, why dispatch skipped a ready issue
(dependency gate? budget? adapter probe failure?), why the watchdog escalated a worker. As
the fleet layer multiplies passes across repos, "why did the orchestrator do X last Tuesday"
becomes a real forensic question — freecode's core insight (decisions must be reconstructable
after the fact) is the right discipline here, at a fraction of freecode's ceremony.

## Port plan (~hours)

1. Copy the three modules into `src/charlie_work/audit/` (new package); replace
   `freecode.config.logs_dir` with a path from `paths.py` under `.var/charlie-work/`
   (per-consumer-repo, like `state.json`). Keep writes append-only JSONL; the atomic-write
   invariant (CLAUDE.md) applies to state files, not append-only logs — document that
   distinction where the sink lives.
2. Namespace the schema `charlie/decision/1` and start an `event_type` registry doc
   (freecode's `docs/decision_events_schema.md` is the template: stable strings, payload
   keys documented, renames require a schema bump).
3. Instrument the four decision choke points first: `janitor.py` (gate verdicts + reason),
   `workflow.py`/`worker.py` dispatch selection (chosen issue, ordering reason, skipped
   candidates + reasons), `reconcile.py` (drift detected/repaired), watchdog tripwire
   transitions (already partially surfaced via `notify` — emit the decision event alongside).
4. Defer `HmacChainSink` wiring until there's a reason to distrust the log; the plain JSONL
   sink is the value. Config knob (`audit.enabled`, default on; `audit.hmac_seed_env`
   optional) via a frozen config dataclass per house style.

## Invariants to carry over

- No prompt/diff/patch content in payloads — IDs, counts, labels, reasons only.
- Stable `event_type` strings; additive schema evolution only.
- Emitting must never throw into orchestrator flow (freecode's `emit_decision` swallows sink
  errors by design — preserve that; errors-as-values is already house style).
