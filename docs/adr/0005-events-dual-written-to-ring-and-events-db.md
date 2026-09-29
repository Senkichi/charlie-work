# Events are dual-written to a ring and to events.db

Every event goes to two places: the capped `events` ring in `state.json` (`state.DEFAULT_EVENT_RING_SIZE` = 2000, overridable through `runtime.event_ring_size`, #525) and an unlimited append-only SQLite log, `events.db`, next to it. The ring is a recent-activity view kept in the state cache, and `events.db` is the audit trail, so root-cause analysis does not depend on what the bounded ring still holds. The events table is indexed on `kind`, `ts`, `correlation_id`, `pr_number` and `issue_number`. Each `loop()` pass shares one correlation ID, so a pass can be replayed with `events_by_correlation_id()`.

## Consequences

- Emission goes through one path. `state.append_event(..., state_path=...)` appends to the ring, trims to the cap, and calls `instrumentation.log_event`. In `OrchestratorApp`, `_record_event` reaches it through `write_gate.record_event`, so dry-run suppresses both writes (#1324). Events outside a state-lock context call `log_event()` directly.
- The database write is best-effort: `log_event` swallows and logs I/O errors so instrumentation never breaks the workflow. The ring append is not, so the two stores can disagree after a database failure. The database is the fuller record, not a guaranteed superset.
- Any query for "did X ever happen" must use `events.db`. The ring drops the oldest entries past the cap.