## Test hygiene

Never hardcode calendar dates in test seeds. A fixture that plants a literal date like `"2026-01-15"` drifts stale the moment a date-window filter (e.g. "issues opened in the last N days") compares it against the real clock. Derive test dates from `datetime.now(...)` (offset with `timedelta` as needed) so date-window filters cannot rot.
