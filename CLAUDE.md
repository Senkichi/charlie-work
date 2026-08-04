# CLAUDE.md — Project Invariants for Worker Agents

## Commands

```bash
uv sync --all-extras             # install all deps (including dev extras: pytest, ruff)
uv run --extra dev pytest -q --tb=short      # run tests
uv run ruff check .              # lint
uv run ruff format .             # format
```

## Invariants — preserve these in every PR

### Config / value objects are frozen dataclasses
All config and value-object types (`LabelConfig`, `DispatchConfig`, `ReviewConfig`,
`AutoMergeConfig`, `RuntimeConfig`, `DevinConfig`, `ClaudeCodeConfig`,
`TestAdequacyConfig`, `OrchestratorConfig`, `SessionRecord`, `JanitorVerdict`, …) use
`@dataclass(frozen=True)`. Never convert them to mutable classes or dicts.

### All JSON state writes are atomic
Every JSON file written by the orchestrator uses a temp-file + `replace()` pattern:
```python
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(...)        # or json.dump + handle.write("\n")
tmp.replace(path)          # atomic rename
```
This is implemented in `state.save_state`, `adapters._write_json`,
`devin_shell._write_json`, and `claude_code`. Never use a plain `open(path, "w")` for
any file that another process may read.

### State lives in GitHub labels + state.json — never in chat memory
Issue workflow state (`queued`, `in-progress`, `pr-open`, …) is stored as GitHub
labels on the issue and mirrored to `.var/charlie-work/state.json`. It is never
inferred from conversation history or process memory.

### Instrumentation: events.db (SQLite) + correlation IDs
Every event written to `state.json`'s capped `events` array (200 entries) is
also dual-written to an unlimited append-only SQLite database (`events.db`)
next to `state.json`. Use `self._record_event()` in `OrchestratorApp` methods
(it passes `state_path` automatically). In standalone functions, pass
`state_path=state_file` to `append_event()`. For events outside state-lock
contexts (e.g. loop-level errors), call `log_event()` directly from
`instrumentation.py`.

The `events` table has indexed columns for `kind`, `ts`, `correlation_id`,
`pr_number`, `issue_number`, and `level` (auto-classified info/warning/error).
Use `query_events()` for structured filtering or `event_counts_by_kind()` for
quick aggregation summaries. A `loop_passes` table records per-pass metadata.

Each `loop()` pass is wrapped in a `correlation_context()` so all events from
that pass share a correlation ID. Use `events_by_correlation_id()` to retrieve
the full event sequence for a specific pass when investigating issues. Legacy
`events.jsonl` files are auto-migrated to SQLite on first access.

### Label state-machine names come from `LabelConfig`
All label strings must be read from a `LabelConfig` instance (default fields in
`config.LabelConfig`). Never hard-code label strings like `"agent:queued"` in
business logic — use `config.labels.queued`, `config.labels.in_progress`, etc.

### Runner slots move by start/park — never by re-registration
**The implementation lives in `ci_fleet`, not here.** Since #869 (2026-08-01) every
runner/fleet consumer in this repo resolves `run_allocation_pass`, `runner_slots`,
and `runners` from `ci_fleet.charlie_work_adapter`. The identically-named modules
under `src/charlie_work/` are a dormant island with no importer in `src/`, kept
only as a rollback path and slated for deletion (#921). `tests/
test_dormant_fleet_marking.py` derives that set from the import graph and fails if
anything repoints back, so trust it over any grep. Note `charlie_work.fleet_dispatch`
is the *logger* name on the live allocation-prologue line — that is the module that
calls the adapter, not the one doing the work.

`ci_fleet`'s `runner_allocation`/`runner_slots` rebalance CI capacity across repos
by starting and stopping *already-configured* listeners. A parked runner keeps its
registration and reports `offline`. Never make reallocation mint a registration
token, run `config.cmd remove`, or delete a runner directory — that is `ci_fleet`'s
provisioning job (`ci_fleet/runners.py`), on a different (much slower) cadence.

Two safety properties must survive any change here. Both are `ci_fleet`'s code now;
they are listed because a change made *here* can still violate them through the
adapter:
- **Never stop a busy listener.** `park_runner_slot` re-checks for a live
  `Runner.Worker` child immediately before terminating, because the plan is a
  snapshot and GitHub's `busy` flag lags. Terminating a working listener aborts
  a CI job.
- **Never traverse outside `managed_root`.** `discover_runner_instances` walks
  exactly the configured root, non-recursively. This host has an unrelated
  runner *service* at `C:\actions-runner` that must never be touched; safety
  comes from the traversal's shape, not from filtering names afterwards.

`charlie runners allocate` is also the *only* thing allowed to decide which
listeners run. Operator scripts and post-reboot procedures must delegate to it
rather than starting every runner directly — a second controller silently undoes
parking and burns a full `demand_idle_samples` hysteresis window reconverging.

The `runner_allocation` **config section** (`config.RunnerAllocationConfig`, and the
cross-section floor check against `runner_scaling` in `config.py`) stays in this repo
regardless — `ci_fleet` reads that knob. It shares a name with the dormant module and
is not part of #921; deleting it would silently disable allocation.

### `EXIT_RESTART_REQUESTED` is a cross-version wire contract — never change it
`supervise_loop.EXIT_RESTART_REQUESTED` (3) is how the `fleet supervise-loop`
wrapper learns that its `fleet supervise` child wants to be replaced. The wrapper
holds the value in memory from *its* commit; the child loads it fresh from disk.
A self-deploy is exactly when those differ. Changing the number makes a stale
wrapper misread a restart request as a normal exit and skip the relaunch — the
#862 outage, reintroduced on the very deploy that changed it. Treat it like a
label string: read from the constant, never re-declare, and never renumber.

### Adapters must not block on worker completion
`devin_shell.launch_devin_session` and `claude_code.launch_claude_worker` both use
`subprocess.Popen` and return immediately — they never call `process.wait()` or
`process.communicate()`. Workers can run for many minutes; any adapter that blocks
will deadlock the orchestrator. Preserve this invariant in any new adapter.

### Errors from external processes come back as values
`subprocess_runner.run_captured` and the launch functions return result objects
(`RunResult`, `SessionRecord`) with `ok`/`error` fields. They never raise on
non-zero exit codes or missing binaries. Callers check `.ok` or `.error`.

The `api` adapter (`api_worker.launch_api_worker`) is subject to both invariants
above: it delegates to `launch_claude_worker` (`Popen`, non-blocking) and returns
launch failures as `ClaudeWorkerRecord` values with `.error` set — it never raises.
