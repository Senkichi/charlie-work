# CLAUDE.md — Project Invariants for Worker Agents

## Commands

```bash
uv sync --all-extras             # install all deps (including dev extras: pytest, ruff)
uv run pytest -q --tb=short      # run tests
uv run ruff check .              # lint
uv run ruff format .             # format
```

## Invariants — preserve these in every PR

### Config / value objects are frozen dataclasses
All config and value-object types (`LabelConfig`, `DispatchConfig`, `ReviewConfig`,
`AutoMergeConfig`, `RuntimeConfig`, `DevinConfig`, `ClaudeCodeConfig`,
`OrchestratorConfig`, `SessionRecord`, `JanitorVerdict`, …) use
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

### Label state-machine names come from `LabelConfig`
All label strings must be read from a `LabelConfig` instance (default fields in
`config.LabelConfig`). Never hard-code label strings like `"agent:queued"` in
business logic — use `config.labels.queued`, `config.labels.in_progress`, etc.

### Adapters must not block on worker completion
`devin_shell.launch_devin_session` and `claude_code.launch_claude_worker` both use
`subprocess.Popen` and return immediately — they never call `process.wait()` or
`process.communicate()`. Workers can run for many minutes; any adapter that blocks
will deadlock the orchestrator. Preserve this invariant in any new adapter.

### Errors from external processes come back as values
`subprocess_runner.run_captured` and the launch functions return result objects
(`RunResult`, `SessionRecord`) with `ok`/`error` fields. They never raise on
non-zero exit codes or missing binaries. Callers check `.ok` or `.error`.
