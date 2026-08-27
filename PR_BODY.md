## Linked issue

Closes #1444

## What changed

Worker dispatch prompts now carry a **module map** section derived at packet build time from the live `src/charlie_work/` tree. For every `.py` module under the package, the section lists the dotted module name, the first line of its docstring, and its public-surface size (`__all__` length if defined, else the count of top-level names not starting with `_`).

This is the generation-time half of the god-file dynamic tracked in #1317 (extraction) and #1442 (CI ratchet stopgap): extraction removes lines, but nothing steered new lines away from the monolith, so it regrew. The map gives a worker a picture of what modules exist and what belongs where, so the largest file no longer wins by default gravity.

### Files

- **`src/charlie_work/module_map.py`** (new) — `build_module_map(package_dir, src_root)` walks the package directory with `pathlib` and parses each `.py` file with `ast` (zero imports — avoids side effects and heavy deps at packet build time). Returns the full markdown section, or an empty string when the package dir is absent/empty. Raises `OSError`/`SyntaxError`/`ValueError` on parse failures so the caller can record the event and degrade; it does not swallow exceptions itself, keeping the single point of enforcement for "dispatch never fails on a map error" at the call site.
- **`src/charlie_work/workflow.py`** — `_build_module_map_value(issue_number)` is the fail-soft wrapper: it calls `build_module_map` and, on a parse failure, logs a `worker_module_map_failed` warning event to `events.db` and returns `""` (omitted section). `_write_worker_prompt` passes the result as the `module_map` value. `module_map` is added to `WORKER_PROMPT_KEYS` so the drift check (`check_prompt_template_drift`) stays honest. `log_event` (not `self._record_event`) is used because `_write_worker_prompt` runs outside a state-lock context.
- **`src/charlie_work/instrumentation.py`** — registers `worker_module_map_failed` at `warning` level in `_LEVEL_BY_KIND`.
- **`src/charlie_work/prompts/worker.md`** and **`worker_claude_code.md`** — reference `$module_map` between the issue body and the scope contract, so the worker sees the module layout before deciding where to place code.
- **Tests** — `tests/test_module_map.py` (new, 12 tests); updated `ISSUE_VALUES` in `tests/test_prompt_sections.py` and two render tests in `tests/test_charlie_work.py` to supply the new `module_map` placeholder.

### Hard constraints (from the issue)

1. **The map is NEVER a hand-maintained list.** `build_module_map` walks the tree with `pathlib.rglob("*.py")` and parses with `ast`. There are zero hardcoded module names in `module_map.py` — verified by `test_newly_added_module_appears_with_no_config_change` (a module added to the tree after the first build appears in the second build with no config change) and `test_build_module_map_lists_modules_with_docstring_and_public_surface`.
2. **Map generation fails soft.** `_build_module_map_value` catches `OSError`/`SyntaxError`/`ValueError`, logs `worker_module_map_failed`, and returns `""`. Verified by `test_unparseable_file_omits_section_and_logs_warning_event` (a `SyntaxError` file → empty `module_map` + one warning event in `events.db`) and `test_missing_package_dir_omits_section_without_event` (a missing package dir → empty string, no failure event).

### Acceptance criteria

1. **Zero hardcoded module names.** ✅ — derivation is `rglob("*.py")` + `ast.parse`; no module name appears as a literal in `module_map.py`.
2. **A newly added module appears with no config change.** ✅ — `test_newly_added_module_appears_with_no_config_change`.
3. **Prompt-size cost measured and reported.** The map for this repo (76 modules) is **7,201 chars / 82 lines**. The base `worker.md` prompt (no map) is 12,710 chars; with the map it is 19,911 chars — a **~56.7% overhead** on the base prompt. The cost is bounded by the module count (one table row per `.py` file) and grows only as the tree grows.
4. **Event kind + consumer.** ✅ — `worker_module_map_failed` is registered at `warning` in `_LEVEL_BY_KIND`. The consumer is `scripts/heartbeat_check.py::check_warning_events`, which reads every `level='warning'` row from `events.db` (derived from the persisted `level` column, never a hardcoded kind list — see its docstring). Verified by `test_worker_module_map_failed_registered_as_warning`.

## Verification

```
uv run --extra dev pytest tests/test_module_map.py tests/test_prompt_sections.py tests/test_prompt_template_drift_check.py tests/test_prompt_render_contract.py tests/test_fix_prompt_template_drift.py tests/test_instrumentation.py tests/test_markdown_fence.py tests/test_issue_comments.py tests/test_doctor.py tests/test_janitor.py --tb=short
455 passed in 249.03s (0:04:09)
```

```
uv run ruff check .
All checks passed!

uv run ruff format .
282 files left unchanged
```

### Mutation check

Reverted each fixed artifact to its merge-base version (`git checkout 87df489 -- <path>`) and confirmed the regression tests fail, then restored the fix and confirmed they pass.

**1. `src/charlie_work/instrumentation.py`** (removed `worker_module_map_failed` from `_LEVEL_BY_KIND`):
```
uv run --extra dev pytest tests/test_module_map.py::test_worker_module_map_failed_registered_as_warning --tb=short
FAILED tests/test_module_map.py::test_worker_module_map_failed_registered_as_warning
E   AssertionError: assert 'worker_module_map_failed' in mappingproxy({...})
1 failed
```
After restore: `1 passed`.

**2. `src/charlie_work/workflow.py`** (removed `module_map` from `WORKER_PROMPT_KEYS`, `_build_module_map_value`, and the values dict):
```
uv run --extra dev pytest tests/test_module_map.py::test_module_map_is_a_worker_prompt_key tests/test_module_map.py::test_unparseable_file_omits_section_and_logs_warning_event tests/test_module_map.py::test_write_worker_prompt_includes_module_map_from_live_tree --tb=short
FAILED tests/test_module_map.py::test_module_map_is_a_worker_prompt_key - Ass...
FAILED tests/test_module_map.py::test_unparseable_file_omits_section_and_logs_warning_event
FAILED tests/test_module_map.py::test_write_worker_prompt_includes_module_map_from_live_tree
3 failed
```
(The `OrchestratorApp` constructor raised `PromptOverrideDriftError` because the templates reference `$module_map` but the writer no longer supplies it — the drift guard catching the regression at the single point of enforcement.) After restore: `12 passed`.

**3. `src/charlie_work/prompts/worker.md` + `worker_claude_code.md`** (removed `$module_map`):
```
uv run --extra dev pytest tests/test_module_map.py::test_write_worker_prompt_includes_module_map_from_live_tree --tb=short
FAILED tests/test_module_map.py::test_write_worker_prompt_includes_module_map_from_live_tree
E   AssertionError: assert '## Module map' in '# Devin Worker Task: Issue #1\n...'
1 failed
```
After restore: `12 passed`.

### Invariant enumeration (fail-soft paths in `_build_module_map_value`)

The fail-soft contract ("dispatch never fails on a map error; omitted section + warning event") has exactly these exit paths in `_build_module_map_value`:
1. `build_module_map` returns a non-empty string → returned directly (success, no event). ✅
2. `build_module_map` returns `""` (missing/empty package dir) → returned directly (omitted section, no event — the map is absent, not broken). ✅
3. `build_module_map` raises `OSError`/`SyntaxError`/`ValueError` → caught, `worker_module_map_failed` logged at `warning`, `""` returned (omitted section + event). ✅

There are no other `return` or `raise` statements between the `try` and the method's end.

## Risks / uncertain areas

- **Prompt-size overhead (~57%).** The map adds 7,201 chars to a ~12,710-char base prompt. This is a meaningful per-packet cost. It is bounded by the module count and is the explicit trade-off the issue asks for (placement steering vs. prompt budget). If this becomes a problem for very large repos, a future change could cap the map to the largest N modules or elide modules with a public surface of 0 — but that is out of scope for this issue, which asks for the full map.
- **`ast.parse` on every packet build.** For this repo (76 modules), `build_module_map` runs in well under a second. It is called once per `_write_worker_prompt` (once per issue at intake/dispatch), not per loop pass. No caching is added; the tree can change between packets by design (a newly added module must appear in the next packet).
- **Public-surface size is a proxy.** `__all__` length (when defined) or the top-level non-underscore name count is a rough measure of a module's public surface. It does not distinguish re-exports from genuine definitions. This matches the issue's specification exactly.

Generated with [Devin](https://devin.ai)
