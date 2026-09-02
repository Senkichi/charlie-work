## Linked issue

Closes #1515

## What changed

Phase 2 Track B (PR #1517) deleted the per-issue adapter routing subsystem
(`routing.py`, `select_adapter`, `_dispatch_partitioned`, `complexity_high`).
This PR cleans up the residue that was out of Track B's file scope:

### 1. Dead `template` parameter dropped from `_write_worker_prompt`

`OrchestratorApp._write_worker_prompt(..., template: str | None = None)` —
the only callers that passed `template=` were the deleted routing branches;
all 3 surviving production callers (`workflow.py:4561`, `5417`, `5997`) omit
it. The parameter and its `template or self.config.dispatch.worker_template`
branch are removed; `_write_worker_prompt` now always renders
`self.config.dispatch.worker_template`.

**Test callers updated** to set `config.dispatch.worker_template` instead of
passing `template=`:
- `test_prompt_render_contract.py` — `test_worker_claude_code_md_renders_via_real_writer`
- `test_prompt_template_drift_check.py` — `test_worker_prompt_keys_match_real_writer`
- `test_markdown_fence.py` — `_render` helper and `_render_via_pre_change_template`
- `test_issue_comments.py` — `_render_via_pre_change_template` and two parametrized tests

### 2. Stale comment/prose references reworded

| File | Old reference | Fix |
|------|--------------|-----|
| `dead_worker_reap.py:609` | `select_adapter` | "the default adapter" |
| `dispatch_selection.py:427` | `routing.record_adapter_choice` | historical note about deleted per-issue adapter selector |
| `doctor.py:283` | `routing.select_adapter` / `policy:complexity` | `config.api_worker.enabled` |
| `workflow.py:2783` | `routing.record_adapter_choice` | historical note |
| `workflow.py:2825` | "adapter_history only grows" (present tense) | "only grew" (past tense) + deletion note |
| `workflow.py:4017` | "api routing falling back to devin-shell for one issue" | "a fallback to a different adapter" |
| `workflow.py:20426` | "independent of per-issue routing" | phrase dropped |
| `worktree.py:2881` | `routing.record_adapter_choice` | historical note about legacy `adapter_history` field |
| `conftest.py:157` | `test_dispatch_partitioned_homogeneous_batch_labels_with_single_kind` | "a partitioned-dispatch batch-labels test" |
| `test_charlie_work.py:45455` | `routing.record_adapter_choice` | historical note |
| `test_charlie_work.py:45063` | `record_adapter_choice` (in `_simulate_redispatch` docstring) | reworded to "per-issue adapter selector … deleted in Phase 2 Track B (PR #1517)" |
| `test_charlie_work.py:45487` | `record_adapter_choice` (inline comment on empty `adapter_history` seed) | same rewording |
| `test_charlie_work.py:45523` | `record_adapter_choice` (in `_simulate_dispatch` docstring) | same rewording |
| `test_charlie_work.py:51943` | `_dispatch_partitioned` | `dispatch_sessions` |
| `test_charlie_work.py:52016` | `_dispatch_partitioned` | `_dispatch_rework_impl` |

### 3. QUICKSTART.md

Dropped "plus `complexity:high`" from the bootstrap-labels line — that label
field was deleted by Track B.

### Preserved (not stale)

- **Historical-context comments** in `api_worker.py:188`,
  `test_api_worker.py:902`, `instrumentation.py:245` that accurately document
  the Phase 2 Track B deletion ("the refusal gate that used to live in
  routing.py before its deletion"). These are historically accurate, not
  stale descriptions of current behavior.
- **Legitimate "routing" uses** — rework routing, escalation routing,
  merge-conflict routing, review routing — are live concepts unrelated to
  the deleted per-issue adapter routing subsystem. The acceptance grep
  `\brouting\b` matches ~160 such legitimate uses; they are not touched.

## Verification

```
uv run --extra dev pytest tests/test_prompt_render_contract.py tests/test_prompt_template_drift_check.py tests/test_markdown_fence.py tests/test_issue_comments.py tests/test_doctor.py tests/test_worktree.py -q --tb=short
```
Result: 394 passed in 137.69s

```
uv run --extra dev pytest tests/test_charlie_work.py::test_orphan_sweep_redispatch_cap_fires_with_api_worker_disabled tests/test_charlie_work.py::test_dispatch_rework_no_rescue_skips_redundant_manifest_write tests/test_workflow_dead_worker_write_gate.py -q --tb=short
```
Result: 6 passed

```
uv run --extra dev pytest tests/test_dispatch_selection_split.py tests/test_dead_worker_reap_split.py -q --tb=short
```
Result: 22 passed

```
uv run ruff check .
```
Result: All checks passed!

```
uv run ruff format .
```
Result: 343 files left unchanged

### Acceptance grep verification

`rg 'complexity_high|_dispatch_partitioned|select_adapter' src/ tests/` returns
zero matches — all unambiguous references to the deleted subsystem are removed.

`rg 'routing\.record_adapter_choice|routing\.select_adapter|routing\._api_preflight' src/ tests/`
returns only 2 matches in `api_worker.py:188` and `test_api_worker.py:902` —
both are historical-context comments that accurately document the deletion
(preserved deliberately).

`rg 'record_adapter_choice' src/ tests/` returns zero matches — the deleted
function's name is fully gone from product code and tests (the three stale
references in `test_charlie_work.py` that the original PR missed were reworded
in the rework commit).

`rg 'complexity' docs/QUICKSTART.md` returns zero matches.

The broad `\brouting\b` pattern also matches ~160 legitimate uses (rework
routing, escalation routing, etc.) that are unrelated to the deleted
per-issue adapter routing subsystem and are not in scope for this issue.

## Rework (review feedback)

The original PR fixed one stale `record_adapter_choice` reference in the
`test_orphan_sweep_redispatch_cap_fires_with_api_worker_disabled` docstring
but left three more identical stale references to the same deleted function
inside the same test function body. This rework commit rewords all three to
match the accurate phrasing already applied at the docstring head
("the per-issue adapter selector that wrote it was deleted in Phase 2
Track B, PR #1517"):

- `_simulate_redispatch` docstring (~`test_charlie_work.py:45063`)
- inline comment on the empty `adapter_history` seed (~`test_charlie_work.py:45487`)
- `_simulate_dispatch` docstring (~`test_charlie_work.py:45523`)

After this rework, `rg 'record_adapter_choice' src/ tests/` returns zero
matches — the symbol name is fully gone from product code and tests. The
issue #1515 acceptance grep (`complexity_high|_dispatch_partitioned|select_adapter`)
did not include `record_adapter_choice`, so it could not catch these; they
were found by the reviewer.

## Risks / uncertain areas

- The `template` parameter was keyword-only with a default (`None`), so
  dropping it can only break callers that explicitly pass `template=`. All
  such callers (4 test files, 6 call sites) have been updated. A grep for
  `_write_worker_prompt.*template=` in tests/ returns zero matches.
- The `_write_worker_prompt` signature change is a private method (`_`
  prefix); no public API or module re-export is affected.
- Two historical-context comments referencing `routing.py` by name
  (`api_worker.py`, `test_api_worker.py`) were deliberately preserved as
  accurate history. If the reviewer prefers these reworded too, they can be
  addressed in a follow-up.
