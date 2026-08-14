## Linked issue

Closes #1010

## What changed

A dispatched worker edited a sibling repo's shared main checkout because the issue's subject code (`suite_coverage.py`) was not in the repo it was dispatched against. Worker containment was enforced only by prompt text, scoped to "the repo" — which does not cover a different repo at all.

Three independent fixes, in enforcement-strength order:

### 1. Pre-flight cross-repo gate (`src/charlie_work/cross_repo_gate.py`)

At dispatch time, extract file-path references from the issue body and check whether any of them exist in the target repo. If the issue references file paths but **none** of them exist in the repo, escalate to `agent:human-needed` with a `cross_repo_target` reason instead of burning a worker and a slot. This makes the invalid state unrepresentable rather than detected late.

- Wired into both the real dispatch path (`_dispatch_impl`) and the dry-run path.
- Escalation uses the existing `_escalate_issue` helper and `transition(..., "redispatch_escalated")` label edge, following the same pattern as dispatch-failed cap exhaustion.
- A new `dispatch_cross_repo_escalated` event is recorded in `events.db` for auditability.
- The gate is conservative: an issue that references no file paths passes; an issue where at least one referenced path exists passes. Only an issue where every referenced path is absent is blocked.
- URLs are stripped before matching so `https://example.com/foo.py` is not treated as a file path.

### 2. Widened containment clause (`src/charlie_work/prompts/worker_sections/scope_contract.md`)

Changed from "never resolve, cd into, or modify any other checkout of the repo" to "never resolve, cd into, or modify any path outside the assigned worktree root — not another checkout of this repo, not a sibling repo, not a shared main checkout, nothing." Also added the "If the file you need to edit is not inside your worktree, stop and explain the blocker instead of going elsewhere" instruction.

Added `$section_scope_contract` to `rework.md` so rework workers get the same widened clause — a rework worker can wander to a sibling repo just as easily as a fresh-dispatch worker.

### 3. Structural containment assertion (`src/charlie_work/prompts.py`)

A post-render guard (`assert_containment`) at the dispatch boundary that catches a repo-local flat override dropping `$section_scope_contract` or reverting to the old repo-scoped wording. Mirrors the existing `assert_no_merge_contract` / `assert_execution_contract` / `assert_conventional_commit_title` pattern (issues #714, #717, #715). Called in both `_write_worker_prompt` and `_write_rework_prompt`.

## Verification

### Tests run

```
uv run --extra dev pytest tests/test_cross_repo_gate.py tests/test_prompt_sections.py tests/test_prompt_template_drift_check.py tests/test_prompt_render_contract.py tests/test_safe_path.py tests/test_markdown_fence.py tests/test_issue_comments.py tests/test_closing_keyword_gate.py -q --tb=short
........................................................................ [ 67%]
..................................                                       [100%]
106 passed
```

### Lint/format

```
uv run ruff check .
All checks passed!

uv run ruff format --check .
185 files already formatted
```

### Mutation checks

**Fix 2 — scope_contract.md reverted to merge-base:**
```
git checkout ec02ffc5dff373d293ccd6f5f4260c9f3f3362d1 -- src/charlie_work/prompts/worker_sections/scope_contract.md

uv run --extra dev pytest tests/test_prompt_sections.py::test_worker_prompts_contain_widened_containment_clause tests/test_prompt_sections.py::test_assert_containment_passes_for_package_templates tests/test_prompt_sections.py::test_rendered_worker_prompts_contain_shared_section_text_and_no_placeholders -q --tb=short
FFF
FAILED tests/test_prompt_sections.py::test_worker_prompts_contain_widened_containment_clause
FAILED tests/test_prompt_sections.py::test_assert_containment_passes_for_package_templates
FAILED tests/test_prompt_sections.py::test_rendered_worker_prompts_contain_shared_section_text_and_no_placeholders

# After restoring fix:
...
3 passed
```

**Fix 3 — assert_containment in prompts.py reverted to merge-base:**
```
git checkout ec02ffc5dff373d293ccd6f5f4260c9f3f3362d1 -- src/charlie_work/prompts.py

uv run --extra dev pytest tests/test_prompt_sections.py::test_assert_containment_rejects_flat_override_without_clause tests/test_prompt_sections.py::test_assert_containment_rejects_old_repo_scoped_wording tests/test_prompt_sections.py::test_assert_containment_accepts_override_with_widened_clause -q --tb=short
FFF
FAILED tests/test_prompt_sections.py::test_assert_containment_rejects_flat_override_without_clause - ImportError: cannot import name 'MissingContainmentError'
FAILED tests/test_prompt_sections.py::test_assert_containment_rejects_old_repo_scoped_wording - ImportError: cannot import name 'MissingContainmentError'
FAILED tests/test_prompt_sections.py::test_assert_containment_accepts_override_with_widened_clause - ImportError: cannot import name 'assert_containment'

# After restoring fix:
...
3 passed
```

**Fix 1 — cross_repo_gate.py reverted (file removed):**
```
mv src/charlie_work/cross_repo_gate.py src/charlie_work/cross_repo_gate.py.bak

uv run --extra dev pytest tests/test_cross_repo_gate.py -q --tb=short
ERROR collecting tests/test_cross_repo_gate.py
ModuleNotFoundError: No module named 'charlie_work.cross_repo_gate'

# After restoring fix:
.............
13 passed
```

## Invariant enumeration

The cross-repo escalation block in `_dispatch_impl` (second lock section) has these exit paths, each of which saves state before the next issue:

1. **Normal escalation path** — `_escalate_issue` + `append_event` + `save_state` + `transition`. If the label transition fails, the `label_error` branch saves state.
2. **Label transition succeeds** — `TransitionOutcome.APPLIED`: no label_error, loop continues to next issue.
3. **Label transition partial failure** — `TransitionOutcome.PARTIAL_FAILURE`: `label_error` recorded, `save_state` called, loop continues.

Every path calls `save_state` before moving to the next issue or exiting the lock.

## Risks / uncertain areas

- **False positives in the cross-repo gate**: an issue that references file paths in prose (e.g. "compare with `src/ci_fleet/suite_coverage.py`") where none exist in the target repo will be escalated. This is a safe failure mode — a human can re-label after confirming the target repo. The gate requires at least one `/` separator in the path to reduce false positives from bare filenames.
- **Path extraction regex**: the regex matches relative paths with at least one separator and a file extension, absolute Windows/POSIX paths, and backtick-quoted paths. It may miss unusual path formats (e.g. paths with spaces, paths without extensions). This is conservative — missing a path means the gate passes, not blocks.
- **Rework prompt change**: adding `$section_scope_contract` to `rework.md` is a new placeholder reference. The `unsupplied_placeholders` startup check validates this against `REWORK_PROMPT_KEYS`, but the section variable is supplied by `section_variables()` so it resolves correctly. All drift-check tests pass.
- **Security-sensitive behavior**: this PR adds a pre-flight gate that reads issue bodies and makes dispatch decisions based on their content. The gate is read-only (no file writes outside the worktree) and escalates rather than dispatching — it cannot be used to bypass the dispatch flow, only to block it.

Generated with [Devin](https://devin.ai)
