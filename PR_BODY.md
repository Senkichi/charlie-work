## Linked issue

Closes #735

## What changed

`migrate-state-dir` relocated the state tree correctly but left every **embedded absolute path** inside `state.json` pointing at the old root. Those paths (`prompt_path`, `decision_path`, `cross_family_report`, `verdict_source`, ...) were only rewritten opportunistically at dispatch/render sites, so a record that got no fresh render never converged. Immediately after a migration the state file was internally inconsistent: the tree at the new root, the pointers naming the old one. On the 2026-07-30 job-cannon migration the compat junction was never created and the source dir was `rmdir`'d, leaving 441 dangling pointers with no reader to notice.

This PR makes the path rewrite part of the migration itself, so the inconsistent state is never representable:

- **`state_migration.py`** — after `apply_state_dir_migration` moves every child, it now walks the parsed `state.json` structurally (not `str.replace` on the file text — that cannot distinguish a path value from a branch name or issue title that happens to contain the token) and rewrites every string leaf under the old root to the new one, verifying each rewritten target exists before committing. The rewrite runs under `state_lock` and saves atomically via `save_state` (temp-file + `replace`).
  - New `StateRewriteResult` frozen dataclass (ok/rewritten/error).
  - New `MigrationOutcome.rewritten_paths` field reports the count so it is observable rather than assumed.
  - New injectable `state_rewriter` seam on `apply_state_dir_migration` (mirrors the existing `mover` seam) for testability.
  - `_try_rewrite_path_string` uses the existing pure `_is_lexically_contained` / `_relative_parts` helpers (exact-prefix containment, separator- and case-folded) — a string that merely *contains* the old-root token (issue title) or a sibling prefix (`<src>-backup`) is not a hit.
  - A missing `state.json` is not an error (fresh tree, nothing to rewrite): `ok=True, rewritten=0`.
  - A rewrite failure (a hit whose rewritten target does not exist, a lock timeout, an `OSError`) returns `ok=False` with `moved` listing what was moved — the children have already moved, so this is an incomplete migration needing manual attention, not a rollback point. `StateLockBusy` and `OSError` are caught and returned as values, never raised (CLAUDE.md invariant). The `except StateLockBusy` and `except OSError` branches in `_rewrite_state_json_paths` are covered by `test_rewrite_state_json_paths_returns_ok_false_on_state_lock_busy` and `test_rewrite_state_json_paths_returns_ok_false_on_oserror` (added in rework), which force each branch via monkeypatching and assert the `ok=False`/error-set contract.
- **`cli.py`** — `run_migrate_state_dir_command` now reports `rewritten_paths` in both the message (`"moved N children, rewrote M embedded paths"`) and the result `data` dict.

This removes the need for both the compat junction and the manual E2 gate, making "migrated" mean the same thing on disk and in state.

### Invariant enumeration — every exit path of `apply_state_dir_migration`

The new invariant is *"after a successful `apply_state_dir_migration`, embedded paths in state.json name the new root"*. Every `return`/`raise` between the rewrite and the end of the function:

1. **Pre-flight refusal (`plan.ok is False`)** — returns before any move or rewrite. No rewrite needed; nothing moved. ✓
2. **Pre-flight refusal (`plan.blocked` non-empty)** — returns before any move or rewrite. No rewrite needed; nothing moved. ✓
3. **`dst_root.mkdir` `OSError`** — returns before any move or rewrite. No rewrite needed; nothing moved. ✓
4. **Containment re-check failure** — returns `ok=False` with `moved` prefix; rewrite not reached. Children may have moved, but the outcome is `ok=False` so the caller knows the migration is incomplete. The rewrite is only run on the all-children-moved path. ✓
5. **Source-vanished TOCTOU** — same as #4. ✓
6. **Destination-appeared TOCTOU** — same as #4. ✓
7. **`mover` `OSError`** — same as #4. ✓
8. **Rewrite failure (`rewrite_result.ok is False`)** — returns `ok=False` with `moved` listing what moved and `rewritten_paths=0`. The caller knows the migration is incomplete. ✓
9. **Full success** — returns `ok=True` with `rewritten_paths` set; the rewrite ran under `state_lock` and saved atomically. ✓

The rewrite is only attempted after the move loop completes without aborting, so it never runs against a partially-moved tree. A failed rewrite does not roll back the moves (the children are already on the new root); it reports `ok=False` so the operator knows manual attention is needed.

## Verification

```
uv run --extra dev pytest tests/test_state_dir_migration.py tests/test_cli.py -q --tb=short
........................................................................ [ 48%]
........................................................................ [ 96%]
......                                                                   [100%]
150 passed
```

```
uv run ruff check .
All checks passed!

uv run ruff format --check .
191 files already formatted
```

### Mutation check

#### Round 1 — rewrite call block

Reverted ONLY the fix — the rewrite call block at the end of `apply_state_dir_migration` in `src/charlie_work/state_migration.py` (the `state_path = plan.dst_root / layout.STATE_FILENAME` block through the final `return`) — to its merge-base version (`return MigrationOutcome(ok=True, moved=tuple(moved))`), keeping the new dataclasses and helpers in place.

Run against the reverted code:
```
uv run --extra dev pytest tests/test_state_dir_migration.py -q --tb=short -k "test_apply_rewrites_embedded_paths_in_state_json or test_apply_rewrite_failure_returns_ok_false_with_moved or test_apply_uses_injected_state_rewriter_seam or test_apply_injected_state_rewriter_failure_propagates"
F..FFF                                                                   [100%]
FAILED tests/test_state_dir_migration.py::test_apply_rewrites_embedded_paths_in_state_json - assert 0 == 8
FAILED tests/test_state_dir_migration.py::test_apply_rewrite_failure_returns_ok_false_with_moved - assert True is False
FAILED tests/test_state_dir_migration.py::test_apply_uses_injected_state_rewriter_seam - assert 0 == 5
FAILED tests/test_state_dir_migration.py::test_apply_injected_state_rewriter_failure_propagates - assert True is False
```

Restored the fix and re-ran:
```
uv run --extra dev pytest tests/test_state_dir_migration.py -q --tb=short
........................................................                 [100%]
56 passed
```

The 2 tests that pass against the reverted code (`test_apply_no_state_json_reports_zero_rewrites`, `test_apply_state_json_with_no_embedded_paths_reports_zero`) test the "0 rewrites" case, which is identical with or without the fix — they are not claimed as mutation-check coverage.

#### Round 2 — exception-handling branches in `_rewrite_state_json_paths` (rework)

Reverted ONLY the `except StateLockBusy` / `except OSError` handlers in `_rewrite_state_json_paths` (`src/charlie_work/state_migration.py`, lines 689-692) — replaced the `try`/`except` block with a bare `with state_lock(...):` body so exceptions propagate instead of being caught and returned as `ok=False` values. The two new tests force each branch via monkeypatching `state_lock` (raises `StateLockBusy`) and `load_state` (raises `OSError`).

Run against the reverted code:
```
uv run --extra dev pytest tests/test_state_dir_migration.py::test_rewrite_state_json_paths_returns_ok_false_on_state_lock_busy tests/test_state_dir_migration.py::test_rewrite_state_json_paths_returns_ok_false_on_oserror -q --tb=short
FF                                                                       [100%]
FAILED tests/test_state_dir_migration.py::test_rewrite_state_json_paths_returns_ok_false_on_state_lock_busy - charlie_work.state.StateLockBusy: lock held by another process
FAILED tests/test_state_dir_migration.py::test_rewrite_state_json_paths_returns_ok_false_on_oserror - OSError: disk I/O error
```

Restored the fix and re-ran:
```
uv run --extra dev pytest tests/test_state_dir_migration.py -q --tb=short
............................................................             [100%]
60 passed
```

## Risks / uncertain areas

- **Rewrite failure leaves an incomplete migration.** If the rewrite fails (a non-path hit, a missing target, a lock timeout), the children have already moved but `state.json` still points at the old root. The outcome is `ok=False` with `moved` listing what moved, so the operator knows manual attention is needed. This is deliberate — rolling back the moves would be more dangerous (re-introducing the partial-migration hazard the all-or-nothing design exists to prevent). The remediation script run against job-cannon on 2026-07-30 is the natural basis for a manual fix.
- **Existence check is filesystem-bound.** `_try_rewrite_path_string` verifies the rewritten target exists on disk. This is correct after a successful all-or-nothing migration (every file that was under the old root is now under the new one), but it means the rewrite cannot run against a tree where some files were not moved. That is by design — a hit whose target does not exist is either a non-path or an unmoved path, both of which must be refused rather than silently skipped.
- **No change to the compat junction behavior.** The issue body's "Also worth fixing" note about the quiesce gate runbook is a documentation fix, not a code fix, and is out of scope for this PR (the issue's "Proposed fix" section is the code change; the runbook note is filed separately). The embedded-path rewrite removes the *need* for the junction, but the junction itself remains a separate operator concern, as documented in the `apply_state_dir_migration` docstring.
