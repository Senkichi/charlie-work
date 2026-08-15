## Linked issue

Closes #1180

## What changed

**Fast-follow to #1176 (PR #1176 is open but not yet merged — this branch is based on #1176's head `e072ec4`). The diff includes #1176's commits until #1176 merges; once it merges, a rebase will shrink the diff to just the #1180 fix.**

Both `verify_shared_venv` and `_repair_venv_pth`'s poisoned-detection pass checked each `.pth` path line against **any** configured root rather than the specific root inferred from that `.pth`'s package. A poisoned `_editable_impl_ci_fleet.pth` repointed at `charlie-work/src` (a configured root, but the wrong one for `ci_fleet`) would pass the "any root" check and produce a silent `ImportError` at import time — the same class of false-green that #969 gap 2 was about, just with a different anchor.

### Changes

1. **`_match_pth_to_root` moved from `supervise.py` to `worktree.py`** — `verify_shared_venv` (in worktree.py) needs to call it, and worktree.py cannot import from supervise.py (supervise imports from worktree, not vice versa). The function is unchanged; only its location moved. `supervise.py` now calls `worktree._match_pth_to_root`.

2. **`verify_shared_venv` (worktree.py)** — For each `.pth`, derives the expected root via `_match_pth_to_root`. When a specific root is derivable, each path line must resolve into THAT root (not just any). When `_match_pth_to_root` returns `None` (unknown package), falls back to the "any configured root" check — a line outside all roots is still poisoned, and "any" is the tightest provable bound.

3. **`_repair_venv_pth` (supervise.py)** — Both the detection pass and the rewrite pass are tightened:
   - **Detection**: When a specific root is derivable, a line resolving into a DIFFERENT configured root is flagged as poisoned. When unknown, falls back to "any root".
   - **Rewrite**: Lines that don't resolve into `correct_root` are rewritten (previously only lines outside ALL roots were rewritten). This is necessary for the repair to actually work when detection catches a cross-root poisoning — without this, the repair would detect the poisoning but fail to rewrite the wrong-root line.

4. **Tests** — Three new regression tests covering the cross-root false-green scenario, plus three existing tests updated to match the new error message text.

### Invariant enumeration

Every exit path from `verify_shared_venv` between the per-package root check and the function's end:

- `return False, "could not locate site-packages..."` — before the root check; no invariant applies.
- `return _verify_shared_venv_by_import(...)` — fallback when no roots; no per-package check applies.
- `return False, "points outside its expected checkout..."` — specific root derivable, line outside it: **poisoned detected** ✓
- `return False, "points outside all configured checkouts..."` — unknown package, line outside all roots: **poisoned detected** ✓
- `return True, "..."` — all lines validated: **clean** ✓

Every exit path from `_repair_venv_pth`'s per-file loop:

- `poisoned = False → continue` — no poisoned lines found: **skip** ✓
- `correct_root is None → unrepairable.append` — poisoned but root unknown: **refuse to repair** ✓
- `not changed → continue` — all lines already correct: **skip** ✓
- `OSError → return False` — write failure: **error returned as value** ✓
- `repaired_files.append` — successfully rewritten: **repaired** ✓

## Verification

### Rework: merge conflict resolution

The PR branch had a merge conflict with `main`@`5ccbfa4` in two files:

- **`PR_BODY.md`** (add/add) — origin/main had a leftover PR body from a different PR (#735). Resolved by keeping this PR's body (HEAD side).
- **`src/charlie_work/instrumentation.py`** (content) — HEAD added `"venv_pth_repair_failed": "error"` to the `_LEVEL_BY_KIND` registry; origin/main added `"venv_editable_anchor_violation": "error"` (from the new `venv_anchor.py` startup guard). Both are legitimate new error-level kinds; resolved by keeping both, in alphabetical order.

All other files (supervise.py, worktree.py, test files) auto-merged cleanly. The per-package root logic in `verify_shared_venv` and `_repair_venv_pth` is intact after the merge.

### Targeted tests (changed test files + modules touched)

```
uv run --extra dev pytest tests/test_supervise.py -q --tb=short --timeout=30 -k "not test_self_deploy_end_to_end_repairs_lossless_blocker_and_retries_pull"
```
Result: 97 passed, 1 deselected (exit 0)

Note: `test_self_deploy_end_to_end_repairs_lossless_blocker_and_retries_pull` was deselected because it hangs on a real `git pull` subprocess in this environment — a pre-existing issue from main, not related to this PR's changes.

```
uv run --extra dev pytest tests/test_worktree.py -q --tb=short --timeout=30 -k "verify_shared_venv or cross_root or wrong_root or peer_root or match_pth"
```
Result: 8 passed, 216 deselected (exit 0)

```
uv run --extra dev pytest tests/test_cli.py tests/test_fleet_dispatch.py -q --tb=short --timeout=30
```
Result: 253 passed (exit 0)

```
uv run --extra dev pytest tests/test_instrumentation.py -q --tb=short --timeout=30
```
Result: 68 passed (exit 0)

```
uv run --extra dev pytest tests/test_venv_anchor.py -q --tb=short --timeout=30
```
Result: 10 passed (exit 0) — new test file from main, verified no interaction with this PR's changes.

### Mutation check

Reverted ONLY the #1180 fix — the per-package root validation in `verify_shared_venv` (worktree.py) and `_repair_venv_pth` detection+rewrite (supervise.py) — to the pre-fix "any root" logic (keeping `_match_pth_to_root` in worktree.py so imports work):

**Reverted code — tests MUST FAIL:**

```
uv run --extra dev pytest tests/test_worktree.py -q --tb=short --timeout=30 -k "cross_root or wrong_root or peer_root"
```
Result:
```
FF                                                                       [100%]
FAILED tests/test_worktree.py::test_verify_shared_venv_catches_foreign_editable_repointed_at_wrong_root
FAILED tests/test_worktree.py::test_verify_shared_venv_catches_main_editable_repointed_at_peer_root
```

```
uv run --extra dev pytest tests/test_supervise.py -q --tb=short --timeout=30 -k "cross_root"
```
Result:
```
F                                                                        [100%]
FAILED tests/test_supervise.py::test_repair_venv_pth_rewrites_cross_root_poisoned_editable
```

**Restored fix — tests MUST PASS:**

```
uv run --extra dev pytest tests/test_worktree.py -q --tb=short --timeout=30 -k "cross_root or wrong_root or peer_root"
```
Result: `..` (2 passed, exit 0)

```
uv run --extra dev pytest tests/test_supervise.py -q --tb=short --timeout=30 -k "cross_root"
```
Result: `.` (1 passed, exit 0)

### Lint and format

```
uv run ruff check .
```
Result: `All checks passed!`

```
uv run ruff format --check .
```
Result: `191 files already formatted`

## Risks / uncertain areas

- **Stacked PR**: This branch is based on #1176's head (`e072ec4`), not `origin/main`, because #1176 is not yet merged. The PR diff includes #1176's commits. Once #1176 merges, this branch should be rebased onto `main` and the diff will shrink to just the #1180 fix. The orchestrator should handle the merge ordering.
- **Message text change**: The error message for the per-package-root case changed from "points outside all configured checkouts" to "points outside its expected checkout". Three existing tests were updated. Any downstream consumer parsing this message text would need updating, but these are diagnostic messages not API contracts.
- **Rewrite pass tightening**: The rewrite pass now rewrites lines that don't resolve into `correct_root` (previously only lines outside all roots). This is necessary for the repair to work end-to-end when detection catches a cross-root poisoning. Without it, the repair would detect but fail to fix, leaving the venv in a broken state.
