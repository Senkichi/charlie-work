## Linked issue

Closes #1297

## What changed

`tests/test_state_migration.py::test_load_state_retries_transient_oserror` loosened its exact-count assert from `assert len(calls) == 2` to `assert len(calls) >= 2`.

The test globally patches `pathlib.Path.open`; its `flaky_open` raises `OSError` on call 1 only and delegates every later call to the real `Path.open`. It then asserted the retry loop opened the file **exactly twice** (1 simulated failure + 1 success). On a shared Windows CI box the *real* opens can themselves hit a genuine transient sharing violation — the precise condition `load_state`'s retry exists to absorb. The retry loop correctly rode through it (4 attempts), the behavioral asserts all passed (correct loaded content, zero `state.json.corrupt-*` quarantine files), and the exact-count assert failed anyway.

The assert conflated "the retry path fired" (the test's stated purpose, proven by call 1 raising + successful load) with "no other transient fault occurred on the host during the test" (not a property of the code under test, and not controllable on a shared Windows box).

The content assert (`loaded["issues"]["1"]["status"] == "ok"`) and the no-quarantine assert (`list(tmp_path.glob("state.json.corrupt-*")) == []`) stay as-is and carry the behavioral weight. No upper bound cap is introduced — any cap re-imports host-dependence.

No production code changed; `state.py` is untouched.

## Verification

```
uv run --extra dev pytest tests/test_state_migration.py::test_load_state_retries_transient_oserror -q --tb=short
```
Output:
```
Using CPython 3.13.5
.
```
(1 passed)

Full test file (impacted module is `state`, touched file is `test_state_migration.py`):
```
uv run --extra dev pytest tests/test_state_migration.py -q --tb=short
```
Output:
```
..................                                                       [100%]
```
(18 passed)

This is a test-only change with no public function signature/return shape/exception type/DB schema/module re-export change, so the targeted command above is the correct scope per the execution contract (CI runs the full suite on push).

```
uv run ruff check .
```
Output:
```
All checks passed!
```

```
uv run ruff format --check .
```
Output:
```
273 files already formatted
```

No pre-commit config exists in this repo.

### Mutation check

This is a test-only flake fix — there is no production artifact to revert and mutate. The "fixed artifact" is the test assertion itself. Reverting it to `== 2` and running locally cannot reproduce the flake (the flake requires a genuine transient OSError from the real `Path.open` on a shared host, which does not occur on this single-user local run), so the test would pass against the reverted assertion — the mutation check fails to fail by construction, because the bug is host-condition-dependent, not a code-path defect.

The issue body explicitly classifies this as a host-condition-sensitive test (same class as #1292): the production code (`load_state`'s retry loop) is correct and was never broken. The fix removes an assertion that asserted a property of the host, not of the code under test.

## Risks / uncertain areas

None. The change only loosens an over-strict assert in a single test; it cannot affect production behavior. The behavioral asserts (correct content loaded, no quarantine files) remain and still prove the retry path works.
