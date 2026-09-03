## Linked issue

Closes #1541

## What changed

Added the AST-equivalence gate for verbatim symbol relocations (Track-1
precondition #4 of 4, llibrary
docs/plans/2026-09-02-god-object-paydown-DECISION.md section 7 row 4).

### New modules

- `src/charlie_work/ast_equivalence_gate.py` (~340 LOC): the core gate.
  - `extract_symbols`: AST symbol extraction with
    `ast.dump(node, include_attributes=False)` for top-level functions,
    classes, and class methods.
  - `derive_moved_symbols`: the diff-derived moved-symbol set (graft E,
    rule #9). Computes `base-not-head(file A)` ∩ `head-not-base(file B)`,
    matched by bare name and verified by `ast.dump` equality. No hardcoded
    symbol-name list anywhere in the source (verified by
    `test_no_hardcoded_symbol_name_list_in_gate_source`).
  - `generate_pep562_shim_source`: generates a module-level `__getattr__`
    (PEP 562) facade, or a class-level `__getattr__` facade for class
    members, from the same diff-derived set.
  - `find_stale_facade_shims`: the vulture "forgotten facade" sweep --
    flags re-export entries nobody imports any more.
  - `render_review_packet`: renders the gate's findings as a review packet
    (evidence, NOT enforcement -- graft C, #1538).

- `src/charlie_work/ast_equivalence_gate_command.py` (~200 LOC): CLI
  command layer following the `private_slug_check_command.py` pattern.
  Owns argparse subparser registration, `git diff`/`git show` subprocess
  calls, and the exit-code decision. The gate **always returns `ok=True`**
  (evidence, not enforcement).

### Modified files

- `src/charlie_work/cli.py`: imports + subparser registration + dispatch
  for `charlie ast-equivalence-check --base <ref> [--shim-file FILE]
  [--output FILE]`.
- `.github/workflows/ci.yml`: added an "AST-equivalence gate (issue 1541)"
  step to the existing "Lint" job (same pattern as mojibake/private-slug
  gates -- rides the already-required "Lint" context, no new job, no
  required_checks change). The step always exits 0 and writes the review
  packet to `$GITHUB_STEP_SUMMARY`.

### New tests

- `tests/test_ast_equivalence_gate.py` (26 tests):
  - `extract_symbols`: top-level functions, classes+methods, async
    functions, empty files.
  - `derive_moved_symbols`: verbatim move (equivalent), modified move
    (non-equivalent), no-move cases, class method move between classes,
    multiple symbols.
  - Rule #9 compliance: `test_no_hardcoded_symbol_name_list_in_gate_source`
    AST-scans the gate's own source for hardcoded symbol-name lists.
  - `generate_pep562_shim_source`: module-level shim, class-level shim,
    no-moves case, and `test_pep562_shim_resolves_old_and_new_import_paths`
    (acceptance: both old and new import paths resolve to the same object
    post-move).
  - `find_stale_facade_shims`: stale shim detected, no stale shims when
    all imported, `parse_shim_mapping` extraction.
  - `render_review_packet`: equivalent move, non-equivalent move, stale
    shims, no moves.
  - CLI command: verbatim move detection via mocked git, always-ok
    behavior (evidence not enforcement).

## Verification

### Tests

```
uv run --extra dev pytest tests/test_ast_equivalence_gate.py --tb=short
```

Output:
```
..........................                                               [100%]
26 passed in 0.18s
```

### Existing tests (no regressions)

```
uv run --extra dev pytest tests/test_cli.py tests/test_mojibake_gate.py tests/test_private_slug_gate.py --tb=short
```

Output:
```
166 passed in 8.23s
```

### Lint + format

```
uv run ruff check .
uv run ruff format --check .
```

Output:
```
All checks passed!
347 files already formatted
```

### Mutation checks

Each regression test was mutation-checked by reverting the specific
function under test and verifying the test fails:

**Mutation 1: `derive_moved_symbols` -> always returns `[]`**

Reverted: `src/charlie_work/ast_equivalence_gate.py`,
`derive_moved_symbols` (added `return moved` before the reverse-index
build).

```
uv run --extra dev pytest tests/test_ast_equivalence_gate.py::test_verbatim_move_is_detected_as_equivalent tests/test_ast_equivalence_gate.py::test_non_equivalent_move_is_flagged tests/test_ast_equivalence_gate.py::test_multiple_symbols_moved --tb=short
```

Output:
```
FFF                                                                      [100%]
FAILED test_verbatim_move_is_detected_as_equivalent - assert 0 == 1
FAILED test_non_equivalent_move_is_flagged - assert 0 == 1
FAILED test_multiple_symbols_moved - assert set() == {'func_a', 'func_b'}
3 failed in 0.19s
```

**Mutation 2: `find_stale_facade_shims` -> always returns `[]`**

Reverted: `src/charlie_work/ast_equivalence_gate.py`,
`find_stale_facade_shims` (added `return []` before the shim-name parse).

```
uv run --extra dev pytest tests/test_ast_equivalence_gate.py::test_stale_shim_detected_when_nobody_imports_it --tb=short
```

Output:
```
F                                                                        [100%]
FAILED test_stale_shim_detected_when_nobody_imports_it - assert 'stale_name' in set()
1 failed in 0.18s
```

**Mutation 3: `generate_pep562_shim_source` -> skip module shim generation**

Reverted: `src/charlie_work/ast_equivalence_gate.py`,
`generate_pep562_shim_source` (replaced `if module_shims:` body with
`pass`, guarded the real body behind `if False:`).

```
uv run --extra dev pytest tests/test_ast_equivalence_gate.py::test_generate_module_level_shim tests/test_ast_equivalence_gate.py::test_pep562_shim_resolves_old_and_new_import_paths --tb=short
```

Output:
```
FF                                                                       [100%]
FAILED test_generate_module_level_shim - assert 'def __getattr__(name):' in '"""PEP 562 ...'
FAILED test_pep562_shim_resolves_old_and_new_import_paths - AttributeError: module 'shim_test_pkg.old_module' has no attribute 'relocated_func'
2 failed in 0.19s
```

All mutations restored after verification.

## Invariant enumeration

The gate command (`run_ast_equivalence_check_command`) has the following
exit paths, all of which satisfy the "always ok=True (evidence, not
enforcement)" invariant:

1. `git diff --name-only` failure -> `return CommandResult(False, ...)` --
   this is the only `ok=False` path, reserved for git infrastructure
   failures (the gate cannot run). It is NOT a non-equivalent-move
   finding.
2. No moves, no stale shims -> `return CommandResult(True, ...)` with
   empty `moved_symbols` and `stale_shims`.
3. Equivalent moves only -> `return CommandResult(True, ...)` with
   `equivalent=True` entries.
4. Non-equivalent moves -> `return CommandResult(True, ...)` with
   `equivalent=False` entries. **The gate does NOT fail on
   non-equivalent moves** -- they are flagged in the review packet for
   the human reviewer, but the gate exits ok=True (evidence, not
   enforcement, per graft C).
5. Stale shims found -> `return CommandResult(True, ...)` with
   `stale_shims` entries. Same rationale: evidence, not enforcement.

## Risks / uncertain areas

- **vulture package not used**: the issue names a "vulture sweep" but the
  `vulture` tool itself cannot detect stale `__getattr__` shim entries
  (they are dynamically resolved via PEP 562, not statically defined).
  The sweep uses custom AST-based import scanning instead, which is what
  vulture would do if it could see dynamic `__getattr__` entries. The
  functionality (flag re-export entries nobody imports) is fully
  implemented; only the tool choice differs from the literal issue text.

- **Self-proving requirement (graft G)**: the issue says "the first thing
  this gate verifies is the verbatim move of `attachment_contracts` into
  its own distribution (#1544)." Issue #1544 (the package relocation) is
  not yet done -- it is a separate issue. The gate's test suite includes
  a verbatim-move test case (`test_verbatim_move_is_detected_as_equivalent`)
  that proves the gate CAN verify a verbatim relocation. When #1544 runs
  the actual relocation, this gate will verify it in CI.

- **CI step is advisory**: the gate always exits 0. Enforcement stays in
  `required_checks` (#1538). A human merging directly to `main` bypasses
  this evidence (same gap as #1537 for all CI gates on this repo).

- **Class-level `__getattr__` shims are generated but not wired**: the
  shim generator produces `_cls_getattr_{ClassName}` functions, but
  wiring them onto the actual class (e.g., `OldClass.__getattr__ =
  _cls_getattr_OldClass`) is the consumer's job (the PR that moves the
  class member). The gate generates the facade source; the consumer
  integrates it.
