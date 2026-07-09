# 03 — AST structural-guard test pattern

## What it is

freecode's cheapest, most transferable discipline: unit tests that parse a module with
`ast` and fail if a structural invariant drifts — e.g. scanning `ast.Call` nodes and
composite string literals to prove "this module never reads bytes from a credential path"
or "every `git worktree add` carries `-c core.symlinks=false`."
`reference/example_structural_guard_test.py` is a complete worked example (freecode's
claude-code credential-isolation guard).

The point: prose invariants drift silently; a guard test turns "we never do X" from a
CLAUDE.md sentence into a CI failure. In freecode these guards repeatedly caught real
regressions during parallel agent-driven development — directly relevant to how this repo
is developed (fleet workers writing PRs against it).

## Why charlie-work wants it now

`CLAUDE.md` already states invariants that are enforced only by review attention:

| Prose invariant (CLAUDE.md) | Guardable assertion |
|---|---|
| "Adapters must not block on worker completion" | In `claude_code.py` / `devin_shell.py` (and any new adapter): no `ast.Attribute` access of `.wait`/`.communicate` on the `Popen` result within the launch call path |
| "All JSON state writes are atomic" | No `open(path, "w")` / `write_text` targeting `state.json`-family paths outside the blessed `_write_json`/`save_state` helpers |
| "Never hard-code label strings" | No string literals matching `agent:*` in business-logic modules outside `config.LabelConfig` defaults |
| "Errors from external processes come back as values" | No `raise` inside `subprocess_runner.run_captured`'s non-timeout paths; launch functions return result objects |
| (future, spec 02) "Secrets only via secrets.py" | No `keyring.` calls outside `secrets.py` |

## Port plan (~1 day)

1. Add `tests/test_structural_guards.py` with one test per invariant above, following the
   exemplar's shape: `ast.parse` the target source, walk nodes, collect violations with
   line numbers, assert empty with a message quoting the CLAUDE.md sentence it enforces.
2. Keep each guard narrowly scoped to named modules (choke points), not repo-wide grep —
   freecode's lesson is that guards earn their keep at boundaries where the invariant is
   load-bearing, and stay cheap when they target one file each.
3. Reference each guard from the CLAUDE.md invariant it enforces ("enforced by
   `tests/test_structural_guards.py::test_adapters_never_block`") so worker agents editing
   an adapter see the contract and its teeth in one place.
4. Bypass requires editing the guard test in the same PR — which is exactly the visibility
   the pattern is designed to force.
