# Contributing

## Dev setup

```bash
uv sync --all-extras   # install all deps including the dev extra (pytest, ruff)
```

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

## Common commands

```bash
uv run --extra dev pytest -q --tb=short   # run tests
uv run ruff check .           # lint
uv run ruff format .          # format
```

## Commit format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>
```

Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

Examples:
- `feat: add cross-family adversarial review pass`
- `fix(state): quarantine corrupt state file instead of crashing`
- `docs: add CONTRIBUTING.md`

## Pull request rules

- **One issue per PR.** Each PR must link exactly one GitHub issue.
- **PR title format:** Conventional-Commits format (`type(scope): description`).
  Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
  The janitor gate enforces this format and warns on non-conventional titles.
- **PR body must include** `Closes #<N>` so GitHub closes the issue on merge.
- **Tests or rationale required.** The janitor gate checks the PR body for test
  evidence (`test`, `verified`, `verification`, …) or an explicit rationale for
  omitting tests. Include the exact commands run and their output.
- Fill in `.github/pull_request_template.md` when creating the PR.

## Citing code in issues

Issues often point a worker at a specific location in the codebase. A bare
`path:line` citation (`workflow.py:4746`) rots fast: `workflow.py` is ~19,000
lines and changes on most merges, so a line number filed one day can point at
unrelated code the next. A worker that lands on plausible-but-wrong code
infers "this was already fixed" and the defect survives — silently, looking
like diligence. This was measured (issue #1000): 6 of 8 dispatch-ready issues
needed citation correction, and a single ordinary merge invalidated citations
in 4 of 13 queued issues.

**Cite the symbol, not the line.** `_collect_external_findings` in `workflow.py`
is stable across every edit that does not rename it; `workflow.py:4746` is stale
the next time anything above it grows. A symbol citation cannot rot the way a
line number does.

**When a line number genuinely adds value** (pointing into the middle of a long
function), give it *alongside* the symbol and the commit it was read at:

```
_render_required_changes_section in workflow.py (≈:4960 @ 27ca3a5)
```

so a reader can tell a drifted citation from one that was always wrong.

**Stamp the commit.** Every issue that quotes code should record the sha it was
read against. Without it there is no way to distinguish "the citation drifted"
from "the citation was always wrong", and those need different responses.

A pre-dispatch check (`citation_check`) flags `path:line` citations whose
coordinates no longer match the working tree (file renamed/deleted, line out of
range, blank line) by posting a comment on the issue before a worker is sent to
it. The comment is visible to the worker, so it is not silently misled. The
check is a backstop — it cannot catch in-range content drift (a valid line
number now pointing at unrelated code), which is why the symbol convention
above is the durable fix. Flagging is non-blocking: the correction needs
judgment, so the check flags rather than auto-edits.
