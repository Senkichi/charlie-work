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
