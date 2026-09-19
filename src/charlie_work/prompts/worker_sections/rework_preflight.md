- Before pushing, match CI locally — `ruff check`, `ruff format --check`, and, if the
  repository has a `.pre-commit-config.yaml`,
  `pre-commit run --files $(git diff --name-only origin/main...HEAD)` — and COMMIT anything
  they fix. Pushing to an existing PR is gated the same as opening one — a CI-dirty tree
  (uncommitted reflow, un-normalized fixture) will be blocked.
