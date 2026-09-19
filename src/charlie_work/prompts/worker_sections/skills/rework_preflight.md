- Before pushing, run `/preflight` (ruff + ruff-format + pre-commit) and COMMIT anything
  it fixes. Pushing to an existing PR is gated the same as opening one — a CI-dirty tree
  (uncommitted reflow, un-normalized fixture) will be blocked.
