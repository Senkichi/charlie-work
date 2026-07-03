## Linked issue

Closes #<!-- issue number -->

## What changed

<!-- Describe what was added, removed, or modified and why. -->

## Verification

<!-- Required: paste the exact commands run and their output.
     The janitor gate checks for test evidence or an explicit rationale. -->

```
uv run pytest -q --tb=short
# paste output here
```

```
uv run ruff check .
uv run ruff format --check .
# paste output here
```

## Risks / uncertain areas

<!-- Anything that could go wrong, edge cases not covered, or areas
     where a reviewer should pay extra attention. Write "None" if clean. -->
