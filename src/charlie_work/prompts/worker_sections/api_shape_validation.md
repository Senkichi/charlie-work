## API shape validation

1. Any code consuming an external API/library response must include evidence (in the PR body) of the real shape: a live call transcript (`gh api` / `curl`), a vendored fixture captured from the real system, or a doc link with version.
2. Any comment claiming behavior of a called function ('single-company', 'returns X') must be verified against that function's signature/docstring in the same session.
3. The janitor will warn if this PR adds a call to `gh api` or an external HTTP endpoint without a test fixture sourced from a live payload. Add the fixture under `tests/fixtures/` or include the evidence in the PR body.
