# Vendored gitleaks rule subset (issue #1505)

`secrets.toml` in this directory is a verbatim subset of gitleaks' built-in
rule set, consumed by `charlie_work.outbound_body_guard` to refuse outbound
PR/issue body writes that would persist a live credential into GitHub's
visible edit history.

## Pin

- Upstream: https://github.com/gitleaks/gitleaks
- Commit: `83d9cd684c87d95d656c1458ef04895a7f1cbd8e` (tag `v8.30.1`)
- Source file: `config/gitleaks.toml`
- Retrieved: 2026-10-06

## Selection rule

Every rule whose `id` starts with `github-`, `gitlab-`, or `slack-`, plus the
named singles `aws-access-token`, `private-key`, `openai-api-key`,
`anthropic-api-key`, `anthropic-admin-api-key`, `npm-access-token`,
`pypi-upload-token`, `stripe-access-token`, `jwt`, `jwt-base64` — the
token / PAT / OAuth / private-key families the issue's needs-design
resolution scoped to. All other upstream rules (generic API-key heuristics,
sidekiq secrets, DB connection strings, …) are deliberately excluded to keep
the false-positive surface small.

## Semantic deviations from upstream `gitleaks detect` (documented, deliberate)

1. **Text scan, not file scan.** Rule `allowlists[].paths` entries are inert —
   there is no file path to match against. `regexes` allowlists are applied to
   each regex match.
2. **`example-secret` fence exemption.** Content inside Markdown code fences
   annotated ` ```example-secret ` is masked before scanning. This carve-out
   exists so that deliberately-labelled *examples* of secret-shaped text
   (e.g. this issue's own discussion of the `gho_…` incident) do not deadlock
   the orchestrator's own comment path. Everything outside such fences,
   including ordinary fenced code blocks, is scanned — and an
   `example-secret` line *inside* an ordinary fence is literal content per
   CommonMark (a closing fence cannot carry an info string), so it does not
   open an exempt block.
3. **Keyword prefilter is case-insensitive** on both sides, matching upstream.

## Refresh procedure

1. Fetch the new `config/gitleaks.toml` at the chosen upstream commit.
2. Re-apply the selection rule above (or regenerate from a pinned upstream
   copy — the worktree that introduced this file kept one at
   `.var/gitleaks-upstream.toml`).
3. Update the pin block in `secrets.toml`'s header comment and this file.
4. Diff the refreshed rules' *fields*, not just their regexes: the loader in
   `outbound_body_guard._rules()` honors `id`, `regex`, `keywords`,
   `entropy`, `secretGroup`, and `allowlists[].regexes` only. If a refreshed
   rule adds `regexTarget`, `stopwords`, `condition`, `path`-gated
   allowlists, or any new field, the loader will silently ignore it — extend
   the loader (and this section) rather than shipping a divergent rule.
5. Run `uv run --extra dev pytest -q tests/test_outbound_body_guard.py`.
