# The GitHub transport takes generic requests, and HTTP is the default adapter

Accepted 2026-09-29; implementation pending (architecture review candidate 7).

Everything below the capability collaborators talks to GitHub through one transport interface. It takes a generic request: a REST method and route with a body, or a GraphQL document with variables. There are two adapters behind it. The pooled HTTPS adapter is the default and is on. The `gh` CLI adapter renders the same request as `gh api …` and is used as a per-call fallback. Every fallback emits an event, and a config key turns HTTP off as a kill switch. Retry, the circuit breaker, dry-run suppression and the in-pass deadline wrap the transport once, so no path can reach GitHub without them.

## Considered Options

- **Keep the `gh` argv as the interface (status quo since #1841).** The HTTP adapter had to reverse-parse `gh` command lines (`http_translate.py`), so it covered only `gh api` GETs and read-only GraphQL. Every `--json` subcommand and every mutation fell back to `gh` by construction. `Transport.validate_field_lists` and `_resolve_token` spawned `gh` outside `run()` and bypassed the breaker.
- **Typed operations per capability (`GetPullRequest(n)`, …).** Rejected at this level because the interface would grow with every new call. Typed operations belong in the capability collaborators *above* the transport.
- **Parse the argv once, centrally.** Rejected because it keeps `gh`'s command language as the contract both adapters must speak.

## Consequences

- "Is this a mutation" is derived from the HTTP method or the GraphQL operation type, never from a hand-maintained list. Dry-run suppression follows from that.
- `gh pr view --json …`-style calls are rewritten as REST or GraphQL requests, so HTTP covers mutations as well as reads.
- The HTTP adapter keeps taking its token from `gh auth token`, resolved through the wrapped transport. Both adapters draw on the one rate-limit budget in `api_budget.py`, because two budgets draining the same quota would hide throttling.
- The `runtime=None → "gh"` default, which existed so tests patching `subprocess.run` kept working, is removed. Transport tests use a fake transport. Business-logic tests keep using `FakeGitHub` at the `GitHubLike` seam, which this ADR does not change.
