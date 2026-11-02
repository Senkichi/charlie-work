# The GitHub transport takes generic requests, and HTTP is the default adapter

Accepted 2026-09-29; implemented (architecture review candidate 7).

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
- Spend accounting (#2439) extends `RateBudgetHolder` rather than adding a second mechanism. Each observed response adds one request, and the `x-ratelimit-used` delta in points, to a per-(capability, resource) ledger. The delta is taken against a process-wide cursor, so concurrent lanes' points sum to the observed counter movement. `gh api rate_limit` is never used for this: it alternates between two counters for one token. A capability is named by an ambient `capability_scope` (collaborator class, or `gh <noun> <verb>` for the legacy shim) and otherwise derived from the request. Fleet lane passes and reap sweeps drain it into one `github_budget_pass` event each, and a primary rate limit emits `github_rate_limited` once per exhausted window.
- The budget governor (#2442) extends the same holder. The observed budget is shared per token (a fingerprint, never the token) across clients and processes through `github-budget.json` in the fleet dir, written with `write_json_atomic` and refreshed from response headers; a snapshot from a past window is ignored. One lane-to-tier mapping (`github_transport/governor.py`) decides which lanes defer as the `graphql` window drains: reconcile, drift and review reaps below 1500, dispatch scans and probes below 800, everything else gated below 300, and never merge, label writes or the unauthorized-merge tripwire. A primary rate limit with a known reset waits for it (bounded) instead of the 1s/2s/4s retries. `runtime.github_budget_governor: false` restores the per-client budget, plain retries and the single threshold.
