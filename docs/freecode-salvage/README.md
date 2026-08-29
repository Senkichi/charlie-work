# freecode salvage — specs, plans, and frozen reference code

**freecode** (a private, now-archived sibling repo) was archived 2026-07-09 after a two-round,
fact-checked evaluation concluded both of its theses were dead: free-tier API arbitrage
priced out at ~$20/month against a $0.10/M-token paid floor, and its compliance-first
architecture was the structural negation of fleet-worker requirements. Full evidence:
freecode `docs/evaluation-2026-07/` (13 documents) and `docs/adr/027-project-archived.md`.

Five components survived the evaluation as genuinely valuable to **this** repo. This
directory holds one spec per component plus verbatim frozen copies of the portable source
(`reference/`, with `PROVENANCE.md`). Nothing here is wired into `charlie_work` — each spec
states its own trigger for when the port becomes worth doing.

## Ranked salvage plan

| # | Component | Spec | Effort | When to act |
|---|---|---|---|---|
| 1 | Decision-event audit trail (schema-versioned, redacting, HMAC-chainable JSONL) | [01](01-decision-event-audit-trail.md) | ~hours | Next time a dispatch/verdict/merge decision is hard to reconstruct post-hoc — or opportunistically with any observability work |
| 2 | AST structural-guard test pattern | [03](03-structural-guard-tests.md) | ~1 day | Now-ish: CLAUDE.md invariants ("adapters never block", atomic writes) are currently prose-only |
| 3 | Keychain-first secret handling (incl. Windows Credential Manager fix) | [02](02-keychain-secrets.md) | ~hours | When charlie-work first holds a long-lived secret beyond ambient `GITHUB_TOKEN`/CLI auth |
| 4 | Cheap-token routing for secondary fleet roles (LiteLLM config, not code) | [05](05-cheap-token-routing.md) | ~hours (config) | When adding a second cross-family reviewer, janitor-assist, or triage role |
| 5 | Session-level admission control / quota-learning (`capacity.py` seed) | [04](04-session-admission-control.md) | days | **Trigger-conditioned:** only when concurrent workers contend on a shared rate-limited credential and `fleet.global_max_concurrent_sessions` stops being a good-enough throttle |

## What was deliberately NOT salvaged

- **Tool/CLI wrappers** (AiderTool/GeminiCliTool/CodexTool/…): charlie-work's own
  `claude_code.py` adapter is better-fitted (non-blocking, PID-recycle-safe liveness,
  stream-json cost telemetry) and battle-run; freecode's wrappers were blocking and
  interactive-first by ADR-locked design.
- **Quota HTTP adapters** (7 providers): no landing spot — charlie-work drives worker CLIs
  directly, not through a proxy.
- **The catalog/ack/evidence governance machinery**: right discipline for freecode's threat
  model, wrong weight class here. If a compliance question arises for a new worker CLI,
  *read* freecode's ADRs 006C/006D/006E — the credential-surface research (which files each
  CLI touches, what's safe to inject vs stat-only) is done and citable.

## Cross-cutting facts worth remembering (fact-checked 2026-07-09)

- Anthropic consumer terms prohibit automated non-human subscription-OAuth access to Claude
  Code — fleet workers should authenticate via API key / Console billing, not a personal
  Pro/Max OAuth session. (Verify how current workers are authed.)
- charlie-work's RUNAWAY cost tripwire depends on `claude -p` stream-json; any non-Claude
  adapter inherits the weaker SLOW-capped supervision ceiling (see spec 05 for implications).
- Best open models trail frontier by ~25 pts on Terminal-Bench 2.1 (~58.7% GLM-5.1 vs ~83.4%
  GPT-5.5) and ~15 pts on SWE-bench-Verified — cheap workers export cost into
  `review.max_rework_cycles`; use them for bounded secondary roles, not primary issue→PR work.
