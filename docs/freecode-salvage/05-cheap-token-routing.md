# 05 — Cheap-token routing for secondary fleet roles (config, not code)

## The surviving 90% of freecode's original thesis

freecode's founding idea — aggregate free-tier capacity behind a router — priced out at
~$20–22/month of equivalent value (fact-checked 2026-07-09). What survives is the adjacent,
simpler play: **capable open models are now nearly free to buy** (verified July 2026:
DeepSeek-V4-Flash $0.09/$0.18 per M tokens via OpenRouter/DeepInfra; GLM-4.7-FlashX
$0.07/$0.40; GLM-4.7-Flash $0 at z.ai), so secondary fleet roles can run on them for cents —
**with one paid API key and a ~100-line LiteLLM config**, not a routing codebase. 2026
LiteLLM natively provides fallbacks, 429 cooldowns, per-deployment RPM/TPM, session
affinity, and per-provider budget caps.

## Roles that fit (and one that doesn't)

| Role | Fit | Notes |
|---|---|---|
| Cross-family adversarial review (`cross_family.*`) | **Best fit** | Findings are leads, never merge gates — quality floor is acceptable by design, and model diversity (GLM/DeepSeek vs Claude) is the point. Adds a second family beyond the current codex-via-Devin pass. |
| Janitor-gate assist / PR triage summaries | Good | Bounded, low-stakes, high-volume. |
| Spec lint (`why-charlie-hate-spec` second opinion) | Good | Same leads-not-gates posture. |
| **Primary issue→PR workers** | **Poor — don't** | Best open models trail frontier ~25 pts on Terminal-Bench 2.1 (~58.7% vs ~83.4%); weak workers export cost into `review.max_rework_cycles` and human attention. Revisit only if open-model scores close the gap. |

## Implementation sketch (~hours, when a role needs it)

1. Run `litellm --config litellm-config.example.yaml` (sibling file) locally; point the
   consuming role at `http://localhost:4000/v1` with any OpenAI-compatible client or CLI
   (mini-swe-agent and opencode both speak custom OpenAI-compatible endpoints; aider does
   via `openai/` model prefixes).
2. For cross-family: add a second `cross_family` engine variant whose command template
   invokes the chosen harness against the local proxy, mirroring how the codex/Devin pass is
   configured today. Review packets already flow as files, so the integration surface is the
   command template.
3. Keys: one OpenRouter (or z.ai/DeepInfra) key covers all listed models; store per spec 02
   once ported, env-inject at spawn until then.
4. Supervision caveat: non-Claude CLIs don't emit `claude -p`-style stream-json, so the
   RUNAWAY cost tripwire won't see them — they inherit the SLOW-capped watchdog ceiling
   (same as Devin-shell today). Acceptable for bounded secondary roles; a reason NOT to use
   this lane for long-leash primary workers.
5. LiteLLM native knobs replace freecode code: `router_settings.fallbacks`,
   `cooldown_time`, per-deployment `rpm/tpm`, `max_budget` per provider. No custom routing
   logic unless spec 04's trigger fires.

## Price sanity anchors (verified 2026-07-09 — re-verify before relying)

- DeepSeek-V4-Flash: $0.09 in / $0.18 out per M (OpenRouter/DeepInfra; official API
  $0.14/$0.28).
- z.ai GLM-4.7-FlashX: $0.07/$0.40; GLM-4.7-Flash free tier; GLM-5.2 flagship $1.40/$4.40.
- Typical autonomous session burn (reported, wide variance): 300K–2M tokens → a
  cross-family review pass on cheap models costs ~$0.03–0.50.
- Flat-fee alternative at scale: z.ai GLM Coding Plan Max (~$112–160/mo, reported) beats
  per-token routing beyond ~50 sessions/day — irrelevant until the fleet is much larger.
