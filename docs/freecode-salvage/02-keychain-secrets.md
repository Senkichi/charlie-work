# 02 — Keychain-first secret handling

## What it is

`reference/keys.py` (~180 LOC): OS-keyring-backed secret storage/retrieval with no plaintext
fallback, plus the part worth preserving verbatim — **the Windows Credential Manager
compound-vs-bare lookup-order fix** (`keys.py:85-113` in source numbering): Windows stores
`keyring` entries under a compound `service:username` target in some backends and a bare
`service` target in others; naive `get_password` calls miss one of them. The fix probes both
orders deterministically and documents which form wins.

## Why charlie-work wants it (eventually)

Today charlie-work mostly rides ambient auth (`gh` CLI token, worker CLIs' own auth) and has
no secret store of its own. That changes the first time it holds a long-lived secret
directly, e.g.:

- an API key for a cheap-token secondary role (spec 05) passed into a worker/reviewer env,
- a webhook signing secret for the `notify` layer,
- per-repo tokens for the fleet registry when operating beyond one GitHub identity.

When that happens, the house answer should already be decided: **OS keyring only, no
plaintext `.env` fallback, secrets injected into child env at spawn** (matching how
`env_sanitize.py` already treats the child env as a curated surface).

## Port plan (~hours)

1. Copy into `src/charlie_work/secrets.py`; drop the optional `DecisionEvent` emit or wire it
   to the ported audit trail (spec 01) if that landed first.
2. Add `keyring` as a dependency **only at this point** — do not pre-add it speculatively.
3. Naming convention: service `"charlie-work"`, username = logical slot (e.g.
   `OPENROUTER_API_KEY`, `NOTIFY_WEBHOOK_SECRET`). Document slots in the config reference.
4. Spawn-time injection only: secrets go into the adapter's child `env` dict, never into
   config files, prompts, state.json, or audit payloads. A structural guard (spec 03) can
   enforce "no `keyring.get_password` calls outside `secrets.py`" — freecode used exactly
   this choke-point pattern.

## Non-goals

Do not port freecode's plaintext-with-explicit-flag escape hatch; charlie-work has no
interactive consent surface to make that auditable, and no current need.
