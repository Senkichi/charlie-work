# API Worker Adapter — Design Spec

- **Date:** 2026-07-20
- **Status:** Approved design, pending implementation
- **Branch:** `senk/api-worker-architecture-b97087`
- **Trial provider:** Kimi K3 via Moonshot ($15 trial budget)
- **Update (2026-08-30):** the per-issue adapter-selection layer this design
  describes in §5 and table row `⑤` was later deleted by Phase 2 of the
  role-config refactor. §5 below has been updated to reflect the current
  mechanism; the rest of this document is left as a historical design record.

## 1. Motivation

The fleet currently has two worker adapters: `devin-shell` (free local swe-1.6
subprocesses — cheap, weak) and `claude-code`. Adapter selection is **global per
repo** (`devin.adapter`), so every issue in a repo gets the same worker
regardless of difficulty, and reworks go back to the same (weak) worker that
produced the flawed first pass.

This design adds a third adapter kind, **`api`**: a provider-agnostic,
paid-API-backed worker routed per-issue. Policy: Devin keeps simple first-pass
work; the API worker takes **reworks** (Phase 1) and **complex first-pass
issues** (Phase 2). This inverts the concern in
`docs/freecode-salvage/05-cheap-token-routing.md` — that doc rejected cheap open
models as *primary* workers versus frontier CLIs; here the API worker is the
*stronger* tier relative to local swe-1.6, applied exactly where the weak tier
demonstrably failed (reworks) or is expected to fail (complex issues).

## 2. Goals / Non-goals

**Goals**

1. Per-issue adapter routing with a single point of policy enforcement.
2. Provider-agnostic via config: adding a provider is a config edit, zero code.
3. Hard spend governance: daily and lifetime USD caps enforced before launch;
   per-session cap enforced in-flight once calibrated (§6).
4. Full reuse of existing supervision: worktree isolation, sidecar records,
   stream-json watchdog telemetry, dead-worker reconciliation.
5. Graceful degradation: API worker unavailable (budget, auth, outage) →
   fall back to `devin-shell`, recorded and auditable.

**Non-goals**

- OpenAI-protocol support / non-Anthropic-compatible endpoints. The harness is
  the Claude Code CLI; providers must expose an Anthropic-compatible endpoint
  (Moonshot, GLM, MiniMax, and vLLM deployments all do). Revisit only when a
  concretely wanted provider lacks one.
- A new agent runtime. No bespoke tool-loop harness.
- Automatic complexity *estimation* (LLM classifier or heuristics). Routing
  signals are deterministic: rework status and an explicit label.

## 3. Architecture

`"api"` joins `"devin"` and `"claude-code"` as an `adapter_kind`. It is the
Claude Code CLI launched with provider environment injected
(`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, model override), so the entire
existing launch/supervision stack carries over. New modules:

| Module | Responsibility | Size |
|---|---|---|
| `api_worker.py` | Resolve active provider → env dict; budget preflight; delegate to `claude_code.launch_claude_worker(adapter_kind="api", ...)` | ~150 lines |
| `api_budget.py` | Atomic-JSON spend ledger; session settlement from events.jsonl usage × provider pricing | small |

**Prerequisite refactor:** `claude_code.launch_claude_worker` gains
`adapter_kind: str = "claude-code"` controlling the sidecar suffix
(`issue-<n>.api.json`) plus a `provider: str` field on `ClaudeWorkerRecord`
(empty for plain claude-code). Defaults preserve current behavior exactly.

**Wiring points** (all follow the existing third-arm pattern):

- `adapters.py`: `elif adapter == "api"` branch in `dispatch_sessions` +
  `_run_api_adapter` helper (staggered, mirrors `_run_claude_code_adapter`).
- `worker.py`: `WorkerView.is_alive()` / `reap_sidecar()` /
  `update_worker_log_stat()` — `"api"` delegates to the claude-code paths;
  `iter_workers()` reads the `.api.json` sidecars.
- `workflow.py._adapter_settings()`: env/venv resolution for the api section.
- `doctor.py`: probe branch (§8).

Cost accounting note: Claude Code's self-reported dollar cost is wrong against
non-Anthropic endpoints; **token counts are correct**. The ledger computes USD
from events.jsonl token usage × configured provider pricing.
`tee_stream_json` is force-enabled for `api` sessions — the ledger and the
in-flight cap depend on it.

## 4. Provider registry (config)

New frozen dataclasses `ApiProviderConfig` and `ApiWorkerConfig` (with nested
`ApiBudgetConfig`), registered on `OrchestratorConfig` as `api_worker` and
merged through the layered global/per-repo config like every other section.

```yaml
api_worker:
  enabled: false            # default OFF everywhere; trial enables charlie-work only
  provider: kimi-k3         # active provider key
  max_concurrent_sessions: 1
  providers:
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic
      api_key_env: MOONSHOT_API_KEY     # env var NAME; the key never lives in config
      model: kimi-k3
      input_usd_per_mtok: 3.0
      output_usd_per_mtok: 15.0
      cached_input_usd_per_mtok: 0.30
  budget:
    max_usd_per_session: 0    # 0/unset = dormant; set after calibration (§6)
    preflight_reserve_usd: 1.00  # headroom estimate while per-session cap is unset
    max_usd_per_day: 5.00
    lifetime_usd: 15.00       # trial ceiling; raise/remove post-trial
  worker_template: worker_claude_code.md
  rework_template: rework.md
```

Validation at config load: `provider` must exist in `providers`; pricing fields
must be > 0 when budget enforcement is enabled; `api_key_env` must be a
non-empty name (presence of the env var itself is a doctor/runtime check, not a
load-time failure, so unrelated repos with the section absent are unaffected).

## 5. Routing policy (superseded)

Phase 2 of the role-config refactor (a separate, later project) deleted the
per-issue adapter-selection layer this section originally described — the
pure selection function, its frozen `AdapterChoice` return type, the
`rework=True` and `complexity:high`-label rules, the preflight checks, and the
fallback-to-a-different-adapter-on-failure chain, along with the
`adapter_history` state recording that made each choice auditable. None of
that exists anymore.

`api` is now selected purely via `worker.harness: api` (and, symmetrically,
`reviewer.harness: api` for review dispatch) — a whole-pass config choice,
not a per-issue policy decision made at dispatch time. Whichever harness the
config names is what every issue in that pass gets; there is no per-issue
rework-routes-to-api rule and no runtime fallback to a different harness when
a preflight-style check would have failed. See
[docs/RUNBOOK.md](../RUNBOOK.md#api-worker-operations) for the current
selection and budget-exhaustion behavior, including the open gap left by this
deletion: no pre-launch budget check currently exists to replace the one
described in §6 below.

## 6. Budget governance

Ledger file: `.var/charlie-work/api-budget.json` (temp + `replace()` atomic
writes, like all state). Structure: per-UTC-day buckets of
`{input_tokens, output_tokens, cached_tokens, usd}` plus a lifetime total and a
per-session detail list `{issue, session_id, started_at, tokens..., usd,
duration_s, outcome}`.

Three enforcement moments:

- **Preflight (routing):** refuse `api` launch when
  `spent_today + session_reserve > max_usd_per_day` or lifetime exceeded.
  `session_reserve` = `max_usd_per_session` when set, else a config
  `preflight_reserve_usd` (default 1.00) used purely as a conservative
  headroom estimate during the calibration window.
- **In-flight (watchdog):** each pass, sum usage from the live session's
  events.jsonl; if cost > `max_usd_per_session` (when set), kill the process
  tree and set `failure_kind="budget_exceeded"`. The existing
  dead-worker-with/without-PR reconciliation then treats it like any other
  death; routing's fallback sends the retry to Devin.
- **Settlement (reap):** on sidecar reap, write the session's final cost and
  outcome into the ledger.

**Calibration-first cap policy (explicit decision):** `max_usd_per_session`
ships **unset**. The enforcement code lands with the feature but stays dormant.
Procedure: run two issues through the api worker, review their per-session
ledger entries plus subsequent sessions, then set the cap from observed cost
distribution (suggested starting point: ~1.5× the observed max of a healthy
session). Daily and lifetime caps protect the wallet during calibration. The
rollout tracking issue (§9) carries this as a checklist item so it cannot be
silently forgotten.

## 7. Failure handling

- **Auth failures (401/403):** new log-tail classification patterns →
  `failure_kind="provider_auth"`, distinct from generic throttles, and the
  provider enters a cooldown (reuses the `throttled_until` state mechanism)
  so dispatch doesn't hammer a dead key.
- **Provider throttles/outages:** existing `throttle_error_markers` machinery
  applies unchanged; cooldown → routing falls back to Devin for the duration.
- **Budget kill:** `budget_exceeded` as above.
- All errors surface as values on records (`.error` / `failure_kind`) — the
  adapter never raises and never blocks (`Popen` path identical to
  claude-code).

## 8. Observability

- **doctor:** when `api_worker.enabled` — key env var present, `base_url`
  well-formed, ledger readable/parsable, remaining daily/lifetime headroom.
  When the section exists but is disabled: one-line notice (part of the
  don't-forget-to-enable surface).
- **Fleet report / status:** one line —
  `api-worker: kimi-k3, $X.XX today / $Y.YY lifetime of $15.00, N live, enabled 1/4 repos`.
  The `enabled N/M repos` fragment renders whenever any registered repo has the
  section configured, keeping partial rollout permanently visible until closed.

## 9. Rollout plan

1. **Trial:** `enabled: true` in charlie-work only; all other repos default off.
2. **Calibration:** two issues routed through the api worker; review ledger
   entries; set `max_usd_per_session` from data (§6).
3. **Fleet-wide enablement:** governed by a **tracked rollout issue filed at
   decomposition time** (not automated-ready — requires human judgment on trial
   results) with checklist: review calibration sessions → set session cap →
   enable in each registered consumer repo's config → raise/remove `lifetime_usd`
   → update runbook. Ambient reminder: the `enabled N/M repos` fleet-report
   fragment (§8) nags until rollout completes.

## 10. Testing

Mirror the per-adapter test pattern: new `tests/test_api_worker.py`,
`tests/test_api_budget.py` (the per-issue selection layer's own test file was
deleted along with it — see §5); third-arm additions to
`tests/test_worker.py` / `tests/test_worker_health.py`; `dispatch_sessions`
partition coverage alongside the existing adapter glue tests. Routing and
budget are pure functions over frozen inputs — exhaustively testable. No
live-API calls in tests; provider env resolution tested against fake env.
Run via `uv run --extra dev pytest -q --tb=short` (targeted files).

## 11. Invariants preserved

Frozen dataclasses for all new config/value types; atomic JSON for ledger and
sidecars; label strings only via `LabelConfig`; adapters never block on worker
completion; external-process errors returned as values, never raised.

## 12. Issue decomposition

Dependency spine: ① → ② → ③ → (④, ⑤) → ⑥ → ⑦; Phase 2 items independent
after ⑤. All are `automated-ready` except ⑪.

**Phase 1 — rework routing**

| # | Issue | Depends on |
|---|---|---|
| ① | `ApiProviderConfig`/`ApiWorkerConfig`/`ApiBudgetConfig` + layered-config parsing + validation | — |
| ② | `claude_code` `adapter_kind` parameterization (sidecar suffix, `provider` field, default-preserving) | ① |
| ③ | `api_worker.py` + `adapters.py` branch + `worker.py` third-arm wiring + `iter_workers` | ② |
| ④ | `api_budget.py` ledger + settlement on reap + force `tee_stream_json` | ③ |
| ⑤ | the per-issue selection module (deleted in Phase 2 of the role-config refactor — see §5) + `adapter_history` state recording | ① |
| ⑥ | Dispatch/rework-dispatch integration: per-issue partition, per-group `dispatch_sessions`, fallback path | ③ ④ ⑤ |
| ⑦ | Watchdog in-flight budget kill + `budget_exceeded`/`provider_auth` failure kinds + auth cooldown | ④ ⑥ |

**Phase 2 — complexity routing + polish**

| # | Issue | Depends on |
|---|---|---|
| ⑧ | `complexity:high` label: `LabelConfig` field + routing rule + issue-template guidance | ⑤ |
| ⑨ | doctor probes + fleet-report budget/enablement line | ④ |
| ⑩ | Runbook: provider onboarding, calibration procedure, budget ops | ⑦ |
| ⑪ | **Rollout tracking issue** (human): calibrate cap → enable fleet-wide → raise lifetime ceiling | ⑦ (execution gated on trial) |
