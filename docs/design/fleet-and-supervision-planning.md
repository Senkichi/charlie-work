# charlie-work: Fleet Management & Worker Supervision — Design (v2, post-critique)

**Status:** REVIEWED (4-lens critique folded in) · **Date:** 2026-07-06

Two operator-reported problems, one unifying architecture. This v2 corrects the blockers a
verification pass found against the real code (label-edge safety, config backward-compat, the
already-on watchdog, stream-json detector regression, orphan-sweep worktree reuse, cross-repo identity,
and issue-filing convention).

---

## 1. Problems

### Problem A — Global, cross-repo worker management
Every real session runs charlie-work against **one or more primary consumer repos at once** while also
improving charlie-work itself ad-hoc. Today there is **no cross-repo concept at all**:
- `--repo` resolves to a bare `Path`; repo identity is never persisted — no `owner/name`, no repo field
  in `state.json` or the `GitHub` dataclass (`inv-multirepo §1,§3`).
- All state lives under `<repo_root>/.var/charlie-work/`; **no user-level/global dir exists** anywhere
  (`Path.home()`, `LOCALAPPDATA`, XDG all absent — `inv-multirepo §2`).
- Two repos = two independent invocations, two uncorrelated `state.json` files. No aggregation, no
  `repos:` config, no shared budget (`inv-multirepo §3,§4`).
- The per-repo governor (`dispatch.max_concurrent_sessions`, shipped #63/#105) caps workers **within one
  repo**; across N repos those caps stack and oversubscribe the host.

### Problem B — Deterministic, on-by-default worker observability & control
Workers hang unnoticed, spin in runaway loops, and tie up throughput.
- **Liveness is PID-alive only**; `(pid, process_start_time)` recycling defense shipped (#114) but legacy
  sidecars fall back to bare pid-alive, and **sidecars are never reaped** (#113 open) — they accumulate
  forever (`inv-worker-observability §2`).
- **The watchdog exists and already runs on every dispatch()/bash-rats pass** (unconditional call at
  `workflow.py:845`, shipped #109/#136). But its only progress signal is **log-mtime staleness OR two
  hardcoded terminal-error strings**. A **chatty runaway** writing any log line every <20 min evades it
  forever. **No wall-clock deadline, no cost/token budget, no loop/no-progress detection**
  (`inv-worker-observability §3,§4`).
- **Kill is tree-scoped, not job-scoped**; detached children escape (#139: a "killed" worker's orphan
  wrote 211 rows to a live DB over 7 hours). No orphan sweep (`inv-worker-observability §5`).
- **Visibility is snapshot-only**; `roll-call` reports alive/dead/stalled but never progress. WORKFLOWS.md:
  "there is no session-status API" (`inv-existing-gaps §1`).
- **Five near-identical two-adapter loops** (`workflow.py:102,124,175,284,414`) hand-duplicate the
  "check devin records, then claude records" pattern (`inv-core §6.7`).

**Correction from critique (arch-fit + operational lenses):** the *active kill-and-mark* watchdog is
**already on-by-default every dispatch()/loop() pass** — the real gaps are (a) an *idle* pass (no
dispatch/rework candidates) never reaches a kill call site, (b) `dispatch_rework`'s kill path is gated on
`max_concurrent>0` while `dispatch`'s is not, (c) a double-invocation when `max_concurrent>0` (governor
`:515` + `:845`), and (d) the *signals* the watchdog uses are too thin. This design does **not** re-enable
an already-enabled watchdog; it widens its signals, closes the coverage gaps, and adds bounded escalation.

---

## 2. Design principles (inherited invariants — non-negotiable)

1. Deterministic hub, no LLM control flow. 2. Invoke-per-pass, no daemon (detection latency = invocation
cadence; surfaced, not solved away). **Amendment (PR #248, 2026-07-10):** `charlie bash-rats` now defaults
to a *supervised foreground loop* (`supervise.py`) — an explicit, operator-attended sequence of unchanged
level-triggered passes triggered by cheap local deltas (sidecar/verdict mtimes, live count), exiting when
drained. This is not a daemon: no auto-start, no restart-on-crash, no background service; each pass remains
a complete re-derivation from ground truth. `--once` preserves single-pass behavior; `fleet bash-rats`
(scheduled-task path) stays one-shot per repo. A non-blocking `supervisor.lock` prevents concurrent passes.
3. Level-triggered reconciliation from ground truth each pass. 4.
GitHub labels = durable state; state.json = derived cache + 200-cap event log (durable counters need
dedicated record fields, per the `request_changes_count` precedent `workflow.py:1334-1338`). 5. Frozen
dataclass config, strict unknown-key validation, **additive-only** (renaming a field `ConfigError`s every
existing config). 6. Atomic JSON writes (temp+`replace()`) under the advisory lock — **reuse
`state.state_lock`/`state.save_state`, which are generic over any path** (`state.py:44-56,165-174`), never
a parallel implementation. 7. Errors as values. 8. Single point of enforcement — label edges live only in
`labels.transition`; we add the same discipline for worker status (one `worker.py`, not a 6th loop) and
must **name the exact module** that resolves per-repo-over-global config merge. 9. Non-blocking adapters
(`Popen`, never `.wait()`). 10. No hardcoded lists — terminal-error markers and the repo set are
config-derived / dynamically collected.

**Do NOT rebuild** (shipped, `inv-existing-gaps §3`): stalled watchdog (#109/#136), PID-recycling defense
(#114), per-repo governor (#63/#105), dead-no-PR relabel (#118), label-write surfacing (#125/#135),
`mop-up`/reconcile, `doctor --adapter-probe`, `roll-call --json`.

---

## 3. Unifying architecture: two reconcilers at two scopes

```
   ┌──────────────────────── FLEET (cross-repo) ────────────────────────┐
   │  user-level registry of managed repos + global budget + one view    │
   │  for each registered repo (carrying repo_key):                       │
   │     ┌────────────── SUPERVISOR (per-worker health) ──────────────┐  │
   │     │ observe: (pid,creation_time) · log activity · structured     │  │
   │     │          progress · cost/token · wall-clock                  │  │
   │     │ classify (pure fn): HEALTHY|SLOW|STALLED|RUNAWAY|DEAD|ORPHAN │  │
   │     │ act (level-triggered): none | kill+reap+orphan-sweep |       │  │
   │     │      redispatch(≤ intensity cap) | escalate→human + notify   │  │
   │     └──────────────────────────────────────────────────────────┘  │
   │     then existing pass: intake → dispatch → review → merge           │
   │  under a global budget; one consolidated attention digest + notify   │
   └────────────────────────────────────────────────────────────────────┘
```

- **Supervisor** = generalize the shipped governor/watchdog into a first-class, adapter-agnostic,
  multi-signal health reconciler. It widens signals and closes coverage gaps; it does not re-enable an
  already-enabled kill path.
- **Fleet** = a user-level registry + aggregation/dispatch commands composing existing single-repo
  primitives under one budget and one view.

Neither needs a daemon; each pass is a complete re-derivation.

---

## 4. Component design

### 4.0 Foundation — unified worker abstraction (`worker.py`)  [F1]
Collapse the five duplicated two-adapter loops into one module. Introduce `WorkerView` — an
adapter-agnostic frozen view over a sidecar exposing `(adapter_kind, issue_number, pid, created_at,
process_start_time)`, `log_path`, **required `repo_key`** (populated by callers so cross-repo aggregation
is never ambiguous — see 4.8/critique), and helpers `is_alive()` / `log_stat()`. `SessionRecord` and
`ClaudeWorkerRecord` are field-for-field identical (verified) — lift the shared shape to a protocol/base;
**also unify `from_dict`** (only `devin_shell` defines it today) so a generic `iter_workers(paths)` reader
has a symmetric constructor for both adapters. Refactor all five call sites. **Also collapse the
double-invocation** of `_detect_and_handle_stalled_sessions` (governor `:515` + dispatch `:845`) into one.
Acceptance: **zero behavior change, full existing suite green, new tests target only the WorkerView
abstraction**; issue body enumerates the five call sites; note backward-compat for reading pre-refactor
sidecars mid-upgrade. Prerequisite for every B feature and fleet aggregation.

### 4.1 Health model (`classify_worker_health`, pure fn)  [B3 core; tripwires B4/B5]
Closed `WorkerHealth` enum: `HEALTHY · SLOW · STALLED · RUNAWAY · DEAD · ORPHANED`. Multiple independent
tripwires, first-to-fire wins (research §2). **Config is additive** — keep `WatchdogConfig.stall_minutes`;
add new fields (do not rename); `WatchdogConfig` stays loadable (extend it, or add a superset
`SupervisorConfig` with `stall_minutes` retained/aliased).

| Signal | Source (per pass) | Trips to | Owner |
|---|---|---|---|
| liveness | `(pid, process_start_time)` match, uniform; **legacy `None` start-time never auto-kills/reaps** | DEAD | B3 |
| progress staleness | `now − log_mtime > stall_minutes` (existing knob, reused) | STALLED | B3 |
| terminal marker | last log line ∈ **config-derived** marker set (default = the current 2 strings) | DEAD | B3 |
| wall-clock | `now − started_at > wall_clock_minutes` (`activeDeadlineSeconds`) | RUNAWAY | B4 |
| cost/token | parsed cumulative usage > `cost_budget`/`token_budget` (Claude only; graceful absence) | RUNAWAY | B5 |
| loop / no-progress churn | **Claude Code only**: no new `tool_call` event for `2×stall_minutes` while log mtime advances → SLOW→(configurable)→RUNAWAY. **Devin: caps at SLOW/WARN, never kill** (no structured signal) | SLOW/RUNAWAY | B4 |
| orphan | non-live sidecar but a process still references the **exact** worktree path | ORPHANED | B6a |

**Defaults ship WARN-first:** new kill-triggering tripwires (wall-clock, cost, loop) default to **SLOW/WARN
(never RUNAWAY/kill)** until real session-duration data justifies a kill threshold — the HEALTHY/SLOW vs
STALLED/RUNAWAY split applied to *rollout*, per critique §op. Every threshold configurable; issue bodies
carry provisional default numbers + rationale.

### 4.2 Progress signal / session-status API (`worker.py` enrichment)  [B2]
Each pass, per live worker: `stat` the log (mtime, size) → record `last_activity_at`, `log_bytes` on the
sidecar (cheap, deterministic, adapter-agnostic — this alone closes gap #3 for both adapters).
**Structured parse, safely:** Claude Code can emit `--output-format stream-json`, but the shipped
detectors (`is_session_stalled` substring match, `_classify_session_failure` regex, `process_utils.py:78-91`,
`claude_code.py:101-144`) read the log as **plaintext** — switching the primary log to JSONL would silently
defeat the #136 watchdog + throttle classification. **Therefore tee stream-json to a *separate* file**
(`issue-<n>.events.jsonl`); the plaintext log and its detectors stay untouched. Parse the JSONL tail for
`tool_call_count`, `turn_count`, cumulative `tokens`/`cost_usd`. Devin `--print` has no structured stream —
degrade gracefully (no cost/tool signal is not "unhealthy"; document the asymmetry). Feeds B4/B5/B9.

### 4.3 Sidecar lifecycle / reaping  [PR against existing #113]
When the supervisor classifies a worker terminal, **archive its sidecar** to `sessions/terminal/` (keeps
forensics; log stays) so live-scan loops stop re-probing dead records. **Safety gate (critique
§correctness):** auto-reap only when `process_start_time is not None` (recycling-safe identity); a legacy
`None`-start-time record requires **N consecutive DEAD classifications across passes** (or an opportunistic
start-time backfill) before archiving — otherwise a transient false-DEAD read permanently vanishes a live
worker (inverse of #113). Idempotent; atomic move. **Files as a PR closing #113** (the open half; #114
fixed miscounting).

### 4.4 Orphan reaping + optional containment  [B6a → PR closing #139; B6b optional]
**Verified design call:** `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is wrong for a fire-and-forget launcher (the
orchestrator closes its handle on exit → would kill the worker); a job handle can't survive an
invoke-per-pass process (cross-checked dotnet/runtime #107992, research §5). So:

- **B6a (primary, cross-platform, closes #139):** a **per-pass orphan sweep** — enumerate processes,
  **exact normalized worktree-path** match (NOT substring) against sessions that were killed/are non-live,
  gated by a recorded **"worktree freed at T"** timestamp so the sweep only targets processes whose start
  time predates the freed time and postdates the kill (worktrees are *recycled* — `reclaimed` field — so a
  loose match would kill a live worker in a reused path). Kill survivors; report in pass JSON. POSIX already
  launches in a new session; kill uses `killpg`. This is one of #139's two "either alone helps" proposals →
  closes it.
- **B6b (optional, deferred, Windows):** a thin launcher **wrapper** holding a kill-on-close Job Object;
  killing the wrapper (pid the orchestrator tracks) tears down the job. If pursued: the sidecar's tracked
  liveness PID must resolve to the **worker** (not the wrapper) so stall detection is unchanged;
  wrapper-health is explicitly out of v1. Filed separately (one-issue-per-PR); does **not** block B6a.

### 4.5 Supervisor coverage + restart-intensity escalation  [B7]
Correcting the premise (critique): the kill watchdog already runs unconditionally in `dispatch()`
(`:845`) and thus every `bash-rats` pass. B7's **actual** scope:
1. **Idle-pass coverage:** run one unconditional supervisor sweep in `loop()` **before intake**, so a pass
   with zero ready/rework candidates still reaps stalled/orphaned sessions (today it never reaches a kill
   site). `roll-call` stays read-only by design.
2. **Rework-path parity:** make `dispatch_rework`'s kill path unconditional to match `dispatch`'s (today
   gated on `max_concurrent>0`).
3. **Restart-intensity cap (OTP MaxR/MaxT, research §3):** a **durable issue-record field holding
   redispatch timestamps pruned to the trailing `redispatch_window_minutes`** (windowed rate, not a
   monotonic int; bounded because time-scoped — unlike the 200-cap event log). If redispatches in the
   window exceed `max_auto_redispatch`, **escalate to `agent:human-needed`**.
   - **New label edge required (critique blocker):** the existing `escalated` edge removes only `reviewing`.
     Firing it from an `in_progress`/`queued` (redispatch) state would leave an active label + a terminal
     label simultaneously, violating the terminal/active invariant. **Add a distinct edge** (e.g.
     `redispatch_escalated`: add `human_needed`, remove the full `active` set — mirroring the `merged`
     edge's `tuple(sorted(active))`) in `labels._edges()`. The transition goes through `labels.transition`,
     never ad-hoc label calls.
4. **Accounting:** an orphan-sweep kill (B6a) is **not** a redispatch and does **not** count toward the cap
   (separate bucket) — resolves the critique's ambiguity.

(Note: #116 — a rework trigger consumed on a *failed* attempt — is a **different** failure mode that B7's
cap does not fix. It gets its own targeted fix against #116; see §5.)

### 4.6 Notification / attention layer  [B8]
Separate **detect** (supervisor, mechanical) from **notify** (pluggable sink: webhook / desktop / shell
command / file) from **escalate** (policy). A `notify` config section selects the sink; at the end of any
pass observing a needs-attention transition, emit a structured digest. Trigger = health-enum transition
(aligned with "state in labels, never chat memory"). **Ship a reference scheduled-invocation artifact**
(cron snippet + Windows Task Scheduler template invoking `charlie fleet bash-rats`) under `examples/`,
following the `examples/orchestrator.config.*.yaml` precedent — otherwise B8 is inert until the operator
solves scheduling themselves; **document plainly that detection latency = invocation cadence.**

### 4.7 roll-call / status health surface  [B9, Blocked by #152]
Extend `roll-call --json` (and `fleet status`) with a `workers` section: per live worker
`{repo, issue, adapter, health, runtime, last_activity_at, tool_calls, tokens/cost, budget_remaining}` —
**`repo` never omitted** (cross-repo disambiguation). **Mechanically `Blocked by #152`** (which adds a
`dependencies` field to the same `status()` JSON and is itself blocked by #153) so the two schema
extensions can't race on the same output contract — prose "coordinate" is not enforceable by the dispatch
gate; the marker is.

### 4.8 Fleet registry + global config layer  [A1]
User-level dir: Windows `%LOCALAPPDATA%\charlie-work\`, POSIX `${XDG_STATE_HOME:-~/.local/state}/charlie-work/`.
`fleet.json`: `{version, repos: {<key>: {repo_root, name_with_owner, config_path, state_dir, first_seen,
last_seen}}}`, `key = nameWithOwner` (resolved once via `gh repo view --json nameWithOwner`, cached; stable
across path moves). **Auto-register/refresh `last_seen` on every `charlie` invocation** in `build_app` — the
registry is *collected*, never hardcoded. **Reuse `state.state_lock` + `state.save_state`** (generic over
any path) for atomic read-modify-write of `fleet.json` — no new lock. Introduce a **global config layer**
(`<global-dir>/config.yaml`) for fleet-wide knobs, layered **under** per-repo config (per-repo wins); name
the resolving module explicitly (single point of enforcement, analogous to `labels.transition`);
additive, absent-file → no-op.

### 4.9 Fleet status (`charlie fleet status`)  [A2]
Read-only aggregation: walk the registry; per repo run `status()` + a read-only supervisor sweep; render
one unified view keyed by `repo_key`. Issue body notes: **ships functional without B9** (aggregation only,
no per-worker health column); re-verify once B9 lands. `--json` + human.

### 4.10 Global concurrency budget  [A3]
Extend the governor to also consult the **fleet-wide** live-worker count (sum across registered repos'
sidecars) against `fleet.global_max_concurrent_sessions`. Tolerate a vanished/moved repo dir (skip; prune
next pass). **Scoped claim (critique):** A3 prevents **worker-count** oversubscription only. It does **not**
by itself prevent **CPU** oversubscription, because each repo's `PYTEST_XDIST_AUTO_NUM_WORKERS` was sized
assuming that repo owns the box (RUNBOOK "Local host saturation ceiling"). The design either documents the
cross-repo discipline `sum(repo.default_limit) × xdist ≤ cores`, or weights each repo's allocation by its
own `default_limit`; it must not over-promise.

### 4.11 Fleet dispatch (`charlie fleet work` / `fleet bash-rats`)  [A4]
One pass across all/selected registered repos in priority order (oldest-`last_seen` or explicit `--repos`),
each composing the existing single-repo pass, under the global budget, emitting one consolidated attention
digest via B8. Composes existing primitives — no new worker logic. **Reuse each repo's own per-repo
`state_lock`** for its writes (do not introduce a fleet-level lock that could interleave non-atomically
with a concurrent manual per-repo invocation's governor read-then-launch window). **Latency note:** the
orphan sweep shells out (PowerShell CIM, ~5s/call) — running it once per repo per fleet pass could dominate
fleet-pass latency; batch/one-sweep-per-pass where possible.

### 4.12 Docs  [D1]
Document supervisor + fleet model in ARCHITECTURE/RUNBOOK/WORKFLOWS; ship the scheduler artifact (or ref
it from B8); **fix the stale RUNBOOK line** ("until an enforced concurrency cap exists in code" — shipped
via #63/#105).

---

## 5. Issue decomposition

**House convention (critique blocker):** work that fixes an *existing open issue* is filed as a PR
`Closes #N` against that issue, not a new duplicate (every merged PR here is "Fix #N: …"). So three items
attach to existing issues, and one net-new bug fix is spun out.

### Attach to / relabel existing issues (no new number)
- **#113** (open) — sidecar reaping on terminal exit → §4.3. **Relabel `automated-ready`**; refine body with
  the reaping-safety gate.
- **#139** (open, unlabeled) — per-pass orphan sweep → §4.4 B6a. **Relabel `automated-ready`**; refine body
  with exact-path + freed-timestamp acceptance.
- **#116** (open, `bug`) — rework trigger consumed on failed attempt. **Relabel `automated-ready`** as a
  standalone targeted fix (`fix(workflow): only consume rework trigger on ok:true dispatch (closes #116)`),
  no dependency on this design. (Removed from B7's scope.)

### New automated-ready issues to create
| id | title | depends on |
|---|---|---|
| **F1** | `refactor(worker): adapter-agnostic worker abstraction + collapse 5 duplicated loops` | — |
| **B2** | `feat(worker): progress/heartbeat enrichment — log activity + tee'd stream-json (session-status API)` | F1 |
| **B3** | `feat(supervisor): multi-signal WorkerHealth classifier + config-derived terminal markers (additive config)` | F1, B2 |
| **B4** | `feat(supervisor): wall-clock + no-progress/loop tripwires (WARN-first defaults)` | B3 |
| **B5** | `feat(supervisor): cost/token budget tripwire (tee'd stream-json usage)` | B2, B3 |
| **B6b** | `feat(supervisor): [optional] Windows Job Object launcher wrapper for containment` | F1 (deferred; not on critical path) |
| **B7** | `feat(supervisor): idle-pass + rework-path sweep coverage + restart-intensity escalation cap (+ new label edge)` | F1, B3 |
| **B8** | `feat(notify): pluggable needs-attention notification layer + reference scheduler artifact` | B7 |
| **B9** | `feat(observability): roll-call/fleet workers health section` — **Blocked by #152** | B2, B3, #152 |
| **A1** | `feat(fleet): user-level repo registry + global config layer (reuse state_lock/save_state)` | — |
| **A2** | `feat(fleet): fleet status — cross-repo aggregation view` | A1, F1 |
| **A3** | `feat(fleet): global concurrency budget across registered repos (worker-count scope)` | A1, F1 |
| **A4** | `feat(fleet): fleet work/bash-rats — cross-repo dispatch pass + attention digest` | A1, A3, B8 |
| **D1** | `docs: supervisor + fleet model; fix stale concurrency-cap RUNBOOK line` | (documents landed work) |

### Dependency graph & critical path
```
F1 ─┬─ B2 ─┬─ B3 ─┬─ B4 ─┐
    │       │      ├─ B5   ├─ B7 ─ B8 ─┐
    │       └───── B9*     │           │
    ├─ #113 (reaping)      │           │
    ├─ #139 (orphan sweep) ┘           │      *B9 also Blocked by #152 (→ #153)
    └─ B6b (optional)                  │
A1 ─┬─ A2                              │
    ├─ A3 ─ A4 ────────────────────────┘   (A4 digest needs B8)
#116 (independent, any time)              D1 (last, documents landed work)
```
**Critical path (corrected to match graph):** F1 → B2 → B3 → {B4, B5} → B7 → B8 → A4.
**Parallel off F1 alone:** #139 orphan sweep (schedule *first* after F1 — the only confirmed production
incident), #113 reaping, B2. **Parallel off A1:** A2, A3. **Independent:** #116.
**Out of scope, noted:** #151 (oldest-first dispatch) and #153 (dependency-gate bug) touch nearby dispatch
code but are pre-existing separate concerns; #152 gates B9 only.

---

## 6. Risks / tradeoffs

1. **Detection latency = invocation cadence** (no daemon). Mitigated by the shipped scheduler artifact
   (§4.6) + docs; not solved to real-time by design.
2. **False-positive kills** — the highest-severity regression risk, on an *already-active* kill path.
   Mitigation: WARN-first defaults for every new kill tripwire; generous provisional numbers; per-repo
   overridable; loop-churn is Claude-only and Devin caps at WARN.
3. **stream-json vs shipped detectors** — resolved by tee-to-separate-file (§4.2); B2 carries regression
   tests proving the existing terminal-error/rate-limit fixtures still trip on the untouched plaintext log.
4. **Orphan-sweep killing a live worker in a recycled worktree** — resolved by exact-path + freed-timestamp
   gating (§4.4); B6a acceptance criterion.
5. **Cross-repo identity ambiguity** — resolved by required `WorkerView.repo_key` and `{repo, issue}` on
   every cross-repo surface (§4.0/4.7/4.9).
6. **Concurrency across two fleet/manual invocations** — reuse per-repo `state_lock` for repo writes and
   `state_lock` for `fleet.json`; global-budget count tolerates a stale sidecar in repo X while acting in Y.
7. **A3 doesn't solve CPU oversubscription** — scoped claim + cross-repo xdist discipline documented (§4.10).
8. **B7 label-edge safety** — new `redispatch_escalated` edge clears the full active set; routed through
   `labels.transition`.
9. **Backward-compat** — additive config only (`WatchdogConfig` stays loadable); sidecars/issue-records
   without new fields still classify; F1 reads pre-refactor sidecars.
10. **Fleet-pass latency** — orphan sweep shells out; one sweep per pass, batched (§4.11).
