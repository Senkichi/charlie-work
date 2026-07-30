# Fleet Orchestration: Session Consumption Protocol — Design

**Status:** IMPLEMENTED (2026-07-20, ceremony skipped per operator — shipped directly, no
writing-plans pass) · **Date:** 2026-07-20

## 1. Problem

Live orchestrating sessions (this Claude Code session, acting as fleet operator across
charlie-work and job-cannon) spend heavy token budget on *active* orchestration — polling
`gh pr list`, reading worker/reviewer logs, and re-deriving worker health via LLM judgment —
even though a deterministic supervisor (`supervise.py` + `worker.py`, dispatched every 5
minutes by the `charlie-fleet-pass` scheduled task) already computes this exact information
every pass.

Two distinct causes, both real:
1. **Redundant work.** The session re-derives signals the deterministic pass already computed
   this cycle, because there's no cheap channel for the session to consume them.
2. **Genuine gaps.** Some things legitimately need a human/session judgment call today:
   detection gaps (new tripwires — wall-clock/cost/loop — still WARN-first or unshipped per
   `fleet-and-supervision-planning.md`), and one-off/novel failure modes no tripwire could
   reasonably pre-empt.

The goal is to eliminate (1) without weakening (2) — stay promptly responsive to things that
actually need attention, stop spending tokens re-discovering things that don't.

## 2. What already exists (verified, not assumed)

- `charlie-fleet-pass` scheduled task: hidden, runs `charlie fleet bash-rats` every 5 minutes,
  cross-repo, `LastTaskResult=0`. This is the detection cadence; already running, no changes
  needed.
- `WorkerHealth` classification + `AttentionDigest` (`notify.py`): per-pass digest of only the
  issues whose health *transitioned* this pass — inherently low-noise, not a steady-state dump.
- Four notify sinks already implemented and tested: `webhook`, `desktop`, `shell`, `file`. The
  `file` sink appends digest entries as JSONL, safe for single-writer append, to a configurable
  path (default `.var/charlie-work/notify/digest.jsonl`).
- **Currently disabled**: `orchestrator.config.yaml` has no `notify:` section, so
  `notify.enabled` defaults to `false`. Nothing is being written today — this is the actual gap,
  not the underlying machinery.
- `scripts/heartbeat_check.py`: deterministic, cross-repo (cw + job-cannon), prints `OK`/`ANOMALY`
  lines per check (dispatch coverage, throttle, stale in-progress, **review-liveness**,
  dispatch-failures, log-freshness, merge-flow, GH rate limit, runner health), exit code 0/1.
- **This session is itself periodically re-invoked externally** (confirmed by the user) with a
  fixed heartbeat-check prompt — i.e. the "coarse fallback wakeup" already exists as
  infrastructure. This design does not need to build scheduling; it needs to codify the contract
  for what those invocations do.
- `charlie mop-up`: existing idempotent reconcile command, already the right first tool for
  claim/state drift rather than ad-hoc fixes.

Net: the deterministic side is already mature. This design changes **session behavior**, plus one
one-line config flip. No new charlie-work features are required.

## 3. Design

### 3.1 Turn on the file sink (config-only)

Add to `orchestrator.config.yaml` (both charlie-work and job-cannon, if job-cannon runs the same
notify module — verify before assuming; job-cannon's state dir naming differs, `devin-orchestrator`
vs `charlie-work`, so it may be a distinct codebase needing its own equivalent):

```yaml
notify:
  enabled: true
  sink: file
  file_path: .var/charlie-work/notify/digest.jsonl
```

Zero new code. This is the one concrete artifact this design produces.

### 3.2 Event-driven consumption during live sessions

When a session is actively orchestrating (not just fielding a one-shot heartbeat wakeup), start a
`Monitor` tailing the digest file(s) instead of polling `gh`/logs manually:

```
tail -f .var/charlie-work/notify/digest.jsonl | grep --line-buffered -E '"health":\s*"(STALLED|RUNAWAY|DEAD)"|"escalated"'
```

- Filtered to transitions that represent something to *act on* (STALLED/RUNAWAY/DEAD/escalation)
  — benign transitions (recoveries, HEALTHY, or things the supervisor already
  auto-remediates within its bounded redispatch cap) don't need to interrupt the session.
- Between events: no polling, no active status-checking. Do other requested work, or go idle.
- One Monitor per repo (cw, job-cannon) covers the cross-repo scope this session operates over.

### 3.3 Standing heartbeat-wakeup protocol (codifying existing external cadence)

This session is already re-invoked periodically with a heartbeat-check prompt. The contract,
as specified and now codified as the standing default:

1. Run `scripts/heartbeat_check.py` (via `env -u VIRTUAL_ENV uv run --active --no-sync`, per the
   existing worktree-venv-shadowing guard). Its stdout is the entire output contract — trust the
   OK/ANOMALY lines, never second-guess with independent log reading.
2. Exit 0: note silently, no user-facing message.
3. Exit 1: for **each** ANOMALY line, dispatch one subagent (sonnet-tier) to investigate and
   remediate, in this preference order:
   - Diagnose root cause directly (process liveness, claim/log inspection) — this is the
     subagent's job, never the orchestrating session's.
   - Prefer existing deterministic tools (`charlie mop-up`) over ad-hoc state edits.
   - Never hand-edit `.var/**/*.json` state directly, never hand-commit a source fix or open a PR
     by hand — file a GitHub issue instead so the fix lands through the normal isolated-worktree /
     PR / adversarial-review / merge pipeline, avoiding a race with the live 5-minute fleet pass.
   - Set `needsHumanDecision=true` on anything genuinely ambiguous (can't confirm a process is
     dead, safety of killing/releasing unclear) rather than guessing.
4. Keep the orchestrating session's own turn minimal — this is a recurring automated check, not a
   user report, unless something needs a human decision.

### 3.4 Delegation discipline (applies to both 3.2 and 3.3)

Legwork (reading raw logs, diagnosing a specific anomaly, gathering PR/issue context) always goes
to a cheap subagent. The orchestrating session's own tool calls are reserved for: deciding what
needs a subagent, synthesizing subagent results, and making judgment calls a deterministic check
or subagent can't (policy tradeoffs, ambiguous review verdicts, anything flagged
`needsHumanDecision`).

## 4. Non-goals

- No changes to `supervise.py`, `worker.py`, the health classifier, or dispatch/review/merge logic
  — that machinery is already correct and sufficient for this design.
- No new scheduling infrastructure — the external wakeup cadence already exists.
- Does not attempt to close every detection gap (wall-clock/cost/loop tripwires) — those remain
  tracked separately per the existing `fleet-and-supervision-planning.md` issue decomposition
  (B4/B5/B7). This design's fallback wakeup is the accepted mitigation for gaps that slip through
  until those ship, not a replacement for shipping them.

## 5. Risks / tradeoffs

1. **Detection latency is unchanged** — still bounded by `min(5-minute fleet pass, heartbeat-wakeup
   cadence)`. This design doesn't tighten it, it just stops the *live session* from re-deriving
   what the next pass will compute anyway.
2. **Monitor only covers active live sessions.** Outside a live session, the external heartbeat
   wakeup (3.3) is the sole safety net — already true today, unchanged by this design.
3. **Silent notify-sink failure.** `NotifyResult(ok=False, ...)` never raises; a sink that starts
   silently failing (e.g. disk full, path issue) would starve the Monitor channel with no signal.
   Not fixed by this design — flagged as a candidate follow-up check for `heartbeat_check.py`
   (e.g. an OK/ANOMALY line for "digest file freshness"), not required to ship this.
4. **job-cannon parity — verified.** job-cannon has no `src/charlie_work` of its own; it's
   orchestrated by the same `charlie_work` package (installed from the charlie-work checkout,
   invoked via the fleet registry against job-cannon's `repo_root`/`config_path`), confirmed by
   `config.load_config()` loading job-cannon's `orchestrator.config.yaml` cleanly from
   charlie-work's own environment. Both repos' `notify:` blocks are live (3.1 shipped, §6).
   Caveat: job-cannon's `watchdog.enabled=false` (shim log-mtime blindness, see that repo's config
   comment) means STALLED transitions specifically won't appear in its digest until that
   structural fix lands — dead-pid reaping and other transitions are unaffected.

## 6. Concrete action items

1. **Done.** `notify:` block (3.1) added to charlie-work's `orchestrator.config.yaml`, validated
   via `config.load_config()`.
2. **Done.** job-cannon parity confirmed (same package, different config); `notify:` block added
   to job-cannon's `orchestrator.config.yaml` too, validated the same way.
3. **Standing practice.** 3.2 (Monitor over digest.jsonl) adopted for live orchestrating sessions
   going forward — session behavior change, not a code change; no further action to "ship."
4. **Standing practice.** 3.3 adopted as the documented contract for externally-triggered
   heartbeat wakeups (already matches observed behavior; now durable/explicit instead of
   re-specified ad hoc each time).
