# charlie-work

Deterministic GitHub-issue orchestration for AI worker fleets. One labeled
issue → one worker session → one PR → adversarial review → gated auto-merge.
Devin workers by default; Claude Code workers first-class.

Extracted from the battle-tested orchestrators that ran inside
[job-cannon](../job-cannon) and [empericus](../empericus) — this repo is the
union of both forks plus the fixes each learned separately.

> The name is a nod to *It's Always Sunny in Philadelphia*: "charlie work" is
> the thankless, unglamorous grunt labor nobody else wants to do. Which is
> exactly what this tool takes off your hands.

## How it works

```
                    ┌─────────────────────────────────────────────┐
                    │            GitHub labels = state            │
                    └─────────────────────────────────────────────┘
  intake ──► dispatch ──► worker session ──► PR ──► review ──► merge-ready
  (issues        │        (Devin / Claude       (adversarial      │
   labeled       │         Code, one issue       packet, cross-   ├─ merge
   automated-    │         per session)          family pass)     ├─ labels
   ready)        └─ writes worker-prompt.md                       └─ branch
                    + session manifest                               delete
                                                                  (best-effort)
```

- **Hub**: a deterministic Python CLI (`charlie`). No chat memory, no
  LLM-driven control flow — state lives in GitHub labels plus a JSON state
  file under `.var/charlie-work/`.
- **Workers**: hermetic one-shot sessions, each bound to exactly one issue,
  driven by a generated `worker-prompt.md`.
- **Review**: a deterministic **janitor gate** runs first (draft/conflict/red-CI/
  no-issue-link checks) and short-circuits obviously-not-ready PRs before any
  LLM spend. PRs that pass get an adversarial review packet (diff, checks,
  metadata) and optionally a **cross-family pass** — a non-Claude model (codex
  via the Devin CLI) attacks the PR; its findings are leads, never merge gates.
- **Merge**: gated on required CI checks + a recorded `approved` decision.
  Branch deletion is remote-only and best-effort — it can never abort the
  merge/label sequence.
- **Reconcile**: `charlie mop-up` detects drift when humans act outside
  the orchestrator (a PR merged by hand leaving stale labels) — read-only by
  default, `--fix` to repair.
- **Supervise**: each pass classifies live worker health (`WorkerView` →
  `classify_worker_health`) and applies tripwires (liveness, staleness,
  terminal-marker, wall-clock, loop/no-progress, cost/token budget),
  escalating to `agent:human-needed` when the restart-intensity cap trips.
  `roll-call` surfaces a `workers` health section; the pluggable `notify`
  layer turns health transitions into outbound signals.
- **Fleet**: `charlie fleet …` composes N per-repo passes across a user-level
  repo registry under one global concurrency budget
  (`fleet.global_max_concurrent_sessions`). Neither the supervisor nor the
  fleet is a daemon — both are invoke-per-pass, so detection latency equals
  invocation cadence.

## Quickstart

`charlie-work` is a standalone dev tool with its own environment — it is **not**
a dependency of the repos it operates on. Clone it as a sibling of your consumer
repo and run it against that repo with `--repo`:

```powershell
# one-time: set up this tool's own environment
uv sync

# operate on a consumer repo by running charlie-work's own uv project against it:
#   --project selects charlie-work's env · --directory sets cwd · --repo is the target
uv run --project ../charlie-work --directory ../job-cannon charlie --repo ../job-cannon roll-call

# copy an example config to the CONSUMER repo's root and adjust it there
cp examples/orchestrator.config.devin.yaml ../job-cannon/orchestrator.config.yaml
```

Most consumers wrap that invocation in a one-line script, so day-to-day use is
just `charlie <command>`:

```powershell
charlie doctor              # preflight: env, labels, CI-check names, config
charlie doctor --adapter-probe   # also probe the worker CLI + surface stale sessions
charlie bootstrap-labels    # create the agent:* labels once
charlie roll-call --json    # what is ready / active / linked (with dependency graph)
charlie work --limit 3      # dependency-ordered dispatch wave (foundational first)
charlie work --issues 565,570   # explicit issue selection (respects dependency gate)
charlie why-charlie-hate --pr 123     # janitor gate → adversarial review packet
charlie verdict --pr 123 --decision approved --summary-file review.md
charlie ship-it --pr 123
charlie bash-rats --limit 3      # intake → dispatch → review → merge in one pass
charlie why-charlie-hate-spec --file docs/SPEC.md   # cross-family pass on a design doc
charlie mop-up              # detect label/state drift (--fix to repair)
charlie fleet status        # aggregate status across every registered repo
charlie fleet work --limit 3     # dispatch-only wave across all registered repos
charlie fleet bash-rats --limit 3   # full loop across all registered repos, one global budget
```

> **Why not an editable path dependency?** Because it breaks consumer CI: a
> locked `uv sync --all-extras` would try to install `../charlie-work`, which CI
> runners never check out. Running it as an external tool (above) leaves the
> consumer's dependency graph and lockfile completely untouched.

## Commands

Every command runs deterministically and is safe to re-run — state lives in
GitHub, not in the CLI. The verbs are themed; the mechanics are boringly
predictable.

| Command | What it does |
|---|---|
| `charlie roll-call` | show what's ready / active / linked (the `status` view) |
| `charlie roll-call --json` | show status with dependency graph and issue metadata |
| `charlie intake` | write worker prompts and state for issues already labeled `automated-ready` |
| `charlie work` | dispatch a dependency-ordered wave of one-issue worker sessions |
| `charlie why-charlie-hate` | janitor gate + adversarial review packet for a PR |
| `charlie why-charlie-hate-spec` | cross-family adversarial pass on a design doc |
| `charlie verdict` | record a review decision (`approved` / `request_changes` / `blocked`) |
| `charlie ship-it` | merge a PR once it's approved and required checks are green |
| `charlie bash-rats` | run one pass of intake → work → review → merge |
| `charlie mop-up` | detect (and with `--fix`, repair) label/state drift |
| `charlie doctor` | preflight diagnostics (env, labels, CI-check names, config, adapter) |
| `charlie bootstrap-labels` | create the nine `agent:*` / `automated-ready` labels once |
| `charlie fleet status` | aggregate `roll-call` across every registered repo |
| `charlie fleet work` | dispatch-only wave across all (or `--repos`-selected) registered repos, under one global concurrency budget |
| `charlie fleet bash-rats` | full intake→work→review→merge pass across all registered repos, under one global budget |
| `charlie runners status` | current runner-pool state for this repo (online/busy/idle, queue depth, host headroom) |
| `charlie runners allocate` | rebalance this host's running runner listeners across every repo by live queue demand (`--dry-run` to preview) |

## Dependencies

The orchestrator supports dependency-aware dispatch through issue body markers
and GitHub's native issue dependencies. This allows you to declare that an issue
depends on other issues, ensuring foundational work is completed before dependent
issues are dispatched.

### Dependency markers

Add any of these patterns to your issue body to declare blockers:

- `Blocked by #N` — case-insensitive, supports comma-separated lists
- `Depends on #N` — case-insensitive, supports comma-separated lists
- `Blocked-by: #N` — case-insensitive, supports comma-separated lists

Examples:
```
Blocked by #123, #124
Depends on #150
Blocked-by: #200, #201, #202
```

### GitHub native dependencies

The orchestrator also respects GitHub's native issue dependency feature
(`blocked_by` relationships). If your repo has GitHub dependencies enabled,
those relationships are automatically included in the dependency graph.

### How it works

1. **Dependency gate**: Issues with open blockers are never dispatched, even if
   they have the `automated-ready` label. They appear in `roll-call --json` under
   the `blocked` field with their blocker list.

2. **Dependency-aware ordering**: Unblocked issues are dispatched
   most-unblocking-first (issues that block the most currently-blocked dependents),
   with oldest-first as the tiebreaker. This maximizes unblocking per wave by
   dispatching critical-path issues first.

3. **Visibility**: `roll-call --json` includes a `dependencies` field for each
   issue with `declared` (all blockers mentioned in body/GitHub) and `open`
   (subset that are currently open) arrays.

### Example workflow

For a dependency-ordered epic:
1. File all issues with `Blocked by #N` markers declaring the dependency chain
2. Label all issues as `automated-ready`
3. Run `charlie work --limit 3` — the orchestrator dispatches foundational
   issues first, then their dependents as blockers close

## Configuration

Config is discovered at `<repo-root>/orchestrator.config.yaml` (override with
`--config`). Absent file → dataclass defaults. See [examples/](examples/) for
the two shipped profiles:

| Profile | Worker runtime | Notes |
|---|---|---|
| `orchestrator.config.devin.yaml` | Devin sessions | skills-based worker loop, cross-family review on |
| `orchestrator.config.claude-code.yaml` | Claude Code | direct-shell worker loop, Claude-only review |

Key knobs: `labels.*` (state-machine label names), `dispatch.default_limit` /
`branch_prefix` / `worker_template` / `order` (`oldest` | `newest`),
`review.max_rework_cycles` (past this many `request_changes` cycles a PR
escalates to `agent:human-needed`), `auto_merge.required_checks` (verify with
`doctor`), `runtime.prompts_dir` (repo-local template overrides),
`devin.adapter` (`manual` | `command` | `devin-shell` | `claude-code`),
`claude_code.*` (worktree/venv settings for the claude-code adapter),
`cross_family.*` (non-Claude adversarial pass), `test_adequacy.*` (opt-in
test-adequacy gate), `watchdog.*` (supervisor tripwires: stall/wall-clock/
loop/cost-token budgets, WARN-first by default), `fleet.*`
(`global_max_concurrent_sessions` — cross-repo worker-count budget),
`notify.*` (opt-in needs-attention sink: webhook | desktop | shell | file), and
`runner_allocation.*` (host-wide elastic CI-runner slots — see below), and
`runner_capacity_escalation.*` (sustained-window starvation escalation — see below).

### Self-hosted runner allocation

Two independent, opt-in features govern self-hosted runners:

| Section | Scope | What it changes |
|---|---|---|
| `runner_scaling.*` | one repo, vertical | Provisions and deregisters runners for *this* repo from its own queue pressure — mints tokens, extracts packages, removes directories. |
| `runner_allocation.*` | one host, horizontal | Redistributes a fixed budget of *running listeners* across every repo with runners registered under `managed_root`. Registration is never touched. |

Without allocation, each repo's CI parallelism is pinned to however many
runners were registered to it: a repo with a deep queue waits behind its own
cap while another repo's runners idle on the same machine. Allocation removes
that by making the *running listener* the unit that moves. A configured runner
whose listener is stopped goes `offline` and keeps its registration, so *moving*
a slot costs about a second — no registration token, no GitHub write, no package
extraction. End-to-end responsiveness is bounded by how often the decision runs,
not by the move: a repo that starts queuing waits until the next fleet pass (up
to `full_pass_interval_seconds`, 5 min) to be granted a slot.

Because it governs one physical host, this section belongs in the **global**
fleet layer (`%LOCALAPPDATA%\charlie-work\config.yaml` on Windows), not in a
per-repo `orchestrator.config.yaml` — three repos must not hold three different
opinions about one machine's capacity:

```yaml
runner_allocation:
  enabled: true
  managed_root: C:/actions-runners   # defaults to runner_scaling.managed_root
  max_running_runners: 8             # host's concurrent-CI-job budget; 0 = cores // 2
  min_running_per_repo: 1            # keep one listener alive per repo
  demand_idle_samples: 3             # slack passes before parking a surplus slot
```

Behavior worth knowing:

- **A running job is never interrupted.** Only listeners GitHub reports as
  not-busy *and* with no local `Runner.Worker` child are parked. A repo that is
  over-allocated but fully working keeps its slots, and the plan says so.
- **`min_running_per_repo: 1` is load-bearing.** A repo whose every runner is
  offline has queued jobs sit unclaimed (GitHub fails them after 24h). One live
  listener keeps pickup latency at zero.
- **Promotion is immediate, demotion is not.** A repo gains slots on the first
  pass that shows demand; a surplus slot is only parked after
  `demand_idle_samples` consecutive slack passes — unless another repo is
  actively waiting for it, in which case it moves at once.
- **Capacity is a hard ceiling.** A repo can never run more listeners than it
  has runners registered. When demand exceeds that, the plan says so — use
  `runner_scaling` (or `config.cmd`) to register more directories.
- **A repo whose demand cannot be read is pinned, not parked.** An API failure
  holds that repo at its current slot count, since parking could strand work
  that simply was not visible. Pins are the one way the host can sit *above*
  `max_running_runners`; when that happens the plan says so explicitly.
- **`max_running_runners` is a budget you assert, not one that is measured.**
  The `cores // 2` default cannot see Devin workers or reviewers competing for
  the same host, and a CI job here is a full pytest matrix entry — 8 concurrent
  jobs on 16 cores is a deliberate choice about memory and cache pressure, not
  a derived fact. Set it explicitly and preview with
  `charlie runners allocate --dry-run` before raising it.
- **Parking uses process termination, not a graceful drain.** A parked listener
  is not-busy at the moment it is stopped, but GitHub can take tens of seconds
  to mark it offline; a job assigned inside that window is re-queued and picked
  up by another runner rather than lost. Listeners this code started are in
  their own process group, so a `CTRL_BREAK_EVENT` drain is available if that
  window ever proves costly.

The fleet pass runs allocation as a prologue *before* autoscale: moving an idle
slot to a starved repo is free, so it is tried before deciding the host needs
more runners registered.

**Only one thing may decide which listeners run.** The host's post-reboot script
(`C:\actions-runners\start-runners.ps1`) delegates to `charlie runners allocate`
and only falls back to starting every runner when allocation is unavailable.
Anything that unconditionally starts all listeners while allocation is enabled
gets undone on the next pass, at the cost of a full hysteresis window of churn.

### Capacity-starvation escalation

`runner_allocation` can only move *already-registered* listeners — it never
mints a registration. So a repo whose live demand exceeds its registered
runner capacity while the host-wide budget has idle slots is permanently
unsatisfiable by allocation alone: the allocator correctly identifies it and
correctly declines to act, but before #763 that conclusion lived only in a
log line and a per-pass `notes` entry the operator digest deliberately drops.
The next starvation was discovered by an operator reading queue times, not
surfaced by the fleet.

`runner_capacity_escalation.*` arms the durable half: when the same repo stays
starved for a sustained window, the fleet prologue raises a structured
`runner_capacity_starvation_escalation` event that surfaces in the operator
attention digest (not just `events.db`). Scope is detection + event only;
provisioning/registration stays operator-gated (issue #826 is the
manual-trigger actuator). It is host-wide (declare it in the global fleet
layer, not a per-repo config) and inert on any host where `runner_allocation`
is disabled, since the prologue returns before reaching it:

```yaml
runner_capacity_escalation:
  enabled: true                       # default; pure observability, no actuation
  starvation_escalation_minutes: 15   # sustained window before escalating (default 15)
```

The window is measured wall-clock from the first starved pass, not by counting
passes, so it is robust to the supervisor's respawn cadence and to a pass that
was skipped. The escalation is edge-triggered per episode: it fires once when
the window is crossed, then stays silent every subsequent pass the starvation
holds, so a reader can tell "still starved" from "signal stopped working". A
repo that recovers is dropped from the tracking sidecar so the next episode
starts a fresh window.

**This repo's own CI check names** (for `auto_merge.required_checks`): `Tests (ubuntu-latest)`,
`Tests (windows-latest)`, and `Lint`. These correspond to the job `name:` fields in
`.github/workflows/ci.yml` and are verified by `charlie doctor`.

**Worker adapters** (`devin.adapter`): `manual` writes a session manifest for
the operator to paste; `command` runs a blocking per-issue launcher;
`devin-shell` launches headless `devin --print --permission-mode dangerous`
sessions non-blocking, each in an isolated per-issue git worktree (creation
before launch; junction-safe teardown on failure), with sidecar tracking;
`claude-code` launches `claude -p` workers in isolated git worktrees (shared
venv junctioned in — teardown is junction-safe). Probe the configured adapter
with `charlie doctor --adapter-probe`.

## Prompt templates

Package defaults live in `src/charlie_work/prompts/`. A repo-local
directory (`runtime.prompts_dir`) overrides them **by filename** — drop in
your own `worker.md` carrying your repo's invariants and canonical commands,
and everything else keeps the defaults. `worker.md` targets Devin sessions;
`worker_claude_code.md` targets Claude Code workers
(`dispatch.worker_template` selects). Blocks shared by both worker templates
live as partials under `prompts/worker_sections/` and render as
`$section_<stem>` — repo-local `worker_sections/` dirs override by filename
too, so a shared change lands in one place instead of drifting across forks.

## State

`.var/charlie-work/` in the consumer repo holds `state.json`
(issues/prs/events, schema v1 — compatible with pre-extraction state), the
per-issue worker prompts, per-PR review packets and decisions, and the
session dispatch manifest/results. All JSON writes are atomic
(temp-file + rename).

## Provenance

Unioned from two production forks (June–July 2026): job-cannon contributed
cross-family review and `why-charlie-hate-spec`; empericus contributed `--issues` wave
dispatch, the `gh pr merge` stdout fix, and the worktree/branch-deletion
failure report that drove the decoupled merge sequence.
