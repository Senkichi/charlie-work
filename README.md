# charlie-work

Deterministic GitHub-issue orchestration for AI worker fleets. One labeled
issue → one worker session → one PR → adversarial review → gated auto-merge.
Devin workers by default; Claude Code workers first-class.

Extracted from the battle-tested orchestrators that ran inside two sibling
repos — this repo is the union of both forks plus the fixes each learned
separately.

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
   labeled       │         Code, one issue       review           ├─ merge
   automated-    │         per session)          packet)          ├─ labels
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
  metadata) for the reviewer to work from.
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
uv run --project ../charlie-work --directory ../your-repo charlie --repo ../your-repo roll-call

# copy an example config to the CONSUMER repo's root and adjust it there
cp examples/orchestrator.config.devin.yaml ../your-repo/orchestrator.config.yaml
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
| `orchestrator.config.devin.yaml` | Devin sessions | skills-based worker loop, no automated reviewer dispatch |
| `orchestrator.config.claude-code.yaml` | Claude Code | direct-shell worker loop, Claude-only review |

Key knobs: `labels.*` (state-machine label names), `dispatch.default_limit` /
`branch_prefix` / `worker_template` / `order` (`oldest` | `newest`),
`review.max_rework_cycles` (past this many `request_changes` cycles a PR
escalates to `agent:human-needed`), `auto_merge.required_checks` (verify with
`doctor`), `runtime.prompts_dir` (repo-local template overrides),
`worker.harness` (`manual` | `command` | `devin-shell` | `claude-code`),
`claude_code.*` (worktree/venv settings for the claude-code adapter),
`test_adequacy.*` (opt-in test-adequacy gate), `watchdog.*` (supervisor tripwires: stall/wall-clock/
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
that by making the *running listener* the unit that moves — a configured
runner whose listener is stopped goes `offline` and keeps its registration, so
moving a slot costs about a second with no registration token, GitHub write,
or package extraction involved. A running job is never interrupted (only idle
listeners are parked), every repo keeps at least one live listener, and
capacity can never exceed however many runners a repo actually has
registered.

Because it governs one physical host's shared capacity, allocation is
configured once at the global fleet layer rather than per repo, under a
`runner_allocation` section (`enabled`, `managed_root`, `max_running_runners`,
`min_running_per_repo`, `demand_idle_samples`). Only `charlie runners
allocate` is meant to decide which listeners run — an operator script that
unconditionally starts every listener will have its choice undone on the next
allocation pass. Preview any config change with `charlie runners allocate
--dry-run` before applying it.

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

**This repo's own CI check names** (for `auto_merge.required_checks`): `Tests` and
`Lint` — single hosted jobs since the 2026-08-28 hosted-CI return (#1500), no OS
matrix. These must match the job `name:` fields in `.github/workflows/ci.yml`
exactly: the merge gate (`checks.py`) compares live check names verbatim, so a
stale matrix-suffixed entry like `Tests (windows-latest)` would count as
`missing` forever. Do not rely on `charlie doctor` to catch that case — its
matrix-suffix tolerance deliberately accepts `Name (suffix)` against a job
named `Name`, so it fails open when a matrix has been collapsed.

**Worker adapters** (`worker.harness`): `manual` writes a session manifest for
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

Unioned from two production forks (June–July 2026) of sibling repos: one
contributed the (since-removed) cross-family review pass and
`why-charlie-hate-spec` command; the other
contributed `--issues` wave dispatch, the `gh pr merge` stdout fix, and the
worktree/branch-deletion failure report that drove the decoupled merge
sequence.
