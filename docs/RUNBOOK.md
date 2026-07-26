# Runbook

Operational procedures for running `charlie` day to day. For the
architecture behind these procedures, see [ARCHITECTURE.md](ARCHITECTURE.md).
For exact command sequences per worker adapter, see
[WORKFLOWS.md](WORKFLOWS.md).

## Reading status and `state.json`

```powershell
uv run charlie roll-call --json
```

`OrchestratorApp.status()` returns: `ready_issue_count` (open issues labeled
`automated-ready`), `available_issue_count` (of those, how many are
actually dispatchable — no active or terminal label already present),
`active_issue_count`, `open_linked_pr_count`, `auto_merge_enabled`, the full
`issues`/`prs` lists (each issue annotated with `dispatchable: bool`), and
`last_generated_at` (from `state.json`'s `generated_at` field).

**`state.json` is a derived cache, not the source of truth** — GitHub labels
are (see [ARCHITECTURE.md](ARCHITECTURE.md#hub-and-spoke-model)). If you
suspect `state.json` has drifted from reality, trust `gh issue list --label
agent:in-progress` / `gh pr list` over what's on disk; `roll-call` itself
already re-queries GitHub live, it does not read state for its counts.

Raw inspection:

```powershell
Get-Content .var\charlie-work\state.json | ConvertFrom-Json | Select-Object -ExpandProperty events | Select-Object -Last 20
```

The `events` array (capped at the most recent 200 entries by
`state.append_event`) is the closest thing to an audit trail — every
`intake`, `dispatch`, `review_packet`, `record_review`, `merge_ready`, and
`spec_review` call appends one entry with a UTC timestamp and a payload.

## Label meanings and legal transitions

| Label | Meaning | Set by |
|---|---|---|
| `automated-ready` | Operator-applied. Issue is eligible for automation. | Human, on GitHub. |
| `agent:queued` | Manual adapter dispatch wrote a session manifest; no worker independently confirmed yet. | `work` → event `queued`. |
| `agent:in-progress` | A non-manual adapter actually launched a worker. | `work` → event `dispatched`. |
| `agent:pr-open` | A PR exists and `review()` has run against it at least once. | `why-charlie-hate` → event `review_started`. |
| `agent:reviewing` | Set alongside `agent:pr-open` in the same transition; distinguished for readability, not a separate state. | `why-charlie-hate` → event `review_started`. |
| `agent:needs-rework` | `verdict --decision request_changes`, under the rework cap. | `verdict` → event `rework_requested`. |
| `agent:blocked` | `verdict --decision blocked` — a product/security decision is needed. | `verdict` → event `blocked`. |
| `agent:done` | PR merged via `ship-it`. Every `active` label is removed in the same transition. | `ship-it` → event `merged`. |
| `agent:human-needed` | Either `blocked`, or the rework cap was exhausted. Terminal — no further automation happens until a human clears it. | `verdict` → event `escalated` or `blocked`. |

Legal transitions are exactly `labels.py`'s `_edges()` table — see the
mermaid diagram in
[ARCHITECTURE.md](ARCHITECTURE.md#label-state-machine). Two invariants worth
internalizing operationally:

- `LabelConfig.terminal = {blocked, done, human_needed}` and
  `LabelConfig.active = {queued, in_progress, pr_open, reviewing,
  needs_rework}`. `_is_dispatchable()` refuses to dispatch an issue that
  already carries any active *or* terminal label — so **removing a stale
  label by hand is sometimes the correct unblock**, not just adding one.
- `agent:reviewing` is removed by both `rework_requested` and `escalated`,
  never left dangling alongside `agent:needs-rework` or
  `agent:human-needed`.

## Handling `agent:human-needed` escalations

An issue lands on `agent:human-needed` for one of two reasons — check
`review-decision.json` under `.var/charlie-work/prs/pr-<n>/` to tell
which:

1. **Explicit block** (`"decision": "blocked"`) — a reviewer decided a
   product or security call is needed before more automation should touch
   this PR. Read `summary` in the decision file for the reviewer's
   reasoning.
2. **Rework cap exhausted** (`"decision": "request_changes", "escalated":
   true`) — the PR has been through `review.max_rework_cycles` (default `2`;
   both shipped example profiles also set `2`) rounds of `request_changes`
   without converging. This usually means the issue brief was wrong,
   ambiguous, or the acceptance criteria are unimplementable as written —
   not that "one more rework round" will fix it.

**Recovery**: fix the underlying problem (rewrite the issue, resolve the
product ambiguity, or manually push a fix to the PR branch yourself), then
either:

- Re-run `charlie verdict --pr <n> --decision approved
  --summary-file <path>` once you've verified it's actually fixed, which
  routes straight to `ship-it` eligibility, or
- Manually swap `agent:human-needed` back to `agent:reviewing` (or
  `agent:needs-rework`) on GitHub and re-run `charlie why-charlie-hate --pr <n>` to
  regenerate a fresh packet before deciding again.

There is no automatic un-escalation — a human decision, once escalated,
requires a human (or an explicit re-`verdict`) to move the issue
forward again.

## Corrupt-state quarantine recovery

`state.load_state()` never raises on a truncated or malformed
`state.json`. On a `JSONDecodeError` or `OSError` reading it, the file is
renamed in place to `state.json.corrupt-<UTC-timestamp-no-colons>` (e.g.
`state.json.corrupt-20260702T180411Z`) and a fresh `empty_state()` is
returned and used for that run.

**What this means operationally**: after a quarantine event, `state.json`
starts fresh (empty `issues`/`prs`/`events`) — but nothing on GitHub was
touched, and none of the per-issue/PR artifact files (`worker-prompt.md`,
`review-decision.json`, etc.) are affected, since they live in separate
files. `roll-call` will still correctly show live GitHub state because it
re-queries `gh`, not the (now-empty) cache. To recover the lost projections:

1. Find the quarantined file: `Get-ChildItem .var\charlie-work\state.json.corrupt-*`.
2. Inspect it for forensics (what was in flight, what the last few `events`
   entries were) — it's valid enough to eyeball even if it failed strict
   JSON parsing (truncation usually cuts off the tail).
3. Do **not** delete it silently — it's your only record of the interrupted
   write. Move it aside for later analysis once you've extracted anything
   useful.
4. Re-run `charlie intake` and `charlie roll-call` to rebuild the
   `issues`/`prs` projections from live GitHub + existing artifact files;
   `review-decision.json` files under `prs/pr-<n>/` are untouched and remain
   the merge-gate authority regardless of what's in `state.json`.

Corruption is almost always caused by a process kill mid-write; the
temp-file-then-`replace` atomic write in `save_state()` makes an actually
torn write on the *final* file rare, but does not protect a write that
never got past the temp-file stage or a manually edited file.

## Reconcile for drift

`charlie mop-up` detects drift between GitHub's actual state and the
orchestrator's recorded labels/state — the concrete, observed gap being a PR
merged outside `merge_ready()` (e.g. a human clicking "Merge" on GitHub
directly) whose issue is still labeled `agent:in-progress` because the
`merged` label transition never ran.

```powershell
charlie mop-up          # read-only: reports every drift item, mutates nothing
charlie mop-up --fix    # repairs labels/state for the detected drift
```

Without `--fix` it is strictly read-only (two `gh` list queries, zero
writes). With `--fix` it repairs labels through the same `labels.transition`
edges the normal pipeline uses and rewrites `state.json` from a fresh copy
(never an in-place mutation). Drift kinds detected: PR merged outside the
orchestrator, PR closed-unmerged with active labels still set, a `state.json`
PR entry GitHub no longer knows about, an issue with active labels but no open
PR, and contradictory label sets (a terminal label alongside active ones).

## Rework-cap behavior

Configured via `review.max_rework_cycles` (default `2` in `config.py` and in
both shipped `examples/*.yaml` profiles — "iteration past ~2 rounds thrashes"
per the operator decision recorded in `config.py`'s `ReviewConfig`
docstring). `record_review()` reads and increments a **durable per-PR
counter** — `state["prs"][<pr_number>]["request_changes_count"]` — and bases
the escalation decision on that field. This counter is intentionally **not**
derived from `state["events"]`: `append_event` truncates the events log to the
last 200 entries (`state.py`), so on a busy repo that eviction could silently
reset an events-derived count and allow a PR to rework indefinitely instead of
escalating. The durable counter survives the events log rolling over. Practical
implication: **the count is per-PR, not per-issue** — if a rework cycle
requires closing a PR and opening a fresh one for the same issue (branch
unrecoverable), the cycle count resets, because it's keyed by
`pr_number`. Worker prompts (`rework.md`) explicitly instruct workers to
update the existing PR rather than open a new one for exactly this reason.

## Test-adequacy gate behavior

Configured via `test_adequacy.enabled` (default `False` in `config.py` and in
both shipped `examples/*.yaml` profiles — opt-in by default). When enabled, the
gate runs a deterministic structural check in `OrchestratorApp.review()` before
packet generation: if the diff adds product code with zero test files touched,
the gate auto-records a `request_changes` decision (bypassing the LLM reviewer)
and routes to rework. This Tier-1 hard gate is self-correcting — a "pure skip"
PR automatically reworks without waiting on an LLM review round. When the gate
passes (tests present or exempt), the review packet includes a "Test-adequacy
facts" section with metrics (added product/test LOC, assertion count, untested
files) and the LLM reviewer receives a stricter written rubric for evaluating
test quality (behavior-coverage table, hollow-test heuristics, exemption
scrutiny). Practical implication: **when enabled, the gate is advisory to the
LLM reviewer (Tier 2) but hard-blocking for pure-skip failures (Tier 1)** — a
PR with no tests and no exemption claim never reaches the LLM, while a PR with
tests present gets the facts block and rubric but the LLM still decides
approval.

## Worktree cleanup gone wrong (junction hazard)

**The hazard**: worker worktrees created by `worktree.create_worktree()`
(used by `claude_code.py`) can share one dev+eval
virtualenv via a Windows junction (or symlink elsewhere) at
`<worktree>/.venv`. A junction is a reparse point — following it is
transparent to most tools. **`git worktree remove --force` and `rm -rf`
both follow the junction into the real shared `.venv` and recursively
delete its contents**, corrupting every other live worktree that shares it,
because they all point at the same target directory.

**Never do this**:

```powershell
# WRONG — if <worktree>/.venv is a junction, this deletes the SHARED venv
Remove-Item -Recurse -Force .var\charlie-work\worktrees\agent-issue-565
git worktree remove --force .var\charlie-work\worktrees\agent-issue-565
```

**Correct teardown order** (what `worktree.remove_worktree()` does
internally — mirror this by hand if you must clean up manually):

1. Check whether `<worktree>/.venv` is a junction/reparse point (`Get-Item
   <worktree>\.venv | Select-Object LinkType,Target`, or in Python,
   `worktree.is_junction(path)`).
2. If it **is** a junction: unlink the reparse point itself, not its
   target — `[System.IO.Directory]::Delete("<worktree>\.venv")` in
   PowerShell (deletes the link, not the folder it points to) or
   `os.rmdir(venv_path)` in Python (Windows `rmdir` on a junction removes
   the link).
3. If it is a **real directory** (not a junction — a worker cold-built its
   own venv instead of linking), only `rm -rf` that specific worktree copy;
   never touch it if you're unsure without checking `LinkType` first.
4. Only then run `git worktree remove <path>` (add `--force` if git
   complains about untracked changes you're intentionally discarding).
5. If `git worktree remove` still fails, run `git worktree prune` to clear
   stale metadata — `remove_worktree()` does this automatically on failure.

**If you already ran the wrong command and corrupted the shared venv**:
every other worktree pointing at that `.venv` junction target is now
broken. Recreate the shared venv from scratch (`uv sync` in a clean
location) and re-link surviving worktrees, or simply tear down and recreate
every active worktree — there is no partial-recovery path once the target
directory's contents are gone.

## Local host saturation ceiling (claude-code adapter)

The `claude-code` adapter runs every worker as a **local** `claude -p` process
in its own worktree on the *same machine* — unlike the Devin adapters, whose
sessions execute in Devin's cloud VMs. (The `command` adapter can also run
locally; if its `dispatch_command` spawns a host-local worker the same ceiling
applies, but charlie can't inject the cap into an operator-supplied command —
bound xdist inside that command yourself.) So a `default_limit`-wide wave puts K
full worker toolchains (each running the repo's tests, and often a heavy
eval/ML import stack) on one host at once. The throughput ceiling here is
**CPU/RAM oversubscription, not the shared-venv junction** — the junction is a
*cleanup* hazard (see the section above), never a runtime bottleneck; a reparse
point is transparent to reads.

The dominant stall is stacked `pytest-xdist` pools. A suite whose `addopts`
carries `-n auto` spawns one worker **per core** on *every* invocation —
including the small targeted runs a worker does for a one-file change. K
concurrent worktrees each doing that is K × cores test processes on a
cores-count box; add each worker's eval-stack RSS and the machine pages into
swap and stalls long before any worker-CLI quota is hit.

Mitigations (the fleet is charlie-work's, but the test config is the
*consumer* repo's — that's where these land):

- **Bound xdist so `default_limit × n ≈ physical cores`**, not `-n auto`. On an
  8-core box with the default 3-wide fleet, `-n 2` (→ 6 workers) leaves
  headroom; `-n auto` (→ 16 × 3 on a 16-thread box) does not. The claude-code
  adapter enforces this at the launch boundary: set
  `claude_code.worker_env: {PYTEST_XDIST_AUTO_NUM_WORKERS: "2"}` (the shipped
  example already does) and every `pytest -n auto` a worker runs is capped to 2
  without editing the suite's `addopts` — CI never sees the var, so it keeps
  every core. Size the value so `default_limit × value ≈ physical cores`. The
  var only governs `-n auto`/`-n logical` resolution — an explicit `-n N` (in a
  worker command or the suite's `addopts`) is not bounded, and a consumer
  `conftest` `pytest_xdist_auto_num_workers` hook overrides it; that's why the
  worker-prompt discipline (below) is the complementary half, not redundant.
- **CI is the gate; local is targeted verify.** Workers should run only the
  tests covering their diff (plus any fast schema/migration guard set) locally
  and delegate full-suite correctness to the CI matrix. `worker_claude_code.md`
  instructs this; reinforce it in the consumer's `CLAUDE.md` canonical test
  command.
- **Exclude the worktrees root from Windows Defender real-time scanning** — it
  re-scans the shared tree on every file touch across all K workers and is the
  single biggest Windows multiplier. From an *elevated* PowerShell:
  `Add-MpPreference -ExclusionPath '<worktrees-root>'` (check first with
  `Get-MpPreference | Select-Object -ExpandProperty ExclusionPath`, which itself
  needs admin).
- **Per-worktree venvs are the default for `claude-code`** (the
  `claude_code.venv_source` option is `null` by default). Each worker builds its
  own isolated `.venv` inside its worktree, which is the only safe choice for a
  shell-capable worker that can run `uv sync` — a shared-venv junction lets a
  worker rewrite the shared venv's editable-install `.pth` to point at the
  worktree and corrupt the operator's `charlie` CLI. If you explicitly set
  `claude_code.venv_source`, you are opting back into the junction and the
  cleanup hazard described above; bound parallelism with `PYTEST_XDIST_AUTO_NUM_WORKERS`
  instead of disabling isolation.

## Shared-venv isolation for devin-shell

Per-worktree venvs are also the default for `devin-shell` (the `devin.venv_source`
option is `null` by default, issue #112). The `devin --prompt-file ... --print`
worker has a full shell and can run `uv sync`, so a shared-venv junction would
let a worker rewrite the shared venv's editable-install `.pth` to point at its
own worktree. Any other worktree that later runs `python -c "import <pkg>"` (or
`uv run --active python ...`) silently imports that worktree's code, producing
fabricated verification results. If you explicitly set `devin.venv_source`, you
are opting back into the junction and the cleanup hazard described above; keep
per-worktree isolation unless you have a specific reason to share a venv.

## GitHub credential isolation for workers (issue #502)

Workers must not be able to merge their own PRs. The orchestrator's adversarial
review gate is the only sanctioned path to `main`, so a worker that inherits the
orchestrator's admin-scoped credentials can bypass review entirely by running
`gh pr merge` itself.

`sanitize_env()` (`env_sanitize.py`, shared by the `devin-shell`, `claude-code`,
and `cross-family` adapters) closes both channels `gh` uses to resolve an
identity:

- **Environment tokens.** `GH_TOKEN`, `GITHUB_TOKEN`, `GH_ENTERPRISE_TOKEN`, and
  `GITHUB_ENTERPRISE_TOKEN` are stripped from the worker's base environment.
- **Stored credentials.** `GH_CONFIG_DIR` is forced to a worktree-local empty
  directory (`<worktree>/.var/gh-config`), created on demand, so `gh` cannot fall
  back to the orchestrator's `gh auth login` state via the platform-default
  config path.

**Operational consequence: workers have no `gh` authentication by default.** This
is intentional, but it means any worker task that needs `gh` — reading issue
comments, pushing via the `gh` credential helper, opening its own PR — will fail
with an auth error unless you supply a credential explicitly.

To give workers a scoped credential, set a **narrowly scoped** PAT in the
adapter's `worker_env`:

```yaml
devin:
  worker_env:
    GH_TOKEN: "<scoped-PAT>"
claude_code:
  worker_env:
    GH_TOKEN: "<scoped-PAT>"
```

`worker_env` is merged **after** `sanitize_env()`, so an explicit value wins over
sanitization while the orchestrator's own token still never reaches the worker.
That merge order is load-bearing and covered by
`test_launch_passes_operator_scoped_gh_token_through`.

Scope the PAT to the minimum the worker actually needs, and specifically **do not
grant it merge rights on protected branches** — a token with merge permission
reopens exactly the bypass this control exists to prevent. The tripwire below
will catch such a merge after the fact, but it cannot prevent it.

**Post-merge tripwire.** Prevention is not assumed to be airtight, so every
`loop()` pass also scans recently-merged PRs on the worker branch prefix and
flags any whose merged head SHA is not covered by an `approved`
`review-decision.json` (`_detect_unauthorized_merges()`). Findings surface in the
pass's `errors` bucket. Treat one as a security event: identify which credential
performed the merge before re-arming anything.

## Fleet: cross-repo dispatch

The fleet layer extends the single-repo orchestrator to operate across multiple
registered repos under a global concurrency budget. This is not a daemon — it
is invoke-per-pass, matching the hub-and-spoke model in ARCHITECTURE.md.

**Registry location**: The fleet registry lives at a user-level path:
- Windows: `%LOCALAPPDATA%\charlie-work\fleet.json`
- POSIX: `${XDG_STATE_HOME:-~/.local/state}/charlie-work/fleet.json`

The registry (`fleet_registry.py`) tracks each registered repo by
`name_with_owner` (resolved via `gh repo view`) and stores:
- `repo_root` (absolute path to the worktree)
- `config_path` (orchestrator.config.yaml location)
- `state_dir` (.var/charlie-work location)
- `first_seen` / `last_seen` timestamps

Registration happens automatically on every command that loads config
(`intake`, `dispatch`, `loop`, etc.) via `touch_repo()`. If `gh repo view`
fails (offline, not a GitHub repo, gh missing), registration is silently
skipped for that invocation.

**Global budget**: `fleet.global_max_concurrent_sessions` caps total live worker
sessions across all registered repos. This is a worker-count budget only — it
does not bound CPU or RAM. The governor applies this cap at every dispatch
path alongside the per-repo `dispatch.max_concurrent_sessions` cap
(`_apply_concurrency_governor()` in `workflow.py`).

**Scoped claim**: The fleet budget bounds worker *count*, not CPU. When running
the fleet across multiple repos on one host, you must still respect the
cross-repo xdist discipline from the Local host saturation ceiling section
above:

```
sum(repo.default_limit) × PYTEST_XDIST_AUTO_NUM_WORKERS ≤ physical cores
```

Today's per-repo guidance only bounds one repo at a time; the fleet multiplies
the risk. If you have 3 repos each with `default_limit: 3` and each worker
runs `pytest -n auto` on a 16-core box, that's 16 × 3 × 3 = 144 test processes
simultaneously. Set `claude_code.worker_env: {PYTEST_XDIST_AUTO_NUM_WORKERS: "2"}`
globally (via the fleet config layer) or per-repo to keep the total under
physical cores.

**Fleet commands**:

- `charlie fleet status` aggregates status across all registered repos (reads
  the registry, calls `OrchestratorApp.status()` per repo with `dry_run=True`,
  and returns a combined JSON or human-readable report).
- `charlie fleet work [--limit N] [--repos owner/a,owner/b]` runs a
  dispatch-only wave across the registry (`fleet_dispatch.fleet_loop`,
  `work_only=True`).
- `charlie fleet bash-rats [--limit N] [--repos …] [--merge/--no-merge]` runs
  the full intake→work→review→merge loop per repo.

> **Previewing a fleet pass.** `charlie --dry-run fleet bash-rats` and
> `charlie --dry-run fleet supervise` gate the self-deploy step, so they do not
> fast-forward-pull `origin/main` into the running checkout or `uv sync` its venv;
> they report what the deploy *would* do instead.
>
> On builds predating that fix (issue #613) they did both, on every invocation —
> and because moving the deployed checkout's HEAD terminates a running supervisor
> by design (drift exit), a "preview" typed while diagnosing a fleet problem could
> deepen it into an outage when the watchdog task was disabled. If you are on such
> a build, preview with `charlie --dry-run runners allocate` or the read-only
> `charlie fleet status` instead.

`fleet work` / `fleet bash-rats` walk the registry oldest-`last_seen`-first (or
the explicit `--repos` order), enforce both the per-repo
`dispatch.max_concurrent_sessions` and the fleet-global cap at every dispatch
path, isolate per-repo failures (a broken/moved repo never aborts the sweep),
and emit a consolidated attention digest at the end of the pass
(`data.digest`: needs-attention event count + orphan-sweep calls).

## Supervisor: worker health & escalation

The supervisor layer runs as a sweep nested inside each pass (it generalizes
the original stalled-session watchdog #109/#136). It classifies every live
worker via the adapter-agnostic `WorkerView` abstraction
(`worker.classify_worker_health`) into a `WorkerHealth` state, applies a table
of tripwires, and escalates to `agent:human-needed` when the restart-intensity
cap is exceeded.

**Health states** (`WorkerHealth` enum in `worker.py`): a worker is classified
from multiple signals — process liveness, log-file staleness (mtime vs.
`watchdog.stall_minutes`), terminal error markers, absolute age, loop/no-
progress detection, and cumulative cost/token usage. `charlie roll-call`
surfaces these under a `workers` section (health per live sidecar) so a
STALLED / RUNAWAY / DEAD worker is visible on the next pass.

**Tripwires** (all under `watchdog.*` in `orchestrator.config.yaml`, WARN-first
by default so nothing kills a worker until you opt in):

| Tripwire | Config knob | Default | Kill? |
|---|---|---|---|
| Stall (log mtime idle) | `watchdog.stall_minutes` | `20` | redispatch (see cap) |
| Wall-clock age cap | `watchdog.wall_clock_minutes` / `wall_clock_kill` | `240` / `false` | only if `wall_clock_kill` |
| Loop / no-progress | `watchdog.loop_stall_multiplier` / `loop_kill` | `2` / `false` | only if `loop_kill` |
| Cost budget | `watchdog.cost_budget_usd` / `cost_budget_action` | `null` (off) / `warn` | only if action `kill` |
| Token budget | `watchdog.token_budget` / `cost_budget_action` | `null` (off) / `warn` | only if action `kill` |

The cost/token tripwires read cumulative usage from Claude Code's tee'd
`events.jsonl` stream (session-status API); they are inert for adapters that
don't emit usage.

**Restart-intensity escalation**: redispatch is capped by
`watchdog.max_auto_redispatch` (default `3`) within
`watchdog.redispatch_window_minutes` (default `240`). Past the cap the
supervisor stops restarting and escalates the issue to `agent:human-needed`
rather than thrashing a genuinely-stuck worker indefinitely.

## Scheduled invocation

The orchestrator is not a daemon — it is invoke-per-pass. Detection latency equals
invocation cadence. For continuous operation, schedule periodic invocation via
your platform's scheduler:

- **cron** (Linux/macOS): Schedule `charlie bash-rats` (single repo) or
  `charlie fleet bash-rats` (multi-repo) at your desired cadence. A commented
  crontab entry ships at `examples/schedule/charlie-fleet.cron`.
- **Windows Task Scheduler**: Import `examples/schedule/charlie-fleet-task.xml`
  with `schtasks /create /xml charlie-fleet-task.xml /tn "charlie-work fleet"`
  and adjust the `<Interval>` (e.g. `PT5M` for every 5 minutes).

Both artifacts are reference templates — adjust the interval, working
directory, and command to your setup before enabling.

## Continuous infill mode

`charlie bash-rats` now runs a **supervised foreground loop** by default.
Each iteration polls cheap local signals (session sidecar mtimes, verdict file
mtimes) and runs a full pass when something actionable changes.  A fallback
pass fires every `full_pass_interval_seconds` (default 5 min) even with no
local delta, to catch GitHub-side changes.  The loop exits automatically when
the system is fully drained (no live workers, no pending dispatches, no open
tracked PRs awaiting verdicts).

### Default mode (supervised loop)

```powershell
uv run charlie bash-rats
```

Press **Ctrl+C** at any time to exit cleanly.  The aggregate summary
(passes run, dispatched, merged, runtime) is printed on exit.

Each pass emits a compact one-line summary:

```
[HH:MM:SS] pass N: dispatched F+R, merged M, reviewed V(+S skipped), live ~K, prs-open P, errors E
```

### Single-pass mode (legacy behavior)

```powershell
uv run charlie bash-rats --once
```

Runs exactly one pass and exits, same as the old behavior.  While a
supervised loop is running, `--once` also refuses with an error (the single
supervisor lock prevents double-dispatch through the concurrency governor's
read→launch window).

### CLI knobs

| Flag | Description | Config equivalent |
|------|-------------|-------------------|
| `--poll-interval N` | Override poll interval (seconds) | `supervisor.poll_interval_seconds` |
| `--max-runtime N` | Stop after N minutes (0 = unlimited) | `supervisor.max_runtime_minutes` |
| `--once` | Single pass, then exit | — |

### Config knobs (`orchestrator.config.yaml`)

```yaml
supervisor:
  poll_interval_seconds: 20        # how often to check local signals
  full_pass_interval_seconds: 300  # fallback pass even with no local delta
  active_cooldown_seconds: 30      # sleep after a pass that dispatched/merged
  max_runtime_minutes: 0           # 0 = unlimited
```

### Detection latency

- **Local events** (worker exits/starts, verdict files written): detected within
  ~2× `poll_interval_seconds` (20 s default → ≤40 s latency).
- **GitHub-side events** (e.g. label changes made manually): detected on the
  next fallback pass (`full_pass_interval_seconds`, 5 min default).

### Lock-file behavior (scheduled invocations)

A non-blocking `supervisor.lock` in `.var/charlie-work/` prevents concurrent
invocations from double-dispatching.  If Task Scheduler triggers a new run
while a supervised loop is already running, the new invocation exits
immediately with:

```
supervisor already running (supervisor.lock held)
```

The lock is released on clean exit, KeyboardInterrupt, or process death.

### `fleet bash-rats` is unchanged

`charlie fleet bash-rats` (multi-repo, scheduled-task path) remains a
one-shot command — it calls `app.loop()` directly per repo and does not enter
the supervised loop.  The supervisor lock is repo-local, so each repo's
single-repo `bash-rats` loop is independent.

## Session-limit / quota discipline

Both non-blocking adapters (`devin_shell.py`'s `launch_devin_session()`,
`claude_code.py`'s `launch_claude_worker()`) are `Popen`-based and return
immediately; they do not enforce a concurrency limit themselves. The Devin CLI's session store (`sessions.db`) is
documented (per the extraction dossier) as **single-threaded / SQLite-
contention-limited** — do not assume parallel local Devin-shell dispatch
works reliably; serialize dispatch waves or shard across machines/profiles
if you need real concurrency.

The orchestrator enforces a code-level concurrency cap via
`dispatch.max_concurrent_sessions` (shipped in #63 and #105). The governor
`_apply_concurrency_governor()` in `workflow.py` counts live sessions at
every dispatch path (`dispatch()`, `dispatch_rework()`, and the combined
wave budget in `loop()`/`bash-rats`) and clamps the dispatch limit to the
available slots. This replaces the need for manual session-count checks
before dispatch — the governor does this automatically.

Operator discipline:

- Keep `dispatch.default_limit` (default `3`) conservative relative to your
  actual quota/rate-limit budget for whichever worker CLI you're driving.
- `doctor`'s `probe_devin()` / `probe_claude()` helpers are cheap
  `--version` checks — use them to confirm the CLI is reachable at all
  before burning a dispatch wave on a broken PATH.

## Needs-attention notification and detection latency

charlie-work has no daemon — detection is strictly invocation-cadence-bound. A stalled session found on pass N is invisible until pass N+1 runs, and nothing runs it automatically unless you schedule it. The pluggable `notify` layer (configured under `notify:` in `orchestrator.config.yaml`) turns health transitions (STALLED / RUNAWAY / DEAD / escalated-to-human) into outbound signals an operator actually sees — webhook, desktop toast, shell command, or file.

**Detection latency = invocation cadence.** There is no background daemon; this design does not add one. If you run `charlie bash-rats` (or `charlie fleet bash-rats` for multi-repo) manually every hour, a stalled session can go undetected for up to an hour. For timely detection, schedule periodic invocation:

- **cron (Linux/macOS)**: See `examples/schedule/charlie-fleet.cron` for a commented crontab entry. Adjust the interval (e.g., `*/5 * * * *` for every 5 minutes) based on how quickly you need to detect issues.
- **Windows Task Scheduler**: See `examples/schedule/charlie-fleet-task.xml` for an XML template. Import with `schtasks /create /xml charlie-fleet-task.xml /tn "charlie-work fleet"` and adjust the `<Interval>` (e.g., `PT5M` for 5 minutes).

The `notify` layer is disabled by default (`notify.enabled: false`). To enable it, add a `notify:` section to your `orchestrator.config.yaml` — see `examples/notify.config.yaml` for sink options (webhook, desktop, shell, file). The layer emits a digest once per pass if any issue's health state changed since the last pass, comparing against a dedicated per-issue `health` field in `state.json` (not the capped `events` log). Sink failures never fail the pass — errors are returned as values and logged.
