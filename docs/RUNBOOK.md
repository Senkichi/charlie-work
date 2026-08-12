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

The `events` array is a bounded recent-activity view, **not** the audit trail —
`state.append_event` caps it at `state.DEFAULT_EVENT_RING_SIZE` (2000) entries,
overridable via the `runtime.event_ring_size` config knob. Every `intake`,
`dispatch`, `review_packet`, `record_review`, `merge_ready`, and `spec_review`
call appends one entry with a UTC timestamp and a payload.

The audit trail is `events.db`, the unlimited append-only SQLite database
written alongside `state.json` — every event is dual-written to it, so anything
evicted from the ring is still there. Reach for it, not the ring, whenever the
question is about history rather than the last few minutes:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from charlie_work.instrumentation import event_counts_by_kind; print(event_counts_by_kind(Path('.var/charlie-work/state.json')))"
```

Note the argument is the **`state.json`** path, not `events.db` — every one of
these helpers takes `state_path: Path` and derives the database beside it
(`instrumentation.py:334`). Passing a string fails; passing the `.db` path only
works by coincidence of that derivation.

`query_events()` filters by `kind`, `ts`, `correlation_id`, `pr_number`, and
`issue_number` (all indexed); `events_by_correlation_id()` returns every event
from a single `loop()` pass.

Scale of the difference, measured on this fleet: 22,467 events in `events.db`
against a 2000-entry ring. If you reason about anything older than the last few
hours from the ring alone, you are reading roughly the most recent 9% of history.

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
| `agent:human-needed` | `blocked`, the rework cap exhausted, or the cross-family regeneration budget exhausted. Terminal — no further automation happens until a human clears it. | `verdict` → event `escalated`/`blocked`, or `loop()` → event `cross_family_report_regen_exhausted`. |

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

An issue lands on `agent:human-needed` for one of three reasons. The first
two are recorded in `review-decision.json` under
`.var/charlie-work/prs/pr-<n>/`; the third is raised by `loop()` itself,
before any reviewer produces a decision file, so check
`state["issues"][<n>]["escalation_reason"]` in `state.json` instead:

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
3. **Cross-family regeneration budget exhausted**
   (`escalation_reason == "cross_family_report_unusable"`, `reason_class ==
   "judgment"`) — the cross-family model **ran** `cross_family.max_regen_attempts`
   times (default `2`) against the PR's *current* head SHA and the report was
   still unusable each time (an `(UNAVAILABLE)` failure stub, or one with no
   head-SHA marker). The budget is per head, not per PR lifetime — a new push
   starts a fresh budget, since the new head has never been tried. Unlike the
   rework cap, this never ends in a caveated `approved`: an unconfirmed
   cross-family head must never pass as reviewed, so it escalates instead (see
   [ARCHITECTURE.md](ARCHITECTURE.md#invariants)).

   Since #1099 this reason means the model genuinely ran and failed. It used to
   also fire for PRs whose model had never been invoked once — `review()`
   returned at the janitor gate long before the regenerator, and the budget was
   charged anyway. If you are triaging escalations created **before** #1099
   shipped, check `events.db` for a `cross_family_report_regen_forced` row for
   that PR with no corresponding model activity: those are false escalations,
   and the PR's real blocker is whatever its `janitor_gate` payload names.

**Recovery**: fix the underlying problem (rewrite the issue, resolve the
product ambiguity, or manually push a fix to the PR branch yourself), then
either:

- Re-run `charlie verdict --pr <n> --decision approved
  --summary-file <path>` once you've verified it's actually fixed, which
  routes straight to `ship-it` eligibility, or
- Manually swap `agent:human-needed` back to `agent:reviewing` (or
  `agent:needs-rework`) on GitHub and re-run `charlie why-charlie-hate --pr <n>` to
  regenerate a fresh packet before deciding again. For reason 3 specifically,
  this is enough on its own, and stays so after #1099 moved the budget claim
  into the regenerator: `charlie why-charlie-hate` passes
  `enforce_regen_budget=False`, so the manual re-run gets a fresh attempt and
  charges nothing. `charlie unescalate` additionally clears the PR's
  `cross_family_regen` record outright, so the automated loop gets a fresh
  budget too — without that the re-arm would be inert, with `loop()` reading
  the spent counters and parking the PR again on its very next pass.

There is no automatic un-escalation — a human decision, once escalated,
requires a human (or an explicit re-`verdict`) to move the issue
forward again.

### A PR making no progress with no `agent:human-needed` label

Escalation is not the only terminal-ish state. A PR whose cross-family report is
unusable **and** whose `review()` never reaches the regenerator is *parked*
after `cross_family.max_regen_attempts` passes: no escalation, no label, and the
report simply stops forcing `review()` on its own (issue #1099). This is by
design — the model was never invoked, so there is nothing to escalate *about*,
and parking self-heals on the next push.

To confirm that is what you are looking at:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from charlie_work.instrumentation import query_events; s=Path('.var/charlie-work/state.json'); [print(e['ts'], e['kind'], e['payload']) for k in ('cross_family_regen_not_reached','janitor_gate') for e in query_events(s, kind=k, pr_number=<n>)]"
```

The `cross_family_regen_not_reached` payload carries `not_reached` and
`max_attempts`; the `janitor_gate` payload names the PR's **actual** blocker,
which is what to fix. Merge conflicts and missing required checks are the
overwhelming majority. Fixing that and pushing resets both budgets, since the
record is keyed by head SHA. A park with no accompanying `janitor_gate` event is
worth investigating — it means `review()` returned somewhere else.

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
derived from `state["events"]`: `append_event` truncates the events log to
`state.DEFAULT_EVENT_RING_SIZE` (2000) entries, so on a busy repo that eviction could silently
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

The tripwire has two dedupe layers so a control that can never go quiet is not
allowed to bury new signal in constant noise:

1. **Pre-arming baseline** (`unauthorized_merge_baseline` in `state.json`,
   issue #510): the first pass records which uncovered merges already predate
   the control and reports nothing. Every later pass reports only merges absent
   from that baseline.
2. **Post-arming acknowledgment** (`unauthorized_merge_acknowledged` in
   `state.json`, issue #673): a post-arming finding that has been triaged (root
   cause fixed, issue filed, or confirmed benign per the #634 audit taxonomy)
   is acknowledged once and then stops pinning `ok=False` on every subsequent
   pass. Acknowledgment is never automatic — it requires an explicit action:

   ```
   charlie tripwire ack <pr> --reason "root cause fixed in #N" [--by <operator>]
   ```

   A non-empty `--reason` is mandatory. Re-acking the same PR updates the
   record (new reason/by/timestamp) rather than duplicating. The ack and its
   audit trail (who/why/when) are recorded as an
   `unauthorized_merge_acknowledged` event in `state.json` and `events.db`.

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

## API worker operations

The `api` adapter is a third worker kind: the Claude Code CLI launched with
provider environment injected so any **Anthropic-compatible** paid-API endpoint
(Moonshot/Kimi K3, GLM, MiniMax, vLLM deployments, …) powers the session. The
entire launch/supervision stack is reused — `api_worker.launch_api_worker`
delegates to `claude_code.launch_claude_worker` with `adapter_kind="api"`, so
sidecars land as `issue-<n>.api.json`, the watchdog/supervisor treats api
workers like any other, and dead-worker reconciliation is unchanged. Routing is
per-issue (not global per repo): reworks and `complexity:high` first-pass issues
go to the api worker; everything else goes to the default adapter.

Design reference: [docs/design/api-worker-adapter.md](design/api-worker-adapter.md).
The config dataclasses are `ApiWorkerConfig`, `ApiProviderConfig`, and
`ApiBudgetConfig` in `config.py`; the ledger is `api_budget.py`; routing is
`routing.py`. This section is verified against those modules, not the design
spec.

### Enabling the api worker per repo

The feature is **off by default** (`api_worker.enabled: false`). Enable it by
adding an `api_worker:` section to your `orchestrator.config.yaml`. The full
annotated schema, matching the shipped frozen dataclasses exactly:

```yaml
api_worker:
  enabled: false                    # master switch; default false
  provider: kimi-k3                 # active provider key (must exist in providers)
  max_concurrent_sessions: 1        # live api workers allowed at once
  providers:                        # registry; adding a provider is a config edit (see below)
    kimi-k3:
      base_url: https://api.moonshot.ai/anthropic   # Anthropic-compatible endpoint
      api_key_env: MOONSHOT_API_KEY                  # NAME of the env var holding the key
      model: kimi-k3                                 # pinned via ANTHROPIC_MODEL
      input_usd_per_mtok: 3.0                        # > 0 required when enabled
      output_usd_per_mtok: 15.0                      # > 0 required when enabled
      cached_input_usd_per_mtok: 0.30                # >= 0; defaults to 0.0
  budget:
    max_usd_per_session: 0          # 0 = per-session cap UNSET (calibration window)
    preflight_reserve_usd: 1.00     # headroom estimate while per-session cap is unset
    max_usd_per_day: 5.00           # daily USD ceiling
    lifetime_usd: 15.00             # lifetime USD ceiling (trial ceiling; raise post-trial)
  fallback_adapter: devin-shell     # adapter used when an api preflight check fails
  worker_template: worker_claude_code.md
  rework_template: rework.md
```

**Where this section lives in the layered config.** Config is loaded by
`global_config.load_layered_config`: a global layer at
`<fleet_dir>/config.yaml` (Windows: `%LOCALAPPDATA%\charlie-work\config.yaml`;
POSIX: `${XDG_STATE_HOME:-~/.local/state}/charlie-work/config.yaml`) supplies
fleet-wide defaults, and the per-repo `orchestrator.config.yaml` overrides on
any key present in both. The `api_worker` section is the **one section that is
deep-merged** (so a per-repo override of, say, `budget.max_usd_per_session` or
an added `providers` entry does not drop the global defaults). All other
sections keep shallow-merge semantics (repo keys fully replace global keys).
Put fleet-wide provider definitions and baseline caps in the global layer; put
per-repo `enabled: true` and any cap overrides in the per-repo file.

**The host environment variable.** The API key **value never lives in config** —
only the *name* of the env var does (`api_key_env`). The host running charlie
must export the env var named by the active provider's `api_key_env` (e.g.
`$env:MOONSHOT_API_KEY = "<key>"` in PowerShell). `launch_api_worker` reads it
via `os.environ.get(provider.api_key_env)`; a missing/empty var is an error
*value* (a `ClaudeWorkerRecord` with `.error` set), never a raise. The token
travels only in the child process env — it is never written to a sidecar, log,
prompt, or command argv. `doctor`'s launched-sessions probe surfaces
`provider-auth` and `budget-exceeded` failure kinds on dead sidecars.

**Validation at config load** (`ApiWorkerConfig.__post_init__`): when
`enabled: true`, `provider` must be a non-empty string naming a key in
`providers`; the active provider's `api_key_env` must be non-empty; its
`input_usd_per_mtok` and `output_usd_per_mtok` must be `> 0`; and
`cached_input_usd_per_mtok` must be `>= 0`. Budget fields must be numbers `>= 0`.
A misconfigured `api_worker` block fails fast at load — fix the config before
re-running.

### Adding a provider (config-only, no code changes)

Any Anthropic-compatible endpoint is a new entry under `providers:` — set its
`base_url`, `api_key_env`, `model`, and per-Mtok pricing, then point
`provider:` at the new key. There are **no code changes**: the registry is read
from config at load time and exposed as an immutable mapping on
`ApiWorkerConfig.providers`. To switch the active provider, change `provider:`
(and ensure the new key's `api_key_env` is exported on the host).

**Explicit non-goal:** OpenAI-protocol-only endpoints are **not** supported.
The harness is the Claude Code CLI, which speaks the Anthropic Messages API;
providers must expose an Anthropic-compatible endpoint. Revisit only when a
concretely wanted provider lacks one.

Pricing fields drive the spend ledger (see Budget operations). Per-Mtok cost is
`usd = tokens / 1_000_000 * usd_per_mtok` per token class: `input_tokens`
(including `cache_creation_input_tokens`, billed at the input rate — there is no
separate cache-write premium) at `input_usd_per_mtok`; `cache_read_input_tokens`
at `cached_input_usd_per_mtok`; `output_tokens` at `output_usd_per_mtok`. Claude
Code's self-reported dollar cost is **wrong** against non-Anthropic endpoints;
token counts are correct, so the ledger derives USD from tokens × your pricing.

### Calibration procedure (per-session cap ships UNSET)

`budget.max_usd_per_session` ships at `0.0` **on purpose** — the in-flight
enforcement code is live but dormant until you calibrate. Do not guess a cap;
measure first:

1. **Enable** the api worker (above) with `max_usd_per_session: 0`, conservative
   `max_usd_per_day` and `lifetime_usd`, and `preflight_reserve_usd` set to a
   headroom estimate (default `1.00`).
2. **Route two issues through the api worker.** Reworks route to api
   automatically (`policy:rework`); a first-pass issue routes to api when it
   carries the `complexity:high` label (`policy:complexity`). Let both sessions
   run to completion.
3. **Read their ledger entries** in `.var/charlie-work/api-budget.json` under
   `sessions`. Each entry records `issue`, `session_id`, `provider`, `model`,
   `started_at`, `ended_at`, `input_tokens`, `output_tokens`, `cached_tokens`,
   `usd`, `duration_s`, and `outcome`. Inspect with:
   ```powershell
   Get-Content .var\charlie-work\api-budget.json | ConvertFrom-Json | Select-Object -ExpandProperty sessions
   ```
4. **Keep reviewing subsequent sessions** — the first two are a sample, not a
   distribution. Watch the `usd` and `duration_s` columns across several healthy
   sessions.
5. **Set `budget.max_usd_per_session`** from the observed cost distribution.
   Starting guideline: **~1.5× the max `usd` of a healthy session**. Put the
   override in the per-repo (or global) config and reload.

**How daily/lifetime caps protect spend while the per-session cap is unset.**
With `max_usd_per_session: 0`, the preflight still enforces `max_usd_per_day`
and `lifetime_usd` every launch: a new api launch is refused when
`spent_today + reserve > max_usd_per_day` or when lifetime spend has reached
`lifetime_usd`, and routing falls back to `fallback_adapter` with
`fallback:budget` (recorded in `adapter_history`). The `reserve` used for the
daily headroom check is `max_usd_per_session` when set, otherwise
`preflight_reserve_usd` — that is the conservative stand-in during the
calibration window. So even with the per-session cap dormant, a runaway day or a
blown lifetime ceiling still trips and diverts to the fallback adapter.

### Budget operations

**Ledger location and shape.** The ledger lives at
`.var/charlie-work/api-budget.json` (`api_budget.LEDGER_FILENAME` joined to the
state dir). Shape: `days` (UTC `YYYY-MM-DD` → `{input_tokens, output_tokens,
cached_tokens, usd}` aggregate), `lifetime_usd` (running total), and `sessions`
(per-session detail list). All writes are atomic (temp-file + `replace()`), and
settlement is idempotent — re-settling the same session
(`issue` + `started_at` + `session_id`) is a no-op, so a double reap never
double-charges.

**Reading the ledger:**
```powershell
Get-Content .var\charlie-work\api-budget.json | ConvertFrom-Json
```
Today's spend is `days[<UTC date>].usd`; lifetime is `lifetime_usd`;
per-session detail is `sessions` (the calibration source).

**What happens at daily/lifetime exhaustion.** Routing's api preflight
(`routing._api_preflight`) runs before every api launch in order: `enabled` →
`auth` → `budget` → `cooldown` → `concurrency`. When
`spent_today + reserve > max_usd_per_day` **or** lifetime spend has reached
`lifetime_usd`, the preflight returns `fallback:budget` and the issue is routed
to `fallback_adapter` (default `devin-shell`) instead. The choice is appended to
that issue's `adapter_history` in `state.json` as
`{ts, kind, provider, reason}` with `reason: "fallback:budget"` (see Reading
routing decisions). No api launch happens until headroom returns (a new UTC day
for the daily cap; a raised `lifetime_usd` for the lifetime cap).

**Raising caps.** Edit `budget.max_usd_per_day` and/or `budget.lifetime_usd` in
config (global or per-repo) and reload. There is no ledger reset — raising
`lifetime_usd` above the current `lifetime_usd` immediately restores headroom.
To start a fresh accounting period (e.g. a new billing month), the lifetime
total is the cumulative sum of all settled sessions; you raise the ceiling, you
do not zero the ledger.

**Corrupt-ledger recovery.** `api_budget.load_ledger` never raises on a
truncated or malformed `api-budget.json`. On a `JSONDecodeError`, `OSError`, or
a wrong-typed field, the file is moved aside to a timestamped sibling
`api-budget.json.corrupt-<UTC-timestamp>` (e.g.
`api-budget.json.corrupt-20260725T020000Z`) and a fresh empty ledger is used
for that run — the corrupt original is preserved on disk for forensics, and a
loud `ERROR` log line names the quarantined path. After a quarantine, the
ledger starts fresh (empty `days`/`sessions`, `lifetime_usd: 0`); GitHub labels
and `state.json` are untouched. To recover: find the quarantined file
(`Get-ChildItem .var\charlie-work\api-budget.json.corrupt-*`), extract any
forensic value, and move it aside — do not delete it silently. Corruption is
almost always a process kill mid-write; the atomic temp-file-then-`replace`
write makes a torn final file rare, but does not protect a manually edited file.

### Failure modes

Two api-specific failure kinds, both surfaced as values on the sidecar
(`failure_kind` field of `issue-<n>.api.json`) and never raised:

**`budget_exceeded` (in-flight per-session cap).** When
`budget.max_usd_per_session` is set (`> 0`), each supervision pass accumulates
token usage from the live session's `events.jsonl` and computes USD via
`api_budget.cost_usd` with the active provider's pricing
(`worker._api_session_over_budget`). If cost exceeds the cap, the process tree
is killed via the shared `kill_process_tree` helper (orphan processes swept
too), the sidecar is marked `failure_kind: "budget_exceeded"` via an atomic
write, and a `session_budget_exceeded` event is appended to `state.json`. The
killed session then flows through the **existing** dead-worker reconciliation
on the next pass: with-PR → review/rework; without-PR → re-dispatch via
`routing.select_adapter`, whose preflight naturally decides api-again vs.
fallback. When the cap is `0`/unset the check is entirely dormant (no cost
computation beyond what settlement already does). Non-api workers are never
budget-evaluated.

**`provider_auth` (dead/invalid API key).** When an api session exits, its log
tail is matched against 401/403/authentication/invalid-api-key patterns
(`claude_code._PROVIDER_AUTH_PATTERN`, api-kind only, checked **before** generic
throttle markers so a dead key is never mislabeled as a throttle). On a match,
`failure_kind` is set to `provider_auth` and the provider enters a cooldown
reusing the existing `throttled_until` state mechanism
(`state.set_throttled_until`) with the quota-exhaustion constant — **24 hours**,
not the 15-minute rate-limit window, because a dead key will not self-heal.
While `throttled_until` is in the future, `routing._api_preflight` returns
`fallback:cooldown` and the issue routes to `fallback_adapter`; the dispatch
throttle gate also defers. Recovery:

1. **Check the key env var** named by the active provider's `api_key_env` is
   present and valid on the host (e.g. `$env:MOONSHOT_API_KEY` is set and the
   key is not expired/revoked).
2. Then either **let the 24h cooldown lapse** (`is_throttled` returns false once
   `now >= throttled_until`) or **clear the throttle state** by setting
   `throttled_until` to `null` in `.var/charlie-work/state.json` (atomic write —
   never edit in place while a pass is running).

`doctor` surfaces both failure kinds: its launched-sessions probe appends
`provider-auth: <issues>` and `budget-exceeded: <issues>` fragments to the
session-record summary when any dead sidecar carries those kinds.

### Reading routing decisions

Every routing decision is appended to that issue's `adapter_history` in
`state.json` by `routing.record_adapter_choice`:

```json
{"ts": "2026-07-25T02:00:00Z", "kind": "api", "provider": "kimi-k3", "reason": "policy:rework"}
```

`kind` is the adapter name (`api`, `devin-shell`, `claude-code`, …). `provider`
is the api provider key for `kind == "api"` and empty for non-api adapters.
`reason` is either a `policy:*` string (a routing rule matched and preflight
passed) or a `fallback:*` string (an api preflight check failed and the
fallback adapter was chosen). The complete set, enumerated from
`routing.select_adapter` / `routing._api_preflight`:

**`policy:*` (api chosen, preflight passed):**

| Reason | When |
|---|---|
| `policy:rework` | `rework=True` (rework dispatch). Evaluated first, so a rework issue carrying `complexity:high` still routes here. |
| `policy:complexity` | First pass (`rework=False`) and the issue carries the `complexity:high` label (from `config.labels.complexity_high`, default `complexity:high` — a routing hint, **not** a workflow state label, so it never blocks dispatch selection). |
| `policy:default` | No api candidate rule matched; the default adapter (`dispatch`-level) is used with no provider. |

**`fallback:*` (api preflight failed → `fallback_adapter`):**

| Reason | Failing check |
|---|---|
| `fallback:disabled` | `api_worker.enabled` is false. |
| `fallback:auth` | The active provider's `api_key_env` env var is missing/empty on the host. |
| `fallback:budget` | Daily headroom exhausted (`spent_today + reserve > max_usd_per_day`) **or** lifetime headroom exhausted (`lifetime_usd` reached). |
| `fallback:cooldown` | Provider is in a throttle cooldown (`throttled_until` in the future — e.g. after a `provider_auth` or quota-exhaustion classification). |
| `fallback:concurrency` | Live api session count (`adapter_kind == "api"`, alive) has reached `max_concurrent_sessions`. |

Preflight checks run in the order above; the **first** failing check wins, so
the recorded `fallback:*` reason identifies the single binding constraint. To
inspect an issue's routing history:

```powershell
Get-Content .var\charlie-work\state.json | ConvertFrom-Json | Select-Object -ExpandProperty issues | ConvertTo-Json -Depth 4
```

and read `adapter_history` for the issue number.

## Needs-attention notification and detection latency

charlie-work has no daemon — detection is strictly invocation-cadence-bound. A stalled session found on pass N is invisible until pass N+1 runs, and nothing runs it automatically unless you schedule it. The pluggable `notify` layer (configured under `notify:` in `orchestrator.config.yaml`) turns health transitions (STALLED / RUNAWAY / DEAD / escalated-to-human) into outbound signals an operator actually sees — webhook, desktop toast, shell command, or file.

**Detection latency = invocation cadence.** There is no background daemon; this design does not add one. If you run `charlie bash-rats` (or `charlie fleet bash-rats` for multi-repo) manually every hour, a stalled session can go undetected for up to an hour. For timely detection, schedule periodic invocation:

- **cron (Linux/macOS)**: See `examples/schedule/charlie-fleet.cron` for a commented crontab entry. Adjust the interval (e.g., `*/5 * * * *` for every 5 minutes) based on how quickly you need to detect issues.
- **Windows Task Scheduler**: See `examples/schedule/charlie-fleet-task.xml` for an XML template. Import with `schtasks /create /xml charlie-fleet-task.xml /tn "charlie-work fleet"` and adjust the `<Interval>` (e.g., `PT5M` for 5 minutes).

The `notify` layer is disabled by default (`notify.enabled: false`). To enable it, add a `notify:` section to your `orchestrator.config.yaml` — see `examples/notify.config.yaml` for sink options (webhook, desktop, shell, file). The layer emits a digest once per pass if any issue's health state changed since the last pass, comparing against a dedicated per-issue `health` field in `state.json` (not the capped `events` log). Sink failures never fail the pass — errors are returned as values and logged.
