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
   `3` in both shipped example profiles) rounds of `request_changes` without
   converging. This usually means the issue brief was wrong, ambiguous, or
   the acceptance criteria are unimplementable as written — not that "one
   more rework round" will fix it.

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
docstring). `record_review()` recomputes the prior-cycle
count fresh from `state["events"]` on every call — counting `record_review`
events for that exact PR number with `decision == "request_changes"` — so
there is no separate mutable counter to get out of sync. Practical
implication: **the count is per-PR, not per-issue** — if a rework cycle
requires closing a PR and opening a fresh one for the same issue (branch
unrecoverable), the cycle count resets, because it's keyed by
`pr_number`. Worker prompts (`rework.md`) explicitly instruct workers to
update the existing PR rather than open a new one for exactly this reason.

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
- **Do not give each worktree its own full venv** to dodge this — it costs disk
  and sync time and does not touch the CPU/RAM ceiling. The shared-venv junction
  (`claude_code.venv_source`) is correct; bound parallelism instead.

## Session-limit / quota discipline

Both non-blocking adapters (`devin_shell.py`'s `launch_devin_session()`,
`claude_code.py`'s `launch_claude_worker()`) are `Popen`-based and return
immediately; they do not enforce a concurrency limit themselves. The Devin CLI's session store (`sessions.db`) is
documented (per the extraction dossier) as **single-threaded / SQLite-
contention-limited** — do not assume parallel local Devin-shell dispatch
works reliably; serialize dispatch waves or shard across machines/profiles
if you need real concurrency. Practical discipline until an enforced
concurrency cap exists in code:

- Keep `dispatch.default_limit` (default `3`) conservative relative to your
  actual quota/rate-limit budget for whichever worker CLI you're driving.
- Before dispatching a new wave, check for still-alive sessions:
  `devin_shell.read_session_records(sessions_dir)` +
  `devin_shell.is_session_alive(record)` (Windows PID liveness via ctypes
  `OpenProcess`+`GetExitCodeProcess`, since `os.kill(pid, 0)` is unreliable on
  Windows; `os.kill(pid, 0)` on POSIX), or the `claude_code` equivalents
  (`read_worker_records`, checking `record.pid`).
- `doctor`'s `probe_devin()` / `probe_claude()` helpers are cheap
  `--version` checks — use them to confirm the CLI is reachable at all
  before burning a dispatch wave on a broken PATH.
