# Workflows

End-to-end operator loops with exact CLI commands. For what each command
does internally, see [ARCHITECTURE.md](ARCHITECTURE.md); for recovery
procedures, see [RUNBOOK.md](RUNBOOK.md).

All commands below assume you're in the consumer repo root with
`orchestrator.config.yaml` present (or accept the dataclass defaults) and
`charlie` resolved via `uv run charlie ...` (or a bare `charlie` if
the venv is activated). Add `--json` for machine-readable output and
`--dry-run` to suppress mutating `gh` calls, the fleet self-deploy FF-pull and
`uv sync`, and the runner scale-event and pool-sample writes. It does **not**
suppress local state writes in general or worker adapter subprocesses (see the
scope caveat in
[QUICKSTART.md](QUICKSTART.md#5-first-cycle-intake--dispatch--review--merge)).

## (a) Devin manual adapter loop

The operator-confirmed default (`worker.harness: manual` — either shipped
example profile). No subprocess launches a worker; you paste the rendered
prompt into a Devin session by hand.

```powershell
# 1. Preflight once per session
charlie doctor

# 2. See what's ready
charlie roll-call --json

# 3. Write worker prompts + issue snapshots for every automated-ready issue
charlie intake

# 4. Select and "dispatch" a wave — writes session-manifest.json and
#    worker-prompt.md per issue, labels agent:queued (manual adapter
#    never promotes straight to agent:in-progress — see
#    ARCHITECTURE.md#invariants)
charlie work --limit 3

# 5. Open .var/charlie-work/dispatches/session-manifest.json.
#    For each session listed, open a Devin session and paste in the
#    contents of that issue's worker-prompt.md
#    (.var/charlie-work/issues/issue-<n>/worker-prompt.md).

# 6. ...worker works, opens a PR referencing the issue...

# 7. Generate the adversarial review packet
charlie why-charlie-hate --pr 123

# 8. Read .var/charlie-work/prs/pr-123/review-prompt.md, do the adversarial
#    review yourself (or via an orchestrating Claude session using
#    prompts/orchestrator.md as its operating brief), write a summary,
#    then record the decision
charlie verdict --pr 123 --decision approved --summary-file review.md
#    or:
charlie verdict --pr 123 --decision request_changes --summary-file review.md --comment
charlie verdict --pr 123 --decision blocked --summary-file review.md

# 9. Merge once approved and required checks are green
charlie ship-it --pr 123
```

Dispatch is dependency-aware by default (issues with open blockers are held
back; the rest go most-unblocking-first, `dispatch.order` — default `oldest` —
breaking ties). To pin an explicit order or subset instead:

```powershell
charlie work --issues 565,570,572
```

Numbers that aren't currently dispatchable (already active, terminal, or
missing the `ready` label) are silently skipped — check the returned
`selected_count` vs. `attempted_count` in `--json` output to confirm what
actually went through.

To run intake + dispatch + review + conditional-merge in one pass (steps
3, 4, 7-9 collapsed, looping over every open linked PR):

```powershell
charlie bash-rats --limit 3
```

## (b) Devin shell adapter loop

Set `worker.harness: devin-shell` (as in
`examples/orchestrator.config.devin.yaml`). Non-blocking, headless:
`launch_devin_session()` spawns `devin --prompt-file <path> --print` via
`Popen` and returns immediately, writing a sidecar JSON
(`sessions_dir/issue-<n>.json`) so the orchestrator can find it again without
blocking on the worker.

```powershell
# 1. Preflight, confirm the devin CLI itself is reachable. --adapter-probe
#    runs devin_shell.probe_devin() and surfaces stale/failed sessions.
charlie doctor --adapter-probe

# 2-3. Same as the manual loop: status, intake
charlie roll-call --json
charlie intake

# 4. Dispatch launches a `devin --print` process per selected issue (instead
#    of only writing a manifest) and labels agent:in-progress immediately —
#    a real worker was launched, not just a manifest written
charlie work --limit 3

# 5. Poll for liveness/completion instead of pasting anything by hand.
#    `charlie doctor --adapter-probe` reports failed/exited sessions, or
#    read the sidecars directly:
python -c "from pathlib import Path; from charlie_work.devin_shell import read_session_records; from charlie_work.worker_fate import is_alive; recs = read_session_records(Path('.var/charlie-work/dispatches/sessions')); [print(r.issue_number, is_alive(r.pid, r.process_start_time)) for r in recs]"

# 6-9. Same why-charlie-hate/verdict/ship-it sequence as the manual loop
charlie why-charlie-hate --pr 123
charlie verdict --pr 123 --decision approved --summary-file review.md
charlie ship-it --pr 123
```

Respect the documented single-threaded SQLite-contention ceiling on the
Devin CLI's own session store — do not dispatch large parallel waves against
this adapter (see
[RUNBOOK.md](RUNBOOK.md#session-limit--quota-discipline)).

## (c) Claude Code worktree adapter loop

Set `worker.harness: claude-code` (as in
`examples/orchestrator.config.claude-code.yaml`).
Each worker gets an isolated git worktree (via `worktree.create_worktree()`,
junction-linked to a shared `.venv` when `venv_source` is given) and a
headless `claude -p --permission-mode acceptEdits` process launched inside
it — promoting an emergent manual pattern (human hand-assembles a
worktree + pastes a prompt into an interactive `claude` session) into code.
The worktree checkout itself carries the repo's tracked `.claude/settings.json`
permissions/hooks for free — no separate config plumbing needed for that.

```powershell
# 1. Preflight — with claude-code.yaml as your config, worker_template is
#    already worker_claude_code.md. --adapter-probe runs claude_code.probe_claude().
Copy-Item ..\charlie-work\examples\orchestrator.config.claude-code.yaml orchestrator.config.yaml
charlie doctor --adapter-probe

# 2-3. status, intake — identical to the other two loops
charlie roll-call --json
charlie intake

# 4. Dispatch creates one worktree + one headless claude -p process per
#    selected issue, writes an issue-<n>.claude.json sidecar, and labels
#    agent:in-progress on a confirmed launch
charlie work --limit 3

# 5. Worker runs inside its own worktree, opens a PR referencing the issue,
#    the orchestrator's dispatch loop or a periodic status check picks up
#    completion via the PR appearing on GitHub. `charlie roll-call` now
#    includes a `workers` health section (classify_worker_health over each
#    live sidecar) so you can see STALLED / RUNAWAY / DEAD workers per pass.

# 6-8. Same why-charlie-hate/verdict/ship-it sequence
charlie why-charlie-hate --pr 123
charlie verdict --pr 123 --decision approved --summary-file review.md
charlie ship-it --pr 123
```

**Worktree cleanup after the PR merges or is abandoned** must go through
`worktree.remove_worktree()` (or its exact manual teardown order) — never a
bare `git worktree remove --force` or `rm -rf` if a `.venv` junction is
present. See
[RUNBOOK.md](RUNBOOK.md#worktree-cleanup-gone-wrong-junction-hazard) for the
full hazard writeup and manual recovery steps.

## (d) opencode worker harness

`worker.harness: opencode` -- or, more usefully, an `opencode` entry in
`worker.fallbacks` -- launches a headless `opencode run` session through the
same worktree/sidecar/terminal-watcher stack as a claude-code worker
(`opencode_worker.launch_opencode_worker`; sidecar `issue-<n>.opencode.json`).
Worker-only: opencode is not a reviewer harness.

```yaml
worker:
  harness: devin-shell
  model: swe-2-high
  fallbacks:
    - {harness: devin-shell, model: gemini-3-8-flash-high}
    - {harness: opencode, model: glm-5.3-flash}   # OpenCode Go subscription

opencode:            # all optional
  provider: opencode-go   # prefixed onto a bare model -> opencode-go/glm-5.3-flash
  variant: ""             # --variant (low | high | max for glm-5.3-flash)
  worker_env: {}          # merged last; operator values win
```

What the launcher pins, and why:

- `opencode run --auto --format json --model <provider>/<model>`, prompt on
  stdin. `--auto` matters: without it `run` auto-*rejects* permission requests
  (e.g. `external_directory`) instead of prompting. An empty model is refused
  as a launch error rather than falling back to opencode's last-used model.
- `OPENCODE_CONFIG_CONTENT` allows every tool (parity with the claude-code
  worker's `bypassPermissions`) and turns off autoupdate, sharing and
  snapshots.
- A per-worker `XDG_DATA_HOME` (`<sessions_dir>/opencode-data/issue-<n>`):
  concurrent runs sharing one `opencode.db` die at startup with `database is
  locked`. The host `opencode auth login` credentials are forwarded in memory
  via `OPENCODE_AUTH_CONTENT` (child env only -- never a sidecar, log or argv).
- `OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1`: opencode otherwise injects the
  operator's personal `~/.claude/CLAUDE.md` into the worker. The repo's own
  `CLAUDE.md`/`AGENTS.md` and `.claude/skills` still load.
- `OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS=1800000`: the default 2-minute
  bash-tool timeout kills a full local test run.

**Usage-limit cooldowns.** OpenCode Go meters a rolling 5-hour, a weekly and a
monthly budget, and opencode logs only "Usage limit reached" -- never which one
tripped or when it resets. When an opencode death classifies `quota_exhausted`,
`opencode_limits.go_quota_reset` (`AdapterFateProfile.quota_reset`) probes the
Go API once (`GET /models`, then a 1-token chat request with the host
credential): a `GoUsageLimitError` 429 gives the real reset (`retry-after`, else
the `limitName` window), so the ledger restricts the entry until exactly then; a
200 means the limit already reset (15-minute cooldown). No credential, a
network error or any other answer keeps the fixed 24h. One probe per 2 minutes
per process.

`charlie doctor --adapter-probe` probes the opencode binary whenever opencode is
the primary *or* any fallback entry, so a missing binary surfaces before a
failover needs it.

## Cross-family review flow (removed)

The automatic non-Claude "second opinion" pass this section used to document
— which ran a `codex`-via-Devin-CLI model against every non-draft PR's diff
inside `review()` — was deleted wholesale by Phase 2 of the role-config
refactor, along with its regeneration-budget bookkeeping (issue #1099) and
its `agent:human-needed` escalation reason. There is no config knob that
reinstates it.

The bounded rescue tier (fires only after rework-cycle exhaustion, not on
every PR) still runs a comparable non-Claude second-opinion pass internally,
but it is not an operator-invoked workflow — see
[RUNBOOK.md](RUNBOOK.md#handling-agenthuman-needed-escalations) for how that
tier's outcome surfaces.

## Role chain (model waterfall)

Both roles may name ordered fallbacks after their primary (issue #2086):

```yaml
worker:
  harness: devin-shell
  model: swe-2
  fallbacks:
    - {harness: claude-code, model: claude-sonnet-5-5}
reviewer:
  harness: claude-code
  model: claude-opus-5-5
  fallbacks:
    - {harness: devin-shell, model: swe-2, effort: high}
```

- **Selection.** Every worker launch (fresh, remote rework, local rework --
  all through the one launch permit) and every reviewer launch (remote and
  local review lanes) calls `role_selection.select_role_entry`: the first
  chain entry whose `(harness, model)` is not restricted in the **quota
  ledger** (`<fleet_dir>/role_quota_ledger.json`) at launch time. The chosen
  entry is stamped on the session sidecar (`role_entry`: role, harness,
  model, `chain_index`); a launch past the primary emits
  `role_fallback_selected` listing each skipped entry and its `until`.
- **Learning.** When a worker or reviewer death is classified
  `rate_limited` / `quota_exhausted` (worker failure classification, the
  stalled rate-limit defer, the stalled-review sweep, a launch-time reviewer
  quota hit), the ledger restricts the `(harness, model)` stamped on *that*
  session until the classified reset. Writes are atomic and monotonic
  (a shorter window never shortens an active one); entries only expire.
  Sessions launched before this shipped carry no stamp and record nothing.
- **Per-repo windows.** `throttled_until` (workers) and `reviewer_quota`
  (reviewers) keep blocking for a chained role unless the ledger explains
  them -- a window that ends no later than the latest ledger restriction on
  a chain entry is *covered*, and the launch proceeds on the selected entry.
  An operator hold or any other unexplained window still blocks. When every
  entry is restricted the launch defers exactly as the per-repo window
  always did (`provider_throttled` / `reviewer_quota_probe_backoff`, plus
  `chain_retry_at`).
- **Unchanged without fallbacks.** A chain of length 1 never reads the
  ledger: per-repo throttles, the reviewer probe backoff and its single-probe
  recovery behave exactly as before. The ledger is still written from its
  sessions, so a chained repo elsewhere in the fleet learns from them.
- **Downstream.** Reap and classification read the harness from the
  sidecar's own type, so a fallback session is classified as what it ran on.
  Worker prompts use the slash-command skills loop only when every chain
  harness can load every declared skill.

## Fleet dispatch loop

Fleet-level commands compose the single-repo loops across every repo in the
user-level registry (`fleet_registry.py`) under one global concurrency budget
(`fleet.global_max_concurrent_sessions` for workers;
`fleet.global_max_concurrent_reviews` for reviewers). A repo joins the registry
automatically the first time any command loads its config (`touch_repo()`), so
there is no explicit "register" step — run charlie once against a repo and it is
enrolled.

```powershell
# Aggregate roll-call across every registered repo (read-only, dry-run per repo)
charlie fleet status

# Dispatch-only wave across all registered repos, sharing the global budget
charlie fleet work --limit 3

# Full intake -> work -> review -> merge pass across all registered repos
charlie fleet bash-rats --limit 3

# Restrict either command to specific repos (overrides the oldest-last_seen order)
charlie fleet work --repos owner/repo-a,owner/repo-b
```

`fleet work` / `fleet bash-rats` walk the registry oldest-`last_seen`-first (or
the explicit `--repos` order), applying the per-repo
`dispatch.max_concurrent_sessions` cap and the fleet-global cap at every
dispatch path. Per-repo errors (a moved/broken repo) are isolated — one repo
failing never aborts the rest of the sweep. Each pass ends with a consolidated
**attention digest** (count of needs-attention events + orphan-sweep calls)
printed in the human-readable output and available under `data.digest` in
`--json`.

The fleet budget bounds worker *count*, not CPU/RAM — respect the cross-repo
xdist discipline in
[RUNBOOK.md](RUNBOOK.md#fleet-cross-repo-dispatch) when running many repos on
one host.

The on-demand `charlie why-charlie-hate-spec` command (an explicit
cross-family pass over a design doc or spec file, independent of the
PR-review path above) was deleted along with the rest of this subsystem;
there is no replacement command for reviewing a standalone spec file.
