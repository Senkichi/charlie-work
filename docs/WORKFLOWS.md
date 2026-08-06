# Workflows

End-to-end operator loops with exact CLI commands. For what each command
does internally, see [ARCHITECTURE.md](ARCHITECTURE.md); for recovery
procedures, see [RUNBOOK.md](RUNBOOK.md).

All commands below assume you're in the consumer repo root with
`orchestrator.config.yaml` present (or accept the dataclass defaults) and
`charlie` resolved via `uv run charlie ...` (or a bare `charlie` if
the venv is activated). Add `--json` for machine-readable output and
`--dry-run` to suppress mutating `gh` calls, the fleet self-deploy FF-pull and
`uv sync`, the runner scale-event and pool-sample writes, and the cross-family
review report/prompt writes. It does **not** suppress local state writes in
general or worker adapter subprocesses (see the
scope caveat in
[QUICKSTART.md](QUICKSTART.md#5-first-cycle-intake--dispatch--review--merge)).

## (a) Devin manual adapter loop

The operator-confirmed default (`devin.adapter: manual` — either shipped
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

# 8. Read .var/charlie-work/prs/pr-123/review-prompt.md (and the
#    cross-family report if cross_family.enabled), do the adversarial
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

Set `devin.adapter: devin-shell` (as in
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
python -c "from pathlib import Path; from charlie_work.devin_shell import read_session_records, is_session_alive; recs = read_session_records(Path('.var/charlie-work/dispatches/sessions')); [print(r.issue_number, is_session_alive(r)) for r in recs]"

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

Set `devin.adapter: claude-code` (as in
`examples/orchestrator.config.claude-code.yaml`).
Each worker gets an isolated git worktree (via `worktree.create_worktree()`,
junction-linked to a shared `.venv` when `venv_source` is given) and a
headless `claude -p --permission-mode acceptEdits` process launched inside
it — promoting the emergent empericus pattern (human hand-assembles a
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

## Cross-family review flow

Runs a non-Claude model (`cross_family.model`, default `codex` via the Devin
CLI) against a PR's diff as a second opinion — its findings are **leads, not
verdicts**; the primary reviewer must verify each against live code and
never let it gate a merge on its own (`cross_family.py`'s `_CAVEAT` text,
reproduced verbatim into every generated report).

Enabled by config (`cross_family.enabled: true`, as in
`examples/orchestrator.config.devin.yaml`) — runs automatically inside
`review()` for every non-draft PR:

```powershell
charlie why-charlie-hate --pr 123
# cross-family-review.md is written to
# .var/charlie-work/prs/pr-123/cross-family-review.md
# and its section is folded into review-prompt.md automatically
```

Or override per-call regardless of config default:

```powershell
charlie why-charlie-hate --pr 123 --cross-family      # force on
charlie why-charlie-hate --pr 123 --no-cross-family   # force off
```

A non-empty successful report is **reused** on repeated `review()`/`loop()`
passes over the same PR (no repeat model spend) — but a failed run's
`(UNAVAILABLE)` stub (or one missing its head-SHA marker) is never reused.
`loop()` forces regeneration instead, bounded by
`cross_family.max_regen_attempts` (default `2`) attempts **per head SHA** —
a new push resets the budget, since a new head has never been tried.
Setting it to `0` disables forced regeneration entirely, so an unusable
report escalates on the first pass instead of being retried. Past the
budget the issue escalates to `agent:human-needed` rather than retrying
forever (see [ARCHITECTURE.md](ARCHITECTURE.md#invariants) and
[RUNBOOK.md](RUNBOOK.md#handling-agenthuman-needed-escalations)). Manually
running `charlie why-charlie-hate --pr <n>` calls `review()` directly and
is not subject to that budget — it always attempts regeneration.

## Fleet dispatch loop

Fleet-level commands compose the single-repo loops across every repo in the
user-level registry (`fleet_registry.py`) under one global concurrency budget
(`fleet.global_max_concurrent_sessions`). A repo joins the registry
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

## Spec-review flow

An explicit, on-demand cross-family pass over a design doc or spec file —
independent of `cross_family.enabled` (that flag only governs the automatic
PR-review path) and always runs when invoked:

```powershell
charlie why-charlie-hate-spec --file docs/SPEC.md
```

Writes `spec-<slug>-prompt.md` and `spec-<slug>-review.md` under
`.var/charlie-work/cross-family/`, using the same non-Claude model
configured under `cross_family.*`. Use this before committing to an
implementation plan, the same way you'd request a second opinion on a
design doc from a different reviewer.
