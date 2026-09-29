# charlie-work

charlie-work turns GitHub issues into merged pull requests. A deterministic orchestrator picks up eligible issues, sends each one to an isolated LLM worker, puts the resulting PR through review, and either merges it or hands it to a human.

## Language

### Issue lifecycle

**Ready**:
An issue that a human has marked as eligible for automation. It is the only way into the lifecycle.
_Avoid_: Approved, enabled, armed

**Dispatchable**:
A Ready issue with no active or terminal state and no gate holding it back. Only dispatchable issues get a worker.
_Avoid_: Eligible, pending

**Queued**:
The issue has been handed to the manual adapter, but nothing has confirmed that a worker started.
_Avoid_: Scheduled

**In progress**:
A worker has actually been launched against the issue.
_Avoid_: Running, active (active is the name for the whole set of in-flight states)

**PR open**:
The issue's worker has published a pull request. The issue stays in this state after approval until the PR merges.

**Reviewing**:
The issue's PR is being reviewed.

**Needs rework**:
The review asked for changes and the rework cap has not been hit yet.

**Done**:
The issue's PR merged. All other lifecycle states are cleared.
_Avoid_: Closed, shipped

**Review ready**:
The successful end state for an issue from a local source that has no remote. The worker's branch is the deliverable, and a human reviews and merges it. This is not an escalation.

**Active state** / **Terminal state**:
The two groups of lifecycle states. An active state means work is in flight. A terminal state means automation has stopped touching the issue. An issue in either group is not dispatchable.

**Escalation**:
Handing an issue from automation to people. It can be caused by a blocked verdict, an exhausted cap, or a mechanical failure, and it always ends in Human needed or Operator queue.
_Avoid_: Blocked (blocked is only a verdict), stuck

**Human needed**:
The terminal state for an escalation that needs a human judgment call.
_Avoid_: Blocked

**Operator queue**:
The terminal state for a mechanical escalation, such as an infrastructure failure. It keeps Human needed free for real judgment calls.

**Dependency gate**:
Keeps an issue out of dispatch while it declares an open blocking issue.

**Escape hatch**:
A label that only a human applies to waive one gate for one issue or PR, for example the collect-gate exemption or the cross-repo override. The automated state machine never adds or removes it.
_Avoid_: Override label, bypass

### Dispatch and workers

**Orchestrator**:
The deterministic Python hub. It reads state, decides what happens next, and launches workers. It never does the coding itself.
_Avoid_: Agent, bot

**Worker**:
One hermetic LLM coding process scoped to a single issue, running in its own worktree.
_Avoid_: Agent (agent is only the label namespace), session, spoke (only for the architecture diagram), bot

**Operator**:
The human who runs charlie-work, or an interactive session acting for them. The operator arms issues, records verdicts, applies escape hatches, and resolves escalations.
_Avoid_: Orchestrating agent, user, agent

**Adapter**:
The mechanism the orchestrator uses to launch a worker on a given harness (manual, command, Devin, Claude Code, API). It returns immediately and never waits for the worker to finish.
_Avoid_: Harness, backend, driver

**Sidecar**:
The durable record of one launched worker. The orchestrator reads it back to decide the worker's fate.
_Avoid_: Manifest, session file

**Rework**:
A new worker run against an existing PR to address review findings or conflicts.
_Avoid_: Retry, fix-up

**Rework cap**:
The maximum number of rework rounds allowed on one PR. When it runs out, the issue goes to the rescue tier or is escalated.

**Rescue tier**:
One optional last attempt after the rework cap runs out and before a human gets involved: rework by a stronger model, then review by a model from a different family.
_Avoid_: Salvage

**Salvage**:
Publishing a PR from commits a worker left stranded, for example because it was superseded or exited without pushing.
_Avoid_: Rescue, park

**Park**:
Keeping a worker's commits on their branch in a repo with no remote, so the issue can end in Review ready. This is the no-remote counterpart of salvage.
_Avoid_: Salvage

### Review and merge

**Janitor gate**:
The deterministic check a PR must pass before any LLM review starts. It costs nothing in model spend.
_Avoid_: Pre-check, lint gate

**Review packet**:
A snapshot of the diff, checks, and metadata that a reviewer judges a PR against.

**Reviewer**:
A worker launched in the review role, not the coding role.

**Verdict**:
The result of reviewing one PR head: approved, request changes, or blocked. The recorded verdict is what authorizes a merge.
_Avoid_: Decision, outcome

**Blocked**:
The verdict saying a PR cannot proceed without a human. It causes an escalation to Human needed. It is not a lifecycle state of its own.
_Avoid_: Using it as a label or state name

**Merge path**:
The steps that take an approved PR to merged, either directly or through a merge queue handoff.
_Avoid_: Merge lane

**Local path**:
The review-and-merge flow for a repo that has no remote and no PRs. Its successful end state is Review ready.
_Avoid_: Local lane

**Merge queue handoff**:
Passing an approved PR to the external merge queue instead of merging it directly. A handed-off PR is not yet merged.
_Avoid_: Auto-merge

**Reconcile**:
Finding and repairing drift between the lifecycle state and what really happened on GitHub, for example a PR someone merged by hand.
_Avoid_: Sync, mop-up (mop-up is the command name)

### Passes, fleet and supervision

**Repo pass**:
One orchestrator iteration over one repo: reconcile, supervise, dispatch, review, merge. It runs once and does not repeat.
_Avoid_: Loop, a bare "pass" without saying which kind

**Fleet**:
The layer that runs repo passes across every registered repo under one shared concurrency budget.
_Avoid_: Supervisor

**Fleet pass**:
One round across every registered repo in the fleet, made up of one repo pass per repo.

**Lane**:
One repo's share of a fleet pass, run alongside the other repos' lanes.
_Avoid_: Using "lane" for the local path or the merge path

**Loop**:
A long-running process that repeats passes on a fixed cadence and can relaunch itself after a self-deploy.
_Avoid_: Using "loop" for a single pass

**Supervisor**:
The sweep inside each repo pass that checks worker health and applies tripwires.
_Avoid_: Fleet, daemon

**Worker health**:
How the supervisor classifies a live worker: healthy, slow, stalled, runaway, dead, or orphaned.

**Tripwire**:
A configured limit on a worker (stall, wall-clock time, no-progress loop, cost) that triggers an intervention when crossed.
_Avoid_: Watchdog, alarm

**Restart intensity**:
How many times a stuck worker has been redispatched automatically. When it hits its cap, the issue is escalated instead of redispatched again.

### State and instrumentation

**Lifecycle state**:
An issue's position in the lifecycle, as recorded in its GitHub labels. That is the only authoritative record. Everything else is derived from it.
_Avoid_: Status

**State cache**:
The orchestrator's local mirror of lifecycle and PR state. It is derived and never takes precedence over the labels.
_Avoid_: State, database

**Event**:
An append-only record of something the orchestrator did or observed. The unlimited history of events is the audit trail.
_Avoid_: Log line

**Correlation ID**:
The identifier that ties together every event from one pass.

### Owned by ci_fleet (defined there, not here)

**Runner allocation**:
Starting and parking CI runners that are already registered, to shift capacity between repos. charlie-work only supplies its config.

**Parked runner**:
A runner that is still registered but has been stopped on purpose. It reports as offline.
