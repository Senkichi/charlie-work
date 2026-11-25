# Foreman Layer: Outcome-Owning Manager Above the Fleet — Design

**Status:** Accepted by the operator 2026-11-25 and revised the same day after a design
review (section 20) · **Tracking issue:** 3112

## 1. Problem

charlie-work turns a Ready issue into a merged PR without a human in the loop. Nothing
turns an *outcome* ("migrate every GitHub call onto the guarded transport") into the
Ready issues that deliver it, watches them land, and re-plans when one escalates. Today
the operator does that by hand, usually from an interactive session that holds the plan
in its context and loses it when the session ends.

Cognition's Devin Bots (2026-11) package exactly that missing role: a persistent manager
that owns an outcome, keeps memory as files, starts a child session per piece of work,
wakes on a schedule or an event, and keeps a "needs you" list for the human. The
child-session half of that product is what this repo already is. This design adds the
manager half without giving up the property the repo is built on: the orchestrator never
thinks.

## 2. What already exists (verified against the tree, not assumed)

- **Work execution.** One Ready issue → one worker → one PR → janitor gate → review →
  merge path, with the dependency gate, rework cap, rescue tier and escalation.
- **An append-only audit trail.** `events.db` has an autoincrement `id`, plus indexed
  `kind`, `ts`, `issue_number`, `pr_number`. `query_events()` filters by `since`
  (timestamp) but has **no `after_id`**, so there is no exact cursor yet.
- **A local, lock-free read model.** Each repo pass writes `status-snapshot.json`
  (`issues[]` with `labels` and `dispatchable`, `prs[]`, `workers[]`). Reading it costs
  no GitHub call.
- **Operator surfaces.** `operator-queue`, `review-queue`, `unescalate`, `claim`, the
  read-only dashboard with its "needs me" zone (ADR-0008), and the `notify` sinks
  (webhook, desktop, shell, file). `notify` only carries worker-health digests today.
- **Launch hygiene worth reusing.** The role chain and quota ledger, `tool_scope`
  (tool allowlist plus MCP isolation), `env_sanitize`, `subprocess_runner.run_captured`
  (errors as values), the host launch port with a fake, `outbound_body_guard` (refuses a
  secret-shaped title or body before it leaves the process), `fleet_pause`.
- **Body-contract parsers.** `github_body_scan.parse_blockers`,
  `detect_prose_only_dependencies` and `citation_check.parse_citations` define what the
  dispatcher reads from an issue body.

Three gaps this design has to close, each found by looking:

1. **The orchestrator cannot file an issue.** No create-issue request exists in the
   GitHub client; filing has always been the operator's job.
2. **The issue-contract lint lives outside the repo.** It is an operator-local script
   that imports the parsers above. Anything in-tree that files issues needs it in-tree.
3. **`CONTEXT.md` defines Ready as something only a human marks.** A manager that arms
   its own issues changes that definition and needs an ADR.

## 3. Goals and non-goals

**Goals**

- An operator writes an outcome once; the fleet keeps moving toward it for days or
  weeks with no interactive session open.
- Zero tokens while nothing has changed.
- Every side effect is applied by deterministic code that can refuse it.
- The operator is asked only for decisions that are theirs, each with a recommendation
  and the exact command that enacts it.

**Non-goals**

- Sharing, access levels, per-foreman secrets, a canvas view, a template marketplace.
  This is a single-operator tool.
- A foreman that codes, reviews, records verdicts or merges. All work still goes through
  an issue.
- Replacing the interactive operator session. The foreman covers the hours no session is
  open.
- iMessage. Inbound Slack is optional and last (section 11).
- Foreman-to-foreman messaging. Two foremen coordinate the way two operators would:
  through issues and `Blocked by` edges. An issue has exactly one owning foreman.

## 4. Design overview

A **Foreman** is a directory plus a deterministic pass. No long-lived model process
exists. Each time the pass finds a reason to wake a foreman, it runs one **Turn**:

```
 foreman pass (deterministic, every few minutes, zero tokens when idle)
   │  wake predicate per foreman: lifecycle fingerprint changed | inbox unread
   │                              | schedule due | quiet too long
   ▼
 ┌────────── Turn ───────────────────────────────────────────────────────────┐
 │ Collect (code)   charter + memory + event delta + snapshot + open items   │
 │      │           → briefing.md                                            │
 │      ▼                                                                    │
 │ Decide (model)   read-only tools, no network, no gh                       │
 │      │           → actions.json (schema-validated)                        │
 │      ▼                                                                    │
 │ Apply (code)     authority check → bounds → lint → guards → effect        │
 │                  → events, items, memory, cursor                          │
 └───────────────────────────────────────────────────────────────────────────┘
        │ issues (Ready)                         ▲ events.db, status snapshot
        ▼                                        │
   existing fleet: dispatch → worker → PR → review → merge path
```

The model proposes; code disposes. The Decide step cannot write anything. It returns a
list of **Actions**, and the Apply step either performs each one through an existing
enforcement point or refuses it and says why in the next briefing. This is the same
shape as worker and verdict today: LLM output crosses into the hub only as validated
data.

A fresh process per Turn gives Devin's "Recycle" behaviour for free, bounds context per
Turn, and means a crashed Turn loses nothing that was not already on disk.

## 5. Vocabulary (additions to `CONTEXT.md`)

`CONTEXT.md` lists "bot" as a word to avoid for both the orchestrator and a worker, so
the new role needs its own term.

| Term | Definition | Avoid |
|---|---|---|
| **Foreman** | A delegated operator with a written, narrower authority. It owns one outcome, files and arms the issues that deliver it, and raises items for the human. It never codes, reviews or merges. | Bot, manager agent, planner |
| **Charter** | The operator-authored file that states a foreman's outcome, repos, authority and bounds. Only a human edits it. | Config, prompt |
| **Foreman pass** | One deterministic iteration over every foreman: evaluate the wake predicate, run a Turn where it holds. Runs once and does not repeat. | Tick, loop |
| **Turn** | One Collect → Decide → Apply cycle for one foreman. | Session, run |
| **Briefing** | The deterministic document a Turn hands to the model. | Context, prompt |
| **Action** | One proposed side effect in a Turn's output. It has no effect until Apply accepts it. | Tool call, command |
| **Item** | Something a foreman needs from, or reports to, the operator: a decision, a permission request or a report. | Ticket, alert |
| **Authored issue** / **Tracked issue** | The two kinds of issue a foreman owns. It filed an authored issue and may amend or withdraw it; the operator handed it a tracked issue by label, and it may only watch and comment. | Adopted, child issue |

Two existing definitions change, and ADR-0010 (section 13) records why:

- **Ready**: "An issue that the operator, or a foreman acting under its charter, has
  marked as eligible for automation."
- **Operator**: unchanged, plus one sentence: "A foreman is not an operator; it holds
  only the authority its charter grants."

## 6. Foreman directory and charter

Foremen live in the fleet dir, because a foreman is user-level, like the fleet registry:

```
<fleet_dir>/foremen/<name>/
  charter.yaml       operator-authored (frozen dataclass, field-metadata validation per ADR-0007)
  instructions.md    operator-authored standing instructions; a "template" is just this file
  memory/*.md        foreman-authored, bounded (section 8.4)
  state.json         cursors, fingerprint, counters, foreman state, body hash per authored issue
  tree/              detached worktree of the charter's repo; the only code Decide reads
  journal.jsonl      one dated line per Turn: what it saw, what it did (code-written)
  inbox.jsonl        operator → foreman messages (instructions)
  signals.jsonl      external events (data, never instructions; section 12)
  items.json         open, answered, snoozed and dismissed items
  turns/<turn_id>/   briefing.md, actions.json, applied.jsonl, decide.log (newest 50 kept)

<fleet_dir>/foremen/_shared/
  *.md               operator-authored partials any foreman's instructions can include
```

`instructions.md` may include shared partials by name (`issue-style`, `operator-facts`),
resolved the way `prompt_sections` resolves worker sections, so guidance common to every
foreman lives in one file. Foremen cannot write to `_shared/`. Turn directories beyond
the newest 50 are pruned; the events and the journal are the permanent record.

```yaml
name: the-gang-migrates-the-transport
outcome: >
  Every GitHub call in src/ goes through the guarded transport; the gh CLI path
  remains only as the transport fallback.
repos: ["<owner>/charlie-work"]      # fleet registry key; exactly one in v1
standing: false                      # true = no end state
done_when:
  check: ["uv", "run", "--extra", "dev", "pytest", "-q", "tests/test_transport_coverage.py"]
arming: auto                         # auto | propose
bounds:
  max_open_issues: 5
  max_issues_per_turn: 3
  min_turn_interval_minutes: 30
  max_quiet_hours: 24
  max_turns_per_day: 12
  max_unanswered_days: 7
cadence:
  report_at: "09:00"                 # local time; one scheduled Turn per day
```

Fleet-wide settings sit in the global config layer under a new `foreman:` section:
`enabled` (kill switch, default true), the foreman **role chain** (harness, model,
fallbacks, resolved through the quota ledger like the worker and reviewer roles),
`tools` (default `Read,Grep,Glob`) and `max_model_turns`. The feature ships on; with no
charter on disk the pass does nothing.

**Foreman state** is one of `active`, `paused` (operator), `degraded` (breaker tripped,
section 10), `faulted` (three consecutive Turn failures) and `done`. It is local to the
foreman and is not a lifecycle state.

Two rules keep a half-written charter from acting:

- **A new foreman is created `paused`.** `charlie gang new` scaffolds the directory and
  nothing runs until `charlie gang resume`. The feature is on by default; a charter
  nobody has finished writing is not.
- **A charter names exactly one repo in v1.** `repos` stays a list so the file format
  does not change later, and the validator rejects any other length. Blockers only work
  inside one repo, and Decide and `done_when` each need one tree to stand in.

`state.json` is a cache. Everything in it that describes GitHub (which issues the
foreman owns, what state they are in) is rebuilt from labels and the status snapshot at
Collect, and a deleted `state.json` loses only counters and cursors.

## 7. The foreman pass

`charlie scheme` runs one foreman pass. It is deterministic and cheap. The fleet's
supervise loop starts it as a child process every `foreman.pass_interval_minutes`
(default 5), by argv, without waiting on it and without importing the foreman package.
The loop skips the start while the previous child is still alive. This needs no second
scheduled task and no operator step, the pass always runs the deployed version of the
code, and a slow model call still cannot delay a fleet pass because the loop never
waits. A fleet-wide pass lock means at most one pass, and so at most one Decide, runs at
a time. A Decide is not counted against the worker or reviewer caps; it draws on the
same provider quota, bounded by `max_turns_per_day` and judged by the section 16
verdict.

For each foreman in `active` or `degraded` state, a Turn runs when the pass holds the
foreman's file lock and any of these is true:

1. **Lifecycle fingerprint changed.** The fingerprint is the sorted list of
   (issue number, lifecycle labels read through `LabelConfig`, present or absent) for
   every issue the foreman owns, taken from each repo's status snapshot. An owned issue
   that has left the snapshot counts as changed; the briefing resolves why from
   `events.db`. No list of event kinds is involved, so a new event kind needs no change
   here.
2. **Inbox has an unread message, an item has a new answer, or a signal is unread.**
3. **The daily `report_at` time has passed** since the last scheduled Turn. A host that
   was off for three days gets one catch-up Turn, not three.
4. **`max_quiet_hours` has elapsed** since the last Turn.

Turns for one foreman never overlap: the file lock is held for the whole Turn, and a
pass that cannot take it moves on to the next foreman.

A Turn is then deferred, with a `foreman_wake_deferred` event naming the reason, when
`min_turn_interval_minutes` has not elapsed, `max_turns_per_day` is reached, or
`fleet-pause.json` exists. A paused fleet gets no Turns: an issue filed into a paused
fleet would sit, and the operator's standing rule is to handle work directly while
paused.

Ownership is a label, `foreman:<name>`, with the prefix read from `LabelConfig`. The
label is the truth; the list in `state.json` is rebuilt from it. Two kinds of owned
issue follow from that:

- **Authored**: the foreman filed it. The body carries the foreman's hidden marker
  (section 8.3) and `state.json` holds the hash of the body the foreman last wrote.
- **Tracked**: the operator added the label by hand to an issue someone else wrote. The
  foreman sees it in every briefing, counts it against its bounds and may comment on
  it, and can never amend or withdraw it.

Pausing a foreman, or a foreman reaching `faulted` or `done`, stops Turns and nothing
else. Issues already armed are ordinary fleet issues and keep flowing; pulling one back
is the operator's act.

## 8. The Turn

### 8.1 Collect (deterministic)

Builds `briefing.md` with fixed sections, each with a size cap:

- charter and `instructions.md`, verbatim;
- every memory file;
- the last few Turn summaries from the journal;
- owned issues (every open issue carrying the foreman's label), each marked authored
  or tracked: number, title, lifecycle state, linked PR, verdict, escalation reason
  codes, stage time so far;
- for each owned issue that escalated or got a request-changes verdict, the reviewer's
  recorded findings from the local review decision record, capped at 4 KB per issue.
  Reason codes alone cannot tell a foreman how to write a better successor issue;
- the event delta since the cursor for owned issues and PRs, reduced deterministically
  (transitions and reason codes, counts by kind), never a raw dump;
- fleet capacity facts that explain delay: capped demand, quota-ledger restrictions,
  pause history;
- open items with their age, new answers, unread inbox messages;
- unread signals, in a fenced block labelled as untrusted data. A briefing that
  contains one marks the Turn as signal-marked (section 12);
- actions the previous Turn had refused, with the refusal reason;
- the current bounds and how much of each is used.

This needs one seam change: `query_events(after_id=...)`, so the cursor is an exact
event id per `events.db`, not a timestamp that can drop or repeat a row.

Collect also moves the foreman's `tree/` worktree to the default branch tip, so Decide
reads what is on main and never the branch or uncommitted edits of whichever checkout
the repo is registered at.

Lifecycle state in the briefing always comes from the snapshot and events, never from
memory. The existing invariant holds unchanged: state lives in labels, never in chat
memory.

### 8.2 Decide (model)

One blocking headless launch through the foreman role chain, with `tool_scope` pinning
the read-only tool list and MCP isolation, `env_sanitize` on the environment, and the
working directory set to the foreman's `tree/` worktree so the model can read the code
on main before it writes an issue about it.

The only output that matters is a schema-validated document:

```json
{
  "turn_summary": "Two of five slices merged. #3101 escalated on a flaky fixture ...",
  "actions": [
    {"type": "file_issue", "key": "rest-labels", "repo": "<owner>/charlie-work",
     "title": "...", "body": "...", "blocked_by": ["key:rest-core", "#3090"]},
    {"type": "raise_item", "key": "flaky-3101", "kind": "decision",
     "question": "...", "options": ["...", "..."], "recommended": 0,
     "command": "charlie unescalate 3101"}
  ]
}
```

Decide is reached through the host launch port, so tests drive Turns with a scripted
fake and never start a model.

### 8.3 Apply (deterministic)

`actions.json` is written to disk before Apply starts. A crashed Turn resumes at Apply
from that file and never asks the model again, so a resume cannot produce a different
plan than the one already half-applied. The cursor and fingerprint advance only after
Apply finishes.

Each action passes, in order: schema → authority (table below) → bounds (section 10) →
type-specific checks → effect → `foreman_action_applied` or `foreman_action_refused`
event.

| Action | Allowed when | Enforcement point |
|---|---|---|
| `file_issue` | repo is in the charter; open-issue and per-Turn bounds have room | issue-contract lint, `outbound_body_guard`, private-slug refusal for a public repo, new create-issue request on the guarded transport |
| `amend_issue` | issue is authored by this foreman, is in no active or terminal state, and its live body still hashes to what the foreman last wrote | same checks as `file_issue` |
| `withdraw_issue` | issue is authored by this foreman and is in no active or terminal state; labels re-read live immediately before | Ready label removed first, then the issue is closed with a comment |
| `comment_issue` | issue carries this foreman's label | `outbound_body_guard` |
| `raise_item` | always | item store, then `notify` |
| `write_memory`, `delete_memory` | path stays inside `memory/`; size and count bounds | `safe_path` containment on the resolved path, atomic write |
| `propose_done` | `standing` is false | runs `done_when.check`; without one, becomes a decision item |

Three details make invalid output unrepresentable instead of merely rejected:

- **Blockers are data, not prose.** The model supplies `blocked_by` as keys or numbers.
  Apply orders the Turn's issues blockers-first, resolves keys to real numbers, and
  renders the `Blocked by #N` lines itself, one per line. The model's body text is then
  linted with zero intended blockers, so any stray `#N` or dependency phrase is refused.
- **Arming is a separate, final step.** The issue is created unlabelled except for
  ownership; the Ready label is added only after creation succeeded and only when the
  charter says `arming: auto`, the foreman is not `degraded`, and the Turn is not
  signal-marked. Otherwise the issue stays unarmed and a decision item asks the
  operator to arm it.
- **The operator's text is never overwritten.** An amend is refused when the live body
  no longer matches the hash the foreman recorded, which is what an operator edit looks
  like. The refusal shows up in the next briefing.
- **Creation is idempotent.** Each issue body ends with a hidden marker carrying the
  foreman name, Turn id and action key. Before creating, Apply searches the foreman's
  own label for that marker, so a crash between create and record cannot file twice.

`done_when.check` is operator-authored argv, never model output. Apply runs it with a
timeout in a fresh detached worktree at the default branch tip, created and removed by
the existing worktree helper. A pass moves the foreman to `done` and raises a final
report; a failure goes into the next briefing with the command's tail.

What a foreman can never do, because no action exists for it: edit any other label,
record a verdict, merge, unescalate, apply an escape hatch, touch an issue it does not
own, change config or its own charter, push code. When one of those is the right move,
it raises a decision item whose `command` field holds the exact `charlie ...` command,
the same pattern the dashboard uses in ADR-0008.

### 8.4 Memory

Memory is what the foreman chose to write down: operator preferences and corrections,
project facts, the current decomposition sketch. Bounds: 32 files, 8 KB each, 24 KB
loaded per briefing (oldest-modified files dropped first, with a line saying so). The
operator can read, edit and delete any of it. Memory is advisory text and is never read
by code as state.

## 9. Operator interface

All of it is CLI first, so an interactive session, a phone session over Remote Control
and a script are the same client:

```
charlie gang new <name> --from examples/foremen/migration   # scaffold a charter (starts paused)
charlie gang list | show <name>                             # state, activity, bounds used, open items
charlie gang say <name> "prefer one PR per module"          # instruction → inbox
charlie gang signal <name> --source sentry --file evt.json  # data → signals
charlie gang interventions [<name>]                         # open items with their commands
charlie gang intervene <name> <item> --choose 0 [--note ...]
charlie gang snooze <name> <item> --until <local time> | dismiss <name> <item>
charlie gang pepe-silvia <name>                             # print the latest briefing
charlie gang dayman [<name>]                                # print the latest daily report
charlie gang pause|resume <name>
charlie scheme [--dry-run [--no-decide]] [--only <name>]    # one foreman pass
```

**Themed surface (decided 2026-11-25).** As with `work`, `bash-rats` and `mop-up`, the
fun names are a thin alias over stable concepts. Code, config keys, event kinds, state
values and `CONTEXT.md` keep the semantic terms from section 5.

| Concept | In code and the glossary | On the CLI and in display text |
|---|---|---|
| Foreman | Foreman | `charlie gang`; a charter is named like an episode, `the-gang-migrates-the-transport` |
| Foreman pass | pass | `charlie scheme` |
| Briefing | Briefing | `pepe-silvia` |
| Item | Item | `interventions`, answered with `intervene` |
| Daily report | report | `dayman` |
| Degraded | `degraded` | shown as `CRICKET` |
| Breaker trip | breaker | shown as `WILD CARD` |
| Refused action | refused | shown as `jabroni` |

`say`, `signal`, `snooze`, `dismiss`, `pause` and `resume` stay literal, the way
`intake` and `doctor` do.

- **`say` versus `signal`.** `say` is the operator speaking and carries instruction
  authority. `signal` is the one seam for every external event source (a CI job, an
  error tracker, a webhook relay, a cron): it wakes the foreman and shows up in the
  briefing as data. Any integration is then a one-line caller of `signal`, and none of
  them can instruct the foreman.
- **`list` shows a derived activity** beside the foreman state: needs you (an open
  decision or permission item), working (owned issues in an active state), idle, done.
- **`pause` is read before each Turn and before each action in Apply.** It stops what
  has not happened yet; it does not undo actions already applied.
- **`--dry-run` runs Collect and Decide and prints what Apply would do.** It launches
  the model, so it costs tokens. `--no-decide` stops after the briefing and costs none.

- **Outbound.** A new item goes to the configured `notify` sink. `notify` gains one
  generic message shape beside the worker-health digest; pointing the existing webhook
  sink at a Slack incoming webhook is then configuration, not code. Under the notify
  config in use when this was written, items do not push to a phone: the operator sees
  them on the dashboard and through `charlie gang interventions`. That is accepted for
  v1.
- **Dashboard.** The "needs me" zone lists open items read from `items.json`, each with
  its command. It stays read-only.
- **Review and merge requests are not duplicated.** `review-queue` and `operator-queue`
  already hold them; the daily report links to them.

## 10. Structural bounds, breaker and cost

Per the standing operator rule there is no hard token or spend cap. Loops are bounded by
structure, usage goes to the ledger, and anomalies raise an alarm.

- **Bounds.** From the charter: open issues, issues per Turn, minimum interval, Turns
  per day. From the fleet config: model turns per launch. A refused action reports which
  bound it hit.
- **Escalated issues count as open.** An issue in Human needed or Operator queue is
  still open and still counts against `max_open_issues`. A foreman with five
  escalations files nothing more until the operator acts on them. That backpressure is
  why no probation period exists for a new foreman, even though the breaker below
  cannot evaluate until ten issues have finished.
- **Escalation breaker.** The escalation rate is the share of finished issues that
  ended in Human needed or Operator queue instead of Done, computed from `events.db`.
  If that rate over this foreman's last 10 finished issues exceeds 1.5× the same rate
  over the repo's last 50 finished issues that no foreman owns, the foreman moves to
  `degraded`: it keeps planning and reporting, but its new issues stay unarmed and an
  item asks the operator to confirm or correct course. One event on the transition, one
  on recovery. The baseline is a count, not a date window, so a long fleet pause cannot
  empty it and switch the breaker off. All three numbers are config.
- **Unattended limit.** Items never expire and never resolve themselves: a decision
  outside the foreman's authority has no safe default. Instead, when the oldest open
  decision item is older than `max_unanswered_days` (default 7), the foreman moves to
  `degraded` until it is answered, so it cannot keep building on a question nobody
  has read. The daily report lists every open item with its age.
- **Fault limit.** Three consecutive failed Turns (launch error, schema-invalid output,
  Apply exception) move the foreman to `faulted` and raise an item. No fourth blind
  attempt.
- **Usage.** Every Turn records tokens and duration on `foreman_turn_completed`. Alarms,
  defined once in the shared heartbeat alarm module so the dashboard and notifications
  agree: Turn usage above 3× the trailing median, a Turns-per-day bound reached, breaker
  tripped, foreman faulted, foreman pass not seen for twice its interval.

An earlier token audit of this fleet is the cautionary number: a role-split
orchestrating session cost about twice the worker plus reviewer per issue. The
idle-costs-nothing wake predicate and the single launch per Turn are the structural
answer; section 16 sets the threshold that
says whether it worked.

## 11. Inbound chat (optional, last)

A small Slack Socket Mode listener (outbound connection only, no public endpoint)
appends messages from one allow-listed Slack user id to the foreman's inbox and posts
the next Turn's summary in the same thread. Everything else it receives is dropped
unread. It needs the operator to create a Slack app and store two tokens, which is why
it is not in the core. Nothing else in this design depends on it.

## 12. Trust boundary

charlie-work is a public repo, so issue and PR text is attacker-influenceable.

- **Only the charter, `instructions.md`, `_shared/` and the inbox are instructions.**
  Everything else in a briefing (signals, issue and PR text, event payloads) is data,
  fenced and labelled as such. Inbox writes come only from `charlie gang say`, which
  the operator runs.
- The model in Decide has no write tool, no shell, no network and no `gh`. A successful
  injection can at worst produce a bad action list.
- **A signal-marked Turn files unarmed.** `charlie gang signal` is the one path where
  text from outside reaches the model. Any issue filed in a Turn whose briefing held an
  unread signal is left unarmed with an arming item, whatever the charter says. Outside
  text can therefore suggest work and can never start it.
- **Review findings are not signals.** They come from the fleet's own reviewer about
  the fleet's own worker PRs, and treating them as signals would put an operator gate
  on every recovery Turn. The accepted residual risk is that reviewer text is shaped by
  worker output.
- Apply validates every action against the charter. Authority cannot widen from model
  output; only a human edit of `charter.yaml` widens it.
- The briefing carries bodies only for issues the foreman authored, and deterministic
  fields for tracked issues and for PRs. Issue and PR comments are never included.
- Every outbound title and body passes `outbound_body_guard`. A foreman also refuses
  any text that names a slug from the list the repo's private-slug gate maintains.
- Inbound chat accepts exactly one user id.

## 13. Invariants: preserved and amended

Preserved without change: the orchestrator has no LLM-driven control flow (the foreman
is a client of the hub, like the operator); labels are the lifecycle source of truth;
atomic JSON writes; errors from external processes as values; frozen config dataclasses;
label strings from `LabelConfig`; the dashboard is read-only; one issue, one worker, one
PR.

Enforced by a new import-graph test: no module outside `charlie_work.foreman` imports
it. The hub can be reasoned about, tested and deployed as if the foreman did not exist.
The supervise loop knows one thing about the foreman: the argv that starts a pass.

**ADR-0010, "A foreman is a chartered client of the hub, and may arm its own issues",**
records the two real changes: Ready can now be set by a foreman under a charter, and
the GitHub client gains create-issue and edit-issue-body requests used only by the
foreman's Apply step. Considered and rejected there: a planning step inside the repo
pass, a foreman with direct `gh` access, a long-lived session per foreman, and a
separate repository.

## 14. Failure handling

| Failure | Behaviour |
|---|---|
| Model launch fails or output fails the schema | Turn recorded as failed, cursor unchanged, fault counter +1 |
| Provider quota exhausted | quota ledger restricts the pair; role chain falls back; if nothing is left the Turn is deferred, not failed |
| Crash during Apply | next pass resumes from `actions.json`; markers prevent duplicates |
| One action refused | the rest still apply; the refusal is in the next briefing |
| Status snapshot missing or stale | Turn deferred with a reason; never falls back to live GitHub reads for the fingerprint |
| Fleet paused | Turn deferred; inbox messages wait |
| Charter invalid | foreman skipped with a `foreman_charter_rejected` event; other foremen unaffected |

## 15. Testing and verification

- **Apply** is tested against `LocalFileGitHub`, a real backend, with a property test
  that replaying any `actions.json` twice yields the same issue set (L4).
- **Collect** has golden briefings built from recorded `events.db` fixtures.
- **Decide** is exercised through the host launch port's fake; one opt-in live test
  proves the real launch returns schema-valid output.
- **Wiring (L3):** an end-to-end test on a local-issues repo runs pass → Turn → issue
  filed and armed → a second pass with no change runs no Turn. The second half is the
  positive control for "idle costs nothing".
- **Authority:** one test per forbidden effect asserts that no action type can express
  it, plus refusal tests for each bound.
- **Import direction:** the graph test in section 13.

## 16. Rollout and go/no-go signals

The first charter is a bounded one on this repo: every command handler in `cli.py`
lives in its own command module and `cli.py` is under 1,000 lines. The extraction
pattern already exists, the blast radius is the CLI layer, and a line count is a
`done_when` a command can check.

0. **Prototype before the build.** One hand-assembled briefing for that outcome, one
   headless Decide call against the action schema, the output linted and read by the
   operator. Stop signal, set up front: fewer than two of three proposed issues pass
   the lint and would be armed unedited. The build is armed only after a pass.
1. **Dry runs.** The first foreman stays `paused` while `charlie scheme --dry-run
   --only <name>` is run by hand for three Turns. Proceed to `charlie gang resume` when
   every `file_issue` action passed the lint and no action was refused on authority.
2. **Live, `arming: auto`.** The bounds and the escalation breaker are the canary.
3. **Verdict after 20 finished authored issues or 14 days, whichever is first.** The
   foreman pass computes it and raises it as a report item; nobody has to remember to
   run the numbers. Two thresholds, set now: the share of the foreman's authored issues
   that reached Done is at least 0.8× the same share over the repo's last 50 finished
   issues that no foreman owns, and foreman Turn tokens are at most 25% of the worker
   plus reviewer tokens spent on those issues. Missing either forces that foreman to
   `propose` arming and raises a "retire or retune" decision item with the numbers.

## 17. Work breakdown

| # | Piece | Issue | Blocked by |
|---|---|---|---|
| 1 | ADR-0010 and the `CONTEXT.md` terms | 3113 (landed with this document) | |
| 2 | `query_events(after_id=)` cursor | 3114 | |
| 3 | Issue-contract lint moved in-tree with a CLI | 3115 | |
| 4 | Create-issue and edit-issue-body requests on both GitHub backends | 3116 | |
| 5 | Foreman store: charter, state, layout, worktree, shared partials, `new`, `list`, `show` | 3117 | |
| 6 | Collect: briefing builder with golden tests, review findings, journal | 3118 | 2, 5 |
| 7 | Action schema and Apply engine | 3119 | 3, 4, 5 |
| 8 | Decide launcher: foreman role, tool scope, host port fake, usage capture | 3120 | 5 |
| 9 | Foreman pass: wake predicate, locks, pause, Turn resume, start from the supervise loop | 3122 | 6, 7, 8 |
| 10 | Items, inbox and signals CLI, generic `notify` message, dashboard read | 3121 | 5 |
| 11 | Breaker, unattended and fault limits, verdict, alarms, `doctor` checks | 3123 | 9, 10 |
| 12 | `done_when` check, the migration example charter, operator docs | 3124 | 9 |

Pieces 2 to 5 have no blocker and can be dispatched together. The Slack listener of
section 11 is not filed.

## 18. Alternatives considered

- **A planning step inside the repo pass.** Rejected: it puts LLM control flow in the
  hub, makes a pass wait on a model, and couples fleet liveness to provider liveness.
- **One long-lived interactive session per foreman**, woken by scheduled prompts.
  Nothing to build, but context grows without bound, a dead session loses the plan, and
  the session holds the operator's full tool surface. This is the status quo that
  motivated the design.
- **Decide with write access (`gh`, shell).** Simpler prompt, no Apply engine. Rejected:
  authority would live in a prompt, a public-repo injection would have real reach, and
  none of the existing guards would sit on the path.
- **A separate repository.** Cleaner boundary on paper, but the foreman needs the body
  parsers, transport, guards, role chain and event log at exactly the deployed commit.
  A subpackage with an enforced import direction gives the boundary without version
  skew.
- **A second scheduled task for the foreman pass.** It keeps the hub ignorant of the
  foreman entirely. Rejected: it is one more thing to register by hand and keep alive,
  it can run a different version of the code than the hub during a self-deploy, and it
  keeps running when the fleet is paused. One argv in the supervise loop is the smaller
  cost.
- **Auto-adopting issues by a label query**, so a standing foreman could triage issues
  another process files. Deferred: tracked issues are comment-only by design, and no
  standing foreman ships until a bounded one has passed the section 16 verdict.

## 19. Prior art: what was taken and what was declined

Read from the vendors' own documentation on 2026-11-25 (Devin Bots at docs.devin.ai,
Grok Bot at docs.x.ai). Neither set of docs states its overlap, missed-schedule or
resume-after-crash behaviour; sections 7 and 8.3 state ours.

| Idea | Source | Here |
|---|---|---|
| One chat owns an outcome; one child session per piece of work | Devin | Taken: Foreman, and the existing one issue, one worker rule |
| Memory as files the manager writes | Devin | Taken (section 8.4), with bounds and operator edits |
| A dated work log separate from memory | Grok | Taken: `journal.jsonl`, written by code, not by the model |
| "Needs you" list with a marked recommendation | Devin | Taken: items, each with the command that enacts it |
| Templates are only a prompt | Devin | Taken: `instructions.md` and example charters |
| One instruction library shared by every bot | Grok | Taken: `_shared/` partials |
| Event triggers from many tools | Both | Taken as one seam: `charlie gang signal` |
| Only the owner's messages are instructions | Grok | Taken and made structural: `say` versus `signal` |
| Pause routines when the owner has been away | Grok | Taken as the unattended limit |
| Draft first, then automate | Grok | Taken: dry runs, then `arming: auto` behind the breaker |
| Per-routine run history, newest 20 | Grok | Taken: newest 50 Turn directories; events are permanent |
| Background approval cards expire after 30 minutes | Grok | Declined: an expired question is a silent default |
| "Always allow" on an approval | Grok | Declined: authority widens only by a human charter edit |
| Bots message each other, group chats | Grok | Declined: coordination goes through issues |
| The manager decides for itself what to remember, no view | Grok | Half taken: it decides, but every file is readable and editable |
| Sharing, access levels, canvas, cloud computer per bot | Both | Declined: single operator, local host |

## 20. Decisions (all made 2026-11-25)

1. **Name.** "Foreman" in code and the glossary, the Gang scheme on the CLI (section 9).
2. **Arming default.** `auto`, guarded by the bounds and the breaker. `propose` remains
   available per charter and makes every issue an operator gate.
3. **Model for the foreman role.** Sonnet-tier primary, with the charter able to raise
   it. Decomposition quality is the product here, so this is the knob most likely to
   move after the section 16 verdict.
4. **Home.** In-tree subpackage (section 18).
5. **Inbound Slack.** Out of the core, and not filed as work. Build it only if
   `charlie gang say` from a phone session proves too clumsy.

Settled in the design review that followed, each folded into the section named:

6. The first charter is the `cli.py` extraction, and a prototype Decide call on it comes
   before any of the build is armed (section 16).
7. The fleet builds the layer; the pause is temporary.
8. One repo per charter in v1 (section 6).
9. The ownership label is the truth and `state.json` is a cache; authored issues can be
   amended and withdrawn, tracked ones only commented on, and an operator edit blocks
   an amend (sections 7 and 8.3).
10. Escalated issues count against the open-issue bound, and there is no probation
    period (section 10).
11. The briefing carries review findings for failed issues, and they do not mark a Turn
    the way a signal does (sections 8.1 and 12).
12. Decide reads a worktree of its own at the default branch tip (sections 6 and 8.1).
13. The supervise loop starts the pass; no second scheduled task (section 7).
14. A signal-marked Turn files unarmed (section 12).
15. Breaker and verdict baselines are the last 50 finished non-foreman issues, not a
    date window (sections 10 and 16).
16. The verdict is computed by the pass, and a failing one forces `propose` (section 16).
17. One Decide at a time fleet-wide, outside the worker and reviewer caps (section 7).
18. Pausing a foreman never touches its issues (section 7).
19. Items reach the operator through the dashboard and `charlie gang interventions`
    under today's notify config; a push is a later config change (section 9).
20. Only the migration example ships; no standing example and no auto-adopt rule
    (section 18).
21. A new foreman is created `paused` (section 6).
22. This document and ADR-0010 live in the tree, because a worker carries no `gh`
    credential and cannot read an issue other than its own.
