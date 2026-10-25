# Main sync: keep registered repos' local mains on origin

Status: design approved 2026-10-25. Not implemented.

## Problem

The fleet merges PRs on GitHub, but nothing moves the operator's local checkouts afterwards.
A stale local main hurts in four ways:

- Code that runs from a checkout keeps running old code.
- A long interactive session reads and edits files that merges have since replaced.
- A checkout nobody has opened in a while falls far behind, and the next pull is slow or blocked.
- Plain hygiene: mains should follow origin without anyone thinking about it.

Two mechanisms exist today, and neither covers this:

- **`supervise.self_deploy`** fast-forwards the checkout the supervisor runs from (`orchestrator_root()`, on the operator host `C:\Users\senki\srv\charlie-work-daemon`) once per pass, with its own restart semantics.
- **The operator's SessionStart `ff-sync` hook** fast-forwards a checkout once, when a session opens in it.

Workers never read a local main. Each worktree branches from a freshly fetched `origin/<default>`, so fleet correctness does not depend on this feature.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Trigger | Reconcile once per full supervisor pass. Not on each merge, not on release tags | Merges also land through the GitHub UI, merge queues and direct pushes. Comparing each checkout to origin catches all of them, at most one pass late. Release tags would leave mains stale between releases |
| Home | The fleet supervisor | Reuses the registry, git retry, events.db and alarm patterns. Accepted cost: nothing syncs while the fleet is paused (the SessionStart hook still covers newly opened sessions) |
| Dependencies | Fast-forward only. The fleet never runs `uv sync` for a repo unless that repo declares a `post_sync` command | See "Why the fleet does not run uv sync" |
| Live sessions | Fast-forward underneath them, then tell them once per prompt via a marker file | Git refuses a fast-forward over overlapping local edits, and Claude Code's Edit/Write refuse a file that changed since it was read. What a session cannot see for itself is that its loaded instructions went stale |

### Why the fleet does not run uv sync

The `uv sync` incident history sets the constraints:

| Failure class | Records | Constraint |
|---|---|---|
| Windows file locks on a venv that is in use | #437, #854 | Never sync a venv something is running from |
| HEAD moved but deps did not, so the next restart crashed | #2312 (fleet down 50 min), fixed by #2325 | Code and deps move together, or neither moves |
| Deferral starvation, restart churn | #1855, #851 | Every deferral has a time bound |
| Shared or junctioned venv poisoning | #274, #649, #112, #1516, #969 | Only a real `.venv` at the checkout root. Never through a junction |
| Sync rewrote `uv.lock` | #889 | Always `--locked` |
| An env that never re-syncs | #2808 (open), #2348 | A fast-forward alone can leave long-running code stale |
| Host-wide cache lock | operator memory | Never `uv cache clean` and never `--reinstall` |

`self_deploy` handles all of these, but only for the one checkout whose runtime it owns. Running `uv sync` eagerly in other repos would reopen each class wherever a daemon, a runner or an editable tool install uses that repo's venv. `uv run` already syncs from the lock on every call. So a plain fast-forward is enough for code started with `uv run`, and long-running code is what `post_sync` exists for.

## Design

### Unit: `charlie_work/main_sync.py`

It has one job: bring one registered repo's primary checkout up to `origin/<default>` when that is safe.

- `classify(state: CheckoutState) -> SyncDecision` is a pure function over a frozen snapshot of the checkout's git state.
- A thin effects layer does the fetch, the fast-forward or ref update, `post_sync`, any rollback, the marker write and the events.
- External processes return their results as values. Nothing in the module raises to its caller, matching the project invariant.
- Nothing is imported from the operator's `~/.claude/scripts`, because this repo is public. The policy is reimplemented here, and the SessionStart hook stays as it is.

### Trigger and scope

- It runs once per full supervisor pass (the `full_pass_interval_seconds` fallback, or a local change), never on the 20-second poll.
- It iterates the same registry the pass uses (`fleet.json` through `_select_repos`), and takes each entry's `repo_root` and `state_dir` from there. No other list of repos exists.
- **It excludes the supervisor's own checkout, compared by resolved path** to `orchestrator_root()`, not by repo key, because `self_deploy` owns that checkout. On the operator host the `Senkichi/charlie-work` registry entry is the operator's dev checkout, a different path, so it is synced.
- It skips repos with no `origin` remote, because the local lane owns their main.
- Kill switches: the global `main_sync.enabled` and the per-repo `main_sync.enabled`, both defaulting to true.
- A wall-clock budget covers the whole step. When it runs out, the remaining repos are skipped with `budget_exhausted` and go first on the next pass.

### Classification

After `git fetch origin <default>` (through `run_git_with_retry`, with a short timeout), exactly one row applies:

| State | Action | Event |
|---|---|---|
| Up to date | none | none |
| On the default branch, behind, worktree clean or dirty only in paths the incoming range does not touch | `git merge --ff-only origin/<default>` | `main_sync_advanced` |
| On another branch, and local `<default>` behind and not ahead | `git fetch origin <default>:<default>` (moves the ref, working tree untouched) | `main_sync_advanced` with `ref_only: true` |
| No merge-base between local `<default>` and origin | skip | `main_sync_skipped` reason `unrelated_history`, plus an alarm on first sight |
| Local `<default>` ahead or diverged, with a merge-base | skip | `main_sync_skipped` reason `diverged` |
| Dirty tracked files overlap the incoming paths, or untracked files would be overwritten | skip, listing the paths | `main_sync_skipped` reason `blocked` |
| Merge, rebase, cherry-pick or revert in progress, stale `index.lock`, or detached HEAD | skip | `main_sync_skipped` with that reason |
| Not a primary checkout, no `origin`, or kill switch off | skip | none |

- **`unrelated_history` alarms at once** because waiting never resolves it: origin was re-rooted, for example by a history rewrite. As of 2026-10-25 the operator's dev checkout is in this state: local `main` at `61fdfe509` (#2785), and `origin/main` re-rooted at `656ed3cf3` (#2812) on 2026-10-23.
- **Skip events are deduplicated** per `(repo, local sha, origin sha, reason)`. A checkout that stays blocked or diverged logs once, not once per pass.
- **There is no blocker repair here.** Lossless pull-blocker repair (`_repair_lossless_pull_blockers`) stays specific to `self_deploy`, which owns its checkout. main_sync owns no checkout, so it reports blockers and touches nothing.

### post_sync contract

This is opt-in, set per repo in that repo's `orchestrator.config.yaml`:

```yaml
main_sync:
  enabled: true
  post_sync:
    command: ["uv", "sync", "--locked"]          # argv list, run with cwd = checkout root
    when_paths: ["uv.lock", "pyproject.toml"]   # optional; default: run on every advance
    timeout_seconds: 300
```

- It is parsed into a frozen dataclass, like all config. `command` is an argv list and never goes through a shell.
- It runs only after a successful working-tree fast-forward (never after a ref-only update, where the working tree did not change), and only when the incoming range touches `when_paths`.
- **Exit 0:** the advance is final.
- **Partial fast-forward.** When the incoming range touches `when_paths`, main_sync first fast-forwards to the last commit *before* the first commit that touches them. That step needs no `post_sync`, so commits that don't change deps keep landing while a deferred one waits. Only the remainder of the range goes through `post_sync`.
- **Exit 75 (`EX_TEMPFAIL`, "not now", e.g. the repo's venv is in use):** roll back the remainder with `git reset --keep <from_sha>`. A defer does not count as a failure. Retries back off, doubling from one pass to at most 60 minutes, so a checkout under a live session does not flip back and forth every pass. After 4h of continuous deferral, raise an alarm once, and never force the command.
- **Any other exit, or a timeout:** roll back the same way. Event `main_sync_failed` with the stderr tail. Three consecutive failures raise an alarm once, and a success resets the count.
- **Rollback refused** (`--keep` refuses to discard local changes): event `main_sync_rollback_refused` and an immediate alarm.
- **Why roll back:** #2312. For a repo that declares a runtime, the committed tree and that runtime must never disagree. Otherwise the next unrelated restart of that runtime starts on a tree its environment cannot run.

### Telling live sessions

**Marker, written by charlie-work.** Path: `<git-common-dir>/charlie-main-sync.json`. It is written through `atomic_write.write_json_atomic` only once an advance is final: after `post_sync` exits 0, or right after the fast-forward when no `post_sync` applies. A rolled-back advance never writes it. It is overwritten on each advance and never deleted.

```json
{"schema": 1, "from": "<sha>", "to": "<sha>", "commits": 4,
 "instruction_paths": ["CLAUDE.md", ".claude/settings.json"], "ts": "<ISO-8601 UTC>"}
```

- `instruction_paths` lists the changed paths matching `CLAUDE.md` or `AGENTS.md` at any depth, or anything under `.claude/`.
- The marker schema is a cross-repo contract with the operator's hook. It is versioned by `schema` and documented in `docs/RUNBOOK.md`.

**Hook, in the operator's claude-config repo.** Stdlib only, registered as a `UserPromptSubmit` hook:

1. Find the git common dir using file reads only. Walk up from the cwd to `.git`. If `.git` is a file (a worktree), follow its `gitdir:` line, then that directory's `commondir`. No subprocess, no git, no network.
2. Inject only when both hold:
   - the marker's `ts` is later than the session's start, taken as the transcript file's creation time;
   - the marker's `to` differs from the last `to` this session was told about, stored per `session_id` under `~/.claude/state/`. Per-session records mean every session in the repo is told once, and no session clears the notice for the others.
3. Print one line:
   - `[main-sync] main advanced 4 commits (abc123→def456) during this session. Changed: CLAUDE.md, .claude/settings.json. Loaded instructions may be stale.`
   - With no instruction paths changed, end with `Files on disk are current.` instead.
4. Exit 0 on every path, and print nothing on any error. The hook never blocks a prompt.

### Observability and alarms

- **Events** go to the repo's own `events.db`, under `state_dir` from the registry, through `log_event`: `main_sync_advanced`, `main_sync_skipped`, `main_sync_deferred`, `main_sync_failed`, `main_sync_rolled_back`, `main_sync_rollback_refused` and `main_sync_alarm`. Each carries `repo`, `from_sha`, `to_sha` and, where relevant, `reason`, `paths` and `stderr_tail`.
- **Alarm counters**, one per repo and stored in the fleet dir through atomic writes, follow the `_record_self_deploy_failure_streak` pattern:
  - 3 consecutive failures;
  - 4h of deferral;
  - unrelated history, on first sight;
  - a refused rollback, at once.

  Each alarm fires once and resets on the next success.
- **Dry run:** under `--dry-run` main_sync classifies and logs what it would do. It never fetches into local refs, fast-forwards, runs `post_sync` or writes the marker (#613).

## Testing

- **Unit:** a table-driven `classify()` test over fake `CheckoutState` values, one case per row of the classification table.
- **Integration:** real temporary repos (a bare origin plus a clone), covering:
  - working-tree fast-forward on the default branch;
  - ref-only update while on a feature branch;
  - skip on overlapping dirty files, with non-overlapping dirty files still fast-forwarding;
  - skip for untracked blockers;
  - skip on divergence;
  - `unrelated_history` raising its alarm;
  - `post_sync` exit 0, exit 75 (rollback, no failure counted, retry back-off) and exit 1 (rollback and failure);
  - a partial fast-forward stopping before the first commit that touches `when_paths`;
  - `when_paths` filtering;
  - rollback refused;
  - dry run touching nothing;
  - skip events deduplicated per sha pair;
  - marker written only for a final advance;
  - the supervisor's own checkout excluded by resolved path.
- **Hook** (claude-config): marker lookup from a primary checkout and from a worktree; per-session dedupe; a session started after the marker's `ts` stays silent; a missing or corrupt marker stays silent.

## Rollout

- Ships ON, as a pass step after `self_deploy`. The fleet is paused today (`fleet-pause.json`), so nothing changes until `fleet resume`.
- **Live check:** after the first full pass, expect one `main_sync_advanced` or `main_sync_skipped` row in the `events.db` of each non-supervisor registry repo (`claude-config`, `job-cannon`, `swole`, and the charlie-work dev checkout). The dev checkout should report `unrelated_history` until the operator resets it.

## Out of scope

- **#2808:** the editable `charlie` uv-tool env does not re-resolve its deps. It has its own fix.
- **Registry coverage:** `fleet.json` lists four repos, with several stale `last_seen` values. Repos that are not registered are not synced. That is a question for the registry, not for main_sync.
- **Webhook push, `git maintenance` prefetch, and a worktree-only workflow:** considered and rejected. Push needs an inbound channel that fails silently. Prefetch never updates the working tree. A worktree-only workflow changes how the operator works.
