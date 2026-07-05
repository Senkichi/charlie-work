# devin-orchestrator Extraction Dossier

Synthesized from 8 recon readers (job-cannon source, empericus source, tests, design-docs,
devin-ops, emp-practice, jc-ops-state, docs) plus 2 external-research passes (papers,
systems/practitioners). Single source of truth for the design-panel and builders.

**Scaffold status**: `C:/Users/senki/repos/devin-orchestrator` already exists — union merge
committed, 41 tests green. DONE: merge/label decoupling, config discovery, prompt-dir
overrides, worker_template selection. This dossier's fix list and open questions are aimed
at what remains: adapters (devin-shell, claude-code), doctor command, fleet-readiness
hardening, logging capture, docs.

---

## 1. System overview

The devin-orchestrator is a **deterministic, gh-CLI-driven pipeline** that turns GitHub
issues labeled `automated-ready` into merged pull requests, using AI coding agents as
disposable workers and a second AI pass as an adversarial reviewer. It is a ~1200-line
Python package (`automation/devin_orchestrator`, 11 modules) with zero framework
dependencies (no LangGraph, no agent SDK) — a plain CLI + JSON state file + gh subprocess
calls. It existed as **duplicated, untracked (git-excluded) code** in two sibling repos
(job-cannon, empericus) that drifted independently; the new standalone repo is the fix for
that root cause.

**End-to-end flow** (`loop` command, or its parts run individually):

1. **intake** — `gh issue list --label automated-ready` → snapshot each issue (`issue.json`)
   → render `worker-prompt.md` from a template.
2. **dispatch** — pick dispatchable issues (has `automated-ready`, no terminal/active label)
   → compute a deterministic branch name (`agent/issue-<N>-<slug>`) → hand off to an
   **adapter** (manual manifest, or a subprocess-launching command) → label `agent:in-progress`
   only for launches that report success.
3. **Worker executes** (out of process, out of Python entirely) — in practice: a git
   **worktree per issue**, checked out off `origin/main`, a **junctioned shared venv**, and a
   headless CLI worker (`devin --prompt-file … --print --permission-mode dangerous` or a
   Claude Code equivalent) that implements the issue, runs tests/lint itself, commits, pushes,
   and opens a PR via `gh pr create`. Results flow back **exclusively through GitHub** (PR
   existence, branch, labels) — not through any return channel from the worker process.
4. **review** — for every open PR linked to an issue: snapshot `pr.json`/`checks.json`/
   `diff.patch`, render an adversarial `review-prompt.md`, optionally run a **cross-family**
   (different-model-family, e.g. codex) adversarial pass, label `agent:pr-open` +
   `agent:reviewing`.
5. **record-review** — a human or the orchestrating AI session reads the packet and records a
   decision (`approved` / `request_changes` / `blocked`) to `review-decision.json`;
   `request_changes` generates a `rework-prompt.md` and flips `agent:needs-rework`.
6. **merge-ready** — gates merge on (a) an `approved` decision file and (b) all configured
   required CI checks passing; on merge, squash-merges via `gh pr merge`, optionally deletes
   the branch, labels the issue `agent:done` and strips active labels.
7. **loop** — chains intake → dispatch → review-every-open-linked-PR →
   merge-ready-if-approved, meant to be re-invoked repeatedly by an orchestrating AI session
   that treats GitHub labels + local JSON/markdown artifacts as the only durable state (never
   chat memory).

**Both worker runtimes** the system must support:

- **Devin CLI** (`devin --prompt-file <file> --print --permission-mode dangerous --model
  <model>`) — Cognition's CLI agent, launched headless/non-interactive, no session-creation
  API exists (confirmed empirically against CLI 2026.8.18); only shell-wrapping is possible.
  Devin-cloud extensibility (rules/skills/hooks in `.devin/`) was built and deployed in
  job-cannon to make worker behavior deterministic.
- **Claude Code** — the runtime actually used in empericus's live practice (despite "Devin"
  naming throughout config/docs): a human or the orchestrating Claude Code session opens a
  session in the per-issue worktree, which inherits the repo's tracked `.claude/settings.json`
  (permissions, hooks) automatically via ordinary worktree checkout. No orchestrator code
  spawns a `claude` process — this "runtime" is emergent from `git worktree add` +
  repo-tracked `.claude/` + human pasting `worker-prompt.md` into a session.

The extraction target is: **Devin worker support as the default/documented path, Claude Code
worker support as a first-class alternative**, unified behind one adapter contract.

---

## 2. Module-by-module inventory (union of both forks)

Both forks (job-cannon "jc", empericus "emp") derive from the same ancestor. 8 of 13 files
are byte-identical: `__init__.py`, `adapters.py`, `checks.py`, `paths.py`, `prompts.py`,
`state.py`, `prompts/orchestrator.md`, `prompts/rework.md`.

### `__main__.py` (131 lines each, divergent)
CLI entry point, argparse-based. Subcommands: `status`, `bootstrap-labels`, `intake`,
`dispatch [--limit]`, `review [--pr]`, `record-review [--pr --decision --summary/
--summary-file --comment]`, `merge-ready [--pr --merge/--no-merge]`, `loop [--limit]`.
Global flags `--config`, `--repo`, `--dry-run`, `--json` (position-independent via an argv
string-strip hack that also swallows a literal `--json` value belonging to another option —
**bug**, fix candidate: replace with proper argparse). Exit codes: 0 ok, 1 `CommandResult`
not-ok, 2 `GitHubError`/`OSError` (uncaught `JSONDecodeError` from corrupt state.json is a
gap).
- **jc-only**: `review --cross-family/--no-cross-family` (tri-state), `spec-review --file`.
- **emp-only**: `dispatch --issues '565,570,572'` — explicit, comma-separated,
  dependency-ordered wave dispatch overriding the newest-first heuristic; skips
  non-dispatchable numbers; NOT plumbed through `loop`.

### `config.py` (155 jc / 127 emp lines, divergent)
Frozen dataclasses: `DispatchConfig` (limit, `worker_model_tier`, `orchestrator_model_tier`
— **both decorative, never wired to `--model`**), `ReviewConfig` (`max_rework_cycles`,
`require_tests_or_rationale`, `require_issue_link` — **all three dead, never read**),
`AutoMergeConfig` (`enabled`, `required_checks` tuple, `delete_branch`,
`allow_auto_merge_when_pending` — **dead**), `LabelConfig` (ready/active/terminal sets),
`DevinConfig` (adapter, `dispatch_command`, `command_timeout_seconds`), `RuntimeConfig`
(state_dir). `load_config` tolerates a missing YAML (all-dataclass defaults) but raises a bare
`TypeError` on any unknown key (unhelpful). `DEFAULT_CONFIG_PATH` lives **inside the package**
— per-repo config edits currently mutate the installed package; extraction must move packaged
defaults out of the installable unit.
- **jc-only**: `CrossFamilyConfig` (enabled, model, command tuple, timeout) — defaults
  `enabled=False` so an absent YAML block is a no-op.
- **required_checks defaults differ per repo** (jc: `['Tests passed','Lint & Format',
  'Pre-commit']`; emp: `['Test','Lint, Types & Pre-commit']`) — this is inherently per-consumer
  deployment data, not a code default that should ship hardcoded.
- **emp bug**: `delete_branch: bool = True` in the dataclass contradicts the shipped YAML's
  `delete_branch: false` — the worktree-safety fix lives ONLY in YAML.

### `github.py` (194 jc / 201 emp lines, divergent)
Frozen-dataclass `gh` CLI wrapper. `run()`: `['gh', *args]`, `text=True, encoding='utf-8',
errors='replace'` (Windows-safe — **but this pattern is inconsistent elsewhere**, see bugs).
`allow_failure` path: non-zero exit + empty stdout → returns **stderr text as the value**
(ambiguous channel — `pr_diff` can write gh error text into `diff.patch` as if it were a real
diff). `dry_run` gates only commands NOT matching a hardcoded readonly-prefix allowlist
(`_is_mutating`) — any new readonly gh subcommand is dry-run-blocked by default until the list
is edited (violates no-hardcoded-lists principle). `linked_issue_number`: matches ANY `#N` in
PR title before checking body Closes/Fixes/Resolves — false-positive-prone; body-only
qualified-keyword matching correctly rejects unqualified cross-repo refs (dependabot
`#2454`-style). `merge_pr`:
- **jc bug**: `return str(self.run(args))` — `gh pr merge` prints its success line to
  **stderr**, so stdout is empty on success → `merged` reads **False** despite a real merge.
- **emp fix** (must carry into union): `output = str(self.run(args)); return output or
  f"merged #{number}"` with an explanatory comment — `run()` already raises on non-zero exit,
  so reaching this line means success; the empty-stdout fallback makes `merged` truthy.

### `checks.py` (46/47 lines, near-identical)
`summarize_checks`: exact-name lookup against the `required_checks` allowlist (a failing
non-required check never blocks); duplicate check names (matrix jobs) collapse to the **last**
dict entry (a failing leg can be masked by a later-listed passing leg with the same name);
pass on `state==SUCCESS`/`conclusion==SUCCESS`/bucket `'pass'`; pending on
`{PENDING,QUEUED,IN_PROGRESS,REQUESTED}`/bucket `'pending'`; **everything else (including
SKIPPED/NEUTRAL/CANCELLED) → failed**; required-but-absent → `missing` (correctly blocks).
`ready` = no pending AND no failed AND no missing.

### `adapters.py` (229 lines, byte-identical across forks)
`dispatch_sessions()` contract: always writes `session-manifest.json` first (even for zero
requests), then per-adapter results, always writes `session-results.json`. Adapters:
`manual` (every request `ok=True`, **no action taken** — this is the fabricated-success bug,
see §7), `command` (sequential subprocess per request, list-form = no-shell per-part
`.format`, string-form = **`shell=True`** whole-string `.format` — **shell injection via
issue_title interpolation**), any other adapter name → per-request failure. `subprocess.run`
has **no `encoding=`/`errors=`** kwargs (unlike `github.py`) — Windows cp1252-decodes worker
output; `TimeoutExpired.stdout` is `str()`-coerced without a bytes check, producing `"b'...'"`
garbage (a bytes-safe pattern exists in `cross_family.py` and was never backported here).
Empty command → clear error. This is the module most in need of a shared subprocess-runner
utility.

### `paths.py` (52/53 lines, byte-identical)
`find_repo_root` via `git rev-parse --show-toplevel` with a `.git`-scan fallback.
`runtime_paths(repo_root, state_dir)` derives all artifact paths under a configurable root
(default `.var/devin-orchestrator`).

### `prompts.py` (14/15 lines, byte-identical)
`render_prompt` = `string.Template.safe_substitute` over a file read fresh every call from a
`prompts/` dir shipped **inside the package**. `safe_substitute` never raises on a
missing/renamed placeholder (silent prompt corruption — e.g. empericus's `rework.md` still
references a dropped `/preflight` skill and nobody would ever be told). **No escaping** of
injected values — an issue body containing `` ``` `` breaks the prompt's own code fence
(prompt-injection surface from issue authors into workers). Command templates use a
**second, different** templating dialect (`str.format` with `{}` placeholders) — two
templating systems in one codebase, each fit-for-purpose (issue bodies full of `{}`/`$` would
break either alone) but confusing; document or unify deliberately.

### `state.py` (54/55 lines, byte-identical)
`load_state`/`save_state`: atomic temp-file + `os.replace` (correctly atomic on Windows, no
fsync, **no file locking** — concurrent invocations silently lose updates, last-writer-wins).
`save_state` **mutates its input dict in place** (`data['generated_at']=utc_now()`) and
`append_event` mutates the events list with an in-place 200-cap trim — both violate the user's
stated immutability convention. `load_state` returns `empty_state()` for missing/non-dict
JSON but **does not catch `JSONDecodeError`** for a truncated/corrupt file — uncaught
traceback, not graceful.

### `workflow.py` (589 jc / 475 emp lines, most-divergent file)
The orchestration core: `intake`, `dispatch`, `review`, `record_review`, `merge_ready`,
`loop`, plus jc-only `spec_review`/`_cross_family_for_pr`. **State-clobbering pattern present
in both forks**: `review()` wholesale-overwrites `state['prs'][n]` (erasing the `decision` key
`record_review` set), `intake()` wholesale-overwrites `state['issues'][n]` (erasing dispatch
status). State.json survives only because **on-disk decision files are the real source of
truth**, read fresh via `_review_decision()` — state.json is an event log/cache, not
authority; the extraction should make this explicit or fix the clobbering.
`loop()` has **no per-PR error isolation**: `gh.merge_pr` uses `allow_failure=False`, so one
merge conflict raises `GitHubError` and aborts the entire remaining batch with exit 2.
`_is_dispatchable` = has `ready` label AND no terminal label AND no active label — the single
enforcement point for the state machine (good pattern to keep).
- **jc-only** (must survive union): `spec_review()` (runs regardless of
  `cross_family.enabled`, no reuse — re-runs codex every call), `_cross_family_for_pr()`
  (skips drafts, reuses a non-empty `cross-family-review.md` from disk across loop passes so
  repeated cycles don't re-burn the external model — **but** the reuse check is
  existence+size>0 only, so a **failure stub is reused as success forever** — bug, see §7).
- **emp-only** (must survive union): `dispatch(only_issues=...)` — parses
  `only_issues.replace(" ", "")` (deletes ALL spaces, not just around commas — `"5 6"` becomes
  issue `56`, a **silent wrong-issue bug**), no dedup of repeated numbers, uncaught `ValueError`
  on non-numeric tokens, `--limit` entirely ignored when `--issues` is given, **not plumbed
  through `loop()`** so wave-ordered dispatch requires bypassing the documented main loop.

### `cross_family.py` (140 lines, jc-only, absent in empericus)
Never-raises hard contract (docstring-enforced): timeout, non-zero exit, and missing-binary
are all captured as `UNAVAILABLE` stub reports with partial output preserved; success writes a
"leads, not verdicts" caveat + model stdout. Injectable `Runner` callable for tests. This is
the adversarial cross-model pass — dropped entirely in empericus with no documented rationale
or replacement.

### `orchestrator.config.yaml` (61 jc / 51 emp lines)
Per-repo deployment data: `required_checks` (must exactly match GitHub check-run `name:`
fields — a rename silently makes checks read as `missing` and permanently blocks merge, a
documented empericus operational trap), `delete_branch` (jc: `true`; emp: `false` with a
comment explaining worker branches live in local worktrees so `--delete-branch` fails on
local-branch deletion and **aborts the post-merge label update**, breaking merge+label
atomicity), `devin.adapter: manual` in both (the `command` adapter is configured-but-unused in
production in both forks), jc-only `cross_family` block.

### `prompts/*.md` (6 templates)
`orchestrator.md`, `rework.md` byte-identical across forks. `worker.md` diverges sharply: jc
is Devin-skills-driven (`/create-branch /commit /test /preflight /push /create-pr /complete`,
PR title `Fix #N: <title>`); emp is raw-shell-driven (`git switch`, `uv run pytest` [now: `uv run --extra dev pytest -q --tb=short`], `git
push`, Conventional-Commits PR titles) with empericus-specific invariants inlined (privacy,
asyncio-single-process, migration-serialization warning). `review.md` differs only by the
jc-only `$cross_family_section` placeholder. jc-only: `cross_family_review.md` (7 attack axes),
`cross_family_spec_review.md` (6 attack axes, spec embedded in prompt).

### Tests
jc: 30 functions (16 shared + 14 cross-family-only). emp: 17 functions (16 shared + 1
`dispatch --issues`-only). See §9.

---

## 3. De facto state.json schema

```
{
  "version": 1,
  "generated_at": "<ISO-Z, refreshed on every save_state call>",
  "issues": {
    "<issue_number_as_string>": {
      "number": int,
      "title": str,
      "url": str,
      "labels": [str, ...],        // GitHub label snapshot, NOT a lifecycle status field
      "prompt_path": "<absolute Windows path to worker-prompt.md>",
      "updated_at": "<ISO-Z>"
      // NOTE: no "status" field — lifecycle is inferred from GitHub labels only.
      // dispatch() sets a transient status="dispatch_failed" on failure but intake()
      // clobbers the whole record on next run, erasing it.
    }
  },
  "prs": {
    "<pr_number_as_string>": {
      "number": int,
      "issue_number": int,
      "url": str,
      "status": "reviewing",       // ONLY value ever observed in production; never
                                    // advances to approved/merged/closed — labels/PR
                                    // state are the only real lifecycle signal
      "prompt_path": "<abs path to review-prompt.md>",
      "decision_path": "<abs path to review-decision.json>",
      "decision": "approved" | "request_changes",   // OPTIONAL — dropped by a later
                                                       // review_packet event in observed
                                                       // production data (pr-497 case)
      "cross_family_ok": bool,      // OPTIONAL, jc-only
      "cross_family_report": "<abs path>"  // OPTIONAL, jc-only
    }
  },
  "events": [
    {
      "at": "<ISO-Z>",
      "kind": "intake" | "dispatch" | "review_packet" | "record_review" | "spec_review",
      "payload": { /* kind-specific, SCHEMA DRIFTS OVER TIME, see below */ }
    }
    // capped at 200 entries via in-place del events[:-200]; append-only otherwise —
    // in production this was the ONLY reliably-surviving history (manifest/PR/issue
    // records all get overwritten)
  ]
}
```

**Event payload schema drift observed in production** (jc-ops-state recon):
`dispatch` payload gained `failed_issue_numbers` partway through the observed history (absent
in the earliest 3 events); `review_packet` gained `cross_family_ok`/`cross_family_reused`
starting 2026-07-01. Any consumer must tolerate multiple historical payload shapes for the
same `kind`.

**On-disk artifacts that are the REAL authority** (state.json is a lossy cache/mirror):
- `issues/issue-<N>/{issue.json, worker-prompt.md}` (+ jc: `worker-prompt.grounded.md`,
  per-subtask prompt variants, `REWORK-PREAMBLE-N.md`)
- `prs/pr-<N>/{pr.json, checks.json, diff.patch, review-prompt.md, review-decision.json,
  rework-prompt.md, cross-family-{prompt,review}.md (jc), review-comment.md}`
- `dispatches/{session-manifest.json, session-results.json}` — **overwritten per dispatch
  call**, retains only the last dispatch; dispatch history lives only in `state.json` events
- `cross-family/spec-<slug>-{prompt,review}.md` (jc-only, spec-review)
- `review-decision.json` schema: `{decision: approved|request_changes|blocked, issue_number,
  pr_number, reviewed_at, summary}` — `summary` can legally be an **empty string** (observed
  in production: approvals with zero rationale)
- `logs/` — worker stdout, **final message only** (see §5)

---

## 4. Devin CLI operational reality + manual adapter loop + worker setup/finish scripts

### CLI reality (Devin CLI 2026.8.18)
- **No programmatic session-creation, lifecycle, monitoring, or termination API exists.**
  Confirmed empirically: `devin session create` errors "unexpected argument 'session'"; no
  `--branch`/`--title`/`--background` flags. Only the interactive CLI and cloud sandbox
  sessions exist as surfaces.
- Relevant flags: `--prompt-file <FILE>`, `-p/--print [<PROMPT>]` (non-interactive, "transient
  session"), `-c/--continue`, `-r/--resume [<ID>]`, `--permission-mode
  <auto|accept-edits|smart|dangerous>` (default `auto`; **`dangerous` never overrides
  org-level permissions** — a documented ceiling on unattended automation), `--model <MODEL>`
  (default `adaptive`), `--export [<PATH>]` (exports conversation **after each turn** — an
  unused progress-monitoring channel), `--sandbox`, `--agent-config <FILE>`.
- Session storage: `%APPDATA%\devin\cli\sessions.db` (SQLite; Linux/Mac
  `~/.local/share/devin/cli/sessions.db`), schema documented (id, working_directory,
  backend_type, model, agent_mode, created_at, last_activity_at, …), IDs are hyphenated word
  pairs (`mammoth-heart`). Transcripts at `…\transcripts\<id>.json` (ATIF-v1.7, undocumented
  schema). Lock files at `…\session_locks\<id>.lock` containing the owning PID.
- Hooks: 8 events (SessionStart, SessionEnd, PreToolUse, PostToolUse, PermissionRequest,
  UserPromptSubmit, Stop, PostCompaction) via `.devin/hooks.v1.json`, **explicitly Claude-Code
  compatible** — also read from `.claude/settings.json`; rules load from `AGENTS.md` OR
  `CLAUDE.md`; `read_config_from: {claude: true}` imports Claude Code config. This is the
  concrete basis for one repo layout serving both worker types.
- `devin acp --agent-type review` exists (read-only + shell code-review agent) — rejected as
  an orchestration primitive (IDE-only, no session creation) but noted as a possible reviewer
  primitive.

### Design-doc feasibility arc (see §6 for full verdicts)
Two same-day docs reach opposite conclusions: `fallback-analysis.md` says shell-wrapping
"fails all" production requirements → stay `adapter: manual`; `realistic-path-forward.md`
says "requesting new features is not realistic" → build a `devin-shell` adapter with care.
**Operative reality**: the fleet subsequently shipped and ran on the shell-wrap model
(`setup_worker.sh` launching headless `devin --print`), so `realistic-path-forward.md` is the
conclusion that was actually acted on, though nothing formally records that supersession.

### The manual adapter loop as actually practiced (both forks)
`devin.adapter: manual` in shipped config → `dispatch_sessions()` writes a manifest and marks
every request `ok=True` with **no process launched**. Production dispatch bypassed this
almost entirely:

1. **`setup_worker.sh`** (host-side, untracked): fetches `origin/main`; creates a git
   worktree at `<sibling-dir>/issue-<N>` branched off **`origin/main`, never local main**
   (the owner commits in parallel); copies git-excluded `.devin/` gate+skill infra into the
   worktree (hooks run cwd-relative); **junctions ONE shared dev+eval venv** in as `.venv` via
   PowerShell `New-Item Junction` (`UV_NO_SYNC=1`, no cold `uv sync` per worker); labels the
   issue `agent:in-progress` via `gh` directly (bypassing `workflow.dispatch()`); **prints**
   (does not run) the launch command, intended to be run as a **backgrounded Bash-tool call so
   worker exit re-invokes the orchestrating agent session** — an event-driven fleet built out
   of a stateless CLI.
2. Launch command: `cd <worktree> && UV_NO_SYNC=1 devin --prompt-file <prompt> --print
   --permission-mode dangerous --model ${DEVIN_MODEL:-swe-1.6} > <log> 2>&1`.
3. Branch name is recovered by `grep -m1 '^agent/issue-'` against the rendered prompt file —
   brittle coupling to prompt formatting (a template reflow silently breaks dispatch/cleanup).
4. **`finish_worker.sh`** (host-side, post-merge): **critical Windows hazard** — the
   worktree's `.venv` is a junction into the one shared venv; `git worktree remove --force`
   (and `rm -rf`) **follow the junction and recursively delete the shared venv's contents**,
   corrupting every other live worktree. Fix (documented only in a shell comment, not
   enforced by code): delete the reparse point itself first via PowerShell
   `(Get-Item …).Delete()` (removes the link, never the target), **then** `git worktree
   remove --force`, then `git branch -D`.

### Output capture reality (weakest link, dominant failure mode)
`devin --print` writes **only the final assistant message to stdout at process exit** — no
incremental transcript. Observed failure signatures in production logs:
- **0-byte log**: session interrupted before emitting a final message (issue-649: recovered by
  hand-writing a `finish-prompt.md` describing what was done vs. remaining and relaunching a
  fresh session in the same worktree — the uncommitted worktree state survived the crash).
- **39-byte log, exactly `"Error: A tool was rejected by the user"`**: observed 8+ times
  across two dispatch days, even under `--permission-mode dangerous` — consistent with
  org-level permission enforcement that `dangerous` cannot override. Working-tree progress
  from the failed run is NOT lost (a relaunch in the same worktree found and finished
  already-completed work in at least one case).
- `--export` (per-turn transcript export) and the SQLite/transcript stores were **never
  wired in** as a capture mechanism — the entire operational history exists only as
  end-of-process stdout dumps.

### Worker-facing scope contracts (REVIEWER_SCOPE.md / REVIEWER_SCOPE_SCANNER.md, jc-only)
Define worker/human division of labor: worker implements + tests + opens PR; worker must
**not** claim live verification it didn't perform; PR body must include a "Live verification
checklist (for reviewer)" section with concrete probe commands. For scanner-type issues,
synthetic fixtures are forbidden — a `require_live_scan.py` PR-create hook **re-executes** the
worker's own claimed sanity check itself and blocks the PR unless a fresh live fetch clears a
minimum threshold — "it trusts its own re-execution, not your reported number." This
enforcement-by-re-execution pattern is architecturally important and worth generalizing.

---

## 5. Claude Code worker loop as practiced in empericus

Despite "Devin" naming throughout config/docs/labels, the **actual** empericus fleet runs
Claude Code workers, reconstructed entirely from indirect evidence (no code path spawns a
`claude` process; the docs were never updated to reflect this):

- `devin.adapter: manual` in shipped config confirms no programmatic launch.
- `worker.md` was rewritten from Devin-skills syntax to **raw shell commands** (`git fetch
  origin && git switch -c $branch_name origin/main`, `uv run --extra dev pytest -q --tb=short`, `uv run
  ruff check .`/`format .`, `git push -u origin $branch_name`) — the portable interface that
  works for any CLI-driven coding agent, which is precisely what let Claude Code substitute
  for Devin with zero orchestrator code changes.
- Workers operate in `emp-devin-wt/issue-N/` git worktrees (branch names match
  `workflow._branch_name()` = `agent/issue-N-slug` exactly), which **inherit the repo's
  tracked `.claude/settings.json`** (permissions allowlist for `uv run --extra dev pytest/ruff/mypy/
  pre-commit` and git commands) **and hooks** at ordinary `git worktree add` checkout time —
  no bespoke launcher code required:
  - `.claude/hooks/enforce-branch.sh` (PreToolUse on every Bash call): blocks `git commit`
    while on `main`, warns above ~600 changed lines.
  - `.claude/hooks/format-on-edit.sh` (PostToolUse on Edit|Write): runs `ruff format
    --quiet` + `ruff check --fix --quiet` on touched `.py` files, silently swallowing
    failures.
  - `.claude/agents/async-reviewer.md` — a concurrency/asyncio reviewer subagent present in
    every worktree checkout.
- **PR merges are done by the human account directly** (`gh`/GitHub UI), not via
  `merge_ready`: every recently-merged PR's `mergedBy` is the human operator, no bot/service
  identity anywhere in PR authorship history.
- **The GitHub-label state machine stalls after dispatch in practice**: `agent:in-progress`
  was applied programmatically (proving `dispatch()` ran at least once), but issues sit there
  indefinitely — `review`/`record-review`/`merge-ready` were essentially never exercised on
  the label layer; `agent:done` label counts (22 issues, #332-#575) all predate the
  orchestrator's introduction (#576+), i.e. are unrelated legacy labeling. **Humans close the
  loop manually** by merging PRs directly; the CLI's review/merge machinery is unused in
  observed practice despite being fully implemented and tested.
- No `.var/devin-orchestrator/` state exists anywhere on disk (consistent with it being
  gitignored+regenerable, but combined with the stalled labels this indicates the pipeline
  past `dispatch` genuinely isn't being driven).
- Some PRs bypass the orchestrator's branch-naming convention entirely, using
  `claude --worktree`-auto-named branches (`senk/competent-hopper-8b2673`,
  `keen-carson-ae00b9`) — ad hoc Claude Code sessions used directly, outside the
  issue-labeling pipeline.

**Conclusion for the new repo**: the Claude Code "adapter" that already works in production
is not code at all — it's (1) `git worktree add`, (2) repo-tracked `.claude/` guardrails
inherited automatically, (3) a human pasting `worker-prompt.md` into a session in that
worktree. The extraction should promote this into a first-class, literal adapter
implementation (worktree lifecycle + prompt hand-off automated) rather than continuing to rely
on it being assembled by hand each time — and should close the observed gap where merges
happen outside `merge_ready()` and issue labels never reconcile.

---

## 6. Design-doc verdicts

### devin-shell adapter feasibility — CONTESTED, operative answer is "feasible with care"
- `architecture-proposal.md` (2026-06-24): **NOT FEASIBLE** — no session-creation API; shell-
  wrapping "technically possible but not recommended due to fragility, race conditions, and
  lack of official support."
- `fallback-analysis.md` (same date): shell-wrapping "fails all" of reliability/
  maintainability/supportability/scalability/security/testability requirements; recommends
  staying `adapter: manual` and requesting APIs from Cognition.
- `realistic-path-forward.md` (same date, contradicts the above): "Requesting new features is
  not realistic" → build a `devin-shell` adapter: `devin --prompt-file <FILE> --print`
  non-interactive launch, pre-create the git branch via `git`, track session ID via **read-only
  SQLite polling** of `sessions.db` with `OperationalError` retry (2s interval), manage the
  subprocess (terminate on timeout), keep `manual` as fallback. Documented limitations: no
  post-creation monitoring, no automatic PR creation (workers self-serve this via `gh`), no
  session termination API, fragile to CLI updates, **"Single-threaded: Cannot launch sessions
  in parallel (SQLite contention)."**
- `acp-analysis.md`: ACP (Agent Client Protocol) is **not viable** for orchestration — it's
  for IDE-consuming sessions, not creating/monitoring/terminating them; would still require
  shell-wrapping `devin acp` as a subprocess anyway. "Direct shell-wrapping is simpler and
  provides the same capabilities."
- Cloud DRS (Devin's remote sandbox product): architecturally misaligned — no issue binding,
  no PR integration, ephemeral-testing lifecycle, per-session cloud cost.
- **Verdict for the new repo**: build the devin-shell adapter per `realistic-path-forward.md`
  §1, but fix its own internal contradictions and gaps before porting (see §7 fix list) —
  particularly the undocumented `--permission-mode bypass` value used in both doc code
  samples (should be `dangerous`), the missing `--print` flag in the reference
  implementation's actual Popen call (contradicts its own prose), and the unresolved question
  of whether `--print` sessions are recorded in `sessions.db` at all.

### Fleet-readiness items — three concrete, unapplied hardening patches
All three target `.devin/hooks/require_live_scan.py`, deliberately **deferred until no worker
is live** (the gate executes from disk on every worker push — editing mid-dispatch races an
in-flight check):
1. **Git-only PR-existence fallback**: `_branch_has_open_pr()` currently fails **open**
   (returns False = not gated) whenever `gh pr list` errors, so a rework push via plain `git
   push` (never re-running `gh pr create`) can skip the breadth-recheck gate on any `gh`
   flakiness. Fix: keep `gh pr list --head <branch> --state open --json number` as
   authoritative, fall back to `git ls-remote --heads origin <branch>` (dependency-free) on gh
   failure — fails **closed** for reworks (branch already on origin = gate it) and **open**
   only for genuinely-new branches (still covered by the first-push gate).
2. **Self-diagnosing zero-tenant probe**: gate blocks currently only annotate tenants that
   threw an exception; the common "returned 0 with no exception" case gives no cause, forcing
   manual robots.txt/sitemap probing per kickback. Fix: a bounded (budget=3, block-path-only,
   ≤12s each) diagnostic pass classifying into NO-SITEMAP / RAW-HAS-JOBS (parse/shape bug —
   "the gold signal") / SITEMAP-INDEX / SITEMAP-EMPTY / SITEMAP-NO-JOBLINKS, surfaced as `WHY
   <diag>` next to the existing `ERR` annotation.
3. **Diff-scoped pytest-contract gate**: the current gate chain (block_destructive →
   require_live_scan → require_ci_clean) **never runs pytest** — a worker can pass every gate
   and still open a PR failing the required "Tests passed" CI check (this exact failure
   occurred). Fix: a `require_contract_tests` gate triggered only when the diff touches
   specific scanner-registry paths, running a fast deterministic pytest subset (not the full
   suite — "minutes/push and flaky-prone") in the venv subprocess. Key quote: **"The
   `/preflight` skill mirrors CI but workers optimize to gates, not skills — so it must be
   enforced, not offered."**

### Extensibility items — implemented and deployed (jc only)
Three-layer determinism stack: **rules** (`.devin/AGENTS.md` — one issue per session,
assigned-branch-only, conventional commits, PR contract) → **skills** (6 SKILL.md files:
create-branch/commit/test/push/create-pr/complete, invokable manually, autonomously, or by
reference from rules) → **hooks** (`.devin/hooks.v1.json` — SessionStart branch-pattern gate,
PreToolUse destructive-command block via naive substring matching on
`DESTRUCTIVE_PATTERNS` — **bug: would block legitimate `ruff format`/`pre-commit`
invocations**, PostToolUse git-ops audit logging, SessionEnd dirty-tree warning). Verdict:
"feasible and recommended within current capabilities" — each layer covers the layer above
it's failure mode (rules can be ignored, skills can be skipped, hooks cannot). **Not ported to
empericus** ("dropped .devin/ Devin-cloud session scaffolding... the fleet here is Claude-Code
workers, not Devin sessions" — per the port commit message), leaving empericus's `rework.md`
referencing a `/preflight` skill that doesn't exist there (byte-identical leftover from jc,
worker.md was rewritten but rework.md wasn't).

---

## 7. COMPLETE deduplicated bug/fix list, prioritized

### P0 — Correctness bugs that silently corrupt state or security

1. **[MERGE] `merge_pr` empty-stdout bug (jc only; emp already fixed)** — `gh pr merge`
   prints success to stderr; jc's `return str(self.run(args))` reports `merged=false` after a
   real merge. Fix: adopt emp's `output or f"merged #{number}"` fallback verbatim, add a unit
   test asserting merged-truthiness on empty stdout (currently masked in both forks' test
   suites because `FakeGitHub.merge_pr` returns a hardcoded truthy string). *Evidence:
   jc-source github.py:142-152; emp-source github.py:142-158; tests recon.*

2. **[SECURITY] Shell injection via `dispatch_command` string form** —
   `adapters.py:148,193-194`: string-form `dispatch_command` runs `shell=True` with
   `issue_title` `.format`-interpolated directly into the shell string; issue titles are
   attacker-influenceable on any repo accepting public issues. Fix: remove shell=True string-
   command support entirely, or pass `issue_title` via env/argv, never string interpolation.
   *Evidence: jc-source adapters.py:148,193-194.*

3. **[MANUAL ADAPTER] Fabricated success** — `_manual_result` returns `ok=True`
   unconditionally with no session ever launched; `dispatch()` then labels the issue
   `agent:in-progress` before any worker exists, and `session-results.json` is a fabricated
   paper trail (`command`/`returncode` null). Fix: `ok` for the manual adapter should mean
   "manifest written, awaiting operator confirmation," not success; don't flip
   `agent:in-progress` until a session is actually observed. *Evidence: jc-source
   adapters.py:120-121; devin-ops recon confirms in production this diverged so far that
   `session-manifest.json` froze at one stale issue while ~60 issue dirs existed.*

4. **[STATE] state.json clobbering drops recorded decisions** — `review()`
   wholesale-overwrites `state['prs'][n]`, erasing the `decision` key `record_review` set;
   `intake()` wholesale-overwrites `state['issues'][n]`, erasing dispatch status. Confirmed in
   production (`pr-497`: decision present on disk, absent in state after a later
   `review_packet` event). Fix: merge-update instead of dict-replace; treat the append-only
   `events` array as canonical and derive `issues`/`prs` views from it, OR make disk artifacts
   (`review-decision.json`) the sole authority and stop mirroring decisions into state.json at
   all. *Evidence: workflow.py:245-254,93-100 both forks; jc-ops-state recon pr-497 case.*

5. **[STATE] No locking, unsafe read-modify-write** — concurrent orchestrator invocations
   lose updates silently (last-writer-wins); `state.py:30` uncaught `JSONDecodeError` on
   corrupt state.json crashes with a raw traceback instead of a graceful reset/exit-2. Fix:
   catch `JSONDecodeError` in `load_state` (reset-with-backup), add a lock file or
   single-writer discipline. *Evidence: state.py:25-37 both forks.*

6. **[CROSS-FAMILY] Failure stub reused as permanent success** — reuse check is
   `report_path.exists() and st_size>0`; a failed run's `UNAVAILABLE` stub satisfies both
   conditions, so **one codex timeout permanently poisons that PR's cross-family slot** —
   every subsequent `review()`/`loop()` pass reuses the stub as `ok=True, reused=True` and
   never retries. Fix: mark stub reports distinguishably (filename suffix or frontmatter
   flag) and retry failed runs; optionally add max-attempt/backoff config. *Evidence:
   jc-source workflow.py:434-439; jc-ops-state confirms this happened in production (pr-651,
   approved 3 minutes after a swallowed timeout).*

7. **[WINDOWS] Inconsistent subprocess encoding** — `adapters.py`'s `subprocess.run` lacks
   `encoding='utf-8', errors='replace'` (present in `github.py` and `cross_family.py`);
   Windows cp1252-decodes worker stdout/stderr, and a `UnicodeDecodeError` (a `ValueError`) is
   **not caught** by the existing exception handlers → dispatch crashes. Also,
   `TimeoutExpired.stdout` is `str()`-coerced without checking for `bytes`, producing `"b'...'"`
   garbage (cross_family.py already has the correct bytes-safe pattern — never backported).
   Fix: centralize subprocess execution in one runner utility with consistent
   encoding/errors/timeout handling. *Evidence: adapters.py:142-160 both forks.*

8. **[WORKTREE] Shared-venv-destruction hazard is enforced only by a shell comment** —
   `git worktree remove --force` / `rm -rf` follow the `.venv` NTFS junction into the ONE
   shared venv and delete its contents, corrupting every other live worktree.
   `finish_worker.sh` does the right thing (delete the reparse point via PowerShell
   `.Delete()` first) but this lives in an untracked shell script, not code. Fix: port this
   teardown ordering into the orchestrator proper (Python, with a test) as a first-class
   worktree-lifecycle helper. *Evidence: devin-ops, jc-ops-state recon both flag this
   identically.*

### P1 — Reliability / operational-integrity bugs

9. **[LOOP] No per-PR error isolation** — `loop()`'s `gh.merge_pr` uses
   `allow_failure=False`; a merge conflict/branch-protection rejection raises `GitHubError`
   and aborts review/merge of ALL remaining PRs in the batch, exiting 2 with no state saved
   for that cycle. Fix: wrap per-PR review/merge in try/except, record failure, continue.
   *Evidence: workflow.py:474-497(jc)/359-382(emp).*

10. **[DISPATCH] `--issues` parsing bugs (emp)** — `only_issues.replace(" ", "")` deletes
    ALL spaces before splitting (`"5 6"` → issue `56`, silently wrong, not an error);
    non-numeric tokens raise an uncaught `ValueError` (no handler catches it, raw traceback);
    duplicate numbers aren't deduped (double-dispatch); `--limit` is silently ignored when
    `--issues` is given; `loop()` cannot pass `only_issues` at all, so wave-ordered dispatch
    requires bypassing the documented main loop entirely. Fix: per-token `.strip()`, explicit
    numeric validation with a `CommandResult` error (not a crash), order-preserving dedup,
    report skipped/unknown numbers in the result data, and plumb `only_issues` through
    `loop --issues`. *Evidence: emp-source workflow.py:105-116; emp-practice/tests recon.*

11. **[MERGE GATE] `delete_branch` default/YAML contradiction (emp)** — dataclass default
    `delete_branch=True` contradicts shipped YAML's `delete_branch: false`; any deployment
    that omits/renames the YAML silently reverts to `--delete-branch` and reintroduces the
    aborted-post-merge-label-update failure for worktree-based workers. Fix: default
    `delete_branch=False` in the dataclass (worktree-safe by default), or decouple merge from
    branch cleanup entirely — attempt deletion after the label transition, treat its failure
    as a warning, not an abort. *Evidence: emp-source config.py:66 vs orchestrator.config.yaml:26-30.*

12. **[LABELS] Asymmetric/accumulating label transitions** — `review()`/`loop()`
    unconditionally re-add `agent:pr-open`+`agent:reviewing` every pass, stomping a
    `needs-rework` signal; `record_review(blocked)` adds `human-needed` but never removes
    `reviewing`; `approved` changes no labels at all (reviewing persists until merge); only
    `merge_ready` sweeps active labels. `agent:queued` is bootstrapped, documented, and
    **never applied by any code path** (vestigial). Fix: one transition function owning
    add/remove pairs, replacing scattered calls across dispatch/review/record_review/
    merge_ready; either wire `agent:queued` or remove it everywhere (labels, bootstrap,
    docs, state diagram). *Evidence: workflow.py both forks; docs recon confirms the
    documented state diagram doesn't match code.*

13. **[REVIEW] Dead review-gating config** — `max_rework_cycles`, `require_tests_or_rationale`,
    `require_issue_link` are parsed from YAML but referenced nowhere in code — rework can
    cycle indefinitely, issue-link and tests requirements are prompt-text-only aspirations.
    Same for `allow_auto_merge_when_pending`, `orchestrator_model_tier`. Fix: enforce
    `max_rework_cycles` as a hard counter (count `request_changes` events per PR, auto-escalate
    to `blocked`/`human-needed` past the cap — directly maps to the external-research finding
    that agents "ghost" rather than converge past 1-2 review-reject-retry cycles, see §10);
    either implement or delete each remaining dead key. *Evidence: config.py both forks;
    papers research idea "bounded retry, not indefinite re-dispatch."*

14. **[LINKING] `linked_issue_number` over-matches** — matches ANY `#N` in a PR title
    (not just an anchored Closes/Fixes clause) before consulting the qualified body keywords —
    a title like "Improve logging (see #12)" mis-binds the PR to issue 12, driving wrong
    labels/review packets/merge actions. Fix: require the branch-prefix pattern for head refs
    and anchored close-keywords for titles; drop the bare `#N`-in-title heuristic or make it
    explicit last-resort behind a config toggle. *Evidence: github.py:166-177 both forks.*

15. **[REVIEW SCHEMA] `record_review` drops `required_changes` field** — the template
    `review()` writes includes a `required_changes` field; `record_review`'s write omits it —
    schema drift between the pending template and the recorded decision. Also: **empty
    `summary` is silently accepted** (`--summary-file` optional, no validation) — approvals
    with zero rationale observed in production (`pr-650`, `pr-651`). Fix: unify the schema
    (always include `required_changes`), require a non-empty summary on `record-review`.
    *Evidence: workflow.py both forks; jc-ops-state pr-650/651.*

16. **[CI CHECKS] SKIPPED/NEUTRAL/CANCELLED bucketed as failed; duplicate check names
    collapse** — arguably-conservative but undocumented; a matrix leg sharing a check name
    with a later-listed passing leg silently masks a failing leg. Fix: document the
    SKIPPED-as-failed decision explicitly (or make it configurable), and either error on
    duplicate required-check names or aggregate them (worst-of) instead of last-write-wins.
    *Evidence: checks.py:20-46 both forks.*

17. **[DRY-RUN] Hardcoded readonly-command allowlist** — `_is_mutating` is a fixed
    string-prefix list; any newly-added readonly `gh` call is dry-run-blocked-by-default
    until the list is manually edited (violates no-hardcoded-lists principle). Also,
    `--dry-run` only suppresses `gh` mutations — state writes, the command adapter's
    subprocess, and the cross-family subprocess all still execute for real, oversold by
    `WORKFLOWS.md`. Fix: replace the allowlist with an explicit per-method mutating flag
    derived from the call site; extend dry-run coverage to state writes and adapter/model
    subprocess calls, or narrow the documented claim. *Evidence: github.py:180-194;
    docs recon fix candidate.*

18. **[PROMPTS] No escaping of injected content; `rework.md` prompt-injection risk** —
    `string.Template.safe_substitute` with no fencing/escaping means an issue body containing
    `` ``` `` breaks the prompt's own markdown fence — a direct injection channel from issue
    authors into worker sessions. Fix: fence/escape injected values (randomized or
    length-prefixed fence, or indented block-quote). *Evidence: prompts.py:14 both forks;
    GitHub Blog research "Untrusted Input in Workflows" red flag corroborates.*

19. **[PROMPT DRIFT] empericus `rework.md` references a dropped skill** — byte-identical
    leftover from jc instructs workers to "run /preflight" — a Devin-cloud skill explicitly
    dropped in the empericus port (worker.md was rewritten to raw shell, rework.md wasn't).
    Fix: generate worker AND rework prompts from one per-fleet tooling profile so they cannot
    drift apart; audit for other such orphaned references before shipping the union.
    *Evidence: emp-source prompts/rework.md:24-26 vs commit b2fd4e0.*

20. **[CONFIG] Unhelpful error on unknown YAML key** — a typo like `requird_checks` raises
    a bare `TypeError` from a frozen-dataclass constructor with no section/key context. Fix:
    add a validation layer (or minimal pydantic/attrs schema) with actionable error messages.
    *Evidence: config.py:128-146 both forks.*

21. **[PACKAGING] `DEFAULT_CONFIG_PATH` lives inside the installable package** — per-repo
    config edits currently mutate the installed package directory. Fix: separate packaged
    defaults from per-repo config/prompt overrides as the primary extraction seam (this
    appears to already be DONE per the scaffold status note — verify and close).

### P2 — Devin-shell / Claude Code adapter build items (net-new for the remaining work)

22. **[ADAPTER] Fix the design docs' own internal contradictions before porting** —
    `realistic-path-forward.md` §1.1 prescribes `devin --prompt-file <FILE> --print` but its
    own reference implementation omits `--print` (would sit at an interactive prompt under
    Popen, not run headlessly); both doc code samples use an **undocumented**
    `--permission-mode bypass` value (valid values are `auto|accept-edits|smart|dangerous`;
    `bypass` is only an interactive `/bypass` alias for `dangerous`). Fix: validate and use
    `devin --prompt-file <FILE> --print --permission-mode dangerous` before writing the
    adapter.

23. **[ADAPTER] Resolve whether `--print` sessions hit `sessions.db` at all** — the design
    corpus contradicts itself (`api-discovery-report.md` line 43 says "creates transient
    session"; its own conclusion says "no persistence"). This is the single biggest open
    premise under the SQLite-polling session-tracking design — if false, session tracking
    must pivot to process exit code + `--export` files + hook logs + GitHub PR-existence
    checks instead.

24. **[ADAPTER] Fix the session-attribution race** in the sample `_wait_for_session_id`:
    `SELECT id FROM sessions WHERE working_directory = ? ORDER BY created_at DESC LIMIT 1`
    with no `created_at` floor and no exclusion of pre-existing IDs returns a **false
    positive** for any prior session in the same directory; two concurrent dispatches in one
    repo cross-attribute sessions. Fix: snapshot existing session IDs (or max `created_at`)
    immediately before dispatch and only accept a session with `created_at >= dispatch_time`
    and `id not in snapshot`; consider `devin list --format json` (documented, programmatic)
    instead of raw undocumented-schema SQLite reads.

25. **[ADAPTER] Output capture** — implement `--export <path>` alongside stdout redirect
    and/or harvest `%APPDATA%\devin\cli\transcripts\<session_id>.json` post-exit; record
    structured exit code + start/end timestamps per run (the old `run.log` header/footer
    convention, machine-written this time, not ad hoc).

26. **[ADAPTER] Detect and auto-handle the 39-byte tool-rejection signature** — classify
    `"Error: A tool was rejected by the user"` as a permission-abort, diff the worktree for
    salvageable progress, and auto-relaunch with a finish-style prompt (the manual issue-649/
    issue-578 recovery, automated).

27. **[ADAPTER] Worker liveness/timeout supervision** — no lifecycle API exists, so track
    spawned PID + start time in the hub; distinguish in-flight-vs-dead 0-byte logs; enforce a
    max session duration with kill + salvage-relaunch, per the empirical interaction-round
    budgets in §10.

28. **[ADAPTER] Honor the documented single-threaded SQLite-contention ceiling** — serialize
    session creation (dispatch lock) or shard workers across machines/profiles; do not assume
    parallel local Devin-shell dispatch works.

29. **[CLAUDE CODE ADAPTER] Promote the emergent worktree+`.claude/`-inheritance pattern
    into real code** — automate worktree creation + prompt hand-off (headless
    `claude -p <rendered-prompt> --cwd <worktree>` or equivalent) instead of relying on a
    human to assemble it by hand each time; add a reconciliation pass that detects PRs merged
    outside `merge_ready()` (mergedBy != automation identity, or a merged PR whose issue is
    still labeled `agent:in-progress`) and corrects the stale label — this is the concrete,
    observed gap in empericus's live usage.

30. **[BRANCH TRACKING] Replace grep-based branch recovery** — `grep -m1
    '^agent/issue-'` against a rendered prompt file is brittle; carry the branch name as
    structured data (manifest field, env var, or a small sidecar JSON) instead.

### P3 — Doctor command / fleet-readiness / hardening (explicitly in scope for remaining work)

31. **Add a `doctor` command** that cross-checks configured `required_checks` against live
    `gh pr checks --json name` output on a recent PR and flags mismatches — turns the
    documented "must match exactly" tribal knowledge into a detectable error instead of a
    silent permanent-block trap. *Evidence: multiple recons flag this exact operational
    pitfall independently (job-cannon required_checks, empericus required_checks).*

32. Port FLEET-READINESS Items 1-3 from job-cannon's design docs into the new repo's
    equivalent gate (see §6): git-ls-remote PR-existence fallback, self-diagnosing zero-tenant
    probe pattern (generalized — this specific probe is job-cannon-domain-specific, but the
    **pattern** of "bounded, block-path-only, budgeted diagnostic sub-probe attached to a gate
    failure" is generalizable), and a diff-scoped contract-test gate mapping touched paths to
    a fast deterministic pytest subset (config-driven mapping, not hardcoded literals).

33. Fix `block_destructive.py`-equivalent naive substring matching (`'rm -r'`, `'format'`
    would false-positive-block legitimate `ruff format`/`pre-commit` invocations) — needs
    word-boundary/argv-aware matching.

34. Version the entire gate/hook layer in-repo (currently git-excluded WIP in both source
    repos) with its own CI: ruff/format checks on the hook files themselves, a pyright scan
    confirming embedded script literals stay f-string/backslash-free, and a smoke test.

### Deduplication notes
Items 1, 6, 7, 8 were independently flagged by 3+ readers each — highest-confidence bugs.
Item 3 (manual-adapter fabricated success) and item 29 (Claude Code adapter promotion) are two
faces of the same root issue: the manual adapter's contract ("ok=True, no verification") is
correct as a *fallback*, but production dispatch never actually goes through it, meaning the
labeling-on-success invariant the tests carefully pin is not actually exercised in the field.

---

## 8. Docs inventory

Both forks ship an identical 5-document suite at `docs/devin-orchestration/`: `README.md`
(61 lines — kit overview, 9-label list, 3-condition merge policy), `ARCHITECTURE.md` (93 —
components, artifact tree, state-machine diagram, merge gate, context-management philosophy,
adapter boundary), `QUICKSTART.md` (112 — 9-step shortest-safe-path, explicit "who runs
this" role split), `RUNBOOK.md` (375 — 10-phase handholding guide incl. 5 documented recovery
cases: context reset, vanished worker, merge conflict, duplicate PRs, mislabeled issue),
`WORKFLOWS.md` (94 — command crib sheet). Only `README.md`'s required-checks list differs
between forks (repo-specific CI job names; empericus's version adds the "must match
`.github/workflows/ci.yml` job `name:` fields exactly" caveat — the single doc improvement
either fork made to the inherited text).

**Staleness/inaccuracy findings** (all confirmed against code):
- `QUICKSTART.md`/`RUNBOOK.md` describe a **paste-into-the-Devin-app** manual model; neither
  fork's live operation matches this — job-cannon runs headless CLI workers via
  `setup_worker.sh`, empericus runs Claude Code sessions in worktrees. Neither the worktree
  fleet model, the `.devin/` hooks+skills infra, cross-family review, spec-review, `dispatch
  --issues`, the `command` adapter, or `session-results.json` are mentioned anywhere in the
  5-doc suite.
- empericus's copies are **stale in repo-identifying specifics**: `QUICKSTART.md:37`/
  `RUNBOOK.md:84` still say "Job Cannon Automated-Ready Orchestrator"; `RUNBOOK.md:18`
  expects repo root `C:/Users/senki/repos/job-cannon`; `RUNBOOK.md:27` expects
  `Senkichi/job-cannon` gh access; `RUNBOOK.md:301` says merge "deletes the branch" — directly
  contradicting empericus's own `delete_branch: false`.
- **Doc promises not kept by code**: merge gate's "PR is associated with an issue" condition
  (README/ARCHITECTURE) is unenforced (`require_issue_link` is dead config); "`.var` can be
  regenerated" (README:21) is false for `review-decision.json` (the actual merge-gate
  authority, local-only); "`--dry-run` suppresses mutating operations" (WORKFLOWS:82-86) is
  true only for `gh` calls, not state writes or adapter/model subprocesses.
- job-cannon ignores `.var` only via **local, unversioned** `.git/info/exclude` — a fresh
  clone would expose (and could commit) the entire runtime tree including ~10 operational
  design docs; empericus does this correctly via committed `.gitignore`.
- `.devin/orchestrator.md` (job-cannon-only, 24 lines) is an extra pointer doc not referenced
  by the main suite.
- **Unversioned design-doc corpus** (job-cannon `.var/devin-orchestrator/`, ~10 files,
  never committed): `FLEET-READINESS-DESIGN.md`, `architecture-proposal.md`,
  `extensibility-design.md`, `devin-cli-reference.md` (24K, the CLI reference of record),
  `api-discovery-report.md`, `fallback-analysis.md`, `realistic-path-forward.md`,
  `acp-analysis.md`, `REVIEWER_SCOPE.md`, `REVIEWER_SCOPE_SCANNER.md` — this is real
  architectural reasoning currently one `git clean` away from permanent loss; needs a
  versioned home in the new repo (e.g. `docs/ops-notes/` or `docs/adr/`).

**Fix list for docs** (folded into §7 conceptually, listed here for completeness): rebuild the
5-doc suite repo-agnostic (template variables for session name/repo root/gh slug/required
checks/branch prefix/merge strategy); document the cross-family pass, `dispatch --issues`,
the command adapter, `session-results.json`, exit codes, `--config`/`--repo`; complete
`ARCHITECTURE.md`'s artifact tree; add a dedicated worktree-fleet-model doc (or RUNBOOK phase)
covering the setup/finish-script lifecycle, the junction-deletion hazard, and the
"backgrounded-launch-so-exit-reinvokes-orchestrator" pattern; migrate the unversioned design
corpus into a versioned docs location.

---

## 9. Test inventory

**job-cannon**: `tests/test_devin_orchestrator.py`, 558 lines, **30 tests, all pass**.
**empericus**: `tests/test_devin_orchestrator.py`, 313 lines, **17 tests, all pass, 0.8s**.

**16 tests shared near-verbatim** across both forks, covering: default-config parsing
(pins repo-specific required-check names — a coupling that should split into
dataclass-defaults-vs-example-config tests in the new repo), runtime-path derivation,
state round-trip, worker-prompt rendering, `slugify`, `label_names` extraction,
`linked_issue_number` (incl. the dependabot-negative case), `summarize_checks` across all 3
gh output shapes, state.json atomicity + `generated_at` refresh, the `--json`-after-subcommand
CLI hack, `github.run`'s `allow_failure` JSON-on-nonzero-exit path, manual-adapter dispatch
(manifest + prompt + label), command-adapter dispatch success (real `sys.executable`
subprocess, label applied), command-adapter dispatch failure (no label, `status=
dispatch_failed`), and merge-ready's approved-decision-required-then-merge path.

**job-cannon-only: 14 cross-family tests** — `render_command` (list vs string templating),
happy-path report-with-caveat write, timeout/nonzero-exit/missing-binary all captured (never
raises), shipped-config-enables-cross-family, absent-block-defaults-disabled, YAML
list-to-tuple coercion, review() injecting the cross-family section exactly once, idempotent
report reuse across a second `review()` call, per-call `--no-cross-family` override, draft-PR
skip, spec_review happy path and missing-artifact-error path.

**empericus-only: 1 test** — `test_dispatch_only_issues_selects_explicit_subset`: pins
space-stripping, comma-split, silent-skip of non-dispatchable/unknown numbers, and that only
the dispatchable explicit match gets dispatched+labeled.

**Shared fixture design** (duplicated per-file, not centralized): a hand-rolled `FakeGitHub`
duck-typed spy (~60 lines each) with one canned issue (#123) and one canned PR (#456);
`issue_view(number)` **ignores its argument and always returns `issues[0]`** — would silently
corrupt any future multi-issue test; `pr_checks` canned check names already **drifted between
forks** (mirrors each fork's `required_checks` default) — proof the duplication itself causes
test rot, independent of the underlying orchestrator code.

**Fragile seams**: 5 occurrences (job-cannon only) of string-target
`monkeypatch.setattr('automation.devin_orchestrator.workflow.run_cross_family_review', ...)`
hardcode the package's dotted import path — any package rename breaks all 30/17 tests at
collection time, as does the `tests/__init__.py` + pytest-rootdir-sys.path-insertion trick
both forks rely on for imports (no editable install, no `pythonpath` in `pyproject.toml`).

**Coverage gaps common to both suites** (none directly tested in either fork): `status()`,
`intake()`, `bootstrap_labels()`, `record_review()`'s full decision-routing logic (rework
prompt content, needs-rework/human-needed label application, `--comment` path), `loop()`
end-to-end orchestration order, `GitHub.run`'s dry-run `_is_mutating` gating, `find_repo_root`
fallback, adapters' unsupported-adapter/timeout/bad-template branches, `github.run`'s
`allow_failure` path with **empty** stdout (returns stderr-as-value — untested), and
`append_event`'s 200-event cap trim. The empericus `merge_pr` empty-stdout fix (item 1 in §7)
is **not pinned by any test in either fork** — `FakeGitHub.merge_pr` returns a hardcoded
truthy string, masking exactly the bug it fixed.

**Recommended for the new repo** (test-infra fix candidates, folded from tests recon):
promote `FakeGitHub` to one shared fixture with a `typing.Protocol` surface definition so
fake/real conformance is statically checked; add the missing merge_pr-empty-stdout test;
port the 14 cross-family tests + the 1 `--issues` test into the union (already reportedly done
per scaffold status — verify); split shipped-config tests from dataclass-default tests; add
coverage for the gap list above; replace string-target monkeypatch of
`run_cross_family_review` with constructor/parameter injection (the pattern the codebase
already uses successfully via `cross_family.py`'s `runner=` kwarg, just not threaded through
`workflow.py`'s call site).

---

## 10. BRIGHT IDEAS FROM THE FIELD (external research, deduplicated)

Two independent research passes (`papers` = academic 2025-2026 arXiv; `systems`/
`practitioners` = OSS projects + vendor/practitioner writeups) returned overlapping findings.
Deduplicated below; each has a source and an applicability verdict **for this specific
system** (single-operator, Windows, deterministic Python hub, no LangGraph, gh-CLI-driven).
Items marked **HIGH** are folded into the fix list in §7 or noted as new items below.

### HIGH applicability — fold into the build

**A. Deterministic, non-LLM verification before spending review budget ("janitor" pattern).**
Don't trust the worker's self-report of "done" — verify via concrete, cheap, deterministic
signals (tests green, diff non-empty, files the issue named actually touched, PR description
non-empty) *before* routing to the adversarial LLM reviewer at all. If the janitor check
fails, don't even generate a review packet — flag `dispatch_failed`-equivalent and skip
straight to a rework/retry decision. — *Source: Bernstein orchestrator
(bernstein.readthedocs.io/architecture), corroborated by GitHub Blog's "Agentic Ghosting" red
flag (empty PR description = auto-request restructure before deep review) and Intercom Eng's
production tiered-reviewer data.* **Verdict: HIGH.** This is cheap (pure Python, no LLM calls)
and directly fixes the observed production pattern where `pr-650`/`pr-651` were approved with
empty review summaries and one was approved while CI was still `IN_PROGRESS` — a janitor pass
would catch both before a human/reviewer ever looks at the packet. **New fix-list item: add a
pre-review deterministic gate (janitor) that checks CI status is terminal (not IN_PROGRESS),
diff is non-empty, PR description is non-empty, and required-check names actually appear in
`gh pr checks` output — before generating `review-prompt.md`.**

**B. Scope AI review to the residual deterministic checks can't cover; strengthen CI as the
primary gate.** Same-training-distribution AI reviewers share blind spots with AI generators
on domain-convention violations — one controlled study found 0-100% detection rate depending
on domain opacity, correlated across model families (all 4 tested models missed the same
healthcare bug; 3 of 4 confidently endorsed the same wrong aviation rule). The adversarial
review prompt's real job is catching **undocumented architectural intent violations**, not
re-deriving correctness that a unit test would already catch. — *Source: "The Specification
as Quality Gate" (arXiv:2603.25773).* **Verdict: HIGH.** Directly informs the review-prompt
rewrite: re-scope `review.md`/`cross_family_review.md` to explicitly ask "does this violate an
architectural convention that isn't written down anywhere" rather than "is this code correct"
— correctness should already be answered by CI + the janitor pass (item A). **Fold into fix
list: rewrite review prompts to target Category-D-style undocumented-intent defects; treat
required-CI-green as the primary correctness gate, not a co-equal signal alongside LLM review.**

**C. Cross-model review as an independent veto channel, not a consensus vote.** Cross-family
panels have only ~2.0-2.4 "effective independent judges" even with different model families,
due to shared training-data conventions; majority-voting among correlated judges dilutes the
single most-informed judge's signal. The system's existing "optional cross-model-family pass"
design is validated, but should never be implemented as "both must approve" or "average the
verdicts" — either reviewer flagging a blocker should block. — *Source: "The Specification as
Quality Gate" (2603.25773); "Correlated Errors in LLMs" (2506.07962); "Nine Judges, Two
Effective Votes" (2605.29800).* **Verdict: HIGH** — validates keeping `cross_family.py`'s
existing non-blocking, additive "leads not verdicts" framing (already correct!) but confirms
it should stay an **independent veto**, never a required-agreement gate, if/when it becomes
mandatory rather than advisory.

**D. Bounded rework-cycle cap with escalation, not indefinite retry.** Agent PRs that receive
open-ended/subjective feedback tend to "ghost" (abandon) rather than converge; empirical
success-round CDFs show 80% of successful resolutions land within 19-25 interaction rounds
while failures show a long tail past 50 with no marginal value; production multi-agent SWE
systems cap retries at exactly one structured second attempt before escalating to a human. —
*Source: Liu et al. arXiv:2509.13941 (RQ1.3); MSR-2026 arXiv:2601.00753 ("ghosting on
subjective feedback"); multi-agent-SWE production-system convergence patterns.* **Verdict:
HIGH.** Directly actionable: this is the concrete backing for enforcing `max_rework_cycles`
(§7 item 13) — cap at 1-2 rework cycles, then auto-escalate to `agent:blocked`/
`agent:human-needed` rather than looping the same worker on reviewer rejections indefinitely.
Also implies rejection feedback (`record-review --decision request_changes`) should be
maximally concrete (specific line, specific failing test, specific named convention) rather
than open-ended stylistic commentary, since vague feedback is exactly where the retry loop
thrashes instead of converging.

**E. Deterministic pre-filters before spending any LLM review tokens.** Cheap, mechanical
gates implementable in plain Python: file-count/diff-line-size threshold (flag PRs spanning
too many unrelated files for split-request rather than reviewing as-is — also caps blast
radius for atomic revertibility, Stripe-style <5-min-reviewable PRs), non-empty PR
description requirement, and a specific "diff touches only test files but the issue wasn't
about tests" check (a strong signal tests were weakened to pass rather than code fixed —
maps onto the existing `checks.py` CI-gaming blind spot). — *Source: GitHub Blog "Agent pull
requests are everywhere" (5 red-flags: CI Gaming, Code Reuse Blindness, Hallucinated
Correctness, Agentic Ghosting, Untrusted-Input-in-Workflows); Paddo "Agents Merge" (Stripe PR
size discipline).* **Verdict: HIGH.** Extends the janitor pattern (item A) with concrete
thresholds; also directly reinforces §7 item 18 (prompt-injection via untrusted PR/issue
text) — GitHub's own red-flag taxonomy names this exact risk independently, corroborating it
as a real, known class of failure, not a theoretical concern.

**F. Task-complexity gate at intake, before any worker is dispatched.** A cheap static-feature
classifier (file types touched, estimated patch size) predicts high-review-effort PRs at
AUC 0.96 using only information available before dispatch; agents excel at narrow-scope
automation (28% zero-friction merge rate in one large study) and struggle specifically with
broad/ambiguous scope. — *Source: arXiv:2601.00753 (MSR 2026); Cognition's own "Devin 2025
Performance Review" pinning the reliable scope sweet-spot at 4-8 human-hours with an
objectively verifiable outcome.* **Verdict: HIGH.** Directly actionable at the
intake/labeling step, cheaper than anything downstream: before an issue is admitted to
`automated-ready`, require (1) stated success criteria, (2) an identifiable verification
command/check, (3) bounded estimated scope. **New fix-list item: add an intake-time
admission checklist (can be a doctor-command-adjacent lint on issue bodies) gating the
`automated-ready` label, rather than accepting any labeled issue as dispatchable.**

**G. Structured worker-prompt template with an explicit Forbidden-Actions/Boundaries
section, and never edit a prompt mid-flight.** Devin's own documented Playbook template
(Overview → Requirements → Procedure → Specifications → Advice → Forbidden Actions →
Required-from-User) and Addy Osmani's independently-converged six-section spec structure
(Commands/Testing/Structure/Style/Git-Workflow/three-tier Always-Ask-Never boundaries) both
land on the same shape; Cognition's own data shows performance measurably degrades when
instructions are appended after a task starts — redispatch fresh instead of patching a live
prompt. — *Source: Devin Docs "Creating Playbooks"; Addy Osmani "How to write a good spec for
AI agents"; Cognition "Devin's 2025 Performance Review."* **Verdict: HIGH.** This is a
concrete template to standardize `worker.md` rendering around (the union already needs to
reconcile jc's skills-based vs. emp's raw-shell worker.md styles — this gives a shape for
both variants to share: fixed sections, with the Forbidden-Actions tier replacing the
mixed "invariants block" pattern each fork improvised independently).

**H. Every worker prompt must name its own self-verification command, and the orchestrator
must independently re-run it rather than trust the worker's claim.** This is the CI-parity
fix for "worker says tests pass but reviewer/CI disagrees." — *Source: Claude Code
best-practices community writeups; converges with REVIEWER_SCOPE_SCANNER.md's own
enforcement-by-re-execution pattern already built in job-cannon.* **Verdict: HIGH** — already
partially implemented (the `require_live_scan.py` re-execution gate); generalize the pattern
explicitly as a named principle in the new repo's docs and extend it via the janitor pass
(item A).

### MEDIUM-HIGH applicability — worth designing for, not blocking the current build

**I. Doom-loop / cognitive-deadlock detection via periodic transcript review, not just a
round-count cap.** A validated failure taxonomy (342 manually-annotated SWE-bench-Verified
failures, Cohen's Kappa 0.72-0.77) found ~65% of agent failures are "flawed reasoning" —
predominantly non-progressive iteration ("cognitive deadlock"). A Passive-Review Expert that
re-reads the worker's transcript at a fixed cadence and injects a course-correcting message
resolved 22.2% of otherwise-failed cases (vs 6.5% for a stronger single-agent baseline). —
*Source: Liu et al. arXiv:2509.13941 (RQ4); reinforced by "doom-loop detection" as a named
concern in production terminal-agent harnesses (arXiv:2603.05344).* **Verdict:
MEDIUM-HIGH.** Not applicable as designed (this system dispatches opaque CLI workers with
end-of-process-only output capture — there is no live transcript to poll mid-session for
Devin `--print` workers today). Becomes directly actionable **once §7 item 25 (output
capture via `--export`/transcript harvesting) ships** — at that point, a periodic
transcript-diff check (hash/diff consecutive tool-call sequences, or literally re-read the
`--export` file at intervals) could implement this cheaply without any new agent framework.
Record as a **follow-on to the devin-shell adapter work**, not a blocker.

**J. Complementary, not interchangeable, worker architectures — route/retry across worker
type on failure.** Different agent architectures have measurably distinct failure
fingerprints (pipeline-style tools fail early at localization; agentic tools fail late via
cognitive deadlock); three architecturally distinct tools in one study each uniquely solved
20-33 issues the others missed despite near-identical aggregate rates. — *Source: Liu et al.
arXiv:2509.13941 (RQ1.1/RQ3).* **Verdict: MEDIUM.** Since this system already targets both
Devin and Claude Code workers, this is empirical justification for: on worker
failure/rejection, retry with a **different** worker architecture rather than re-running the
same one (coverage is complementary, not just noisy), and potentially route issue types by
worker-family strength once enough dispatch history accumulates to know the fingerprints for
these two specific worker types. Not implementable day-one (needs dispatch-history data
first) — flag as a design principle for the retry/escalation logic, not a concrete feature.

**K. Never use "two workers produced similar-looking fixes" as a correctness signal (the
"popularity trap").** Consensus/majority-vote among multiple candidate solutions
systematically filters out minority-correct answers and amplifies shared-but-wrong ones,
because same-distribution models converge on the same plausible mistakes; a
diversity-favoring, test-execution-verified selection strategy captures up to 95% of the
theoretical ensemble upper bound. — *Source: arXiv:2510.21513.* **Verdict: MEDIUM** — only
relevant if/when this system ever dispatches the same issue to two workers in parallel
(currently it doesn't). Record as a **guardrail for a future "parallel dispatch" feature**:
always gate on deterministic test execution first; use similarity/diversity only as a
tiebreaker signal to route to deeper review, never as a pass/fail criterion by itself.

**L. Per-worker cost/token circuit breaker, independent of the retry/backoff state
machine.** Track each worker's spend against its own historical baseline (not one fixed
global ceiling) via anomaly detection, with a kill-path that works even mid-tool-call, not
just at FSM checkpoints — distinct from "retry on failure," this is "kill something still
nominally running but burning resources abnormally." — *Source: Bernstein orchestrator
CircuitBreaker subsystem; corroborated by production cost-circuit-breaker writeups.*
**Verdict: MEDIUM-HIGH** for a single-operator system paying per-ACU/per-token — directly
actionable once worker liveness tracking (§7 item 27) exists: extend it to track
token/cost-so-far per session, not just wall-clock, and expose it in `status`.

**M. State reconciliation against the tracker (GitHub) every poll cycle, with local state
as a recovery checkpoint only, never the source of truth.** — *Source: sortie
(docs.sortie-ai.com); directly matches this system's own "GitHub labels/PRs are the durable
source of truth, .var is regenerable" design philosophy already documented (but, per §7
items 4-5 and the docs-inventory findings, not actually true in the current implementation).*
**Verdict: HIGH as a design principle, already the stated intent** — but §7 items 4 and 15
are exactly the places where the current code violates it (decisions live only in local
files, state.json can diverge from GitHub reality). Treat this less as a "new idea" and more
as confirmation that fixing §7's state-clobbering bugs *is* the fix for this principle
already being violated.

**N. Codified three-tier context ("project constitution" + domain-expert specs + on-demand
retrieval with null-result-as-signal) to reduce knowledge-deficiency failures.**
Knowledge deficiency (missing codebase-specific context) accounts for ~25% of agent failures
in the validated taxonomy. — *Source: arXiv:2602.20478; Liu et al. arXiv:2509.13941.*
**Verdict: MEDIUM** — directly maps onto the worker-prompt-generation step's existing
"embed the full issue body" pattern; the "null retrieval = documentation gap, log it" idea
is a cheap, generalizable addition if the prompt-writer ever gains a context-retrieval step
beyond the issue body itself. Not urgent for the current build; worth a design note.

### MEDIUM applicability — situational, record but don't build now

**O. Risk-scored routing at PR ingestion to decide review depth.** Compute a cheap
blast-radius signal (paths touched: auth/payments/migrations vs. docs/tests-only) and route
low-risk diffs to a lighter/faster path (or straight to auto-merge on CI-green) while routing
high-risk diffs through the full adversarial + cross-model pass. — *Source: Paddo (citing
Greptile v3); DeployHQ agentic CI/CD overview.* **Verdict: MEDIUM.** Useful lever for
controlling cross-family review cost (which already has real $ cost per invocation) — could
gate `cross_family.enabled`-equivalent decisions per-PR on a cheap risk heuristic instead of
a single repo-wide on/off flag. Worth an open design question (see §11), not yet a concrete
fix-list item since it needs a risk-signal definition first.

**P. Structured-disagreement review protocol: treat reviewer/critic convergence as a red
flag, not confirmation.** Instead of running two reviewers and merging on double-approve,
feed the second (cross-family) reviewer the first reviewer's findings and instruct it to try
to falsify each one with evidence; require the merge gate to weight *disagreement*, not just
agreement. — *Source: "Adversarial Review: Structured Disagreement" (OpenReview); Claude
Code's own Agent Teams "competing hypotheses" pattern.* **Verdict: MEDIUM** — a genuine
prompt-engineering improvement to `cross_family_review.md`, but changes the semantics of the
existing "leads not verdicts, independent veto" design (item C above) in a way that needs
explicit design-panel sign-off before implementing (do we want disagreement-seeking, or
independent-veto? They're compatible but not identical).

**Q. Hook-veto chain as the general shape for the merge gate**, instead of one monolithic
boolean `can_merge`: implement the gate as a sequence of independently-vetoing checks
(CI-green hook, review-approved hook, no-secrets hook, janitor-pass hook), each of which can
reject-with-reason and re-queue rather than the whole thing collapsing into one function. —
*Source: Anthropic Claude Code Agent Teams (`TaskCompleted` hook, exit-2-to-veto pattern).*
**Verdict: MEDIUM** — a clean refactor target for `merge_ready()`'s currently-monolithic gate
logic, consistent with the janitor-pass addition (item A); not urgent but worth doing at the
same time as item A since they compose naturally.

**R. Status-grid dashboard for supervising N concurrent workers** (session id / issue /
state / elapsed / cost-so-far on one screen, instead of tailing individual logs), plus a
lightweight file/module-overlap pre-check before dispatching a new worker onto paths another
in-flight worker already owns. — *Source: practitioner roundups (Batty, AgentsRoom); Bernstein
architecture.* **Verdict: MEDIUM** for a single-operator system currently running a handful
of concurrent workers — the existing `status` CLI command is a lighter-weight version of
this; a genuine multi-column live view only pays for itself once concurrency is high enough
that log-tailing becomes the bottleneck. Record as a UX enhancement, not a correctness fix.

### LOW applicability — noted for completeness, not recommended

**S. GitHub Copilot's initiator-cannot-self-approve branch-protection quirk and
firewall-scoped-to-Bash-tool-only egress control.** — *Source: GitHub Copilot coding agent
docs.* **Verdict: LOW-MEDIUM** as a security baseline worth replicating conceptually (the
adversarial reviewer's verdict should come from a genuinely separate credential/actor than
whatever dispatched the worker — already naturally true here since review is a separate CLI
invocation, but worth stating as an explicit invariant), but the specific GitHub-platform
mechanics (branch protection self-approval blocking) don't transfer to a single-operator
repo where the operator IS the approver by design.

**T. Kanban-card=worktree=agent 1:1 correspondence with periodic reconciliation GC.** —
*Source: vibe-kanban, conductor.build and similar GUI tools.* **Verdict: LOW-MEDIUM** — the
underlying invariant (every open worktree should have a corresponding tracked issue-state
entry, and vice versa; garbage-collect orphans left by crashes) is worth a periodic
reconciliation pass in the hub, but the GUI-specific tooling itself isn't relevant to a CLI
hub.

**U. Manager-of-managers pattern (Devin managing Devins).** — *Source: Cognition "Devin can
now manage Devins."* **Verdict: LOW** for the current scope — this system already IS a
manager-of-workers; the specific idea of persisting full worker trajectories (not just
diffs) for a coordinator to mine for prompt-improvement feedback is interesting but
speculative and depends on item 25 (output capture) shipping first. Record as a longer-term
possibility, not a near-term item.

---

## 11. Open design questions

Genuinely open questions requiring design-panel input, deduplicated across all readers:

1. **Canonical strategy doc conflict**: `fallback-analysis.md` (stay manual) and
   `realistic-path-forward.md` (build devin-shell) share the same date with opposite
   recommendations, and nothing formally records which one the org actually adopted (the
   subsequent worktree-fleet build suggests `realistic-path-forward.md` won in practice, but
   this was never written down as a decision).

2. **Does `devin --print` actually persist a session to `sessions.db`/transcripts?** This is
   the load-bearing premise for the entire SQLite-polling session-tracking design in the
   devin-shell adapter (item 23/§7). One line of the design corpus says yes ("creates
   transient session"), another says no ("no persistence"). Needs an empirical check before
   committing to the SQLite-based tracking approach vs. pivoting to exit-code+export-file
   tracking.

3. **What exactly triggers the "Error: A tool was rejected by the user" 39-byte failure**
   under `--permission-mode dangerous`? An org-enforced deny, a `.devin/` hook veto, or a CLI
   permission-handling bug in non-interactive mode? No artifact in either source repo answers
   this definitively.

4. **Where should review decisions durably live**, given the stated "GitHub is authoritative,
   `.var` is regenerable" design promise is currently false for decisions (they exist only in
   local `review-decision.json` files)? Options: keep local-only (status quo, fix the
   clobbering bugs instead), post as a structured PR comment, apply as a PR label, or use a
   real GitHub PR review. This is upstream of several §7 fixes (items 4, 15) and should be
   settled before those are implemented, not after.

5. **Is the `.devin/` hooks+skills infrastructure (push gates, session hooks, 8 skills) in
   scope for this extraction**, or does it stay host-repo-specific WIP? job-cannon's
   Devin-skills-based `worker.md` is unusable without it; empericus proves the kit runs fine
   without it (raw-shell `worker.md`). This decision determines whether the new repo ships a
   default `.devin/`/`.claude/` scaffold or documents it as a bring-your-own per-consumer
   layer.

6. **Should the new repo own worktree lifecycle** (creation, naming convention, post-merge
   pruning) as first-class code, or remain operator-managed as empericus's `delete_branch`
   comment currently prescribes ("the operator removes the worktree and prunes the branch
   after each merge")? Directly determines the scope of §7 item 29.

7. **What is the intended default dispatch ordering** — keep "newest-first" (today's implicit
   `gh issue list` ordering, formalized only by empericus's `--issues` escape hatch), or
   build first-class dependency-ordered dispatch (e.g. parsing "Depends on #N" from issue
   bodies)? The empericus wave-dispatch evidence (`.planning/2026-07-01-wave-...md`) shows
   real operational need for foundations-before-leaves ordering within hours of first live
   use.

8. **What is the cross-family reviewer's runner when Devin's CLI isn't present** (e.g., a
   pure-Claude-Code deployment with no Devin CLI installed at all)? Today it's hardcoded to
   `devin --model codex`. Should `spec_review`/cross-family remain independent of a single
   `cross_family.enabled` toggle, or become risk-scored per-PR (external-research item O)?

9. **Should worker prompts carry host-repo invariants via a per-repo template-override
   directory**, or by instructing workers to read the host `CLAUDE.md` only (job-cannon's
   lighter-weight approach)? empericus inlines specific named invariants directly into
   `worker.md`; job-cannon points at a document. This is the concrete design decision behind
   external-research item G (structured prompt template) — needs to be settled as part of
   that template design, not left as an implicit per-fork style choice.

10. **What triggers `require_contract_tests`-equivalent test subsets** in the new repo's
    fleet-readiness gate (§7 item 32)? job-cannon's example is domain-specific
    (scanner-registry paths); the new repo needs a config-driven path→test-subset mapping
    mechanism, not a hardcoded example, per the user's own no-hardcoded-lists rule — but the
    mapping's shape (glob patterns? label-driven? diff-content-driven?) is undecided.

11. **Does the new repo need explicit per-worker-type routing/retry logic** (external-research
    item J — retry failed dispatches with a *different* worker architecture rather than the
    same one), or is that premature until there's enough dispatch history from both worker
    types to know their actual failure fingerprints in this system specifically?

12. **How aggressively should `max_rework_cycles` escalate** (§7 item 13, external-research
    item D)? A hard cap of 1 structured retry (matching cited production multi-agent-SWE
    practice) vs. 2-3 (matching the dead `max_rework_cycles: 3` config value both forks
    already shipped, unenforced) — needs a decision, not just "enforce whatever number is
    already in the YAML."

13. **Is a janitor-style pre-review deterministic gate (external-research item A) additive to
    the existing review pipeline, or does it replace part of it?** E.g., does a janitor-fail
    skip packet generation entirely (cheaper) or generate the packet but pre-annotate it with
    the janitor's findings for the human/reviewer to see (more transparent, costs the same
    generation work)?

14. **What is the supported Python/OS floor for the standalone repo?** RUNBOOK names
    Python 3.12+ while job-cannon runs 3.13; both source repos are Windows-first
    (junction-handling, `cp1252` issues, PowerShell wrapper scripts) — should Linux/Mac be a
    first-class target for the extraction, or explicitly out of scope given the
    single-operator-on-Windows framing?
