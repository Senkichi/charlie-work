# charlie-work

Deterministic GitHub-issue orchestration for AI worker fleets. One labeled
issue → one worker session → one PR → adversarial review → gated auto-merge.
Devin workers by default; Claude Code workers first-class.

Extracted from the battle-tested orchestrators that ran inside
[job-cannon](../job-cannon) and [empericus](../empericus) — this repo is the
union of both forks plus the fixes each learned separately.

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
   labeled       │         Code, one issue       packet, cross-   ├─ merge
   automated-    │         per session)          family pass)     ├─ labels
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
  metadata) and optionally a **cross-family pass** — a non-Claude model (codex
  via the Devin CLI) attacks the PR; its findings are leads, never merge gates.
- **Merge**: gated on required CI checks + a recorded `approved` decision.
  Branch deletion is remote-only and best-effort — it can never abort the
  merge/label sequence.
- **Reconcile**: `charlie mop-up` detects drift when humans act outside
  the orchestrator (a PR merged by hand leaving stale labels) — read-only by
  default, `--fix` to repair.

## Quickstart

`charlie-work` is a standalone dev tool with its own environment — it is **not**
a dependency of the repos it operates on. Clone it as a sibling of your consumer
repo and run it against that repo with `--repo`:

```powershell
# one-time: set up this tool's own environment
uv sync

# operate on a consumer repo by running charlie-work's own uv project against it:
#   --project selects charlie-work's env · --directory sets cwd · --repo is the target
uv run --project ../charlie-work --directory ../job-cannon charlie --repo ../job-cannon roll-call

# copy an example config to the CONSUMER repo's root and adjust it there
cp examples/orchestrator.config.devin.yaml ../job-cannon/orchestrator.config.yaml
```

Most consumers wrap that invocation in a one-line script, so day-to-day use is
just `charlie <command>`:

```powershell
charlie doctor              # preflight: env, labels, CI-check names, config
charlie doctor --adapter-probe   # also probe the worker CLI + surface stale sessions
charlie bootstrap-labels    # create the agent:* labels once
charlie roll-call --json    # what is ready / active / linked
charlie work --limit 3      # newest-first dispatch wave
charlie work --issues 565,570   # dependency-ordered wave
charlie why-charlie-hate --pr 123     # janitor gate → adversarial review packet
charlie verdict --pr 123 --decision approved --summary-file review.md
charlie ship-it --pr 123
charlie bash-rats --limit 3      # intake → dispatch → review → merge in one pass
charlie why-charlie-hate-spec --file docs/SPEC.md   # cross-family pass on a design doc
charlie mop-up              # detect label/state drift (--fix to repair)
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
| `charlie intake` | label eligible open issues `automated-ready` |
| `charlie work` | dispatch a newest-first wave of one-issue worker sessions |
| `charlie why-charlie-hate` | janitor gate + adversarial review packet for a PR |
| `charlie why-charlie-hate-spec` | cross-family adversarial pass on a design doc |
| `charlie verdict` | record a review decision (`approved` / `request_changes` / `blocked`) |
| `charlie ship-it` | merge a PR once it's approved and required checks are green |
| `charlie bash-rats` | run the whole cycle (intake → work → review → merge) until the queue's dry |
| `charlie mop-up` | detect (and with `--fix`, repair) label/state drift |
| `charlie doctor` | preflight diagnostics (env, labels, CI-check names, config, adapter) |
| `charlie bootstrap-labels` | create the nine `agent:*` / `automated-ready` labels once |

## Configuration

Config is discovered at `<repo-root>/orchestrator.config.yaml` (override with
`--config`). Absent file → dataclass defaults. See [examples/](examples/) for
the two shipped profiles:

| Profile | Worker runtime | Notes |
|---|---|---|
| `orchestrator.config.devin.yaml` | Devin sessions | skills-based worker loop, cross-family review on |
| `orchestrator.config.claude-code.yaml` | Claude Code | direct-shell worker loop, Claude-only review |

Key knobs: `labels.*` (state-machine label names), `dispatch.default_limit` /
`branch_prefix` / `worker_template`, `review.max_rework_cycles` (past this many
`request_changes` cycles a PR escalates to `agent:human-needed`),
`auto_merge.required_checks` (verify with `doctor`), `runtime.prompts_dir`
(repo-local template overrides), `devin.adapter` (`manual` | `command` |
`devin-shell` | `claude-code`), `claude_code.*` (worktree/venv settings for the
claude-code adapter), `cross_family.*` (non-Claude adversarial pass).

**This repo's own CI check names** (for `auto_merge.required_checks`): `Tests (ubuntu-latest)`,
`Tests (windows-latest)`, and `Lint`. These correspond to the job `name:` fields in
`.github/workflows/ci.yml` and are verified by `charlie doctor`.

**Worker adapters** (`devin.adapter`): `manual` writes a session manifest for
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

Unioned from two production forks (June–July 2026): job-cannon contributed
cross-family review and `why-charlie-hate-spec`; empericus contributed `--issues` wave
dispatch, the `gh pr merge` stdout fix, and the worktree/branch-deletion
failure report that drove the decoupled merge sequence.
