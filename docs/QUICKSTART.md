# Quickstart

This walks a new consumer repo through install, config, label bootstrap, a
preflight check, and one full intake → dispatch → review → merge cycle.

## 1. Set up charlie-work as an external tool

`charlie-work` is a standalone dev tool with its own environment — it operates
*on* your target repo but is **not** a dependency of it. (An editable path dep
would break the consumer's CI: a locked `uv sync --all-extras` tries to install
`../charlie-work`, which CI runners never check out.) Clone it next to your
consumer repo and set up its environment once:

```powershell
# from the charlie-work repo
uv sync
```

Invoke it against a consumer repo by running charlie-work's own uv project with
the consumer as the `--repo` target:

```powershell
uv run --project ..\charlie-work --directory ..\your-repo charlie --repo ..\your-repo --help
```

- `--project` selects charlie-work's environment
- `--directory` sets the working directory to the consumer repo
- `--repo` is charlie-work's explicit target — config, state, prompts, and every
  `gh`/`git` call resolve against it

Most consumers wrap this in a one-line script so daily use is just
`charlie <command>`. The console script is `charlie` (with a `charlie-work`
alias), defined by `[project.scripts]` in this repo's `pyproject.toml`
(`charlie = "charlie_work.cli:main"`). The `uv run charlie …` examples below are
shorthand for that wrapped invocation.

## 2. Add a config file

Config is discovered at `<repo-root>/orchestrator.config.yaml`
(`config.find_config_path`); an explicit `--config <path>` overrides
discovery. If no file is present, the orchestrator runs on pure dataclass
defaults from `config.py` — that's a valid state, but `required_checks`
defaults to empty, which `doctor` will flag once `auto_merge.enabled` is
true.

Copy one of the two shipped profiles from this repo's `examples/` directory
into your consumer repo's root as `orchestrator.config.yaml`, then edit the
`required_checks` list to match your CI's actual job `name:` fields:

```powershell
Copy-Item ..\charlie-work\examples\orchestrator.config.devin.yaml orchestrator.config.yaml
```

Minimal example (everything not listed keeps its dataclass default):

```yaml
labels:
  ready: automated-ready

auto_merge:
  enabled: true
  required_checks:
    - Tests passed
    - Lint & Format

devin:
  adapter: manual
```

Key knobs, all overridable per-section (see `src/charlie_work/config.py`
for the full dataclass list and defaults):

| Key | Meaning |
|---|---|
| `labels.*` | The `agent:*` / `automated-ready` label strings that make up the state machine (see [ARCHITECTURE.md](ARCHITECTURE.md#label-state-machine)). |
| `dispatch.default_limit` / `branch_prefix` / `worker_template` | Wave size, branch-name prefix, which prompt template renders per-issue worker prompts (`worker.md` for Devin, `worker_claude_code.md` for Claude Code). |
| `review.max_rework_cycles` | `request_changes` cycles allowed before escalating to `agent:human-needed` instead of dispatching another rework round. |
| `auto_merge.required_checks` | CI check-run names that must be green before `ship-it` will merge. **Must match your `.github/workflows/*.yml` job `name:` fields exactly** — `doctor` verifies this. |
| `auto_merge.merge_flags` | Extra flags appended to `gh pr merge` (e.g., `["--admin"]` for protected-base merges, `["--auto"]` for merge-queue flows). Takes precedence over the legacy `admin` field. Flags must start with `--` and cannot be strategy flags (`--merge`/`--rebase`/`--squash`) or `--delete-branch` (branch deletion is handled separately). |
| `auto_merge.admin` | Legacy field for `gh pr merge --admin` (required when the base branch is protected and your gh auth has admin on the repo). Superseded by `merge_flags` but preserved for backward compatibility. |
| `runtime.prompts_dir` | Repo-local directory that overrides package prompt templates **by filename** — drop in your own `worker.md` and everything else keeps the package default. |
| `devin.adapter` | How a worker is launched: `manual` (write a session manifest for a human to paste), `command` (blocking subprocess via `devin.dispatch_command`), `devin-shell` (non-blocking headless `devin --print`), or `claude-code` (non-blocking headless `claude -p` in an isolated worktree). See [ARCHITECTURE.md](ARCHITECTURE.md#adapter-boundary). |
| `watchdog.*` | Supervisor tripwires (stall, wall-clock, loop/no-progress, cost/token budget) and restart-intensity cap. WARN-first by default — see [RUNBOOK.md](RUNBOOK.md#supervisor-worker-health--escalation). |
| `fleet.global_max_concurrent_sessions` | Cross-repo worker-count budget for `charlie fleet …` (default `0` = unlimited). |
| `notify.*` | Opt-in needs-attention sink (webhook \| desktop \| shell \| file); `enabled: false` by default. See `examples/notify.config.yaml`. |

## 3. Preflight with `doctor`

```powershell
uv run charlie doctor
```

`run_doctor` (in `doctor.py`) checks, in order: `gh` on PATH and
authenticated, config file presence, `required_checks` configured (if
`auto_merge.enabled`), each required check name matched against live
`.github/workflows/*.yml` job names, all `LabelConfig.all` labels exist on
the repo, the state file loads cleanly, the dispatch adapter is sane
(`command` adapter requires a non-empty `dispatch_command`), and the
configured worker template resolves to a real file. Exit code is non-zero
only on hard (`severity="error"`) failures — warnings don't block. Fix
everything `doctor` flags before your first real dispatch.

## 4. Bootstrap labels

One-time per repo (idempotent — re-running is safe, `gh label create` is
called with `allow_failure=True`):

```powershell
uv run charlie bootstrap-labels
```

Creates all labels from `LabelConfig.all` (`automated-ready`,
`agent:queued`, `agent:in-progress`, `agent:pr-open`, `agent:reviewing`,
`agent:needs-rework`, `agent:blocked`, `agent:done`, `agent:human-needed`,
plus `complexity:high`) with descriptions.

## 5. First cycle: intake → dispatch → review → merge

Label a real issue `automated-ready` on GitHub, then run the loop by hand
(one step at a time, so you can see each artifact) or via `bash-rats` (all
steps, one pass):

```powershell
# See what's ready, what's active, what's linked
uv run charlie roll-call --json

# Write worker-prompt.md + issue.json for every automated-ready issue
uv run charlie intake

# Select a wave (dependency-aware, most-unblocking-first with dispatch.order —
# default oldest — as tiebreaker, up to dispatch.default_limit) and write the
# session manifest / launch workers per the configured adapter
uv run charlie work --limit 3

# ...worker does its thing out-of-band (manual paste, or a launched process)
# and opens a PR that references the issue (branch name, title, or
# "Closes #<n>" in the body) ...

# Generate an adversarial review packet for that PR
uv run charlie why-charlie-hate --pr 123

# Read .var/charlie-work/prs/pr-123/review-prompt.md, do the review,
# then record a decision
uv run charlie verdict --pr 123 --decision approved --summary-file review.md

# Merge once checks + decision are green
uv run charlie ship-it --pr 123

# Or run intake + dispatch + review + conditional-merge in one pass:
uv run charlie bash-rats --limit 3
```

Every command accepts `--json` (either before or after the subcommand — the
CLI strips `--json` from `argv` before `argparse` sees it) for
machine-readable output, and `--dry-run` to suppress **mutating** `gh` calls
(the `_is_mutating` guard in `github.py`).

`--dry-run` additionally suppresses these local mutations, each of which used to
run during a "preview":

- the `fleet bash-rats` / `fleet supervise` **self-deploy**, which otherwise
  fast-forward-pulls `origin/main` into the running checkout and may `uv sync`
  its venv (issue #613). Because moving that checkout's HEAD terminates a running
  supervisor by design, an ungated preview could take the fleet down.
- the runner **scale-event cooldown** write, which gates *both* scale directions
  (issue #609), and the runner **pool-sample** write that feeds idle detection.

It does **not** suppress local state writes in general. **Worker** adapter
launches (`devin-shell` / `claude-code`) are a separate mechanism —
`AdapterSettings.dry_run`, threaded from the same flag — and are not covered by
the audit above. Treat any other state write as unsuppressed unless you have
checked it — see [WORKFLOWS.md](WORKFLOWS.md) for the exact scope.

## 6. Choose a worker adapter profile

Two profiles ship in `examples/`, matching the two first-class worker
runtimes:

| Profile | `dispatch.worker_template` | `devin.adapter` | Notes |
|---|---|---|---|
| `examples/orchestrator.config.devin.yaml` | `worker.md` | `devin-shell` | Skills-based worker loop (`/create-branch`, `/commit`, `/test`, `/preflight`, `/push`, `/create-pr`, `/complete`); no automated reviewer dispatch (`review_dispatch.enabled: false`). |
| `examples/orchestrator.config.claude-code.yaml` | `worker_claude_code.md` | `claude-code` | Direct-shell worker loop (no Devin skills, plain git/test commands in the prompt); no automated reviewer dispatch (Claude-only review). |

Both shipped profiles use a non-blocking adapter that actually launches a
worker (`devin-shell` / `claude-code`); each comments `# Fall back to
harness: manual` inline if you'd rather have the orchestrator only write the
session manifest and prompt files and paste them into a worker session (Devin
app, or a `claude` terminal in a worktree) by hand. The `command` adapter
(blocking subprocess-launch via `devin.dispatch_command`) is a fourth option —
see [ARCHITECTURE.md](ARCHITECTURE.md#adapter-boundary) and
[WORKFLOWS.md](WORKFLOWS.md) for each adapter's exact invocation shape. Confirm
the configured adapter's CLI is reachable with `charlie doctor --adapter-probe`.

## 7. Prompt templates

Package defaults live in `src/charlie_work/prompts/`
(`orchestrator.md`, `worker.md`, `worker_claude_code.md`, `review.md`,
`rework.md`). A
repo-local `runtime.prompts_dir` overrides these **by filename** — point it
at a tracked directory in your consumer repo and drop in your own
`worker.md` carrying repo-specific invariants and canonical commands; every
other template keeps the package default (`prompts.resolve_template`
searches the override dir first, falls back to the package `prompts/`
directory per-file).
