# Quickstart

This walks a new consumer repo through install, config, label bootstrap, a
preflight check, and one full intake → dispatch → review → merge cycle.

## 1. Install as an editable path dependency

`devin-orchestrator` is consumed as a library + CLI from your target repo,
not installed standalone. In your consumer repo's `pyproject.toml`:

```toml
[tool.uv.sources]
devin-orchestrator = { path = "../devin-orchestrator", editable = true }

[project]
dependencies = [
    "devin-orchestrator",
    # ...your other deps
]
```

Adjust the relative path to wherever you've cloned this repo. Then, from the
consumer repo:

```powershell
uv sync
```

This installs the `devin-orch` console script (`[project.scripts]` in this
repo's `pyproject.toml`: `devin-orch = "devin_orchestrator.cli:main"`).
Verify it resolved:

```powershell
uv run devin-orch --help
```

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
Copy-Item ..\devin-orchestrator\examples\orchestrator.config.devin.yaml orchestrator.config.yaml
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

Key knobs, all overridable per-section (see `src/devin_orchestrator/config.py`
for the full dataclass list and defaults):

| Key | Meaning |
|---|---|
| `labels.*` | The nine `agent:*` / `automated-ready` label strings that make up the state machine (see [ARCHITECTURE.md](ARCHITECTURE.md#label-state-machine)). |
| `dispatch.default_limit` / `branch_prefix` / `worker_template` | Wave size, branch-name prefix, which prompt template renders per-issue worker prompts (`worker.md` for Devin, `worker_claude_code.md` for Claude Code). |
| `review.max_rework_cycles` | `request_changes` cycles allowed before escalating to `agent:human-needed` instead of dispatching another rework round. |
| `auto_merge.required_checks` | CI check-run names that must be green before `merge-ready` will merge. **Must match your `.github/workflows/*.yml` job `name:` fields exactly** — `doctor` verifies this. |
| `runtime.prompts_dir` | Repo-local directory that overrides package prompt templates **by filename** — drop in your own `worker.md` and everything else keeps the package default. |
| `devin.adapter` | `manual` (write a session manifest for a human to paste into a Devin session) or `command` (subprocess-launch via `devin.dispatch_command`). |
| `cross_family.*` | Enables the non-Claude adversarial pass (`enabled: false` by default; both example profiles show how to turn it on/off). |

## 3. Preflight with `doctor`

```powershell
uv run devin-orch doctor
```

`run_doctor` (in `doctor.py`) checks, in order: `gh` on PATH and
authenticated, config file presence, `required_checks` configured (if
`auto_merge.enabled`), each required check name matched against live
`.github/workflows/*.yml` job names, all nine orchestration labels exist on
the repo, the state file loads cleanly, the dispatch adapter is sane
(`command` adapter requires a non-empty `dispatch_command`), the
cross-family binary is on PATH (if `cross_family.enabled`), and the
configured worker template resolves to a real file. Exit code is non-zero
only on hard (`severity="error"`) failures — warnings don't block. Fix
everything `doctor` flags before your first real dispatch.

## 4. Bootstrap labels

One-time per repo (idempotent — re-running is safe, `gh label create` is
called with `allow_failure=True`):

```powershell
uv run devin-orch bootstrap-labels
```

Creates all nine labels from `LabelConfig.all` (`automated-ready`,
`agent:queued`, `agent:in-progress`, `agent:pr-open`, `agent:reviewing`,
`agent:needs-rework`, `agent:blocked`, `agent:done`, `agent:human-needed`)
with descriptions.

## 5. First cycle: intake → dispatch → review → merge

Label a real issue `automated-ready` on GitHub, then run the loop by hand
(one step at a time, so you can see each artifact) or via `loop` (all
steps, one pass):

```powershell
# See what's ready, what's active, what's linked
uv run devin-orch status --json

# Write worker-prompt.md + issue.json for every automated-ready issue
uv run devin-orch intake

# Select a wave (newest-first, up to dispatch.default_limit) and write the
# session manifest / launch workers per the configured adapter
uv run devin-orch dispatch --limit 3

# ...worker does its thing out-of-band (manual paste, or a launched process)
# and opens a PR that references the issue (branch name, title, or
# "Closes #<n>" in the body) ...

# Generate an adversarial review packet for that PR
uv run devin-orch review --pr 123

# Read .var/devin-orchestrator/prs/pr-123/review-prompt.md, do the review,
# then record a decision
uv run devin-orch record-review --pr 123 --decision approved --summary-file review.md

# Merge once checks + decision are green
uv run devin-orch merge-ready --pr 123

# Or run intake + dispatch + review + conditional-merge in one pass:
uv run devin-orch loop --limit 3
```

Every command accepts `--json` (either before or after the subcommand — the
CLI strips `--json` from `argv` before `argparse` sees it) for
machine-readable output, and `--dry-run` to suppress **mutating** `gh` calls
(the `_is_mutating` guard in `github.py`; note this does not suppress local
state writes or adapter/model subprocess calls — see
[WORKFLOWS.md](WORKFLOWS.md) for the exact scope).

## 6. Choose a worker adapter profile

Two profiles ship in `examples/`, matching the two first-class worker
runtimes:

| Profile | `dispatch.worker_template` | `devin.adapter` | Notes |
|---|---|---|---|
| `examples/orchestrator.config.devin.yaml` | `worker.md` | `manual` (default) | Skills-based worker loop (`/create-branch`, `/commit`, `/test`, `/preflight`, `/push`, `/create-pr`, `/complete`); cross-family review **on**. |
| `examples/orchestrator.config.claude-code.yaml` | `worker_claude_code.md` | `manual` | Direct-shell worker loop (no Devin skills, plain git/test commands in the prompt); cross-family review **off** (Claude-only review). |

`manual` is the operator-confirmed default for both profiles today: the
orchestrator writes the session manifest and prompt files, and a human opens
the worker session (Devin app, or a `claude` terminal in a worktree) by
hand. The `command` adapter (subprocess-launch via
`devin.dispatch_command`) and the in-flight non-blocking `devin_shell`/
`claude_code` adapters are alternatives — see
[ARCHITECTURE.md](ARCHITECTURE.md#adapter-boundary) and
[WORKFLOWS.md](WORKFLOWS.md) for their exact invocation shape and current
integration status.

## 7. Prompt templates

Package defaults live in `src/devin_orchestrator/prompts/`
(`orchestrator.md`, `worker.md`, `worker_claude_code.md`, `review.md`,
`rework.md`, `cross_family_review.md`, `cross_family_spec_review.md`). A
repo-local `runtime.prompts_dir` overrides these **by filename** — point it
at a tracked directory in your consumer repo and drop in your own
`worker.md` carrying repo-specific invariants and canonical commands; every
other template keeps the package default (`prompts.resolve_template`
searches the override dir first, falls back to the package `prompts/`
directory per-file).
