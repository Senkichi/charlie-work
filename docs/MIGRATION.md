# Migration: moving job-cannon / empericus onto this package

This repo is the extraction target for the orchestrators that previously
lived in-tree in job-cannon and empericus (see this repo's README
"Provenance" section and `docs/design/extraction-dossier.md` for the full
fork-comparison history). This doc is the cutover procedure for each
consumer repo.

`charlie-work` is consumed as an **external dev tool**, not a dependency —
see step 2 for why. The steps below are ordered; run them per consumer repo.

## 1. Capture a rollback point

Before touching either consumer repo, record the current commit so the
cutover is trivially reversible:

```powershell
$pre_fix_hash = git rev-parse HEAD
```

Do this **in each consumer repo separately** (job-cannon and empericus each
get their own hash) immediately before step 3 below.

## 2. Wire it as an external tool (not a dependency)

`charlie-work` operates *on* your repo (dispatch, review, merge via `gh`); it
is not a runtime or test dependency of your app. It runs from its **own**
environment, so nothing changes in the consumer's `pyproject.toml`, lockfile,
or CI dependency graph.

Clone `charlie-work` as a sibling of the consumer repo, then add a thin
wrapper that runs charlie-work's uv project against this repo:

```powershell
# scripts/charlie.ps1  (any name; the operator invokes it directly)
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$orchRoot = Join-Path (Split-Path -Parent $repoRoot) "charlie-work"
if (-not (Test-Path $orchRoot)) {
    throw "charlie-work not found at '$orchRoot' (expected as a sibling of this repo). " +
        "Clone https://github.com/Senkichi/charlie-work there, or edit this path."
}
uv run --project $orchRoot --directory $repoRoot charlie --repo $repoRoot @Arguments
```

- `--project` selects charlie-work's environment (its deps, not yours)
- `--directory` sets cwd to this repo (for any cwd-relative git/gh logic)
- `--repo` is charlie-work's explicit target — config, state, prompts, and
  every `gh`/`git` call resolve against it

Set up the tool's environment once, from the charlie-work repo:

```powershell
uv sync
```

> **Why not an editable path dependency?** An earlier plan added
> `charlie-work = { path = "../charlie-work", editable = true }` to the
> consumer's `[tool.uv.sources]`. It **breaks CI**: a locked
> `uv sync --all-extras` force-installs every optional dep including the
> path-sourced tool, but `../charlie-work` is never checked out on the CI
> runner. The external-tool wrapper keeps CI's dependency graph and lockfile
> untouched (idiomatic, like `uvx ruff`).

## 3. Delete the in-repo automation/ copy

Once the wrapper resolves, the consumer repo's own copy of the orchestrator
becomes dead code — delete it rather than leaving two implementations to
drift:

- job-cannon: the in-tree `automation/devin_orchestrator/` package (confirm
  the exact path in that repo; the dossier refers to it as
  `automation.devin_orchestrator`) and its `.devin/` hooks/skills
  scaffolding, **if** you are not also relying on the rules/skills/hooks
  three-layer determinism stack described in the dossier's §6
  "Extensibility items" — if you are, that scaffolding is orthogonal to the
  orchestrator package and can stay.
- empericus: the equivalent in-tree orchestrator module.
- Both: any wrapper scripts (`setup_worker.sh` / `finish_worker.sh` or
  similar) whose logic has been ported into this package's `worktree.py`
  (worktree lifecycle) or `adapters.py` (dispatch). Verify the ported
  behavior covers your script's exact semantics before deleting the script —
  see the module map in [ARCHITECTURE.md](ARCHITECTURE.md#module-map).
- Any repo-specific `orchestrator.py` / `__main__.py` entry point that
  shelled out to the old in-tree package — replace call sites with your
  `charlie <command>` wrapper invocation, or drop them if scripts can call
  the wrapper directly.

Do **not** delete the old package until step 6 (consumer test suite green)
confirms the new one is a working replacement.

## 4. Add repo-local config and prompt overrides

Copy the profile matching your worker runtime into the **consumer** repo root
as `orchestrator.config.yaml`:

```powershell
# job-cannon (Devin workers, skills-based loop, cross-family on)
Copy-Item ..\charlie-work\examples\orchestrator.config.devin.yaml orchestrator.config.yaml

# empericus (Claude Code workers, direct-shell loop, Claude-only review)
Copy-Item ..\charlie-work\examples\orchestrator.config.claude-code.yaml orchestrator.config.yaml
```

Then port anything repo-specific from the old in-tree config that isn't in
the shipped example:

- `auto_merge.required_checks` — **must match your `.github/workflows/*.yml`
  job `name:` fields exactly**; `charlie doctor` verifies this for you after
  the cutover (see step 6).
- Any repo-specific label overrides under `labels:` (unlikely — both forks
  used the same nine default names, but confirm).
- If your repo's worker prompt carries repo-specific invariants or canonical
  commands beyond what `worker.md` / `worker_claude_code.md` cover, set
  `runtime.prompts_dir` to a tracked directory (e.g. `.devin/prompts/`) and
  drop your customized template in there under the same filename — it
  overrides the package default for that file only; every other template
  stays default (`prompts.resolve_template` searches your override dir first,
  per filename).

## 5. Preserve existing `.var` state — pin it, and it loads as-is

The state schema is versioned (`state.STATE_VERSION = 1`) and explicitly
**backward compatible with pre-extraction state**: `load_state()` uses
`.setdefault()` to fill any missing top-level keys (`version`,
`generated_at`, `issues`, `prs`, `events`) rather than requiring an exact
shape. You do not need to migrate or transform the old `state.json`.

**But you must pin the state directory.** charlie-work's *package default* is
now `.var/charlie-work`, whereas both forks wrote to `.var/devin-orchestrator`.
The state file also stores **absolute artifact paths** (per-issue worker
prompts, per-PR review packets/decisions) that point into that directory, so
renaming it would orphan every one of them. Keep your existing state by
pinning `runtime.state_dir` to the directory the old package used:

```yaml
runtime:
  state_dir: .var/devin-orchestrator   # keep pointing at existing live state
```

(A brand-new consumer with no prior state can simply take the
`.var/charlie-work` default.)

Verify after cutover:

```powershell
.\scripts\charlie.ps1 roll-call --json
```

If issue/PR counts look sane relative to what you expected before the
cutover, the state carried over correctly. If the old state file is malformed
or was already corrupt, `load_state()` quarantines it exactly as it would
post-cutover (see
[RUNBOOK.md](RUNBOOK.md#corrupt-state-quarantine-recovery)) — not a
migration-specific risk, just the same safety net that always applies.

## 6. Label parity check via `doctor`

```powershell
.\scripts\charlie.ps1 doctor
```

This is the single most important post-cutover check — it confirms the nine
`agent:*` / `automated-ready` labels the package expects already exist on the
repo (they do, since both forks used the same label set by default; `doctor`
flags any mismatch instead of letting it fail silently at the first `work`
dispatch), and separately verifies every configured `required_checks` name
actually matches a job in your live `.github/workflows/*.yml` files — the
exact class of bug the dossier flags as a "silent permanent-block trap" when
check names drift from CI job names.

If `doctor` reports missing labels (shouldn't happen on a straight migration,
but possible if your repo customized `labels:` in the old config and you
didn't port that customization in step 4):

```powershell
.\scripts\charlie.ps1 bootstrap-labels
```

This is additive/idempotent (`gh label create` with `allow_failure=True`) —
safe to run even if most labels already exist.

## 7. Remove old in-tree orchestrator tests, then run the suite

Under the external-tool model the consumer no longer imports the orchestrator
at all, so any consumer tests that referenced the old in-tree module directly
(import paths, monkeypatched dotted strings like
`automation.devin_orchestrator.workflow.*` — the dossier flags this exact
fragility in both forks' suites) are **deleted** along with the package in
step 3. That coverage now lives in charlie-work's own suite. Then run the
consumer suite to confirm nothing else depended on the deleted module:

```powershell
uv run --active pytest -q --tb=short
```

The suite should be green before you consider the cutover complete. If it
fails in a way that traces back to the old in-tree module actually being gone
(rather than a genuine regression), that's a signal you deleted something in
step 3 the suite still depended on directly — restore it from
`git show $pre_fix_hash:<path>` and re-evaluate.

## Rollback

If the cutover doesn't work out cleanly:

```powershell
git reset --hard $pre_fix_hash
```

This is safe precisely because step 1 captured the pre-cutover commit before
any later step touched the working tree. Since `.var/` state is untouched by
the wrapper swap itself (step 5 pins it in place), rolling back the code does
not lose any orchestration history.
