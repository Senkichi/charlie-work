# Migration: moving job-cannon / empericus onto this package

This repo is the extraction target for the orchestrators that previously
lived in-tree in job-cannon and empericus (see this repo's README
"Provenance" section and `docs/design/extraction-dossier.md` for the full
fork-comparison history). This doc is the cutover procedure for each
consumer repo.

## 1. Capture a rollback point

Before touching either consumer repo, record the current commit so the
cutover is trivially reversible:

```powershell
$pre_fix_hash = git rev-parse HEAD
```

Do this **in each consumer repo separately** (job-cannon and empericus each
get their own hash) immediately before step 3 below.

## 2. Add the editable path dependency

In the consumer repo's `pyproject.toml`:

```toml
[tool.uv.sources]
devin-orchestrator = { path = "../devin-orchestrator", editable = true }

[project]
dependencies = [
    "devin-orchestrator",
    # ...existing deps
]
```

The relative path assumes the standard sibling-repo layout
(`C:/Users/senki/repos/devin-orchestrator` next to
`C:/Users/senki/repos/job-cannon` / `.../empericus`); adjust if your layout
differs. Then:

```powershell
uv sync
uv run devin-orch --help    # confirm the console script resolved
```

## 3. Delete the in-repo automation/ copy

Once the editable dependency resolves, the consumer repo's own copy of the
orchestrator becomes dead code — delete it rather than leaving two
implementations to drift:

- job-cannon: the in-tree `automation/devin_orchestrator/` package (or
  equivalent — confirm the exact path in that repo; the dossier refers to it
  as `automation.devin_orchestrator`) and its `.devin/` hooks/skills
  scaffolding, **if** you are not also relying on the rules/skills/hooks
  three-layer determinism stack described in the dossier's §6
  "Extensibility items" — if you are, that scaffolding is orthogonal to the
  orchestrator package and can stay.
- empericus: the equivalent in-tree orchestrator module.
- Both: any wrapper scripts (`setup_worker.sh` / `finish_worker.sh` or
  similar) whose logic has been ported into this package's
  `worktree.py` (worktree lifecycle) or `adapters.py` (dispatch). Verify the
  ported behavior covers your script's exact semantics before deleting the
  script — see the module map in
  [ARCHITECTURE.md](ARCHITECTURE.md#module-map).
- Any repo-specific `orchestrator.py`/`__main__.py` entry point that shelled
  out to the old in-tree package — replace call sites with `devin-orch
  <command>` (or `uv run devin-orch <command>`) invocations, or drop them if
  CI/scripts can call the console script directly.

Do **not** delete the old package until step 6 (both consumer test suites
green) confirms the new one is a working replacement.

## 4. Add repo-local config and prompt overrides

Copy the profile matching your worker runtime as the new
`orchestrator.config.yaml`:

```powershell
# job-cannon (Devin workers, skills-based loop, cross-family on)
Copy-Item ..\devin-orchestrator\examples\orchestrator.config.devin.yaml orchestrator.config.yaml

# empericus (Claude Code workers, direct-shell loop, Claude-only review)
Copy-Item ..\devin-orchestrator\examples\orchestrator.config.claude-code.yaml orchestrator.config.yaml
```

Then port over anything repo-specific from the old in-tree config that
isn't in the shipped example:

- `auto_merge.required_checks` — **must match your `.github/workflows/*.yml`
  job `name:` fields exactly**; `devin-orch doctor` verifies this for you
  after the cutover (see step 5).
- Any repo-specific label name overrides under `labels:` (unlikely — both
  forks used the same nine default names, but confirm).
- If your repo's worker prompt carries repo-specific invariants or canonical
  commands beyond what `worker.md` / `worker_claude_code.md` cover, set
  `runtime.prompts_dir` to a tracked directory (e.g. `orchestrator-prompts/`)
  and drop your customized template in there under the same filename — it
  overrides the package default for that file only, every other template
  stays default (`prompts.resolve_template` searches your override dir
  first, per filename).

## 5. Preserve existing `.var` state — it loads as-is

The state schema is versioned (`state.STATE_VERSION = 1`) and explicitly
**backward compatible with pre-extraction state** — the README says so and
`load_state()`'s implementation confirms it: `.setdefault()` calls fill in
any missing top-level keys (`version`, `generated_at`, `issues`, `prs`,
`events`) rather than requiring an exact shape match. You do not need to
migrate or transform the old `state.json` — just make sure
`runtime.state_dir` in the new config points at the **same** directory the
old package used (default `.var/devin-orchestrator` in both forks, per the
dossier's schema section), and the new package will pick up the existing
file on its next `load_state()` call.

If the old state file is malformed or was already corrupt before the
cutover, `load_state()` quarantines it exactly as it would post-cutover (see
[RUNBOOK.md](RUNBOOK.md#corrupt-state-quarantine-recovery)) — this is not a
migration-specific risk, it's the same safety net that always applies.

Verify after cutover:

```powershell
uv run devin-orch status --json
```

If issue/PR counts look sane relative to what you expected before the
cutover, the state carried over correctly.

## 6. Label parity check via `doctor`

```powershell
uv run devin-orch doctor
```

This is the single most important post-cutover check — it confirms the
nine `agent:*`/`automated-ready` labels the new package expects already
exist on the repo (they do, since both forks used the same label set by
default; `doctor` flags any mismatch instead of letting it fail silently
at the first `dispatch`), and separately verifies every configured
`required_checks` name actually matches a job in your live
`.github/workflows/*.yml` files — the exact class of bug the dossier flags
as a "silent permanent-block trap" when check names drift from CI job
names.

If `doctor` reports missing labels (shouldn't happen on a straight
migration, but possible if your repo customized `labels:` in the old
config and you didn't port that customization in step 4):

```powershell
uv run devin-orch bootstrap-labels
```

This is additive/idempotent (`gh label create` with `allow_failure=True`) —
safe to run even if most labels already exist.

## 7. Run both consumer test suites after the cutover

If the consumer repo's own test suite references the old in-tree
orchestrator module directly (import paths, monkeypatched dotted strings —
the dossier flags this exact fragility pattern in both forks' test suites),
update those imports to `devin_orchestrator.<module>` before running:

```powershell
# in job-cannon
uv run --active pytest -q --tb=short

# in empericus
uv run --active pytest -q --tb=short
```

Both suites should be green before you delete anything from step 3 for
good, or before you consider the cutover complete. If either suite fails
in a way that traces back to the old in-tree module actually being gone
(rather than a genuine regression in the new package), that's a signal you
deleted something in step 3 the test suite still depended on directly —
restore it from `git show $pre_fix_hash:<path>` and re-evaluate.

## Rollback

If the cutover doesn't work out cleanly:

```powershell
git reset --hard $pre_fix_hash
```

This is safe precisely because step 1 captured the pre-cutover commit
before any of steps 2-4 touched the working tree. Since `.var/` state is
untouched by the package swap itself (step 5 confirms it loads as-is either
way), rolling back the code does not lose any orchestration history.
