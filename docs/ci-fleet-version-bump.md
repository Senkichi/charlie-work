# ci-fleet version-bump workflow

`ci-fleet` (imported as `ci_fleet`) is consumed from PyPI as a versioned wheel
— since 2026-08-28 it is no longer an editable path dependency on a
`../ci_runners` sibling checkout. Under the old arrangement a sibling edit was
live in this repo's venv immediately via an editable `.pth`; under the new one,
what runs is whatever wheel the last `uv sync` installed into site-packages,
and it only changes when the lockfile and a sync say so. This doc covers the
three places that difference bites: consuming a release here, the fleet daemon
checkout, and local cross-repo development.

## Consuming a new ci-fleet release

```powershell
# 1. Raise the floor on the `ci-fleet` entry in pyproject.toml's
#    [project.dependencies] ONLY if the new release's changes require it —
#    e.g. charlie_work starts calling an API that only exists in the new
#    version. If the existing constraint already admits the release, skip
#    this and let the lockfile carry the bump.
#
# 2. Re-resolve ci-fleet. Plain `uv lock` prefers versions already in
#    uv.lock, so --upgrade-package is what actually picks up the new release
#    when the constraint admits both old and new:
uv lock --upgrade-package ci-fleet

# 3. Install the locked version into .venv (dev extras included):
uv sync --all-extras

# 4. Verify what actually landed:
uv run python -c "import importlib.metadata; print(importlib.metadata.version('ci-fleet'))"
```

Commit `pyproject.toml` and `uv.lock` together: CI's Lint job runs
`uv lock --check` as a lock-currency gate, so a constraint bump without a
relock is a red check. If the constraint already admitted the release and only
`uv.lock` moved, commit the lockfile alone.

## The daemon-checkout gotcha

The fleet daemon (`charlie fleet supervise-loop`, launched by the scheduled
task through `scripts/fleet-pass.ps1`) runs from a dedicated checkout whose
`.venv` has charlie-work installed **editable**: a `.pth` file in
site-packages puts `<checkout>/src` on `sys.path`, so a `git pull` alone makes
new `charlie_work` source live on the next supervisor restart. No `uv sync` is
needed for code-only changes — that is exactly what the editable install is
for.

The `.pth` covers *only* charlie-work's own source. `ci_fleet` is a regular
wheel in site-packages: its version is whatever the last `uv sync` resolved
and installed, and no file the `.pth` points at describes it. `self_deploy`
knows this — when *its own* `git pull` touches `pyproject.toml` or `uv.lock`
it runs `uv sync` (deferring via a pending-sync marker while workers are
live, then retrying on later passes).

The gap is any HEAD move `self_deploy` did not perform itself: a manual
`git pull` in the daemon checkout, a cherry-pick, a branch switch. The next `self_deploy` pass diffs only what its own pull fetched —
with the bump already landed it reports "already up to date" and never syncs.
The `.pth` then makes the deploy *look* complete — new orchestrator code runs
after restart — while `ci_fleet` silently stays at the old wheel. No startup
check catches the skew: the anchors verify *where* `ci_fleet` loads from
(`ci_fleet_anchor` records `__file__`; `venv_anchor` checks `.pth` targets),
not *which version* is installed.

So: after any out-of-band update that touched `pyproject.toml`/`uv.lock` in
the daemon checkout, run `uv sync` there *before* restarting the supervisor —
the launch line runs `uv run --no-sync`, so nothing at daemon startup
reconciles the venv for you — and verify with the daemon's own interpreter:

```powershell
uv sync
.venv\Scripts\python.exe -c "import importlib.metadata; print(importlib.metadata.version('ci-fleet'))"
```

Bare `uv sync`, not `--all-extras` — the same command `self_deploy` runs; the
daemon does not need the dev extra.

One related window, self-healing and not a manual case: when `self_deploy`'s
pull lands a dep bump while workers are live, the sync is deferred and the
supervisor still restarts on the moved HEAD — new code against old deps until
the marker retries once workers drain. A stale `ci-fleet` version observed
right after such a deploy is expected; a stale version that persists after
workers drain is the deferred-sync path failing and worth checking
`events.db` for `self_deploy_failed`.

## Cross-repo dev loop (testing unreleased ci_fleet changes)

To run charlie-work against a `ci_runners` working tree without publishing,
temporarily restore the pre-PyPI source override in `pyproject.toml`:

```toml
# temporary — DO NOT COMMIT
[tool.uv.sources]
ci-fleet = { path = "../ci_runners", editable = true }
```

```powershell
uv lock
uv sync --all-extras
```

`editable = true` writes a `.pth` at `../ci_runners/src`, so sibling edits are
live on the next import — the arrangement this repo ran on before the PyPI
switch.

Use the declared override rather than `uv pip install -e ../ci_runners`. Both
anchors read `[tool.uv.sources]`: the declared entry is what makes the sibling
a *legal* `.pth` target for `venv_anchor` (an undeclared editable is flagged
as escaping the interpreter's checkout and the supervisor refuses to start),
and it re-activates `ci_fleet_anchor`'s provenance check against the sibling.
A `uv pip install -e` shortcut also gets silently reverted by the next
`uv sync`, which syncs the venv back to the lockfile.

Worktree note: the path resolves relative to the *worktree* root, so inside
`.var/charlie-work/worktrees/<name>/` the relative `../ci_runners` names a
sibling of the worktree — a directory that does not exist — not the sibling
of the main checkout. Use an absolute path from a worktree, or run the dev
loop from a checkout that is a real sibling of `ci_runners`. (The same
relative-resolution is why `ci_fleet_anchor` abstains from worktrees rather
than blocking the fleet.)

Before committing, remove the override and re-run `uv lock` + `uv sync` to
restore the PyPI resolution. Both `pyproject.toml` and `uv.lock` record the
override — a committed path source fails CI outright, because `../ci_runners`
does not exist on the runners and `uv sync` cannot resolve it. `git diff` both
files before pushing.

The usual worktree-venv isolation rules apply unchanged while the override is
in play — including why `uv run --active` is unsafe here. See the per-worktree
venv sections under
[Local host saturation ceiling](RUNBOOK.md#local-host-saturation-ceiling-claude-code-adapter)
and
[Shared-venv isolation for devin-shell](RUNBOOK.md#shared-venv-isolation-for-devin-shell)
in RUNBOOK.md rather than a restatement here.
