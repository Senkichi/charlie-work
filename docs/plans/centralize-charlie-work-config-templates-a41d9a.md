# Centralize charlie-work Configuration, Modules, and Templates (Round 3)

This revised plan consolidates charlie-work's scattered configuration, state, and template pathways into a single canonical location per concern, incorporating all findings from the round-2 adversarial review (every claim verified against live code 2026-07-29).

---

## Background — the current split (corrected)

charlie-work is an **external dev tool** that operates *on* consumer repos. It ships canonical prompt templates in `src/charlie_work/prompts/` (resolved via `prompts.resolve_template` — repo-local `prompts_dir` shadows the package defaults, first-hit-wins by filename) and shared section partials in `src/charlie_work/prompts/worker_sections/` (resolved via `prompt_sections.section_variables`).

### Path inventory — every config key AND hardcoded path that controls where data lives

Overriding `runtime.state_dir` relocates *some* data and not the rest. The full inventory:

| Data | Source | Default | job-cannon override |
|---|---|---|---|
| state.json, events.db, dispatches/, issues/, prs/, logs/ | `runtime.state_dir` | `.var/charlie-work` | `.var/devin-orchestrator` |
| worktrees/ | `claude_code.worktrees_dir` (`None` → `worktree.py` default) | `<repo_root>/.var/charlie-work/worktrees` | **not overridden** → `.var/charlie-work/worktrees` |
| review sidecars | `review_dispatch.reviews_dir` | `.var/charlie-work/dispatches/reviews` | `.var/devin-orchestrator/dispatches/reviews` (inert: `review_dispatch.enabled: false`) |
| spec cross-family reviews | **hardcoded** `workflow.py:10250`: `self.paths.root / "cross-family"` | follows `state_dir` | `.var/devin-orchestrator/cross-family/` (3 live artifacts) |
| devin sessions | `devin.session_manifest` / `session_results` / `sessions_dir` | `.var/charlie-work/dispatches/...` | `.var/devin-orchestrator/dispatches/...` |
| notify digest | `notify.file_path` | `.var/charlie-work/notify/digest.jsonl` | `.var/devin-orchestrator/notify/digest.jsonl` |
| worktrees (for `worktree-clean`) | **hardcoded** `cli.py:344`: `paths.root / "worktrees"` | follows `state_dir` | `.var/devin-orchestrator/worktrees` (does not exist — command is a silent no-op) |

**Critical consequence**: `.var/charlie-work/` in job-cannon is **not stale**. It contains **74 live `agent-issue-*` worktrees** plus an empty `dispatches/reviews/`. The 0-byte `events.db` and stray finding docs are the *only* inert things in it. `.var/devin-orchestrator/` has **no `worktrees/` directory** — worktrees were never redirected by the `state_dir` override because `worktrees_dir` defaults independently.

The job-cannon consumer repo has **three overlapping locations**:

1. **`.var/devin-orchestrator/`** — the *active* state dir for state.json/events.db/dispatches/prs/issues/notify/cross-family. Holds the live `state.json` (~1 MB, 517 embedded legacy paths), `events.db` (3.2 MB + 4.1 MB WAL), the launch shim, and stale design docs. **No worktrees here.**
2. **`.var/charlie-work/`** — **live worktrees** (74 dirs) + empty `dispatches/reviews/` + stray docs + 0-byte `events.db`. NOT stale; NOT deletable.
3. **`.devin/`** — consumer-local Devin-CLI infrastructure. Only **3 files are git-tracked** (`hooks/require_platform_contract.py`, `prompts/rework.md`, `prompts/worker.md`); 12+ files are untracked. Materialized into every worktree by the launch shim's `cp -r` (NOT by `materialize_dirs` — see §1.2 correction below).

### Materialization: the real two-writer conflict (corrected from round 2)

**job-cannon does NOT set `materialize_dirs: [".devin"]`.** It sets `dispatch.injected_paths: [.devin]`. These are different keys with different semantics:

- `injected_paths` — paths excluded from the `_worker_authored_dirty` probe (`worktree.py:912-913`). Never copied.
- `materialize_dirs` — paths copied into the worktree by `_materialize_directory` + shielded via `git update-index --assume-unchanged` (`worktree.py:1392+`).

The `_materialize_directory` `assume-unchanged` shielding has **never** applied to `.devin` in job-cannon. The real second writer is the launch shim's `rm -rf .devin && cp -r "$MAIN/.devin" .devin` (`launch_devin_worker.sh:38`). The migration's real shape is: **the shim's `cp -r` is the incumbent writer; the package materializer is the challenger.**

### Prompt structural difference

The package `worker.md` is a **composition root** referencing 8 `$section_*` partials (renders to ~8.7 KB). jc's override is a **flat file** referencing **zero** `$section_*` placeholders (renders to ~9.7 KB). Removing jc's override begins injecting 8 new sections into every job-cannon worker prompt **for the first time** — a large behavioral change requiring explicit review.

### Packaging: assets already ship (corrected from round 2)

Hatchling is directory-based, not manifest-based. `packages = ["src/charlie_work"]` includes the whole package tree. Built wheel verified: all 16 non-`.py` assets (8 prompts + 8 worker_sections) already ship. **No `package-data` or `force-include` is needed.** (`package-data` is a setuptools key that hatchling silently ignores — adding it would be a no-op.)

### Fleet registry: self-heals on next command

`fleet_registry.py:137`: `touch_repo` writes `"state_dir": str(paths.root)` on every call, and runs from `build_app` (`cli.py:302`) and `run_doctor_command` (`cli.py:314`). The registry self-heals on the next command after the config change. Manual `fleet.json` editing is redundant (belt-and-braces only).

### Watchdog interaction

job-cannon's config sets `watchdog.enabled: false` because the bash.exe-wrapping shim leaves the sidecar log never growing past the shim marker. Phase 4 rewrites that shim — either preserve the behavior (watchdog stays off) or change it (re-enabling is an explicit decision with validation).

---

## Goals — what success looks like

1. **One state location per consumer**: every consumer uses `.var/charlie-work/` for **all** path defaults. The legacy `devin-orchestrator` name is retired. The fleet registry self-heals on next invocation.
2. **One template source**: `src/charlie_work/prompts/` + `worker_sections/` is canonical. Consumer repos carry no prompt files unless overriding by filename. Reconciliation is by **rendered output**.
3. **One Devin-CLI infra source**: hooks, skills, and `AGENTS.md` move into the package. Consumer repos carry no `.devin/` directory. **Hook paths are parameterized** before promotion — no consumer-specific literals in the shared package.
4. **No stale provenance**: no consumer or package file references `devin-orchestrator`.
5. **No dual materialization**: the shim no longer re-copies `.devin`; the package materializer is the single source.
6. **Zero functional regression**: dispatch, review, merge, fleet supervise, worktree creation, hook execution, skill invocation, the `_worker_authored_dirty` probe, and the issue #487 stale-claim recovery path all behave identically post-migration.
7. **No data loss**: state.json (with rewritten paths), events.db (checkpointed), dispatch/sidecar records, cross-family spec reviews, worktrees, and the 12 untracked `.devin/` files are all preserved with verified backups.

---

## Non-goals

- Changing the layered-config merge semantics.
- Changing the `prompts_dir` override-by-filename mechanism.
- Migrating the empericus consumer (separate PR).
- Rewriting the launch shim's venv/re-base/sanitization logic.
- Changing the fleet dir.
- Fixing the `state_dir` vs `worktrees_dir`/`reviews_dir` default split itself (tracked as a separate issue — see "Separate issues" below).

---

## Cross-repo ordering constraint

- Phase 1 (package materializer) must ship and be installed before Phase 4 removes the shim's `.devin` copy.
- Phase 3 (state migration) and Phase 4 (shim move) must land **together** in the consumer.
- Rollback is: restore `Copy-Item` backups first (state, worktrees, `.devin`, `.git/info/exclude`), then `git reset --hard` for tracked files. **`git reset --hard` restores none of what matters** — `.var/`, `.devin/`, and `.git/info/exclude` are all excluded from git.
- Rehearse the rollback once on a scratch copy before Phase 3 runs for real.
- **Scheduling note**: job-cannon's `worker_model` / `cross_family.model` were changed today. Leave a day between those changes and the migration to avoid conflating variables if anything regresses.

---

## Phased plan

### Phase 0 — Freeze, snapshot, checkpoint, and gitignore (preconditions)

**Goal**: establish a safe rollback point, confirm no active writers, checkpoint SQLite, and move ignore rules into tracked `.gitignore`.

1. **Move `/.var/` and `/.devin/` into tracked `.gitignore`** (currently in `.git/info/exclude:29,33,35` under a "TEMP" preface — does not survive clone, fresh CI checkout, or `git reset --hard`). Add both patterns to `.gitignore`, commit. Back up `.git/info/exclude` to `.git/info/exclude.backup-<timestamp>`. This is a **prerequisite for the migration**: on any fresh clone, untracked `.devin/` without an ignore rule would be swept into the first worker PR by `git add -A`.
2. **Disable the scheduled task first** (before killing the supervisor — killing first leaves a 5-min window for the task to relaunch):
   ```powershell
   schtasks /Change /TN "charlie-work fleet" /Disable
   ```
3. **Stop the supervisor by PID**:
   ```powershell
   Get-Process | Where-Object { $_.CommandLine -like '*fleet supervise*' } | Stop-Process
   ```
   Verify fleet-level locks released: `fleet-supervisor.lock`, `fleet.lock`, `fleet.json.lock`.
4. **Stop CI runners**: `charlie runners scale-down`. Confirm `charlie runners status` → zero.
5. **Confirm zero in-flight workers**: inspect `state.json` `dispatches` for live PIDs; check `sessions_dir` sidecars. `charlie doctor --adapter-probe` surfaces stale/failed sessions.
6. **Check per-repo locks**: verify `state.json.lock` and `supervisor.lock` in `.var/devin-orchestrator/` are not held (0-byte or absent = not locked).
7. **Record pre-migration SHAs**: `git -C ~/repos/job-cannon rev-parse HEAD` and `git -C ~/repos/charlie-work rev-parse HEAD`.
8. **Checkpoint SQLite** — `events.db-wal` is 4.1 MB (larger than the 3.2 MB main DB, unclean shutdown):
   ```python
   import sqlite3
   conn = sqlite3.connect(r"C:\Users\senki\repos\job-cannon\.var\devin-orchestrator\events.db")
   conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
   conn.execute("PRAGMA integrity_check")
   conn.close()
   ```
   Verify `-wal`/`-shm` gone. Record event count for post-migration comparison.
9. **Backup both state dirs**:
   ```powershell
   Copy-Item -Recurse ~/repos/job-cannon/.var/devin-orchestrator ~/repos/job-cannon/.var/devin-orchestrator.backup-<timestamp>
   Copy-Item -Recurse ~/repos/job-cannon/.var/charlie-work ~/repos/job-cannon/.var/charlie-work.backup-<timestamp>
   ```
   Verify `state.json` and `events.db` byte sizes match. The `.var/charlie-work/` backup protects the 74 live worktrees.
10. **Backup `.devin/`** (12 untracked files have no git history):
    ```powershell
    Copy-Item -Recurse ~/repos/job-cannon/.devin ~/repos/job-cannon/.devin.backup-<timestamp>
    ```
    **Do NOT use `git stash`** — it removes `.devin/` from the working tree (breaking the shim's `cp -r`), requires git ≥ 2.35 for pathspec-limited stash, and `CLAUDE.md` bans `git stash` inside worktrees (shared `.git/` ref hazard).

**Validation gate**: `.gitignore` committed with `/.var/` + `/.devin/`; scheduled task disabled; supervisor gone; fleet + per-repo locks released; `runners status` → zero; zero in-flight workers; SQLite checkpointed; all three backups exist and byte-match; both SHAs recorded.

---

### Phase 1 — Centralize Devin-CLI infrastructure into charlie-work

**Goal**: move hooks, skills, and `AGENTS.md` from job-cannon's `.devin/` into the charlie-work package, with hook paths parameterized.

1. **Per-file disposition table** (corrected: `create-pr` not `create-PR`):
   - `hooks/block_destructive.py` — promote (parameterize `devin-orchestrator` literal first)
   - `hooks/log_git_ops.py` — promote (parameterize)
   - `hooks/require_ci_clean.py` — promote (parameterize)
   - `hooks/require_live_scan.py` — promote (parameterize)
   - `hooks/require_platform_contract.py` — promote (parameterize, git-tracked)
   - `hooks/session_end.py` — promote (parameterize `_VAR = Path(".var/devin-orchestrator")` → `Path(os.environ.get("CHARLIE_WORK_STATE_DIR", ".var/charlie-work"))`)
   - `hooks/session_start.py` — promote
   - `hooks/hooks.v1.json` — drop (legacy, unreferenced)
   - `hooks/__pycache__/` — **exclude**
   - `skills/{capture-fixture,commit,complete,create-branch,create-pr,preflight,push,test}/` — promote each
   - `AGENTS.md` — promote (parameterize `devin-orchestrator` reference at line 71)
   - `orchestrator.md` — drop (references dead `../devin-orchestrator` sibling)
   - `worker.md` (loose) — drop (references dead in-tree state paths)
   - `prompts/worker.md` — Phase 2 disposition
   - `prompts/rework.md` — Phase 2 disposition
2. **Parameterize all hook paths before promotion**: 6+ sites across 5 hook files hardcode `.var/devin-orchestrator`. These are cwd-relative (worker's cwd = worktree), so writer and reader are mutually consistent today — but promoting them verbatim hardcodes a consumer-specific legacy literal into the shared package, violating Goal 4. Replace with env var or config-derived path.
3. Create `src/charlie_work/devin_infra/` with `hooks/`, `skills/`, `AGENTS.md`. Copy the promoted (parameterized) files.
4. Add a **single** package materializer (`_materialize_package_infra` in `worktree.py`) that copies `src/charlie_work/devin_infra/` into `<worktree>/.devin/` at worktree-creation time. This is the **only** materializer for `.devin/`.
5. Add `DispatchConfig.devin_infra_dir` (default `None` = use package). When set, consumer-relative path shadows the package.
6. **No `pyproject.toml` packaging change needed** — hatchling's directory-based `packages` directive already ships non-`.py` assets. Verified by building the wheel.

**Inertness note**: until Phase 4 removes the shim's `cp -r`, the shim (launch-time) always overwrites the package materializer's output (worktree-creation-time). The new materializer is **inert** during Phase 1 — a broken materializer cannot break production until Phase 4 deliberately hands it the baton. This is a safety property worth stating: gate Phase 1 on unit/integration evidence only; move the "materializer feeds a real worker" gate to Phase 4.

**Validation gate**:
- New test: `test_package_devin_infra_materializes_into_worktree` — worktree with `devin_infra_dir=None`, asserts `.devin/hooks/block_destructive.py` exists and matches package source byte-for-byte.
- New test: `test_devin_infra_consumer_override_shadows_package` — `devin_infra_dir=.devin`, asserts consumer copy wins.
- New test: `test_materialize_directory_is_last_writer_wins` — documents actual semantics (differing bytes overwritten).
- New test: `test_wheel_contains_devin_infra_assets` — builds wheel, asserts `devin_infra/**` present (regression test for the property that is already true).
- New test: `test_promoted_hooks_have_no_legacy_literal` — greps `devin_infra/hooks/*.py` for `devin-orchestrator` — zero matches.
- `uv run --extra dev pytest -q` green.

---

### Phase 2 — Standardize and centralize prompt templates (rendered-output reconciliation)

**Goal**: make `src/charlie_work/prompts/` + `worker_sections/` the single canonical source; reconcile by **rendered output**.

1. **Render both `worker.md` variants** with an identical fixture value set under `strict=True`. Diff the **rendered** output. Map jc's flat content onto the 8 existing `$section_*` partials — general content goes into package partials, jc-specific stays as a consumer override.
2. **Reconcile `rework.md`** the same way — render both, diff rendered output.
3. If any jc prompt content is genuinely consumer-specific, leave *only that file* in `.devin/prompts/`. If nothing is, remove `.devin/prompts/` entirely. Document per-section in the commit.
4. **Remove `runtime.prompts_dir` from job-cannon's config** as an explicit numbered step in this phase (not just in a summary table). `doctor.py:920` fails when `prompts_dir` points at a non-existent directory — if the directory is deleted but the config key remains, `charlie doctor` breaks.
5. Remove the loose `.devin/orchestrator.md` and `.devin/worker.md`.
6. Update `examples/orchestrator.config.devin.yaml`.
7. **Produce a rendered-diff review artifact** (not a test — a charlie-work package test cannot assert on a job-cannon working-copy file). Attach it to the PR for human review.

**Validation gate**:
- `render_prompt("worker.md", {...}, search_dirs=())` produces no literal `$placeholder` text.
- New test: `test_canonical_prompts_have_no_consumer_refs` — greps `src/charlie_work/prompts/**/*.md` for `devin-orchestrator` — zero matches.
- New test: `test_all_shipped_templates_render_strict_per_entry_point` — a table of `(template_name, variable_set_factory)` pairs covering each real dispatch path (`worker`, `review`, `rework`, `cross_family_review`, `cross_family_spec_review`, `fleet_burndown`), asserting `render_prompt(..., strict=True)` does not raise. Partials are validated via the parent's variable set (`prompts.py:92-93`), not standalone.
- `uv run --extra dev pytest -q` green.

---

### Phase 3 — Consolidate state directory to `.var/charlie-work` (operator-invoked migration)

**Goal**: migrate job-cannon off the legacy state dir via an explicit operator-invoked command.

1. **Add a `charlie migrate-state` subcommand** (new `state_migration.py` + CLI wiring). Never called from `paths.runtime_paths` (which is pure — join → `resolve()` → frozen dataclass; `ensure()` is deliberately separate).
2. **Migration procedure** (copy-then-verify-then-delete, NOT `os.replace`):
   - **Pre-flight gate**: verify quiescence (no live sessions, no held locks including `state.json.lock` and `supervisor.lock`). Refuse if unexpected files in target.
   - **Checkpoint SQLite**: `PRAGMA wal_checkpoint(TRUNCATE)` + `PRAGMA integrity_check`.
   - **Copy state files** from `.var/devin-orchestrator/` into `.var/charlie-work/`: `state.json`, `events.db` (post-checkpoint), `dispatches/`, `prs/`, `issues/`, `notify/`, `logs/`, **`cross-family/`** (3 live spec-review artifacts — hardcoded `workflow.py:10250`, not in any config-key inventory). Do NOT touch `.var/charlie-work/worktrees/`.
   - **Overwrite the 0-byte `events.db`** with `-Force` — `Copy-Item` without `-Force` would leave the 0-byte file and silently orphan 10,345+ events.
   - **Verify**: compare file counts and byte sizes; re-run `PRAGMA integrity_check`; compare event counts pre/post.
   - **Rewrite embedded legacy paths in `state.json`**: 517 occurrences of `devin-orchestrator` are embedded as absolute paths in `prompt_path` (268), `decision_path` (135), `cross_family_report` (108, informational only), and `verdict_source` (3). Post-migration these point at a deleted directory. The issue #487 stale-claim recovery path (`workflow.py:2495-2500`) silently `continue`s when `prompt_path` doesn't exist — permanently unreachable for 268 PRs. Rewrite via JSON load → recursive string-replace of the old state-dir prefix → atomic write (tmp + `replace()`). **Not a text `sed`** — risks corrupting escaped-backslash Windows paths. Gate with count assertion: post-rewrite legacy prefix occurrences = 0, new prefix occurrences = 517.
   - **Delete source**: only after verification, delete the copied files from `.var/devin-orchestrator/` (not the whole dir — see dispositions below).
   - **Write marker**: `.var/charlie-work/.migrated-from-devin-orchestrator`.
   - **Fleet registry**: self-heals on next `charlie` invocation via `touch_repo` (`fleet_registry.py:137`). No manual `fleet.json` edit needed (belt-and-braces optional, noting it is redundant).
3. **Remove all `devin-orchestrator` path references from job-cannon's `orchestrator.config.yaml`**:
   - `state_dir` → delete. `devin.session_manifest` / `session_results` / `sessions_dir` → delete. `reviews_dir` → delete. `notify.file_path` → delete. `devin.shell_command` → update to package shim location (Phase 4, lands together).
   - **Note**: deleting a per-repo key adopts the **global layer's** value where one is set, not necessarily the dataclass default. Validation must assert **resolved** config values, not grep for string absence.
4. **Disposition of remaining `.var/devin-orchestrator/` files** (full enumeration):
   - `launch_devin_worker.sh` → Phase 4 (move to package).
   - `finish_worker.sh`, `setup_worker.sh` → **confirm-then-delete** (uninvoked — `launch_devin_worker.sh` never references them; porting dead scripts carries their legacy paths into the package for no benefit).
   - `bin/powershell.cmd` → delete (no `bin`/`PATH` reference found in shim).
   - `_stash/` → delete or archive.
   - `git-ops.log`, `session-events.log`, `last_gate_block.txt` → delete (runtime logs, no migration value).
   - `state.json.lock`, `supervisor.lock` → delete (0-byte lock files).
   - `state.json.corrupt-2026-07-12T051315Z` → delete (corrupt backup).
   - `events.jsonl.migrated` → delete (migration marker, already processed).
   - `INCOMPLETE-issue-306.flag` → delete or archive.
   - Design docs (`acp-analysis.md`, `api-discovery-report.md`, `architecture-proposal.md`, `devin-cli-reference.md`, `extensibility-design.md`, `extensibility-launch-prompt.md`, `fallback-analysis.md`, `FLEET-READINESS-DESIGN.md`, `realistic-path-forward.md`, `REVIEWER_SCOPE.md`, `REVIEWER_SCOPE_SCANNER.md`) → move to job-cannon `docs/` if worth keeping.
5. **Disposition of `.var/charlie-work/` stray files** (do NOT delete the directory — 74 live worktrees):
   - 0-byte `events.db` → overwrite with migrated real one (`-Force`).
   - Stray finding docs → move to `docs/` or archive.
   - `worktrees/`, `dispatches/reviews/`, `fable-logs/` → **leave untouched**.
6. **Junction-aware deletion**: `.var/test-junction` is a reparse point (junction → `.var/test-target`). Recursive delete follows junctions. Enumerate and handle explicitly.

**Validation gate**:
- New test: `test_migrate_state_copies_and_verifies` — temp tree with legacy state + target with live worktrees, asserts state files moved, worktrees untouched, event counts match.
- New test: `test_migrate_state_rewrites_embedded_paths` — asserts 517 legacy prefix occurrences → 0, new prefix → 517.
- New test: `test_migrate_state_overwrites_zero_byte_events_db` — asserts `-Force` semantics (0-byte target overwritten).
- New test: `test_migrate_state_copies_cross_family_dir` — asserts `cross-family/` is in the copy list.
- New test: `test_migrate_state_idempotent`.
- New test: `test_migrate_state_checkpoints_sqlite`.
- New test: `test_migrate_state_junction_aware`.
- New test: `test_no_hardcoded_paths_root_dir_un inventoried` — greps `src/charlie_work/` for `paths.root /` and `paths.root.joinpath`, requires every hit in the inventory table.
- Manual: `charlie migrate-state --dry-run` lists exactly what will move; `charlie migrate-state` executes; `charlie doctor` confirms `state_file` resolves to `.var/charlie-work/state.json`.
- Manual: assert resolved config values: `load_layered_config(repo_root, None)` → `config.runtime.state_dir` == `.var/charlie-work`, etc.

---

### Phase 4 — Consolidate the launch shim and eliminate dual materialization

**Goal**: move the shim into charlie-work, parameterize it, remove its `.devin` re-copy, and answer the watchdog question.

1. **Move `launch_devin_worker.sh`** into charlie-work (`src/charlie_work/devin_infra/` or `scripts/`). **Do NOT port `finish_worker.sh` or `setup_worker.sh`** — they are uninvoked (confirm-then-delete in Phase 3).
2. **Parameterize**: replace hardcoded `MAIN="/c/Users/senki/repos/job-cannon"` with `CHARLIE_WORK_CONSUMER_ROOT` env var.
3. **Remove the shim's `.devin` materialization step** (`rm -rf .devin && cp -r "$MAIN/.devin" .devin`). Replace with an assertion: `.devin/hooks/` must exist; fail loudly if not. The package materializer (Phase 1) is now the single source — and it is no longer inert after this step.
4. **Resolve the shim path in `shell_command`**: `shell_command` is a static YAML list with a closed placeholder set `{"prompt_path", "issue_number", "branch", "model_args"}` (`config.py:1639`). A `{shim_path}` placeholder requires adding it to that set plus a resolver. **Prefer a `console_scripts` entry point** instead — the shim is resolved via `PATH`, needs no placeholder, and sidesteps editable-vs-wheel path divergence.
5. **Watchdog decision**: either (a) preserve the log-not-growing behavior (bash.exe wrapping) so `watchdog.enabled: false` stays correct, or (b) change it, in which case re-enabling is an explicit decision with validation. Document which in the commit.
6. **`_worker_authored_dirty` interaction**: after `.devin/` is untracked and materialized from the package, the probe (`worktree.py:860-923`) uses `git status --porcelain=v2 -z --untracked-files=all` and excludes paths in `injected_paths` via the `excluded_path in path.parents` predicate (line 919). `.devin` is untracked (no tracked files to report as modified) but still in `injected_paths` (excludes untracked files under `.devin/`). The probe should see no dirty paths from `.devin/`. **Verify on one real dispatch before Phase 5.**

**Validation gate**:
- New test: `test_launch_shim_does_not_recopy_devin` — asserts no `cp -r` of `.devin` in the shim.
- New test: `test_shim_parameterized_consumer_root`.
- New test: `test_worker_authored_dirty_excludes_untracked_devin`.
- New test: `test_shell_command_resolves_shim_via_entry_point` (or `test_shell_command_shim_path_placeholder` if going the placeholder route).
- Manual: `charlie work --dry-run --limit 1` and confirm worktree has `.devin/hooks/` from the package materializer (not the shim).
- Manual: one real dispatch → no `worktree_unsafe` event.
- `uv run --extra dev pytest -q` green.

---

### Phase 5 — Remove the consumer `.devin/` directory and legacy state dir

**Goal**: delete the now-redundant consumer-local copies after all upstream phases are verified in production.

1. **Confirm** Phase 1–4 have run in production for at least one full dispatch+review cycle with no regressions.
2. **Delete `.devin/`** — all 15+ files (tracked and untracked). The Phase 0 backup is the safety net.
3. **Delete `.var/devin-orchestrator/`** — after confirming the backup is intact and a second full cycle is clean. Use junction-aware deletion.
4. **Remove `injected_paths: [".devin"]`** from job-cannon's config ONLY if the dirty-check probe is verified to not need it (Phase 4 analysis). If still needed for exclusion, keep it.
5. **Re-enable the scheduled task**: `schtasks /Change /TN "charlie-work fleet" /Enable`.
6. **Run `charlie worktree-clean --dry-run`** as an explicit post-migration step — the command has been a silent no-op (pointing at `.var/devin-orchestrator/worktrees` which doesn't exist) and will now flip to live against a 74-worktree backlog on its first invocation. Read the dry-run output before the first real run.
7. Update `docs/MIGRATION.md`.

**Validation gate**:
- `git -C ~/repos/job-cannon status` shows no `.devin/` tracked or untracked.
- `charlie doctor` passes.
- `grep -r devin-orchestrator ~/repos/job-cannon/` → zero matches **outside `.var/charlie-work/{state.json,events.db}`** (517 + 3 legacy occurrences remain inside migrated state files by design).
- **Re-scope the grep to also cover the installed package**: `python -c "import charlie_work; ..."` → grep `devin_infra/` for `devin-orchestrator` → zero matches (hooks were parameterized in Phase 1).
- `charlie fleet status` → job-cannon present with correct `state_dir`.
- Scheduled task re-enabled and running.

---

## Verification and validation checklist

### Automated (charlie-work repo)
- [ ] `uv run --extra dev pytest -q --tb=short` — full suite green.
- [ ] `uv run ruff check .` + `uv run ruff format --check .`
- [ ] `test_package_devin_infra_materializes_into_worktree`
- [ ] `test_devin_infra_consumer_override_shadows_package`
- [ ] `test_materialize_directory_is_last_writer_wins`
- [ ] `test_wheel_contains_devin_infra_assets` (regression test — property already true)
- [ ] `test_promoted_hooks_have_no_legacy_literal`
- [ ] `test_canonical_prompts_have_no_consumer_refs`
- [ ] `test_all_shipped_templates_render_strict_per_entry_point`
- [ ] `test_migrate_state_copies_and_verifies`
- [ ] `test_migrate_state_rewrites_embedded_paths` (517 → 0 legacy, 517 new)
- [ ] `test_migrate_state_overwrites_zero_byte_events_db`
- [ ] `test_migrate_state_copies_cross_family_dir`
- [ ] `test_migrate_state_idempotent`
- [ ] `test_migrate_state_checkpoints_sqlite`
- [ ] `test_migrate_state_junction_aware`
- [ ] `test_no_hardcoded_paths_root_dir_uninventoried`
- [ ] `test_launch_shim_does_not_recopy_devin`
- [ ] `test_shim_parameterized_consumer_root`
- [ ] `test_worker_authored_dirty_excludes_untracked_devin`
- [ ] `test_shell_command_resolves_shim_via_entry_point`

### Automated (job-cannon consumer)
- [ ] `charlie doctor` — all checks pass, `state_file` resolves to `.var/charlie-work/state.json`.
- [ ] `charlie doctor --adapter-probe` — no stale/failed sessions.
- [ ] Assert **resolved** config values (not grep): `load_layered_config(repo_root, None)` → `config.runtime.state_dir` == `.var/charlie-work`, `config.devin.sessions_dir` == `.var/charlie-work/dispatches/sessions`, `config.review_dispatch.reviews_dir` == `.var/charlie-work/dispatches/reviews`.
- [ ] `charlie fleet status` → job-cannon present with correct `state_dir`.

### Manual / behavioral
- [ ] One full `charlie bash-rats --once` (or `charlie work --limit 1` + `charlie review-queue`) dispatch cycle completes, all reading from `.var/charlie-work/`.
- [ ] Worktree `.devin/hooks/` from the package, verified by `diff` (zero diff).
- [ ] Rendered worker prompt matches package `worker.md` (or deliberate override), verified by inspecting `.var/charlie-work/issues/issue-*/worker-prompt.md`.
- [ ] `events.db` queryable: event counts match pre-migration.
- [ ] `charlie runners status` — runners reconverge.
- [ ] No `worktree_unsafe` / `session_failed_escalated` events in the first 24 h.
- [ ] Watchdog behavior unchanged (or explicitly re-enabled with validation).
- [ ] `charlie worktree-clean --dry-run` output reviewed before first real run.

### Rollback readiness (until Phase 5 completes)
- [ ] `.var/devin-orchestrator.backup-<timestamp>/` intact and byte-matches.
- [ ] `.var/charlie-work.backup-<timestamp>/` intact (protects 74 worktrees).
- [ ] `.devin.backup-<timestamp>/` intact.
- [ ] `.git/info/exclude.backup-<timestamp>` intact.
- [ ] Pre-migration SHAs recorded.
- [ ] **Rollback truth stated plainly**: `git reset --hard` restores *tracked files only*. State, worktrees, `.devin`, and `.git/info/exclude` all depend solely on the `Copy-Item` backups. Restore backups first, then `git reset --hard`.
- [ ] **Rollback rehearsed once** on a scratch copy before Phase 3 runs for real.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Deleting 74 live worktrees | `.var/charlie-work/` is NOT stale; Phase 3 migrates state files into it without touching `worktrees/`. Phase 0 backs up the full dir. |
| 12 untracked `.devin/` files lost | Phase 0 backs up `.devin/` via `Copy-Item` (not `git stash` — which removes from working tree and has shared-ref hazard). |
| 517 embedded legacy paths dangle | Phase 3 rewrites `state.json` paths via JSON load → recursive string-replace → atomic write. Count assertion: 517 → 0. |
| `cross-family/` spec reviews lost | Added to Phase 3 copy list. Hardcoded path inventory gate prevents future misses. |
| Fresh clone sweeps `.devin/` into PR | Phase 0 moves `/.var/` + `/.devin/` into tracked `.gitignore`. |
| `os.replace` fails on Windows | Migration is copy-then-verify-then-delete, NOT `os.replace`. |
| SQLite WAL corruption | `wal_checkpoint(TRUNCATE)` + `integrity_check` before copy; verify `-wal` gone; re-check after copy; compare counts. |
| 0-byte `events.db` not overwritten | Use `-Force` on copy; test asserts overwrite. |
| Fleet registry stale | Self-heals via `touch_repo` on next command. No manual edit needed. |
| Two materializers clobber | Re-derived: the real conflict is shim `cp -r` vs package materializer. Package materializer is sole source after Phase 4. |
| Removing jc `worker.md` injects 8 new sections | Reconciliation by rendered output. Section-level decomposition. Behavioral change is explicit. |
| `_worker_authored_dirty` false positives | Phase 4 analyzes untracked + `injected_paths` interaction, verifies on one real dispatch. |
| Watchdog re-enablement | Phase 4 documents whether shim preserves log-not-growing behavior. |
| Junction-following deletion | `.var/test-junction` is a reparse point; junction-aware deletion. |
| `shell_command` can't express package path | Use `console_scripts` entry point (resolved via `PATH`, no placeholder needed). |
| Hook paths hardcode legacy literal | Parameterize before promotion; test asserts zero `devin-orchestrator` in `devin_infra/`. |
| Dead scripts ported for no benefit | `setup_worker.sh`/`finish_worker.sh` are uninvoked — confirm-then-delete, don't port. |
| `worktree-clean` flips from no-op to live | Run `--dry-run` post-Phase-3; read output before first real run. |
| Cross-repo ordering | Package ships first, consumer second, rollback in reverse. Rollback rehearsed. |
| `doctor` breaks on stale `prompts_dir` | `prompts_dir` config key removed in same phase as directory deletion (Phase 2, explicit step). |
| Model changes confound migration | Leave a day between model config changes and migration. |

---

## Separate issues (pre-existing bugs, not this plan's fault but surfaced by it)

1. **Three-way worktree path divergence**: `worktree.py` default, `config.claude_code.worktrees_dir`, and `cli.py:344` `paths.root / "worktrees"` are three different conventions. `worktree-clean` has been a silent no-op in job-cannon as a result. Fix: derive `worktrees_dir` from one resolver all three callers use. **Record**: 74-worktree backlog, actively growing (was 72 earlier today).

2. **`count_fleet_live_sessions` fails open**: `fleet_registry.py:206-209` — silent `continue` when `sessions_dir` missing (no warning, no skip record). Both callers discard the skip list (`supervise.py:665`, `workflow.py:4993`). A concurrency cap that silently undercounts is an over-dispatch hazard independent of this migration.

3. **`state_dir` vs `worktrees_dir`/`reviews_dir` default split**: overriding `runtime.state_dir` silently relocates some data and not the rest. Worth its own issue so future overrides are all-or-nothing.

---

## File-level change summary

### charlie-work package (new + modified)
- **New**: `src/charlie_work/devin_infra/` (hooks/, skills/, AGENTS.md, launch_devin_worker.sh) — all paths parameterized
- **New**: `src/charlie_work/state_migration.py` (operator-invoked `charlie migrate-state`)
- **Modified**: `src/charlie_work/cli.py` — add `migrate-state` subcommand
- **Modified**: `src/charlie_work/worktree.py` — add `_materialize_package_infra` (sole materializer for `.devin/`)
- **Modified**: `src/charlie_work/config.py` — add `DispatchConfig.devin_infra_dir` field; add `{shim_path}` to placeholder set OR add `console_scripts` entry point
- **Modified**: `pyproject.toml` — add `console_scripts` entry point for the shim (if going that route); **no `package-data`/`force-include`**
- **Modified**: `examples/orchestrator.config.devin.yaml`
- **Modified**: `docs/MIGRATION.md`
- **New tests**: `tests/test_devin_infra_materialize.py`, `tests/test_state_migration.py`, additions to `tests/test_prompts.py` and `tests/test_worktree.py`

### job-cannon consumer (modified + deleted)
- **Modified**: `.gitignore` — add `/.var/` + `/.devin/` (Phase 0)
- **Modified**: `orchestrator.config.yaml` — remove all `devin-orchestrator` paths, remove `prompts_dir` (Phase 2 explicit step), update `devin.shell_command` to entry point (Phase 4)
- **Deleted**: `.devin/` (Phase 5)
- **Deleted**: `.var/devin-orchestrator/` (Phase 5, junction-aware)
- **Migrated**: state files + `cross-family/` from `.var/devin-orchestrator/` into `.var/charlie-work/` (Phase 3)
- **Rewritten**: 517 embedded legacy paths in `state.json` (Phase 3)
- **Moved**: stray finding docs + design docs into `docs/`
- **Handled**: `.var/test-junction` reparse point (explicit disposition)
