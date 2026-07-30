# Adversarial review — `centralize-charlie-work-config-templates-a41d9a.md` (round 2)

Reviewed 2026-07-29. Every claim below carries a `file:line` citation or the exact command
that produced it. The plan's credibility rests on "verified against live code 2026-07-29";
this review is held to the same standard.

**Verdict: do not execute as written.** Phase 1 and Phase 5 are built on two premises that
are factually false, and Phase 3 omits a data-rewrite step whose absence silently degrades
the review pipeline for every pre-migration PR. The phase *ordering* is sound and several
safety analyses are better than the first round — the defects are concentrated in
(a) premises never empirically tested, (b) one missing migration step, and (c) validation
gates that cannot pass or cannot be written.

---

## Bucket 1 — Factually wrong premises

These are not gaps. They are assertions the plan makes that the code contradicts. Each one
invalidates work the plan schedules.

### 1.1 The packaging premise is false; the prescribed fix is a no-op (BLOCKER)

The plan treats "make the package ship its `.md` assets" as work to be done, and prescribes
adding `package-data` to `pyproject.toml`.

Empirically tested by building the wheel and listing its contents:

```
TOTAL 65
NONPY: charlie_work/prompts/cross_family_review.md
NONPY: charlie_work/prompts/cross_family_spec_review.md
NONPY: charlie_work/prompts/fleet_burndown.md
NONPY: charlie_work/prompts/orchestrator.md
NONPY: charlie_work/prompts/review.md
NONPY: charlie_work/prompts/rework.md
NONPY: charlie_work/prompts/worker.md
NONPY: charlie_work/prompts/worker_claude_code.md
NONPY: charlie_work/prompts/worker_sections/*.md   (8 files)
```

All 16 non-`.py` assets already ship. `pyproject.toml`'s
`[tool.hatch.build.targets.wheel] packages = ["src/charlie_work"]` includes the whole
package tree — hatchling is directory-based, not manifest-based. There is no `force-include`
and no `package-data` today, and none is needed.

Worse, `package-data` is a **setuptools** key. Hatchling ignores it silently. Adding it
produces no error, no warning, and no behavior change — so the plan's step would appear to
succeed while doing nothing, and would then be cited as the reason a future asset ships when
the real reason is the `packages` directive.

**Fix:** delete the `package-data` step. Replace it with a regression test that asserts the
built wheel contains the asset tree (the check above, as a test), so the property that is
already true stays true if someone narrows `packages` later.

### 1.2 `materialize_dirs: [".devin"]` does not exist in job-cannon's config (BLOCKER)

A whole strand of the plan — Phase 1 step 3, the cross-repo ordering note, Phase 5 step 4,
a risk-register row, and the file-level change summary — reasons about a conflict between
two materializers, on the premise that job-cannon sets `materialize_dirs: [".devin"]`.

`grep -n "materialize_dirs" ~/repos/job-cannon/orchestrator.config.yaml` returns nothing.
The key appears nowhere in job-cannon's config. What exists is
`dispatch.injected_paths: [.devin]`.

Two consequences the plan has backwards:

- The "two materializers clobber each other" risk is against a phantom. The real second
  writer is the launch shim's `rm -rf .devin && cp -r "$MAIN/.devin" .devin`
  (`.var/devin-orchestrator/launch_devin_worker.sh:38`).
- `_materialize_directory`'s `git update-index --assume-unchanged` shielding
  (`src/charlie_work/worktree.py:1392+`) has **never** applied to `.devin` in job-cannon,
  because that code only runs for paths in `materialize_dirs`. Any reasoning that assumes
  `.devin` is currently shielded from dirty-tree detection is wrong.

**Fix:** re-derive that entire strand from `injected_paths` + the shim. The migration's real
shape is "the shim's `cp -r` is the incumbent writer; the package materializer is the
challenger" — which, as §3.1 notes, is actually *safer* than the plan realizes.

---

## Bucket 2 — Missing work

### 2.1 514 absolute legacy paths embedded in `state.json` are never rewritten (BLOCKER)

Phase 3 copies `state.json` byte-for-byte. Phase 5 deletes the directory those bytes point
into. Counted in the live `state.json` (976,489 B):

| key | legacy-prefixed | consumer |
|---|---|---|
| `prompt_path` | 268 | `workflow.py:2496` |
| `decision_path` | 135 | `workflow.py:2503` |
| `cross_family_report` | 108 | informational only (see below) |
| `verdict_source` | 3 of 27 | mtime comparison |
| **total** | **514** | |

514 is also the *raw* count of `devin-orchestrator` substrings in the file, so these four
keys account for every legacy occurrence — there is no fifth hiding place.

The consumers treat non-existence as "skip", not as an error:

```python
# workflow.py:2495-2510
prompt_path_str = pr_state.get("prompt_path")
if not prompt_path_str:
    continue
prompt_path = Path(prompt_path_str)
if not prompt_path.exists():
    continue                      # <-- silently skips, post-migration: always
```

Blast radius, stated precisely (the first-round instinct to call this "lost approvals" is
wrong and worth correcting):

- **Not** a lost-approval bug. `reconcile.py:186-190` re-derives
  `paths.prs / f"pr-{pr_number}" / "review-decision.json"` from config, so the primary
  approval path is config-derived and self-heals.
- **Is** a silent capability regression, and this is the whole of the finding: the issue-#487
  stale-claim recovery path reads `prompt_path` *from state* and skips on non-existence —
  `continue` for all 268 affected PRs, permanently unreachable post-migration, with no log
  line.
- **Not** a cost regression. An earlier draft of this review claimed the 108 stale
  `cross_family_report` paths would re-burn paid cross-family reviews. That is wrong:
  `_cross_family_for_pr` computes `report_path = pr_dir / "cross-family-review.md"`
  (`workflow.py:10300`) — config-derived, never read from state — and all 108 stored values
  point under `prs/pr-N/`, which Phase 3 *does* copy. Reuse detection self-heals exactly like
  `reconcile.py:186`. The stored values are informational only.
- Minor: `_is_verdict_newer_than_brief` (`workflow.py:3597-3610`) compares mtimes on these
  paths; with them dangling the comparison is meaningless.

**Fix:** add a Phase 3 step that rewrites the embedded prefix. Do it as a JSON
load → recursive string-replace of the old state-dir prefix → atomic write (respecting the
tmp + `replace()` invariant), *not* as a text `sed` — a text substitution on a 1 MB JSON
file risks corrupting escaped-backslash Windows paths. Gate it with a count assertion:
post-rewrite occurrences of the legacy prefix must be 0 and occurrences of the new prefix
must equal 514.

Related, and much smaller: `events.db` payloads contain only **3** legacy-path rows (all
`record_review`), so the plan's event-*count* integrity check is adequate there. Not worth
a step.

### 2.2 `paths.root / "cross-family"` is in no inventory and no copy list

`workflow.py:10250` hardcodes `reviews_dir = self.paths.root / "cross-family"`. It is not a
config key, so it does not appear in the plan's config-keyed path inventory table. It holds
4 live spec-review artifacts. (These are distinct from per-PR cross-family reviews, which
live under `prs/pr-N/` — see §2.1.)

Phase 3's copy list enumerates `state.json`, `events.db`, `dispatches/`, `prs/`, `issues/`,
`notify/`, `logs/` — **not** `cross-family/`. So the directory is left behind in
`.var/devin-orchestrator/` and deleted by Phase 5.

The inventory's failure mode is sharper than "incomplete": it tracks
`review_dispatch.reviews_dir`, which for job-cannon is **inert** (`review_dispatch.enabled:
false`; reviewers run via `cross_family`), while missing the directory that is actually live.
A config-key-driven inventory structurally cannot see hardcoded paths.

**Fix:** add `cross-family/` to the Phase 3 copy list. Then add a gate that greps
`src/charlie_work/` for `paths.root /` and `paths.root.joinpath` and requires every hit to
appear in the inventory table — so the next hardcoded path cannot slip through.

### 2.3 No `.gitignore` migration; the only safety net is an untracked file marked "TEMP"

`/.var/` and `/.devin/` are excluded via `.git/info/exclude:33,35` — **not** `.gitignore`,
which has zero `.var`/`.devin` patterns. The block is prefaced:

```
29:# --- TEMP: devin-orchestration session (added by Claude Code) ---
33:/.var/
35:/.devin/
```

The hazard is **portability, not today's state**, and it is worth being precise because the
two get conflated easily:

- *On this machine, after migration:* `.devin` is untracked **and** ignored, so
  `git add -A` (`preflight/SKILL.md:37`) adds nothing. This is strictly better than today,
  where `.devin/hooks/require_platform_contract.py`, `.devin/prompts/rework.md` and
  `.devin/prompts/worker.md` are **tracked**, the shim's `rm -rf && cp -r` replaces them,
  `git status` reports them modified, and `git add -A` stages them into the PR. The
  migration fixes a live bug. Credit where due.
- *On any fresh clone, new machine, or re-cloned CI checkout:* `.git/info/exclude` does not
  come with the repository. `.devin` is then untracked **and not ignored**, and the first
  `git add -A` sweeps the entire package-materialized tree into the first worker PR.

So the migration converts a visible, self-limiting bug into an invisible one that only
appears off this host. Compounding it: `git reset --hard` — the plan's stated rollback
mechanism — cannot restore `.git/info/exclude`, Phase 0 does not back it up, and the block
is prefaced with a comment a past agent explicitly labeled temporary.

**Fix:** move `/.var/` and `/.devin/` into the tracked `.gitignore` as a Phase 0 step, and
add `.git/info/exclude` to the Phase 0 backup set. Tracked ignore rules survive
`reset --hard`, clone, and worktree creation.

### 2.4 `devin.shell_command` has no mechanism to express a package path

Phase 4 requires the shim to be invoked from the installed package. But `shell_command` is a
static YAML list (`config.py:557-597`, `shell_command: tuple[str, ...] = ()`), and
placeholder validation at `config.py:1622-1640` checks against a **closed set**
`{"prompt_path", "issue_number", "branch", "model_args"}`.

So a `{shim_path}` placeholder requires a code change — a new member in that set plus a
resolver — which the plan does not list among its changes. The alternative (hardcoding a
`site-packages` path in YAML) is fragile across `uv sync` and diverges between editable and
wheel installs.

**Fix:** add the `config.py` placeholder-set change explicitly to Phase 4's file list, or
choose a `console_scripts` entry point instead so the shim is resolved via `PATH` and needs
no placeholder at all. The entry-point route is preferable — it also sidesteps §3.2.

### 2.5 Phase 0 quiescence misses the per-repo locks

The plan checks the fleet-dir locks. It does not check `supervisor.lock` or
`state.json.lock`, both of which live in the per-repo state dir being moved. Copying a state
dir while a lock is held is exactly the race Phase 0 exists to prevent.

**Fix:** add both to the Phase 0 validation gate.

### 2.6 Undispositioned inventory entries

Present on disk, absent from the disposition table: `bin/powershell.cmd`, `_stash/`,
`git-ops.log`, `session-events.log`, `last_gate_block.txt`, `state.json.lock`,
`supervisor.lock`, and ~9 design docs the table does not enumerate individually. No claim
here that deleting any of them breaks a worker — the shim grep found no `bin`/`PATH`
reference for `powershell.cmd`. The defect is that they are *undispositioned*, and Phase 5
deletes the directory.

**Fix:** enumerate each with an explicit keep/copy/delete decision before Phase 5.

---

## Bucket 3 — Gates that cannot pass or cannot be written

A validation gate that cannot pass is worse than no gate: it trains the operator to override
it, which disarms the gates that do work.

### 3.1 `grep -r devin-orchestrator ~/repos/job-cannon/ → zero matches` is unsatisfiable

Phase 5's own gate. After Phase 3 there are 514 occurrences inside the migrated
`state.json` (§2.1) plus 3 in `events.db`. The gate fails on a successful migration.

**Fix:** scope it — `zero matches outside .var/charlie-work/{state.json,events.db}` — and
pair it with §2.1's count assertion, which is the check that actually matters.

### 3.2 "Assert exec bit on the shipped `.sh`" cannot be satisfied — and is unnecessary

Verified: a `chmod +x` `.sh` file ships in the wheel as `mode=100644`. The exec bit does not
survive the build. The gate is unachievable.

It is also pointless: `shell_command[0]` is `C:/Program Files/Git/bin/bash.exe` with the
shim as `argv[1]` (job-cannon `orchestrator.config.yaml`), so the shim is never executed
directly and its mode is irrelevant.

**Fix:** delete the gate. Replace with an assertion that `shell_command[0]` resolves to a
real interpreter — which is what `doctor.py:95-140` (`_probe_adapter` runs
`effective_template[0] --version`) already does.

### 3.3 `test_all_shipped_templates_render_strict` is ill-defined as specified

Two problems:

- There is no single "orchestrator's actual variable set". `worker`, `review`, `rework`,
  `cross_family`, and `fleet_burndown` each supply different variables. A test asserting all
  templates render against one set either passes vacuously (union of all sets) or fails
  spuriously (intersection).
- `prompts/worker_sections/*.md` are **partials**, not standalone templates. `render_prompt`
  already validates referenced partials against the parent's variable set
  (`prompts.py:92-93`: `for key in set(template.get_identifiers()) & set(sections)`), and
  deliberately does *not* validate unreferenced ones (`prompts.py:89-91` explains why: a
  stale unused partial must not block an unrelated render).

**Fix:** make it per-entry-point — a table of `(template_name, variable_set_factory)` pairs
covering each real dispatch path, asserting `render_prompt(..., strict=True)` does not raise.
That tests the property #589 was about (`prompts.py:16-27`) without inventing a fictional
global variable set.

### 3.4 `test_jc_worker_md_rendered_diff_documented` is not implementable as a test

A charlie-work package test cannot assert on a job-cannon working-copy file. It would fail
in CI, on a fresh clone, and for every other consumer.

**Fix:** demote to a one-off Phase 1 review artifact — produce the rendered diff, have a
human read it, attach it to the PR. Not a test.

### 3.5 `charlie doctor` fails if `runtime.prompts_dir` outlives its directory

`doctor.py:910-920` fails when `prompts_dir` does not exist. Phase 2 step 4 deletes
`.devin/prompts` but never says to remove `runtime.prompts_dir` from job-cannon's config
(only the file-level summary mentions it). Executed as written, Phase 5's "doctor is green"
gate fails.

**Fix:** make the config-key removal an explicit numbered step in the same phase as the
directory deletion, not an entry in a summary table.

### 3.6 Phase 1's validation gate cannot prove production behavior — but the ordering is safe

Until Phase 4 removes the shim's `cp -r`, the shim (launch-time) always overwrites the
package materializer's output (worktree-creation-time). The new materializer is therefore
**inert** during Phase 1, and no Phase 1 gate can demonstrate it working in production.

This is a point in the plan's favor that the plan never makes: the ordering means a broken
package materializer cannot break production until Phase 4 deliberately hands it the baton.

**Fix:** state the inertness explicitly, gate Phase 1 on unit/integration evidence only, and
move the "materializer actually feeds a real worker" gate into Phase 4 where it can be true.

### 3.7 Phase 0's `git stash --include-untracked -- .devin` is a footgun

Three independent problems:

1. `git stash` **removes** `.devin/` from the working tree. The shim's
   `cp -r "$MAIN/.devin"` then copies from an empty source and every launch is broken.
2. Pathspec-limited `git stash push` requires git ≥ 2.35 — unstated prerequisite.
3. The user's own `CLAUDE.md` bans `git stash` inside worktrees: stash refs live in the
   shared parent `.git/`, so a later `pop` can apply a sibling worktree's WIP onto the wrong
   branch.

**Fix:** replace with `Copy-Item -Recurse .devin .devin.backup-<timestamp>`. Same
durability, no working-tree mutation, no shared-ref hazard.

---

## Bucket 4 — Pre-existing bugs surfaced (file as issues; not the plan's fault)

Per global rule #12 these belong as tracked issues and as additions to the plan's existing
"Separate issue" section — folding them into plan criticism dilutes the parts that are
genuinely the plan's responsibility.

### 4.1 `worktree-clean` has been a silent no-op in job-cannon

`cli.py:344` passes `paths.root / "worktrees"` = `.var/devin-orchestrator/worktrees`, which
does not exist. The real worktrees — **74** of them — are at
`.var/charlie-work/worktrees/`, because `worktrees_dir` defaults independently of
`state_dir` (`worktree.py:259-260`). Three different worktree-path conventions coexist:
`_default_worktrees_dir`, `config.claude_code.worktrees_dir`, and `cli.py:344`.

Post-Phase-3 the two paths coincide and the command flips from no-op to live against a
74-worktree backlog on its first invocation. Severity is bounded, and the first-round
"could mass-delete 74 worktrees" framing overstates it: per-worktree gating is strong
(`worktree.py:3011-3024`) — live `gh pr view` MERGED, clean tree, HEAD contained in the
merged `headRefOid`, no live worker, fail-**closed** on any `gh` error. It is also
manual-only (sole caller `cli.py:342`).

Two residual concerns worth stating: the final orphan sweep is gated only on "git admin
record gone, tree remains", not on the merged-PR checks; and 74 live `gh pr view` calls in
one invocation is an API burst.

**Recommend:** run `charlie worktree clean --dry-run` as an explicit post-Phase-3 step and
read the plan before the first real run. File an issue for the three-way path divergence —
the single-point-of-enforcement fix is to derive `worktrees_dir` from one resolver that all
three callers use.

### 4.2 The 74-worktree backlog is a symptom, not a side issue

The plan defers the `state_dir` vs `worktrees_dir`/`reviews_dir` split to a "separate issue."
But that split is *why* the backlog exists and why 4.1 was invisible. The count is also
drifting — the plan says 72, today's count is 74 — so it is actively growing.

**Recommend:** keep the deferral, but record the count and its growth rate in the issue so
it is not rediscovered a third time.

### 4.3 Promoted hooks hardcode `.var/devin-orchestrator/`

Five files, six-plus sites: `block_destructive.py:28`, `require_ci_clean.py:51`,
`require_live_scan.py:93`, `require_platform_contract.py:217`, `log_git_ops.py:24`,
`session_end.py:26`, plus `AGENTS.md:71`.

The functional-break claim from round 1 should be dropped: `session_end.py:26` is
`_VAR = Path(".var/devin-orchestrator")` — **cwd-relative**, and the worker's cwd is the
worktree. `AGENTS.md:71` names the same worktree-relative path. Writer and reader stay
mutually consistent regardless of `runtime.state_dir`. Nothing breaks.

What remains is still real and still blocking for Goal 4: promoting these verbatim
hardcodes a consumer-specific legacy literal into the *shared package*, violating the plan's
own Goal 4 and global rule #9. And Phase 5's grep gate is scoped to
`~/repos/job-cannon/` — so it passes while the string lives on in `site-packages` and is
materialized back into every worktree on every launch.

**Fix (in-plan):** parameterize the hook path before promotion, and re-scope the Phase 5
grep to cover the installed package as well as the consumer repo.

### 4.4 `count_fleet_live_sessions` fails open when `sessions_dir` is absent

The plan proposes adding warn-and-record for missing `repo_root`/`state_dir` — already
implemented (`fleet_registry.py:181-204`). The genuinely silent path is the bare `continue`
when `sessions_dir` is missing:

```python
# fleet_registry.py:206-209
sessions_dir = state_dir / "dispatches" / "sessions"
if not sessions_dir.exists():
    # No sessions dir means no live sessions for this repo
    continue                     # <-- no warning, no skip record
```

This contributes 0 *for that repo* and continues the loop — other repos still count, so it
is an undercount, not a fleet-wide zero. The over-dispatch hazard is proportional to the
affected repo's live worker count, and the Phase-3-to-Phase-5 window is exactly the shape
that triggers it. Both callers then discard the skip list anyway: `supervise.py:665`
(`live_count, _`) and `workflow.py:4993` (`_skipped_repos`), so nothing surfaces it.

Also: `touch_repo` writes `"state_dir": str(paths.root)` on every call
(`fleet_registry.py:139`) and runs from `build_app` (`cli.py:302`) and `run_doctor_command`
(`cli.py:314`) — so the registry self-heals on the next command, making the plan's manual
`fleet.json` edit largely redundant.

**Fix:** drop the manual `fleet.json` edit (or keep it as belt-and-braces, noting it is
redundant). File an issue for the silent `continue` and the two discarded skip lists — a
concurrency cap that silently undercounts is an over-dispatch hazard independent of this
migration.

### 4.5 `setup_worker.sh` and `finish_worker.sh` are uninvoked

Phase 4 ports both. `grep -rn "setup_worker\|finish_worker"` across job-cannon returns only
vendored mypy `setup_worker_manager` false positives inside
`.claude/worktrees/*/.venv/Lib/site-packages/mypy/`. `shell_command` invokes only
`launch_devin_worker.sh`.

They read `$MAIN/.var/devin-orchestrator/issues/issue-$ISSUE/worker-prompt.md`
(`setup_worker.sh:30`, `finish_worker.sh:10`) — a path Phase 3 relocates, which would be a
genuine phase-ordering defect *if* anything called them. Nothing does.

**Fix:** confirm-then-delete rather than port. Porting dead scripts carries their legacy
paths into the package for no benefit.

### 4.6 `create-pr` vs `create-PR`

On-disk directory is `create-pr`; the plan's disposition table writes `create-PR`. Invisible
on Windows, breaks on case-sensitive CI or any Linux consumer.

---

## What the plan gets right

Worth recording so the revision does not regress these:

- **SQLite WAL handling is thorough and correct.** `wal_checkpoint(TRUNCATE)` +
  `integrity_check` before backup, verify `-wal`/`-shm` gone, re-check after copy, compare
  event counts (plan lines 104-112, 194-196, 224, 329). Verified live: `events.db` is 3.29 MB
  with a 3.95 MB `-wal` in `journal_mode=wal` — the plan correctly identifies that the WAL
  exceeds the main DB and that a naive copy would lose it.
- **The "`.var/charlie-work/` is not stale, not deletable" correction** is right and is the
  single most important thing the first revision fixed.
- **Phase ordering is safe** — see §3.6; safer than the plan itself claims.
- **Refusing to put migration I/O in `runtime_paths()`** is correct. `paths.py`'s
  `runtime_paths()` is pure (join → `resolve()` → frozen dataclass) with `ensure()`
  deliberately separate; a migration side effect there would fire from every CLI entry point.
- **No `dispatches/` collision.** Verified: `.var/charlie-work/dispatches/reviews/` is
  **empty**, so Phase 3's copy of `dispatches/` lands clean. Similarly
  `.var/charlie-work/events.db` is 0 bytes with no `events` table — nothing to lose, though
  Phase 3 must overwrite rather than skip it (`Copy-Item` without `-Force` would leave the
  0-byte file and silently orphan 10,345 events).

---

## Recommended restructure

1. **Delete** the `package-data` step (§1.1) and the exec-bit gate (§3.2).
2. **Re-derive** the materializer-conflict strand from `injected_paths` + the shim's
   `cp -r` (§1.2).
3. **Add to Phase 0**: `.gitignore` migration for `/.var/` + `/.devin/`; back up
   `.git/info/exclude`; check `supervisor.lock` and `state.json.lock`; replace the
   `git stash` step with `Copy-Item`.
4. **Add to Phase 3**: the `state.json` prefix rewrite with its count assertion (§2.1);
   `cross-family/` in the copy list (§2.2); `-Force` on the `events.db` copy.
5. **Add to Phase 4**: the `config.py:1622-1640` placeholder-set change, or switch to a
   `console_scripts` entry point (§2.4); parameterize the hook paths before promotion
   (§4.3); confirm-then-delete the two dead scripts (§4.5).
6. **Rewrite** the three broken gates (§3.1, §3.3, §3.4) and make the `prompts_dir` config
   removal an explicit step (§3.5).
7. **File issues** for §4.1, §4.2, §4.4 and extend the existing "Separate issue" section.
8. **State the rollback truth plainly.** The plan frames rollback as "consumer revert first
   (`git reset --hard <jc-sha>`)". But `/.var/` and `/.devin/` are excluded, so
   `git reset --hard` restores *none* of what matters — state, worktrees, `.devin`, and
   `.git/info/exclude` all depend solely on the `Copy-Item` backups. An operator who reads
   the current wording will over-trust git.

One scheduling note: job-cannon's `worker_model` / `cross_family.model` were changed today
(`orchestrator.config.yaml:108,132`). Running the migration on the same day conflates two
variables if anything regresses — leave a day between them.
