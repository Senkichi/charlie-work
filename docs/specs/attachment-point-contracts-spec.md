# Attachment-Point Contracts — pilot implementation spec

Authority: `llibrary/docs/plans/2026-08-24-god-object-mitigation-DECISION.md` (operator-approved
2026-08-24). Design: `llibrary/raw/analyses/2026-08-god-object/design-candidates/design-contract-boundary.md`.
Stopgap #1442 remains live and concurrent; this system is its successor.

**Binding constraint: no line count is ever read anywhere in this package.** The unit of
measure is bound-member count per attachment point, gated by an archetype-relative outlier
test against the repo's own distribution. No constant threshold appears anywhere.

## Package layout (many small files; ruff line-length 99; Python >= 3.11)

```
src/charlie_work/attachment_contracts/
  __init__.py      # re-export public API only
  model.py         # frozen dataclasses (below)
  archetypes.py    # AST archetype detection -> list[AttachmentPoint]
  ledger.py        # linear-ledger detection (structural, no allowlist)
  excludes.py      # G3 exclude-set
  outliers.py      # saturation via Q3 + 1.5*IQR, small-sample floor
  baseline.py      # freeze-on-adopt, ratchet, tamper/reason-token guard
  redirect.py      # sibling-destination naming + G2 pre-wired scaffold
  check.py         # check_file / check_tree orchestration (hook + CI entries)
  hook_entry.py    # PreToolUse stdin protocol, interactive-vs-unattended split
  backtest.py      # G1 run positive control over git history
  __main__.py      # CLI: scan | baseline | check-file | check-tree | backtest
tests/attachment_contracts/
  test_archetypes.py test_ledger.py test_excludes.py
  test_outliers.py test_baseline.py
  test_redirect.py test_check.py test_hook_entry.py
.github/workflows/attachment-contracts.yml   # Week-1 SHADOW: report-only
```

NEVER touch `src/charlie_work/cli.py`, `workflow.py`, or `tests/test_charlie_work.py` —
entry is `python -m charlie_work.attachment_contracts`, tests live in their own directory.
(Appending this system to the monoliths it polices would be self-refuting.)

## model.py (shared contract — all builders code against this exactly)

```python
Kind = Literal["typer_app", "click_group", "blueprint", "class", "migration_runner", "test_module"]

@dataclass(frozen=True)
class AttachmentPoint:
    kind: Kind
    identity: str          # e.g. "OrchestratorApp", "cli:app", "tests/test_x.py::module"
    file: str              # repo-relative posix path
    members: tuple[str, ...]
    is_linear_ledger: bool = False
    @property
    def member_count(self) -> int: ...

@dataclass(frozen=True)
class ScanResult:
    root: str
    points: tuple[AttachmentPoint, ...]
    parse_failures: tuple[str, ...]   # G6: files that failed AST parse — NEVER dropped

@dataclass(frozen=True)
class SaturationVerdict:
    point: AttachmentPoint
    saturated: bool
    q3: float; iqr: float; boundary: float
    population: int                   # same-kind APs in repo; < floor -> never saturated

@dataclass(frozen=True)
class Finding:
    severity: Literal["block", "advise", "error"]   # error = G6 parse-failure/tamper
    file: str
    identity: str
    message: str                      # includes the redirect text
    redirect: str | None              # suggested destination path or scaffold summary
```

## Archetype detection (archetypes.py)

Pure function `scan_source(text: str, path: str) -> list[AttachmentPoint]` plus
`scan_tree(root: Path, excludes) -> ScanResult` (walks `src/` and `tests/`, honors excludes).
Detection is structural, derived, never listed:

- **typer_app / click_group**: `Assign` whose value is a call to `typer.Typer(...)` /
  `click.Group(...)` (or `@click.group` decorated fn). Members = functions decorated
  `@<name>.command(...)` / `@<name>.command`.
- **blueprint**: `Assign` value calling `Blueprint(...)` (flask/quart import). Members =
  functions decorated `@<name>.route(...)` / method-verb decorators (`get/post/...`).
- **class**: every `ClassDef`; members = its direct `FunctionDef`/`AsyncFunctionDef`
  children (dunders `__init__` etc. INCLUDED — they are responsibilities too).
- **migration_runner**: module or class whose member family matches a monotonic numbered
  pattern (see ledger.py); the AP is emitted with `is_linear_ledger=True` when contiguous.
- **test_module**: any `test_*.py`/`*_test.py` under `tests/`; identity is the module;
  members = top-level `test_*` functions + `Test*` classes (a Test class counts as ONE
  member; its methods are members of its own separate `class` AP).

A file may emit multiple APs. A new file with a new `app = typer.Typer()` is itself a new
AP entering the same distribution (cannot dodge by adding files).

## Ledger detection (ledger.py) — structural exemption, no allowlist

`classify_ledger(members: Sequence[str]) -> bool`: True iff >= 3 members match a common
prefix + integer suffix pattern (`_migrate_v(\d+)`, `_apply_m(\d+)`, generally
`^(?P<prefix>[A-Za-z_]+?)(?P<n>\d+)$` with one dominant prefix covering >= 80% of members)
AND the integers are strictly increasing and contiguous-modulo-gaps<=1. Ledger APs are
exempt from the member ratchet but still tracked; a NON-matching member on a ledger AP
breaks the pattern-dominance and the AP loses its exemption (caught).

## Excludes (excludes.py) — G3

`load_excludes(root) -> Excludes` combining:
1. The ONE sanctioned config list: `[tool.attachment-contracts] exclude_globs` in
   pyproject.toml (tree globs only; default empty). Bounded, auditable config — not code.
2. `.git-blame-ignore-revs` SHAs (for backtest replay: skip those commits).
3. Codemod-shape heuristic (backtest only): a commit touching > 20 files is skipped as a
   bulk reformat candidate.
Plus always-on structural excludes: `.venv`, `.var`, `node_modules`, `__pycache__`,
`.claude/worktrees`, any dir named `generated` or `vendor`.

## Saturation (outliers.py)

```
saturate(points, kind) per repo:
  counts = [p.member_count for p in same-kind, non-ledger points]
  if len(counts) < FLOOR (=4): nothing of this kind is saturated (population too small)
  q1, q3 = nearest-rank quartiles; iqr = q3 - q1
  boundary = q3 + 1.5 * iqr
  saturated(p) = p.member_count > boundary
```
Deterministic nearest-rank quartiles (no interpolation ambiguity across platforms).
FLOOR is a named module constant with the rationale in a docstring — it is a
statistical-validity floor (outlier tests are meaningless at n<4), not a size threshold.

## Baseline (baseline.py) — freeze-on-adopt + ratchet + tamper guard

`.attachment-budgets.json` at repo root, GENERATED only (`baseline` CLI cmd), schema:
```json
{"version": 1, "generated_by": "charlie_work.attachment_contracts <pkg-version>",
 "generated_at": "<iso8601>", "floor": 4,
 "entries": [{"kind": "class", "identity": "OrchestratorApp",
              "file": "src/charlie_work/workflow.py", "member_count": 134,
              "boundary": 12.5, "bumps": []}]}
```
Entries = saturated points only, sorted (kind, file, identity) for stable diffs.
- `compare(current_scan, baseline)`: a saturated point above its baselined member_count
  without a bump entry -> Finding(block). A point now BELOW baseline -> ratchet down
  (rewrite entry lower; CLI `baseline --ratchet` does this; CI verifies monotonic-down).
- Bump schema: `{"to": N, "reason": str, "actor": "interactive"|"worker", "ack": str}`.
  **G4:** `actor=worker` REQUIRES a non-empty `ack` referencing an external source
  (issue URL / dispatch-prompt id / human handle). Worker bump without external ack ->
  Finding(error). Interactive bumps self-ack.
- Tamper guard: `check-tree` recomputes what the baseline SHOULD contain for unchanged
  points; a baseline entry raised without a bump record -> Finding(error).

## Redirect + scaffold (redirect.py) — G2

`suggest(point, scan) -> Redirect`: nearest non-saturated same-kind sibling (fewest
members, same package dir preferred), else propose a new module name derived from the
member being added (`<verb>_ops.py` for class methods, `tests/<topic>/test_<topic>.py`
for tests). `scaffold(redirect, member_name) -> ScaffoldPlan{path, content}`: renders the
sibling file carrying the source file's relevant imports, the registration wiring
(`app.add_typer` / blueprint registration / class skeleton / fixture imports for tests),
so the agent writes only the member body. Scaffold WRITES nothing itself; it returns the
plan (the hook prints it; callers decide).

## check.py — the two entry points

- `check_file(path, root) -> list[Finding]`: single-file VIEW over a full-tree scan
  (delegates to `check_tree` and filters to `path`), compare against committed baseline.
  Used by the hook. **Not sub-second on a repo this package's own size (~3s
  measured)**: a single point's saturation verdict depends on its whole archetype
  distribution's outlier boundary, so classifying one file correctly requires scanning
  the tree, not just that file. This is a deliberate parity tradeoff (hook and CI share
  exactly one comparison algorithm, so they can never diverge in behavior) traded
  against hook latency; round-1 review (finding #6) confirmed the tradeoff is
  real and this line is the amendment closing that gap rather than leaving spec and
  behavior in conflict. Parse failure -> Finding(error) (G6 — fail TOWARD CI, never
  silently pass).
- `check_tree(root) -> list[Finding]`: full scan + baseline compare + tamper guard +
  G6 (any parse_failures -> error findings).
Exit codes for `__main__`: 0 clean, 1 findings-with-error/block (CI fail), 0 with
report when `--report-only` (Week-1 shadow mode).

## hook_entry.py — PreToolUse protocol

stdin JSON: `{"tool_name": "Write|Edit|MultiEdit", "tool_input": {"file_path": ...}}`.
- No `.attachment-budgets.json` found walking up from target -> exit 0 silently (fast
  no-op outside piloted repos).
- Unattended detection: env `CHARLIE_FLEET_WORKER=1` (the fleet dispatch env) OR
  `CLAUDE_CODE_UNATTENDED=1` -> ALWAYS advisory (never exit 2) — print redirect JSON
  `{"hookSpecificOutput": {"additionalContext": ...}}`, append a marker line to
  `.var/attachment-contracts/advisories.jsonl` (best-effort).
- Interactive + mode=enforce -> exit 2 with redirect message on stderr.
- Mode source: `ATTACHMENT_CONTRACTS_MODE` env, else `mode` key in the baseline file
  (default `"advise"`). **Week 1 ships with `advise`. Week 2 flips to `enforce`.**

## CI (.github/workflows/attachment-contracts.yml) — Week-1 SHADOW

Job on pull_request + push to main: `uv run python -m charlie_work.attachment_contracts
check-tree --report-only --github-annotations`. `--report-only` => always exit 0 in
Week 1; the Week-2 flip is deleting that flag (leave a `# WEEK-2:` comment showing the
enforce line). Findings surface as PR annotations + step summary.

## backtest.py — G1 (Deliverable 0, gates the pilot)

`backtest --repo <path> --months 6`: sample the first commit of each month on main +
the anchor SHAs `1ead858`, `7373d47`, `9de0b9f`; for each, `git worktree add --detach`
a temp dir, run scan_tree + saturate, record which APs are saturated, then remove the
temp worktree (via `git worktree remove`, NEVER rm -rf). PASS criteria (hard gate):
1. `OrchestratorApp` (src/charlie_work/workflow.py) saturated at every sampled point
   where it exists;
2. `tests/test_charlie_work.py` test_module AP saturated; `tests/test_worktree.py`
   saturated at the anchors;
3. ZERO of the 13 counterexample modules produce a saturated AP at any sample:
   prompt_sections.py, event_kinds.py, safe_path.py, file_lock.py, markdown_fence.py,
   closing_keyword_gate.py, dirty_tree.py, safe_ref.py, git_pull_blockers.py,
   throttle_signatures.py, fleet_paths.py, logging_setup.py, rescue.py;
4. Cluster-B score (informational, reported not gated): whether any bare-function
   module registers via the `class`/module archetypes — expected partial miss, report it.
Output: `docs/plans/attachment-contracts-backtest-report.md` + JSON next to it.

## Conventions

- `uv run` (NO `--active` — this is a worktree with its own venv), `pytest -q --tb=short`.
- Immutability: frozen dataclasses, no in-place mutation; return new objects.
- Type hints on all signatures; f-strings; no hand-maintained module lists anywhere
  (grep-test: the only path literals allowed are the structural excludes + test fixtures).
- Every module 200-400 lines target, 800 hard max.
