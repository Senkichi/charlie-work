"""G1 (Deliverable 0): positive-control backtest over the repo's own git history.

Two halves, deliberately separated:

- Pure logic (`select_samples`, `evaluate`, report formatting) takes plain
  data in and plain data out — no git subprocess calls — so it is unit
  testable without a git checkout at all.
- Orchestration (`run_backtest`) is the only part that shells out to git. It
  samples commits, materializes each one into a disposable detached
  `git worktree add`, runs scan_tree + saturation against it, and always
  removes the worktree via `git worktree remove --force` — never `rm -rf`
  (a worktree can hold a junction/reparse point back into the real checkout;
  a recursive delete would follow it).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.model import AttachmentPoint, Kind, ScanResult
from charlie_work.attachment_contracts.outliers import saturate

# The three explicit anchor SHAs named in the spec (Deliverable 0, G1).
ANCHOR_SHAS: tuple[str, ...] = ("1ead858", "7373d47", "9de0b9f")

ORCHESTRATOR_FILE = "src/charlie_work/workflow.py"
ORCHESTRATOR_IDENTITY = "OrchestratorApp"
TEST_CHARLIE_WORK_FILE = "tests/test_charlie_work.py"
TEST_WORKTREE_FILE = "tests/test_worktree.py"

# The 13 counterexample modules named in the spec: small, legitimately-scoped
# modules that must NEVER produce a saturated attachment point. A false
# positive on any of these means the outlier test is miscalibrated.
COUNTEREXAMPLE_MODULES: tuple[str, ...] = (
    "prompt_sections.py",
    "event_kinds.py",
    "safe_path.py",
    "file_lock.py",
    "markdown_fence.py",
    "closing_keyword_gate.py",
    "dirty_tree.py",
    "safe_ref.py",
    "git_pull_blockers.py",
    "throttle_signatures.py",
    "fleet_paths.py",
    "logging_setup.py",
    "rescue.py",
)


# ---------------------------------------------------------------------------
# Pure data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitRef:
    """One sampled commit: enough identity to run a backtest sample."""

    sha: str
    date: str  # ISO "YYYY-MM-DD", commit author date
    label: str  # e.g. "2026-03" (first-of-month) or "anchor"


@dataclass(frozen=True)
class SampleResult:
    """The outcome of scanning + saturating one sampled commit's checkout."""

    ref: CommitRef
    points: tuple[AttachmentPoint, ...]
    saturated_keys: frozenset[tuple[Kind, str, str]]  # (kind, file, identity)
    parse_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BacktestVerdict:
    samples: tuple[SampleResult, ...]
    criteria: tuple[CriterionResult, ...]
    cluster_b_score: int  # informational only, never gates pass/fail
    cluster_b_detail: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)


# ---------------------------------------------------------------------------
# Pure sampling
# ---------------------------------------------------------------------------


def select_samples(
    commits: tuple[CommitRef, ...],
    months: int,
    anchor_shas: tuple[str, ...] = ANCHOR_SHAS,
) -> tuple[CommitRef, ...]:
    """Pick the first commit of each of the most recent `months` calendar
    months present in `commits`, plus the explicit anchor commits.

    `commits` may be given in any order and need not be pre-filtered; this
    function derives the (year, month) grouping itself from each ref's date.
    Anchors are matched by SHA prefix (git short-SHA semantics) against
    `commits` and always included, even if their month was not otherwise
    selected — they are explicit, not derived. Deduplicated by sha; result is
    sorted chronologically by date.
    """
    by_month: dict[str, CommitRef] = {}
    for ref in sorted(commits, key=lambda r: r.date):
        month_key = ref.date[:7]  # "YYYY-MM"
        if month_key not in by_month:
            by_month[month_key] = ref  # first (earliest) commit of that month

    recent_months = sorted(by_month.keys())[-months:] if months > 0 else []
    selected: dict[str, CommitRef] = {
        by_month[m].sha: CommitRef(sha=by_month[m].sha, date=by_month[m].date, label=m)
        for m in recent_months
    }

    for anchor in anchor_shas:
        match = next((r for r in commits if r.sha.startswith(anchor)), None)
        if match is not None and match.sha not in selected:
            selected[match.sha] = CommitRef(sha=match.sha, date=match.date, label="anchor")

    return tuple(sorted(selected.values(), key=lambda r: r.date))


# ---------------------------------------------------------------------------
# Pure criteria evaluation
# ---------------------------------------------------------------------------


def _find_point(sample: SampleResult, file: str, identity: str) -> AttachmentPoint | None:
    return next((p for p in sample.points if p.file == file and p.identity == identity), None)


def _is_saturated(sample: SampleResult, point: AttachmentPoint) -> bool:
    return (point.kind, point.file, point.identity) in sample.saturated_keys


def _criterion_orchestrator(samples: tuple[SampleResult, ...]) -> CriterionResult:
    misses: list[str] = []
    present_count = 0
    for sample in samples:
        point = _find_point(sample, ORCHESTRATOR_FILE, ORCHESTRATOR_IDENTITY)
        if point is None:
            continue
        present_count += 1
        if not _is_saturated(sample, point):
            misses.append(sample.ref.sha)
    if present_count == 0:
        return CriterionResult(
            name="orchestrator_saturated",
            passed=False,
            detail=f"{ORCHESTRATOR_IDENTITY} not found in any sample; cannot validate",
        )
    passed = not misses
    detail = (
        f"saturated at all {present_count} samples where present"
        if passed
        else f"NOT saturated at: {', '.join(misses)}"
    )
    return CriterionResult(name="orchestrator_saturated", passed=passed, detail=detail)


def _find_test_module_point(sample: SampleResult, file: str) -> AttachmentPoint | None:
    return next((p for p in sample.points if p.kind == "test_module" and p.file == file), None)


def _criterion_test_charlie_work(samples: tuple[SampleResult, ...]) -> CriterionResult:
    misses: list[str] = []
    present_count = 0
    for sample in samples:
        point = _find_test_module_point(sample, TEST_CHARLIE_WORK_FILE)
        if point is None:
            continue
        present_count += 1
        if not _is_saturated(sample, point):
            misses.append(sample.ref.sha)
    if present_count == 0:
        return CriterionResult(
            name="test_charlie_work_saturated",
            passed=False,
            detail=f"{TEST_CHARLIE_WORK_FILE} test_module not found in any sample",
        )
    passed = not misses
    detail = (
        f"saturated at all {present_count} samples where present"
        if passed
        else f"NOT saturated at: {', '.join(misses)}"
    )
    return CriterionResult(name="test_charlie_work_saturated", passed=passed, detail=detail)


def _criterion_test_worktree_at_anchors(
    samples: tuple[SampleResult, ...], anchor_shas: tuple[str, ...]
) -> CriterionResult:
    anchor_samples = [s for s in samples if any(s.ref.sha.startswith(a) for a in anchor_shas)]
    misses: list[str] = []
    present_count = 0
    for sample in anchor_samples:
        point = _find_test_module_point(sample, TEST_WORKTREE_FILE)
        if point is None:
            misses.append(f"{sample.ref.sha} (not found)")
            continue
        present_count += 1
        if not _is_saturated(sample, point):
            misses.append(sample.ref.sha)
    passed = not misses and present_count > 0
    detail = (
        f"saturated at all {present_count} anchor samples"
        if passed
        else f"failed at: {', '.join(misses) if misses else '(no anchors sampled)'}"
    )
    return CriterionResult(name="test_worktree_saturated_at_anchors", passed=passed, detail=detail)


def _criterion_counterexamples_clean(samples: tuple[SampleResult, ...]) -> CriterionResult:
    hits: list[str] = []
    for sample in samples:
        for point in sample.points:
            module_name = Path(point.file).name
            if module_name in COUNTEREXAMPLE_MODULES and _is_saturated(sample, point):
                hits.append(f"{sample.ref.sha}:{point.file}:{point.identity}")
    passed = not hits
    detail = "zero false-positive saturations" if passed else f"false positives: {', '.join(hits)}"
    return CriterionResult(name="counterexamples_clean", passed=passed, detail=detail)


def _cluster_b_informational(samples: tuple[SampleResult, ...]) -> tuple[int, str]:
    """Informational only (reported, never gates pass/fail).

    Counts, per sample, bare-function (module-level) files whose responsibility
    surface is captured only as a `migration_runner` module-ledger AP or not
    captured by any class/typer/blueprint archetype at all — i.e. cases where
    a bare-function module's growth would register via the `class`/module
    archetypes only partially, which is the expected, documented gap.
    """
    hits = 0
    examples: list[str] = []
    for sample in samples:
        files_with_class_or_router_ap = {
            p.file
            for p in sample.points
            if p.kind in ("class", "typer_app", "click_group", "blueprint")
        }
        for point in sample.points:
            if point.kind == "migration_runner" and point.identity.startswith("module:"):
                if point.file not in files_with_class_or_router_ap:
                    hits += 1
                    examples.append(f"{sample.ref.sha}:{point.file}")
    detail = (
        f"{hits} bare-function module(s) registering only via module-ledger archetype "
        f"(expected partial miss)"
    )
    if examples:
        detail += f": {', '.join(examples[:10])}"
    return hits, detail


def evaluate(
    samples: tuple[SampleResult, ...], anchor_shas: tuple[str, ...] = ANCHOR_SHAS
) -> BacktestVerdict:
    """Evaluate the PASS criteria (spec section `backtest.py`) over `samples`."""
    criteria = (
        _criterion_orchestrator(samples),
        _criterion_test_charlie_work(samples),
        _criterion_test_worktree_at_anchors(samples, anchor_shas),
        _criterion_counterexamples_clean(samples),
    )
    cluster_b_score, cluster_b_detail = _cluster_b_informational(samples)
    return BacktestVerdict(
        samples=samples,
        criteria=criteria,
        cluster_b_score=cluster_b_score,
        cluster_b_detail=cluster_b_detail,
    )


# ---------------------------------------------------------------------------
# Report formatting (pure)
# ---------------------------------------------------------------------------


def render_markdown(verdict: BacktestVerdict) -> str:
    lines = ["# Attachment-Point Contracts backtest report", ""]
    lines.append(f"**Overall: {'PASS' if verdict.passed else 'FAIL'}**")
    lines.append("")
    lines.append(f"Samples: {len(verdict.samples)}")
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    for c in verdict.criteria:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"- [{mark}] `{c.name}` — {c.detail}")
    lines.append("")
    lines.append("## Cluster-B score (informational, not gated)")
    lines.append("")
    lines.append(verdict.cluster_b_detail)
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    for s in verdict.samples:
        lines.append(
            f"- `{s.ref.sha}` ({s.ref.label}, {s.ref.date}): "
            f"{len(s.points)} points, {len(s.saturated_keys)} saturated, "
            f"{len(s.parse_failures)} parse failures"
        )
    return "\n".join(lines) + "\n"


def render_json(verdict: BacktestVerdict) -> str:
    document = {
        "passed": verdict.passed,
        "criteria": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in verdict.criteria
        ],
        "cluster_b_score": verdict.cluster_b_score,
        "cluster_b_detail": verdict.cluster_b_detail,
        "samples": [
            {
                "sha": s.ref.sha,
                "label": s.ref.label,
                "date": s.ref.date,
                "point_count": len(s.points),
                "saturated_count": len(s.saturated_keys),
                "saturated": sorted(f"{k}:{f}:{i}" for k, f, i in s.saturated_keys),
                "parse_failures": list(s.parse_failures),
            }
            for s in verdict.samples
        ],
    }
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Orchestration (git subprocess) — the only impure part of this module
# ---------------------------------------------------------------------------


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def load_commit_log(repo_path: Path, branch: str = "main") -> tuple[CommitRef, ...]:
    """`git log` on `branch`, one CommitRef per commit (full sha + author date)."""
    output = _run_git(repo_path, "log", "--pretty=format:%H|%ad", "--date=short", branch)
    refs: list[CommitRef] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        sha, _, date = line.partition("|")
        refs.append(CommitRef(sha=sha, date=date, label=""))
    return tuple(refs)


def _scan_sample(repo_root: Path, ref: CommitRef) -> SampleResult:
    excludes = load_excludes(repo_root)
    # Imported lazily to avoid a hard import-time dependency cycle between
    # backtest.py and archetypes.py during concurrent development.
    from charlie_work.attachment_contracts.archetypes import scan_tree

    scan: ScanResult = scan_tree(repo_root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    saturated_keys: set[tuple[Kind, str, str]] = set()
    for kind in kinds:
        for verdict in saturate(scan.points, kind):
            if verdict.saturated:
                saturated_keys.add(
                    (verdict.point.kind, verdict.point.file, verdict.point.identity)
                )
    return SampleResult(
        ref=ref,
        points=scan.points,
        saturated_keys=frozenset(saturated_keys),
        parse_failures=scan.parse_failures,
    )


def _worktree_add(repo_path: Path, worktree_dir: Path, sha: str) -> None:
    _run_git(repo_path, "worktree", "add", "--detach", str(worktree_dir), sha)


def _worktree_remove(repo_path: Path, worktree_dir: Path) -> None:
    # NEVER rm -rf a worktree directory: it can contain a junction/reparse
    # point back into the real checkout (e.g. a .venv junction), and a
    # recursive delete would follow it. `git worktree remove` is the only
    # sanctioned removal path.
    _run_git(repo_path, "worktree", "remove", "--force", str(worktree_dir))


def run_backtest(
    repo_path: Path,
    months: int = 6,
    anchor_shas: tuple[str, ...] = ANCHOR_SHAS,
    branch: str = "main",
) -> BacktestVerdict:
    """Full orchestration: sample commits, materialize each into a disposable
    detached worktree under `repo_path`'s parent, scan + saturate, tear down.
    """
    commits = load_commit_log(repo_path, branch=branch)
    samples_to_run = select_samples(commits, months, anchor_shas)

    results: list[SampleResult] = []
    parent_dir = repo_path.resolve().parent
    with tempfile.TemporaryDirectory(dir=str(parent_dir), prefix=".apc-backtest-") as tmp_root:
        for ref in samples_to_run:
            worktree_dir = Path(tmp_root) / ref.sha
            _worktree_add(repo_path, worktree_dir, ref.sha)
            try:
                results.append(_scan_sample(worktree_dir, ref))
            finally:
                _worktree_remove(repo_path, worktree_dir)

    return evaluate(tuple(results), anchor_shas)


def write_report(verdict: BacktestVerdict, out_dir: Path) -> tuple[Path, Path]:
    """Write the markdown + JSON report next to each other; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "attachment-contracts-backtest-report.md"
    json_path = out_dir / "attachment-contracts-backtest-report.json"
    md_path.write_text(render_markdown(verdict), encoding="utf-8")
    json_path.write_text(render_json(verdict), encoding="utf-8")
    return md_path, json_path
