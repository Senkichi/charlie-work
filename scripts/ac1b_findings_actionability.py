"""AC-1b findings-actionability measurement harness.

Background: docs/plans/rework-findings-channel.md (untracked planning doc,
main checkout only -- read it there, not in this worktree, before touching
this script).

AC-1 (structural) asks only "is the rendered findings section non-empty?".
The plan explicitly calls AC-1-alone a FALSE GREEN (section 8): it passes
mechanically for cross-family verdicts whose entire content is a
content-free placeholder string. AC-1b (actionability) is the discriminator
this script measures: does the rendered section name at least one concrete
referent (file path, code symbol, or line reference) a worker could act on?

This script:
  1. Discovers the on-disk `request_changes` verdict corpus dynamically (no
     hardcoded PR list) under ``<repo>/.var/charlie-work/prs/pr-*/``.
  2. Renders each one through the REAL production renderer
     (``charlie_work.workflow._render_required_changes_section``) -- it does
     not reimplement any rendering logic.
  3. Classifies each verdict into one of three categories (derived from
     verdict content/provenance, not a hardcoded PR-to-category map): cross-
     family generic collapse, synthetic CI-failure, real reviewer prose.
  4. Scores actionability per verdict using an explicit, documented
     definition of "concrete referent".
  5. Reports BY CATEGORY, never as a single aggregate (docs/plans/
     rework-findings-channel.md section 8, AC-1b: "A single aggregate number
     here is not an acceptable report").
  6. Is re-runnable with no baked-in baseline numbers, so it can be re-run
     after the fixes (F1, F5) land to produce "after" numbers against the
     same corpus.

Mandated invocation (see docs/plans/rework-findings-channel.md and the
worktree-discipline memory on inherited VIRTUAL_ENV silently testing the
wrong checkout):

    VIRTUAL_ENV= PYTHONPATH="$PWD/src" uv run --no-sync python \
        scripts/ac1b_findings_actionability.py --repo C:/Users/senki/repos/charlie-work

``--repo`` points at wherever ``.var/charlie-work/prs`` (the runtime corpus)
lives -- this is normally the long-running main checkout, not whatever
pinned worktree this script's own code came from. Those two are
independent: the CODE under test is whatever ``charlie_work`` package
``PYTHONPATH`` resolves to (its commit SHA is derived and printed
automatically, not hand-typed), the DATA is wherever ``--repo`` points.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Fallback import path so this also works when invoked without PYTHONPATH
# pre-set (e.g. `python scripts/ac1b_findings_actionability.py` directly from
# a checkout). The mandated invocation sets PYTHONPATH explicitly; this is
# just a convenience net, not the primary mechanism.
_FALLBACK_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(_FALLBACK_SRC))

from charlie_work import cross_family  # noqa: E402
from charlie_work.global_config import load_layered_config  # noqa: E402
from charlie_work.paths import RepoNotFoundError, find_repo_root, runtime_paths  # noqa: E402
from charlie_work.workflow import _render_required_changes_section  # noqa: E402

# --------------------------------------------------------------------------
# Actionability detector -- the single definition of "concrete referent".
#
# Per docs/plans/rework-findings-channel.md section 8 (AC-1b): "a file path,
# a symbol name, or a line reference". Three independent regex classes, each
# documented so a reader can audit exactly what counts without re-deriving
# it from behavior.
# --------------------------------------------------------------------------

_SOURCE_EXTENSIONS = (
    "py",
    "md",
    "json",
    "yml",
    "yaml",
    "toml",
    "ps1",
    "sh",
    "js",
    "ts",
    "tsx",
    "jsx",
    "cfg",
    "ini",
    "txt",
    "html",
    "css",
    "sql",
)
_EXT_ALTERNATION = "|".join(_SOURCE_EXTENSIONS)

#: A file path: either a `dir/dir/file.ext` shape, or a bare `file.ext`
#: filename carrying one of the known source/config extensions.
_FILE_PATH_RE = re.compile(
    r"(?:[\w.\-]+/)+[\w.\-]+\.(?:" + _EXT_ALTERNATION + r")\b"
    r"|\b[\w\-]+\.(?:" + _EXT_ALTERNATION + r")\b"
)

#: A code symbol: a backtick-quoted identifier, or an `identifier(`
#: call/def-shaped token.
_CODE_SYMBOL_RE = re.compile(
    r"`[A-Za-z_][A-Za-z0-9_.]*`"
    r"|\b[A-Za-z_][A-Za-z0-9_]*\("
)

#: A line-number reference: `:123`, `line 123`, `L123`. Known false-positive
#: risk (timestamps, ratios) is accepted per the task's explicit definition;
#: not tightened beyond spec.
_LINE_NUMBER_RE = re.compile(
    r":\d{1,6}\b"
    r"|\bline\s+\d+\b"
    r"|\bL\d+\b"
)

_REFERENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("file_path", _FILE_PATH_RE),
    ("code_symbol", _CODE_SYMBOL_RE),
    ("line_number", _LINE_NUMBER_RE),
)


def find_concrete_referents(text: str) -> list[tuple[str, str]]:
    """Return every ``(kind, matched substring)`` concrete referent in *text*."""
    found: list[tuple[str, str]] = []
    for kind, pattern in _REFERENT_PATTERNS:
        for m in pattern.finditer(text):
            found.append((kind, m.group(0)))
    return found


def is_actionable(text: str) -> bool:
    return bool(find_concrete_referents(text))


# --------------------------------------------------------------------------
# Scoring the reviewer's words, not the renderer's own chrome.
#
# F1's heading/lead-in wording is explicit implementer discretion (plan
# section 6/F1: "Exact wording is implementer discretion... Do not treat a
# differing choice of heading as a review defect"). If that wording happens
# to contain a backtick or a path, every verdict would score actionable
# regardless of reviewer content -- the exact false-green failure mode this
# harness exists to catch, one layer up. Score only the reviewer-authored
# body: either the fenced ```...``` block (F1's mandated fallback shape) or
# the bullet list items (the structured-list shape), falling back to
# non-heading lines only when neither recognized shape is present.
# --------------------------------------------------------------------------

_FENCED_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
_BULLET_LINE_RE = re.compile(r"^-\s+(.*)$", re.MULTILINE)


def extract_scoreable_body(rendered_section: str) -> str:
    """Return only the reviewer-authored content of a rendered section."""
    fenced = _FENCED_BLOCK_RE.findall(rendered_section)
    if fenced:
        return "\n".join(fenced)
    bullets = _BULLET_LINE_RE.findall(rendered_section)
    if bullets:
        return "\n".join(bullets)
    return "\n".join(
        line for line in rendered_section.splitlines() if not line.strip().startswith("#")
    )


# --------------------------------------------------------------------------
# Category classification -- derived from verdict content/provenance via the
# REAL parsers, not a hardcoded PR-to-category map.
# --------------------------------------------------------------------------

CROSS_FAMILY_COLLAPSE = "cross_family_generic_collapse"
SYNTHETIC_CI_FAILURE = "synthetic_ci_failure"
REAL_REVIEWER_PROSE = "real_reviewer_prose"
CATEGORY_ORDER = (CROSS_FAMILY_COLLAPSE, SYNTHETIC_CI_FAILURE, REAL_REVIEWER_PROSE)


def derive_cross_family_collapse_sentinel() -> str:
    """Derive the generic-collapse fallback string via the REAL parser.

    ``cross_family.parse_cross_family_verdict`` (src/charlie_work/
    cross_family.py:316-356) returns the literal constant "Cross-family
    review found BLOCKER/MAJOR findings" as its ``summary`` whenever its
    ``_VERDICT_RE`` fails to find a ``Verdict:`` marker while a
    BLOCKER/MAJOR severity marker is present. Deriving it here by calling
    the real function with a crafted probe -- rather than hardcoding a
    second copy of the literal -- means this classifier cannot silently
    drift from the parser it is measuring.

    Raises RuntimeError (loudly, not silently) if the parser's return shape
    or behavior no longer matches what this probe expects. F5 (plan section
    6) rewrites this exact function; if F5 changes the contract, this probe
    must be re-validated by a human, not silently misclassify every
    cross-family verdict as "real reviewer prose".
    """
    probe = "## Report\n\n**BLOCKER** unparseable body with no Verdict: marker\n"
    result = cross_family.parse_cross_family_verdict(probe)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise RuntimeError(
            "cross_family.parse_cross_family_verdict returned an unexpected "
            f"shape ({result!r}) for the generic-collapse sentinel probe. "
            "The parser's contract has likely changed (see F5 in docs/plans/"
            "rework-findings-channel.md) -- update this probe, do not ignore "
            "this error."
        )
    decision, summary = result
    if decision != "request_changes" or "BLOCKER" not in summary.upper():
        raise RuntimeError(
            f"cross_family.parse_cross_family_verdict returned {result!r} "
            "for a synthetic BLOCKER-only probe with no Verdict: marker -- "
            "this does not look like the known generic-collapse fallback. "
            "Refusing to use it as a classification sentinel."
        )
    return summary


#: Mirrors the f-string template at src/charlie_work/workflow.py:7231:
#:     summary = f"CI failed on {', '.join(verdict.failed_required_checks)}; push a fix"
#: Matched structurally (the failed-check list varies), not as a duplicated
#: literal -- an exact-literal match would silently miss verdicts naming a
#: different failed check.
_SYNTHETIC_CI_FAILURE_RE = re.compile(r"^CI failed on .+; push a fix$")


def classify_verdict(summary: str, sentinel: str) -> str:
    stripped = (summary or "").strip()
    if stripped == sentinel:
        return CROSS_FAMILY_COLLAPSE
    if _SYNTHETIC_CI_FAILURE_RE.match(stripped):
        return SYNTHETIC_CI_FAILURE
    return REAL_REVIEWER_PROSE


# --------------------------------------------------------------------------
# Corpus discovery -- dynamic glob, no hardcoded PR list.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    pr_dir: str
    path: Path
    decision: dict[str, Any]


def discover_request_changes_verdicts(prs_dir: Path) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for path in sorted(prs_dir.glob("pr-*/review-decision.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or data.get("decision") != "request_changes":
            continue
        verdicts.append(Verdict(pr_dir=path.parent.name, path=path, decision=data))
    return verdicts


# --------------------------------------------------------------------------
# Self-test: positive/negative controls on the DETECTOR, independent of the
# renderer. Required to run and print before the main table -- an empty or
# uniform result later is not evidence of anything until this shows the
# detector could have gone the other way.
# --------------------------------------------------------------------------


def run_self_test() -> bool:
    positive = "See src/charlie_work/workflow.py:3678 in `_render_required_changes_section`."
    negative = "Cross-family review found BLOCKER/MAJOR findings"
    pos_referents = find_concrete_referents(positive)
    neg_referents = find_concrete_referents(negative)
    ok = bool(pos_referents) and not neg_referents

    print("=== Self-test: actionability detector controls ===")
    print(f"  positive control text : {positive!r}")
    print(f"    referents found     : {pos_referents}")
    print(f"    actionable          : {bool(pos_referents)}  (expected True)")
    print(f"  negative control text : {negative!r}")
    print(f"    referents found     : {neg_referents}")
    print(f"    actionable          : {bool(neg_referents)}  (expected False)")
    print(f"  SELF-TEST: {'PASSED' if ok else 'FAILED'}")
    print()
    return ok


# --------------------------------------------------------------------------
# Mutation checks: prove the harness's number would move if the underlying
# renderer output changed, using the REAL production renderer (never a
# reimplementation) on a copy of one real verdict.
# --------------------------------------------------------------------------


def run_mutation_checks(sample_decision: dict[str, Any]) -> bool:
    print("=== Mutation check 1: monkeypatch `summary` only (as literally specified) ===")
    before = _render_required_changes_section(sample_decision)
    mutated_summary = copy.deepcopy(sample_decision)
    mutated_summary["summary"] = "See src/charlie_work/workflow.py:3678 for details."
    after = _render_required_changes_section(mutated_summary)
    moved = before != after
    print(f"  rendered before                : {before!r}")
    print(f"  rendered after (summary edited) : {after!r}")
    print(f"  count moved                     : {moved}")
    if not moved:
        print(
            "  EXPECTED on this PRE-F1 baseline: _render_required_changes_section\n"
            '  (workflow.py:3678) early-returns "" whenever `required_changes` is\n'
            "  empty, WITHOUT ever consulting `summary` -- so a summary-only\n"
            "  mutation cannot move anything before F1 lands. This null result IS\n"
            "  itself evidence of the exact defect this harness measures (plan\n"
            "  section 2.1), not a broken check."
        )
    print()

    print(
        "=== Mutation check 2: mutate `required_changes` (the field the PRE-F1\n"
        "    renderer actually reads), through the REAL unmodified renderer ==="
    )
    mutated_rc = copy.deepcopy(sample_decision)
    mutated_rc["required_changes"] = ["Fix the null check in src/charlie_work/workflow.py:3700"]
    rendered2 = _render_required_changes_section(mutated_rc)
    body2 = extract_scoreable_body(rendered2)
    referents2 = find_concrete_referents(body2)
    actionable2 = bool(referents2)
    print(f"  rendered section : {rendered2!r}")
    print(f"  referents found  : {referents2}")
    print(f"  actionable       : {actionable2}  (expected True)")
    check2_ok = actionable2
    if not check2_ok:
        print(
            "  MUTATION CHECK 2 FAILED -- harness did not flag an obviously actionable referent."
        )
    print()
    return check2_ok


# --------------------------------------------------------------------------
# Post-F1 projection -- diagnostic only, NOT the authoritative AC-1/AC-1b
# number. F1 has not merged at the pinned baseline SHA this script's `src/`
# was imported from, so there is no real post-F1 renderer to call yet. This
# local stand-in exists solely to make the "would the count move" evidence
# concrete and to preview the CI-failure-category conflict (see report).
# Re-run this SAME script (unmodified) against the real post-F1 code once
# it lands -- do not treat this projection as a substitute for that re-run.
# --------------------------------------------------------------------------


def project_f1_rendering(decision: dict[str, Any]) -> str:
    """Local stand-in for F1's contract (plan section 6/F1): fence the
    verdict `summary` as the fallback body when `required_changes` is empty
    but `summary` is non-empty. NOT the real renderer -- diagnostic use only.
    """
    if decision.get("decision") != "request_changes":
        return ""
    required_changes = decision.get("required_changes")
    if isinstance(required_changes, list) and required_changes:
        lines = ["## Required changes", ""]
        for change in required_changes:
            text = str(change).strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)
    summary = (decision.get("summary") or "").strip()
    if not summary:
        return ""
    return "## Required changes (fallback: reviewer summary)\n\n```md\n" + summary + "\n```\n"


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------


@dataclass
class CategoryStats:
    total: int = 0
    ac1_non_empty: int = 0
    ac1b_actionable: int = 0
    projected_ac1b_actionable: int = 0
    rows: list[str] = field(default_factory=list)


def resolve_code_sha() -> str | None:
    """`git rev-parse HEAD` for wherever the imported charlie_work package
    lives -- derived from the actual code executing, not hand-typed, so the
    printed SHA cannot drift from what actually ran.
    """
    import charlie_work

    pkg_dir = Path(charlie_work.__file__).resolve().parent
    code_root = pkg_dir.parent.parent  # charlie_work/ -> src/ -> repo root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=code_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help=(
            "Repo whose .var/<state_dir>/prs holds the review-decision.json "
            "corpus (normally the long-running main checkout, not this "
            "script's own worktree -- .var is gitignored runtime state)."
        ),
    )
    args = parser.parse_args()

    try:
        data_repo_root = find_repo_root(args.repo, explicit=args.repo is not None)
    except RepoNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    config = load_layered_config(data_repo_root)
    prs_dir = runtime_paths(data_repo_root, config.runtime.state_dir).prs
    code_sha = resolve_code_sha()

    print("=" * 78)
    print("AC-1b findings-actionability measurement harness")
    print("=" * 78)
    print(f"  code under test (charlie_work src) pinned at SHA: {code_sha or 'UNKNOWN'}")
    print(f"  corpus directory                                : {prs_dir}")
    print()

    self_test_ok = run_self_test()

    verdicts = discover_request_changes_verdicts(prs_dir)
    print("=== Corpus discovery ===")
    print(f"  request_changes verdicts found: {len(verdicts)}")
    if not verdicts:
        print(
            "  ERROR: zero verdicts found. Per verification discipline, an empty\n"
            "  result is a claim about the query, not the world -- this is NOT\n"
            "  being reported as '0/0 pass'. Check --repo and the corpus path\n"
            "  printed above before trusting any report from this run."
        )
        return 1
    print()

    mutation_ok = run_mutation_checks(verdicts[0].decision)

    try:
        sentinel = derive_cross_family_collapse_sentinel()
        sentinel_ok = True
    except RuntimeError as exc:
        print(f"ERROR deriving cross-family sentinel: {exc}", file=sys.stderr)
        sentinel = None
        sentinel_ok = False

    print("=== Sentinel derivation ===")
    print(f"  cross-family generic-collapse sentinel: {sentinel!r}")
    print(f"  derivation OK: {sentinel_ok}")
    print()

    stats: dict[str, CategoryStats] = {c: CategoryStats() for c in CATEGORY_ORDER}
    stats["UNKNOWN_provenance_unavailable"] = CategoryStats()

    for v in verdicts:
        summary = v.decision.get("summary") or ""
        category = (
            classify_verdict(summary, sentinel)
            if sentinel_ok
            else "UNKNOWN_provenance_unavailable"
        )
        cat_stats = stats[category]
        cat_stats.total += 1

        rendered = _render_required_changes_section(v.decision)
        ac1_pass = bool(rendered.strip())
        if ac1_pass:
            cat_stats.ac1_non_empty += 1

        body = extract_scoreable_body(rendered)
        referents = find_concrete_referents(body)
        ac1b_pass = bool(referents)
        if ac1b_pass:
            cat_stats.ac1b_actionable += 1

        projected_rendered = project_f1_rendering(v.decision)
        projected_body = extract_scoreable_body(projected_rendered)
        projected_referents = find_concrete_referents(projected_body)
        if projected_referents:
            cat_stats.projected_ac1b_actionable += 1

        # Show referents from whichever pass actually found something, so a
        # reader can see WHERE a match came from (baseline vs. projection)
        # instead of a confusing "actionable=True, referents=[]" pairing.
        referent_source = "baseline" if referents else "projected"
        referent_pool = referents if referents else projected_referents
        referent_preview = ", ".join(f"{k}:{s!r}" for k, s in referent_pool[:3])
        cat_stats.rows.append(
            f"    {v.pr_dir:>10}  len(summary)={len(summary):>5}  "
            f"AC1={ac1_pass!s:<5}  AC1b={ac1b_pass!s:<5}  "
            f"proj_AC1b={bool(projected_referents)!s:<5}  "
            f"referents[{referent_source}]=[{referent_preview}]"
        )

    print("=== Results BY CATEGORY (never a single aggregate) ===")
    print(
        f"{'category':<32} {'count':>6} {'AC-1 (non-empty)':>18} "
        f"{'AC-1b (actionable)':>20} {'proj. post-F1 AC-1b':>20}"
    )
    total_all = total_ac1 = total_ac1b = total_proj = 0
    for category in (*CATEGORY_ORDER, "UNKNOWN_provenance_unavailable"):
        s = stats[category]
        if s.total == 0 and category == "UNKNOWN_provenance_unavailable":
            continue
        print(
            f"{category:<32} {s.total:>6} {s.ac1_non_empty:>18} "
            f"{s.ac1b_actionable:>20} {s.projected_ac1b_actionable:>20}"
        )
        total_all += s.total
        total_ac1 += s.ac1_non_empty
        total_ac1b += s.ac1b_actionable
        total_proj += s.projected_ac1b_actionable
    print(
        f"{'TOTAL (sum, not a substitute for the rows above)':<32} {total_all:>6} "
        f"{total_ac1:>18} {total_ac1b:>20} {total_proj:>20}"
    )
    print()

    print("=== Per-verdict detail ===")
    for category in (*CATEGORY_ORDER, "UNKNOWN_provenance_unavailable"):
        s = stats[category]
        if s.total == 0:
            continue
        print(f"  -- {category} ({s.total}) --")
        for row in s.rows:
            print(row)
    print()

    print("=== Summary ===")
    print(f"  self-test passed        : {self_test_ok}")
    print(f"  mutation check 2 passed : {mutation_ok}")
    print(f"  sentinel derivation OK  : {sentinel_ok}")
    print(f"  corpus size             : {total_all}")
    print(
        "  Baseline AC-1 / AC-1b are measured against the REAL production\n"
        "  renderer at the pinned SHA above. 'proj. post-F1 AC-1b' is a\n"
        "  diagnostic projection using a local stand-in for F1's contract\n"
        "  (NOT the real F1 code) -- re-run this unmodified script against\n"
        "  the real post-F1 checkout to get the authoritative post-fix number."
    )
    return 0 if (self_test_ok and mutation_ok and sentinel_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
