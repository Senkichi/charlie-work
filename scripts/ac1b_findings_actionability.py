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
        scripts/ac1b_findings_actionability.py --repo /path/to/charlie-work

``--repo`` points at wherever ``.var/charlie-work/prs`` (the runtime corpus)
lives -- this is normally the long-running main checkout, not whatever
pinned worktree this script's own code came from. Those two are
independent: the CODE under test is whatever ``charlie_work`` package
``PYTHONPATH`` resolves to (its commit SHA is derived and printed
automatically, not hand-typed), the DATA is wherever ``--repo`` points.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import io
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
#: call/def-shaped token. The bare-call alternative is filtered post-match
#: (see ``_is_builtin_call``) to exclude Python builtins -- `print(`, `int(`,
#: `str(`, etc. constantly appear in ordinary prose and verification-command
#: snippets (e.g. `python -c "...; print(x)"`) without a reviewer naming any
#: project symbol a worker could act on. Backtick-quoted names are never
#: builtin-shaped (the pattern excludes `(`) so they are unaffected either way.
_CODE_SYMBOL_RE = re.compile(
    r"`[A-Za-z_][A-Za-z0-9_.]*`"
    r"|\b[A-Za-z_][A-Za-z0-9_]*\("
)

#: Every public Python builtin name (functions, types, exception classes,
#: constants) -- derived from the ``builtins`` module itself, not hand-typed,
#: so `open(`, `set(`, `len(`, etc. are excluded on the same footing as
#: `print(`/`int(`/`str(` without per-name upkeep as builtins are added.
_BUILTIN_NAMES = frozenset(name for name in dir(builtins) if not name.startswith("_"))


def _is_builtin_call(match_text: str) -> bool:
    """True if *match_text* is a bare call to a Python builtin, e.g. ``"print("``.

    Only applies to the un-quoted ``identifier(`` shape (no surrounding
    backticks are part of this match to begin with -- see ``_CODE_SYMBOL_RE``)
    so a genuine backtick-quoted reference is never affected by this check.
    """
    return match_text.endswith("(") and match_text[:-1] in _BUILTIN_NAMES


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
    """Return every ``(kind, matched substring)`` concrete referent in *text*.

    A ``code_symbol`` match is dropped when it is a bare call to a Python
    builtin (see ``_is_builtin_call``) -- e.g. the `print(` in a verification
    snippet's `python -c "import charlie_work; print(charlie_work.__file__)"`
    is not a reviewer naming a symbol to change, it is the language itself.
    """
    found: list[tuple[str, str]] = []
    for kind, pattern in _REFERENT_PATTERNS:
        for m in pattern.finditer(text):
            match_text = m.group(0)
            if kind == "code_symbol" and _is_builtin_call(match_text):
                continue
            found.append((kind, match_text))
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
    """Return the legacy generic-collapse fallback string, for classifying
    PRE-issue-#784 on-disk verdicts.

    This function used to derive the sentinel by calling the real parser
    with a crafted BLOCKER-only/no-``Verdict:``-marker probe (its
    docstring's own "F5", per ``docs/plans/rework-findings-channel.md``).
    Issue #784 IS that rewrite: ``cross_family.parse_cross_family_verdict``'s
    legacy fallback can no longer construct a content-free
    ``CrossFamilyVerdict`` for that exact shape at all --
    ``CrossFamilyVerdict.__post_init__`` now raises on it, and the parser
    catches that and returns a ``MalformedCrossFamilyVerdict`` instead.
    Deriving the historical literal by probing the live parser is therefore
    no longer possible; that capability is precisely what #784 removed, by
    design (unrepresentable content-free verdicts is the whole point of the
    fix).

    The literal itself survives as ``cross_family.LEGACY_VACUOUS_SUMMARY``,
    exported specifically so this script and
    ``workflow._is_carry_forward_eligible`` share one source of truth for
    recognizing the 8 pre-#784 broken records rather than each hardcoding a
    second copy. This function still probes the live parser with the same
    input as before -- not to derive the string, but to prove #784's fix is
    actually live in the code under test (not just merged elsewhere) before
    trusting the historical constant for classification.

    Raises RuntimeError (loudly, not silently) if the live parser's
    behavior for this probe is neither the expected post-#784 shape
    (``MalformedCrossFamilyVerdict``) nor recognizable at all -- so a future
    behavior change is caught, not silently misclassified as "real reviewer
    prose".
    """
    probe = "## Report\n\n**BLOCKER** unparseable body with no Verdict: marker\n"
    result = cross_family.parse_cross_family_verdict(probe)
    if not isinstance(result, cross_family.MalformedCrossFamilyVerdict):
        raise RuntimeError(
            "cross_family.parse_cross_family_verdict returned an unexpected "
            f"shape ({result!r}) for the generic-collapse sentinel probe. "
            "Expected MalformedCrossFamilyVerdict (issue #784's fix): a "
            "BLOCKER/MAJOR marker with no extractable summary must no "
            "longer construct a content-free CrossFamilyVerdict. Update "
            "this probe or investigate a regression -- do not ignore this "
            "error."
        )
    return cross_family.LEGACY_VACUOUS_SUMMARY


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
        sample_has_changes = bool(
            isinstance(sample_decision.get("required_changes"), list)
            and sample_decision["required_changes"]
        )
        if sample_has_changes:
            print(
                "  EXPECTED: the real renderer reads `required_changes` first, so\n"
                "  a summary-only mutation does not move the output when the\n"
                "  structured list is non-empty."
            )
        else:
            print(
                "  EXPECTED on this PRE-F1 checkout: _render_required_changes_section\n"
                '  early-returns "" whenever `required_changes` is empty, WITHOUT\n'
                "  ever consulting `summary` -- so a summary-only mutation cannot\n"
                "  move anything before F1 lands. This null result IS itself\n"
                "  evidence of the exact defect this harness measures (plan\n"
                "  section 2.1), not a broken check."
            )
    print()

    print(
        "=== Mutation check 2: mutate `required_changes` (the field the real\n"
        "    renderer reads first), through the REAL unmodified renderer ==="
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
# number. When the real renderer already implements F1's summary fallback,
# the projection is obsolete and is suppressed so the report does not show a
# "proj. post-F1 AC-1b" column alongside the real post-F1 numbers. On a
# pre-F1 checkout the local stand-in still runs and makes the "would the
# count move" evidence concrete.
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


def _renderer_has_f1_summary_fallback() -> bool:
    """Probe the REAL renderer for F1's summary fallback.

    Returns True when a ``request_changes`` verdict with an empty
    ``required_changes`` list but a non-empty ``summary`` renders a non-empty
    section. That is the exact contract ``project_f1_rendering`` simulates;
    when the real code already implements it, the projection column is
    obsolete and must not be printed next to the real post-F1 numbers.
    """
    probe: dict[str, Any] = {
        "decision": "request_changes",
        "required_changes": [],
        "summary": "Probe summary containing `src/charlie_work/workflow.py:1`.",
    }
    return bool(_render_required_changes_section(probe).strip())


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
    # `sys.stdout` is statically typed as `TextIO` (no `.reconfigure`); the
    # concrete runtime class that actually declares it is `io.TextIOWrapper`.
    # `isinstance` narrows to that concrete class so this is provably safe
    # under static analysis, not just at runtime -- and it still safely
    # no-ops when stdout has been replaced by something else (e.g. pytest's
    # capture writer, which is not a TextIOWrapper), matching the previous
    # `hasattr` guard's behavior exactly without the type-checker complaint.
    if isinstance(sys.stdout, io.TextIOWrapper):
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

    show_projection = not _renderer_has_f1_summary_fallback()

    print("=" * 78)
    print("AC-1b findings-actionability measurement harness")
    print("=" * 78)
    print(f"  code under test (charlie_work src) pinned at SHA: {code_sha or 'UNKNOWN'}")
    print(f"  corpus directory                                : {prs_dir}")
    print(
        f"  projection column                               : "
        f"{'suppressed (real F1 renderer in use)' if not show_projection else 'shown (pre-F1 renderer; local stand-in)'}"
    )
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

    # `sentinel` (Optional[str]) is the single source of truth for whether
    # derivation succeeded -- there is deliberately no separate boolean
    # flag alongside it. Two independently-assigned variables that must
    # stay in sync (a `str | None` plus a `bool` asserting whether it is
    # `None`) is exactly the kind of state a type checker cannot verify is
    # consistent, and a maintenance hazard if a future edit updates one
    # without the other. Testing `sentinel is not None` directly lets
    # static analysis (and readers) prove the classification call below is
    # never reached with `None`, not just trust that it happens to line up.
    try:
        sentinel: str | None = derive_cross_family_collapse_sentinel()
    except RuntimeError as exc:
        print(f"ERROR deriving cross-family sentinel: {exc}", file=sys.stderr)
        sentinel = None
    sentinel_ok = sentinel is not None

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
            if sentinel is not None
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

        projected_referents: list[tuple[str, str]] = []
        if show_projection:
            projected_rendered = project_f1_rendering(v.decision)
            projected_body = extract_scoreable_body(projected_rendered)
            projected_referents = find_concrete_referents(projected_body)
            if projected_referents:
                cat_stats.projected_ac1b_actionable += 1

        # Show referents from whichever pass actually found something, so a
        # reader can see WHERE a match came from (baseline vs. projection)
        # instead of a confusing "actionable=True, referents=[]" pairing.
        if referents:
            referent_source = "baseline"
            referent_pool = referents
        elif show_projection and projected_referents:
            referent_source = "projected"
            referent_pool = projected_referents
        else:
            referent_source = "baseline"
            referent_pool = []
        referent_preview = ", ".join(f"{k}:{s!r}" for k, s in referent_pool[:3])
        proj_part = f"proj_AC1b={bool(projected_referents)!s:<5}  " if show_projection else ""
        cat_stats.rows.append(
            f"    {v.pr_dir:>10}  len(summary)={len(summary):>5}  "
            f"AC1={ac1_pass!s:<5}  AC1b={ac1b_pass!s:<5}  "
            f"{proj_part}"
            f"referents[{referent_source}]=[{referent_preview}]"
        )

    print("=== Results BY CATEGORY (never a single aggregate) ===")
    header = f"{'category':<32} {'count':>6} {'AC-1 (non-empty)':>18} {'AC-1b (actionable)':>20}"
    if show_projection:
        header += f" {'proj. post-F1 AC-1b':>20}"
    print(header)
    total_all = total_ac1 = total_ac1b = total_proj = 0
    for category in (*CATEGORY_ORDER, "UNKNOWN_provenance_unavailable"):
        s = stats[category]
        if s.total == 0 and category == "UNKNOWN_provenance_unavailable":
            continue
        row = f"{category:<32} {s.total:>6} {s.ac1_non_empty:>18} {s.ac1b_actionable:>20}"
        if show_projection:
            row += f" {s.projected_ac1b_actionable:>20}"
        print(row)
        total_all += s.total
        total_ac1 += s.ac1_non_empty
        total_ac1b += s.ac1b_actionable
        total_proj += s.projected_ac1b_actionable
    total_row = (
        f"{'TOTAL (sum, not a substitute for the rows above)':<32} {total_all:>6} "
        f"{total_ac1:>18} {total_ac1b:>20}"
    )
    if show_projection:
        total_row += f" {total_proj:>20}"
    print(total_row)
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
    if show_projection:
        print(
            "  Baseline AC-1 / AC-1b are measured against the REAL production\n"
            "  renderer at the pinned SHA above. 'proj. post-F1 AC-1b' is a\n"
            "  diagnostic projection using a local stand-in for F1's contract\n"
            "  (NOT the real F1 code) -- re-run this unmodified script against\n"
            "  the real post-F1 checkout to get the authoritative post-fix number."
        )
    else:
        print(
            "  Baseline AC-1 / AC-1b are measured against the REAL production\n"
            "  renderer at the pinned SHA above. The diagnostic 'proj. post-F1\n"
            "  AC-1b' projection is suppressed because the real F1 renderer is\n"
            "  already in use."
        )
    return 0 if (self_test_ok and mutation_ok and sentinel_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
