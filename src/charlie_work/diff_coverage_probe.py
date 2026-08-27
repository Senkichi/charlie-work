"""Static diff-coverage / unwired-symbol probes (issues #1260/#1261).

Two mechanical, advisory-only checks that run inside ``OrchestratorApp.review()``
and inject findings into the review packet. Neither ever blocks dispatch,
review, or merge -- see ``config.CoverageProbeConfig``'s docstring. Promotion to
a hard gate is explicitly deferred past a 2-week false-positive measurement
window (the #1260/#1261 scoping comment); this module intentionally has no
auto-reject path.

- **W3** (#1260): a pure diff-text heuristic. Counts added "branch token" lines
  (``if ``, ``elif ``, ``except ``, ...) in non-test files against added
  assertion/test-function lines in test files, and flags a non-test file when
  its branch-adds outran the diff's test-adds.
- **W20 item 1** (#1261, mechanical half): an AST walk of newly-added
  functions/methods/classes in non-test files, flagging any new public symbol
  referenced only from ``tests/`` and nowhere in ``src/``. There is no PR-branch
  checkout at packet-assembly time (``create_review_checkout``'s only call site
  is the reviewer's later session, not ``review()`` itself), so candidates and
  same-diff wiring both come from the diff text; the operator's own
  ``repo_root`` main-branch checkout is consulted read-only, best-effort, to
  catch a consumer this diff never re-states.

Both halves are pure with respect to state (no writes) and never raise to
their caller: ``run_static_probe`` is the single entry point and the one place
that turns an internal exception into a rendered warning instead of letting it
propagate into ``review()`` -- mirrors ``janitor.check_test_adequacy``'s
contract. The W20 half does read files under ``repo_root`` (``check_operator_containment``'s
same read-only posture), which is exactly why that visible-warning contract
matters more here than in the pure-text W3 half.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from charlie_work.janitor import iter_diff_files

if TYPE_CHECKING:
    from charlie_work.config import CoverageProbeConfig

logger = logging.getLogger(__name__)

# Matches a def/async def/class statement's own added line, capturing the name.
_DEF_LINE_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class BranchCoverageFinding:
    """One non-test file whose added branch logic outran the diff's added tests."""

    filename: str
    branch_adds: int
    test_adds: int
    reason: str  # "no_test_adds" | "ratio_exceeded"


@dataclass(frozen=True)
class UnwiredSymbolFinding:
    """A new public symbol referenced only from tests/, never from src/."""

    symbol: str
    filename: str
    kind: str  # "function" | "class"


@dataclass(frozen=True)
class StaticProbeVerdict:
    """Combined W3 + W20-item-1 result for one PR diff.

    ``warnings`` covers internal-error visible-degradation text (design item 7
    of the #1260/#1261 scoping comment) -- never silently empty on error.
    """

    branch_findings: tuple[BranchCoverageFinding, ...] = ()
    unwired_findings: tuple[UnwiredSymbolFinding, ...] = ()
    warnings: tuple[str, ...] = ()


def _is_test_file(filename: str, config: CoverageProbeConfig) -> bool:
    return any(fnmatch.fnmatch(filename, glob) for glob in config.test_path_globs)


def _is_exempt_file(filename: str, config: CoverageProbeConfig) -> bool:
    return any(fnmatch.fnmatch(filename, glob) for glob in config.exempt_path_globs)


def _added_lines(hunk_lines: Sequence[str], config: CoverageProbeConfig) -> list[str]:
    """Non-blank, non-comment added lines from a file's hunks, '+' stripped."""
    out: list[str] = []
    for line in hunk_lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in config.comment_prefixes):
            continue
        out.append(line[1:])
    return out


def check_branch_coverage(
    diff: str, config: CoverageProbeConfig
) -> tuple[BranchCoverageFinding, ...]:
    """W3: branch-token-vs-test-add heuristic. Pure diff-text scan, no I/O.

    Reuses ``janitor.iter_diff_files`` for hunk splitting, which already
    treats a rename-only diff (no hunk body) as zero files -- a rename-only
    diff therefore never produces a finding without any special-casing here.
    """
    branch_by_file: dict[str, int] = {}
    test_adds_total = 0

    for filename, _is_new_file, hunk_lines in iter_diff_files(diff):
        if _is_exempt_file(filename, config):
            continue
        added = _added_lines(hunk_lines, config)
        if _is_test_file(filename, config):
            for line in added:
                if any(marker in line for marker in config.assertion_markers):
                    test_adds_total += 1
                elif config.test_function_prefix and line.lstrip().startswith(
                    config.test_function_prefix
                ):
                    test_adds_total += 1
            continue
        branch_adds = sum(
            1 for line in added if any(token in line for token in config.branch_tokens)
        )
        if branch_adds:
            branch_by_file[filename] = branch_adds

    total_branch_adds = sum(branch_by_file.values())
    if not total_branch_adds:
        return ()

    if test_adds_total == 0:
        return tuple(
            BranchCoverageFinding(filename, branch_adds, test_adds_total, "no_test_adds")
            for filename, branch_adds in branch_by_file.items()
        )

    ratio = total_branch_adds / test_adds_total
    if ratio > config.branch_to_assert_ratio_threshold:
        return tuple(
            BranchCoverageFinding(filename, branch_adds, test_adds_total, "ratio_exceeded")
            for filename, branch_adds in branch_by_file.items()
        )
    return ()


def _reconstruct_new_text(hunk_lines: Sequence[str]) -> str:
    """Rebuild one file's post-diff text from its hunks: keep context (' ')
    and added ('+') lines in order, drop removed ('-') lines and '@@' headers.

    This is a per-hunk reconstruction, not a full-file one -- a diff with
    partial hunks yields partial (but syntactically plausible, in the common
    case) text. Callers wrap ``ast.parse`` of the result in a try/except and
    treat a parse failure as "skip this file's AST pass", never as a
    probe-wide error.
    """
    out: list[str] = []
    for line in hunk_lines:
        if line.startswith("@@"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
        elif line.startswith(" "):
            out.append(line[1:])
        # '---', '+++', and bare '\' (no-newline marker) are metadata, dropped.
    return "\n".join(out)


def _new_top_level_defs(hunk_lines: Sequence[str]) -> list[tuple[str, str]]:
    """(name, kind) for every module- or class-level def/class whose own
    line was added by this diff (not merely present as unchanged context).
    """
    added_names: set[str] = set()
    for line in hunk_lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        match = _DEF_LINE_RE.match(line[1:])
        if match:
            added_names.add(match.group(1))
    if not added_names:
        return []

    try:
        tree = ast.parse(_reconstruct_new_text(hunk_lines))
    except SyntaxError:
        # Best-effort hunk reconstruction failed to parse (e.g. a hunk that
        # doesn't carry enough surrounding context to be self-contained).
        # Skip this file's AST pass rather than raising -- the diff-text-only
        # scan already ran for the rest of the probe.
        return []

    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in added_names:
            found.append((node.name, "function"))
        elif isinstance(node, ast.ClassDef) and node.name in added_names:
            found.append((node.name, "class"))
    return found


def _is_definition_line(line: str, symbol: str) -> bool:
    match = _DEF_LINE_RE.match(line)
    return bool(match and match.group(1) == symbol)


def _word_match(symbol: str, line: str) -> bool:
    return re.search(rf"\b{re.escape(symbol)}\b", line) is not None


def _collect_repo_referenced_names(repo_root: Path) -> tuple[set[str], tuple[str, ...]]:
    """AST-walk ``repo_root/src`` once, returning every Name/Attribute
    identifier seen plus the (repo-relative) paths of any file that failed
    to parse.

    Read-only, best-effort. A symbol collision with an unrelated existing
    name elsewhere in the tree is a known heuristic limitation (this probe
    is advisory-only and never blocks); a parse failure is reported back to
    the caller as a warning rather than silently narrowing the search.
    """
    names: set[str] = set()
    failures: list[str] = []
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return names, ()
    for path in sorted(src_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            try:
                failures.append(str(path.relative_to(repo_root)).replace("\\", "/"))
            except ValueError:
                failures.append(str(path))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names, tuple(failures)


def check_unwired_symbols(
    diff: str, repo_root: Path, config: CoverageProbeConfig
) -> tuple[tuple[UnwiredSymbolFinding, ...], tuple[str, ...]]:
    """W20 item 1: flag a new public function/class whose only reference
    anywhere is from a test file.

    Returns ``(findings, warnings)``. ``warnings`` surfaces per-file AST-parse
    failures hit while walking ``repo_root`` -- a broken consumer search must
    read as visible degradation, never as silent-clean.
    """
    candidates: list[tuple[str, str, str]] = []  # (symbol, filename, kind)
    for filename, _is_new_file, hunk_lines in iter_diff_files(diff):
        if _is_test_file(filename, config) or _is_exempt_file(filename, config):
            continue
        for name, kind in _new_top_level_defs(hunk_lines):
            if name.startswith(config.private_name_prefix):
                continue
            candidates.append((name, filename, kind))

    if not candidates:
        return (), ()

    # Same-diff wiring: does any OTHER added line anywhere in this diff
    # (test or product) reference the candidate symbol by name?
    src_refs: set[str] = set()
    test_refs: set[str] = set()
    for filename, _is_new_file, hunk_lines in iter_diff_files(diff):
        is_test = _is_test_file(filename, config)
        for line in _added_lines(hunk_lines, config):
            for symbol, def_filename, _kind in candidates:
                if def_filename == filename and _is_definition_line(line, symbol):
                    continue
                if _word_match(symbol, line):
                    (test_refs if is_test else src_refs).add(symbol)

    repo_names, warnings = _collect_repo_referenced_names(repo_root)

    findings: list[UnwiredSymbolFinding] = []
    for symbol, filename, kind in candidates:
        if symbol in src_refs or symbol in repo_names:
            continue
        if symbol not in test_refs:
            # Referenced nowhere at all (not even tests) is out of this
            # probe's stated scope ("referenced only from tests/").
            continue
        findings.append(UnwiredSymbolFinding(symbol, filename, kind))

    return tuple(findings), warnings


def run_static_probe(
    diff: str, repo_root: Path, config: CoverageProbeConfig
) -> StaticProbeVerdict:
    """Compute both probe halves for one PR diff. Never raises -- any
    internal exception becomes a rendered warning instead of propagating to
    ``review()`` (mirrors ``janitor.check_test_adequacy``'s never-raises
    contract).
    """
    warnings: list[str] = []
    branch_findings: tuple[BranchCoverageFinding, ...] = ()
    unwired_findings: tuple[UnwiredSymbolFinding, ...] = ()

    try:
        branch_findings = check_branch_coverage(diff, config)
    except Exception as exc:
        logger.warning("static probe: branch-coverage heuristic failed: %s", exc)
        warnings.append(f"static probe degraded: branch-coverage heuristic failed: {exc}")

    if config.check_unwired_symbols:
        try:
            unwired_findings, unwired_warnings = check_unwired_symbols(diff, repo_root, config)
            warnings.extend(unwired_warnings)
        except Exception as exc:
            logger.warning("static probe: unwired-symbol probe failed: %s", exc)
            warnings.append(f"static probe degraded: unwired-symbol probe failed: {exc}")

    return StaticProbeVerdict(branch_findings, unwired_findings, tuple(warnings))
