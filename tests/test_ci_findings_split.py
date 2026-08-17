"""Seam integrity for the workflow.py -> ci_findings.py split (#1283 Phase A, PR 5/~6).

``ci_findings.py`` holds the CI-checks-findings free-function family -- the
review-packet CI-status section renderer, the dispatch-staleness detector,
and the required-check-annotation-to-required-changes converter --
verbatim-moved out of ``workflow.py``. ``workflow.py`` re-exports every
moved name through a facade import block (mirroring ``config.py``'s
``RunnerAllocationConfig`` pattern and this repo's own
``dispatch_selection.py``/``escalation.py``/``verdict_parsing.py``/
``rework_prompts.py`` precedents) so every existing
``charlie_work.workflow.<name>`` import path and monkeypatch target keeps
resolving unchanged.

Unlike every prior Phase-A family, these 8 names are NOT one call-graph-
connected cluster -- they are three mutually disconnected sub-clusters,
joined only by issue #1283's own binding text and a shared destination
theme (workflow-side consumers of CI check data), not by call edges:

* ``_ci_status_section`` -> ``_non_required_check_findings`` (imports
  ``summarize_checks``, ``_classify_check_run``, ``_CheckClassification``
  from ``.checks``).
* ``check_dispatch_staleness`` -> {``_parse_iso_ts``,
  ``_backlog_is_non_empty``, ``_latest_non_empty_dispatch``} (imports
  ``query_events`` from ``.instrumentation`` and ``DispatchConfig`` from
  ``.config`` -- nothing from ``.checks``).
* ``_required_changes_from_checks`` -> ``_annotation_to_required_change``
  (imports ``_is_failing_run`` from ``.checks``).

This disclosure mirrors ``ci_findings.py``'s own module docstring rather than
smoothing the grouping over as natural cohesion -- it is a judgment call
bound by the issue's own recorded clustering.

This file mirrors ``tests/test_rework_prompts_split.py`` (the current
precedent for the FIXED 3-branch ``_module_imports_in`` helper -- issue
#1300 documents the gap in the three older suites this one does not
inherit) structurally, with deliberate departures documented at the point
they occur:

* The write/event-emission surface re-confirmation (AC8/AC9) asserts an
  EMPTY result -- zero hits, the opposite of ``rework_prompts.py``'s two
  known ``_write_text_atomic`` calls. Because a zero-hits result is
  unfalsifiable on its own (verification-ladder: empty result vs. broken
  query are indistinguishable without a control), the positive control
  below is load-bearing, not decoration: it re-runs the identical scanner
  over ``workflow.py`` and asserts it finds every one of the write/event
  call names it is known to contain.
* The AC5 vacuity/mutation control (operator decision #3, binding) targets
  a FUNCTION, not a constant-branch claim: ``ci_findings.py`` has zero
  top-level constants (all 8 module-level names are ``FunctionDef``), so a
  ``len(names) == 8`` assertion proves membership/drift only, never that
  ``_module_level_defined_names`` still walks the ``Assign``/``AnnAssign``
  branch. That branch is kept in the shared helper below (so a constant
  added to ``ci_findings.py`` later is not silently invisible to AC4/AC5),
  but no test in this file claims the constant branch is exercised. The
  actual mutation/revert cycle (commenting out ``check_dispatch_staleness``'s
  facade re-export line, confirming both the identity and completeness
  tests fail closed naming the symbol and its real consumer at
  ``tests/test_dispatch_staleness.py:21``, then restoring) was performed as
  a one-off verification during this PR and is documented in the commit
  body -- not shipped as a permanent test, since a real edit to
  ``workflow.py``'s facade block is not something a test file can safely
  perform on itself at collection time.
* Only 4 of the 8 names (``_annotation_to_required_change``,
  ``_required_changes_from_checks``, ``_non_required_check_findings``,
  ``check_dispatch_staleness``) have any live consumer reference under
  ``tests/``/``scripts``/``src`` at all -- confirmed by direct scan, not
  merely cited from Preflight. The other 4 (``_ci_status_section``,
  ``_backlog_is_non_empty``, ``_latest_non_empty_dispatch``,
  ``_parse_iso_ts``) are exercised only behaviorally through
  ``OrchestratorApp.review()``/``OrchestratorApp._dispatch_impl()``, so the
  AC5 completeness scan finds no anchor for them. They are still required
  to be re-exported by the unconditional facade-obligation rule (AC4
  covers that), just not because the live scan demands it -- mirrors the
  same shape ``test_rework_prompts_split.py`` documents for its own
  8-of-14 split.

Three ways the facade promise can quietly break, none of which fails
loudly on its own (same three failure modes every prior split suite
documents):

* ``ci_findings.py`` could grow an import of ``workflow.py`` -- workflow.py
  already imports ``ci_findings.py`` for the facade, so that would be a
  real import cycle.
* The facade could re-declare a name instead of importing it (a copy-paste
  that silently duplicates a function). Both copies look correct in
  isolation; only identity distinguishes them.
* The facade's import list could fall out of sync with what real consumers
  (tests, scripts) actually reach through ``charlie_work.workflow``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_CI_FINDINGS_PATH = _REPO_ROOT / "src" / "charlie_work" / "ci_findings.py"
_WORKFLOW_PATH = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"

_MOVED_NAMES = (
    "_ci_status_section",
    "_non_required_check_findings",
    "_backlog_is_non_empty",
    "_latest_non_empty_dispatch",
    "_parse_iso_ts",
    "check_dispatch_staleness",
    "_annotation_to_required_change",
    "_required_changes_from_checks",
)


# ---------------------------------------------------------------------------
# Shared derivation helpers -- both AC4 and AC5 draw their name universe from
# the actual module content, never from a hand-typed list. A hand-typed list
# is exactly the kind of thing that drifts the moment either file is next
# edited without updating this test.
# ---------------------------------------------------------------------------


def _module_level_defined_names(path: Path) -> list[str]:
    """Top-level function/class/constant names a module defines.

    Module-level ``Assign``/``AnnAssign`` with a simple ``Name`` target are
    treated as constants. Dunders are skipped. This is the module's own
    public surface, read straight off its AST -- not restated by hand
    anywhere in this file.

    ``ci_findings.py`` happens to have zero constants at the top level (all
    8 moved names are functions), but the ``Assign``/``AnnAssign`` branches
    below are kept regardless: dropping them would make a constant added to
    ``ci_findings.py`` later silently invisible to both the AC4 identity
    test and the AC5 completeness test below, which both draw their
    candidate set from this helper.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("__"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("__"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("__"):
                names.append(node.target.id)
    return names


def _facade_reexported_names(workflow_path: Path) -> set[str]:
    """Names workflow.py's facade block currently re-exports from ``.ci_findings``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "ci_findings":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# AC3: import-cycle guard
# ---------------------------------------------------------------------------


def _module_imports_in(
    source: str,
    *,
    relative_module: str,
    absolute_module: str,
    filename: str = "<string>",
) -> list[str]:
    """AST-derived list of any import of the given module.

    AST-based rather than a substring grep so a prose mention of the module
    name in a comment or docstring (this file's own module docstring has
    several) can never false-positive the check, and so the check isn't
    fooled by formatting a plain grep might not anticipate (``as`` aliases,
    multi-line ``from ... import (...)`` blocks).

    Covers four distinct import spellings that all resolve to the same
    module at runtime: ``from .<relative_module> import X`` (module-qualified
    relative), ``from . import <relative_module>`` (bare package-relative --
    parses as ``ImportFrom(module=None, level=1, names=[alias(relative_module)])``,
    a shape the module-qualified branch below does not match since its own
    ``node.module`` is ``None``, not ``relative_module``), ``from
    <absolute_module> import X``, and ``import <absolute_module>``. This is
    the FIXED 3-branch ``ImportFrom`` handling ``test_rework_prompts_split.py``
    introduced -- the three older split suites
    (``test_dispatch_selection_split.py``, ``test_escalation_split.py``,
    ``test_verdict_parsing_split.py``) lack the middle (bare package-relative)
    branch, a gap issue #1300 documents. Copied here verbatim rather than
    inherited from those older suites.
    """
    tree = ast.parse(source, filename=filename)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module == relative_module:
                offenders.append(f"line {node.lineno}: from .{relative_module} import ...")
            elif node.level == 1 and node.module is None:
                for alias in node.names:
                    if alias.name == relative_module:
                        offenders.append(f"line {node.lineno}: from . import {relative_module}")
            elif node.module == absolute_module:
                offenders.append(f"line {node.lineno}: from {absolute_module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == absolute_module:
                    offenders.append(f"line {node.lineno}: import {absolute_module}")
    return offenders


def _workflow_imports_in(source: str, *, filename: str = "<string>") -> list[str]:
    """AST-derived list of any import of ``workflow``/``charlie_work.workflow``."""
    return _module_imports_in(
        source,
        relative_module="workflow",
        absolute_module="charlie_work.workflow",
        filename=filename,
    )


def test_ci_findings_has_no_workflow_import() -> None:
    """AC3: ci_findings.py must never import from workflow.py.

    workflow.py's facade imports FROM ci_findings.py; the reverse import
    would be the exact cycle the issue's own Traps section warns against
    (import charlie_work.ci_findings would then transitively require
    charlie_work.workflow to already be fully initialized, and vice versa).
    """
    offenders = _workflow_imports_in(
        _CI_FINDINGS_PATH.read_text(encoding="utf-8"),
        filename=str(_CI_FINDINGS_PATH),
    )
    assert offenders == [], (
        "ci_findings.py imports from charlie_work.workflow -- this creates an "
        f"import cycle with workflow.py's facade block: {offenders}"
    )


def test_workflow_import_detector_flags_a_real_violation() -> None:
    """Control for the AST detector above -- proves it can actually fire.

    Without this, a detector that had quietly become incapable of finding
    anything (e.g. a typo'd node-type check) would leave the assertion above
    vacuously true forever.

    Includes the bare ``from . import workflow`` package-relative spelling
    (``ImportFrom(module=None, level=1, names=[alias("workflow")])``)
    alongside the module-qualified relative, absolute-``from``, and plain
    ``import`` spellings -- omitting that form would mean the detector could
    fail open on it while this control still passed.
    """
    relative_violation = "from .workflow import OrchestratorApp\n"
    relative_package_violation = "from . import workflow\n"
    absolute_violation = "import charlie_work.workflow\n"
    absolute_from_violation = "from charlie_work.workflow import OrchestratorApp\n"
    innocent = (
        '"""A docstring that merely mentions workflow.py and charlie_work.workflow.foo."""\n'
    )

    assert _workflow_imports_in(relative_violation) != []
    assert _workflow_imports_in(relative_package_violation) != []
    assert _workflow_imports_in(absolute_violation) != []
    assert _workflow_imports_in(absolute_from_violation) != []
    assert _workflow_imports_in(innocent) == [], "prose mention must not be flagged"


def test_ci_findings_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not error.

    The AST check above is necessary but not sufficient -- it only rules out
    one specific kind of cycle (an explicit import of workflow.py). This
    drives the real interpreter through the module's actual import
    statements, which also catches any other cyclical dependency the module
    might have picked up (e.g. through ``.checks``, ``.instrumentation``, or
    ``.config``).

    Kept for precedent-parity even though this family has no module-load-time
    side effect analogous to ``verdict_parsing.py``'s
    ``_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS = OrchestratorConfig()``
    construction -- Preflight's fresh AST re-derivation (restricted to
    exactly the 3 fragment line ranges) found zero top-level
    ``Assign``/``AnnAssign`` among the 8 moved units, only the 8
    ``FunctionDef`` nodes themselves. This test still earns its place: it
    drives the interpreter through the real import statements (``.checks``,
    ``.instrumentation``, ``.config``), which the AST check above does not
    exercise, and would catch any cycle picked up through one of those three
    modules that the workflow-only check can't see.
    """
    import importlib

    module = importlib.import_module("charlie_work.ci_findings")
    assert module.__name__ == "charlie_work.ci_findings"

    # Confirms the module body actually executed to completion (not merely
    # "importlib didn't raise") by checking real symbols landed in the
    # module's namespace with the expected type -- one from each of the
    # three disconnected sub-clusters.
    assert callable(module._ci_status_section)
    assert callable(module.check_dispatch_staleness)
    assert callable(module._required_changes_from_checks)


# ---------------------------------------------------------------------------
# AC4: seam-guard identity checks (mirrors test_rework_prompts_split.py's pattern)
# ---------------------------------------------------------------------------


def test_all_ci_findings_names_are_reexported_by_identity() -> None:
    """AC4: every name ci_findings.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of a
    function would compare unequal-but-structurally-similar in ways that are
    easy to miss.

    The ``len(names) == 8`` assertion below is a membership/drift guard
    only -- unlike the analogous assertion in
    ``test_rework_prompts_split.py`` (which additionally proves
    ``_module_level_defined_names`` still walks the ``Assign`` branch, since
    that family has 3 constants), ``ci_findings.py`` has zero top-level
    constants, so this count cannot and does not claim to exercise that
    branch. It still catches drift: a future edit that added or removed a
    moved unit without updating this test would change ``len(names)`` and
    fail loudly here.
    """
    import charlie_work.ci_findings as ci_findings
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_CI_FINDINGS_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 8, f"expected 8 moved units, found {len(names)}: {sorted(names)}"
    assert set(names) == set(_MOVED_NAMES), (
        f"AST-derived names {sorted(names)} do not match the expected moved set "
        f"{sorted(_MOVED_NAMES)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    not_identical = [n for n in names if getattr(workflow, n) is not getattr(ci_findings, n)]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"ci_findings.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above (mirrors
    test_rework_prompts_split.py's own control, itself mirroring
    test_ci_fleet_seams.py::test_identity_check_would_fail_on_a_redefined_class).

    Two structurally-identical-but-distinct objects are not ``is``-equal --
    proves the check discriminates, rather than passing for any two things
    that merely look alike.
    """
    from dataclasses import dataclass

    def helper_a() -> int:
        return 1

    def helper_b() -> int:
        return 1

    assert helper_a is not helper_b

    @dataclass(frozen=True)
    class ResultA:
        value: int = 1

    @dataclass(frozen=True)
    class ResultB:
        value: int = 1

    assert ResultA is not ResultB
    # dataclass __eq__ checks `other.__class__ is self.__class__` by default,
    # so two independently-declared dataclasses with identical fields are
    # not equal either -- there is no value-equality shortcut that would let
    # a redeclaration slip past a caller who does `isinstance`/`is` checks.
    assert ResultA(value=1) != ResultB(value=1)
    assert not isinstance(ResultA(value=1), ResultB)


# ---------------------------------------------------------------------------
# AC5: re-export completeness, derived from live consumer references
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """POSIX-style path for messages/comparisons, repo-relative when possible.

    Falls back to the absolute path when ``path`` isn't under the repo (the
    scan's own control tests point it at a ``tmp_path`` fixture directory).
    Always forward-slashed so assertions comparing against a literal string
    don't have to special-case Windows' backslash separators.
    """
    try:
        display = path.relative_to(_REPO_ROOT)
    except ValueError:
        display = path
    return display.as_posix()


def _consumer_referenced_names(candidates: set[str], search_roots: list[Path]) -> dict[str, str]:
    """name -> one file that reaches it through ``charlie_work.workflow``.

    Walks every ``.py`` file under ``search_roots`` for three reference forms
    (confirmed to be the only forms in use for this family by a direct scan
    during preflight):

    1. ``from charlie_work.workflow import <name>`` / ``from .workflow import
       <name>`` (AST ``ImportFrom``, so a multi-line parenthesized import
       block is caught the same as a single-line one).
    2. ``workflow.<name>`` / ``workflow_module.<name>`` / ``wf.<name>``
       attribute access (regex, word-bounded so e.g.
       ``check_dispatch_staleness`` cannot partially match a longer name).
    3. The string-dotted monkeypatch form
       ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)`` -- matched
       only when the token is quote-delimited, so a prose/docstring
       cross-reference like ```` ``charlie_work.workflow._required_changes_from_checks`` ````
       (backtick-quoted, not a real string literal) is not mistaken for a
       real monkeypatch target.
    """
    attr_patterns = {
        name: re.compile(
            rf"(?<![.\w])(?:workflow|workflow_module|cw_workflow|wf)\.{re.escape(name)}\b"
        )
        for name in candidates
    }
    string_patterns = {
        name: re.compile(rf"""(['"])charlie_work\.workflow\.{re.escape(name)}\b""")
        for name in candidates
    }

    found: dict[str, str] = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:  # pragma: no cover - a broken tree is a different failure
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_is_workflow = node.module == "charlie_work.workflow" or (
                        node.level == 1 and node.module == "workflow"
                    )
                    if not module_is_workflow:
                        continue
                    for alias in node.names:
                        if alias.name in candidates:
                            found.setdefault(alias.name, _display_path(path))
            for name in candidates:
                if name in found:
                    continue
                if attr_patterns[name].search(source) or string_patterns[name].search(source):
                    found[name] = _display_path(path)
    return found


def test_facade_reexports_every_name_consumers_reach_through_workflow() -> None:
    """AC5: re-export completeness, derived from a live scan -- not hardcoded.

    Mirrors test_rework_prompts_split.py's own AC5 test (itself mirroring
    test_verdict_parsing_split.py / test_escalation_split.py /
    test_dispatch_selection_split.py / test_ci_fleet_seams.py's
    ``test_adapter_exports_every_name_the_consumers_import``): the expected
    set comes from walking the actual consumers under tests/scripts/src, not
    from a list restated by hand in this test (which is exactly the kind of
    copy that drifts the moment a consumer changes).

    Only 4 of ci_findings.py's 8 module-level names have any consumer
    reference outside workflow.py at all -- ``_non_required_check_findings``,
    ``check_dispatch_staleness``, ``_annotation_to_required_change``, and
    ``_required_changes_from_checks``. The other 4 (``_ci_status_section``,
    ``_backlog_is_non_empty``, ``_latest_non_empty_dispatch``,
    ``_parse_iso_ts``) are reached only by bare-name call sites inside
    OrchestratorApp methods that stay in workflow.py (``review()`` and
    ``_dispatch_impl()``), so this scan imposes no obligation for them. They
    are still required to be re-exported by the unconditional
    facade-obligation rule (AC4 covers that), just not because this live
    scan demands it.
    """
    candidates = set(_module_level_defined_names(_CI_FINDINGS_PATH))
    referenced = _consumer_referenced_names(
        candidates,
        [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"],
    )

    # Positive control (mirrors the rework_prompts/verdict_parsing/escalation/
    # dispatch_selection tests' own control): an empty walk would pass the
    # assertion below vacuously, and "no consumer references any of these
    # names" is exactly what a scan that stopped working would look like.
    assert referenced, (
        "no consumer under tests/scripts/src references any ci-findings name "
        "through charlie_work.workflow -- the scan itself is broken"
    )

    facade_names = _facade_reexported_names(_WORKFLOW_PATH)
    missing = {n: f for n, f in sorted(referenced.items()) if n not in facade_names}

    assert missing == {}, (
        "workflow.py's facade re-export block is missing names real consumers still "
        f"reach through charlie_work.workflow: {missing}"
    )


def test_consumer_reference_scan_finds_the_known_anchors() -> None:
    """Control for the scan above: it must find the specific references this
    extraction's Preflight step identified by hand, so a regression in the
    scan's own patterns (e.g. an over-tightened regex) shows up here instead
    of silently shrinking the ``referenced`` set in the completeness test
    above.

    Anchors verified by directly running this file's own
    ``_consumer_referenced_names`` scan against this tree (not merely copied
    from Preflight's notes):

    - ``_non_required_check_findings``: real single-name import at
      ``tests/test_checks.py:17``.
    - ``check_dispatch_staleness``: real single-name import at
      ``tests/test_dispatch_staleness.py:21``.
    - ``_annotation_to_required_change`` and ``_required_changes_from_checks``:
      both resolve to ``tests/test_charlie_work.py`` -- both names are part
      of that file's large shared ``from charlie_work.workflow import
      (...)`` block starting ~L110, so both land on the same file. This is
      unlike ``test_rework_prompts_split.py``'s own anchor set, where each
      of its 6 referenced names landed on a distinct file -- there is no
      significance to two names sharing an anchor file here beyond both
      being consumed by the same giant direct-import test file.

    The remaining 4 names (``_ci_status_section``, ``_backlog_is_non_empty``,
    ``_latest_non_empty_dispatch``, ``_parse_iso_ts``) have NO anchor at
    all, in either ``tests/`` alone or the full ``tests+scripts+src`` scan --
    confirmed identical between the two scopes below, unlike
    ``_read_review_decision`` in the rework_prompts precedent (which needed
    ``scripts/`` to find its anchor). Nothing in this family's 8 names is
    reached only through ``scripts/``.
    """
    candidates = set(_module_level_defined_names(_CI_FINDINGS_PATH))
    referenced = _consumer_referenced_names(candidates, [_REPO_ROOT / "tests"])
    referenced_all = _consumer_referenced_names(
        candidates, [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"]
    )

    assert "_non_required_check_findings" in referenced
    assert referenced["_non_required_check_findings"] == "tests/test_checks.py"

    assert "check_dispatch_staleness" in referenced
    assert referenced["check_dispatch_staleness"] == "tests/test_dispatch_staleness.py"

    assert "_annotation_to_required_change" in referenced
    assert referenced["_annotation_to_required_change"] == "tests/test_charlie_work.py"

    assert "_required_changes_from_checks" in referenced
    assert referenced["_required_changes_from_checks"] == "tests/test_charlie_work.py"

    # The 4 behaviorally-only-exercised names have no anchor anywhere, in
    # either scope -- adding scripts/src does not surface one for this
    # family (unlike the rework_prompts precedent's _read_review_decision).
    no_anchor = {
        "_ci_status_section",
        "_backlog_is_non_empty",
        "_latest_non_empty_dispatch",
        "_parse_iso_ts",
    }
    for name in no_anchor:
        assert name not in referenced, f"{name} unexpectedly has a tests/-only anchor"
        assert name not in referenced_all, (
            f"{name} unexpectedly has an anchor even with scripts/src"
        )

    assert (
        set(referenced.keys())
        == set(referenced_all.keys())
        == {
            "_non_required_check_findings",
            "check_dispatch_staleness",
            "_annotation_to_required_change",
            "_required_changes_from_checks",
        }
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'monkeypatch.setattr("charlie_work.workflow._non_required_check_findings", x)\n',
            id="setattr-string",
        ),
        pytest.param(
            "from charlie_work.workflow import _non_required_check_findings\n",
            id="absolute-import",
        ),
        pytest.param("from .workflow import _non_required_check_findings\n", id="relative-import"),
        pytest.param(
            "workflow._non_required_check_findings(checks, required)\n",
            id="attribute-access",
        ),
        pytest.param(
            "wf._non_required_check_findings(checks, required)\n",
            id="wf-alias-attribute-access",
        ),
    ],
)
def test_reference_scan_recognizes_every_documented_form(tmp_path: Path, source: str) -> None:
    """Control: each of the documented reference forms is individually
    detectable, isolated from the noise of the real tree (so a form that
    stopped matching wouldn't be masked by the other forms still working).

    Uses ``_non_required_check_findings`` (rather than
    ``check_dispatch_staleness``) as the illustrative target deliberately:
    the probe strings below embed literal
    ``workflow._non_required_check_findings``/
    ``charlie_work.workflow._non_required_check_findings`` text inside THIS
    file's own source, and this file is itself part of the real tree the
    completeness test above walks. ``tests/test_checks.py`` --
    ``_non_required_check_findings``'s real anchor -- sorts alphabetically
    before ``tests/test_ci_findings_split.py``, so ``found.setdefault``
    locks in the real anchor before this file is ever scanned regardless of
    what these probe strings contain. ``check_dispatch_staleness``'s real
    anchor (``tests/test_dispatch_staleness.py``) sorts AFTER this filename,
    so using it here would let these very probe strings hijack that anchor
    in the completeness test above -- confirmed the hard way while writing
    this file, not merely reasoned about in the abstract.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")

    referenced = _consumer_referenced_names({"_non_required_check_findings"}, [tmp_path])

    assert "_non_required_check_findings" in referenced


def test_reference_scan_ignores_a_backtick_docstring_mention(tmp_path: Path) -> None:
    """Control for the false-positive class this family's precedent already
    hit once (A1's ``scripts/heartbeat_check.py``): a backtick-quoted RST
    cross-reference in a docstring must not be mistaken for a real
    monkeypatch string-literal target.

    This control only proves the specific quoted-dotted form
    (```` ``charlie_work.workflow.<name>`` ````) is excluded -- it is NOT a
    claim that every prose mention is ignored (the bare ``workflow.<name>``
    attribute-access form, without the ``charlie_work.`` prefix, IS matched
    by the attr-access pattern even inside backticks, which is why this
    file's own docstrings above are careful to always use the full
    ``charlie_work.workflow.<name>`` dotted form when referring to a moved
    name in prose, never the bare ``workflow.<name>`` form). Uses
    ``_non_required_check_findings`` for the same self-pollution reason
    documented on the control immediately above.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        '    """Companion to ``charlie_work.workflow._non_required_check_findings``."""\n'
        "    return None\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"_non_required_check_findings"}, [tmp_path])

    assert referenced == {}


# ---------------------------------------------------------------------------
# AC8/AC9 (post-move re-confirmation): the write/event-emission surface in
# the moved units' final spans in ci_findings.py must be EXACTLY EMPTY.
# ---------------------------------------------------------------------------

_WRITE_EVENT_CALL_NAMES = {
    "_write_json",
    "_write_text_atomic",
    "append_event",
    "log_event",
    "_record_event",
}
_RAW_WRITE_ATTRS = {"write_text", "write_bytes"}


def _write_event_call_sites(source: str, *, filename: str = "<string>") -> list[dict[str, object]]:
    """AST-derived call sites matching COMMON's write/event-surface name
    list (``_write_json``, ``_write_text_atomic``, ``write_text``,
    ``write_bytes``, ``append_event``, ``log_event``, ``_record_event``).

    AST ``Call`` nodes, not a raw substring/regex ``findall`` over the whole
    source -- this family's own docstrings discuss the atomic-write and
    event-instrumentation invariants by name in several places (e.g.
    ``check_dispatch_staleness``'s docstring references events.db and the
    ``dispatch_stale`` warning event it produces upstream of this function,
    in ``OrchestratorApp``), which a raw text search would miscount as call
    sites. Walking real ``Call`` nodes only counts code that actually
    executes.

    A single ``ast.NodeVisitor`` pass (not the nested-``ast.walk`` shape
    that would double-count every call).
    """
    tree = ast.parse(source, filename=filename)
    hits: list[dict[str, object]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _WRITE_EVENT_CALL_NAMES or name in _RAW_WRITE_ATTRS:
                hits.append({"lineno": node.lineno, "name": name})
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def test_ci_findings_write_event_surface_is_exactly_empty() -> None:
    """AC8 (post-move, re-confirmed): the moved units' FINAL spans in
    ci_findings.py (not just the pre-move workflow.py spans Preflight
    checked) emit zero writes and zero events.

    Unlike rework_prompts.py (whose ``_write_rework_prompt`` is the one
    place in the codebase that produces ``rework-prompt.md`` and its
    sidecar, so its own analogous test asserts exactly TWO
    ``_write_text_atomic`` hits), this family reads events.db
    (``query_events``, a read, not a write) and formats packet text -- it
    never writes a file and never emits an event. The required positive
    control that this zero-hits assertion is not merely a broken query is
    ``test_write_event_call_scanner_has_a_positive_control`` below, which
    runs the identical scanner over workflow.py and confirms it DOES find
    hits there.
    """
    source = _CI_FINDINGS_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(source, filename=str(_CI_FINDINGS_PATH))

    assert hits == [], (
        f"ci_findings.py has {len(hits)} write/event-emission call site(s), expected zero: "
        f"{hits} -- this family should only read events.db (query_events) and format packet "
        "text, never write files or emit events"
    )


def test_write_event_call_scanner_has_a_positive_control() -> None:
    """Required positive control (operator decision #4, binding) for the
    empty-result assertion above: a scanner that always returns ``[]`` (a
    typo'd node-type check, a name-set that no longer matches anything)
    would make the assertion above pass vacuously forever. This proves the
    identical scanner is capable of matching real write/event call sites
    when they exist, by running it over workflow.py -- which is known to
    still contain ``OrchestratorApp._write_json``, ``append_event``/
    ``log_event`` calls, and ``_write_text_atomic`` call sites inside
    ``record_review``'s own archive-copy logic.
    """
    workflow_source = _WORKFLOW_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(workflow_source, filename=str(_WORKFLOW_PATH))

    found_names = {h["name"] for h in hits}
    assert found_names, "the scanner found zero call sites in workflow.py -- the scanner is broken"
    # workflow.py is expected to contain call sites for every one of these
    # names somewhere -- a scanner that only found a subset would be
    # silently blind to some of the forms it's supposed to catch.
    expected_broad_coverage = {"_write_json", "_write_text_atomic", "append_event", "log_event"}
    missing = expected_broad_coverage - found_names
    assert missing == set(), (
        f"scanner found zero call sites for {sorted(missing)} in workflow.py, which is known to "
        "contain calls to all of them -- the scanner's Call-node matching is broken"
    )


def test_write_event_call_scanner_ignores_docstring_mentions() -> None:
    """Control proving the AST scanner (unlike a raw substring search) does
    NOT count prose mentions of write/event names in docstrings as call
    sites.

    Without this, a regression that accidentally switched the scanner back
    to a text-based search would silently inflate the empty-result test
    above from 0 to a nonzero count, and nothing else in this file would
    catch it -- the exact-match assertion above would just start failing
    with a confusing nonzero count rather than this test explaining why.
    """
    probe_source = (
        "def f(path):\n"
        '    """Reads events.db via query_events; see append_event and\n'
        "    _write_text_atomic in workflow.py for the write-side counterpart\n"
        '    this function has no analog of.\n"""\n'
        "    return None\n"
    )
    hits = _write_event_call_sites(probe_source)
    assert hits == [], (
        f"scanner counted prose-only mentions as call sites: {hits} -- it must walk ast.Call "
        "nodes only, never search raw source text"
    )
