"""Seam integrity for the workflow.py -> verdict_parsing.py split (#1283 Phase A, PR 3/~6).

``verdict_parsing.py`` holds the reviewer-verdict-parsing free-function
family -- fenced-JSON extraction, stream-json event decoding, mtime-gated
file fallback recovery, reviewer-session-summary reconstruction, and the
frozen dataclass that summary returns -- verbatim-moved out of
``workflow.py``. ``workflow.py`` re-exports every moved name through a
facade import block (mirroring ``config.py``'s ``RunnerAllocationConfig``
pattern and this repo's own ``dispatch_selection.py``/``escalation.py``
precedents) so every existing ``charlie_work.workflow.<name>`` import path
and monkeypatch target keeps resolving unchanged.

Three ways that promise can quietly break, none of which fails loudly on its
own:

* ``verdict_parsing.py`` could grow an import of ``workflow.py`` (e.g. to
  reach a helper still living there) -- workflow.py already imports
  verdict_parsing.py for the facade, so that would be a real import cycle.
  Nothing raises until some import order happens to hit it first. The
  module-load-time ``OrchestratorConfig()`` construction for
  ``_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS`` makes this family specifically
  sensitive to import-order problems that a purely static check would not
  catch, which is why this file pairs the AST check with a behavioral
  "actually imports cleanly" smoke test.
* The facade could re-declare a name instead of importing it (a copy-paste
  that silently duplicates a function). Both copies look correct in
  isolation; only identity distinguishes them, exactly the ``GitHubError``
  hazard ``test_ci_fleet_seams.py`` documents for the ci_fleet extraction.
* The facade's import list could fall out of sync with what real consumers
  (tests, scripts) actually reach through ``charlie_work.workflow`` -- a
  line dropped during a later edit would not fail *this* file, only
  whatever test happens to import the now-missing name, at whatever time
  someone next touches this block.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_VERDICT_PARSING_PATH = _REPO_ROOT / "src" / "charlie_work" / "verdict_parsing.py"
_WORKFLOW_PATH = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"


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
    """Names workflow.py's facade block currently re-exports from ``.verdict_parsing``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "verdict_parsing"
        ):
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

    Shared by the workflow-cycle check (AC3) and the cross_family-independence
    check below -- one walk, parameterized by which module name pair to look
    for, so both detectors are exercised by the SAME positive control instead
    of two independently-typo-able copies of this walk.

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
    <absolute_module> import X``, and ``import <absolute_module>``.
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


def _cross_family_imports_in(source: str, *, filename: str = "<string>") -> list[str]:
    """AST-derived list of any import of ``cross_family``/``charlie_work.cross_family``."""
    return _module_imports_in(
        source,
        relative_module="cross_family",
        absolute_module="charlie_work.cross_family",
        filename=filename,
    )


def test_verdict_parsing_has_no_workflow_import() -> None:
    """AC3: verdict_parsing.py must never import from workflow.py.

    workflow.py's facade imports FROM verdict_parsing.py; the reverse import
    would be the exact cycle the issue's own Traps section warns against
    (import charlie_work.verdict_parsing would then transitively require
    charlie_work.workflow to already be fully initialized, and vice versa).
    """
    offenders = _workflow_imports_in(
        _VERDICT_PARSING_PATH.read_text(encoding="utf-8"),
        filename=str(_VERDICT_PARSING_PATH),
    )
    assert offenders == [], (
        "verdict_parsing.py imports from charlie_work.workflow -- this creates an "
        f"import cycle with workflow.py's facade block: {offenders}"
    )


def test_verdict_parsing_has_no_cross_family_import() -> None:
    """verdict_parsing.py must never import from cross_family.py.

    cross_family.py maintains its own independent, byte-identical copy of the
    fenced-code-block regex that ``_VERDICT_FENCE_RE`` also implements,
    deliberately NOT imported from workflow (workflow imports cross_family,
    not the reverse -- see cross_family.py's own dependency-direction
    comment). The new module inherits that same constraint: it must not
    import cross_family.py or deduplicate that regex against it.
    """
    offenders = _cross_family_imports_in(
        _VERDICT_PARSING_PATH.read_text(encoding="utf-8"),
        filename=str(_VERDICT_PARSING_PATH),
    )
    assert offenders == [], (
        "verdict_parsing.py imports from charlie_work.cross_family -- these two "
        f"modules must keep independent copies of the fence regex: {offenders}"
    )


def test_workflow_import_detector_flags_a_real_violation() -> None:
    """Control for the AST detector above -- proves it can actually fire.

    Without this, a detector that had quietly become incapable of finding
    anything (e.g. a typo'd node-type check) would leave the assertion above
    vacuously true forever.

    Includes the bare ``from . import workflow`` package-relative spelling
    (``ImportFrom(module=None, level=1, names=[alias("workflow")])``)
    alongside the module-qualified relative, absolute-``from``, and plain
    ``import`` spellings -- a prior version of this control omitted that
    form, which meant it could not reveal that the detector itself failed
    open on it (the detector matched nothing, the assertion above passed
    vacuously, and only the separate behavioral smoke test caught the
    resulting cycle via a real ``ImportError``).
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


def test_cross_family_import_detector_flags_a_real_violation() -> None:
    """Control for the cross_family detector used by
    ``test_verdict_parsing_has_no_cross_family_import`` above.

    ``_cross_family_imports_in`` shares its walk (``_module_imports_in``)
    with ``_workflow_imports_in``, but the two are invoked with different
    module-name arguments -- a mistake in how the module-name pair is wired
    (e.g. the ``cross_family`` check accidentally reusing the ``workflow``
    module names) would not be caught by the workflow-detector's own control
    above, since that control never calls the cross_family entry point at
    all. Without this test, a wiring mistake here would leave the assertion
    in ``test_verdict_parsing_has_no_cross_family_import`` vacuously true
    forever -- the exact failure mode the workflow-side control exists to
    rule out, just for the sibling detector.

    Includes the bare ``from . import cross_family`` package-relative
    spelling alongside the module-qualified relative, absolute-``from``, and
    plain ``import`` spellings, for the same reason the workflow-side
    control above does: a prior version omitted this form, which meant it
    could not reveal that the shared ``_module_imports_in`` walk failed open
    on it.
    """
    relative_violation = "from .cross_family import _VERDICT_FENCE_RE\n"
    relative_package_violation = "from . import cross_family\n"
    absolute_violation = "import charlie_work.cross_family\n"
    absolute_from_violation = "from charlie_work.cross_family import _VERDICT_FENCE_RE\n"
    innocent = (
        '"""A docstring that merely mentions cross_family.py and '
        'charlie_work.cross_family.foo."""\n'
    )

    assert _cross_family_imports_in(relative_violation) != []
    assert _cross_family_imports_in(relative_package_violation) != []
    assert _cross_family_imports_in(absolute_violation) != []
    assert _cross_family_imports_in(absolute_from_violation) != []
    assert _cross_family_imports_in(innocent) == [], "prose mention must not be flagged"


def test_verdict_parsing_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not error.

    The AST check above is necessary but not sufficient -- it only rules out
    one specific kind of cycle (an explicit import of workflow.py or
    cross_family.py). This drives the real interpreter through the module's
    actual import statements, which also catches any other cyclical
    dependency the module might have picked up.

    Load-bearing specifically for this family (not merely redundant with the
    AST check above): ``_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS`` constructs an
    ``OrchestratorConfig()`` instance AT MODULE IMPORT TIME, not lazily
    inside a function. An import-order problem or a broken default in
    ``OrchestratorConfig`` would raise the moment this module is imported --
    something a static AST walk over import statements can never detect,
    because it never actually executes the module body.
    """
    import importlib

    module = importlib.import_module("charlie_work.verdict_parsing")
    assert module.__name__ == "charlie_work.verdict_parsing"

    # The module-load-time construction actually ran and produced a real
    # value -- not merely "importing didn't raise", but "the specific
    # load-time side effect happened and is usable".
    markers = module._DEFAULT_REVIEW_SESSION_LIMIT_MARKERS
    assert isinstance(markers, (list, tuple)), (
        "_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS must be constructed from "
        "OrchestratorConfig().runtime.session_limit_markers at import time"
    )


# ---------------------------------------------------------------------------
# AC4: seam-guard identity checks (mirrors test_ci_fleet_seams.py's pattern)
# ---------------------------------------------------------------------------


def test_all_verdict_parsing_names_are_reexported_by_identity() -> None:
    """AC4: every name verdict_parsing.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of a
    function would compare unequal-but-structurally-similar in ways that are
    easy to miss.

    The ``len(names) == 23`` assertion below is a second, independent
    anti-vacuity guard, not mere brittleness: it is the only thing in this
    file that proves ``_module_level_defined_names`` is still walking the
    ``Assign``/``AnnAssign`` branches (the 12 module constants) and the
    ``ClassDef`` branch (``ReviewSessionOutcome``), not just
    ``FunctionDef``. A regression that silently dropped either branch would
    shrink ``names`` to the 10 functions, and every remaining assertion in
    this test -- and in the AC5 completeness test below, which draws its own
    candidate set from this same helper -- would keep passing while quietly
    stopping to cover 13 of the 23 moved units. This count is also a genuine
    tripwire, not just a guard: a later Phase-A/B PR that adds a symbol to
    ``verdict_parsing.py`` will legitimately trip it, and should update the
    count (and the comment above it) rather than deleting the check -- issue
    #1269 (W12) is exactly that PR, adding ``REVIEW_SESSION_FAILED_HEADING``,
    ``REVIEW_SESSION_SUMMARY_HEADING`` (2 constants: 10 -> 12) and
    ``body_has_crash_signature`` (1 function: 9 -> 10), for 20 -> 23 overall.
    Issue #1354 adds ``CAUSE_UNKNOWN``, ``_RESULT_EVENT_CAUSE_FIELDS`` (2
    constants: 12 -> 14) and ``_extract_terminating_cause`` (1 function:
    10 -> 11), for 23 -> 26 overall.
    Issue #1485 adds ``_EXTRACTED_VERDICT_SOURCES`` (1 constant: 14 -> 15)
    and ``is_extracted_verdict_source``, ``provenance_caveat_for`` (2
    functions: 11 -> 13), for 26 -> 29 overall.
    """
    import charlie_work.verdict_parsing as verdict_parsing
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_VERDICT_PARSING_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 29, (
        f"expected 29 moved units (13 functions + ReviewSessionOutcome + 15 constants), "
        f"found {len(names)}: {sorted(names)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    not_identical = [n for n in names if getattr(workflow, n) is not getattr(verdict_parsing, n)]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"verdict_parsing.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above (mirrors
    test_ci_fleet_seams.py::test_identity_check_would_fail_on_a_redefined_class
    and test_escalation_split.py's own control).

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
    (confirmed to be the only forms in use for this family by a direct grep
    sweep during preflight):

    1. ``from charlie_work.workflow import <name>`` / ``from .workflow import
       <name>`` (AST ``ImportFrom``, so a multi-line parenthesized import
       block is caught the same as a single-line one).
    2. ``workflow.<name>`` / ``workflow_module.<name>`` / ``wf.<name>``
       attribute access (regex, word-bounded so e.g.
       ``_extract_review_session_summary`` cannot partially match a longer
       name).
    3. The string-dotted monkeypatch form
       ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)`` -- matched
       only when the token is quote-delimited, so a prose/docstring
       cross-reference like ```` ``charlie_work.workflow._extract_verdict_from_text`` ````
       (backtick-quoted, not a real string literal) is not mistaken for a
       real monkeypatch target.
    """
    # (?<![.\w]) rather than a plain \b: "workflow." is also a substring of
    # the fully-qualified "charlie_work.workflow." (e.g. inside a
    # backtick-quoted docstring cross-reference), and a bare \b boundary
    # between "." and "w" does not exclude that. The lookbehind requires
    # "workflow"/etc. to start a standalone token, not the tail of a longer
    # dotted chain -- the fully-qualified form is form 3's job below.
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

    Mirrors test_escalation_split.py's own AC5 test (itself mirroring
    test_dispatch_selection_split.py / test_ci_fleet_seams.py's
    ``test_adapter_exports_every_name_the_consumers_import``): the expected
    set comes from walking the actual consumers under tests/scripts/src, not
    from a list restated by hand in this test (which is exactly the kind of
    copy that drifts the moment a consumer changes).

    Only 6 of verdict_parsing.py's 26 module-level names have any consumer
    reference outside workflow.py at all -- the core-chain functions
    ``_validate_review_verdict``, ``_extract_verdict_from_text``,
    ``_parse_review_verdict_from_log``, ``_parse_review_verdict_from_events``,
    ``_parse_review_verdict_from_files``, ``_extract_review_session_summary``.
    ``ReviewSessionOutcome``, ``_log_tail_throttled``,
    ``_reviewer_session_metrics``, and all 14 module constants are reached
    only by bare-name call sites inside workflow.py (OrchestratorApp methods,
    and -- since issue #1269, W12 -- the module-level
    ``_collect_external_findings``), so this scan imposes no obligation for
    them. They are still required to be re-exported by the unconditional
    facade-obligation rule (AC4 covers that), just not because this live
    scan demands it. ``body_has_crash_signature`` and the two
    ``REVIEW_SESSION_*_HEADING`` constants (W12) join this same
    reached-only-by-bare-name set: their other consumers
    (``rework_prompts.py``, ``scripts/backfill_stale_rework_briefs.py``, and
    this issue's new tests) import directly from ``charlie_work.verdict_parsing``,
    never through ``charlie_work.workflow``, so they add no new obligation
    here either. Issue #1354's ``CAUSE_UNKNOWN``,
    ``_RESULT_EVENT_CAUSE_FIELDS``, and ``_extract_terminating_cause`` join
    the same set for the same reason.
    """
    candidates = set(_module_level_defined_names(_VERDICT_PARSING_PATH))
    referenced = _consumer_referenced_names(
        candidates,
        [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"],
    )

    # Positive control (mirrors the escalation/dispatch_selection tests' own
    # control): an empty walk would pass the assertion below vacuously, and
    # "no consumer references any of these names" is exactly what a scan
    # that stopped working would look like.
    assert referenced, (
        "no consumer under tests/scripts/src references any verdict-parsing name "
        "through charlie_work.workflow -- the scan itself is broken"
    )

    facade_names = _facade_reexported_names(_WORKFLOW_PATH)
    missing = {n: f for n, f in sorted(referenced.items()) if n not in facade_names}

    assert missing == {}, (
        "workflow.py's facade re-export block is missing names real consumers still "
        f"reach through charlie_work.workflow: {missing}"
    )


def test_consumer_reference_scan_finds_the_known_anchors() -> None:
    """Control for the scan above: it must find the specific references the
    preflight notes for this extraction already identified by hand, so a
    regression in the scan's own patterns (e.g. an over-tightened regex)
    shows up here instead of silently shrinking the ``referenced`` set in the
    completeness test above.

    Anchors verified by direct grep during preflight (not merely copied from
    the recon):

    - ``_validate_review_verdict``, ``_extract_verdict_from_text``,
      ``_parse_review_verdict_from_files``: all three imported together in
      one parenthesized block at ``tests/test_fix_fabricated_verdicts.py:26``.
    - ``_parse_review_verdict_from_log``: ``tests/test_charlie_work.py``.
    - ``_parse_review_verdict_from_events``: ``tests/test_fix_verdict_stream_json.py``.
    - ``_extract_review_session_summary``: ``tests/test_fix_escalated_dispatch_gate.py``.

    ``_extract_verdict_from_text`` IS hard-asserted below despite this file's
    own reference-form probes (parametrized examples, the backtick
    false-positive control) legitimately embedding
    ``workflow._extract_verdict_from_text``/
    ``charlie_work.workflow._extract_verdict_from_text`` string literals as
    example content: ``sorted(root.rglob("*.py"))`` walks
    ``test_fix_fabricated_verdicts.py`` before
    ``test_verdict_parsing_split.py`` (``"f" < "v"``), and ``found.setdefault``
    only writes the first file it sees a name in -- so the real anchor wins
    and this file's own probe strings, visited later, are no-ops against an
    already-populated key. (This is the mirror image of the escalation
    precedent's own control for ``_escalate_issue``, where the precedent's
    test file genuinely does sort ahead of its anchor and therefore leaves
    that one name presence-only -- the ordering differs per file, so it must
    be re-derived here rather than copied.)
    """
    candidates = set(_module_level_defined_names(_VERDICT_PARSING_PATH))
    referenced = _consumer_referenced_names(candidates, [_REPO_ROOT / "tests"])

    assert "_extract_verdict_from_text" in referenced
    assert referenced["_extract_verdict_from_text"] == "tests/test_fix_fabricated_verdicts.py"

    assert "_validate_review_verdict" in referenced
    assert referenced["_validate_review_verdict"] == "tests/test_fix_fabricated_verdicts.py"

    assert "_parse_review_verdict_from_files" in referenced
    assert (
        referenced["_parse_review_verdict_from_files"] == "tests/test_fix_fabricated_verdicts.py"
    )

    assert "_parse_review_verdict_from_log" in referenced
    assert referenced["_parse_review_verdict_from_log"] == "tests/test_charlie_work.py"

    assert "_parse_review_verdict_from_events" in referenced
    assert (
        referenced["_parse_review_verdict_from_events"] == "tests/test_fix_verdict_stream_json.py"
    )

    assert "_extract_review_session_summary" in referenced
    assert (
        referenced["_extract_review_session_summary"]
        == "tests/test_fix_escalated_dispatch_gate.py"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'monkeypatch.setattr("charlie_work.workflow._extract_verdict_from_text", x)\n',
            id="setattr-string",
        ),
        pytest.param(
            "from charlie_work.workflow import _extract_verdict_from_text\n",
            id="absolute-import",
        ),
        pytest.param("from .workflow import _extract_verdict_from_text\n", id="relative-import"),
        pytest.param("workflow._extract_verdict_from_text(text)\n", id="attribute-access"),
        pytest.param("wf._extract_verdict_from_text(text)\n", id="wf-alias-attribute-access"),
    ],
)
def test_reference_scan_recognizes_every_documented_form(tmp_path: Path, source: str) -> None:
    """Control: each of the documented reference forms is individually
    detectable, isolated from the noise of the real tree (so a form that
    stopped matching wouldn't be masked by the other forms still working).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")

    referenced = _consumer_referenced_names({"_extract_verdict_from_text"}, [tmp_path])

    assert "_extract_verdict_from_text" in referenced


def test_reference_scan_ignores_a_backtick_docstring_mention(tmp_path: Path) -> None:
    """Control for the false-positive this family's precedent already hit
    once (A1's ``scripts/heartbeat_check.py``): a backtick-quoted RST
    cross-reference in a docstring must not be mistaken for a real
    monkeypatch string-literal target."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        '    """Companion to ``charlie_work.workflow._extract_verdict_from_text``."""\n'
        "    return None\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"_extract_verdict_from_text"}, [tmp_path])

    assert referenced == {}


# ---------------------------------------------------------------------------
# AC7 (post-move re-confirmation): zero write/event-emission surface in the
# moved units' final spans in verdict_parsing.py.
# ---------------------------------------------------------------------------

_WRITE_EVENT_SURFACE_RE = re.compile(
    r"record_review\("
    r'|"review-decision\.json"'
    r'|"rework-prompt\.md"'
    r'|"rework-dispatch-note\.txt"'
    r"|_write_json"
    r"|write_text"
    r"|write_bytes"
    r"|append_event"
    r"|log_event"
)


def test_verdict_parsing_module_has_zero_write_or_event_surface() -> None:
    """AC7 (post-move): re-confirms, against the FINAL moved spans in
    verdict_parsing.py (not just the pre-move workflow.py spans the recon
    checked), that this family never writes state or emits events.

    The family only parses verdict TEXT/JSON into a dict; the actual
    ``record_review()`` call and ``review-decision.json`` write happen later,
    inside ``_reap_review_verdicts`` itself (an ``OrchestratorApp`` method
    that stays in workflow.py, Phase B scope). A hit here would mean either
    the move picked up something it shouldn't have, or the recon's
    pre-move analysis was wrong -- either way this must fail loudly, not
    silently.
    """
    source = _VERDICT_PARSING_PATH.read_text(encoding="utf-8")
    hits = _WRITE_EVENT_SURFACE_RE.findall(source)
    assert hits == [], (
        f"verdict_parsing.py contains write/event-emission surface: {hits} -- "
        "this family should only parse verdict text, never write state or emit events"
    )


def test_write_event_surface_pattern_has_a_positive_control() -> None:
    """Positive control for the zero-hits assertion above (this repo's own
    verification discipline: an absence used as evidence needs a control
    proving the query itself isn't broken).

    The identical pattern, run over workflow.py (which the recon's own sweep
    found 149 hits in file-wide), must find at least one hit -- proving the
    pattern is capable of matching real write/event surface, not merely
    incapable of matching anything.
    """
    workflow_source = _WORKFLOW_PATH.read_text(encoding="utf-8")
    hits = _WRITE_EVENT_SURFACE_RE.findall(workflow_source)
    assert hits, (
        "the write/event-emission pattern found zero hits in workflow.py itself -- "
        "the pattern is broken, not evidence that verdict_parsing.py is clean"
    )


# ---------------------------------------------------------------------------
# Issue #1269 (W12): body_has_crash_signature -- crash-comment noise
# suppression. REVIEW_SESSION_FAILED_HEADING/REVIEW_SESSION_SUMMARY_HEADING
# are the wire contract between _extract_review_session_summary (the
# emitter) and two independent downstream consumers (workflow.py's
# collector-side filter, rework_prompts.py's render-side guard); these tests
# pin the predicate's behavior directly against the module under test.
# ---------------------------------------------------------------------------


def test_body_has_crash_signature_matches_a_synthetic_summary_heading() -> None:
    """A body opening with REVIEW_SESSION_SUMMARY_HEADING is recognized.

    Built from the shared constant itself (never a hardcoded copy of the
    literal string) -- open question 1 of the W12 implementation plan: this
    is the real, frequently-observed shape (jc#1394's 6 unstamped crash
    comments all carry this exact heading, never the launch-failed one).
    """
    from charlie_work.verdict_parsing import (
        REVIEW_SESSION_SUMMARY_HEADING,
        body_has_crash_signature,
    )

    body = (
        f"{REVIEW_SESSION_SUMMARY_HEADING}\n\n"
        "The automated reviewer ran for 4 turns (2 tool calls) but did not "
        "produce a structured verdict.\n"
    )
    assert body_has_crash_signature(body) is True


def test_body_has_crash_signature_matches_a_synthetic_failed_heading() -> None:
    """A body opening with REVIEW_SESSION_FAILED_HEADING is recognized.

    Open question 1's other branch: no captured fixture carries this
    heading (jc#1394's population happens to be entirely the summary
    variant), so this specimen is synthetic -- built from the shared
    constant, never a hardcoded copy, exactly per the plan's resolution.
    """
    from charlie_work.verdict_parsing import (
        REVIEW_SESSION_FAILED_HEADING,
        body_has_crash_signature,
    )

    body = (
        f"{REVIEW_SESSION_FAILED_HEADING}\n\n"
        "The automated reviewer exited before running a single turn, so no "
        "review was performed.\n"
    )
    assert body_has_crash_signature(body) is True


def test_review_session_failed_heading_exact_literal() -> None:
    """Value-pins REVIEW_SESSION_FAILED_HEADING to its exact literal text.

    test_body_has_crash_signature_matches_a_synthetic_failed_heading (above)
    derives its test body FROM the constant, so it would keep passing even
    if the constant's literal value silently drifted -- it only proves the
    predicate is self-consistent with whatever the constant currently says,
    not that the constant still says the right thing.
    test_body_has_crash_signature_real_captured_specimen_with_crlf pins
    REVIEW_SESSION_SUMMARY_HEADING the same way a real captured specimen
    does, hardcoding literal text independent of the constant. No real
    captured specimen carries the FAILED heading (see this file's docstring
    a few tests up), so this test does the equivalent job directly: hardcode
    the literal and assert the constant still equals it.
    """
    from charlie_work.verdict_parsing import REVIEW_SESSION_FAILED_HEADING

    assert REVIEW_SESSION_FAILED_HEADING == "## Reviewer session failed to start"


def test_body_has_crash_signature_real_captured_specimen_with_crlf() -> None:
    """A real, unmodified crash comment (jc#1394, GitHub comment id
    5067515891) is recognized, CRLF line endings and all.

    Captured verbatim via the GitHub API (``\\r\\n``, not normalized) --
    the one specimen in this file not derived from the shared constant, kept
    specifically because a synthetic ``\\n``-only body would not exercise
    the real line-ending shape GitHub actually returns, and would not catch
    a future rewrite of the predicate that started splitting on lines
    instead of a plain prefix check. Written as escaped ``\\r\\n`` sequences
    (not literal CR bytes) so the source text itself -- not just the string
    value -- is immune to git's line-ending normalization. No secrets: the
    body is boilerplate crash/hook-failure text plus a local tool path, both
    already public on the source PR.
    """
    from charlie_work.verdict_parsing import body_has_crash_signature

    real_specimen = (
        "## Reviewer session summary (no verdict produced)\r\n"
        "\r\n"
        "The automated reviewer did not produce a structured verdict.\r\n"
        "\r\n"
        "\r\n"
        "### Recent analysis from the reviewer:\r\n"
        "\r\n"
        "Error: When using --print, --output-format=stream-json requires --verbose\r\n"
        "\r\n"
        "---\r\n"
        "\r\n"
        'SessionEnd hook ["C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" '
        '-NoProfile -File "C:\\Users\\senki\\repos\\llibrary\\hooks\\session-end.ps1"] '
        "failed: Hook cancelled\r\n"
        "\r\n"
        "---\r\n"
    )
    assert "\r\n" in real_specimen, "control: this specimen must actually carry CRLF"
    assert body_has_crash_signature(real_specimen) is True


def test_body_has_crash_signature_prefix_not_substring() -> None:
    """A GitHub "Quote reply" that quotes a crash comment is NOT flagged.

    Mirrors ``_is_orchestrator_comment``'s own rationale in workflow.py:
    GitHub's quote-reply blockquotes every line of the quoted body with
    ``"> "``, so the heading no longer sits at the start of the
    (``lstrip()``-ped) text. A substring match would still catch this and
    wrongly discard a genuine human reply discussing the crash; the prefix
    check must not.
    """
    from charlie_work.verdict_parsing import (
        REVIEW_SESSION_SUMMARY_HEADING,
        body_has_crash_signature,
    )

    quoted_reply = (
        f"> {REVIEW_SESSION_SUMMARY_HEADING}\n"
        "> \n"
        "> The automated reviewer ran for 4 turns...\n"
        "\n"
        "This looks like a session crash, not a real review -- can we re-run it?\n"
    )
    assert body_has_crash_signature(quoted_reply) is False


def test_body_has_crash_signature_false_for_unrelated_finding() -> None:
    """Control: ordinary reviewer/human finding text is never flagged.

    Without this, a predicate that had quietly become unconditionally True
    (or matched on a much broader substring) would pass every test above
    vacuously.
    """
    from charlie_work.verdict_parsing import body_has_crash_signature

    assert body_has_crash_signature("The retry loop does not cap its backoff.") is False
    assert body_has_crash_signature("") is False


def test_body_has_crash_signature_tolerates_leading_blank_lines() -> None:
    """A body with leading blank lines before the heading is still matched
    (``lstrip()`` in the predicate), mirroring GitHub occasionally rendering
    a leading newline before the first Markdown heading."""
    from charlie_work.verdict_parsing import (
        REVIEW_SESSION_FAILED_HEADING,
        body_has_crash_signature,
    )

    assert body_has_crash_signature(f"\n\n{REVIEW_SESSION_FAILED_HEADING}\nbody\n") is True
