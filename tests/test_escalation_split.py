"""Seam integrity for the workflow.py -> escalation.py split (#1283 Phase A, PR 2/~6).

``escalation.py`` holds the escalation free-function family -- the shared
escalation-flag check, the de-escalation skip-outcome builder, the
escalate-issue mutator, and the escalated-label repair predicates --
verbatim-moved out of ``workflow.py``. ``workflow.py`` re-exports every
moved name through a facade import block (mirroring ``config.py``'s
``RunnerAllocationConfig`` pattern and this repo's own
``dispatch_selection.py`` precedent) so every existing
``charlie_work.workflow.<name>`` import path and monkeypatch target keeps
resolving unchanged.

Three ways that promise can quietly break, none of which fails loudly on its
own:

* ``escalation.py`` could grow an import of ``workflow.py`` (e.g. to reach a
  helper still living there) -- workflow.py already imports escalation.py
  for the facade, so that would be a real import cycle. Nothing raises until
  some import order happens to hit it first.
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
_ESCALATION_PATH = _REPO_ROOT / "src" / "charlie_work" / "escalation.py"
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
    """Names workflow.py's facade block currently re-exports from ``.escalation``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "escalation":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# AC3: import-cycle guard
# ---------------------------------------------------------------------------


def _workflow_imports_in(source: str, *, filename: str = "<string>") -> list[str]:
    """AST-derived list of any import of ``workflow``/``charlie_work.workflow``.

    AST-based rather than a substring grep so a prose mention of "workflow.py"
    in a comment or docstring (this file's own module docstring has several)
    can never false-positive the check, and so the check isn't fooled by
    formatting a plain grep might not anticipate (``as`` aliases, multi-line
    ``from ... import (...)`` blocks).
    """
    tree = ast.parse(source, filename=filename)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module == "workflow":
                offenders.append(f"line {node.lineno}: from .workflow import ...")
            elif node.module == "charlie_work.workflow":
                offenders.append(f"line {node.lineno}: from charlie_work.workflow import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "charlie_work.workflow":
                    offenders.append(f"line {node.lineno}: import charlie_work.workflow")
    return offenders


def test_escalation_has_no_workflow_import() -> None:
    """AC3: escalation.py must never import from workflow.py.

    workflow.py's facade imports FROM escalation.py; the reverse import
    would be the exact cycle the issue's own Traps section warns against
    (import charlie_work.escalation would then transitively require
    charlie_work.workflow to already be fully initialized, and vice versa).
    """
    offenders = _workflow_imports_in(
        _ESCALATION_PATH.read_text(encoding="utf-8"),
        filename=str(_ESCALATION_PATH),
    )
    assert offenders == [], (
        "escalation.py imports from charlie_work.workflow -- this creates an "
        f"import cycle with workflow.py's facade block: {offenders}"
    )


def test_workflow_import_detector_flags_a_real_violation() -> None:
    """Control for the AST detector above -- proves it can actually fire.

    Without this, a detector that had quietly become incapable of finding
    anything (e.g. a typo'd node-type check) would leave the assertion above
    vacuously true forever.
    """
    relative_violation = "from .workflow import OrchestratorApp\n"
    absolute_violation = "import charlie_work.workflow\n"
    absolute_from_violation = "from charlie_work.workflow import OrchestratorApp\n"
    innocent = (
        '"""A docstring that merely mentions workflow.py and charlie_work.workflow.foo."""\n'
    )

    assert _workflow_imports_in(relative_violation) != []
    assert _workflow_imports_in(absolute_violation) != []
    assert _workflow_imports_in(absolute_from_violation) != []
    assert _workflow_imports_in(innocent) == [], "prose mention must not be flagged"


def test_escalation_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not error.

    The AST check above is necessary but not sufficient -- it only rules out
    one specific kind of cycle (an explicit import of workflow.py). This
    drives the real interpreter through the module's actual import
    statements, which also catches any other cyclical dependency the module
    might have picked up.
    """
    import importlib

    module = importlib.import_module("charlie_work.escalation")
    assert module.__name__ == "charlie_work.escalation"


# ---------------------------------------------------------------------------
# AC4: seam-guard identity checks (mirrors test_ci_fleet_seams.py's pattern)
# ---------------------------------------------------------------------------


def test_all_escalation_names_are_reexported_by_identity() -> None:
    """AC4: every name escalation.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of a
    function would compare unequal-but-structurally-similar in ways that are
    easy to miss.
    """
    import charlie_work.escalation as escalation
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_ESCALATION_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    not_identical = [n for n in names if getattr(workflow, n) is not getattr(escalation, n)]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"escalation.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above (mirrors
    test_ci_fleet_seams.py::test_identity_check_would_fail_on_a_redefined_class
    and test_dispatch_selection_split.py's own control).

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
       ``_collect_escalated_label_subjects`` cannot partially match a longer
       name). ``wf`` is included alongside ``workflow``/``workflow_module``
       because ``tests/test_fix_escalation_paths.py`` uses ``from
       charlie_work import workflow as wf`` as its local per-test-function
       alias.
    3. The string-dotted monkeypatch form
       ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)`` -- matched
       only when the token is quote-delimited, so a prose/docstring
       cross-reference like ```` ``charlie_work.workflow._escalate_issue`` ````
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

    Mirrors test_dispatch_selection_split.py's own AC5 test (itself mirroring
    test_ci_fleet_seams.py::test_adapter_exports_every_name_the_consumers_import):
    the expected set comes from walking the actual consumers under
    tests/scripts/src, not from a list restated by hand in this test (which
    is exactly the kind of copy that drifts the moment a consumer changes).

    Only 2 of escalation.py's 5 module-level names have any consumer
    reference outside workflow.py at all (`_escalate_issue` and
    `_collect_escalated_label_subjects`) -- `_escalation_flags`,
    `_deescalation_skip`, and `_escalated_label_needs_repair` are reached
    only by bare-name call sites inside OrchestratorApp methods that stay in
    workflow.py, so this scan imposes no obligation for them. They are still
    required to be re-exported by the unconditional facade-obligation rule
    (AC4 covers that), just not because this live scan demands it.
    """
    candidates = set(_module_level_defined_names(_ESCALATION_PATH))
    referenced = _consumer_referenced_names(
        candidates,
        [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"],
    )

    # Positive control (mirrors the dispatch_selection test's own control):
    # an empty walk would pass the assertion below vacuously, and "no
    # consumer references any of these names" is exactly what a scan that
    # stopped working would look like.
    assert referenced, (
        "no consumer under tests/scripts/src references any escalation name "
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

    Unlike A1's dispatch_selection control (3 anchors), only 2 of
    escalation.py's 5 names have any external reference at all -- see the
    completeness test's docstring above for why the other 3 are legitimately
    absent, not a scan failure.

    ``_escalate_issue`` itself is checked for presence only, not a specific
    resolved file: this test file's own reference-form probes below
    (parametrized examples, the backtick false-positive control) legitimately
    embed ``workflow._escalate_issue``/``charlie_work.workflow._escalate_issue``
    string literals as example content, and ``tests/test_escalation_split.py``
    sorts alphabetically ahead of the two real consumer files, so the scan's
    ``setdefault`` may resolve to this file instead of a real anchor -- exactly
    the same shape A1's own control uses for ``_reviewer_pid_alive`` (which its
    own parametrize block also embeds), while hard-anchoring only names that
    don't self-collide. ``_collect_escalated_label_subjects`` is never embedded
    in this file's own probe strings (confirmed by direct grep), so its
    specific resolved file is safe to hard-assert.
    """
    candidates = set(_module_level_defined_names(_ESCALATION_PATH))
    referenced = _consumer_referenced_names(candidates, [_REPO_ROOT / "tests"])

    assert "_escalate_issue" in referenced

    assert "_collect_escalated_label_subjects" in referenced
    assert referenced["_collect_escalated_label_subjects"] == "tests/test_fix_escalation_paths.py"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'monkeypatch.setattr("charlie_work.workflow._escalate_issue", x)\n',
            id="setattr-string",
        ),
        pytest.param("from charlie_work.workflow import _escalate_issue\n", id="absolute-import"),
        pytest.param("from .workflow import _escalate_issue\n", id="relative-import"),
        pytest.param("workflow._escalate_issue(state, 1)\n", id="attribute-access"),
        pytest.param("wf._escalate_issue(state, 1)\n", id="wf-alias-attribute-access"),
    ],
)
def test_reference_scan_recognizes_every_documented_form(tmp_path: Path, source: str) -> None:
    """Control: each of the documented reference forms is individually
    detectable, isolated from the noise of the real tree (so a form that
    stopped matching wouldn't be masked by the other forms still working).

    Includes the ``wf`` alias form (not present in A1's precedent) because
    ``tests/test_fix_escalation_paths.py`` is this family's real consumer of
    that exact pattern.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")

    referenced = _consumer_referenced_names({"_escalate_issue"}, [tmp_path])

    assert "_escalate_issue" in referenced


def test_reference_scan_ignores_a_backtick_docstring_mention(tmp_path: Path) -> None:
    """Control for the false-positive this family already hit once (A1's
    precedent, ``scripts/heartbeat_check.py``): a backtick-quoted RST
    cross-reference in a docstring must not be mistaken for a real
    monkeypatch string-literal target."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        '    """Companion to ``charlie_work.workflow._escalate_issue``."""\n'
        "    return None\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"_escalate_issue"}, [tmp_path])

    assert referenced == {}
