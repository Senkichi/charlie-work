"""Seam integrity for the workflow.py -> rework_prompts.py split (#1283 Phase A, PR 4/~6).

``rework_prompts.py`` holds the rework-prompt-rendering free-function family
-- the required-changes-section tiered fallback renderer, the external-findings
join helper, the review-decision JSON reader and freshness comparison used to
detect a stale brief, the per-round retry/distinct-verdict archive numbering,
and the pure-render / atomic-write split behind ``_write_rework_prompt`` --
verbatim-moved out of ``workflow.py``. ``workflow.py`` re-exports every moved
name through a facade import block (mirroring ``config.py``'s
``RunnerAllocationConfig`` pattern and this repo's own
``dispatch_selection.py``/``escalation.py``/``verdict_parsing.py``
precedents) so every existing ``charlie_work.workflow.<name>`` import path
and monkeypatch target keeps resolving unchanged.

This file mirrors ``tests/test_verdict_parsing_split.py`` (the current
precedent) structurally, with two deliberate departures documented at the
point they occur rather than silently:

* No cross_family-independence test. verdict_parsing.py's precedent guards
  against re-deduplicating a regex against ``cross_family.py`` -- a concern
  specific to that family's own fenced-JSON parsing. As of issue #1269
  (W12), ``rework_prompts.py`` does import one name from ``rescue_review.py``
  (``LEGACY_VACUOUS_SUMMARY``, a plain constant reference, not a
  re-implementation of any cross_family logic), so "not among this module's
  imports" is no longer literally true -- but that import creates no cycle
  (``rescue_review.py`` imports neither ``rework_prompts`` nor ``workflow``,
  confirmed by ``test_rework_prompts_has_no_workflow_import``'s sibling
  concern) and duplicates no regex, so there remains no analogous hazard
  for a dedicated test to guard against.
* The write/event-emission surface re-confirmation (AC9) asserts a
  non-empty result -- exactly two real call sites, both
  ``_write_text_atomic`` -- rather than zero. Unlike verdict_parsing.py
  (which only parses text and writes nothing), this family's
  ``_write_rework_prompt`` is the one place in the codebase that produces
  ``rework-prompt.md`` and its ``rework-dispatch-note.txt`` sidecar. The
  dispatch prompt's own AC9 text ("exactly ONE hit... matching the recon's
  pre-move finding") repeats the *original* recon's count, which this PR's
  own Preflight step already re-derived and corrected to two hits (see
  ``a4-rework-prompts-notes.md``, "SURPRISES" item on the write/event
  re-grep): the recon treated the ``rework-prompt.md`` and
  ``rework-dispatch-note.txt`` writes asymmetrically for no structural
  reason -- both are the same ``_write_text_atomic(path, text)`` call shape,
  one line apart, inside the same function. This file's own AST-based
  re-verification (independent of both the recon and Preflight's) confirms
  two, not one; the test below asserts what is actually there, not the
  stale count.

Three ways the facade promise can quietly break, none of which fails loudly
on its own (same three failure modes ``test_verdict_parsing_split.py``
documents for its own family):

* ``rework_prompts.py`` could grow an import of ``workflow.py`` -- workflow.py
  already imports rework_prompts.py for the facade, so that would be a real
  import cycle.
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
_REWORK_PROMPTS_PATH = _REPO_ROOT / "src" / "charlie_work" / "rework_prompts.py"
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
    """Names workflow.py's facade block currently re-exports from ``.rework_prompts``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "rework_prompts"
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


def test_rework_prompts_has_no_workflow_import() -> None:
    """AC3: rework_prompts.py must never import from workflow.py.

    workflow.py's facade imports FROM rework_prompts.py; the reverse import
    would be the exact cycle the issue's own Traps section warns against
    (import charlie_work.rework_prompts would then transitively require
    charlie_work.workflow to already be fully initialized, and vice versa).
    """
    offenders = _workflow_imports_in(
        _REWORK_PROMPTS_PATH.read_text(encoding="utf-8"),
        filename=str(_REWORK_PROMPTS_PATH),
    )
    assert offenders == [], (
        "rework_prompts.py imports from charlie_work.workflow -- this creates an "
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
    ``import`` spellings -- a prior version of this control omitted that
    form, which meant it could not reveal that the detector itself failed
    open on it (the detector matched nothing, the assertion below passed
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


def test_rework_prompts_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not error.

    The AST check above is necessary but not sufficient -- it only rules out
    one specific kind of cycle (an explicit import of workflow.py). This
    drives the real interpreter through the module's actual import
    statements, which also catches any other cyclical dependency the module
    might have picked up (e.g. through ``.config``, ``.github``,
    ``.markdown_fence``, or ``.prompts``).

    Kept for precedent-parity with ``test_verdict_parsing_split.py`` even
    though this family has no module-load-time side effect analogous to that
    family's ``_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS = OrchestratorConfig()``
    construction -- Preflight's fresh AST re-derivation (restricted to
    exactly the 3 fragment line ranges) found zero top-level ``Expr``/
    ``Call``/``If`` statements among the 14 moved units, only the 11
    ``FunctionDef`` + 3 ``Assign`` nodes themselves. This test still earns
    its place: it drives the interpreter through the real import statements
    (``.config``, ``.github``, ``.markdown_fence``, ``.prompts``), which the
    AST check above does not exercise, and would catch any cycle picked up
    through one of those four modules that the workflow-only check can't see.
    """
    import importlib

    module = importlib.import_module("charlie_work.rework_prompts")
    assert module.__name__ == "charlie_work.rework_prompts"

    # Confirms the module body actually executed to completion (not merely
    # "importlib didn't raise") by checking a real symbol landed in the
    # module's namespace with the expected type.
    assert callable(module._write_rework_prompt)
    assert isinstance(module._ROUND_COMPARE_KEYS, tuple)


# ---------------------------------------------------------------------------
# AC4: seam-guard identity checks (mirrors test_verdict_parsing_split.py's pattern)
# ---------------------------------------------------------------------------


def test_all_rework_prompts_names_are_reexported_by_identity() -> None:
    """AC4: every name rework_prompts.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of a
    function would compare unequal-but-structurally-similar in ways that are
    easy to miss.

    The ``len(names) == 16`` assertion below is a second, independent
    anti-vacuity guard: it is the only thing in this file that proves
    ``_module_level_defined_names`` is still walking the ``Assign`` branch
    (the 4 module constants: ``_EXTERNAL_FINDINGS_POINTER``,
    ``_EXTERNAL_FINDINGS_SECTION_INTRO``, ``_REQUIRED_CHANGES_TIER1_INTRO``,
    ``_ROUND_COMPARE_KEYS``), not just ``FunctionDef``. A regression that
    silently dropped that branch would shrink ``names`` to the 12 functions,
    and every remaining assertion in this test -- and in the AC5
    completeness test below, which draws its own candidate set from this
    same helper -- would keep passing while quietly stopping to cover 12 of
    the 16 moved/added units.

    Issue #1270 (W13) added two more free functions here --
    ``_round_history_entries`` and ``_render_round_findings`` -- bringing the
    count from 14 to 16 (12 core-chain functions + _write_text_atomic + 3
    constants). Neither is reached by this file's own consumer-reference
    scan (AC5 below): both were called only by bare-name call sites inside
    ``OrchestratorApp._build_prior_review_section``, which stays in
    workflow.py, so they joined the "reached only by bare-name call sites"
    bucket AC5's docstring already describes for 8 of the original 14 names.
    A #1270 review-round-1 fix then hoisted a fifth constant,
    ``_REQUIRED_CHANGES_TIER1_INTRO`` (previously a literal restated in two
    places), bringing the count to 17; it joined the same bare-name-only
    bucket, referenced only inside ``_render_required_changes_section`` and
    ``_render_round_findings``, both of which stay in rework_prompts.py.

    Issue #1362 Stage 1 then hoisted ``_round_history_entries`` itself OUT
    of this module into ``charlie_work.review_decision`` (the new
    single-reader module) -- this file now imports it rather than defining
    it, so it drops out of ``_module_level_defined_names``'s AST walk
    entirely (that helper only sees ``FunctionDef``/``Assign``/
    ``AnnAssign``, never ``ImportFrom``), bringing the count back down to
    16. ``workflow.py``'s facade import of ``_round_history_entries`` at
    line 334 keeps resolving through the new import chain (``workflow`` ->
    ``rework_prompts`` -> ``review_decision``), so the identity check below
    still holds for it even though it is no longer in ``names``.
    Issue #1485 adds ``_provenance_caveat_from_decision`` (1 function:
    11 -> 12 core-chain functions), for 16 -> 17 overall.
    """
    import charlie_work.rework_prompts as rework_prompts
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_REWORK_PROMPTS_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 17, (
        f"expected 17 moved/added units (12 core-chain functions + "
        f"_write_text_atomic + 4 constants, after issue #1362 Stage 1 hoisted "
        f"_round_history_entries out to review_decision.py), found {len(names)}: "
        f"{sorted(names)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    not_identical = [n for n in names if getattr(workflow, n) is not getattr(rework_prompts, n)]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"rework_prompts.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above (mirrors
    test_verdict_parsing_split.py's own control, itself mirroring
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
    (confirmed to be the only forms in use for this family by a direct grep
    sweep during preflight):

    1. ``from charlie_work.workflow import <name>`` / ``from .workflow import
       <name>`` (AST ``ImportFrom``, so a multi-line parenthesized import
       block is caught the same as a single-line one).
    2. ``workflow.<name>`` / ``workflow_module.<name>`` / ``wf.<name>``
       attribute access (regex, word-bounded so e.g.
       ``_render_required_changes_section`` cannot partially match a longer
       name).
    3. The string-dotted monkeypatch form
       ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)`` -- matched
       only when the token is quote-delimited, so a prose/docstring
       cross-reference like ```` ``charlie_work.workflow._write_rework_prompt`` ````
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

    Mirrors test_verdict_parsing_split.py's own AC5 test (itself mirroring
    test_escalation_split.py / test_dispatch_selection_split.py /
    test_ci_fleet_seams.py's ``test_adapter_exports_every_name_the_consumers_import``):
    the expected set comes from walking the actual consumers under
    tests/scripts/src, not from a list restated by hand in this test (which
    is exactly the kind of copy that drifts the moment a consumer changes).

    Only 6 of rework_prompts.py's 17 module-level names have any consumer
    reference outside workflow.py at all -- ``_render_required_changes_section``,
    ``_is_verdict_newer_than_brief``, ``_read_review_decision``,
    ``_existing_round_numbers``, ``_write_text_atomic``, and
    ``_write_rework_prompt``. The other 11 (``_rework_prompt_search_dirs``,
    ``_finish_required_changes_section``, ``_render_external_findings_section``,
    ``_EXTERNAL_FINDINGS_POINTER``, ``_EXTERNAL_FINDINGS_SECTION_INTRO``,
    ``_next_round_number``, ``_ROUND_COMPARE_KEYS``, ``_render_rework_prompt``,
    and, since issue #1270/W13, ``_round_history_entries``,
    ``_render_round_findings``, and ``_REQUIRED_CHANGES_TIER1_INTRO``) are
    reached only by bare-name call sites inside OrchestratorApp methods (or,
    for the constant, inside other rework_prompts.py functions) that stay
    in workflow.py or rework_prompts.py respectively, so this scan imposes
    no obligation for them. They are still required to be re-exported by
    the unconditional facade-obligation rule (AC4 covers that), just not
    because this live scan demands it.
    """
    candidates = set(_module_level_defined_names(_REWORK_PROMPTS_PATH))
    referenced = _consumer_referenced_names(
        candidates,
        [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"],
    )

    # Positive control (mirrors the verdict_parsing/escalation/dispatch_selection
    # tests' own control): an empty walk would pass the assertion below
    # vacuously, and "no consumer references any of these names" is exactly
    # what a scan that stopped working would look like.
    assert referenced, (
        "no consumer under tests/scripts/src references any rework-prompts name "
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
    extraction's Preflight/SeamTests steps identified by hand, so a
    regression in the scan's own patterns (e.g. an over-tightened regex)
    shows up here instead of silently shrinking the ``referenced`` set in the
    completeness test above.

    Anchors verified by direct grep against this tree (not merely copied
    from the recon):

    - ``_render_required_changes_section``: real top-of-file import at
      ``tests/test_charlie_work.py:123`` (``sorted(root.rglob("*.py"))``
      visits ``tests/`` before ``scripts/``, and within ``tests/`` this file
      sorts ahead of the production consumer's own test file, so the found
      import wins over ``scripts/ac1b_findings_actionability.py:72`` even
      though both are real).
    - ``_is_verdict_newer_than_brief``: real import at
      ``tests/test_backfill_stale_rework_briefs.py:23``.
    - ``_read_review_decision``: NOT imported by name in
      ``tests/test_backfill_stale_rework_briefs.py`` (only
      ``_is_verdict_newer_than_brief`` is), so the scan correctly falls
      through ``tests/`` with no match and lands on the real production
      import at ``scripts/backfill_stale_rework_briefs.py:113``.
    - ``_existing_round_numbers``: real import at
      ``tests/test_review_round_archive.py:57``.
    - ``_write_text_atomic``: real import at
      ``tests/test_review_event_payload.py:46``.
    - ``_write_rework_prompt``: the scan's winning match is
      ``tests/test_charlie_work.py``, but -- unlike the five anchors above --
      this one is NOT a real import in that file. It is a parenthesized
      prose mention in a docstring, ``(workflow._write_rework_prompt)`` at
      line 26358, that is not backtick-quoted and so is not excluded by the
      backtick-mention control below (which only excludes RST-style
      backtick cross-references, the one false-positive class this family's
      own precedent already hit once). ``test_charlie_work.py`` sorts ahead
      of this name's REAL import sites (``tests/test_markdown_fence.py``,
      ``tests/test_prompt_render_contract.py:41``,
      ``tests/test_prompt_template_drift_check.py:38``), so
      ``found.setdefault`` locks in the docstring mention first. This is the
      mirror image of the verdict_parsing precedent's own control for
      ``_escalate_issue``/``_extract_verdict_from_text`` (where the ordering
      quirk ran the other way) -- re-derived here rather than assumed, since
      the winner differs per name and per file tree.
    """
    candidates = set(_module_level_defined_names(_REWORK_PROMPTS_PATH))
    referenced = _consumer_referenced_names(candidates, [_REPO_ROOT / "tests"])
    referenced_all = _consumer_referenced_names(
        candidates, [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"]
    )

    assert "_render_required_changes_section" in referenced
    assert referenced["_render_required_changes_section"] == "tests/test_charlie_work.py"

    assert "_is_verdict_newer_than_brief" in referenced
    assert (
        referenced["_is_verdict_newer_than_brief"] == "tests/test_backfill_stale_rework_briefs.py"
    )

    # _read_review_decision has no match anywhere under tests/ alone -- its
    # real anchor is a production script, only visible once scripts/ is
    # included in the search roots.
    assert "_read_review_decision" not in referenced
    assert "_read_review_decision" in referenced_all
    assert referenced_all["_read_review_decision"] == "scripts/backfill_stale_rework_briefs.py"

    assert "_existing_round_numbers" in referenced
    assert referenced["_existing_round_numbers"] == "tests/test_review_round_archive.py"

    assert "_write_text_atomic" in referenced
    assert referenced["_write_text_atomic"] == "tests/test_review_event_payload.py"

    assert "_write_rework_prompt" in referenced
    assert referenced["_write_rework_prompt"] == "tests/test_charlie_work.py"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'monkeypatch.setattr("charlie_work.workflow._render_required_changes_section", x)\n',
            id="setattr-string",
        ),
        pytest.param(
            "from charlie_work.workflow import _render_required_changes_section\n",
            id="absolute-import",
        ),
        pytest.param(
            "from .workflow import _render_required_changes_section\n", id="relative-import"
        ),
        pytest.param(
            "workflow._render_required_changes_section(decision)\n", id="attribute-access"
        ),
        pytest.param(
            "wf._render_required_changes_section(decision)\n", id="wf-alias-attribute-access"
        ),
    ],
)
def test_reference_scan_recognizes_every_documented_form(tmp_path: Path, source: str) -> None:
    """Control: each of the documented reference forms is individually
    detectable, isolated from the noise of the real tree (so a form that
    stopped matching wouldn't be masked by the other forms still working).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")

    referenced = _consumer_referenced_names({"_render_required_changes_section"}, [tmp_path])

    assert "_render_required_changes_section" in referenced


def test_reference_scan_ignores_a_backtick_docstring_mention(tmp_path: Path) -> None:
    """Control for the false-positive this family's precedent already hit
    once (A1's ``scripts/heartbeat_check.py``): a backtick-quoted RST
    cross-reference in a docstring must not be mistaken for a real
    monkeypatch string-literal target.

    Not to be confused with the *non*-backtick-quoted prose mention this
    family genuinely does have (documented in
    ``test_consumer_reference_scan_finds_the_known_anchors`` above) -- this
    control only proves the specific quoted form is excluded, not that every
    prose mention is.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        '    """Companion to ``charlie_work.workflow._render_required_changes_section``."""\n'
        "    return None\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"_render_required_changes_section"}, [tmp_path])

    assert referenced == {}


# ---------------------------------------------------------------------------
# AC9 (post-move re-confirmation): the write/event-emission surface in the
# moved units' final spans in rework_prompts.py.
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
    source: this family's docstrings discuss the atomic-write invariant by
    name (e.g. ``_write_rework_prompt``'s own docstring says "a plain
    write_text here is the same ... failure class" and mentions
    ``_write_text_atomic`` and ``OrchestratorApp._write_json`` in prose) --
    a raw text search over this module finds 6 lines, 4 of them pure
    docstring/comment mentions with zero write behavior behind them. Walking
    real ``Call`` nodes only counts code that actually executes, which is
    what "write/event-emission surface" means for this check's purpose.

    The first positional arg is resolved one level through the enclosing
    function's local variable assignments (mirroring
    ``test_review_round_archive.py``'s own AC6 scanner's ``_resolve``
    helper): ``_write_rework_prompt`` calls
    ``_write_text_atomic(prompt_path, prompt)`` where ``prompt_path`` is
    assigned two lines earlier from ``pr_dir / "rework-prompt.md"`` -- an
    unparse of the bare ``Name`` node alone would just say ``"prompt_path"``
    and never surface the literal filename a caller of this scanner needs to
    assert against.

    A single ``ast.NodeVisitor`` pass (not the nested-``ast.walk`` shape
    that would double-count every call), tracking the innermost enclosing
    function's assignment table as a stack so a call inside a nested
    function only resolves against that function's own locals.
    """
    tree = ast.parse(source, filename=filename)
    hits: list[dict[str, object]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope_stack: list[dict[str, str]] = [{}]

        def _enter_scope(self, node: ast.AST) -> None:
            assigned_rhs: dict[str, str] = {}
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned_rhs[target.id] = ast.unparse(child.value)
            self.scope_stack.append(assigned_rhs)
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._enter_scope(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._enter_scope(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _WRITE_EVENT_CALL_NAMES or name in _RAW_WRITE_ATTRS:
                assigned_rhs = self.scope_stack[-1]
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Name) and first_arg.id in assigned_rhs:
                        arg_text = f"{first_arg.id} = {assigned_rhs[first_arg.id]}"
                    else:
                        arg_text = ast.unparse(first_arg)
                else:
                    arg_text = ""
                hits.append({"lineno": node.lineno, "name": name, "first_arg": arg_text})
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def test_rework_prompts_write_event_surface_is_exactly_the_two_known_atomic_writes() -> None:
    """AC9 (post-move, corrected): re-confirms, against the FINAL moved spans
    in rework_prompts.py (not just the pre-move workflow.py spans the recon
    checked), the family's write/event-emission surface.

    The dispatch prompt's own AC9 text asserts "EXACTLY ONE hit... matching
    the recon's pre-move finding" -- but the recon's original count was
    itself wrong, already caught and corrected by this PR's own Preflight
    step (see ``a4-rework-prompts-notes.md``, "Write/event-emission re-grep
    result -- CORRECTED vs. recon"): ``_write_rework_prompt`` makes TWO
    ``_write_text_atomic`` calls, not one -- the live brief
    (``rework-prompt.md``) and its raw-note sidecar
    (``rework-dispatch-note.txt``), one line apart, same call shape. This
    test asserts the corrected, empirically-verified count (independently
    re-derived here via AST ``Call`` walking, a third confirmation after the
    recon's own re-read and Preflight's re-grep) rather than the stale
    number, and pins down exactly which two call sites they are so a future
    edit that adds a third write (or swaps one for a raw ``write_text``)
    fails loudly here.

    Zero ``append_event``/``log_event``/``_record_event``/``_write_json``
    hits and zero raw ``write_text``/``write_bytes`` calls (which would
    bypass the atomic-write invariant CLAUDE.md requires) -- both confirmed
    correct by the recon and unchanged by this re-verification.
    """
    source = _REWORK_PROMPTS_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(source, filename=str(_REWORK_PROMPTS_PATH))

    names = sorted(h["name"] for h in hits)
    assert names == ["_write_text_atomic", "_write_text_atomic"], (
        f"expected exactly two _write_text_atomic call sites and nothing else, found: {hits}"
    )

    targets = sorted(str(h["first_arg"]) for h in hits)
    # The first positional arg to _write_text_atomic is the destination path
    # expression; both target expressions must mention their respective
    # filename literal so a future edit that redirects one of these writes
    # to a different artifact is caught here.
    assert any("rework-prompt.md" in t for t in targets), (
        f"no _write_text_atomic call targets rework-prompt.md: {targets}"
    )
    assert any("rework-dispatch-note.txt" in t for t in targets), (
        f"no _write_text_atomic call targets rework-dispatch-note.txt: {targets}"
    )

    raw_write_hits = [h for h in hits if h["name"] in _RAW_WRITE_ATTRS]
    assert raw_write_hits == [], (
        f"found raw write_text/write_bytes call(s) bypassing _write_text_atomic: {raw_write_hits}"
    )

    event_hits = [
        h
        for h in hits
        if h["name"] in ("append_event", "log_event", "_record_event", "_write_json")
    ]
    assert event_hits == [], (
        f"found event-emission or _write_json call(s) inside rework_prompts.py: {event_hits} -- "
        "this family should only write the rework brief and its sidecar, never emit events or "
        "write review-decision.json (that stays in OrchestratorApp.record_review, workflow.py)"
    )


def test_write_event_call_scanner_has_a_positive_control() -> None:
    """Positive control for the scanner above (this repo's own verification
    discipline: a result used as evidence needs a control proving the query
    itself isn't broken).

    The identical scanner, run over workflow.py, must find call sites for
    every one of COMMON's target names -- proving the scanner is capable of
    matching real write/event surface broadly, not merely capable of finding
    exactly the two sites this test expects in rework_prompts.py.
    """
    workflow_source = _WORKFLOW_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(workflow_source, filename=str(_WORKFLOW_PATH))

    found_names = {h["name"] for h in hits}
    assert found_names, "the scanner found zero call sites in workflow.py -- the scanner is broken"
    # workflow.py is expected to contain call sites for every one of these
    # names somewhere (OrchestratorApp._write_json, append_event/log_event
    # helpers, and _write_text_atomic call sites inside record_review's own
    # archive-copy logic) -- a scanner that only found a subset would be
    # silently blind to some of the forms it's supposed to catch.
    expected_broad_coverage = {"_write_json", "_write_text_atomic", "append_event", "log_event"}
    missing = expected_broad_coverage - found_names
    assert missing == set(), (
        f"scanner found zero call sites for {sorted(missing)} in workflow.py, which is known to "
        "contain calls to all of them -- the scanner's Call-node matching is broken"
    )


def test_write_event_call_scanner_ignores_docstring_mentions() -> None:
    """Control proving the AST scanner (unlike a raw substring search) does
    NOT count the four purely-prose mentions of write/event names in
    rework_prompts.py's own docstrings.

    Without this, a regression that accidentally switched the scanner back
    to a text-based search would silently inflate the hit count in the test
    above from 2 to 6, and nothing else in this file would catch it -- the
    exact-match assertion above would just start failing with a confusing
    "found 6, expected 2" rather than this test explaining why.
    """
    probe_source = (
        "def f(path):\n"
        '    """Mirrors OrchestratorApp._write_json\'s tmp+replace shape.\n\n'
        "    A plain write_text here would be the same failure class that\n"
        "    _write_text_atomic exists to close -- see append_event too.\n"
        '    """\n'
        "    return None\n"
    )
    hits = _write_event_call_sites(probe_source)
    assert hits == [], (
        f"scanner counted prose-only mentions as call sites: {hits} -- it must walk ast.Call "
        "nodes only, never search raw source text"
    )
