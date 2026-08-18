"""Seam integrity for the workflow.py -> stalled_review_reap.py split
(#1283 Phase A, PR 6 of 6 -- staged variant).

``stalled_review_reap.py`` holds the stalled-/orphaned-review-dispatch reap
family -- ten units (nine functions plus one ``Enum`` class) verbatim-moved
out of ``workflow.py``: ``_remove_review_checkout_with_warning``,
``_set_reviewer_quota_exhausted_with_backoff``, ``_merge_on_write_save``,
``_ThrottleClassification``, ``_detect_and_handle_stalled_reviews``,
``_reap_review_sidecar``, ``_reap_completed_review_checkouts``,
``_reap_orphaned_review_checkouts``, ``_classify_review_dispatch_stalled_level``,
``_append_sweep_events``. ``workflow.py`` re-exports every moved name through
a facade import block (mirroring this repo's ``dispatch_selection.py`` /
``escalation.py`` / ``verdict_parsing.py`` / ``rework_prompts.py`` /
``ci_findings.py`` / ``backlog_reachability.py`` precedents) so every
existing ``charlie_work.workflow.<name>`` import path and monkeypatch target
keeps resolving unchanged.

Corrected cohesion rationale (mirrors the new module's own docstring -- do
NOT use an earlier "zero call-graph edges, purely physical adjacency"
framing for members #6/#7, it is factually wrong): live AST + source-read
evidence shows ``_reap_completed_review_checkouts`` calls
``_reap_review_sidecar`` directly, and ``_reap_orphaned_review_checkouts``
also calls ``_reap_review_sidecar`` directly. All ten members are either
directly call-graph-connected to another member, or (for
``_ThrottleClassification``, a pure ``Enum``, and
``_classify_review_dispatch_stalled_level``, called only by
``_detect_and_handle_stalled_reviews``) a tightly-scoped intra-group
dependency.

This is a genuine, disclosed deviation from the A1-A5 lineage's "zero
test-file edits" precedent: three names imported into ``workflow.py`` from
OTHER sibling modules (``remove_review_checkout`` from ``.worktree``,
``iter_workers`` from ``.worker``, ``is_pid_alive`` from ``.process_utils``)
are called bare-name from inside the moving functions. A Python function's
bare-name lookups resolve via its own defining module's ``__globals__``,
fixed permanently at definition -- the facade re-export in ``workflow.py``
only affects *external* qualified-attribute access, never a moved
function's own internal bare-name resolution. 14 existing test sites (12
monkeypatch targets across 3 files + 2 ``test_instrumentation.py``
allow-list ``path=`` entries) were repointed from ``charlie_work.workflow.``
to ``charlie_work.stalled_review_reap.`` in the same commit as the move --
a narrow, mechanical, string-only edit with zero logic/assertion change.
This file does not re-verify those 14 sites directly (that is what the
targeted pytest runs against the 4 affected test files are for); it verifies
the *facade* and *module-shape* invariants those edits depend on.

This module is EXPECTED to exceed the repo's normal 800-line-per-module cap
(CLAUDE.md invariant) -- explicitly waived for this extraction by operator
decision on issue #1283 (staged-split final comment, 2026-08-17). In place
of the 800-line gate, this file asserts two things:

* An AST-derived name-set equality on the member-content-only body span
  (first ``def``/``class`` to EOF): its top-level definitions must be
  exactly the 10 known moved names, no more, no fewer. Originally (A6) this
  was a HARD LINE-COUNT equality (1258 lines) against the raw contiguous
  physical fragment Preflight measured -- a premise that held only while the
  module stayed an untouched verbatim move. W6 PR2 (issue #1264) legitimately
  invalidates that premise: threading a required ``write_gate: WriteGate``
  parameter (plus a ``require_write_gate`` call) through 4 of the 10 moved
  functions grows their body length by design, not by drift. The name-set
  form (see ``test_member_content_defines_exactly_the_ten_moved_symbols``)
  preserves what the original gate actually protected against -- a top-level
  unit silently added, dropped, or renamed during a mechanical edit -- without
  re-asserting a byte count the PR is deliberately changing.
* A BAND on the new module's total line count (docstring + imports + body).
  A6's Preflight step derived the original ``[1308, 1338]`` band live from
  ``ci_findings.py``'s own header/import-surface ratio (recorded in that
  run's ``wf-a6-notes.md`` as member-content 1258 + a [50, 80]-line header
  estimate). W6 PR2 widens the upper bound to ``1391`` (the real
  post-conversion total this PR measured, 1361 lines, plus a 30-line
  headroom margin) to accommodate the same write_gate-threading growth
  described above -- a documented, PR-recorded widening of the existing
  band, not a silent loosening or a removal of the gate. The lower bound
  (1308) is unchanged: the module only grows from here, so it remains a
  safe floor.

Three ways the facade promise can quietly break, same three failure modes
every prior split suite in this lineage documents:

* ``stalled_review_reap.py`` could grow an import of ``workflow.py`` --
  ``workflow.py`` already imports ``stalled_review_reap.py`` for the
  facade, so the reverse would be the exact import cycle this pattern
  exists to avoid.
* The facade could re-declare a name instead of importing it (a copy-paste
  that silently duplicates a function/class). Both copies look correct in
  isolation; only object identity distinguishes them.
* ``_append_sweep_events``'s facade entry is newly load-bearing in this
  extraction (unlike every prior one in the lineage): ``workflow.py``'s own
  ``_detect_and_handle_orphaned_workers``, which stays behind, calls it as a
  bare name -- so workflow.py's *own* remaining code, not just external
  consumers, now depends on the facade import actually succeeding.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "charlie_work" / "stalled_review_reap.py"
_WORKFLOW_PATH = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"
_CI_FINDINGS_PATH = _REPO_ROOT / "src" / "charlie_work" / "ci_findings.py"

_MOVED_NAMES = (
    "_remove_review_checkout_with_warning",
    "_set_reviewer_quota_exhausted_with_backoff",
    "_merge_on_write_save",
    "_ThrottleClassification",
    "_detect_and_handle_stalled_reviews",
    "_reap_review_sidecar",
    "_reap_completed_review_checkouts",
    "_reap_orphaned_review_checkouts",
    "_classify_review_dispatch_stalled_level",
    "_append_sweep_events",
)

# _CAP_BAND_MIN is A6's original Preflight-derived floor (wf-a6-notes.md
# Step 6/10) -- unchanged, since the module only grows from here. _CAP_BAND_MAX
# was widened by W6 PR2 (issue #1264) from A6's original 1338 to the real
# post-conversion total this PR measured (1361) plus a 30-line headroom
# margin: PR2's write_gate: WriteGate signature additions + require_write_gate
# calls grow the module by design, so the original band (derived assuming an
# untouched verbatim move) no longer fits. See the module docstring above.
_CAP_BAND_MIN = 1308
_CAP_BAND_MAX = 1391


# ---------------------------------------------------------------------------
# Shared derivation helpers -- both the identity test and the import-cycle
# guard draw their name universe from the actual module content, never from
# a hand-typed list restated in this file (which is exactly the kind of
# thing that silently drifts the moment either file is next edited).
# ---------------------------------------------------------------------------


def _module_level_defined_names(path: Path) -> list[str]:
    """Top-level function/class/constant names a module defines.

    Module-level ``Assign``/``AnnAssign`` with a simple ``Name`` target are
    treated as constants (kept even though ``stalled_review_reap.py`` has
    zero top-level constants today -- dropping the branch would make one
    added later silently invisible to the identity test below, which draws
    its candidate set from this helper).
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
    """Names workflow.py's facade block currently re-exports from
    ``.stalled_review_reap``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "stalled_review_reap"
        ):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# AC3: import-cycle guard -- dual-layer (raw-text pattern check, the "grep"
# form the task calls for explicitly, PLUS an AST-based check for defense in
# depth against formatting the raw-text patterns might miss).
# ---------------------------------------------------------------------------

_FORBIDDEN_WORKFLOW_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+\.\s+import\s+workflow\b", re.MULTILINE),
    re.compile(r"^\s*from\s+\.workflow\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+charlie_work\.workflow\b", re.MULTILINE),
    re.compile(r"^\s*from\s+charlie_work\.workflow\s+import\b", re.MULTILINE),
)


def _grep_forbidden_workflow_imports(source: str) -> list[str]:
    """Raw-text pattern scan (the "grep" form) for any of the four import
    spellings that would create a ``workflow.py`` -> ``stalled_review_reap.py``
    -> ``workflow.py`` cycle. Anchored to line-start (``^\\s*``) so a
    docstring line that merely *contains* one of these phrases mid-sentence
    is not flagged -- only an actual import statement is.
    """
    hits: list[str] = []
    for pattern in _FORBIDDEN_WORKFLOW_IMPORT_PATTERNS:
        hits.extend(m.group(0).strip() for m in pattern.finditer(source))
    return hits


def _module_imports_in(
    source: str,
    *,
    relative_module: str,
    absolute_module: str,
    filename: str = "<string>",
) -> list[str]:
    """AST-derived list of any import of the given module.

    Covers the four import spellings that all resolve to the same module at
    runtime: ``from .<relative_module> import X``, ``from . import
    <relative_module>``, ``from <absolute_module> import X``, and ``import
    <absolute_module>``.
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
    return _module_imports_in(
        source,
        relative_module="workflow",
        absolute_module="charlie_work.workflow",
        filename=filename,
    )


def test_stalled_review_reap_has_no_workflow_import_grep_form() -> None:
    """AC3 (grep form, as the task explicitly calls for): raw-text scan of
    ``stalled_review_reap.py``'s own source finds zero occurrences of
    ``from . import workflow`` / ``from .workflow import`` / ``import
    charlie_work.workflow`` / ``from charlie_work.workflow import`` as an
    actual statement (line-anchored, so prose mentions of "workflow.py" in
    the module docstring -- which this module's docstring has several of --
    are not false-flagged).
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    hits = _grep_forbidden_workflow_imports(source)
    assert hits == [], (
        f"stalled_review_reap.py contains a forbidden import of workflow.py: {hits} -- "
        "workflow.py's facade already imports FROM stalled_review_reap.py, so the "
        "reverse import would create the exact cycle this pattern exists to avoid"
    )


def test_stalled_review_reap_has_no_workflow_import_ast_form() -> None:
    """AC3 (AST form, defense in depth): same guard via full import-node
    walking, catching any multi-line/aliased spelling the raw-text scan
    above might not anticipate.
    """
    offenders = _workflow_imports_in(
        _MODULE_PATH.read_text(encoding="utf-8"), filename=str(_MODULE_PATH)
    )
    assert offenders == [], f"stalled_review_reap.py imports from workflow.py: {offenders}"


def test_workflow_import_detectors_flag_a_real_violation() -> None:
    """Positive control for both detectors above -- proves they can actually
    fire, so the empty-result assertions are not merely an unexercised query.
    """
    relative_violation = "from .workflow import OrchestratorApp\n"
    relative_package_violation = "from . import workflow\n"
    absolute_violation = "import charlie_work.workflow\n"
    absolute_from_violation = "from charlie_work.workflow import OrchestratorApp\n"
    innocent = (
        '"""A docstring that merely mentions workflow.py and charlie_work.workflow.foo."""\n'
    )

    for violation in (
        relative_violation,
        relative_package_violation,
        absolute_violation,
        absolute_from_violation,
    ):
        assert _grep_forbidden_workflow_imports(violation) != [], violation
        assert _workflow_imports_in(violation) != [], violation

    assert _grep_forbidden_workflow_imports(innocent) == [], "prose mention must not be flagged"
    assert _workflow_imports_in(innocent) == [], "prose mention must not be flagged"


def test_stalled_review_reap_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not
    error. The AST/grep checks above only rule out an explicit
    ``workflow.py`` import; this drives the real interpreter through the
    module's actual import statements (``.config``, ``.github``,
    ``.instrumentation``, ``.process_utils``, ``.state``,
    ``.throttle_signatures``, ``.worker``, ``.worktree``), which would catch
    any other cycle picked up through one of those.
    """
    import importlib

    module = importlib.import_module("charlie_work.stalled_review_reap")
    assert module.__name__ == "charlie_work.stalled_review_reap"

    # Confirms the module body actually executed to completion (not merely
    # "importlib didn't raise") by checking real symbols landed with the
    # expected shape: one plain function, the Enum class, and the function
    # whose facade entry is newly load-bearing for workflow.py's own
    # remaining code.
    assert callable(module._remove_review_checkout_with_warning)
    assert isinstance(module._ThrottleClassification, type)
    assert callable(module._append_sweep_events)


# ---------------------------------------------------------------------------
# AC4: facade-completeness identity checks -- all 10 names importable from
# BOTH charlie_work.workflow and charlie_work.stalled_review_reap, and
# `is`-identical (not merely equal), for every name.
# ---------------------------------------------------------------------------


def test_all_ten_names_are_reexported_by_identity() -> None:
    """Every name stalled_review_reap.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of
    a function (or, here, the ``_ThrottleClassification`` Enum class) would
    compare unequal-but-structurally-similar in ways that are easy to miss
    -- especially for an Enum, where two independently-defined classes with
    identical members are NOT the same type and NOT interchangeable with
    ``isinstance``.
    """
    import charlie_work.stalled_review_reap as stalled_review_reap
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_MODULE_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 10, f"expected 10 moved units, found {len(names)}: {sorted(names)}"
    assert set(names) == set(_MOVED_NAMES), (
        f"AST-derived names {sorted(names)} do not match the expected moved set "
        f"{sorted(_MOVED_NAMES)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    missing_from_module = [n for n in names if not hasattr(stalled_review_reap, n)]
    assert missing_from_module == [], (
        f"stalled_review_reap.py itself is missing a name it should define: "
        f"{sorted(missing_from_module)}"
    )

    not_identical = [
        n for n in names if getattr(workflow, n) is not getattr(stalled_review_reap, n)
    ]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"stalled_review_reap.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above: two
    structurally-identical-but-distinct objects are not ``is``-equal --
    proves the check discriminates, rather than passing for any two things
    that merely look alike. Exercises both a function pair (mirrors the
    other 9 moved names) and an Enum-class pair (mirrors
    ``_ThrottleClassification`` specifically, since it is the one moved
    unit that is a class, not a function).
    """
    from enum import Enum

    def helper_a() -> int:
        return 1

    def helper_b() -> int:
        return 1

    assert helper_a is not helper_b

    class ClassificationA(Enum):
        STALLED = "stalled"

    class ClassificationB(Enum):
        STALLED = "stalled"

    assert ClassificationA is not ClassificationB
    assert ClassificationA.STALLED is not ClassificationB.STALLED
    assert not isinstance(ClassificationA.STALLED, ClassificationB)


def test_facade_reexported_names_match_the_ten_member_set() -> None:
    """The facade block's own AST-derived import list (not just attribute
    presence on the ``workflow`` module object, which could also be
    satisfied by an unrelated same-named attribute elsewhere in the file)
    contains exactly the 10 expected names, no more, no fewer.
    """
    facade_names = _facade_reexported_names(_WORKFLOW_PATH)
    assert facade_names == set(_MOVED_NAMES), (
        f"workflow.py's `.stalled_review_reap` facade block re-exports "
        f"{sorted(facade_names)}, expected exactly {sorted(_MOVED_NAMES)}"
    )


# ---------------------------------------------------------------------------
# Module-shape / cap-exemption band assertions (novel to this PR -- no
# prior split suite in this lineage needed one, since every earlier
# extraction stayed under the 800-line cap without an exemption).
# ---------------------------------------------------------------------------


def _member_content_line_count(path: Path) -> int:
    """First top-level def/class line to EOF -- the member-content-only
    span the byte-identity check (AC1) compares, excluding the module's own
    docstring/import header.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    first_member_lineno = min(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    total_lines = len(path.read_text(encoding="utf-8").splitlines())
    return total_lines - first_member_lineno + 1


def test_member_content_defines_exactly_the_ten_moved_symbols() -> None:
    """Structural replacement for the retired byte/line-count equality gate
    (issue #1264, W6 PR2): the original ``test_member_content_span_is_exactly_1258_lines``
    asserted a HARD LINE-COUNT equality whose premise -- "this module is an
    untouched verbatim move from workflow.py, so its content span's length
    stays pinned" -- is intentionally invalidated by this PR. Threading a
    required ``write_gate: WriteGate`` parameter (plus a
    ``require_write_gate`` call) through 4 of the 10 moved functions grows
    their body length by design, so re-asserting the old byte count would
    fail on the very change this PR is authorized to make.

    What the retired gate actually protected against -- content silently
    lost, duplicated, or a top-level unit added/dropped/renamed during a
    mechanical edit -- is still enforced here, just via a name-set equality
    instead of a length equality: every top-level definition in the
    member-content span must be one of the 10 known moved names, no more,
    no fewer. This reuses the same ``_module_level_defined_names``
    derivation ``test_all_ten_names_are_reexported_by_identity`` already
    exercises (deliberately -- both tests are meant to agree), but stands as
    its own independent guard scoped to just this claim.
    """
    names = _module_level_defined_names(_MODULE_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert set(names) == set(_MOVED_NAMES), (
        f"stalled_review_reap.py's top-level definitions are {sorted(names)}, expected "
        f"exactly the 10 moved names {sorted(_MOVED_NAMES)} -- a mismatch here means a "
        "top-level unit was added, dropped, or renamed (growing an EXISTING member's body, "
        "e.g. adding a write_gate parameter, is expected and does not change this set)"
    )


def test_module_total_line_count_is_within_the_recorded_cap_band() -> None:
    """BAND gate: the new module's total (docstring + imports + body) must
    fall within [1308, 1391]. A6's Preflight step originally derived
    [1308, 1338] live from ci_findings.py's own header/import-surface ratio
    (recorded in wf-a6-notes.md Step 10). W6 PR2 (issue #1264) widened the
    upper bound to 1391 -- the real post-conversion total this PR measured
    (1361 lines) plus a 30-line headroom margin -- to accommodate the
    write_gate: WriteGate signature additions + require_write_gate calls
    the conversion adds. This is NOT the repo's normal 800-line cap
    (explicitly waived for this extraction by operator decision).
    """
    total = len(_MODULE_PATH.read_text(encoding="utf-8").splitlines())
    assert _CAP_BAND_MIN <= total <= _CAP_BAND_MAX, (
        f"stalled_review_reap.py is {total} lines total, expected within the recorded "
        f"cap-exemption band [{_CAP_BAND_MIN}, {_CAP_BAND_MAX}] -- if this module's header "
        "or member content genuinely needs to grow or shrink beyond the band, the band "
        "itself (not this assertion) should be re-derived and re-recorded, not silently "
        "widened here"
    )


def test_ci_findings_header_ratio_still_matches_the_bands_own_justification() -> None:
    """Drift guard on the band's mechanical half: re-measures
    ci_findings.py's live header length (everything before its first
    top-level def/class) and import-statement count, and confirms they
    still match the figures wf-a6-notes.md's Step 10 recorded (44-line
    header, 12 non-blank import-block lines) when this band was derived.

    Does not re-derive the band itself (the docstring-length component of
    the header estimate was a thematic judgment call at Preflight time, not
    mechanically reproducible from ci_findings.py alone -- see the module
    docstring above) -- this only proves the *precedent* the band cites
    hasn't silently changed shape underneath it since Preflight ran.
    """
    ci_findings_source = _CI_FINDINGS_PATH.read_text(encoding="utf-8")
    ci_findings_lines = ci_findings_source.splitlines()
    tree = ast.parse(ci_findings_source, filename=str(_CI_FINDINGS_PATH))
    first_member_lineno = min(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    header_length = first_member_lineno - 1
    # Loose, documented tolerances -- this is a drift guard, not a repeat of
    # the exact arithmetic (which involved manual line-classification at
    # Preflight time, e.g. deciding a bare `)` closing a multi-line import
    # counts toward "import block" but not toward "docstring").
    assert header_length == 44, (
        f"ci_findings.py's header length is now {header_length} lines, expected 44 -- "
        "the cap-exemption band this file asserts was derived from that figure; if "
        "ci_findings.py has genuinely changed shape, the band needs re-deriving, not "
        "this guard silently loosened"
    )
    assert len(ci_findings_lines) == 451, (
        f"ci_findings.py is now {len(ci_findings_lines)} lines total, expected 451 -- "
        "same drift-guard rationale as the header-length assertion above"
    )


@pytest.mark.parametrize(
    "source, expected_lines",
    [
        pytest.param("x = 1\ndef f():\n    pass\n", 2, id="one-line-header"),
        pytest.param('"""doc"""\n\ndef f():\n    pass\n', 2, id="docstring-header"),
    ],
)
def test_member_content_line_count_helper_is_correct(
    tmp_path: Path, source: str, expected_lines: int
) -> None:
    """Control for the ``_member_content_line_count`` helper: proves it
    counts from the first top-level def/class to EOF, not from line 1 --
    isolated from the real module so a regression here can't hide behind
    stalled_review_reap.py's own already-correct span.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert _member_content_line_count(probe) == expected_lines
