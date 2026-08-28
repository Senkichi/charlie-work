"""Seam integrity for the workflow.py -> dead_worker_reap.py split (#1317,
spun off from #1283's Phase-A recon; seventh module in the same
verbatim-move lineage as ``dispatch_selection.py``, ``escalation.py``,
``verdict_parsing.py``, ``rework_prompts.py``, ``ci_findings.py``,
``backlog_reachability.py``, and ``stalled_review_reap.py`` (#1320)).

``dead_worker_reap.py`` holds the dead-worker/session-reap free-function
family -- 25 functions plus two threshold constants
(``STARTUP_DEATH_THRESHOLD_SECONDS``, ``_ZERO_ARTIFACT_ESCALATION_THRESHOLD``),
verbatim-moved out of ``workflow.py``. ``workflow.py`` re-exports every moved
name through a facade import block (the same pattern the six prior Phase-A
extractions plus ``stalled_review_reap.py`` use) so every existing
``charlie_work.workflow.<name>`` import path and monkeypatch target keeps
resolving unchanged.

``_detect_and_handle_orphaned_workers`` deliberately stays in ``workflow.py``
-- at ~1,384 lines it is heavily entangled with ``OrchestratorApp`` state
(dispatch scheduling, throttle bookkeeping, worker-view construction) in ways
that would require a real refactor, not a verbatim move, to extract cleanly.
This is the same judgment #1283 recorded for this exact function. It calls
many of the functions defined in this module by bare name; those calls
resolve correctly only because the facade import block in ``workflow.py``
binds each re-exported name into ``workflow.py``'s own module globals at
import time -- a Python function's bare-name lookups resolve via its own
defining module's ``__globals__``, fixed at definition, so the facade only
affects *external* qualified-attribute access, never
``_detect_and_handle_orphaned_workers``'s own internal bare-name resolution.
``test_orphaned_workers_stays_in_workflow_and_resolves_moved_names_via_facade``
below verifies this dependency actually works, not just that it is claimed.

Two disclosed judgment calls in family membership: ``_is_pr_updated_at_older_than``
and ``_dispatching_repo_name`` both have callers outside the reap family, but
moved anyway -- leaving them in ``workflow.py`` while their only definitions
of "dead worker" / "dispatching repo" context live here would recreate the
``workflow.py <-> dead_worker_reap.py`` import cycle this split exists to
avoid. Their non-family callers reach them through the facade, unchanged.

Favorable deviation from the ``stalled_review_reap.py`` precedent: this
extraction needed ZERO test-file monkeypatch-target edits. Every monkeypatched
name in this family is patched via ``charlie_work.workflow.<name>`` at a call
site that itself remains defined in ``workflow.py`` (either inside
``_detect_and_handle_orphaned_workers`` or an ``OrchestratorApp`` method) --
those call sites resolve the patched bare name via ``workflow.py``'s own
globals, which the facade import keeps populated correctly.

This module is EXPECTED to exceed the repo's normal 800-line-per-module cap
(CLAUDE.md invariant) -- the same operator-recorded exemption #1283 granted
``stalled_review_reap.py`` (2026-08-17 decision) to preserve verbatim-move/
byte-identity discipline over the cap. In place of the 800-line gate, this
file asserts two things, mirroring ``test_stalled_review_reap_split.py``:

* An AST-derived name-set equality on the module's top-level definitions:
  must be exactly the 27 known moved names, no more, no fewer.
* A BAND on the new module's total line count (docstring + imports + body),
  derived live from this PR's own measured total (2667 lines) with headroom
  margin on both sides for legitimate future per-function growth (e.g. a
  write_gate-threading conversion of the kind #1264 W6 PR2 did to
  ``stalled_review_reap.py``) without silently loosening the gate to "any
  size."

Three ways the facade promise can quietly break, same three failure modes
every prior split suite in this lineage documents:

* ``dead_worker_reap.py`` could grow an import of ``workflow.py`` --
  ``workflow.py`` already imports ``dead_worker_reap.py`` for the facade, so
  the reverse would be the exact import cycle this pattern exists to avoid.
* The facade could re-declare a name instead of importing it (a copy-paste
  that silently duplicates a function/constant). Both copies look correct in
  isolation; only object identity distinguishes them.
* ``_detect_and_handle_orphaned_workers``'s bare-name calls into this module
  are newly load-bearing for ``workflow.py``'s *own* remaining code, not just
  external consumers -- the facade import actually succeeding is a
  precondition for that function to even be callable.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _REPO_ROOT / "src" / "charlie_work" / "dead_worker_reap.py"
_WORKFLOW_PATH = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"

_MOVED_NAMES = (
    "STARTUP_DEATH_THRESHOLD_SECONDS",
    "_is_startup_death",
    "_worker_death_bounded_runtime_seconds",
    "_session_failed_relabeled_payload",
    "_emit_session_failed_relabeled",
    "_count_live_sessions",
    "_detect_stalled_sessions",
    "_detect_and_handle_stalled_sessions",
    "_worker_pid_alive",
    "_orphan_head_fingerprint",
    "_ZERO_ARTIFACT_ESCALATION_THRESHOLD",
    "_is_zero_artifact_dispatch_loop",
    "_sweep_orphan_processes_for_dead_sessions",
    "_log_worker_census",
    "_rework_pr_for_worker",
    "_reap_restore_rework_requested",
    "_is_pr_updated_at_older_than",
    "_is_pre_review_rework_candidate",
    "_route_dead_worker_to_pre_review_rework",
    "_classify_dead_sessions_and_update_throttle_state",
    "_safe_repo_slug",
    "_dispatching_repo_name",
    "_open_salvage_pr",
    "_salvage_already_landed",
    "_attempt_salvage",
    "_open_pr_for_orphaned_branch",
    "_issues_with_live_workers",
)

# Band derived live from this PR's own measured total (2667 lines): +/- 150
# lines of headroom on either side, enough to absorb a future write_gate
# threading pass (the same class of legitimate growth #1264 W6 PR2 applied
# to stalled_review_reap.py) without the band needing re-derivation for
# routine per-function growth, while still catching a gross structural
# change (a member silently dropped or duplicated).
_CAP_BAND_MIN = 2517
_CAP_BAND_MAX = 2817


# ---------------------------------------------------------------------------
# Shared derivation helpers -- both the identity test and the import-cycle
# guard draw their name universe from the actual module content, never from
# a hand-typed list restated in this file.
# ---------------------------------------------------------------------------


def _module_level_defined_names(path: Path) -> list[str]:
    """Top-level function/class/constant names a module defines."""
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
    ``.dead_worker_reap``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "dead_worker_reap"
        ):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


# ---------------------------------------------------------------------------
# Import-cycle guard -- dual-layer (raw-text pattern check plus an AST-based
# check for defense in depth against formatting the raw-text patterns might
# miss).
# ---------------------------------------------------------------------------

_FORBIDDEN_WORKFLOW_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+\.\s+import\s+workflow\b", re.MULTILINE),
    re.compile(r"^\s*from\s+\.workflow\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+charlie_work\.workflow\b", re.MULTILINE),
    re.compile(r"^\s*from\s+charlie_work\.workflow\s+import\b", re.MULTILINE),
)


def _grep_forbidden_workflow_imports(source: str) -> list[str]:
    """Raw-text pattern scan (the "grep" form) for any of the four import
    spellings that would create a ``workflow.py`` -> ``dead_worker_reap.py``
    -> ``workflow.py`` cycle. Anchored to line-start so a docstring line that
    merely *contains* one of these phrases mid-sentence is not flagged.
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


def test_dead_worker_reap_has_no_workflow_import_grep_form() -> None:
    """Grep-form check: raw-text scan of ``dead_worker_reap.py``'s own
    source finds zero occurrences of a forbidden workflow.py import as an
    actual statement.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    hits = _grep_forbidden_workflow_imports(source)
    assert hits == [], (
        f"dead_worker_reap.py contains a forbidden import of workflow.py: {hits} -- "
        "workflow.py's facade already imports FROM dead_worker_reap.py, so the "
        "reverse import would create the exact cycle this pattern exists to avoid"
    )


def test_dead_worker_reap_has_no_workflow_import_ast_form() -> None:
    """AST form, defense in depth: same guard via full import-node walking,
    catching any multi-line/aliased spelling the raw-text scan might not
    anticipate.
    """
    offenders = _workflow_imports_in(
        _MODULE_PATH.read_text(encoding="utf-8"), filename=str(_MODULE_PATH)
    )
    assert offenders == [], f"dead_worker_reap.py imports from workflow.py: {offenders}"


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


def test_dead_worker_reap_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not
    error. The AST/grep checks above only rule out an explicit
    ``workflow.py`` import; this drives the real interpreter through the
    module's actual import statements, which would catch any other cycle
    picked up through one of those.
    """
    import importlib

    module = importlib.import_module("charlie_work.dead_worker_reap")
    assert module.__name__ == "charlie_work.dead_worker_reap"

    # Confirms the module body actually executed to completion (not merely
    # "importlib didn't raise") by checking real symbols landed with the
    # expected shape.
    assert callable(module._is_startup_death)
    assert callable(module._detect_and_handle_stalled_sessions)
    assert isinstance(module.STARTUP_DEATH_THRESHOLD_SECONDS, int)
    assert isinstance(module._ZERO_ARTIFACT_ESCALATION_THRESHOLD, int)


# ---------------------------------------------------------------------------
# Facade-completeness identity checks -- all 27 names importable from BOTH
# charlie_work.workflow and charlie_work.dead_worker_reap, and `is`-identical
# (not merely equal), for every name.
# ---------------------------------------------------------------------------


def test_all_moved_names_are_reexported_by_identity() -> None:
    """Every name dead_worker_reap.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality: a re-declared copy of
    a function or constant would compare equal-but-distinct in ways that are
    easy to miss.
    """
    import charlie_work.dead_worker_reap as dead_worker_reap
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_MODULE_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 27, f"expected 27 moved units, found {len(names)}: {sorted(names)}"
    assert set(names) == set(_MOVED_NAMES), (
        f"AST-derived names {sorted(names)} do not match the expected moved set "
        f"{sorted(_MOVED_NAMES)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    missing_from_module = [n for n in names if not hasattr(dead_worker_reap, n)]
    assert missing_from_module == [], (
        f"dead_worker_reap.py itself is missing a name it should define: "
        f"{sorted(missing_from_module)}"
    )

    not_identical = [n for n in names if getattr(workflow, n) is not getattr(dead_worker_reap, n)]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"dead_worker_reap.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above: two
    structurally-identical-but-distinct objects are not ``is``-equal --
    proves the check discriminates, rather than passing for any two things
    that merely look alike.
    """

    def helper_a() -> int:
        return 1

    def helper_b() -> int:
        return 1

    assert helper_a is not helper_b

    const_a = 60
    const_b = 60
    # Small ints are cached by CPython, so identity alone can't discriminate
    # constants the way it discriminates functions/classes -- equality is
    # the correct check for the constant members, and this control documents
    # why the identity assertion above is still meaningful for them: it only
    # fails to catch a redefinition when the redefinition happens to produce
    # the exact same cached small-int object, which a real copy-paste bug
    # (different formula, different threshold value) would not.
    assert const_a == const_b


def test_facade_reexported_names_match_the_moved_set() -> None:
    """The facade block's own AST-derived import list (not just attribute
    presence on the ``workflow`` module object, which could also be
    satisfied by an unrelated same-named attribute elsewhere in the file)
    contains exactly the 27 expected names, no more, no fewer.
    """
    facade_names = _facade_reexported_names(_WORKFLOW_PATH)
    assert facade_names == set(_MOVED_NAMES), (
        f"workflow.py's `.dead_worker_reap` facade block re-exports "
        f"{sorted(facade_names)}, expected exactly {sorted(_MOVED_NAMES)}"
    )


# ---------------------------------------------------------------------------
# Module-shape / cap-exemption band assertions -- mirrors
# test_stalled_review_reap_split.py's approach for a module that exceeds the
# repo's normal 800-line cap under the same #1283 operator exemption.
# ---------------------------------------------------------------------------


def test_module_defines_exactly_the_27_moved_symbols() -> None:
    """Every top-level definition in dead_worker_reap.py must be one of the
    27 known moved names, no more, no fewer -- catches content silently
    lost, duplicated, or a top-level unit added/dropped/renamed during a
    mechanical edit.
    """
    names = _module_level_defined_names(_MODULE_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert set(names) == set(_MOVED_NAMES), (
        f"dead_worker_reap.py's top-level definitions are {sorted(names)}, expected "
        f"exactly the 27 moved names {sorted(_MOVED_NAMES)}"
    )


def test_module_total_line_count_is_within_the_recorded_cap_band() -> None:
    """BAND gate: the new module's total (docstring + imports + body) must
    fall within [2517, 2817] -- derived live from this PR's own measured
    total (2667 lines) with +/-150 lines of headroom on either side. This is
    NOT the repo's normal 800-line cap (explicitly waived for this
    extraction under the same #1283 operator exemption
    ``stalled_review_reap.py`` used).
    """
    total = len(_MODULE_PATH.read_text(encoding="utf-8").splitlines())
    assert _CAP_BAND_MIN <= total <= _CAP_BAND_MAX, (
        f"dead_worker_reap.py is {total} lines total, expected within the recorded "
        f"cap-exemption band [{_CAP_BAND_MIN}, {_CAP_BAND_MAX}] -- if this module's header "
        "or member content genuinely needs to grow or shrink beyond the band, the band "
        "itself (not this assertion) should be re-derived and re-recorded, not silently "
        "widened here"
    )


def test_orphaned_workers_stays_in_workflow_and_resolves_moved_names_via_facade() -> None:
    """Proves the facade dependency ``_detect_and_handle_orphaned_workers``
    relies on actually works, not just that it is claimed in the docstring
    above.

    AST-extracts ``_detect_and_handle_orphaned_workers``'s source from
    ``workflow.py``, finds every bare-Name load inside it that matches one
    of the 27 moved names, and asserts each one resolves as a real
    ``workflow`` module attribute post-import -- i.e. the facade import
    block actually populated ``workflow.py``'s own globals with these names.
    """
    import charlie_work.workflow as workflow

    source = _WORKFLOW_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_WORKFLOW_PATH))
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_detect_and_handle_orphaned_workers"
        ),
        None,
    )
    assert target is not None, (
        "_detect_and_handle_orphaned_workers not found as a top-level function in "
        "workflow.py -- it is supposed to stay there (see module docstring); if it "
        "moved, this test (and the extraction's central claim) needs updating"
    )

    moved_set = set(_MOVED_NAMES)
    referenced = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in moved_set
    }
    assert referenced, (
        "_detect_and_handle_orphaned_workers references zero moved names by bare name -- "
        "either the facade dependency this test exists to verify no longer applies, or "
        "the derivation above is broken"
    )

    missing = [n for n in sorted(referenced) if not hasattr(workflow, n)]
    assert missing == [], (
        f"_detect_and_handle_orphaned_workers calls these moved names by bare name, but "
        f"they do not resolve as workflow.py module attributes post-import: {missing} -- "
        "the facade import block is not populating workflow.py's own globals correctly"
    )
