"""Seam integrity for the workflow.py -> backlog_reachability.py split (#1283).

``backlog_reachability.py`` holds the backlog-reachability free-function
pair -- the shared open-blocker check (``_get_open_blockers_for_issue``) and
the unfiltered-backlog reachability classifier that calls it
(``classify_backlog_reachability``) -- verbatim-moved out of ``workflow.py``.
``workflow.py`` re-exports both moved names through a facade import block
(mirroring ``config.py``'s ``RunnerAllocationConfig`` pattern and this
repo's own ``dispatch_selection.py``/``escalation.py``/``verdict_parsing.py``/
``rework_prompts.py``/``ci_findings.py`` precedents) so every existing
``charlie_work.workflow.<name>`` import path and monkeypatch target keeps
resolving unchanged.

Unlike ``ci_findings.py``'s 8-name family (three mutually disconnected
sub-clusters), this pair IS call-graph connected --
``classify_backlog_reachability`` calls ``_get_open_blockers_for_issue``
directly -- but the pair as a whole is disconnected from every other
Phase-A family (A1-A5) already extracted from ``workflow.py``. Confirmed by
the A6 recon and operator-approved in issue #1283's newest comment
(2026-08-17) as its own standalone standard-shape PR.

This file mirrors ``tests/test_ci_findings_split.py`` structurally (itself
mirroring ``tests/test_rework_prompts_split.py``'s FIXED 3-branch
``_module_imports_in`` helper), scaled down for a 2-name pair instead of an
8-name family:

* Only ``classify_backlog_reachability`` has a live consumer anchor under
  ``tests/`` -- the direct import at ``tests/test_backlog_reachability.py:21``.
  ``_get_open_blockers_for_issue`` has NO anchor anywhere in
  ``tests/``/``scripts``/``src`` (confirmed by direct scan): its only
  caller is the bare-name call inside ``OrchestratorApp._get_open_blockers``
  in ``workflow.py`` itself, which is a bare-name reference, not a
  ``workflow.<name>``-qualified one, so the reference-form scan below
  (deliberately) does not count it. It is still required to be re-exported
  by the unconditional facade-obligation rule (AC4 covers that), same shape
  as ``ci_findings.py``'s 4 anchor-less names.
* The write/event-emission surface (AC8/AC9) asserts an EMPTY result, same
  as ``ci_findings.py`` -- both functions only read (``gh.are_issues_open``,
  ``gh.issue_list``), they never write a file or emit an event.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_BACKLOG_REACHABILITY_PATH = _REPO_ROOT / "src" / "charlie_work" / "backlog_reachability.py"
_WORKFLOW_PATH = _REPO_ROOT / "src" / "charlie_work" / "workflow.py"

_MOVED_NAMES = (
    "_get_open_blockers_for_issue",
    "classify_backlog_reachability",
    "compute_mention_coverage_map",
    "fetch_merged_prs_fail_open",
    "resolve_dispatch_mention_coverage",
    "scan_merged_pr_references",
)


# ---------------------------------------------------------------------------
# Shared derivation helpers -- both AC4 and AC5 draw their name universe from
# the actual module content, never from a hand-typed list.
# ---------------------------------------------------------------------------


def _module_level_defined_names(path: Path) -> list[str]:
    """Top-level function/class/constant names a module defines.

    Module-level ``Assign``/``AnnAssign`` with a simple ``Name`` target are
    treated as constants. Dunders are skipped. ``backlog_reachability.py``
    has zero top-level constants (both moved names are functions), but the
    ``Assign``/``AnnAssign`` branches are kept regardless: dropping them
    would make a constant added to the module later silently invisible to
    both the AC4 identity test and the AC5 completeness test below.
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
    """Names workflow.py's facade block currently re-exports from ``.backlog_reachability``."""
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"), filename=str(workflow_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "backlog_reachability"
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

    Covers four distinct import spellings that all resolve to the same
    module at runtime: ``from .<relative_module> import X``, ``from . import
    <relative_module>`` (bare package-relative), ``from <absolute_module>
    import X``, and ``import <absolute_module>``. This is the FIXED 3-branch
    ``ImportFrom`` handling ``test_rework_prompts_split.py`` introduced,
    copied here verbatim.
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


def test_backlog_reachability_has_no_workflow_import() -> None:
    """AC3: backlog_reachability.py must never import from workflow.py.

    workflow.py's facade imports FROM backlog_reachability.py; the reverse
    import would be the exact cycle the issue's own Traps section warns
    against.
    """
    offenders = _workflow_imports_in(
        _BACKLOG_REACHABILITY_PATH.read_text(encoding="utf-8"),
        filename=str(_BACKLOG_REACHABILITY_PATH),
    )
    assert offenders == [], (
        "backlog_reachability.py imports from charlie_work.workflow -- this creates an "
        f"import cycle with workflow.py's facade block: {offenders}"
    )


def test_workflow_import_detector_flags_a_real_violation() -> None:
    """Control for the AST detector above -- proves it can actually fire.

    Without this, a detector that had quietly become incapable of finding
    anything would leave the assertion above vacuously true forever.
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


def test_backlog_reachability_module_actually_imports_cleanly() -> None:
    """Behavioral form: importing the new module in isolation must not error.

    The AST check above only rules out one specific kind of cycle. This
    drives the real interpreter through the module's actual import
    statements (``.config``, ``.github``), which also catches any other
    cyclical dependency the module might have picked up.
    """
    import importlib

    module = importlib.import_module("charlie_work.backlog_reachability")
    assert module.__name__ == "charlie_work.backlog_reachability"

    assert callable(module._get_open_blockers_for_issue)
    assert callable(module.classify_backlog_reachability)


# ---------------------------------------------------------------------------
# AC4: seam-guard identity checks
# ---------------------------------------------------------------------------


def test_all_backlog_reachability_names_are_reexported_by_identity() -> None:
    """AC4: every name backlog_reachability.py defines resolves through the
    workflow.py facade to the SAME object, not a re-declared copy.

    Module-attribute IDENTITY (``is``), not equality.
    """
    import charlie_work.backlog_reachability as backlog_reachability
    import charlie_work.workflow as workflow

    names = _module_level_defined_names(_BACKLOG_REACHABILITY_PATH)
    assert names, "AST derivation found zero module-level names -- derivation is broken"
    assert len(names) == 6, f"expected 6 moved units, found {len(names)}: {sorted(names)}"
    assert set(names) == set(_MOVED_NAMES), (
        f"AST-derived names {sorted(names)} do not match the expected moved set "
        f"{sorted(_MOVED_NAMES)}"
    )

    missing_from_facade = [n for n in names if not hasattr(workflow, n)]
    assert missing_from_facade == [], (
        f"workflow.py's facade does not re-export: {sorted(missing_from_facade)}"
    )

    not_identical = [
        n for n in names if getattr(workflow, n) is not getattr(backlog_reachability, n)
    ]
    assert not_identical == [], (
        "workflow.py re-exports these names as objects DIFFERENT from "
        f"backlog_reachability.py's own -- the facade must import, never redeclare: "
        f"{sorted(not_identical)}"
    )


def test_identity_check_would_fail_on_a_redefined_name() -> None:
    """Positive control for the identity check above (mirrors
    test_ci_findings_split.py's own control).

    Two structurally-identical-but-distinct objects are not ``is``-equal --
    proves the check discriminates, rather than passing for any two things
    that merely look alike.
    """

    def helper_a() -> int:
        return 1

    def helper_b() -> int:
        return 1

    assert helper_a is not helper_b


# ---------------------------------------------------------------------------
# AC5: re-export completeness, derived from live consumer references
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """POSIX-style path for messages/comparisons, repo-relative when possible."""
    try:
        display = path.relative_to(_REPO_ROOT)
    except ValueError:
        display = path
    return display.as_posix()


def _consumer_referenced_names(candidates: set[str], search_roots: list[Path]) -> dict[str, str]:
    """name -> one file that reaches it through ``charlie_work.workflow``.

    Walks every ``.py`` file under ``search_roots`` for three reference
    forms: a direct ``ImportFrom`` of ``charlie_work.workflow``/``.workflow``,
    a ``workflow.<name>``/``wf.<name>`` attribute access, and the
    string-dotted monkeypatch form
    ``monkeypatch.setattr("charlie_work.workflow.<name>", ...)``.
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

    Only ``classify_backlog_reachability`` has a consumer reference outside
    workflow.py (``tests/test_backlog_reachability.py``).
    ``_get_open_blockers_for_issue`` has none -- its only caller is a
    bare-name call inside ``workflow.py`` itself, so this scan imposes no
    obligation for it. It is still required to be re-exported by the
    unconditional facade-obligation rule (AC4 covers that), just not because
    this live scan demands it.
    """
    candidates = set(_module_level_defined_names(_BACKLOG_REACHABILITY_PATH))
    referenced = _consumer_referenced_names(
        candidates,
        [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"],
    )

    # Positive control: an empty walk would pass the assertion below
    # vacuously.
    assert referenced, (
        "no consumer under tests/scripts/src references any backlog-reachability name "
        "through charlie_work.workflow -- the scan itself is broken"
    )

    facade_names = _facade_reexported_names(_WORKFLOW_PATH)
    missing = {n: f for n, f in sorted(referenced.items()) if n not in facade_names}

    assert missing == {}, (
        "workflow.py's facade re-export block is missing names real consumers still "
        f"reach through charlie_work.workflow: {missing}"
    )


def test_consumer_reference_scan_finds_the_known_anchor() -> None:
    """Control for the scan above: it must find the specific reference this
    extraction's recon identified by hand, so a regression in the scan's own
    patterns shows up here instead of silently shrinking the ``referenced``
    set in the completeness test above.

    ``classify_backlog_reachability`` resolves to
    ``tests/test_backlog_reachability.py`` (a real single-name import at
    line 21). ``_get_open_blockers_for_issue`` has NO anchor at all, in
    either ``tests/`` alone or the full ``tests+scripts+src`` scan.
    """
    candidates = set(_module_level_defined_names(_BACKLOG_REACHABILITY_PATH))
    referenced = _consumer_referenced_names(candidates, [_REPO_ROOT / "tests"])
    referenced_all = _consumer_referenced_names(
        candidates, [_REPO_ROOT / "tests", _REPO_ROOT / "scripts", _REPO_ROOT / "src"]
    )

    assert "classify_backlog_reachability" in referenced
    assert referenced["classify_backlog_reachability"] == "tests/test_backlog_reachability.py"

    # Issue #1337: resolve_dispatch_mention_coverage is imported by
    # tests/test_backlog_reachability.py's caller-side wiring tests.
    assert "resolve_dispatch_mention_coverage" in referenced
    assert referenced["resolve_dispatch_mention_coverage"] == "tests/test_backlog_reachability.py"

    assert "_get_open_blockers_for_issue" not in referenced, (
        "_get_open_blockers_for_issue unexpectedly has a tests/-only anchor"
    )
    assert "_get_open_blockers_for_issue" not in referenced_all, (
        "_get_open_blockers_for_issue unexpectedly has an anchor even with scripts/src"
    )

    # The remaining 4 names (compute_mention_coverage_map,
    # fetch_merged_prs_fail_open, scan_merged_pr_references) have no
    # tests/-only anchor -- they are reached only through workflow.py's
    # facade re-export and the thin wrapper methods on OrchestratorApp.
    assert (
        set(referenced.keys())
        == set(referenced_all.keys())
        == {"classify_backlog_reachability", "resolve_dispatch_mention_coverage"}
    )


def test_reference_scan_recognizes_the_import_form(tmp_path: Path) -> None:
    """Control: the direct-import reference form is individually detectable,
    isolated from the noise of the real tree.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from charlie_work.workflow import classify_backlog_reachability\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"classify_backlog_reachability"}, [tmp_path])

    assert "classify_backlog_reachability" in referenced


def test_reference_scan_ignores_a_backtick_docstring_mention(tmp_path: Path) -> None:
    """Control for the false-positive class this family's precedent already
    hit once (A1's ``scripts/heartbeat_check.py``): a backtick-quoted RST
    cross-reference in a docstring must not be mistaken for a real
    monkeypatch string-literal target.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def f():\n"
        '    """Companion to ``charlie_work.workflow.classify_backlog_reachability``."""\n'
        "    return None\n",
        encoding="utf-8",
    )

    referenced = _consumer_referenced_names({"classify_backlog_reachability"}, [tmp_path])

    assert referenced == {}


# ---------------------------------------------------------------------------
# AC8/AC9 (post-move re-confirmation): the write/event-emission surface in
# the moved units' final spans in backlog_reachability.py must be EXACTLY
# EMPTY.
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
    """AST-derived call sites matching COMMON's write/event-surface name list."""
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


def test_backlog_reachability_write_event_surface_is_exactly_empty() -> None:
    """AC8 (post-move, re-confirmed): backlog_reachability.py emits zero
    writes and zero events.

    Both functions only read (``gh.are_issues_open``, ``gh.issue_list``) --
    the required positive control that this zero-hits assertion is not
    merely a broken query is
    ``test_write_event_call_scanner_has_a_positive_control`` below.
    """
    source = _BACKLOG_REACHABILITY_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(source, filename=str(_BACKLOG_REACHABILITY_PATH))

    assert hits == [], (
        f"backlog_reachability.py has {len(hits)} write/event-emission call site(s), expected "
        f"zero: {hits} -- this pair should only read via the GitHub client, never write files "
        "or emit events"
    )


def test_write_event_call_scanner_has_a_positive_control() -> None:
    """Required positive control for the empty-result assertion above: runs
    the identical scanner over workflow.py, which is known to still contain
    ``append_event``/``log_event`` and ``_write_text_atomic`` call sites.
    """
    workflow_source = _WORKFLOW_PATH.read_text(encoding="utf-8")
    hits = _write_event_call_sites(workflow_source, filename=str(_WORKFLOW_PATH))

    found_names = {h["name"] for h in hits}
    assert found_names, "the scanner found zero call sites in workflow.py -- the scanner is broken"
    expected_broad_coverage = {"_write_json", "_write_text_atomic", "append_event", "log_event"}
    missing = expected_broad_coverage - found_names
    assert missing == set(), (
        f"scanner found zero call sites for {sorted(missing)} in workflow.py, which is known to "
        "contain calls to all of them -- the scanner's Call-node matching is broken"
    )


def test_write_event_call_scanner_ignores_docstring_mentions() -> None:
    """Control proving the AST scanner does NOT count prose mentions of
    write/event names in docstrings as call sites.
    """
    probe_source = (
        "def f(path):\n"
        '    """Reads via gh.issue_list; see append_event and\n'
        "    _write_text_atomic in workflow.py for the write-side counterpart\n"
        '    this function has no analog of.\n"""\n'
        "    return None\n"
    )
    hits = _write_event_call_sites(probe_source)
    assert hits == [], (
        f"scanner counted prose-only mentions as call sites: {hits} -- it must walk ast.Call "
        "nodes only, never search raw source text"
    )
