"""Tests for issue #1274 (W17), the detection half.

``_detect_ci_run_never_created`` (job-cannon, 2026-08-06/07) is PRE-EXISTING
behavior -- W17 does not modify it (binding comment item 1: "reuse the
existing detector, do not build a second, independent predicate"). This file
does two things, both about that existing method, neither of which is new
production code:

* AC1 -- a characterization/regression test pinning the detector's three-way
  contract ("zero suites vs failed suites vs pending -> only zero triggers")
  directly, by calling the method with hand-built inputs rather than routing
  through a full ``review()`` fixture. This is the same contract
  ``test_fix_event_dedup.py``'s ``test_ci_run_never_created_*`` tests already
  exercise end-to-end; this file pins the narrower unit-level shape the issue
  literally asks for, and says so.
* AC2 -- a structural fence, using the same AST parent-map technique as
  ``test_verdict_provenance_enforcement.py``'s bypass scanner, proving the
  design invariant from binding comment item 2: two independently-tuned
  grace periods gating the same underlying "is CI missing" condition is the
  invalid-state smell this codebase's design explicitly avoids.
  ``config.review.stale_checks_grace_minutes`` governs ONLY the
  post-retrigger wait (inside ``_attempt_stale_checks_retrigger``) and is
  never threaded into detection; ``workflow_runs_for_head`` (the one gh API
  call that answers "did Actions ever create a run for this head") had
  exactly one call site in ``src/``, inside ``_detect_ci_run_never_created``
  itself -- until #1936 registered a second consumer
  (``_drive_infra_rerun_or_escalate``, misc_checks.py, asking the different
  question "is this run terminal"). The fence is now a pair: the src-wide
  call-site SET is pinned by
  ``test_workflow_runs_for_head_call_sites_are_the_registered_ones``, and
  the stale-checks-internal uniqueness guarantee stays on the original
  leaf name -- the collect-only gate (#1538) fails a test rename as
  ``removed`` + ``missing_sibling`` and waiving one needs an
  operator-applied ``collect-gate-exempt`` label, so the fence is split
  rather than renamed.

  This fence test was verified by mutation during implementation (per the
  issue's own instruction): a scratch second call to
  ``self.gh.workflow_runs_for_head(...)`` was added inside
  ``_attempt_stale_checks_retrigger``, confirmed to fail
  ``test_exactly_one_workflow_runs_for_head_call_site_and_it_is_inside_the_detector``
  (call-site count became 2), and then removed. Not committed as a
  self-mutating test -- this repo's existing structural/AST fences
  (``test_verdict_provenance_enforcement.py``,
  ``test_closing_reference.py``) all follow the same shape: a static
  positive-count assertion against the real tree, verified by hand during
  authorship rather than re-mutated on every run.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from charlie_work.config import AutoMergeConfig, OrchestratorConfig
from charlie_work.janitor import JanitorVerdict
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub
from _src_ast import parsed, source_files

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "charlie_work"


class _FakeGitHubWithRuns(FakeGitHub):
    """Minimal double: only ``workflow_runs_for_head`` is under test here."""

    def __init__(self, runs: list[dict[str, Any]] | None) -> None:
        super().__init__()
        self._runs = runs

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        return self._runs


def _app(tmp_path: Path, *, runs: list[dict[str, Any]] | None) -> OrchestratorApp:
    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed",),
        enabled=True,
        ci_run_never_created_grace_minutes=5,
    )
    config = OrchestratorConfig(auto_merge=auto_merge)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _FakeGitHubWithRuns(runs)
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _stale_pr(head_sha: str = "abc123abc123") -> dict[str, Any]:
    # Real GitHub head SHAs are always hex -- require_valid_sha rejects the
    # suite-wide "sha-abc123" placeholder, so this fixture uses a conforming
    # value (matches test_fix_event_dedup.py's convention).
    stale_updated_at = (
        (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )
    return {"headRefOid": head_sha, "updatedAt": stale_updated_at}


def _missing_checks_verdict(missing: tuple[str, ...] = ("Tests passed",)) -> JanitorVerdict:
    return JanitorVerdict(
        ok=False,
        failures=("Required check(s) missing: Tests passed",),
        warnings=(),
        missing_required_checks=missing,
    )


# ---------------------------------------------------------------------------
# AC1: detection-predicate characterization (pre-existing behavior)
# ---------------------------------------------------------------------------


def test_zero_actions_runs_ever_created_triggers(tmp_path: Path) -> None:
    """(a) ``workflow_runs_for_head`` returns ``[]`` (a successful query that
    found zero runs) -- positive evidence Actions never created a run for
    this head -- returns the head SHA (triggers).
    """
    app = _app(tmp_path, runs=[])
    pr = _stale_pr()
    verdict = _missing_checks_verdict()

    result = app._detect_ci_run_never_created(pr, verdict, known_head=None)

    assert result == "abc123abc123"


def test_required_checks_failed_not_missing_does_not_trigger(tmp_path: Path) -> None:
    """(b) Required checks are present but FAILED, not missing --
    ``verdict.missing_required_checks`` is empty (the janitor only populates
    it for checks GitHub reports as absent, never for ones that ran and
    failed -- ``failed_required_checks`` is the separate field for that).
    The detector requires a missing check as a precondition and returns None
    before ever touching ``workflow_runs_for_head`` -- this is the
    ``verdict.missing_required_checks`` empty-guard at the top of the
    method, not a ``workflow_runs_for_head`` case at all. A fake GitHub with
    no ``workflow_runs_for_head`` override at all still works here, since the
    empty-guard returns before that call would happen; using ``runs=None``
    (query-would-fail) makes that ordering explicit rather than accidental.
    """
    app = _app(tmp_path, runs=None)
    pr = _stale_pr()
    verdict = JanitorVerdict(
        ok=False,
        failures=("Required check(s) failed: Tests passed",),
        warnings=(),
        failed_required_checks=("Tests passed",),
        missing_required_checks=(),
    )

    result = app._detect_ci_run_never_created(pr, verdict, known_head=None)

    assert result is None


def test_runs_pending_not_missing_does_not_trigger(tmp_path: Path) -> None:
    """(c) Checks are "missing" per the janitor gate, but Actions HAS
    already created run(s) for the head (pending/propagating, not
    never-created) -- ``workflow_runs_for_head`` returns a non-empty list.
    This is the discriminator the detector exists to isolate: "missing" alone
    is ambiguous between "never started" and "started, not reported yet";
    only a successful EMPTY response is positive evidence of the former.
    """
    app = _app(tmp_path, runs=[{"id": 1, "status": "queued"}])
    pr = _stale_pr()
    verdict = _missing_checks_verdict()

    result = app._detect_ci_run_never_created(pr, verdict, known_head=None)

    assert result is None


# ---------------------------------------------------------------------------
# AC2: no second detection grace window (binding comment item 2 fence)
# ---------------------------------------------------------------------------


def _guard_member_modules() -> tuple[ModuleType, ...]:
    """The distinct modules that host the two members this fence spans -- the
    retrigger (which reads ``stale_checks_grace_minutes``) and the detector
    (which must not). Both are located through ``OrchestratorApp`` so the scan
    follows each member to whatever module a later leaf moves it into, instead
    of hard-coding ``workflow.py``. Deduped by module object: before any leaf
    splits them they share one module, afterwards they may differ.
    """
    modules: list[ModuleType] = []
    for member in (
        OrchestratorApp._attempt_stale_checks_retrigger,
        OrchestratorApp._detect_ci_absence,
    ):
        module = inspect.getmodule(member)
        assert module is not None, f"could not locate module for {member!r}"
        if module not in modules:
            modules.append(module)
    return tuple(modules)


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_function_name(node: ast.AST, parents: dict[int, ast.AST]) -> str | None:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(id(current))
    return None


def test_stale_checks_grace_minutes_never_referenced_inside_the_detector() -> None:
    """``config.review.stale_checks_grace_minutes`` (the post-RETRIGGER wait)
    must never be read inside ``_detect_ci_run_never_created`` (the DETECTION
    method) -- that would recreate the "two independently-tuned grace periods
    gating the same condition" smell binding comment item 2 explicitly rules
    out. Structural (AST attribute-name scan), not textual: catches the field
    being threaded in under any call shape, not just a literal string match
    for "grace".
    """
    referencing_functions: set[str | None] = set()
    total_references = 0
    for module in _guard_member_modules():
        source = inspect.getsource(module)
        tree = ast.parse(source, filename=getattr(module, "__file__", module.__name__))
        parents = _build_parent_map(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "stale_checks_grace_minutes":
                total_references += 1
                referencing_functions.add(_enclosing_function_name(node, parents))

    # Positive control: the field IS referenced somewhere in the scanned source
    # (the retrigger-wait logic) -- an empty result here would mean the scan
    # itself is broken (wrong attribute name, wrong module), not that the
    # invariant holds.
    assert total_references > 0, (
        "expected at least one reference to stale_checks_grace_minutes in the "
        "retrigger's module (the retrigger-wait logic) -- scan found none, "
        "which means this test isn't actually exercising the field"
    )
    assert "_detect_ci_absence" not in referencing_functions
    assert referencing_functions == {"_attempt_stale_checks_retrigger"}


def test_exactly_one_workflow_runs_for_head_call_site_and_it_is_inside_the_detector() -> None:
    """Within the stale-checks guard modules -- the modules hosting the
    detector and the retrigger, located via ``_guard_member_modules()`` --
    ``workflow_runs_for_head`` still has exactly one call site, inside
    ``_detect_ci_run_never_created``.

    Before #1936 this fence asserted a single call site across all of
    ``src/``: ``workflow_runs_for_head`` was the one gh API call answering
    "did Actions ever create a run for this head", and a second,
    independently-gated call site anywhere else would be exactly the
    second-detector-predicate binding comment item 1 forbids. Issue #1936
    registered a second consumer OUTSIDE the stale-checks family
    (``_drive_infra_rerun_or_escalate`` in misc_checks.py asks "is this
    run terminal" -- a different question), so the src-wide assertion
    moved to the sibling set test below. This leaf keeps the narrower
    guarantee: inside the stale-checks machinery itself no second call
    site has appeared -- a differently-phrased second window inside
    ``_attempt_stale_checks_retrigger`` still shows up here.

    The leaf name is deliberately unchanged: the collect-only gate
    (#1538) reads a rename as ``removed`` + ``missing_sibling`` and fails
    the required check; waiving it needs an operator-applied
    ``collect-gate-exempt`` label, so the fence is split instead of
    renamed.

    Verified by mutation during implementation: temporarily adding
    ``self.gh.workflow_runs_for_head(head_sha)`` inside
    ``_attempt_stale_checks_retrigger`` made this test fail (2 call sites
    in the guard modules, the second inside the wrong function); removed
    after confirming the failure.
    """
    call_sites: list[tuple[str, str | None]] = []
    for module in _guard_member_modules():
        module_file = getattr(module, "__file__", None)
        assert module_file is not None, f"could not locate source file for {module!r}"
        tree = ast.parse(inspect.getsource(module), filename=module_file)
        parents = _build_parent_map(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "workflow_runs_for_head"
            ):
                enclosing = _enclosing_function_name(node, parents)
                call_sites.append((Path(module_file).name, enclosing))

    assert call_sites == [("reap_dispatch.py", "_detect_ci_absence")], (
        f"expected exactly one workflow_runs_for_head call site across the "
        f"stale-checks guard modules, inside _detect_ci_absence; "
        f"found {call_sites!r}"
    )


def test_workflow_runs_for_head_call_sites_are_the_registered_ones() -> None:
    """``workflow_runs_for_head`` (the one gh API call answering "did
    Actions ever create a run for this head" / "is this run terminal")
    has a deliberately small call-site set -- every site named here.

    ``_detect_ci_run_never_created`` (reap_dispatch.py) is the original
    consumer: the stale-checks detector. A second, independently-gated
    "did CI ever create a run" predicate anywhere else would be exactly
    the second-detector-predicate binding comment item 1 forbids and
    this fence exists to catch.

    Issue #1936 adds ``_drive_infra_rerun_or_escalate`` (misc_checks.py):
    a deferred infra-rerun follow-up probes the CONTAINING run's
    terminal status before re-calling ``gh run rerun`` -- a different
    question (per-run status, not run existence), deliberately registered
    rather than smuggled in. Call-site SET, not a text-pattern match for
    "grace" nearby -- a differently-phrased second window still shows up
    here as another ``ast.Call`` node. The stale-checks-internal half of
    the original fence (exactly one call site inside the detector) lives
    on the ``test_exactly_one_workflow_runs_for_head_call_site_...`` leaf
    above -- kept under its original name because the collect-only gate
    (#1538) enforces leaf-name stability.

    Verified by mutation during implementation: temporarily adding
    ``self.gh.workflow_runs_for_head(head_sha)`` inside
    ``_attempt_stale_checks_retrigger`` made this test fail (3 call sites,
    the third unregistered); removed after confirming the failure.
    """
    call_sites: list[tuple[str, str | None]] = []
    for py_file in source_files(_SRC_ROOT):
        tree = parsed(py_file)
        parents = _build_parent_map(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "workflow_runs_for_head"
            ):
                enclosing = _enclosing_function_name(node, parents)
                call_sites.append((py_file.name, enclosing))

    assert call_sites == [
        ("misc_checks.py", "_drive_infra_rerun_or_escalate"),
        ("reap_dispatch.py", "_detect_ci_absence"),
    ], f"unexpected workflow_runs_for_head call site(s); found {call_sites!r}"
