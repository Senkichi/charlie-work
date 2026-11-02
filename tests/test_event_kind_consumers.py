"""Issue #1364: every emitted event ``kind`` is either consumed or declared audit-only.

The single most recurrent defect class in this codebase's review history is
*signal without a consumer*: an event kind gets emitted (into ``state.json``'s
ring and ``events.db``) and nothing ever reads it. The signal exists, so the
author and reviewer both believe the condition is "handled"; in reality it is
write-only. Standing review guidance is "grep per EVENT KIND, not per
subsystem" -- this test applies that check structurally, once, the same way
``tests/test_dormant_fleet_marking.py`` derives the dormant-module set from
the import graph instead of hand-maintaining it.

Two sets are built by static analysis (AST, not regex -- kinds are string
literals but call *shapes* vary):

**Emitted kinds** -- every string-literal ``kind``/``event_kind`` argument
passed to the sanctioned emitters (``append_event``, ``log_event``,
``OrchestratorApp._record_event``, the ``WriteGate`` equivalents that share
their name, and the ``_route_to_rework`` wrapper), across every module under
``src/charlie_work``. A call site whose kind argument cannot be reduced to a
literal string is a **failure** unless the emitting line carries a
``# event-consumer:`` marker (see below) -- a variable or f-string kind is
exactly how a kind escapes the registry unnoticed.

**Consumed kinds** -- a kind counts as consumed if its literal string turns
up in a read position: a ``query_events(kind=...)`` call, an equality/
membership comparison against an event's ``kind`` field (including via a
dict/frozenset classification table), anywhere in the heartbeat scripts --
``scripts/heartbeat_check.py`` plus ``scripts/heartbeat_event_alarms.py``,
the module #1895 extracted the events.db anomaly checks into -- or anywhere
under ``tests/`` (a weak, test-only consumer, reported separately
so it can be upgraded deliberately).

**The escape hatch** is declared at the emission site, never in a
hand-maintained central list (which would itself rot): a trailing comment
``# event-consumer: audit-only <justification>``, ``# event-consumer:
<pointer>``, or ``# event-consumer: pending #NNNN``. The marker's own source
line is read directly (line-based, but only over a *comment*, never over code
structure) against the AST call site it decorates -- a marker that decorates
no real emission site is orphaned and fails the same as a missing one.

Baseline handling (issue's own explicit instruction): the backlog this test
surfaced was NOT mass-marked audit-only. Every one of the 13 ``audit-only``
markers below carries a one-clause justification tied to the surrounding
code (usually: the real state mutation already happened inline, or a sibling
kind is the actionable one). The two kinds that originally looked like they
should have a real consumer -- ``draft_pr_blocked`` and
``venv_editable_anchor_violation`` -- carried ``pending #1366`` markers until
issue #1366 added real heartbeat consumers for each
(``check_draft_pr_blocked_events`` -- since #1895 in
``scripts/heartbeat_event_alarms.py`` -- and
``check_supervisor_venv_refusal`` in ``scripts/heartbeat_check.py``); the
markers were dropped at the emission sites and the pending backlog is now
empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _dws_emit_scan import is_decide_module
from _event_kind_scan import Report, _analyze, _parse_marker, _scan_emit_sites

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "charlie_work"
TESTS_DIR = REPO_ROOT / "tests"
# The heartbeat consumers are physically split across the entry script and
# the sibling module #1895 extracted the events.db anomaly checks into --
# scanning only heartbeat_check.py would silently reclassify the moved
# kinds' literal strings as test-only consumption.
HEARTBEAT = (
    REPO_ROOT / "scripts" / "heartbeat_check.py",
    REPO_ROOT / "scripts" / "heartbeat_event_alarms.py",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _real_tree_report() -> Report:
    """Own the full-tree extractor scan so the measured call stays the assertion.

    The three real-tree tests below (`test_the_extractor_actually_reaches_things`,
    `test_every_emitted_kind_is_consumed_or_declared`, and the #1895 regression
    pin) each used to run the whole `_analyze(SRC, TESTS_DIR, HEARTBEAT)` walk in
    their own call phase. One walk costs the per-process cold `ast.parse` of every
    top-level ``test_*.py`` under ``tests/`` (~850 files -- ``tests/`` is
    deliberately outside the ``_src_ast`` cache) plus two full AST passes over the
    888-module ``src/charlie_work`` tree -- what the ledger flagged 2.1x on the
    canary (issue #2885: recent median 14.126 s / n=10 vs baseline 6.886 s / n=499).

    One module-scoped fixture computes the report ONCE per pytest process and
    hands the same :class:`Report` to every consumer: the assertions stay in the
    tests, verbatim, while the shared scaffold charges to fixture setup where
    ``ci-fleet ledger report --section slow-fixtures`` owns it (the accepted
    #2662/#2867/#2875/#2882 remedy family). Sharing also removes the two
    redundant whole-tree walks a module run used to pay after the first
    consumer's call. The ``Report`` is frozen, derived, read-only scan data --
    same discipline as the ``_src_ast`` shared trees ("shared trees must not be
    mutated"); the consumers only read/iterate, and the synthetic-tmp-tree tests
    below keep building their own reports.
    """
    return _analyze(SRC, TESTS_DIR, HEARTBEAT)


def test_the_extractor_actually_reaches_things(_real_tree_report: Report) -> None:
    """Positive control / canary (AC1).

    An empty or near-empty emitted set would mean the extractor is broken,
    and every assertion below would be measuring nothing -- the same "an
    absence is not evidence until you show it could have been non-empty"
    shape as ``test_dormant_fleet_marking.py``'s reachability control. Pin a
    floor, and pin the specific kinds this session's real incidents relied
    on, so a regression in the AST walk (or in the Call-to-local-function
    resolution ``self_deploy_{failed,succeeded}`` specifically exercises)
    fails loudly instead of silently under-collecting.
    """
    report = _real_tree_report
    assert len(report.emitted_kinds) > 100, (
        f"emitted-kind extraction found only {len(report.emitted_kinds)} kinds -- "
        "the AST walk is broken"
    )
    required = {
        "loop_started",
        "loop_completed",
        "fleet_pass_config_error",
        "review_verdict_missed",
        "self_deploy_failed",
        "self_deploy_succeeded",
    }
    missing = required - report.emitted_kinds
    assert not missing, f"canary kinds not found by the extractor: {sorted(missing)}"


def test_every_emitted_kind_is_consumed_or_declared(
    _real_tree_report: Report, request: object
) -> None:
    """The core assertion (AC5/AC6): the real repo's current inventory.

    Every literal emitted kind is either consumed (src/heartbeat/tests) or
    carries a valid ``# event-consumer:`` marker at one of its emission
    sites. Every dynamic (non-literal) call site carries a marker too (AC4).
    A failure here names the offending kind(s)/site(s) and the two remedies:
    add a consumer, or mark it (audit-only with a justification, or
    ``pending #NNNN`` pointing at a real tracking issue).
    """
    report = _real_tree_report

    if report.unaccounted:
        lines = []
        for entry in report.unaccounted:
            where = "; ".join(f"{s.path}:{s.lineno}" for s in entry.sites) or "<no emission site>"
            lines.append(f"  {entry.kind} (emitted at {where})")
        assert False, (
            "event kind(s) emitted with no consumer and no marker -- add a consumer "
            "(query_events/the heartbeat scripts/a test), or mark the emission site "
            "`# event-consumer: audit-only <reason>` or `# event-consumer: pending #NNNN`:\n"
            + "\n".join(lines)
        )

    if report.unmarked_dynamic:
        lines = [
            f"  {u.site.path}:{u.site.lineno} in {u.site.scope}() -- kind expression "
            f"`{u.site.kind_source}` is not a string literal (variable/f-string/call); "
            "dynamic kinds escape the registry and MUST carry a "
            "`# event-consumer:` marker"
            for u in report.unmarked_dynamic
        ]
        assert False, "non-literal kind argument(s) without a marker:\n" + "\n".join(lines)

    if report.orphan_markers:
        lines = [f"  {path}:{lineno}" for path, lineno in report.orphan_markers]
        assert False, (
            "`# event-consumer:` marker(s) decorate no real emission site (marker rot -- "
            "the emission was removed/moved, or the marker was misplaced):\n" + "\n".join(lines)
        )

    if report.invalid_markers:
        lines = [f"  {path}:{lineno}: {error}" for path, lineno, error in report.invalid_markers]
        assert False, "invalid `# event-consumer:` marker(s):\n" + "\n".join(lines)

    # Issue #1366 resolved the entire pending backlog: ``draft_pr_blocked``
    # and ``venv_editable_anchor_violation`` now have real heartbeat consumers
    # (``check_draft_pr_blocked_events`` in scripts/heartbeat_event_alarms.py
    # since #1895 / ``check_supervisor_venv_refusal`` in
    # scripts/heartbeat_check.py), so no emitted kind should carry a
    # ``pending #NNNN`` marker. A kind regressing back to pending means a
    # consumer was removed or a marker was re-added -- investigate rather
    # than silently accepting it.
    assert not report.pending, (
        "event kind(s) carry a `pending #NNNN` marker -- resolve the tracking "
        "issue by adding a real consumer instead of leaving the marker: "
        f"{[(p.kind, p.issue) for p in report.pending]}"
    )


def test_extracted_alarms_module_consumers_are_scanned(_real_tree_report: Report) -> None:
    """Regression pin for the #1895 extraction blind spot (PR #1908 review).

    The events.db anomaly checks physically moved out of
    ``scripts/heartbeat_check.py`` into ``scripts/heartbeat_event_alarms.py``,
    taking the ``draft_pr_blocked`` / ``ci_headroom_unavailable`` /
    ``infra_blocked_escalated`` literal strings with them. All three kinds
    also appear in ``tests/`` -- so a heartbeat scan covering only
    ``heartbeat_check.py`` would not leave them unaccounted (the test
    literals still satisfy ``is_consumed``); it would silently demote them
    to *test-only* consumers, the exact weak signal ``test_only_consumed``
    exists to flag. Pin that each kind is found in the scanned heartbeat
    literals and absent from ``report.test_only_consumed``.
    """
    report = _real_tree_report
    moved_kinds = {
        "ci_headroom_unavailable",
        "draft_pr_blocked",
        "infra_blocked_escalated",
    }
    unemitted = moved_kinds - set(report.emitted_kinds)
    assert not unemitted, (
        f"kind(s) expected to be emitted from src/ were not extracted: {sorted(unemitted)}"
    )
    missing_literals = moved_kinds - set(report.heartbeat_literals)
    assert not missing_literals, (
        "kind(s) whose heartbeat consumer moved to heartbeat_event_alarms.py are absent "
        "from the scanned heartbeat literals -- the scan is not covering both files: "
        f"{sorted(missing_literals)}"
    )
    demoted = moved_kinds & set(report.test_only_consumed)
    assert not demoted, (
        "kind(s) with a real heartbeat consumer were classified as test-only consumed "
        "-- the heartbeat scan is missing heartbeat_event_alarms.py: "
        f"{sorted(demoted)}"
    )


def test_new_unconsumed_kind_without_marker_fails(tmp_path: Path) -> None:
    """AC2: a fresh ``log_event(..., "totally_new_kind", ...)`` with no
    consumer and no marker must be caught, naming the kind and file:line --
    proven via a synthetic module rather than mutating the real source tree.
    """
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "totally_new_kind", payload)\n',
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert "totally_new_kind" in report.emitted_kinds
    offending = {u.kind: u for u in report.unaccounted}
    assert "totally_new_kind" in offending, (
        "a brand-new emitted kind with no consumer and no marker must be reported as unaccounted"
    )
    site = offending["totally_new_kind"].sites[0]
    assert site.path == "injected.py"
    assert site.lineno == 4


def test_new_unconsumed_kind_with_marker_is_accepted(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        "    log_event(\n"
        '        path, "totally_new_kind", payload  # event-consumer: audit-only test fixture\n'
        "    )\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert not report.unaccounted
    assert not report.unmarked_dynamic
    assert not report.orphan_markers


def _sweep_fixture(tmp_path: Path, body: str) -> Path:
    root = tmp_path / "src"
    package = root / "dead_worker_sweep"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "decide_fixture.py").write_text(body, encoding="utf-8")
    return root


def test_sweep_decide_emit_without_consumer_or_marker_fails(tmp_path: Path) -> None:
    """#2112: ``emit("<kind>", ...)`` in a ``dead_worker_sweep/decide*.py`` is an emit site."""
    root = _sweep_fixture(
        tmp_path,
        "def decide():\n"
        '    return emit("fixture_unconsumed_kind", {})\n'
        "\n"
        "def decide_stalled():\n"
        '    return StateTxn(event_kind="fixture_unconsumed_txn_kind")\n',
    )
    report = _analyze(root, tests_root=None, heartbeat_paths=None)
    offending = {u.kind: u for u in report.unaccounted}
    assert set(offending) == {"fixture_unconsumed_kind", "fixture_unconsumed_txn_kind"}
    assert offending["fixture_unconsumed_kind"].sites[0].lineno == 2


def test_sweep_decide_emit_with_marker_is_accepted(tmp_path: Path) -> None:
    root = _sweep_fixture(
        tmp_path,
        "def decide():\n"
        '    return emit("fixture_unconsumed_kind", {})  # event-consumer: audit-only fixture\n',
    )
    report = _analyze(root, tests_root=None, heartbeat_paths=None)
    assert not report.unaccounted
    assert not report.orphan_markers


def test_emit_outside_a_decide_module_is_not_a_sweep_site(tmp_path: Path) -> None:
    root = _sweep_fixture(tmp_path, "")
    (root / "dead_worker_sweep" / "apply_fixture.py").write_text(
        'def go():\n    return emit("fixture_unconsumed_kind", {})\n', encoding="utf-8"
    )
    assert not _analyze(root, tests_root=None, heartbeat_paths=None).emit_sites


def test_the_real_sweep_decide_modules_yield_emit_sites() -> None:
    """Positive control: the real package is reached, so a clean report is not a blind one."""
    sites = [s for s in _scan_emit_sites(SRC) if s.func_name in {"emit", "event_kind"}]
    assert len(sites) > 10, f"only {len(sites)} sweep emit sites found -- the walk is broken"
    assert all(is_decide_module(s.path) for s in sites)


def test_non_literal_kind_argument_without_marker_fails(tmp_path: Path) -> None:
    """AC4: a variable/f-string kind with no marker fails, explaining why."""
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload, some_dynamic_kind):\n"
        "    log_event(path, some_dynamic_kind, payload)\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert len(report.unmarked_dynamic) == 1
    assert report.unmarked_dynamic[0].site.path == "injected.py"
    assert report.unmarked_dynamic[0].site.lineno == 4


def test_non_literal_kind_argument_with_marker_is_accepted(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload, some_dynamic_kind):\n"
        "    log_event(  # event-consumer: pending #9999\n"
        "        path, some_dynamic_kind, payload\n"
        "    )\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert not report.unmarked_dynamic


def test_marker_grammar_audit_only() -> None:
    marker = _parse_marker(
        '    log_event(path, "x", {})  # event-consumer: audit-only -- forensic record only',
        1,
    )
    assert marker is not None
    assert marker.kind == "audit-only"
    assert marker.valid


def test_marker_grammar_audit_only_without_justification_is_invalid() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: audit-only', 1)
    assert marker is not None
    assert not marker.valid


def test_marker_grammar_pending_with_issue() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: pending #123', 1)
    assert marker is not None
    assert marker.kind == "pending"
    assert marker.valid
    assert marker.pending_issue == 123


def test_marker_grammar_pending_without_issue_is_invalid() -> None:
    marker = _parse_marker('    log_event(path, "x", {})  # event-consumer: pending', 1)
    assert marker is not None
    assert not marker.valid


def test_marker_grammar_pointer_form() -> None:
    marker = _parse_marker(
        '    log_event(path, "x", {})  # event-consumer: dashboards.external_widget',
        1,
    )
    assert marker is not None
    assert marker.kind == "pointer"
    assert marker.valid


def test_orphan_marker_on_non_emitting_line_fails(tmp_path: Path) -> None:
    """AC3: a marker decorating a line with no real emission call site fails."""
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "x = 1  # event-consumer: audit-only nothing emits here\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert ("injected.py", 1) in report.orphan_markers


def test_self_match_is_not_a_consumer(tmp_path: Path) -> None:
    """The literal kind string appearing a second time in the SAME function
    that emits it (e.g. a comment, or an unrelated local comparison) must
    not be mistaken for a genuine downstream consumer.
    """
    module = tmp_path / "src"
    module.mkdir()
    (module / "injected.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "self_matching_kind", payload)\n'
        '    also_kind = {"self_matching_kind"}\n'
        '    if payload.get("kind") in also_kind:\n'
        "        pass\n",
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    offending = {u.kind for u in report.unaccounted}
    assert "self_matching_kind" in offending, (
        "a same-function 'consumer' occurrence must not exempt a kind from the unaccounted list"
    )


def test_genuine_downstream_consumer_in_a_different_scope_counts(tmp_path: Path) -> None:
    module = tmp_path / "src"
    module.mkdir()
    (module / "emitter.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "genuinely_consumed_kind", payload)\n',
        encoding="utf-8",
    )
    (module / "reader.py").write_text(
        "def check(events):\n"
        '    return [e for e in events if e["kind"] == "genuinely_consumed_kind"]\n',
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    offending = {u.kind for u in report.unaccounted}
    assert "genuinely_consumed_kind" not in offending


def test_subpackage_emit_site_is_seen_without_a_declared_list(tmp_path: Path) -> None:
    """Coverage control (#1687): a package directory that exists only in the
    tree -- named by no tuple, list, or set in this guard -- is scanned.

    Fails against the pre-#1687 implementation: ``_SCANNED_SUBPACKAGES`` only
    ever named ``orchestration``, so ``new_delegate_pkg/emitter.py`` would
    never reach the extractor and ``subpkg_emitted_kind`` would not enter
    ``emitted_kinds`` at all.
    """
    module = tmp_path / "src"
    module.mkdir()
    pkg = module / "new_delegate_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "emitter.py").write_text(
        "from .instrumentation import log_event\n\n"
        "def do_thing(path, payload):\n"
        '    log_event(path, "subpkg_emitted_kind", payload)\n',
        encoding="utf-8",
    )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    assert "subpkg_emitted_kind" in report.emitted_kinds
    offending = {u.kind: u for u in report.unaccounted}
    assert "subpkg_emitted_kind" in offending
    assert offending["subpkg_emitted_kind"].sites[0].path == "new_delegate_pkg/emitter.py"


def test_same_basename_in_two_subpackages_attributes_by_relative_path(
    tmp_path: Path,
) -> None:
    """Collision control (#1687): two files sharing a basename in different
    subpackages attribute emit sites and markers to their own relative path.

    Fails under basename keying: both files share the ``file_markers`` /
    ``file_sites`` slot ``"emit.py"``, so the later scan clobbers the earlier
    one's marker map -- ``pkg_a``'s own ``audit-only`` marker would be lost and
    its kind would turn up unaccounted.
    """
    module = tmp_path / "src"
    module.mkdir()
    for pkg_name, kind in (("pkg_a", "kind_from_pkg_a"), ("pkg_b", "kind_from_pkg_b")):
        pkg = module / pkg_name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        marker = "  # event-consumer: audit-only test fixture" if pkg_name == "pkg_a" else ""
        (pkg / "emit.py").write_text(
            "from .instrumentation import log_event\n\n"
            "def do_thing(path, payload):\n"
            f'    log_event(path, "{kind}", payload){marker}\n',
            encoding="utf-8",
        )
    report = _analyze(module, tests_root=None, heartbeat_paths=None)
    unaccounted = {u.kind: u for u in report.unaccounted}
    assert "kind_from_pkg_a" not in unaccounted, (
        "pkg_a's own audit-only marker was not honored -- file_markers keyed by "
        "basename let pkg_b's marker map clobber pkg_a's"
    )
    assert "kind_from_pkg_b" in unaccounted
    site = unaccounted["kind_from_pkg_b"].sites[0]
    assert site.path == "pkg_b/emit.py"
