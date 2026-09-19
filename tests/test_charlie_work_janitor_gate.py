"""Janitor-gate fall-through behavior and the verdict_source scope fence.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import pytest
from _dispatch_fixtures import _fail_if_launched
from _fakes_github import FakeGitHubWithMissingRequired
from _review_fixtures import _required_checks_config
from charlie_work.config import ReviewDispatchConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_co_occurring_ci_red_branch_stays_inside_janitor_ok_gate() -> None:
    """Issue #1258 (AC8): a mutation-testing gap found in review that AC8's
    literal construction (disable the co-occurring branch's own boolean
    guard, expect a launch) cannot be satisfied against this architecture,
    and explains why, before pinning the mutation that CAN be.

    Why the local mutation is structurally inert: ``janitor.run_janitor``
    (janitor.py) always appends "Required check(s) failed: ..." to
    ``failures`` whenever ``failed_required_checks`` is truthy, and
    ``JanitorVerdict.ok = not failures`` -- so ``verdict.ok`` is NEVER True
    while CI is red, sole or co-occurring. ``review()``'s
    ``if not verdict.ok:`` gate (workflow.py) is therefore always entered on
    red CI, and every branch inside it is an early ``return`` ending in an
    UNCONDITIONAL default that overwrites ``status`` to ``"janitor_blocked"``
    and returns ``CommandResult(False, ...)`` before the packet-write/
    dispatch code textually after (i.e. outside) the gate is ever reached.
    Disabling ``is_co_occurring_check_failure_block``'s own guard therefore
    cannot produce a ``launch_claude_worker`` call -- it can only fall
    through to the pre-existing ``janitor_blocked`` stall, which
    ``test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework``'s
    ``launched == []`` assertion cannot distinguish from the branch actually
    firing.

    The mutation that DOES reach a launch is hoisting a red-CI exclusion out
    to the outer gate itself, e.g. rewriting
    ``if not verdict.ok:`` as
    ``if not verdict.ok and not bool(verdict.failed_required_checks):`` --
    that skips the fail-safe default entirely for every red-CI PR (sole or
    co-occurring) and falls through to the packet-write/dispatch path.
    Applied by hand against this diff and reverted immediately (not part of
    this suite's harness -- AST source-mutation isn't a fixture here), it
    made
    ``test_janitor_required_check_failure_with_co_occurring_body_failure_routes_to_rework``
    fail exactly at the ``_fail_if_launched`` fake's
    ``launch_claude_worker`` call:
    ``AssertionError: launch_claude_worker must not be called on red CI``,
    raised from ``dispatch_reviews`` -- a real, reachable launch-on-red-CI,
    not a hypothetical one.

    This test is the permanent guard against that specific hoist: it
    AST-scans ``review()`` and asserts (a) the outer gate's test is exactly
    ``not verdict.ok`` with no additional ``and``/``or`` operand, and (b) the
    ``is_co_occurring_check_failure_block`` branch stays lexically nested
    inside that outer gate's body rather than becoming a sibling of it (or
    being folded into its condition). Either change is exactly the refactor
    mistake that would reopen this gap; this test fails CI the moment either
    lands, rather than relying on a mutation that this architecture makes
    unreachable at the branch's own guard.

    Both assertions were verified live (positive control), applied by hand
    against this diff and reverted immediately (confirmed byte-identical via
    diff against a pre-mutation backup) -- neither is a mutation this suite
    runs automatically:
    - Rewriting the outer gate as
      ``if not verdict.ok and not bool(verdict.failed_required_checks):``
      makes the AST node a ``BoolOp``, not the ``UnaryOp``-wrapping-
      ``Attribute`` shape ``is_outer_gate`` matches, so ``outer_gates``
      drops to 0 and assertion (a) (``len(outer_gates) == 1``) fires:
      "found 0 -- a rewrite changed the outer janitor-blocked gate's shape".
    - Dedenting the ``is_co_occurring_check_failure_block`` ``if``-statement
      (and its body) by one level so it becomes a sibling statement
      immediately after the outer gate's closing ``)`` -- textually after,
      not inside, ``if not verdict.ok:`` -- still parses (this is valid
      Python) and still yields exactly one ``co_occurring_ifs`` match, so
      assertion (a) and the count check in (b) both stay green; it is
      specifically the ``nested_inside_outer_gate`` assertion that fires:
      "must stay lexically nested inside `if not verdict.ok:` ... or moved
      to be a sibling of it". This is the assertion that actually guards
      the sibling-hoist shape, distinct from the one guarding the
      condition-hoist shape above.
    """
    import ast

    src_path = Path(__file__).parents[1] / "src" / "charlie_work" / "workflow.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))

    review_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "review":
            review_fn = node
            break
    assert review_fn is not None, "could not find review() -- a rename invalidated this probe"

    def is_outer_gate(stmt: ast.AST) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        test = stmt.test
        return (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Attribute)
            and test.operand.attr == "ok"
            and isinstance(test.operand.value, ast.Name)
            and test.operand.value.id == "verdict"
        )

    outer_gates = [stmt for stmt in ast.walk(review_fn) if is_outer_gate(stmt)]
    assert len(outer_gates) == 1, (
        "expected exactly one `if not verdict.ok:` gate (no additional and/or "
        f"operand) in review(), found {len(outer_gates)} -- a rewrite changed "
        "the outer janitor-blocked gate's shape"
    )
    outer_gate = outer_gates[0]

    def has_co_occurring_guard(stmt: ast.AST) -> bool:
        if not isinstance(stmt, ast.If):
            return False
        names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
        return "is_co_occurring_check_failure_block" in names

    co_occurring_ifs = [stmt for stmt in ast.walk(review_fn) if has_co_occurring_guard(stmt)]
    assert len(co_occurring_ifs) == 1, (
        "expected exactly one `is_co_occurring_check_failure_block` guard in "
        f"review(), found {len(co_occurring_ifs)} -- a rename/duplication invalidated this probe"
    )
    co_occurring_if = co_occurring_ifs[0]

    nested_inside_outer_gate = any(
        stmt is co_occurring_if for stmt in ast.walk(outer_gate) if stmt is not outer_gate
    )
    assert nested_inside_outer_gate, (
        "the co-occurring CI-red branch must stay lexically nested inside "
        "`if not verdict.ok:`, not hoisted into the outer gate's own condition "
        "(e.g. `if not verdict.ok and not is_co_occurring_check_failure_block:`) "
        "or moved to be a sibling of it -- either change skips the fail-safe "
        "janitor_blocked default and is the one refactor mistake that makes "
        "launch_claude_worker reachable on red CI (confirmed by hand-mutation, "
        "see this test's docstring)"
    )


def test_scope_fence_no_verdict_source_added() -> None:
    """Issue #1258 (AC7, structural half): this item's diff must add NO
    ``verdict_source`` field/enum anywhere in the new CI-red gate (W8's
    "ci_gate_auto_reject provenance enum" -- W8 lands after W1).

    Originally this test also pinned ``ReviewConfig.stale_checks_grace_minutes``
    / ``max_retriggers`` absent, guarding against W1 (or #1258 itself)
    re-adding W17's fields prematurely. W17 (issue #1274, this same lane) has
    now landed those two fields on ``ReviewConfig`` on purpose -- see
    ``ReviewConfig.stale_checks_grace_minutes``/``stale_checks_max_retriggers``
    and their loader validation block in config.py. That half of this test is
    therefore retired; the ``verdict_source`` guard below is unrelated
    (W8/#1258 concern) and still applies.

    ``verdict_source`` already exists elsewhere in this codebase
    (``_reap_review_verdicts`` provenance, an unrelated pre-existing
    mechanism -- see workflow.py's own field of that name) -- this test does
    not (and must not) assert the string is absent from the whole repo, only
    that ``JanitorVerdict`` (the structure this item actually touches) never
    gained the field.
    """
    from charlie_work.janitor import JanitorVerdict

    janitor_verdict_fields = {f.name for f in dataclasses.fields(JanitorVerdict)}
    assert "verdict_source" not in janitor_verdict_fields


def test_missing_checks_only_pr_falls_through_to_janitor_blocked_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC7, behavioral half): a PR whose required checks are
    entirely MISSING (never reported), as opposed to FAILED, is the
    pre-existing "absent checks" code path this item must leave untouched
    -- it is classification-only (``is_missing_checks_only_block``, issue
    #1133) with no retrigger, and W17's not-yet-built
    stale-checks-grace/retrigger policy is what would eventually act on it.

    Both of this item's new/extended gates require
    ``verdict.failed_required_checks`` to be truthy
    (``is_check_failure_block`` and the new co-occurring branch alike) --
    a purely-missing check leaves that tuple empty, so neither can fire by
    construction. This PR must fall straight through to the passive
    ``janitor_blocked`` bookkeeping exactly as it did before this item's
    diff: no ``record_review`` decision, no ``review_dispatch_skipped_ci_red``
    event, no reviewer launch.

    ``review_dispatch`` is explicitly enabled here (``_required_checks_config()``
    alone leaves it at its ``enabled=False`` default) and the launch seam is
    driven for real via ``_fail_if_launched`` -- with dispatch left disabled,
    ``dispatch_reviews()`` returns ``launched_count == 0`` on its very first
    line for every fixture, sole-failure and co-occurring alike, which would
    make that assertion pass for any mutation of the gate. The positive
    control proving this zero is real, not vacuous, is the AC1 sibling
    ``test_janitor_all_checks_green_dispatches_reviewer_ci_red_kind_absent``,
    which drives the identical enabled seam and gets ``launched_count == 1``.
    """
    config = dataclasses.replace(
        _required_checks_config(), review_dispatch=ReviewDispatchConfig(enabled=True)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    launched = _fail_if_launched(monkeypatch)

    result = app.review(456)

    assert result.ok is False
    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["status"] == "janitor_blocked"
    assert pr_state["is_missing_checks_only_block"] is True
    # Neither the pre-existing sole-failure short-circuit nor the new
    # co-occurring branch fired -- no request_changes decision was recorded.
    assert "decision" not in pr_state
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []

    # No packet was written (the janitor gate blocked before packet-build),
    # so dispatch_reviews has structurally nothing to launch for this PR --
    # driven for real, with dispatch enabled and the launch seam wired to
    # fail loudly, not inferred from a disabled-dispatch early return.
    dispatch_result = app.dispatch_reviews()
    assert dispatch_result.ok is True
    assert dispatch_result.data["launched_count"] == 0
    assert dispatch_result.data["selected_count"] == 0
    assert launched == []
