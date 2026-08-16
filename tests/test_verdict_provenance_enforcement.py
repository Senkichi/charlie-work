"""Tests for issue #1265: ``verdict_provenance`` on every review-decision
writer.

Three writer shapes must independently carry the field (per the issue's
binding comment, which corrects the plan text where they conflict):

  (a) ``record_review()`` -- a required (no-default) keyword-only parameter,
      threaded through every call site.
  (b) ``_update_approval_head`` -- stamps ``"carried_forward"`` directly on
      both the decision-file patch and the event payload, bypassing
      ``record_review`` entirely (issue #638's carry-forward mechanism,
      which never routes through ``record_review`` by design).
  (c) The pending-reset ``decision_template`` written by ``review()`` when
      it regenerates a packet -- stamps an explicit ``None`` sentinel (not
      an omitted key): "pending" is not a verdict yet, so it has no
      provenance to report, but the key's presence keeps the no-default
      contract structurally checkable.

This file covers:
  AC1 -- every record_review() call site supplies the EXACT literal from the
         issue #1265 mapping table (9 sites: 8 in workflow.py across 6
         functions + 1 in cli.py -- includes _reconcile_stranded_verdicts'
         ``stranded_reconciliation``, a 9th enum value this lane's recon
         found that neither the plan nor the binding comment's table listed;
         see w8-impl-notes.md finding 1), not just that some non-None string
         was passed (that weaker check is AC5, below).
  AC2 -- record_review() has no default for verdict_provenance anywhere in
         its signature: a static AST check on the def, plus a runtime test
         that omitting the argument raises TypeError at the call boundary
         (Python's own argument binding) before any body code runs.
  AC3 -- carry-forward stamping, tested separately per tier/event-kind (not
         one code path exercised three times with identical assertions).
  AC4 -- pending-reset ``None`` sentinel (key present, value ``None``).
  AC5 -- AST bypass-enforcement scanner covering both ``record_review`` call
         sites and the two direct ``review-decision.json`` writers
         (carry-forward, pending-reset), mirroring
         ``test_closing_reference.py``'s scanner pattern: no hardcoded file
         list, alias-aware, a "scanned" positive control alongside an
         "offenders" list that must end up empty.
  AC6 -- the pre-existing ``session_metrics.verdict_source`` (which parser
         found the reviewer's fenced verdict block) is unchanged and
         coexists, uncollided, with the new top-level ``verdict_provenance``
         (which mechanism produced the verdict at all) on the same
         ``record_review`` event payload.
  AC7 -- ``merge_authorize``'s write is unaffected (regression only; it
         patches only ``authorized_override``, never a verdict field).
  AC8 -- ``"derived_from_prose"`` (issue #792's ``findings_channel`` value --
         what produced a verdict's *content*, not what produced the verdict)
         appears nowhere in the enum or as a literal anywhere in ``src/``.
  AC9 -- a fixture events.db / review-decision.json set drives every one of
         the 9 enum values (through record_review() for the 8 it accepts as
         a fresh decision, and through _update_approval_head for
         "carried_forward", which never routes through record_review) plus
         the pending-reset None sentinel, then queries events.db
         (``query_events``, the same indexed-column API the real dashboards
         use) and re-reads review-decision.json to confirm every row/write
         carries the field. Stands in for "events.db rows post-deploy all
         carry it" -- the fresh clone's own events.db has no populated rows
         and is off-limits regardless.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import VERDICT_PROVENANCE_VALUES, OrchestratorApp

from test_charlie_work import (
    FakeGitHub,
    _cross_family_app,
    _dispatch_reviews_app,
    _make_dead_review_sidecar,
    _merge_check_app,
    _set_review_dispatched_state,
    _write_review_packet,
)

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "charlie_work"


# ---------------------------------------------------------------------------
# AC3: _update_approval_head stamps "carried_forward" on BOTH the
# decision-file patch and the event payload, for all three carry-forward
# event kinds -- three separately parametrized cases (distinct old/new
# heads and expected event kind per case), not one path run three times
# with identical assertions.
# ---------------------------------------------------------------------------


def _carry_forward_app(tmp_path: Path) -> tuple[OrchestratorApp, Any]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    return app, paths


def test_carry_forward_patch_id_tier_stamps_provenance_on_decision_and_event(
    tmp_path: Path,
) -> None:
    """tier="patch-id" -> verdict_carried_forward_clean_rebase."""
    app, paths = _carry_forward_app(tmp_path)
    pr_number, issue_number = 456, 123
    decision_dir = paths.prs / f"pr-{pr_number}"
    decision_dir.mkdir(parents=True)
    decision = {
        "decision": "approved",
        "reviewed_head_sha": "old-patch-id",
        "reviewed_patch_id": "pid-xyz",
        # The ORIGINAL provenance -- carry-forward must overwrite this, not
        # preserve it, per the binding comment's "always stamps
        # carried_forward regardless of what produced the verdict".
        "verdict_provenance": "fresh_llm_review",
    }
    (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")

    applied = app._update_approval_head(
        pr_number,
        decision,
        "new-patch-id",
        old_head="old-patch-id",
        issue_number=issue_number,
        tier="patch-id",
    )
    assert applied is True

    on_disk = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert on_disk["verdict_provenance"] == "carried_forward"

    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "verdict_carried_forward_clean_rebase"]
    assert len(events) == 1
    assert events[0]["payload"]["verdict_provenance"] == "carried_forward"


def test_carry_forward_line_content_tier_stamps_provenance_on_decision_and_event(
    tmp_path: Path,
) -> None:
    """tier="line-content" -> verdict_carried_forward_line_content."""
    app, paths = _carry_forward_app(tmp_path)
    pr_number, issue_number = 456, 123
    decision_dir = paths.prs / f"pr-{pr_number}"
    decision_dir.mkdir(parents=True)
    decision = {
        "decision": "request_changes",
        "reviewed_head_sha": "old-line-content",
        "verdict_provenance": "ci_gate_auto_reject",
    }
    (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")

    applied = app._update_approval_head(
        pr_number,
        decision,
        "new-line-content",
        old_head="old-line-content",
        issue_number=issue_number,
        tier="line-content",
    )
    assert applied is True

    on_disk = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert on_disk["verdict_provenance"] == "carried_forward"

    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "verdict_carried_forward_line_content"]
    assert len(events) == 1
    assert events[0]["payload"]["verdict_provenance"] == "carried_forward"


def test_carry_forward_verified_sync_tier_stamps_provenance_on_decision_and_event(
    tmp_path: Path,
) -> None:
    """tier="verified-sync" (the default) -> verdict_carried_forward_verified_sync."""
    app, paths = _carry_forward_app(tmp_path)
    pr_number, issue_number = 456, 123
    decision_dir = paths.prs / f"pr-{pr_number}"
    decision_dir.mkdir(parents=True)
    decision = {
        "decision": "approved",
        "reviewed_head_sha": "old-verified-sync",
        "verdict_provenance": "rescue_review",
    }
    (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")

    applied = app._update_approval_head(
        pr_number,
        decision,
        "new-verified-sync",
        old_head="old-verified-sync",
        issue_number=issue_number,
        tier="verified-sync",
    )
    assert applied is True

    on_disk = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert on_disk["verdict_provenance"] == "carried_forward"

    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "verdict_carried_forward_verified_sync"]
    assert len(events) == 1
    assert events[0]["payload"]["verdict_provenance"] == "carried_forward"


# ---------------------------------------------------------------------------
# AC4: the pending-reset write stamps an explicit "verdict_provenance": None
# -- the key must be PRESENT with value None, not merely absent (an omitted
# key is indistinguishable from a bypass writer that never considered the
# contract).
# ---------------------------------------------------------------------------


def test_pending_reset_packet_stamps_none_sentinel_not_omission(tmp_path: Path) -> None:
    """review()'s fresh-packet path (no prior review-decision.json) writes
    the pending-reset decision_template. It must carry an explicit
    "verdict_provenance": None key."""
    app = _cross_family_app(tmp_path, enabled=False)
    decision_path = app.paths.prs / "pr-456" / "review-decision.json"
    assert not decision_path.exists()

    app.review(456)

    assert decision_path.exists()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "pending"
    assert "verdict_provenance" in decision, (
        "the pending-reset template omitted the key entirely -- it must be an "
        "explicit sentinel, not an absence"
    )
    assert decision["verdict_provenance"] is None


def test_pending_reset_voids_stale_terminal_decision_with_none_sentinel(
    tmp_path: Path,
) -> None:
    """The second pending-reset call site (an existing terminal decision
    voided because the head advanced past it, workflow.py ~12674) must stamp
    the same explicit None sentinel -- both write call sites share the one
    decision_template literal, so this pins that both are actually reached,
    not just the fresh-packet one."""
    app = _cross_family_app(tmp_path, enabled=False)
    decision_dir = app.paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": "sha-superseded",
                "verdict_provenance": "fresh_llm_review",
            }
        ),
        encoding="utf-8",
    )

    # The live PR head (sha-abc123, the FakeGitHub default) has moved past
    # the recorded reviewed_head_sha -- the terminal decision is stale and
    # must be voided back to the pending template.
    app.review(456)

    decision = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "pending"
    assert "verdict_provenance" in decision
    assert decision["verdict_provenance"] is None


# ---------------------------------------------------------------------------
# AC6: session_metrics.verdict_source (pre-existing, "which parser found the
# fenced verdict block") is unchanged and coexists, uncollided, with the new
# top-level verdict_provenance ("which mechanism produced the verdict at
# all") on the same record_review event payload for a fresh-LLM-review row.
# Driven through the real _reap_review_verdicts production path, not a
# hand-built payload, so the test is evidence about the actual wiring.
# ---------------------------------------------------------------------------


def test_reap_review_verdicts_payload_carries_both_provenance_fields(
    monkeypatch, tmp_path: Path
) -> None:
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    reviews_dir = app._layout.reviews_dir

    verdict_log = (
        "Final verdict:\n```json\n{\n"
        '  "decision": "approved",\n'
        '  "summary": "lgtm",\n'
        '  "required_changes": []\n'
        "}\n```\n"
    )
    _make_dead_review_sidecar(reviews_dir, 100, verdict_log)
    _set_review_dispatched_state(app, 100, 10, "2026-07-06T12:00:00Z")

    events_path = reviews_dir / "issue-100-review.events.jsonl"
    events = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}]}},
        {
            "type": "result",
            "num_turns": 2,
            "total_cost_usd": 0.5,
            "usage": {"input_tokens": 400, "output_tokens": 100},
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)

    result = app._reap_review_verdicts(reviews_dir)

    assert result["recorded"] == [
        {"pr": 100, "issue": 10, "decision": "approved", "verdict_source": "log"}
    ]

    state = load_state(app.paths.state_file)
    record_events = [e for e in state["events"] if e["kind"] == "record_review"]
    assert record_events, "expected a record_review event"
    payload = record_events[-1]["payload"]

    assert payload["verdict_provenance"] == "fresh_llm_review"
    assert "session_metrics" in payload
    assert payload["session_metrics"]["verdict_source"] == "log"


# ---------------------------------------------------------------------------
# AC7: merge_authorize's review-decision.json patch is unaffected -- it
# touches only authorized_override, never a verdict field. Regression only:
# no verdict_provenance key may ever appear here.
# ---------------------------------------------------------------------------


def test_merge_authorize_write_never_gains_verdict_provenance_key(tmp_path: Path) -> None:
    app, paths, _ = _merge_check_app(tmp_path)

    result = app.merge_authorize(456, "CI green, stale decision overridden", by="senkichi")
    assert result.ok is True

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "authorized_override" in decision
    assert "verdict_provenance" not in decision, (
        "merge_authorize is an operator's merge authorization, not a review "
        "verdict -- it must never gain a verdict_provenance key (issue #1265 "
        "binding comment scopes it out explicitly)"
    )


# ---------------------------------------------------------------------------
# AC8: "derived_from_prose" (issue #792's findings_channel value, tagging a
# verdict's CONTENT) must never appear as a verdict_provenance enum member
# or literal anywhere in src/ -- including it would make the enum
# non-exclusive with an orthogonal, already-present field.
# ---------------------------------------------------------------------------


def test_derived_from_prose_absent_from_verdict_provenance_surface() -> None:
    assert "derived_from_prose" not in VERDICT_PROVENANCE_VALUES
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "derived_from_prose" not in text, (
            f"{path.name} contains the literal string 'derived_from_prose' -- "
            "issue #1265's binding comment excludes it from the "
            "verdict_provenance surface entirely"
        )


# ---------------------------------------------------------------------------
# AC5: bypass-enforcement AST scanner, mirroring
# tests/test_closing_reference.py's
# test_every_pr_create_call_site_routes_through_the_validator pattern: an
# rglob walk over src/*.py with no hardcoded file list, alias-aware call
# detection (any receiver, not just "self."), a "scanned" positive control
# alongside an "offenders" list that must end up empty.
#
# Covers two categories:
#   1. Every record_review() call site (any file, any receiver) must supply
#      a non-None verdict_provenance keyword.
#   2. Every self._write_json(...) call whose first argument resolves to a
#      path ending in "review-decision.json" -- derived from the filename
#      literal, not an assumed variable name, so a future bypass writer
#      spelled differently is still caught -- must write a dict containing a
#      "verdict_provenance" key. record_review's own write and
#      merge_authorize's write are excluded by function name: the former is
#      already covered by category 1 (a caller can't omit the value that
#      flows into record_review's own write) and the latter is a deliberate
#      exclusion (AC7), the same way test_closing_reference.py excludes
#      pr_create's own definition from the scan of pr_create's callers.
# ---------------------------------------------------------------------------

_PROVENANCE_WRITER_EXEMPT_FUNCTIONS = frozenset({"record_review", "merge_authorize"})


def _nodes_outside_function_definition(tree: ast.AST, function_name: str) -> list[ast.AST]:
    skip: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            skip.update(id(child) for child in ast.walk(node))
    return [n for n in ast.walk(tree) if id(n) not in skip]


def _record_review_call_sites(tree: ast.AST) -> list[ast.Call]:
    sites: list[ast.Call] = []
    for node in _nodes_outside_function_definition(tree, "record_review"):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "record_review"
        ):
            sites.append(node)
    return sites


def _call_supplies_provenance(call_node: ast.Call) -> bool:
    for kw in call_node.keywords:
        if kw.arg == "verdict_provenance":
            # An explicit literal None is a bypass in spirit -- AC2's
            # contract is "no default anywhere", and a caller passing None
            # is functionally the same evasion as omitting the keyword.
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                return False
            return True
    return False


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(id(current))
    return None


def _last_name_assignment(func_node: ast.AST, name: str) -> ast.expr | None:
    result: ast.expr | None = None
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    result = node.value
    return result


def _contains_review_decision_literal(expr: ast.expr) -> bool:
    return any(
        isinstance(n, ast.Constant) and n.value == "review-decision.json" for n in ast.walk(expr)
    )


def _resolves_to_review_decision_path(
    expr: ast.expr, func_node: ast.AST, _seen: frozenset[str] = frozenset()
) -> bool:
    if _contains_review_decision_literal(expr):
        return True
    if isinstance(expr, ast.Name) and expr.id not in _seen:
        value = _last_name_assignment(func_node, expr.id)
        if value is not None:
            return _resolves_to_review_decision_path(value, func_node, _seen | {expr.id})
    return False


def _dict_var_has_key(func_node: ast.AST, var_name: str, key: str) -> bool:
    """True if ``var_name`` is ever assigned a Dict literal containing
    ``key``, OR ever subscript-assigned that key directly (the
    read-modify-write shape ``_update_approval_head`` uses: ``updated_decision
    = dict(current_decision)`` followed by
    ``updated_decision["verdict_provenance"] = "carried_forward"``)."""
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == var_name
                and isinstance(node.value, ast.Dict)
                and any(isinstance(k, ast.Constant) and k.value == key for k in node.value.keys)
            ):
                return True
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == var_name
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key
            ):
                return True
    return False


def _write_site_supplies_provenance(call_node: ast.Call, func_node: ast.AST) -> bool:
    if len(call_node.args) < 2:
        return False
    value_arg = call_node.args[1]
    if isinstance(value_arg, ast.Dict):
        return any(
            isinstance(k, ast.Constant) and k.value == "verdict_provenance" for k in value_arg.keys
        )
    if isinstance(value_arg, ast.Name):
        return _dict_var_has_key(func_node, value_arg.id, "verdict_provenance")
    return False


def _review_decision_write_sites(
    tree: ast.AST,
) -> tuple[list[tuple[str, ast.Call, ast.AST]], list[tuple[str, ast.Call, ast.AST]]]:
    """Returns ``(checked_sites, exempted_sites)``.

    Exempted sites are *collected*, not silently skipped, because the
    exemption in ``_PROVENANCE_WRITER_EXEMPT_FUNCTIONS`` matches by function
    *name*, not identity. Matching by name alone fails open: a future
    function anywhere under ``_SRC_ROOT`` that happens to be named
    ``record_review`` or ``merge_authorize`` would silently inherit the
    exemption and could write review-decision.json with no provenance while
    this scanner stays green. Returning the exempted sites lets the caller
    assert the exemption still resolves to exactly the two known, deliberate
    exclusions instead of trusting the name match blindly.
    """
    parents = _build_parent_map(tree)
    checked: list[tuple[str, ast.Call, ast.AST]] = []
    exempted: list[tuple[str, ast.Call, ast.AST]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_write_json"
            and len(node.args) >= 1
        ):
            continue
        func_node = _enclosing_function(node, parents)
        if func_node is None:
            continue
        if not _resolves_to_review_decision_path(node.args[0], func_node):
            continue
        if func_node.name in _PROVENANCE_WRITER_EXEMPT_FUNCTIONS:
            exempted.append((func_node.name, node, func_node))
        else:
            checked.append((func_node.name, node, func_node))
    return checked, exempted


def test_every_record_review_call_site_and_review_decision_writer_supplies_provenance() -> None:
    offenders: list[str] = []
    scanned_call_sites: list[str] = []
    scanned_write_sites: list[str] = []
    exempted_write_sites: list[str] = []

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for call in _record_review_call_sites(tree):
            scanned_call_sites.append(f"{path.name}:{call.lineno}")
            if not _call_supplies_provenance(call):
                offenders.append(f"{path.name}:{call.lineno} record_review() call site")

        checked_sites, exempt_sites = _review_decision_write_sites(tree)
        for func_name, call, _func_node in checked_sites:
            scanned_write_sites.append(f"{path.name}:{call.lineno} ({func_name})")
            if not _write_site_supplies_provenance(call, _func_node):
                offenders.append(
                    f"{path.name}:{call.lineno} {func_name}() review-decision.json write"
                )
        for func_name, call, _func_node in exempt_sites:
            exempted_write_sites.append(f"{path.name}:{call.lineno} ({func_name})")

    # Positive control: the scan must actually find the known call/write
    # sites, or an empty offenders list below would be indistinguishable
    # from "the scan never matched anything". cw#1265's own recon pass
    # missed a call site (_reconcile_stranded_verdicts) by trusting a
    # remembered count instead of a fresh derivation -- this asserts the
    # scan finds real sites in both files/functions it is supposed to cover,
    # not just that it ran without error.
    assert len(scanned_call_sites) >= 9, scanned_call_sites
    assert any("cli.py" in site for site in scanned_call_sites), scanned_call_sites
    assert any("workflow.py" in site for site in scanned_call_sites), scanned_call_sites
    assert any("_update_approval_head" in site for site in scanned_write_sites), (
        scanned_write_sites
    )
    # Pin the pending-reset writer inside review() specifically -- a bare
    # substring check ("review" in site) would also match _reap_review_verdicts
    # or review_queue helpers and silently stop asserting what it claims to.
    assert any(site.endswith("(review)") for site in scanned_write_sites), scanned_write_sites

    # _PROVENANCE_WRITER_EXEMPT_FUNCTIONS matches by function *name*, which
    # fails open: any future function elsewhere in src/ named record_review
    # or merge_authorize would silently inherit the exemption and this scan
    # would stay green while it wrote review-decision.json with no
    # provenance. Asserting the exemption resolves to exactly the two known,
    # deliberate sites (both in workflow.py) turns that into a hard failure
    # instead of a silent widening.
    assert len(exempted_write_sites) == 2, exempted_write_sites
    assert all(site.startswith("workflow.py:") for site in exempted_write_sites), (
        exempted_write_sites
    )
    exempted_func_names = {site.rsplit(" (", 1)[1].rstrip(")") for site in exempted_write_sites}
    assert exempted_func_names == {"record_review", "merge_authorize"}, exempted_write_sites

    assert offenders == [], (
        "record_review call site(s) or review-decision.json writer(s) bypass "
        f"the verdict_provenance contract (issue #1265): {offenders}"
    )


# ---------------------------------------------------------------------------
# AC1: every record_review() call site supplies the EXACT literal from the
# issue #1265 mapping table -- not just that some non-None string was passed
# (AC5, above, only checks that). Grouped by (file, enclosing function)
# rather than by line number: line numbers drift with every unrelated edit
# (this lane's own recon notes document ~1000 lines of drift since the plan
# was written), while a call site's enclosing function is stable, and no
# function here has more than one provenance value except review() itself
# (two ci_gate_auto_reject sites -- the original CI-gate exit and W1's
# co-occurring-failure variant -- plus one test_adequacy_auto_reject site),
# which is why the expected table stores a Counter per function rather than
# a single string.
# ---------------------------------------------------------------------------

_EXPECTED_RECORD_REVIEW_PROVENANCE_BY_SITE: dict[tuple[str, str], Counter[str]] = {
    ("workflow.py", "review"): Counter({"ci_gate_auto_reject": 2, "test_adequacy_auto_reject": 1}),
    ("workflow.py", "_reap_review_verdicts"): Counter({"fresh_llm_review": 1}),
    ("workflow.py", "_reconcile_stranded_verdicts"): Counter({"stranded_reconciliation": 1}),
    ("workflow.py", "_record_cross_family_verdicts"): Counter({"cross_family_review": 1}),
    ("workflow.py", "_handle_malformed_cross_family_verdict"): Counter(
        {"unparseable_exhausted": 1}
    ),
    ("workflow.py", "_process_rescue_review"): Counter({"rescue_review": 1}),
    ("cli.py", "run_command"): Counter({"operator_manual": 1}),
}


def test_record_review_call_sites_map_to_expected_provenance_literal() -> None:
    actual: dict[tuple[str, str], Counter[str]] = {}

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = _build_parent_map(tree)
        for call in _record_review_call_sites(tree):
            func = _enclosing_function(call, parents)
            assert func is not None, (
                f"{path.name}:{call.lineno} record_review() call site is not "
                "inside any function -- AC1's per-function grouping assumes "
                "every call site has an enclosing def"
            )
            for kw in call.keywords:
                if kw.arg != "verdict_provenance":
                    continue
                assert isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str), (
                    f"{path.name}:{call.lineno} passes a non-literal or "
                    "non-string verdict_provenance -- AC1 requires a "
                    "resolvable literal at every call site so this table "
                    "can pin the exact value, not just that some value "
                    "was passed"
                )
                key = (path.name, func.name)
                actual.setdefault(key, Counter())[kw.value.value] += 1

    assert actual == _EXPECTED_RECORD_REVIEW_PROVENANCE_BY_SITE, (
        "record_review() call sites no longer match the issue #1265 mapping "
        "table (w8-impl-notes.md section 2's per-site mapping, corrected "
        "for the stranded_reconciliation site this lane's recon found).\n"
        f"expected: {dict(_EXPECTED_RECORD_REVIEW_PROVENANCE_BY_SITE)}\n"
        f"actual:   {dict(actual)}"
    )

    # Positive control: the table itself must cover all 9 known call sites --
    # an accidentally-empty expected table would make the equality assertion
    # above vacuously true if `actual` were also empty (e.g. a scanner
    # regression that stopped matching anything).
    total_sites = sum(
        sum(counter.values()) for counter in _EXPECTED_RECORD_REVIEW_PROVENANCE_BY_SITE.values()
    )
    assert total_sites == 9, _EXPECTED_RECORD_REVIEW_PROVENANCE_BY_SITE


# ---------------------------------------------------------------------------
# AC2: record_review() has NO default value for verdict_provenance anywhere
# in its signature. Two halves: a static AST check on the def (no default
# expression exists on the parameter, in any of the three places Python
# allows one), and a runtime check that omitting the argument fails at the
# call/type layer -- a TypeError from Python's own argument binding -- before
# any body code (the decision-value check, the verdict_provenance membership
# check, any state I/O) runs.
# ---------------------------------------------------------------------------


def _find_function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level or nested def named {name!r} found")


def test_record_review_verdict_provenance_has_no_default_in_the_ast() -> None:
    tree = ast.parse(
        (_SRC_ROOT / "workflow.py").read_text(encoding="utf-8"), filename="workflow.py"
    )
    func = _find_function_def(tree, "record_review")

    positional_names = [a.arg for a in func.args.posonlyargs + func.args.args]
    assert "verdict_provenance" not in positional_names, (
        "verdict_provenance must not be a positional(-or-keyword) parameter "
        "-- a positional parameter with a default in func.args.defaults "
        "would satisfy AC2's letter while defeating its intent (a caller "
        "could still omit it if every parameter after it also has a "
        "default)"
    )

    kwonly_names = [a.arg for a in func.args.kwonlyargs]
    assert "verdict_provenance" in kwonly_names, (
        "verdict_provenance must be a required keyword-only parameter"
    )
    index = kwonly_names.index("verdict_provenance")
    assert func.args.kw_defaults[index] is None, (
        "verdict_provenance has a default expression in record_review()'s "
        "keyword-only signature -- AC2 requires no default anywhere"
    )


def test_record_review_missing_verdict_provenance_raises_typeerror_at_call_boundary(
    tmp_path: Path,
) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    with pytest.raises(TypeError, match="verdict_provenance"):
        app.record_review(456, "approved", summary="lgtm")  # type: ignore[call-arg]

    # Proves the TypeError fired at the call boundary, before any body code
    # ran: no pr_dir/review-decision.json was ever created.
    assert not (paths.prs / "pr-456" / "review-decision.json").exists()


# ---------------------------------------------------------------------------
# AC9: a fixture events.db / review-decision.json set drives every one of
# the 9 enum values plus the pending-reset None sentinel through the real
# writers, then queries events.db (query_events -- the same indexed-column
# API the real dashboards use) and re-reads review-decision.json to confirm
# every row/write carries the field as designed. This is a fixture, not the
# live database, per the task's own framing (a fresh clone's events.db has
# no populated rows and is off-limits regardless).
# ---------------------------------------------------------------------------


def test_verdict_provenance_fixture_drives_every_enum_value_and_null_sentinel(
    tmp_path: Path,
) -> None:
    # 8 of the 9 enum values are ever passed to record_review() as a fresh
    # decision; "carried_forward" is stamped only by _update_approval_head,
    # which never calls record_review (see the AC3 tests above) -- so it is
    # exercised separately below, not through this loop.
    record_review_values = sorted(VERDICT_PROVENANCE_VALUES - {"carried_forward"})
    assert record_review_values == [
        "ci_gate_auto_reject",
        "cross_family_review",
        "fresh_llm_review",
        "operator_manual",
        "rescue_review",
        "stranded_reconciliation",
        "test_adequacy_auto_reject",
        "unparseable_exhausted",
    ], record_review_values

    for index, value in enumerate(record_review_values):
        case_root = tmp_path / f"case-{index}"
        config = OrchestratorConfig()
        paths = runtime_paths(case_root, config.runtime.state_dir)
        app = OrchestratorApp(case_root, paths, config, FakeGitHub())

        result = app.record_review(456, "approved", summary="lgtm", verdict_provenance=value)
        assert result.ok is True, (value, result.message)

        decision_path = paths.prs / "pr-456" / "review-decision.json"
        on_disk = json.loads(decision_path.read_text(encoding="utf-8"))
        assert on_disk["verdict_provenance"] == value

        rows = query_events(paths.state_file, kind="record_review")
        assert len(rows) == 1, (value, rows)
        assert rows[0]["payload"]["verdict_provenance"] == value

    # "carried_forward": _update_approval_head bypasses record_review
    # entirely (issue #638's carry-forward mechanism).
    app, paths = _carry_forward_app(tmp_path / "case-carried-forward")
    decision_dir = paths.prs / "pr-456"
    decision_dir.mkdir(parents=True)
    decision = {
        "decision": "approved",
        "reviewed_head_sha": "old-sha",
        "verdict_provenance": "fresh_llm_review",
    }
    (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")

    applied = app._update_approval_head(
        456,
        decision,
        "new-sha",
        old_head="old-sha",
        issue_number=123,
        tier="verified-sync",
    )
    assert applied is True

    on_disk = json.loads((decision_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert on_disk["verdict_provenance"] == "carried_forward"
    rows = query_events(paths.state_file, kind="verdict_carried_forward_verified_sync")
    assert len(rows) == 1
    assert rows[0]["payload"]["verdict_provenance"] == "carried_forward"

    # The pending-reset None sentinel, driven through review()'s real
    # fresh-packet path (same mechanism as the AC4 tests above).
    none_app = _cross_family_app(tmp_path / "case-none-sentinel", enabled=False)
    none_decision_path = none_app.paths.prs / "pr-456" / "review-decision.json"
    none_app.review(456)
    none_on_disk = json.loads(none_decision_path.read_text(encoding="utf-8"))
    assert none_on_disk["decision"] == "pending"
    assert "verdict_provenance" in none_on_disk
    assert none_on_disk["verdict_provenance"] is None

    # Positive control: every enum member (8 driven through record_review +
    # "carried_forward" driven through _update_approval_head) plus the None
    # sentinel were actually exercised above -- 9 + 1 = 10 distinct writes.
    assert len(record_review_values) + 1 == len(VERDICT_PROVENANCE_VALUES)
