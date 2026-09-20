"""Review-packet rendering: command templates, test-adequacy / static-probe sections, and the required-changes section's vacuous / crash-signature tiers.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY, render_command
from charlie_work.verdict_parsing import REVIEW_SESSION_SUMMARY_HEADING
from charlie_work.workflow import _render_required_changes_section
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


# --- Cross-family adversarial review ------------------------------------------


def test_render_command_templates_list_and_string() -> None:
    values = {"model": "codex", "prompt_path": "/tmp/p.md"}
    assert render_command(
        ("devin", "--model", "{model}", "-p", "--prompt-file", "{prompt_path}"), values
    ) == ["devin", "--model", "codex", "-p", "--prompt-file", "/tmp/p.md"]
    assert render_command("devin --model {model}", values) == "devin --model codex"


def test_render_test_adequacy_section_unit() -> None:
    """Unit test for render_test_adequacy_section (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Test with None (gate disabled)
    assert render_test_adequacy_section(None, ()) == ""

    # Test with populated facts
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=("src/foo.py", "src/bar.py"),
        exempt=False,
        exempt_reason="",
    )
    warnings = ("Zero recognized assertions in added test lines",)

    section = render_test_adequacy_section(facts, warnings)
    assert "## Test-adequacy facts (Tier 1, deterministic)" in section
    assert "Added product LOC: 100" in section
    assert "Added test LOC: 50" in section
    assert "Assertion-bearing added test lines: 10" in section
    assert "Test files changed: 2" in section
    assert "Untested product files: src/foo.py, src/bar.py" in section
    assert "Zero recognized assertions in added test lines" in section

    # Test with empty warnings
    section_no_warnings = render_test_adequacy_section(facts, ())
    assert "Zero recognized assertions" not in section_no_warnings

    # Test with exempt claim
    facts_exempt = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=0,
        assertion_count=0,
        test_files_changed=0,
        untested_product_files=(),
        exempt=True,
        exempt_reason="n/a - pure refactoring",
    )
    section_exempt = render_test_adequacy_section(facts_exempt, ())
    assert (
        'Test-exempt claim: "n/a - pure refactoring" (verify against the diff)' in section_exempt
    )


def test_render_static_probe_section_unit() -> None:
    """Unit test for render_static_probe_section (issues #1260/#1261)."""
    from charlie_work.diff_coverage_probe import (
        BranchCoverageFinding,
        StaticProbeVerdict,
        UnwiredSymbolFinding,
    )
    from charlie_work.workflow import render_static_probe_section

    # Disabled (probe never ran) -> "".
    assert render_static_probe_section(None) == ""

    # Enabled, zero findings, zero warnings -> explicit visible "no findings"
    # line, never a bare "" -- an advisory probe that goes silent on a clean
    # run must not read as "never ran".
    clean_section = render_static_probe_section(StaticProbeVerdict())
    assert clean_section == "Static probe: no findings.\n"

    # Enabled, internal error -> visible degradation warning, not silent-empty.
    degraded = StaticProbeVerdict(
        warnings=("static probe degraded: branch-coverage heuristic failed: boom",)
    )
    degraded_section = render_static_probe_section(degraded)
    assert "static probe degraded" in degraded_section

    # Enabled, findings present -> both W3 and W20 findings concatenated
    # into the one section.
    verdict = StaticProbeVerdict(
        branch_findings=(BranchCoverageFinding("src/foo.py", 3, 0, "no_test_adds"),),
        unwired_findings=(UnwiredSymbolFinding("helper", "src/bar.py", "function"),),
    )
    section = render_static_probe_section(verdict)
    assert "Branch-coverage heuristic (W3)" in section
    assert "src/foo.py" in section
    assert "Unwired-symbol probe (W20)" in section
    assert "helper" in section
    assert "src/bar.py" in section


def test_render_required_changes_section_populated_list_unaffected_by_fallback() -> None:
    """Non-regression: when required_changes IS populated, the enumerated
    list renders exactly as before even though a summary is also present --
    the summary-fallback and escape-hatch tiers must never fire when there is
    real structured content to show."""
    decision = {
        "decision": "request_changes",
        "summary": "This summary must not appear in the rendered section.",
        "required_changes": ["fix the off-by-one", "add a regression test"],
    }

    section = _render_required_changes_section(decision)

    assert "## Required changes" in section
    assert "- fix the off-by-one" in section
    assert "- add a regression test" in section
    # The fallback/escape-hatch framing must not leak in alongside the list.
    assert "did not record a structured findings list" not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section
    assert "This summary must not appear" not in section


def test_render_required_changes_section_both_empty_renders_escape_hatch() -> None:
    """Hard requirement (F1): when both required_changes and summary are
    empty on a request_changes verdict, the section must never look like
    "nothing to change" -- it must loudly say the findings are unavailable
    and point the worker at the PR's review comments on GitHub instead of
    silently rendering an empty section."""
    decision = {"decision": "request_changes", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert section.strip() != ""
    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert "is NOT a signal that there is nothing to change" in section
    assert "GitHub" in section


def test_render_required_changes_section_blocked_both_empty_renders_escape_hatch() -> None:
    """Defensive extension of the same hard requirement to `blocked`: the
    decision-agnostic janitor-gate rework routes (merge-conflict / no-op
    repair) can carry forward whatever verdict was last on disk, including a
    `blocked` one. Suppressing the enumerated list / summary fallback for
    `blocked` (see the omits-for-approved-style suppression covered above)
    is deliberate, but suppressing it AND leaving the worker with zero signal
    that something was withheld is not -- the both-empty escape hatch still
    fires."""
    decision = {"decision": "blocked", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section


# --------------------------------------------------------------------------
# Issue #792: verdicts recorded by the current record_review carry an
# explicit findings_channel marker ("vacuous" or "derived"). These tests
# cover the renderer's handling of that marker directly, independent of the
# shape-based (required_changes/summary) tiers above, which exist only to
# infer the same distinction for pre-#792 records with no marker at all.
# --------------------------------------------------------------------------


def test_render_required_changes_section_vacuous_marker_renders_escape_hatch() -> None:
    """A findings_channel="vacuous" verdict always renders tier 3, even
    though `summary` is technically non-blank (it may carry the historical
    placeholder) -- rendering it as real content (tier 2) would silently
    present content-free text as the reviewer's actual findings."""
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
        "findings_channel": "vacuous",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert LEGACY_VACUOUS_SUMMARY not in section


def test_render_required_changes_section_vacuous_marker_fires_for_blocked_too() -> None:
    """The vacuous marker's escape hatch is decision-agnostic -- it fires for
    `blocked` exactly as it does for `request_changes`, unlike the "derived"
    marker below which is request_changes-specific."""
    decision = {
        "decision": "blocked",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
        "findings_channel": "vacuous",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section


def test_render_required_changes_section_derived_marker_renders_summary_verbatim() -> None:
    """A findings_channel="derived" request_changes verdict renders the
    tier-2-shaped verbatim-summary section even though required_changes is
    now populated (record_review copied summary into it) -- it must not
    fall into tier 1's bullet-list rendering, which would wrap an entire
    multi-sentence summary as a single bullet."""
    prose = "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    decision = {
        "decision": "request_changes",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert prose in section
    assert f"- {prose}" not in section
    assert "did not record a structured findings list" in section


def test_render_required_changes_section_derived_marker_suppressed_for_blocked() -> None:
    """Unlike "vacuous", the "derived" marker's special-cased rendering only
    applies to request_changes (see the docstring's blocked-suppression
    rule) -- for `blocked` it falls through to the pre-existing shape-based
    tiers, where a populated required_changes on a blocked verdict is
    suppressed by design (blocked's "what must change before approval"
    framing doesn't fit blocked's routes)."""
    prose = "Security review flagged an unauthenticated endpoint."
    decision = {
        "decision": "blocked",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert section == ""


def test_render_required_changes_section_defangs_live_keyword_in_list_tier() -> None:
    # Issue #781 AC4: reviewer prose in the required_changes list tier must
    # not carry a live closing keyword into the rendered brief -- a worker
    # reads this brief and writes its own PR body from it, a boundary
    # linked_issue_number's hijack-safety check never sees.
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["this does not fix #649, dig deeper"],
    }

    section = _render_required_changes_section(decision)

    assert "does not fix issue 649" in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live keyword survived"


def test_render_required_changes_section_defangs_live_keyword_in_summary_tier() -> None:
    # Same guarantee for the summary-fallback tier (tier 2).
    from charlie_work.issue_linking import _CLOSING_KEYWORD_REF

    decision = {
        "decision": "request_changes",
        "summary": "BLOCKER - does not fix #649. Still broken.",
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert "does not fix issue 649" in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live keyword survived"


# --------------------------------------------------------------------------
# Issue #1269 (W12): the render-side crash-signature guard is the single
# enforcement point that reaches records already persisted before the
# collector-side fix in _collect_external_findings shipped -- these tests
# cover both the new-shape (external_findings field) and old-shape
# (findings_channel == "external", merged into required_changes) paths.
# --------------------------------------------------------------------------


def test_render_required_changes_section_strips_crash_signature_from_external_findings() -> None:
    """New shape: a crash-comment body sitting in `external_findings` from
    before the collector-side fix is stripped at render time; a genuine
    external finding alongside it survives untouched."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "Reviewer summary text.",
        "required_changes": ["fix the off-by-one"],
        "external_findings": ["A human found a real bug in the retry loop.", crash_body],
    }

    section = _render_required_changes_section(decision)

    assert "A human found a real bug in the retry loop." in section
    assert crash_body not in section
    assert "did not produce a structured verdict" not in section
    assert "## Findings posted on the PR itself" in section, "new_shape must still be True"


def test_render_required_changes_section_all_crash_external_findings_falls_back_to_pointer() -> (
    None
):
    """New shape, entirely crash noise: external_findings ends up empty
    after filtering, so `new_shape` becomes False and this falls to the
    ordinary pointer-style ending (_finish_required_changes_section) rather
    than rendering an external-findings section with nothing real in it."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["fix the off-by-one"],
        "external_findings": [crash_body],
    }

    section = _render_required_changes_section(decision)

    assert "fix the off-by-one" in section
    assert crash_body not in section
    assert "## Findings posted on the PR itself" not in section
    assert "none of which reach this brief" in section, (
        "the ordinary external-findings pointer must reappear once external_findings "
        "filters down to empty"
    )


def test_render_required_changes_section_old_shape_strips_crash_signature_from_changes() -> None:
    """Old shape (findings_channel == "external"): a crash comment merged
    into required_changes before the collector-side fix is stripped; the
    reviewer's own genuine item survives (defense-in-depth for a reopened
    old-shape PR, per issue #1269 open question 3)."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": ["fix the off-by-one", crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "- fix the off-by-one" in section
    assert crash_body not in section


def test_render_required_changes_section_old_shape_vacuous_all_crash_changes_renders_tier3() -> (
    None
):
    """Old shape, all-crash: when the crash filter empties `changes`
    entirely AND the leftover summary is the vacuous placeholder
    (record_review's vacuous-replace path -- the only way this exact
    combination is reachable), the section must render tier 3, NOT fall
    through to tier 2 and present the content-free placeholder as if it
    were real findings. This is the failure mode the "vacuous" marker
    branch already guards against, reached here through the old-shape
    "external" path instead of a fresh "vacuous" marker."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert LEGACY_VACUOUS_SUMMARY not in section
    assert crash_body not in section
    assert "did not record a structured findings list" not in section, (
        "must not fall through to the tier-2 verbatim-summary rendering"
    )


def test_render_required_changes_section_old_shape_blank_summary_all_crash_changes_renders_tier3() -> (
    None
):
    """Same guard, other half of the "vacuous" OR-condition: a blank
    summary (not the LEGACY_VACUOUS_SUMMARY placeholder) alongside an
    all-crash `changes` list also renders tier 3, not tier 2's empty-prose
    rendering."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert crash_body not in section


def test_render_required_changes_section_old_shape_all_crash_changes_preserves_genuine_summary() -> (
    None
):
    """Guard against over-aggression: when the crash filter empties
    `changes` entirely but the summary is genuine, non-vacuous reviewer
    prose (the pre-#999 "reviewer chose prose over an itemized list"
    population -- see `_is_carry_forward_eligible`'s docstring), tier 2
    must still fire and render that real summary. The vacuous-neutralization
    guard must only fire on the two known placeholder shapes (blank or
    LEGACY_VACUOUS_SUMMARY), never on genuine content."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    genuine_summary = (
        "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    )
    decision = {
        "decision": "request_changes",
        "summary": genuine_summary,
        "required_changes": [crash_body],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert genuine_summary in section
    assert crash_body not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "a genuine, non-vacuous summary must not be discarded down to tier 3"
    )


def test_render_required_changes_section_vacuous_guard_does_not_drop_populated_external_findings() -> (
    None
):
    """Hardening (review of 63ce581): the vacuous-neutralization guard's
    `and not external_findings` clause must not let a co-persisted, populated
    `external_findings` field get silently discarded.

    No current writer produces this exact shape -- `findings_channel ==
    "external"` old-shape records and populated `external_findings` are
    disjoint in every writer today -- but the guard's condition must not
    silently assume that forever. Before the `and not external_findings`
    clause, an all-crash old-shape `changes` plus a vacuous summary would
    neutralize summary_text to "" regardless of `external_findings`, landing
    in the tier-3 "both empty" branch below -- which returns immediately
    without ever consulting `external_findings` at all, dropping genuine
    findings on the floor. With the clause, a populated `external_findings`
    keeps the guard from firing, so the section falls through to the
    tier-2 summary-fallback branch instead, which -- because `new_shape` is
    True -- still calls `_render_external_findings_section` and renders
    every genuine finding."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    genuine_external_finding = "the retry wrapper swallows the exception type"
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [crash_body],
        "findings_channel": "external",
        "external_findings": [genuine_external_finding],
    }

    section = _render_required_changes_section(decision)

    assert genuine_external_finding in section, (
        "populated external_findings must still be rendered, not dropped by "
        "the vacuous-neutralization guard"
    )
    assert crash_body not in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "must not fall through to tier 3 and discard the genuine external findings"
    )


# --------------------------------------------------------------------------
# Issue #1310: the tier-2 (summary-verbatim) render path -- both the
# marker-less `summary_text` fallback and the `"derived"` marker branch --
# must crash-filter `summary` exactly as W12 (#1269) filtered
# `external_findings`/`required_changes`. A crash-signature body or the
# LEGACY_VACUOUS_SUMMARY placeholder arriving as a verdict `summary` with
# an empty findings list must degrade to tier 3, not render verbatim.
# Production-unreachable today (crash bodies are posted as PR comments,
# never written into a verdict `summary`), but the suppression contract is
# "crash noise cannot reach prompt content through any render path."
# --------------------------------------------------------------------------


def test_render_required_changes_section_tier2_crash_summary_degrades_to_tier3() -> None:
    """Marker-less tier-2 path: a request_changes round whose `summary` is a
    crash body and whose findings list is empty must NOT render the crash
    text verbatim -- it degrades to tier 3 (the "findings unavailable"
    escape hatch), exactly as the other tiers do."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": crash_body,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert crash_body not in section, (
        "tier-2 verbatim emit must not render a crash-signature summary"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the summary is a crash body"
    )
    assert "did not record a structured findings list" not in section, (
        "must not fall through to the tier-2 verbatim-summary rendering"
    )


def test_render_required_changes_section_tier2_vacuous_summary_degrades_to_tier3() -> None:
    """Marker-less tier-2 path: a request_changes round whose `summary` is
    the LEGACY_VACUOUS_SUMMARY placeholder and whose findings list is empty
    must degrade to tier 3, not present the content-free placeholder as if
    it were real findings."""
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert LEGACY_VACUOUS_SUMMARY not in section, (
        "tier-2 verbatim emit must not render the vacuous placeholder"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the summary is the vacuous placeholder"
    )
    assert "did not record a structured findings list" not in section


def test_render_required_changes_section_tier2_derived_crash_summary_degrades() -> None:
    """`"derived"` marker branch: a request_changes round stamped
    `findings_channel == "derived"` whose `summary` is a crash body and
    whose findings list is empty must degrade to tier 3, not render the
    crash text verbatim. Without the `and summary_text` guard on the
    derived branch, the crash guard would neutralize `summary_text` to ""
    but the branch would still fire and render an empty verbatim summary."""
    crash_body = f"{REVIEW_SESSION_SUMMARY_HEADING}\n\nNo verdict was produced."
    decision = {
        "decision": "request_changes",
        "summary": crash_body,
        "required_changes": [],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert crash_body not in section, (
        "derived tier-2 verbatim emit must not render a crash-signature summary"
    )
    assert "REVIEWER FINDINGS UNAVAILABLE" in section, (
        "must degrade to tier 3 when the derived summary is a crash body"
    )
    assert "did not record a structured findings list" not in section


def test_render_required_changes_section_tier2_genuine_summary_still_renders() -> None:
    """Guard against over-aggression: a genuine, non-crash, non-vacuous
    summary with an empty findings list must still render verbatim at
    tier 2. The crash/vacuous guard must only fire on crash-signature
    bodies or the LEGACY_VACUOUS_SUMMARY placeholder, never on real
    reviewer prose."""
    genuine_summary = (
        "The retry wrapper swallows the exception type; callers cannot distinguish causes."
    )
    decision = {
        "decision": "request_changes",
        "summary": genuine_summary,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert genuine_summary in section, "a genuine summary must still render verbatim at tier 2"
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section, (
        "a genuine summary must not be discarded down to tier 3"
    )
