"""Issue #950: the external-findings pointer must reach every substantive
tier of ``_render_required_changes_section``, not just the two degraded
tiers that already had it.

Before this fix, the instruction to go read the PR's own review comments
existed only in the ``findings_channel == "vacuous"`` tier and the
both-empty (no marker) escape-hatch tier -- both of which fire only when the
orchestrator's own internal findings came back empty. The three tiers that
render substantive content (the structured ``required_changes`` list, the
``findings_channel == "derived"`` summary, and the shape-based
summary-fallback for pre-#792 records) carried no such pointer, so a human's
or peer agent's PR-comment findings were silently discarded exactly when the
internal review was good (PR #948).

These tests exercise ``_render_required_changes_section`` directly as a pure
function of a ``decision`` dict -- no ``OrchestratorApp``/state needed.
"""

from __future__ import annotations

from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY
from charlie_work.github import _CLOSING_KEYWORD_REF
from charlie_work.workflow import _render_required_changes_section

# Distinctive substring unique to the new pointer's section header -- does
# not overlap with either degraded tier's own "go check GitHub" prose, so
# its presence/absence is an unambiguous signal that _finish_required_changes
# _section did (or did not) run.
_POINTER_HEADER = "Also required: findings posted on the PR itself"
# A second, prose-level distinctive phrase from inside the pointer body.
_POINTER_PHRASE = "Read the PR's review comments and review threads on GitHub before you start"
# Issue #999: the external-findings section's own heading (distinct from the
# pointer's header so the two cannot be confused).
_EXTERNAL_SECTION_HEADER = "## Findings posted on the PR itself"


# --------------------------------------------------------------------------
# The three substantive tiers now carry the pointer, additively.
# --------------------------------------------------------------------------


def test_external_findings_tier_renders_list_and_pointer() -> None:
    """Issue #950: when findings are folded in from the PR itself,
    ``findings_channel == "external"`` renders the itemized tier with an
    external-aware intro and the pointer still appended."""
    decision = {
        "decision": "request_changes",
        "summary": "The retry wrapper swallows the exception type.",
        "required_changes": [
            "add a regression test",
            "fix the null check in parse()",
        ],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    assert "## Required changes" in section
    assert "verified findings posted on the PR itself" in section
    assert "- add a regression test" in section
    assert "- fix the null check in parse()" in section
    assert _POINTER_HEADER in section
    assert section.count(_POINTER_HEADER) == 1
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section


def test_list_tier_contains_pointer_and_keeps_all_items() -> None:
    """Tier 1 (enumerated ``required_changes``): the pointer is appended and
    every structured item still renders -- additive, not a replacement."""
    decision = {
        "decision": "request_changes",
        "summary": "",
        "required_changes": [
            "fix the off-by-one",
            "add a regression test",
            "update the docstring (Fixes #649)",
        ],
    }

    section = _render_required_changes_section(decision)

    assert "## Required changes" in section
    assert "- fix the off-by-one" in section
    assert "- add a regression test" in section
    # Reviewer prose in the list tier is still defanged even with the
    # pointer appended after it.
    assert "update the docstring (Fixes issue 649)" in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live closing keyword survived"
    # The pointer is present exactly once.
    assert section.count(_POINTER_HEADER) == 1
    assert _POINTER_PHRASE in section
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section


def test_derived_marker_tier_contains_pointer_and_keeps_summary() -> None:
    """The ``findings_channel == "derived"`` tier (tier-2-shaped, request_changes
    only) gets the pointer, and the summary prose it renders verbatim
    survives -- defanged -- alongside it."""
    prose = "The retry wrapper swallows the exception type (Fixes #649)."
    decision = {
        "decision": "request_changes",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
    }

    section = _render_required_changes_section(decision)

    assert "did not record a structured findings list" in section
    assert "The retry wrapper swallows the exception type (Fixes issue 649)." in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live closing keyword survived"
    assert section.count(_POINTER_HEADER) == 1
    assert _POINTER_PHRASE in section


def test_summary_fallback_tier_contains_pointer_and_keeps_summary() -> None:
    """Shape-based tier 2 (no ``findings_channel`` marker, ``required_changes``
    empty, ``summary`` populated -- the pre-#792 record shape) gets the
    pointer too, with the summary still present and defanged."""
    prose = "BLOCKER - does not fix #649. The underlying conflict is still unresolved."
    decision = {
        "decision": "request_changes",
        "summary": prose,
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert "did not record a structured findings list" in section
    assert "does not fix issue 649" in section
    assert "does not fix #649" not in section
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live closing keyword survived"
    assert section.count(_POINTER_HEADER) == 1
    assert _POINTER_PHRASE in section


# --------------------------------------------------------------------------
# The two degraded tiers keep rendering their own "unavailable" text and do
# NOT pick up a (duplicated) pointer -- they already tell the worker to go
# check GitHub via their own inline prose.
# --------------------------------------------------------------------------


def test_vacuous_marker_tier_has_no_pointer_duplication() -> None:
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": [],
        "findings_channel": "vacuous",
    }

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert LEGACY_VACUOUS_SUMMARY not in section
    assert _POINTER_HEADER not in section
    assert section.count(_POINTER_HEADER) == 0


def test_both_empty_escape_hatch_tier_has_no_pointer_duplication() -> None:
    decision = {"decision": "request_changes", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert "is NOT a signal that there is nothing to change" in section
    assert _POINTER_HEADER not in section
    assert section.count(_POINTER_HEADER) == 0


def test_blocked_both_empty_escape_hatch_tier_has_no_pointer_duplication() -> None:
    """Same degraded escape hatch, decision-agnostic for `blocked` -- also no
    pointer duplication."""
    decision = {"decision": "blocked", "summary": "", "required_changes": []}

    section = _render_required_changes_section(decision)

    assert "REVIEWER FINDINGS UNAVAILABLE" in section
    assert _POINTER_HEADER not in section


# --------------------------------------------------------------------------
# Paths that must still render the empty string are unchanged by this fix.
# --------------------------------------------------------------------------


def test_no_decision_returns_empty_string() -> None:
    assert _render_required_changes_section(None) == ""


def test_non_dict_decision_returns_empty_string() -> None:
    assert _render_required_changes_section("not a dict") == ""  # type: ignore[arg-type]
    assert _render_required_changes_section([1, 2, 3]) == ""  # type: ignore[arg-type]


def test_approved_verdict_returns_empty_string() -> None:
    decision = {
        "decision": "approved",
        "summary": "Looks good.",
        "required_changes": [],
    }

    assert _render_required_changes_section(decision) == ""


def test_blocked_with_both_changes_and_summary_returns_empty_string() -> None:
    """`blocked` with real findings content (not both empty) is suppressed
    by design -- the "what must change before approval" framing does not fit
    the decision-agnostic janitor-gate routes that carry a `blocked` verdict
    forward. Untouched by this fix: no pointer, no content, empty string."""
    decision = {
        "decision": "blocked",
        "summary": "merge conflict, do not re-litigate",
        "required_changes": ["some leftover finding"],
    }

    section = _render_required_changes_section(decision)

    assert section == ""


def test_blocked_with_summary_only_returns_empty_string() -> None:
    """Same suppression when only one of the two (summary, no structured
    list) is populated -- still not "both empty", so the escape hatch does
    not fire either; the section is fully suppressed."""
    decision = {
        "decision": "blocked",
        "summary": "merge conflict, do not re-litigate",
        "required_changes": [],
    }

    section = _render_required_changes_section(decision)

    assert section == ""


# --------------------------------------------------------------------------
# Issue #999: external findings render as their own section, separate from
# the reviewer's required_changes. findings_channel keeps describing ONLY
# the reviewer's list, so "derived" is never overwritten. Old-shape records
# (no external_findings field) render exactly as before this fix.
# --------------------------------------------------------------------------


def test_derived_with_external_findings_renders_summary_verbatim_and_external_section() -> None:
    """The core #999 acceptance criterion: a verdict that is BOTH ``derived``
    AND carries external findings renders the derived summary verbatim (not
    as a single bullet) under its own heading AND the external items as
    bullets under theirs.

    Before #999, ``record_review`` overwrote ``findings_channel`` to
    ``"external"`` and merged the external items into ``required_changes``,
    so the renderer took the itemized path and the multi-paragraph derived
    summary was emitted as one ``- {...}`` bullet -- destroying the only
    representation of what the reviewer wanted changed.
    """
    prose = (
        "The retry wrapper swallows the exception type. Callers cannot "
        "distinguish a transient failure from a permanent one, so every "
        "retry loop masks real bugs. This affects the parse() and fetch() "
        "paths alike.\n\nA targeted regression test should cover both."
    )
    decision = {
        "decision": "request_changes",
        "summary": prose,
        "required_changes": [prose],
        "findings_channel": "derived",
        "external_findings": [
            "The migration needs a rollback path before this can land.",
            "Add a changelog entry (Fixes #649).",
        ],
    }

    section = _render_required_changes_section(decision)

    # The derived summary renders verbatim, NOT as a bullet.
    assert prose in section
    assert f"- {prose}" not in section
    assert "did not record a structured findings list" in section
    # The external findings render under their own heading as bullets.
    assert _EXTERNAL_SECTION_HEADER in section
    assert "- The migration needs a rollback path before this can land." in section
    assert "- Add a changelog entry (Fixes issue 649)." in section
    # The external closing keyword is defanged.
    assert _CLOSING_KEYWORD_REF.search(section) is None, "live closing keyword survived"
    # The pointer is NOT appended -- the external section replaces it.
    assert _POINTER_HEADER not in section
    assert _POINTER_PHRASE not in section
    assert section.count(_EXTERNAL_SECTION_HEADER) == 1


def test_itemized_with_external_findings_renders_both_sections() -> None:
    """A reviewer-provided itemized list plus external findings: the
    reviewer's list renders as the itemized tier (normal intro, not the
    external-aware one) and the external items render under their own
    heading. ``findings_channel`` is unset (the reviewer provided a list,
    so the derivation block never ran)."""
    decision = {
        "decision": "request_changes",
        "summary": "fix A",
        "required_changes": ["fix the off-by-one", "add a regression test"],
        "external_findings": ["The migration needs a rollback path."],
    }

    section = _render_required_changes_section(decision)

    # Reviewer's itemized list with the normal (non-external) intro.
    assert "## Required changes" in section
    assert "authoritative list of what must change" in section
    assert "- fix the off-by-one" in section
    assert "- add a regression test" in section
    # External findings under their own heading.
    assert _EXTERNAL_SECTION_HEADER in section
    assert "- The migration needs a rollback path." in section
    # No pointer (the external section replaces it).
    assert _POINTER_HEADER not in section


def test_old_shape_external_channel_renders_unchanged_with_pointer() -> None:
    """Migration guarantee (issue #999): a verdict written before this fix
    has external findings already merged into ``required_changes`` with
    ``findings_channel == "external"`` and NO ``external_findings`` field.
    It must render exactly as before -- the itemized tier with the
    external-aware intro and the pointer -- so no content is lost from any
    verdict already on disk."""
    decision = {
        "decision": "request_changes",
        "summary": "The retry wrapper swallows the exception type.",
        "required_changes": [
            "add a regression test",
            "The migration needs a rollback path.",
        ],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    # Old-shape itemized tier with the external-aware intro.
    assert "## Required changes" in section
    assert "verified findings posted on the PR itself" in section
    assert "- add a regression test" in section
    assert "- The migration needs a rollback path." in section
    # The pointer is still appended (old shape).
    assert _POINTER_HEADER in section
    assert section.count(_POINTER_HEADER) == 1
    # No new-shape external section (no external_findings field).
    assert _EXTERNAL_SECTION_HEADER not in section


def test_vacuous_with_external_findings_still_replaces_not_separate_section() -> None:
    """The ``vacuous`` case is the one that must NOT become a separate
    section: a content-free reviewer summary has nothing worth rendering
    above the external items, so ``record_review`` still *replaces*
    ``required_changes`` with the external findings (channel ``"external"``,
    no ``external_findings`` field). The renderer's old-shape ``"external"``
    path handles it -- the vacuous tier-3 warning does NOT fire and the
    external items render as the itemized list."""
    decision = {
        "decision": "request_changes",
        "summary": LEGACY_VACUOUS_SUMMARY,
        "required_changes": ["The migration needs a rollback path."],
        "findings_channel": "external",
    }

    section = _render_required_changes_section(decision)

    # The vacuous tier-3 warning does NOT fire (the external items replaced
    # the vacuous content).
    assert "REVIEWER FINDINGS UNAVAILABLE" not in section
    assert LEGACY_VACUOUS_SUMMARY not in section
    # The external items render as the itemized list (old-shape external).
    assert "- The migration needs a rollback path." in section
    assert "verified findings posted on the PR itself" in section
    # No new-shape external section.
    assert _EXTERNAL_SECTION_HEADER not in section
