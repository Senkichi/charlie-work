"""Issue #1642: a ``request_changes`` verdict whose findings ask for a
human/operator decision must be reclassified to ``blocked`` and routed to
the operator queue -- never to automated rework.

Regression tests for the incident on PR #1641: the fleet reviewer returned
``decision: request_changes`` with a ``required_changes`` item that asked
for an explicit human/operator call. ``record_review`` recorded it as an
ordinary ``request_changes``, ``dispatch_rework`` launched a rework worker,
and the worker asserted an operator decision that never occurred ("The
operator declined to grant that sign-off").

Covers four layers:

1. **Helper unit tests** -- ``human_decision_marker_match`` classifies
   required-changes items against the configured markers, and
   ``reclassify_human_call_verdict`` gates that match on decision +
   provenance.
2. **``record_review`` integration tests** -- the incident fixture verdict
   routes to ``blocked`` (issue escalated ``review_blocked``/judgment,
   ``agent:human-needed``, no rework brief, no rework budget consumed), the
   ``review_decision_reclassified_blocked`` event carries the matched item,
   and the positive control (an ordinary ``request_changes``) still routes
   to rework. Mechanically generated verdicts (``*_auto_reject`` gates) are
   exempt.
3. **Rework-prompt test** -- the rendered brief carries the "No operator
   decisions in this brief" section and the ``.worker-outcome.json``
   blocked channel.
4. **Config tests** -- ``review.human_decision_markers`` parses from YAML
   and both shipped example configs carry the default set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
    ReviewConfig,
    load_config,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.review_decision import (
    human_decision_marker_match,
    reclassify_human_call_verdict,
)
from charlie_work.rework_prompts import _render_rework_prompt
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub


# The exact required_changes item the fleet reviewer emitted on PR #1641
# (issue #1642's "Observed" section, verbatim).
_INCIDENT_REQUIRED_CHANGE = (
    "Confirm explicitly (human/operator call, not automated rework) whether "
    "landing as_property/as_staticmethod now with no current caller is "
    "acceptable given they're scoped for the future L09 leaf per the design "
    "doc Section 3.3 -- either accept as-is, or defer them to the PR that "
    "first consumes them."
)

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# ---------------------------------------------------------------------------
# Layer 1: human_decision_marker_match unit tests
# ---------------------------------------------------------------------------


def test_marker_match_incident_text() -> None:
    """The PR #1641 incident item must match the default marker set."""
    markers = ReviewConfig().human_decision_markers
    result = human_decision_marker_match([_INCIDENT_REQUIRED_CHANGE], markers)
    assert result is not None
    item, marker = result
    assert item == _INCIDENT_REQUIRED_CHANGE
    assert marker in _INCIDENT_REQUIRED_CHANGE.lower()


def test_marker_match_ordinary_finding_returns_none() -> None:
    """An ordinary rework finding must not match (positive control)."""
    markers = ReviewConfig().human_decision_markers
    assert (
        human_decision_marker_match(["Add a regression test for the empty-list case"], markers)
        is None
    )


def test_marker_match_is_case_insensitive() -> None:
    markers = ("SIGN-OFF",)
    assert human_decision_marker_match(["needs sign-off from QA"], markers) == (
        "needs sign-off from QA",
        "sign-off",
    )


def test_marker_match_empty_inputs() -> None:
    markers = ReviewConfig().human_decision_markers
    assert human_decision_marker_match([], markers) is None
    assert human_decision_marker_match([_INCIDENT_REQUIRED_CHANGE], ()) is None
    assert human_decision_marker_match([_INCIDENT_REQUIRED_CHANGE], []) is None


def test_marker_match_tolerates_non_str_items() -> None:
    """The external-findings replacement path can put non-str values in the
    list -- the guard must not crash on them (mirrors record_review's own
    ``str(item)`` event-payload coercion)."""
    assert human_decision_marker_match([42], ("operator",)) is None
    assert human_decision_marker_match(["call the operator", 42], ("operator",)) == (
        "call the operator",
        "operator",
    )


def test_reclassify_human_call_verdict_reclassifies_matching_request_changes() -> None:
    """A ``request_changes`` verdict with a human-call finding under an
    authored provenance becomes ``blocked`` with the match carried through."""
    markers = ReviewConfig().human_decision_markers
    decision, match = reclassify_human_call_verdict(
        "request_changes",
        [_INCIDENT_REQUIRED_CHANGE],
        markers=markers,
        verdict_provenance="fresh_llm_review",
    )
    assert decision == "blocked"
    assert match is not None and match[0] == _INCIDENT_REQUIRED_CHANGE


def test_reclassify_human_call_verdict_passthrough_cases() -> None:
    """Everything else passes through unchanged: non-request_changes
    decisions, mechanical ``*_auto_reject`` provenances, and findings with
    no human-call language."""
    markers = ReviewConfig().human_decision_markers

    # ``blocked`` and ``approved`` are never reclassified.
    for decision in ("blocked", "approved"):
        assert reclassify_human_call_verdict(
            decision,
            [_INCIDENT_REQUIRED_CHANGE],
            markers=markers,
            verdict_provenance="fresh_llm_review",
        ) == (decision, None)

    # Mechanical auto-reject provenances are exempt even when the text
    # matches a marker (CI check annotations, not reviewer judgment).
    for provenance in ("ci_gate_auto_reject", "janitor_auto_reject"):
        assert reclassify_human_call_verdict(
            "request_changes",
            ["Lint & Format: missing whitespace around operator"],
            markers=markers,
            verdict_provenance=provenance,
        ) == ("request_changes", None)

    # An ordinary finding never reclassifies.
    assert reclassify_human_call_verdict(
        "request_changes",
        ["Add a regression test for the empty-list case"],
        markers=markers,
        verdict_provenance="fresh_llm_review",
    ) == ("request_changes", None)


# ---------------------------------------------------------------------------
# Layer 2: record_review integration tests
# ---------------------------------------------------------------------------


def _app(
    tmp_path: Path, config: OrchestratorConfig | None = None
) -> tuple[OrchestratorApp, FakeGitHub]:
    config = config or OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _decision_file(app: OrchestratorApp, pr_number: int = 456) -> dict[str, Any]:
    path = app.paths.prs / f"pr-{pr_number}" / "review-decision.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_human_call_verdict_reclassifies_to_blocked(tmp_path: Path) -> None:
    """The incident verdict must be recorded as ``blocked`` and routed to
    the operator queue -- not to the rework lane."""
    app, fake_gh = _app(tmp_path)
    result = app.record_review(
        456,
        "request_changes",
        summary="Reviewer asks for an explicit human call on landing order.",
        required_changes=[_INCIDENT_REQUIRED_CHANGE],
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True, result.message
    assert result.data["decision"] == "blocked"
    assert result.data["reclassified_from"] == "request_changes"
    # No rework brief was written: the verdict went to the blocked lane.
    assert result.data["rework_path"] is None
    assert not (app.paths.prs / "pr-456" / "rework-prompt.md").exists()

    decision = _decision_file(app)
    assert decision["decision"] == "blocked"
    assert decision["escalated"] is False

    state = load_state(app.paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["status"] == "blocked"
    assert pr_state["decision"] == "blocked"
    # A reclassified verdict is not a rework cycle: the cap budget is untouched.
    assert pr_state["request_changes_count"] == 0
    # Same escalation shape as a correctly filed ``blocked`` verdict:
    # judgment-class operator queue, not the rework lane.
    issue_state = state["issues"]["123"]
    assert issue_state["status"] == "blocked"
    assert issue_state["escalation_reason"] == "review_blocked"
    assert issue_state["reason_class"] == "judgment"
    assert (123, app.config.labels.human_needed) in fake_gh.labels_added
    assert (123, app.config.labels.needs_rework) not in fake_gh.labels_added


def test_reclassification_event_carries_matched_item(tmp_path: Path) -> None:
    """``review_decision_reclassified_blocked`` must name the matched item
    so the operator queue entry is actionable without reading the verdict."""
    app, _ = _app(tmp_path)
    app.record_review(
        456,
        "request_changes",
        summary="Reviewer asks for an explicit human call on landing order.",
        required_changes=[_INCIDENT_REQUIRED_CHANGE],
        verdict_provenance="fresh_llm_review",
    )
    events = query_events(app.paths.state_file, kind="review_decision_reclassified_blocked")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 456
    assert payload["issue_number"] == 123
    assert payload["original_decision"] == "request_changes"
    assert payload["matched_item"] == _INCIDENT_REQUIRED_CHANGE
    assert payload["matched_marker"] in _INCIDENT_REQUIRED_CHANGE.lower()


def test_ordinary_request_changes_still_routes_to_rework(tmp_path: Path) -> None:
    """Positive control: a request_changes verdict with ordinary findings
    must keep routing to the rework lane, unchanged."""
    app, fake_gh = _app(tmp_path)
    result = app.record_review(
        456,
        "request_changes",
        summary="Missing regression coverage for the empty-list case.",
        required_changes=["Add a regression test for the empty-list case"],
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True, result.message
    assert result.data["decision"] == "request_changes"
    assert result.data["reclassified_from"] is None
    assert result.data["rework_path"] is not None
    assert Path(result.data["rework_path"]).exists()

    decision = _decision_file(app)
    assert decision["decision"] == "request_changes"

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, app.config.labels.needs_rework) in fake_gh.labels_added
    assert (123, app.config.labels.human_needed) not in fake_gh.labels_added
    assert query_events(app.paths.state_file, kind="review_decision_reclassified_blocked") == []


def test_human_call_in_summary_only_also_reclassifies(tmp_path: Path) -> None:
    """A reviewer that writes the human call in ``summary`` instead of the
    structured list is caught identically: the #792 "derived" channel folds
    the summary into effective_required_changes before the guard runs."""
    app, _ = _app(tmp_path)
    result = app.record_review(
        456,
        "request_changes",
        summary=(
            "This needs a human/operator call on whether the unused helpers "
            "may land now -- not automated rework."
        ),
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True, result.message
    assert result.data["decision"] == "blocked"
    assert result.data["reclassified_from"] == "request_changes"
    assert _decision_file(app)["decision"] == "blocked"


def test_auto_reject_provenance_not_reclassified(tmp_path: Path) -> None:
    """Mechanically generated verdicts are exempt: a CI-gate auto-reject's
    required_changes come from check annotations, and an annotation like
    pycodestyle's "missing whitespace around operator" must not park a
    mechanical CI-red verdict in the operator queue."""
    app, fake_gh = _app(tmp_path)
    result = app.record_review(
        456,
        "request_changes",
        summary="CI failed on Lint & Format; push a fix",
        required_changes=["Lint & Format: missing whitespace around operator"],
        verdict_provenance="ci_gate_auto_reject",
    )
    assert result.ok is True, result.message
    assert result.data["decision"] == "request_changes"
    assert result.data["reclassified_from"] is None
    assert result.data["rework_path"] is not None
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, app.config.labels.needs_rework) in fake_gh.labels_added
    assert query_events(app.paths.state_file, kind="review_decision_reclassified_blocked") == []


def test_custom_markers_config_is_respected(tmp_path: Path) -> None:
    """The marker list is config, not code: a configured marker the defaults
    lack must reclassify, and an empty list disables the guard entirely."""
    config = OrchestratorConfig(review=ReviewConfig(human_decision_markers=("needs-decision",)))
    app, _ = _app(tmp_path, config)
    result = app.record_review(
        456,
        "request_changes",
        summary="One item is a needs-decision call, not rework.",
        required_changes=["Flag the branch ordering as needs-decision"],
        verdict_provenance="fresh_llm_review",
    )
    assert result.data["decision"] == "blocked"

    config_off = OrchestratorConfig(review=ReviewConfig(human_decision_markers=()))
    app_off, _ = _app(tmp_path / "off", config_off)
    result_off = app_off.record_review(
        456,
        "request_changes",
        summary="Reviewer asks for an explicit human call on landing order.",
        required_changes=[_INCIDENT_REQUIRED_CHANGE],
        verdict_provenance="fresh_llm_review",
    )
    assert result_off.data["decision"] == "request_changes"
    assert result_off.data["rework_path"] is not None


# ---------------------------------------------------------------------------
# Layer 3: rework-prompt section
# ---------------------------------------------------------------------------


def test_rendered_rework_prompt_carries_no_operator_decisions_section(
    tmp_path: Path,
) -> None:
    """The rendered rework brief must state that it carries no operator
    decisions and point at the ``.worker-outcome.json`` blocked channel --
    the fix for the rework worker asserting an operator decision (PR #1641)."""
    config = OrchestratorConfig()
    state_file = runtime_paths(tmp_path, config.runtime.state_dir).state_file
    pr_dir = state_file.parent / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "needs rework",
                "required_changes": ["fix the bug"],
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": 456,
        "title": "Fake PR title",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-test",
    }
    rendered = _render_rework_prompt(state_file, pr, 123, "A dispatch note.", config)
    assert "## No operator decisions in this brief" in rendered
    assert "NO operator decisions" in rendered
    assert ".worker-outcome.json" in rendered
    assert '"outcome": "blocked"' in rendered


# ---------------------------------------------------------------------------
# Layer 4: config parsing
# ---------------------------------------------------------------------------


def test_load_config_human_decision_markers_yaml_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "review:\n  human_decision_markers:\n    - needs-decision\n    - human call\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.review.human_decision_markers == ("needs-decision", "human call")


def test_load_config_human_decision_markers_rejects_non_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("review:\n  human_decision_markers: 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_load_config_human_decision_markers_rejects_non_string_element(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("review:\n  human_decision_markers:\n    - 42\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_example_configs_load_with_default_markers() -> None:
    """Both shipped example configs document the knob (commented out) and
    load cleanly with the default marker set."""
    for name in (
        "orchestrator.config.devin.yaml",
        "orchestrator.config.claude-code.yaml",
    ):
        cfg = load_config(_EXAMPLES_DIR / name)
        assert "operator" in cfg.review.human_decision_markers
        assert "not automated rework" in cfg.review.human_decision_markers
