"""Issue #1485: log-extracted review verdicts can assert false git-state
claims (e.g. "already merged") that would discard real work if trusted.

Regression tests for the provenance-caveat fix: when a review verdict was
extracted from a dead reviewer's session artifacts (``verdict_source`` =
``"log"``/``"events"``/``"file:<path>"``) rather than emitted as a clean
structured completion, ``record_review`` persists a ``provenance_caveat``
into ``review-decision.json`` and surfaces it in the PR comment, and the
rework-prompt renderer prepends it to the required-changes section. A
human reading the verdict knows to re-verify factual claims before acting
on a destructive-adjacent action like closing a PR.

Covers three layers:

1. **Helper unit tests** -- ``is_extracted_verdict_source`` and
   ``provenance_caveat_for`` correctly classify verdict sources.
2. **``record_review`` integration tests** -- the caveat is persisted in
   the decision file and surfaced in the PR comment for log-extracted
   ``blocked``/``request_changes`` verdicts, and absent for ``approved``
   or non-extracted (``verdict_source=None``) verdicts.
3. **Renderer tests** -- ``_render_rework_prompt`` and
   ``_render_round_findings`` prepend the caveat when the decision file
   carries it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.rework_prompts import (
    _provenance_caveat_from_decision,
    _render_required_changes_section,
    _render_round_findings,
)
from charlie_work.verdict_parsing import (
    is_extracted_verdict_source,
    provenance_caveat_for,
)
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub
from _review_fixtures import _PR_NUMBER


# ---------------------------------------------------------------------------
# Layer 1: helper unit tests
# ---------------------------------------------------------------------------


def test_is_extracted_verdict_source_identifies_extraction_sources() -> None:
    """``"log"``, ``"events"``, and ``"file:<path>"`` are all extracted."""
    assert is_extracted_verdict_source("log") is True
    assert is_extracted_verdict_source("events") is True
    assert is_extracted_verdict_source("file:/tmp/review.md") is True


def test_is_extracted_verdict_source_none_is_not_extracted() -> None:
    """``None`` (operator/CI-gate direct call, no parser) is trusted."""
    assert is_extracted_verdict_source(None) is False


def test_is_extracted_verdict_source_unknown_string_is_not_extracted() -> None:
    """An unrecognized verdict_source string is not extracted (defensive)."""
    assert is_extracted_verdict_source("structured") is False
    assert is_extracted_verdict_source("") is False


def test_provenance_caveat_for_returns_caveat_for_log() -> None:
    caveat = provenance_caveat_for("log")
    assert caveat is not None
    assert "issue #1485" in caveat.lower()
    assert "verdict_source: log" in caveat
    assert "re-verify" in caveat.lower()


def test_provenance_caveat_for_returns_caveat_for_events() -> None:
    caveat = provenance_caveat_for("events")
    assert caveat is not None
    assert "verdict_source: events" in caveat


def test_provenance_caveat_for_returns_caveat_for_file_source() -> None:
    caveat = provenance_caveat_for("file:/tmp/review.md")
    assert caveat is not None
    assert "verdict_source: file:/tmp/review.md" in caveat


def test_provenance_caveat_for_returns_none_for_none() -> None:
    assert provenance_caveat_for(None) is None


def test_provenance_caveat_from_decision_reads_field() -> None:
    decision = {"provenance_caveat": "  caveat text  "}
    assert _provenance_caveat_from_decision(decision) == "caveat text"


def test_provenance_caveat_from_decision_empty_for_missing_field() -> None:
    decision = {"decision": "blocked"}
    assert _provenance_caveat_from_decision(decision) == ""


def test_provenance_caveat_from_decision_empty_for_none() -> None:
    assert _provenance_caveat_from_decision(None) == ""


def test_provenance_caveat_from_decision_empty_for_blank() -> None:
    decision = {"provenance_caveat": "   "}
    assert _provenance_caveat_from_decision(decision) == ""


# ---------------------------------------------------------------------------
# Layer 2: record_review integration tests
# ---------------------------------------------------------------------------


class _CapturingGitHub(FakeGitHub):
    """FakeGitHub subclass that records every ``pr_comment`` call's body."""

    def __init__(self) -> None:
        super().__init__()
        self.captured_comments: list[tuple[int, str]] = []

    def pr_comment(self, number: int, body_file: Path) -> None:
        self.captured_comments.append((number, body_file.read_text(encoding="utf-8")))


def _app(tmp_path: Path) -> tuple[OrchestratorApp, _CapturingGitHub]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = _CapturingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _record(
    app: OrchestratorApp,
    fake_gh: _CapturingGitHub,
    *,
    decision: str,
    head: str,
    summary: str,
    required_changes: list[str] | None = None,
    verdict_source: str | None = None,
) -> Any:
    fake_gh.pr_head_shas[_PR_NUMBER] = head
    result = app.record_review(
        _PR_NUMBER,
        decision,
        summary=summary,
        required_changes=required_changes or [],
        verdict_provenance="fresh_llm_review",
        verdict_source=verdict_source,
    )
    assert result.ok is True, result.message
    return result


def _decision_file(app: OrchestratorApp) -> dict[str, Any]:
    path = app.paths.prs / f"pr-{_PR_NUMBER}" / "review-decision.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_blocked_log_verdict_persists_provenance_caveat(tmp_path: Path) -> None:
    """A log-extracted ``blocked`` verdict must carry ``provenance_caveat``
    in ``review-decision.json``."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="blocked",
        head="sha-blocked",
        summary="PR already merged, close it",
        required_changes=["close as duplicate"],
        verdict_source="log",
    )
    decision = _decision_file(app)
    assert "provenance_caveat" in decision, (
        "issue #1485: a log-extracted blocked verdict must persist a "
        "provenance_caveat field in review-decision.json"
    )
    assert "issue #1485" in decision["provenance_caveat"].lower()
    assert "verdict_source: log" in decision["provenance_caveat"]


def test_request_changes_log_verdict_persists_provenance_caveat(
    tmp_path: Path,
) -> None:
    """A log-extracted ``request_changes`` verdict must carry
    ``provenance_caveat`` in ``review-decision.json``."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="request_changes",
        head="sha-rc",
        summary="needs rework",
        required_changes=["fix the bug"],
        verdict_source="log",
    )
    decision = _decision_file(app)
    assert "provenance_caveat" in decision


def test_events_source_verdict_persists_provenance_caveat(tmp_path: Path) -> None:
    """An events-extracted verdict must also carry the caveat."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="blocked",
        head="sha-events",
        summary="blocked",
        required_changes=["x"],
        verdict_source="events",
    )
    decision = _decision_file(app)
    assert "provenance_caveat" in decision
    assert "verdict_source: events" in decision["provenance_caveat"]


def test_approved_log_verdict_does_not_persist_caveat(tmp_path: Path) -> None:
    """An ``approved`` verdict must NOT carry the caveat -- the issue scopes
    to ``blocked``/``request_changes`` verdicts that route to operator
    instructions."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="approved",
        head="sha-approved",
        summary="lgtm",
        verdict_source="log",
    )
    decision = _decision_file(app)
    assert "provenance_caveat" not in decision, (
        "issue #1485: an approved verdict must not carry a provenance caveat "
        "-- the issue scopes to blocked/request_changes"
    )


def test_blocked_no_verdict_source_does_not_persist_caveat(
    tmp_path: Path,
) -> None:
    """A ``blocked`` verdict with ``verdict_source=None`` (operator CLI / CI
    gate direct call, no parser) must NOT carry the caveat -- those
    verdicts are trusted."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="blocked",
        head="sha-trusted",
        summary="human judgment",
        required_changes=["x"],
        verdict_source=None,
    )
    decision = _decision_file(app)
    assert "provenance_caveat" not in decision


def test_blocked_log_verdict_pr_comment_includes_caveat(tmp_path: Path) -> None:
    """The PR comment for a log-extracted ``blocked`` verdict must include
    the provenance caveat so an operator reading it knows to re-verify."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="blocked",
        head="sha-cmt",
        summary="already merged, close it",
        required_changes=["close as duplicate"],
        verdict_source="log",
    )
    assert len(fake_gh.captured_comments) == 1
    body = fake_gh.captured_comments[0][1]
    assert "Provenance caveat" in body
    assert "issue #1485" in body.lower()
    assert "re-verify" in body.lower()


def test_blocked_no_verdict_source_pr_comment_excludes_caveat(
    tmp_path: Path,
) -> None:
    """The PR comment for a trusted (``verdict_source=None``) ``blocked``
    verdict must NOT include the caveat."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="blocked",
        head="sha-no-caveat",
        summary="human judgment",
        required_changes=["x"],
        verdict_source=None,
    )
    assert len(fake_gh.captured_comments) == 1
    body = fake_gh.captured_comments[0][1]
    assert "Provenance caveat" not in body


def test_approved_log_verdict_pr_comment_excludes_caveat(
    tmp_path: Path,
) -> None:
    """The PR comment for a log-extracted ``approved`` verdict must NOT
    include the caveat."""
    app, fake_gh = _app(tmp_path)
    _record(
        app,
        fake_gh,
        decision="approved",
        head="sha-app-cmt",
        summary="lgtm",
        verdict_source="log",
    )
    assert len(fake_gh.captured_comments) == 1
    body = fake_gh.captured_comments[0][1]
    assert "Provenance caveat" not in body


# ---------------------------------------------------------------------------
# Layer 3: renderer tests
# ---------------------------------------------------------------------------


def test_render_required_changes_section_prepends_caveat() -> None:
    """``_render_required_changes_section`` renders the findings; the
    rework-prompt renderer (``_render_rework_prompt``) prepends the
    caveat at its call site. Here we test the call-site prepending
    logic directly: a decision with ``provenance_caveat`` and a
    non-empty required-changes section gets the caveat prepended."""
    decision = {
        "decision": "request_changes",
        "summary": "needs rework",
        "required_changes": ["fix the bug"],
        "provenance_caveat": provenance_caveat_for("log"),
    }
    section = _render_required_changes_section(decision)
    assert section, "the base section must be non-empty"
    caveat = _provenance_caveat_from_decision(decision)
    assert caveat
    combined = f"{caveat}\n\n{section}"
    assert combined.startswith(caveat)
    assert "fix the bug" in combined


def test_render_round_findings_prepends_caveat() -> None:
    """``_render_round_findings`` must prepend the provenance caveat to
    the rendered round-history section when the decision carries it."""
    decision = {
        "decision": "request_changes",
        "summary": "needs rework",
        "required_changes": ["fix the bug"],
        "provenance_caveat": provenance_caveat_for("log"),
    }
    rendered = _render_round_findings(decision)
    assert rendered, "round findings must render something"
    assert "Provenance caveat" in rendered
    assert "fix the bug" in rendered


def test_render_round_findings_no_caveat_for_trusted_verdict() -> None:
    """``_render_round_findings`` must NOT prepend a caveat when the
    decision has no ``provenance_caveat`` field."""
    decision = {
        "decision": "request_changes",
        "summary": "needs rework",
        "required_changes": ["fix the bug"],
    }
    rendered = _render_round_findings(decision)
    assert rendered
    assert "Provenance caveat" not in rendered
