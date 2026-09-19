"""Reviewer-verdict extraction from session logs and cross-family verdict normalization.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import (
    OrchestratorApp,
    _parse_review_verdict_from_log,
    _summary_is_vacuous,
)


def test_parse_review_verdict_from_log_extracts_last_fenced_json(tmp_path: Path) -> None:
    """Issue #507: parse the last fenced JSON verdict block from a log."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        'Some earlier output\n```json\n{"decision": "blocked", "summary": "old"}\n```\n'
        'Final verdict:\n```json\n{\n  "decision": "approved",\n  "summary": "lgtm",\n  "required_changes": []\n}\n```\n',
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "approved"
    assert verdict["summary"] == "lgtm"
    assert verdict["required_changes"] == []


def test_parse_review_verdict_from_log_extracts_json_after_language_tagged_fence(
    tmp_path: Path,
) -> None:
    """Regression: ``_VERDICT_FENCE_RE`` previously only recognized an opening
    fence tagged bare or ``json`` (``` ```(?:json)?\\s*\\n `` ``), so a
    reviewer quoting evidence in a ```python fence before its final verdict
    fence would desync the pairing entirely -- the ```python fence's own
    opening backtick never matched, so its *closing* bare ``` got misread as
    a new opening and swallowed everything up to the *next* fence's opening,
    permanently misaligning the scan. This mirrors the exact structure that
    hid a real, well-formed verdict in a production cross-family report
    (PR #802); the same regex is duplicated here in ``workflow.py`` (kept
    latent so far by per-event stream-json decoding, but a real defect)."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        "Citing the bug:\n```python\ndef broken():\n    return None\n```\n"
        "That's a real problem.\n\n"
        'Final verdict:\n```json\n{\n  "decision": "request_changes",\n'
        '  "summary": "broken() returns None instead of raising",\n'
        '  "required_changes": ["Raise instead of returning None"]\n}\n```\n',
        encoding="utf-8",
    )

    verdict = _parse_review_verdict_from_log(log)

    assert verdict is not None
    assert verdict["decision"] == "request_changes"
    assert verdict["summary"] == "broken() returns None instead of raising"
    assert verdict["required_changes"] == ["Raise instead of returning None"]


def test_parse_review_verdict_from_log_requires_valid_decision(tmp_path: Path) -> None:
    """Issue #507: only accepted decisions and non-empty summaries are valid."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        '```json\n{"decision": "maybe", "summary": "?"}\n```\n',
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_log(log) is None


def test_parse_review_verdict_from_log_rejects_empty_request_changes_summary(
    tmp_path: Path,
) -> None:
    """Issue #507: request_changes with an empty summary is not a valid verdict."""
    log = tmp_path / "review.claude.log"
    log.write_text(
        '```json\n{"decision": "request_changes", "summary": "   "}\n```\n',
        encoding="utf-8",
    )

    assert _parse_review_verdict_from_log(log) is None


def test_cross_family_request_changes_verdict_persists_required_changes(
    tmp_path: Path,
) -> None:
    """End-to-end: a rescue-tier request_changes verdict's decision/summary/
    required_changes, recorded the same way the rescue tier's own
    record_review call site does, ends up with a populated
    ``required_changes`` in review-decision.json -- the exact defect this fix
    closes (8 of 20 request_changes verdicts had it silently empty).

    The decision/summary/required_changes below are the literal values a
    JSON verdict block of
    ``{"decision": "request_changes", "summary": "file.py:10 has a real bug
    that breaks X", "required_changes": ["Fix the off-by-one in
    file.py:10", "Add a regression test for the empty-list case"]}``
    used to parse to, back when ``parse_cross_family_verdict`` (deleted in
    the role-config Phase 2 cleanup) extracted them from a report body."""
    verdict_decision = "request_changes"
    verdict_summary = "file.py:10 has a real bug that breaks X"
    verdict_required_changes = (
        "Fix the off-by-one in file.py:10",
        "Add a regression test for the empty-list case",
    )

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mirrors the rescue tier's own record_review call site.
    # PR 456 is FakeGitHub's seeded default PR.
    result = app.record_review(
        456,
        verdict_decision,
        summary=verdict_summary,
        required_changes=verdict_required_changes,
        verdict_provenance="rescue_review",
    )
    assert result.ok

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == [
        "Fix the off-by-one in file.py:10",
        "Add a regression test for the empty-list case",
    ]
    assert decision["summary"] == "file.py:10 has a real bug that breaks X"


def test_cross_family_legacy_path_verdict_with_empty_required_changes_gets_derived(
    tmp_path: Path,
) -> None:
    """AC-6 (rescue-tier producer): a request_changes verdict with an empty
    required_changes and a real, non-vacuous summary -- the shape the legacy
    Markdown-only cross-family parse path (no JSON verdict block, deleted
    along with ``parse_cross_family_verdict`` in the role-config Phase 2
    cleanup) used to produce, since it only ever extracted a summary, never
    itemized findings -- always arrives at record_review as
    required_changes=(). This is the shape record_review's derivation
    exists for. Contrast with the (deleted) AC-8 malformed-verdict tests: a
    JSON verdict block declaring request_changes with an empty
    required_changes was diverted to MalformedCrossFamilyVerdict before ever
    reaching record_review (issue #795) -- this scenario has no JSON block
    at all, so that defense-in-depth layer never applied here and
    record_review's own derivation is what prevents the content-free
    outcome.

    ``verdict_summary`` below is the literal value
    ``"**MAJOR**\\nreal bug\\n\\nVerdict: MAJOR issues block merge"``
    used to parse to via the legacy path's verdict-line extraction."""
    verdict_decision = "request_changes"
    verdict_summary = "MAJOR issues block merge"
    verdict_required_changes: tuple[str, ...] = ()
    assert verdict_summary and not _summary_is_vacuous(verdict_summary)

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Mirrors the rescue tier's own record_review call site.
    result = app.record_review(
        456,
        verdict_decision,
        summary=verdict_summary,
        required_changes=verdict_required_changes,
        verdict_provenance="rescue_review",
    )
    assert result.ok is True

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == [verdict_summary]
    assert decision["findings_channel"] == "derived"
