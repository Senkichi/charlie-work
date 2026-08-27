"""Tests for issue #1460: attachment-budget dispatch clause.

Mirrors ``tests/test_module_map.py``'s fail-soft pattern:

* Marker file absent -> empty clause, no warning event (this feature is
  opt-in per repo, gated purely on `.attachment-budgets.json` presence).
* Marker file present and structurally valid -> the static placement clause,
  containing the ``check-file`` command a worker is told to run.
* Marker file present but malformed -> empty clause plus a
  ``worker_attachment_budget_failed`` warning event in events.db.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, dumps, generate
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp, WORKER_PROMPT_KEYS


def _app(tmp_path: Path) -> OrchestratorApp:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh=None)


def _events_with_kind(paths, kind: str) -> list[tuple]:
    db_path = paths.state_file.parent / "events.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT kind, level, payload FROM events WHERE kind = ?", (kind,)
        ).fetchall()
    finally:
        conn.close()


def test_marker_absent_yields_empty_clause_no_event(tmp_path: Path) -> None:
    app = _app(tmp_path)

    value = app._build_attachment_budget_value(issue_number=1)

    assert value == ""
    rows = _events_with_kind(app.paths, "worker_attachment_budget_failed")
    assert rows == [], "marker-file absence is not a failure; no warning expected"


def test_marker_valid_yields_nonempty_clause_with_check_file(tmp_path: Path) -> None:
    baseline_doc = generate((), generated_by="test", generated_at="2026-08-25T00:00:00Z", floor=1)
    (tmp_path / BASELINE_FILENAME).write_text(dumps(baseline_doc), encoding="utf-8")
    app = _app(tmp_path)

    value = app._build_attachment_budget_value(issue_number=2)

    assert value != ""
    assert "check-file" in value
    assert "attachment_contracts" in value
    rows = _events_with_kind(app.paths, "worker_attachment_budget_failed")
    assert rows == []


def test_marker_valid_clause_instructs_worker_to_post_advisories_comment(
    tmp_path: Path,
) -> None:
    """Issue #1466: the dispatch clause must instruct the worker to publish
    the advisories PR comment with the stable marker line, so the review-
    packet builder's PR-comment channel has something to read."""
    from charlie_work.attachment_contracts.hook_entry import ADVISORY_COMMENT_MARKER

    baseline_doc = generate((), generated_by="test", generated_at="2026-08-25T00:00:00Z", floor=1)
    (tmp_path / BASELINE_FILENAME).write_text(dumps(baseline_doc), encoding="utf-8")
    app = _app(tmp_path)

    value = app._build_attachment_budget_value(issue_number=2)

    assert ADVISORY_COMMENT_MARKER in value, (
        "dispatch clause must tell the worker to start the advisories PR "
        "comment with the stable marker line"
    )
    # The clause must name the schema fields so the worker posts the right
    # shape -- not a free-form comment the builder cannot parse.
    for field in ("severity", "file", "identity", "message", "redirect", "timestamp"):
        assert field in value, f"dispatch clause must name the `{field}` schema field"


def test_marker_malformed_yields_empty_clause_and_warning_event(tmp_path: Path) -> None:
    (tmp_path / BASELINE_FILENAME).write_text("not valid json {{{", encoding="utf-8")
    app = _app(tmp_path)

    value = app._build_attachment_budget_value(issue_number=3)

    assert value == ""
    rows = _events_with_kind(app.paths, "worker_attachment_budget_failed")
    assert len(rows) == 1, f"expected one worker_attachment_budget_failed event; got {rows}"
    kind, level, _payload = rows[0]
    assert kind == "worker_attachment_budget_failed"
    assert level == "warning"


def test_marker_wrong_schema_version_yields_empty_clause_and_warning(tmp_path: Path) -> None:
    """A structurally-valid JSON object that fails baseline.load's own
    version check (TamperError) must also be treated as malformed."""
    import json

    (tmp_path / BASELINE_FILENAME).write_text(
        json.dumps({"version": 999, "entries": []}), encoding="utf-8"
    )
    app = _app(tmp_path)

    value = app._build_attachment_budget_value(issue_number=4)

    assert value == ""
    rows = _events_with_kind(app.paths, "worker_attachment_budget_failed")
    assert len(rows) == 1


def test_worker_attachment_budget_failed_registered_as_warning() -> None:
    assert "worker_attachment_budget_failed" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["worker_attachment_budget_failed"] == "warning"


def test_attachment_budget_is_a_worker_prompt_key() -> None:
    assert "attachment_budget" in WORKER_PROMPT_KEYS
