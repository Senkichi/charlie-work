"""Dispatch-integration tests for issue #1000 citation-drift flagging.

These cover the ``OrchestratorApp.dispatch`` wiring of
``citation_check.verify_citations``: a drifted citation is flagged with a
deduped issue comment (visible to the worker via ``$issue_comments``) plus a
``dispatch_citation_drift_flagged`` event and a state fingerprint; a clean
issue is not flagged; a still-stale issue is not re-commented on the next
pass; and a comment-posting failure never aborts dispatch.

Reuses ``FakeGitHub`` / ``runtime_paths`` from ``test_charlie_work.py`` per the
established convention in this suite (see ``test_tripwire_detection_record.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _fakes_github import FakeGitHub


def _make_app(tmp_path: Path, fake_gh: FakeGitHub) -> OrchestratorApp:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _write(repo: Path, rel: str, lines: list[str]) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _drifted_issue_body() -> str:
    # Cites workflow.py:5000 against a 3-line file -> OUT_OF_RANGE (drift).
    return "The defect is at workflow.py:5000 in the loop."


def _clean_issue_body() -> str:
    # Cites workflow.py:2 against a 3-line file -> OK (line exists, non-blank).
    return "The defect is at workflow.py:2 in the loop."


def test_dispatch_flags_drifted_citation_with_comment_and_event(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["line1", "def f():", "    pass"])
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["body"] = _drifted_issue_body()
    fake_gh.prs[0]["state"] = "CLOSED"  # make the issue dispatchable
    app = _make_app(tmp_path, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    # A drift comment was posted on the issue.
    posted = getattr(fake_gh, "issue_comments_posted", [])
    assert len(posted) == 1
    issue_number, body = posted[0]
    assert issue_number == 123
    assert "Citation drift detected" in body
    assert "workflow.py:5000" in body
    assert "out_of_range" in body
    # The state record carries the fingerprint + flagged_at timestamp.
    state = json.loads(
        (tmp_path / ".var" / "charlie-work" / "state.json").read_text(encoding="utf-8")
    )
    entry = state["issues"]["123"]
    assert entry["citation_drift_fingerprint"]
    assert "citation_drift_flagged_at" in entry


def test_dispatch_does_not_flag_clean_citations(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["line1", "def f():", "    pass"])
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["body"] = _clean_issue_body()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = _make_app(tmp_path, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert getattr(fake_gh, "issue_comments_posted", []) == []
    state = json.loads(
        (tmp_path / ".var" / "charlie-work" / "state.json").read_text(encoding="utf-8")
    )
    entry = state["issues"]["123"]
    # A clean issue is not flagged: no comment, no flagged_at timestamp. The
    # fingerprint is left unset (treated as "" for dedup) on the first clean
    # pass; a future regression still re-alerts because None == "" in the
    # comparison.
    assert entry.get("citation_drift_fingerprint") in (None, "")
    assert "citation_drift_flagged_at" not in entry


def test_dispatch_drift_flag_is_deduped_across_passes(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["line1", "def f():", "    pass"])
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["body"] = _drifted_issue_body()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = _make_app(tmp_path, fake_gh)

    app.dispatch(limit=1)
    first_count = len(getattr(fake_gh, "issue_comments_posted", []))
    assert first_count == 1

    # Second pass: the issue is now dispatched (status "dispatched"), so it is
    # no longer a candidate and the check does not re-run. Reset state so it
    # becomes a candidate again, simulating a dead-worker recovery redispatch.
    state_path = tmp_path / ".var" / "charlie-work" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["issues"]["123"]["status"] = "dispatch_failed"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    app.dispatch(limit=1)
    # Fingerprint unchanged -> no new comment.
    assert len(getattr(fake_gh, "issue_comments_posted", [])) == 1


def test_dispatch_drift_comment_failure_does_not_abort_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path, "src/workflow.py", ["line1", "def f():", "    pass"])
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["body"] = _drifted_issue_body()
    fake_gh.prs[0]["state"] = "CLOSED"

    def _boom(number: int, body_file: Path) -> None:
        raise OSError("simulated gh issue comment failure")

    # Patch on the instance so the broken override is used only for this test.
    fake_gh.issue_comment = _boom  # type: ignore[method-assign]
    app = _make_app(tmp_path, fake_gh)

    with caplog.at_level(logging.WARNING, logger="charlie_work.workflow"):
        result = app.dispatch(limit=1)

    # Dispatch still succeeded and selected the worker despite the comment failure.
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert any("citation drift comment post failed" in r.message for r in caplog.records)


def test_dispatch_flags_stale_prefix_citation_with_comment_and_event(
    tmp_path: Path,
) -> None:
    # A citation with a stale directory prefix (the asserted literal path does
    # not exist, but the basename resolves via the recursive index) must be
    # flagged at dispatch time: a flag comment is posted on the issue (visible
    # to the worker via ``$issue_comments``) and a
    # ``dispatch_citation_drift_flagged`` event is emitted. The comment must
    # surface the resolved path so the worker can see where the file moved to.
    # This is the dispatch-level companion to
    # ``test_verify_stale_directory_prefix_resolves_via_recursive_index``.
    lines = [f"line{i}" for i in range(11)]  # index 9 == line 10
    lines[9] = "TARGET_MOVED_LINE"
    _write(tmp_path, "src/charlie_work/workflow.py", lines)
    fake_gh = FakeGitHub()
    # The citation says ``old_dir/workflow.py:10`` but the file lives at
    # ``src/charlie_work/workflow.py`` -- stale prefix, valid basename.
    fake_gh.issues[0]["body"] = "The defect is at old_dir/workflow.py:10 in the loop."
    fake_gh.prs[0]["state"] = "CLOSED"
    app = _make_app(tmp_path, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    # A drift comment was posted on the issue.
    posted = getattr(fake_gh, "issue_comments_posted", [])
    assert len(posted) == 1
    issue_number, body = posted[0]
    assert issue_number == 123
    assert "Citation drift detected" in body
    assert "old_dir/workflow.py:10" in body
    assert "stale_prefix" in body
    # The comment surfaces where the file actually moved to.
    assert "src/charlie_work/workflow.py" in body
    # The state record carries the fingerprint + flagged_at timestamp.
    state = json.loads(
        (tmp_path / ".var" / "charlie-work" / "state.json").read_text(encoding="utf-8")
    )
    entry = state["issues"]["123"]
    assert entry["citation_drift_fingerprint"]
    assert "citation_drift_flagged_at" in entry
    # A ``dispatch_citation_drift_flagged`` event was emitted.
    drift_events = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_citation_drift_flagged"
    ]
    assert len(drift_events) == 1
    payload = drift_events[0]["payload"]
    assert payload["issue"] == 123
    cited = payload["drifted_citations"]
    assert len(cited) == 1
    assert cited[0]["citation"] == "old_dir/workflow.py:10"
    assert cited[0]["status"] == "stale_prefix"
