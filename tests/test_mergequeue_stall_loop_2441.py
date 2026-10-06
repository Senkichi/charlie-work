"""Issue #2441: the stall alarm is wired into ``reap_loop._loop_body`` and is best-effort."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _merge_ready_fixtures import _mergequeue_automerge

from charlie_work import mergequeue_stall
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

QUEUED_FOR = timedelta(hours=3)


def _app(tmp_path: Path) -> tuple[OrchestratorApp, FakeGitHub]:
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)
    # PR 456: already handed to the queue three hours ago, still labelled, never merged.
    gh.prs[0]["labels"] = [{"name": "mergequeue"}]
    state = load_state(paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "issue_number": 123,
        "status": "mergequeue",
        "decision": "approved",
        "reviewed_head_sha": "sha-abc123",
        "mergequeue_since": (datetime.now(UTC) - QUEUED_FOR).isoformat(),
        "mergequeue_head_sha": "sha-abc123",
    }
    save_state(paths.state_file, state)
    return app, gh


def _add_second_approved_pr(app: OrchestratorApp, gh: FakeGitHub) -> None:
    """PR 457 sits AFTER the stalled PR in the scan and is ready for the queue hand-off."""
    gh.prs.append(
        {
            **gh.prs[0],
            "number": 457,
            "headRefName": "agent/issue-124-other",
            "headRefOid": "sha-457",
            "body": "Closes #124\n\nTests: covered.",
            "labels": [],
        }
    )
    gh.issues.append({**gh.issues[0], "number": 124, "title": "Other", "labels": []})
    app.record_review(457, "approved", summary="ok", verdict_provenance="fresh_llm_review")


def test_loop_pass_emits_mergequeue_stalled_for_overdue_queued_pr(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)

    app.loop(limit=0)

    events = query_events(app.paths.state_file, kind="mergequeue_stalled")
    assert [e["payload"]["pr_number"] for e in events] == [456]
    assert load_state(app.paths.state_file)["prs"]["456"]["mergequeue_stalled_since"]

    app.loop(limit=0)  # the same episode stays silent on later passes
    assert len(query_events(app.paths.state_file, kind="mergequeue_stalled")) == 1


def test_failing_alarm_does_not_abort_the_pass_or_skip_later_prs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, gh = _app(tmp_path)
    _add_second_approved_pr(app, gh)

    def boom(*_a: Any, **_k: Any) -> bool:
        raise TimeoutError("state lock busy")

    monkeypatch.setattr(mergequeue_stall, "alarm_if_stalled", boom)

    result = app.loop(limit=0)

    assert result.data["errors"] == []  # the failure never reaches the per-PR error bucket
    # PR 457, scanned after the failing alarm, still got its merge-queue hand-off.
    assert (457, "mergequeue") in gh.pr_labels_added
    failures = query_events(app.paths.state_file, kind="mergequeue_stall_alarm_failed")
    # The alarm ran (and failed) for BOTH PRs: the scan went on past the first failure.
    assert sorted(e["payload"]["pr_number"] for e in failures) == [456, 457]
    assert "state lock busy" in failures[0]["payload"]["error"]
    # The row really is a ``mergequeue_stall_alarm_failed`` event (kind literal in the kind
    # slot, payload in the payload slot), not one filed under a swapped argument.
    every_kind = {e["kind"] for e in query_events(app.paths.state_file)}
    assert "mergequeue_stall_alarm_failed" in every_kind
    assert all(isinstance(e["payload"].get("pr_number"), int) for e in failures)
    assert not [k for k in every_kind if k.startswith("{") or "pr_number" in k]
    assert query_events(app.paths.state_file, kind="mergequeue_stalled") == []
