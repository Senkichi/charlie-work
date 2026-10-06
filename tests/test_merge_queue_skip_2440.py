"""Issue #2440: a PR already handed to the merge queue is not fully re-checked each pass.

The safety property under test: a hold label added to the PR or to its linked
issue is honoured on the very next pass -- the skip never outlives a change.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _merge_ready_fixtures import _mergequeue_automerge

from charlie_work.config import OrchestratorConfig
from charlie_work.merge_path import queue_skip
from charlie_work.merge_path.model import PersistedPr
from charlie_work.merge_path.queue_skip import (
    RECHECK_INTERVAL,
    QueueSkipFacts,
    decide,
    merge_ready_unless_queued,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
HOLD = "agent:merge-hold"


def _persisted(**over: Any) -> PersistedPr:
    fields: dict[str, Any] = {
        "status": "mergequeue",
        "mergequeue_head_sha": "sha-1",
        "mergequeue_checked_at": (NOW - timedelta(minutes=5)).isoformat(),
    }
    fields.update(over)
    return PersistedPr(**fields)


def _facts(**over: Any) -> QueueSkipFacts:
    base = QueueSkipFacts(
        mergequeue_label="mergequeue",
        hold_labels=frozenset({HOLD}),
        priority_prefix="priority:",
        pr_labels=frozenset({"mergequeue"}),
        pr_head_sha="sha-1",
        persisted=_persisted(),
        issue_bound=True,
        issue_labels=frozenset({"automated-ready"}),
        now=NOW,
    )
    return replace(base, **over)


def test_decide_skips_unchanged_queued_pr() -> None:
    assert decide(_facts()) is True


@pytest.mark.parametrize(
    "over",
    [
        {"pr_labels": frozenset()},  # label removed (Aviator reverted / human)
        {"pr_labels": frozenset({"mergequeue", HOLD})},  # hold on the PR
        {"issue_labels": frozenset({HOLD})},  # hold on the issue
        {"issue_labels": frozenset({"priority:critical"})},  # late critical
        {"issue_labels": None},  # issue not in the cached list: unproven
        {"pr_head_sha": "sha-2"},  # head moved since hand-off
        {"pr_head_sha": None},
        {"persisted": _persisted(mergequeue_checked_at=None)},  # never checked
        {"persisted": _persisted(status="pr_open")},
        {"persisted": _persisted(mergequeue_revoked_reason="x")},
        {"mergequeue_label": None},
    ],
)
def test_decide_rechecks_on_any_change(over: dict[str, Any]) -> None:
    assert decide(_facts(**over)) is False


def test_decide_forces_recheck_after_interval() -> None:
    stale = _persisted(mergequeue_checked_at=(NOW - RECHECK_INTERVAL).isoformat())
    assert decide(_facts(persisted=stale)) is False
    # A checked_at in the future (clock skew) is not trusted either.
    future = _persisted(mergequeue_checked_at=(NOW + timedelta(minutes=1)).isoformat())
    assert decide(_facts(persisted=future)) is False


def test_decide_unbound_pr_needs_no_issue_labels() -> None:
    assert decide(_facts(issue_bound=False, issue_labels=None)) is True


class _CountingGitHub(FakeGitHub):
    """Records every GitHub read/write ``merge_ready`` would make."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.pr_labels_posted: list[str] = []

    def pr_view(self, number: int, *, fields: str | None = None):  # type: ignore[override]
        self.calls.append("pr_view")
        return super().pr_view(number, fields=fields)

    def pr_checks(self, number: int):  # type: ignore[override]
        self.calls.append("pr_checks")
        return super().pr_checks(number)

    def issue_view(self, number: int):  # type: ignore[override]
        self.calls.append("issue_view")
        return super().issue_view(number)

    def add_pr_label(self, number: int, label: str) -> bool:  # type: ignore[override]
        self.calls.append("add_pr_label")
        self.pr_labels_posted.append(label)
        return super().add_pr_label(number, label)


def _queued_app(tmp_path: Path) -> tuple[OrchestratorApp, _CountingGitHub, dict[str, Any]]:
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _CountingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=True)
    assert result.data["mergequeue_label_applied"] is True
    state = load_state(paths.state_file)["prs"]["456"]
    assert state["status"] == "mergequeue"
    assert state["mergequeue_head_sha"] == "sha-abc123"  # the hand-off SHA
    assert state["mergequeue_checked_at"]
    gh.calls.clear()
    gh.pr_labels_posted.clear()
    # The per-pass PR snapshot (pr_list row) after Aviator's label landed.
    snapshot = {**gh.prs[0], "labels": [{"name": "mergequeue"}]}
    return app, gh, snapshot


def _run(app: OrchestratorApp, snapshot: dict[str, Any]) -> bool:
    state = load_state(app.paths.state_file)["prs"]["456"]
    return merge_ready_unless_queued(
        app, snapshot, state, 123, merge=True, merge_train_head=None, errors=[], merges=[]
    )


def test_queued_pr_with_unchanged_head_costs_zero_github_calls(tmp_path: Path) -> None:
    app, gh, snapshot = _queued_app(tmp_path)

    assert _run(app, snapshot) is True
    assert gh.calls == []


def test_issue_hold_added_after_hand_off_is_honoured_next_pass(tmp_path: Path) -> None:
    app, gh, snapshot = _queued_app(tmp_path)
    assert _run(app, snapshot) is True

    gh.issues[0]["labels"].append({"name": HOLD})

    assert _run(app, snapshot) is False
    assert "pr_view" in gh.calls  # the full merge_ready ran


def test_pr_hold_added_after_hand_off_is_honoured_next_pass(tmp_path: Path) -> None:
    app, gh, snapshot = _queued_app(tmp_path)
    held = {**snapshot, "labels": [*snapshot["labels"], {"name": HOLD}]}

    assert _run(app, held) is False
    assert "pr_view" in gh.calls


def test_head_change_and_interval_expiry_force_full_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, gh, snapshot = _queued_app(tmp_path)

    assert _run(app, {**snapshot, "headRefOid": "sha-new"}) is False

    gh.calls.clear()
    later = datetime.now(UTC) + RECHECK_INTERVAL + timedelta(minutes=1)
    monkeypatch.setattr(
        queue_skip,
        "_host_current",
        lambda: SimpleNamespace(clock=SimpleNamespace(now=lambda: later)),
    )
    assert _run(app, snapshot) is False
    assert "pr_view" in gh.calls


def test_full_check_of_already_labelled_pr_does_not_repost_label(tmp_path: Path) -> None:
    app, gh, _ = _queued_app(tmp_path)
    gh.prs[0]["labels"] = [{"name": "mergequeue"}]

    result = app.merge_ready(456, merge=True)

    assert result.data["mergequeue_label_applied"] is True
    assert "add_pr_label" not in gh.calls
    # ...and the full check re-stamps the freshness clock for the next skip window.
    assert load_state(app.paths.state_file)["prs"]["456"]["mergequeue_checked_at"]


def test_full_check_reads_issue_labels_from_cached_list_not_issue_view(tmp_path: Path) -> None:
    app, gh, _ = _queued_app(tmp_path)
    gh.prs[0]["labels"] = []  # label absent: the hand-off path runs and needs the hold read

    app.merge_ready(456, merge=True)

    assert "issue_view" not in gh.calls


def test_late_critical_issue_gets_skip_line_label_without_reposting_queue_label(
    tmp_path: Path,
) -> None:
    app, gh, _ = _queued_app(tmp_path)
    gh.prs[0]["labels"] = [{"name": "mergequeue"}]  # already queued
    gh.issues[0]["labels"].append({"name": "priority:critical"})  # promoted after hand-off

    app.merge_ready(456, merge=True)

    skip_line = app.config.auto_merge.mergequeue_skip_line_label
    assert gh.pr_labels_posted == [skip_line]  # skip-line added, mergequeue not re-POSTed


def test_skip_line_label_already_on_pr_is_not_re_added(tmp_path: Path) -> None:
    app, gh, _ = _queued_app(tmp_path)
    skip_line = app.config.auto_merge.mergequeue_skip_line_label
    gh.prs[0]["labels"] = [{"name": "mergequeue"}, {"name": skip_line}]
    gh.issues[0]["labels"].append({"name": "priority:critical"})

    app.merge_ready(456, merge=True)

    assert gh.pr_labels_posted == []
