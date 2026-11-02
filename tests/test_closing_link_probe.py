"""Tests for ``closing_reference.probe_closing_link`` (cw#1868).

Split out of ``test_closing_reference.py`` (a frozen attachment point): the probe
settles GitHub's asynchronous closing-keyword indexing before a caller may log
``pr_closing_ref_unlinked``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.closing_reference import probe_closing_link
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import query_events


class _LaggingIndexGitHub:
    """``pr_view`` fake whose link list appears only from the Nth read on.

    Each entry of ``script`` is one read: a list of issue numbers, or an
    Exception instance to raise.
    """

    def __init__(self, script: list[list[int] | Exception]) -> None:
        self._script = script
        self.calls = 0

    def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
        step = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return {"closingIssuesReferences": [{"number": n} for n in step]}


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    import charlie_work.closing_reference as mod

    sleeps: list[float] = []
    monkeypatch.setattr(mod, "_default_sleep", sleeps.append)
    return sleeps


def test_probe_closing_link_matched_first_read_does_not_sleep(
    recorded_sleeps: list[float],
) -> None:
    gh = _LaggingIndexGitHub([[42]])
    assert probe_closing_link(gh, 7, 42, fields="closingIssuesReferences") == {42}
    assert gh.calls == 1
    assert recorded_sleeps == []


def test_probe_closing_link_recovers_when_index_catches_up(recorded_sleeps: list[float]) -> None:
    """The #1868 race: empty on the first read, linked on a later one."""
    gh = _LaggingIndexGitHub([[], [], [42]])
    assert probe_closing_link(gh, 7, 42, fields="closingIssuesReferences") == {42}
    assert gh.calls == 3
    assert recorded_sleeps == [3.0, 10.0]


def test_probe_closing_link_persistent_miss_returns_last_linked_set(
    recorded_sleeps: list[float],
) -> None:
    gh = _LaggingIndexGitHub([[], [], [999]])
    assert probe_closing_link(gh, 7, 42, fields="closingIssuesReferences") == {999}
    assert gh.calls == 3


def test_probe_closing_link_failed_final_probe_is_none(recorded_sleeps: list[float]) -> None:
    gh = _LaggingIndexGitHub([[], [], RuntimeError("boom")])
    assert probe_closing_link(gh, 7, 42, fields="closingIssuesReferences") is None


def test_probe_closing_link_early_raise_then_late_success(recorded_sleeps: list[float]) -> None:
    """A transient error on an early read does not poison a later linked read."""
    gh = _LaggingIndexGitHub([RuntimeError("boom"), [], [42]])
    assert probe_closing_link(gh, 7, 42, fields="closingIssuesReferences") == {42}
    assert gh.calls == 3


def test_open_salvage_pr_no_unlinked_event_when_index_lags(
    tmp_path: Path, recorded_sleeps: list[float]
) -> None:
    """End-to-end at the call site: a lagging read must not produce the warning."""
    from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels

    from charlie_work.workflow import _open_salvage_pr

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path, closing_issue_numbers=[42])
    reads = iter([{"closingIssuesReferences": []}, {"closingIssuesReferences": [{"number": 42}]}])
    gh.pr_view = lambda number, *, fields="": next(reads)  # type: ignore[method-assign]

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        source_description="worker branch",
        state_file=state_file,
    )

    assert (pr_number, error) == (101, None)
    assert recorded_sleeps == [3.0]
    assert query_events(state_file, kind="pr_closing_ref_unlinked", issue_number=42) == []
