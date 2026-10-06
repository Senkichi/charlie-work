"""Issue #2441: ``mergequeue_stalled`` alarm and the missing-``.aviator/config.yml`` warning."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _merge_ready_fixtures import _mergequeue_automerge

from charlie_work import global_config, mergequeue_stall
from charlie_work.config import OrchestratorConfig
from charlie_work.global_config import load_layered_config, warn_if_aviator_config_missing
from charlie_work.instrumentation import query_events
from charlie_work.mergequeue_stall import STALL_AFTER, stalled_since
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

# ci_runners #169 entered the queue at ~04:42Z on 2026-10-06 and never merged.
QUEUED_AT = datetime(2026, 10, 6, 4, 42, 30, tzinfo=UTC)
ENTRY = {
    "number": 169,
    "issue_number": 7,
    "status": "mergequeue",
    "mergequeue_since": QUEUED_AT.isoformat(),
    "mergequeue_head_sha": "sha-169",
}


def _since(now: datetime, **over: Any) -> datetime | None:
    kwargs: dict[str, Any] = {
        "mergequeue_label": "mergequeue",
        "pr_labels": {"mergequeue"},
        "pr_entry": ENTRY,
        "now": now,
    }
    kwargs.update(over)
    return stalled_since(**kwargs)


def test_alarm_would_have_fired_for_ci_runners_169_at_0643z() -> None:
    # Queued 04:42:30Z + 2h = 06:42:30Z: silent just before, fires by 06:43Z.
    assert _since(datetime(2026, 10, 6, 6, 42, 0, tzinfo=UTC)) is None
    assert _since(datetime(2026, 10, 6, 6, 43, 0, tzinfo=UTC)) == QUEUED_AT


@pytest.mark.parametrize(
    "over",
    [
        {"mergequeue_label": None},
        {"pr_labels": set()},  # label gone: merged, reverted or dequeued
        {"pr_entry": {**ENTRY, "status": "pr_open"}},
        {"pr_entry": {**ENTRY, "mergequeue_since": None}},
        {"pr_entry": {**ENTRY, "mergequeue_since": "garbage"}},
        {"pr_entry": {**ENTRY, "mergequeue_stalled_since": ENTRY["mergequeue_since"]}},
    ],
)
def test_no_alarm_when_condition_absent_or_already_alarmed(over: dict[str, Any]) -> None:
    assert _since(QUEUED_AT + STALL_AFTER + timedelta(hours=5), **over) is None


def _app(tmp_path: Path, **cfg: Any) -> tuple[OrchestratorApp, Path]:
    config = OrchestratorConfig(auto_merge=_mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state = load_state(paths.state_file)
    state["prs"]["169"] = dict(ENTRY)
    save_state(paths.state_file, state)
    return app, paths.state_file


def _set_now(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    monkeypatch.setattr(
        mergequeue_stall,
        "_host_current",
        lambda: SimpleNamespace(clock=SimpleNamespace(now=lambda: now)),
    )


def _pr(**over: Any) -> dict[str, Any]:
    return {"number": 169, "labels": [{"name": "mergequeue"}], **over}


def test_emits_once_per_episode_and_again_for_a_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state_file = _app(tmp_path)
    _set_now(monkeypatch, datetime(2026, 10, 6, 6, 43, tzinfo=UTC))

    assert mergequeue_stall.alarm_if_stalled(app, _pr(), load_state(state_file)["prs"]["169"])
    events = query_events(state_file, kind="mergequeue_stalled")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 169
    assert payload["issue_number"] == 7
    assert payload["queued_since"] == QUEUED_AT.isoformat()
    assert payload["threshold_hours"] == 2.0
    assert load_state(state_file)["prs"]["169"]["mergequeue_stalled_since"] == (
        QUEUED_AT.isoformat()
    )

    # Later passes of the same episode stay silent.
    _set_now(monkeypatch, datetime(2026, 10, 6, 9, 0, tzinfo=UTC))
    assert not mergequeue_stall.alarm_if_stalled(app, _pr(), load_state(state_file)["prs"]["169"])
    assert len(query_events(state_file, kind="mergequeue_stalled")) == 1

    # A new episode (head moved -> since re-stamped) alarms again once overdue.
    state = load_state(state_file)
    state["prs"]["169"]["mergequeue_since"] = datetime(2026, 10, 6, 9, 1, tzinfo=UTC).isoformat()
    save_state(state_file, state)
    _set_now(monkeypatch, datetime(2026, 10, 6, 11, 2, tzinfo=UTC))
    assert mergequeue_stall.alarm_if_stalled(app, _pr(), load_state(state_file)["prs"]["169"])
    assert len(query_events(state_file, kind="mergequeue_stalled")) == 2


def test_no_event_without_the_label_or_under_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state_file = _app(tmp_path)
    _set_now(monkeypatch, datetime(2026, 10, 7, 0, 0, tzinfo=UTC))
    entry = load_state(state_file)["prs"]["169"]

    assert not mergequeue_stall.alarm_if_stalled(app, _pr(labels=[]), entry)
    app.dry_run = True
    assert not mergequeue_stall.alarm_if_stalled(app, _pr(), entry)
    assert query_events(state_file, kind="mergequeue_stalled") == []


# --- config-load warning ----------------------------------------------------


def _repo(tmp_path: Path, *, label: str | None, aviator: bool) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    text = (
        f"auto_merge:\n  mergequeue_label: {label}\n"
        if label
        else "auto_merge:\n  enabled: true\n"
    )
    (repo / "orchestrator.config.yaml").write_text(text, encoding="utf-8")
    if aviator:
        (repo / ".aviator").mkdir()
        (repo / ".aviator" / "config.yml").write_text("merge_rules: {}\n", encoding="utf-8")
    return repo


@pytest.fixture(autouse=True)
def _fresh_warn_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config, "_AVIATOR_WARNED", set())


def _load(repo: Path, tmp_path: Path) -> OrchestratorConfig:
    return load_layered_config(repo, fleet_dir_override=str(tmp_path / "fleet"))


def test_load_warns_when_mergequeue_label_set_without_aviator_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _repo(tmp_path, label="mergequeue", aviator=False)

    with caplog.at_level(logging.WARNING, logger="charlie_work.global_config"):
        config = _load(repo, tmp_path)
        _load(repo, tmp_path)  # second load: warned once per process

    assert config.auto_merge.mergequeue_label == "mergequeue"
    hits = [r for r in caplog.records if ".aviator/config.yml" in r.getMessage()]
    assert len(hits) == 1
    assert "'mergequeue'" in hits[0].getMessage()


@pytest.mark.parametrize(("label", "aviator"), [("mergequeue", True), (None, False), (None, True)])
def test_load_is_quiet_when_aviator_config_present_or_label_unset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, label: str | None, aviator: bool
) -> None:
    repo = _repo(tmp_path, label=label, aviator=aviator)

    with caplog.at_level(logging.WARNING, logger="charlie_work.global_config"):
        config = _load(repo, tmp_path)

    assert not [r for r in caplog.records if ".aviator/config.yml" in r.getMessage()]
    assert warn_if_aviator_config_missing(config, repo) is False
