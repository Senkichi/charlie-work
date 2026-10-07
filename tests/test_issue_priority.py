"""TIS-CW-7: ``priority:<level>`` orders fresh dispatch and puts critical PRs at the queue front."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _merge_ready_fixtures import _mergequeue_automerge
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    AutoMergeConfig,
    DispatchConfig,
    LabelConfig,
    OrchestratorConfig,
    load_config,
)
from charlie_work.host.fakes import FakeWorkerLauncher
from charlie_work.issue_priority import is_critical, order_by_priority, priority_level
from charlie_work.merge_path.apply_merge import _add_skip_line_if_critical
from charlie_work.merge_path.model import MergePathConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

P = "priority:"


def _issue(number: int, created: str, *labels: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "work",
        "labels": [{"name": "automated-ready"}] + [{"name": label} for label in labels],
        "createdAt": created,
        "state": "OPEN",
    }


# Oldest-first base order: 100, 300, 200, 400, 500.
BACKLOG = [
    _issue(100, "2026-07-01T00:00:00Z"),
    _issue(300, "2026-07-02T00:00:00Z", "priority:low"),
    _issue(200, "2026-07-03T00:00:00Z", "priority:critical"),
    _issue(400, "2026-07-04T00:00:00Z", "priority:high"),
    _issue(500, "2026-07-05T00:00:00Z", "priority:urgent"),
]


# --- the label vocabulary -----------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([], "normal"),
        (["priority:critical"], "critical"),
        (["Priority: High "], "high"),
        (["priority:low", "priority:critical"], "critical"),
        (["priority:urgent"], "normal"),
        (["priority:urgent", "priority:low"], "low"),
        (["model:opus"], "normal"),
    ],
)
def test_priority_level(labels: list[str], expected: str) -> None:
    assert priority_level(labels, P) == expected


def test_order_is_critical_first_and_stable_within_a_level() -> None:
    ordered = order_by_priority(BACKLOG, P)
    assert [issue["number"] for issue in ordered] == [200, 400, 100, 500, 300]


def test_an_empty_prefix_turns_priority_off() -> None:
    assert priority_level(["priority:critical"], "") == "normal"
    assert is_critical(["priority:critical"], "") is False
    assert [i["number"] for i in order_by_priority(BACKLOG, "")] == [100, 300, 200, 400, 500]


# --- fresh dispatch claims in priority order ----------------------------------


def _dispatch_app(
    tmp_path: Path, *, dry_run: bool = True, order: str = "oldest", prefix: str = P
) -> OrchestratorApp:
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            default_limit=5,
            order=order,
            # Host-load brake reads live pytest processes; keep it out of ordering tests.
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=0,
        ),
        labels=LabelConfig(priority_prefix=prefix),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = copy.deepcopy(BACKLOG)
    fake_gh.prs[0]["state"] = "CLOSED"
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)


def _dry_claims(app: OrchestratorApp, limit: int = 5) -> list[int]:
    result = app.dispatch(limit=limit)
    assert result.ok, result.message
    return [session["issue_number"] for session in result.data["sessions"]]


@pytest.mark.parametrize(
    ("order", "prefix", "expected"),
    [
        ("oldest", P, [200, 400, 100, 500, 300]),
        ("newest", P, [200, 400, 500, 100, 300]),
        ("oldest", "", [100, 300, 200, 400, 500]),
    ],
)
def test_dry_run_claims_in_priority_order(
    tmp_path: Path, order: str, prefix: str, expected: list[int]
) -> None:
    assert _dry_claims(_dispatch_app(tmp_path, order=order, prefix=prefix)) == expected


def test_a_tight_limit_claims_the_critical_issue_first(tmp_path: Path) -> None:
    assert _dry_claims(_dispatch_app(tmp_path), limit=1) == [200]


def test_the_real_pass_launches_in_priority_order(tmp_path: Path, fake_host) -> None:
    launched: list[int] = []

    def _fake(_repo_root, _manifest, _results, settings, requests):
        launched.extend(request.issue_number for request in requests)
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter=settings.adapter,
                ok=True,
            )
            for request in requests
        ]

    fake_host(worker_launch=FakeWorkerLauncher([_fake]))
    app = _dispatch_app(tmp_path, dry_run=False)
    result = app.dispatch(limit=2)
    assert launched == [200, 400], result.message


# --- the Aviator hand-off -----------------------------------------------------


def _handoff_app(
    tmp_path: Path, *issue_labels: str, auto_merge: AutoMergeConfig | None = None
) -> tuple[OrchestratorApp, FakeGitHub]:
    config = OrchestratorConfig(auto_merge=auto_merge or _mergequeue_automerge())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": "automated-ready"}] + [
        {"name": label} for label in issue_labels
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    return app, fake_gh


def test_a_critical_issue_pr_gets_skip_line_before_the_queue_label(tmp_path: Path) -> None:
    app, fake_gh = _handoff_app(tmp_path, "priority:critical")
    result = app.merge_ready(456, merge=True)
    assert result.data["mergequeue_label_applied"] is True
    assert fake_gh.pr_labels_added == [(456, "mergequeue-skip-line"), (456, "mergequeue")]


def test_a_non_critical_issue_pr_queues_in_normal_order(tmp_path: Path) -> None:
    app, fake_gh = _handoff_app(tmp_path, "priority:high")
    app.merge_ready(456, merge=True)
    assert fake_gh.pr_labels_added == [(456, "mergequeue")]


def test_a_null_skip_line_label_is_the_kill_switch(tmp_path: Path) -> None:
    auto_merge = replace(_mergequeue_automerge(), mergequeue_skip_line_label=None)
    app, fake_gh = _handoff_app(tmp_path, "priority:critical", auto_merge=auto_merge)
    app.merge_ready(456, merge=True)
    assert fake_gh.pr_labels_added == [(456, "mergequeue")]


def test_self_merge_mode_adds_no_skip_line(tmp_path: Path) -> None:
    auto_merge = AutoMergeConfig(required_checks=(), require_approved_review=True)
    app, fake_gh = _handoff_app(tmp_path, "priority:critical", auto_merge=auto_merge)
    result = app.merge_ready(456, merge=True)
    assert result.data["can_merge"] is True
    assert fake_gh.pr_labels_added == []


def test_an_unreadable_issue_queues_without_skip_line() -> None:
    added: list[tuple[int, str]] = []

    def _unreadable(_number: int) -> dict[str, Any]:
        raise ValueError("issue gone")

    def _add(number: int, label: str) -> bool:
        added.append((number, label))
        return True

    app = SimpleNamespace(
        gh=SimpleNamespace(
            issue_view=_unreadable,
            issue_list=lambda state=None: [],  # cache miss -> live read
            add_pr_label=_add,
        )
    )
    cfg = MergePathConfig(
        mergequeue_label="mergequeue", skip_line_label="mergequeue-skip-line", priority_prefix=P
    )
    assert _add_skip_line_if_critical(app, 456, 123, cfg) is False
    assert added == []


def test_config_defaults_on_with_null_as_kill_switch(tmp_path: Path) -> None:
    assert AutoMergeConfig().mergequeue_skip_line_label == "mergequeue-skip-line"
    assert LabelConfig().priority_prefix == "priority:"
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text(
        'auto_merge:\n  mergequeue_skip_line_label: null\nlabels:\n  priority_prefix: ""\n',
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.auto_merge.mergequeue_skip_line_label is None
    assert config.labels.priority_prefix == ""
    config_file.write_text(
        "auto_merge:\n  mergequeue_skip_line_label: '  skip  '\n", encoding="utf-8"
    )
    assert load_config(config_file).auto_merge.mergequeue_skip_line_label == "skip"
