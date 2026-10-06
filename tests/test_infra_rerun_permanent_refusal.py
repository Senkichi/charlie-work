"""Permanent ``gh run rerun`` refusals: one attempt, one escalation (#2445).

GitHub answers a rerun of a run created over a month ago with HTTP 403 forever
(live: ci_runners PR 73). The driver used to re-request it every pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github_rerun import FakeGitHubWithRerunCapture
from _review_fixtures import _required_checks_config
from charlie_work.infra_rerun_refusal import is_permanent_rerun_refusal
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

_RUN_LINK = "https://github.com/owner/repo/actions/runs/12345/job/67890"
_CHECKS = [
    {"name": "Tests passed", "state": "CANCELLED", "link": _RUN_LINK},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]
_REFUSAL_403 = (
    "failed to rerun: HTTP 403: Unable to rerun the workflow run: "
    "the run was created over a month ago (https://api.github.com/repos/o/r/actions/runs/12345/rerun)"
)


def _app(tmp_path: Path, fake_gh: FakeGitHubWithRerunCapture):
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )
    return app, config, paths


def _events(paths, kind: str) -> list[dict]:
    return [e for e in load_state(paths.state_file).get("events", []) if e["kind"] == kind]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_REFUSAL_403, True),
        ("HTTP 403: CREATED OVER A MONTH AGO", True),
        ("run 1 cannot be rerun; This workflow is already running", False),
        ("HTTP 502: Bad Gateway", False),
        ("HTTP 403: API rate limit exceeded", False),
        ("", False),
    ],
)
def test_classifier(error: str, expected: bool) -> None:
    assert is_permanent_rerun_refusal(error) is expected


def test_403_one_attempt_one_escalation(tmp_path: Path) -> None:
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CHECKS), rerun_ok=False, rerun_error=_REFUSAL_403
    )
    app, config, paths = _app(tmp_path, fake_gh)

    first = app.merge_ready(456)
    assert first.data.get("infra_escalated") is True
    for _ in range(3):
        app.merge_ready(456)

    assert fake_gh.rerun_calls == [["run", "rerun", "12345"]]
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["infra_rerun_refused"] == {"sha-abc123": [12345]}
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "infra_rerun_refused"
    assert state["issues"]["123"]["reason_class"] == "mechanical"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    escalated = _events(paths, "infra_rerun_escalated")
    assert len(escalated) == 1
    assert escalated[0]["payload"]["refused_run_ids"] == [12345]
    assert len(_events(paths, "infra_rerun_failed")) == 1


def test_refused_run_skipped_even_without_escalation_record(tmp_path: Path) -> None:
    """The per-run refusal record alone stops re-requests (state, not escalation)."""
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CHECKS), rerun_ok=False, rerun_error=_REFUSAL_403
    )
    app, _, paths = _app(tmp_path, fake_gh)
    app.merge_ready(456)
    state = load_state(paths.state_file)
    state["issues"]["123"].pop("escalation_reasons_seen", None)
    state["prs"]["456"].pop("escalation_reasons_seen", None)
    save_state(paths.state_file, state)
    app.merge_ready(456)
    assert len(fake_gh.rerun_calls) == 1


def test_transient_error_keeps_retrying_without_escalation(tmp_path: Path) -> None:
    fake_gh = FakeGitHubWithRerunCapture(
        checks=list(_CHECKS), rerun_ok=False, rerun_error="HTTP 502: Bad Gateway"
    )
    app, config, paths = _app(tmp_path, fake_gh)

    for _ in range(3):
        app.merge_ready(456)

    assert len(fake_gh.rerun_calls) == 3
    state = load_state(paths.state_file)
    assert "infra_rerun_refused" not in state["prs"].get("456", {})
    assert not _events(paths, "infra_rerun_escalated")
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
