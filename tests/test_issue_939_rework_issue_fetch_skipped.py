"""Tests for issue #939: a failed ``gh.issue_view`` in ``dispatch_rework`` must not be silent.

In the rework-dispatch candidate scan, an issue in ``rework_requested`` with an
open PR is dropped silently when ``gh.issue_view`` raises ``GitHubError``. The
issue stays ``rework_requested`` and is retried on the next pass, but a later
escalation (e.g. ``rework_stall_minutes``) leaves no record of *why* rework was
never dispatched. These tests pin the ``rework_issue_fetch_skipped`` warning
event.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.instrumentation import _classify_level
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import (
    OrchestratorApp,
    _build_rework_issue_fetch_skip_payload,
)

from test_charlie_work import FakeGitHub


def _make_app(tmp_path: Path, fake_gh: FakeGitHub, **kwargs) -> tuple[OrchestratorApp, object]:
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                "python",
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, **kwargs)
    return app, paths


class FakeGitHubIssueViewFailing(FakeGitHub):
    """A gh whose ``issue_view`` always raises, e.g. a transient outage."""

    def __init__(self, message: str = "gh unavailable") -> None:
        super().__init__()
        self._message = message

    def issue_view(self, number: int):
        raise GitHubError(self._message)


def _seed_rework_requested_issue(paths, number: int = 123) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(number)] = {
            "number": number,
            "title": f"Fix {number}",
            "url": f"https://example.test/issues/{number}",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)


def test_rework_issue_fetch_skipped_records_warning_event(tmp_path: Path) -> None:
    """A raising ``issue_view`` in the rework scan must leave an audit trail."""
    paths = runtime_paths(tmp_path, OrchestratorConfig().runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ensure()
    fake_gh = FakeGitHubIssueViewFailing("gh: command not found")
    _seed_rework_requested_issue(paths, 123)
    app, _ = _make_app(tmp_path, fake_gh)

    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 0
    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "rework_issue_fetch_skipped"]
    assert len(events) == 1, (
        f"expected exactly one rework_issue_fetch_skipped event, got {len(events)}"
    )
    payload = events[0]["payload"]
    assert payload["issue_numbers"] == [123]
    assert payload["reason"] == "gh: command not found"
    assert payload["error_type"] == "GitHubError"
    assert payload["issue_numbers_truncated"] == 0


def test_rework_issue_fetch_skipped_fires_once_per_pass_not_deduped(tmp_path: Path) -> None:
    """Each pass is a distinct occurrence; do not collapse across passes."""
    paths = runtime_paths(tmp_path, OrchestratorConfig().runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ensure()
    fake_gh = FakeGitHubIssueViewFailing("gh: command not found")
    _seed_rework_requested_issue(paths, 123)
    app, _ = _make_app(tmp_path, fake_gh)

    app.dispatch_rework()
    app.dispatch_rework()
    app.dispatch_rework()

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "rework_issue_fetch_skipped"]
    assert len(events) == 3, (
        f"expected 3 rework_issue_fetch_skipped events across 3 calls, got {len(events)}"
    )


def test_rework_issue_fetch_skip_payload_caps_issue_numbers() -> None:
    """The payload caps the issue list and reports how many were elided."""
    failures = [(n, GitHubError(f"err {n}")) for n in range(1, 26)]
    payload = _build_rework_issue_fetch_skip_payload(failures, max_issue_numbers=20)

    assert payload["issue_numbers"] == list(range(1, 21))
    assert payload["issue_numbers_truncated"] == 5
    assert payload["reason"] == "err 1"
    assert payload["error_type"] == "GitHubError"


def test_rework_issue_fetch_skip_payload_truncates_long_reason() -> None:
    """Long exception text is truncated so the payload does not explode."""
    long_message = "x" * 1000
    failures = [(1, GitHubError(long_message))]
    payload = _build_rework_issue_fetch_skip_payload(failures, reason_chars=300)

    assert len(payload["reason"]) == 300
    assert payload["reason"].endswith("...")
    assert payload["reason"] == "x" * 297 + "..."


def test_rework_issue_fetch_skipped_classified_as_warning() -> None:
    """A skipped rework issue fetch is a handled degradation, not an error."""
    assert _classify_level("rework_issue_fetch_skipped") == "warning"
