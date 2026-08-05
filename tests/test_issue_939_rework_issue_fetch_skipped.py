"""Tests for issue #939: a failed ``gh.issue_view`` in ``dispatch_rework`` must not be silent.

In the rework-dispatch candidate scan, an issue in ``rework_requested`` with an
open PR is dropped silently when ``gh.issue_view`` raises ``GitHubError``. The
issue stays ``rework_requested`` and is retried on the next pass, but a later
escalation (e.g. ``rework_stall_minutes``) leaves no record of *why* rework was
never dispatched. These tests pin the ``rework_issue_fetch_skipped`` warning
event.
"""

from __future__ import annotations

import sys
from pathlib import Path

from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.instrumentation import _classify_level
from charlie_work.paths import runtime_paths
from charlie_work.state import StateLockBusy, load_state, save_state, state_lock
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


def test_rework_issue_fetch_skipped_lock_busy_does_not_defer_healthy_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    """StateLockBusy from the diagnostic write must not defer the whole pass.

    Regression test for the fix: the diagnostic write's ``try/except (OSError,
    ValueError)`` did not cover ``StateLockBusy`` (a ``RuntimeError``
    subclass), so lock contention on that best-effort audit write propagated
    up to ``dispatch_rework``'s own ``except StateLockBusy``, which defers the
    ENTIRE call with ``selected_count=0`` -- discarding issue 123, a healthy
    rework candidate that had already been found and was ready to dispatch,
    purely because of contention writing an event about a *different*,
    unrelated issue (999) whose ``gh.issue_view`` failed.
    """
    import charlie_work.workflow as workflow_module

    class ReworkGitHubMixedFetch(FakeGitHub):
        """``issue_view`` succeeds for 123 (healthy) and raises for 999 (failing)."""

        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]
            self.issues.append(
                {
                    "number": 999,
                    "title": "Other broken issue",
                    "url": "https://example.test/issues/999",
                    "body": "Also broken",
                    "labels": [{"name": "agent:needs-rework"}],
                    "state": "OPEN",
                }
            )
            self.prs.append(
                {
                    "number": 457,
                    "title": "Fix #999: other",
                    "url": "https://example.test/pull/457",
                    "headRefName": "agent/issue-999-other",
                    "baseRefName": "main",
                    "headRefOid": "sha-def999",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #999\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            )

        def issue_view(self, number: int):
            if number == 999:
                raise GitHubError("gh: transient outage")
            return super().issue_view(number)

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ensure()
    _seed_rework_requested_issue(paths, 123)
    _seed_rework_requested_issue(paths, 999)

    fake_gh = ReworkGitHubMixedFetch()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    # --- Target ONLY the diagnostic write's state_lock call ---
    # ``_build_rework_issue_fetch_skip_payload`` is called exactly once per
    # pass, immediately before the diagnostic write's own ``with
    # state_lock(...)``, with nothing else in between but a log call. Arm a
    # flag there and consume it on the very next ``state_lock`` call, so the
    # simulated contention lands on that specific ``with`` block -- not the
    # earlier state-scan lock, not the later throttle-check or dispatch-claim
    # locks that also use ``state_lock`` in the same pass.
    real_build_payload = workflow_module._build_rework_issue_fetch_skip_payload
    real_state_lock = workflow_module.state_lock
    armed = {"value": False}
    payload_built = {"value": False}

    def fake_build_payload(*args, **kwargs):
        result = real_build_payload(*args, **kwargs)
        armed["value"] = True
        payload_built["value"] = True
        return result

    def fake_state_lock(*args, **kwargs):
        if armed["value"]:
            armed["value"] = False
            raise StateLockBusy("state lock held (simulated contention)")
        return real_state_lock(*args, **kwargs)

    monkeypatch.setattr(
        workflow_module, "_build_rework_issue_fetch_skip_payload", fake_build_payload
    )
    monkeypatch.setattr(workflow_module, "state_lock", fake_state_lock)

    result = app.dispatch_rework()

    # Control: prove the fake issue_view actually raised for 999 and the
    # diagnostic payload was actually built -- otherwise this test would pass
    # even if the patch never engaged (e.g. if failed_issue_fetches were
    # empty, or if the code path bypassed the diagnostic write entirely).
    assert payload_built["value"] is True
    assert armed["value"] is False, "the armed StateLockBusy was never consumed"

    # The regression: contention on the diagnostic write must not be mistaken
    # for whole-call state-lock-busy deferral. Without the fix, this whole
    # block would fail: dispatch_rework's except StateLockBusy would catch the
    # re-raised StateLockBusy and return exactly this deferred shape instead.
    assert result.data.get("state_lock_busy") is not True
    assert result.data.get("deferred_reason") != "state_lock_busy"
    assert "state lock held" not in result.message

    # Issue 123 -- found, healthy, and ready before the diagnostic write ever
    # ran -- must still be dispatched.
    assert result.ok is True
    assert result.data["selected_count"] == 1
    dispatched_numbers = {s["issue_number"] for s in result.data["sessions"]}
    assert dispatched_numbers == {123}

    # The diagnostic write itself failed, so no event should have landed --
    # that is the whole point of the scenario, not a bug to additionally fix.
    state_after = load_state(paths.state_file)
    events = [e for e in state_after["events"] if e["kind"] == "rework_issue_fetch_skipped"]
    assert events == []
