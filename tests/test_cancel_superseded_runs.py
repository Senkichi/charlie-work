"""Tests for cancel_superseded_runs, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub


def test_cancel_superseded_runs_no_workflow_name(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs returns error when workflow_name is empty."""
    from charlie_work.github import cancel_superseded_runs

    fake_gh = FakeGitHub()
    result = cancel_superseded_runs(fake_gh, "main", "")
    assert result["errors"] == ["workflow_name is empty - cannot cancel runs"]
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0


def test_cancel_superseded_runs_no_queued_runs(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles no queued runs correctly."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithEmptyRuns(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = []

    fake_gh = FakeGitHubWithEmptyRuns()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_one_queued_run(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs keeps the single queued run."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithOneRun(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                }
            ]

    fake_gh = FakeGitHubWithOneRun()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 1
    assert result["kept"] == 1
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_multiple_queued_runs(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs keeps newest and cancels older runs."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithMultipleRuns(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "queued",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 125,
                    "status": "queued",
                    "createdAt": "2026-07-09T02:00:00Z",
                    "headBranch": "main",
                },
            ]
            self.cancelled_runs = []

        def run(self, args, *, json_output=False, allow_failure=False):
            if "cancel" in args:
                run_id = int(args[-1])
                self.cancelled_runs.append(run_id)
                return "Cancelled"
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithMultipleRuns()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 3
    assert result["kept"] == 1
    assert result["cancelled"] == 2
    assert result["cancelled_run_ids"] == [124, 123]  # Oldest two cancelled
    assert result["errors"] == []
    # Verify the newest (125) was kept, older ones cancelled
    assert 123 in fake_gh.cancelled_runs  # Oldest cancelled
    assert 124 in fake_gh.cancelled_runs  # Middle cancelled
    assert 125 not in fake_gh.cancelled_runs  # Newest kept


def test_cancel_superseded_runs_ignores_in_progress(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs ignores in_progress runs."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithInProgress(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "in_progress",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
            ]

    fake_gh = FakeGitHubWithInProgress()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 1  # Only queued runs counted
    assert result["kept"] == 1
    assert result["cancelled"] == 0
    assert result["cancelled_run_ids"] == []
    assert result["errors"] == []


def test_cancel_superseded_runs_handles_cancel_error(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles individual cancel failures gracefully."""
    from charlie_work.github import cancel_superseded_runs

    class FakeGitHubWithCancelError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.runs_response = [
                {
                    "databaseId": 123,
                    "status": "queued",
                    "createdAt": "2026-07-09T00:00:00Z",
                    "headBranch": "main",
                },
                {
                    "databaseId": 124,
                    "status": "queued",
                    "createdAt": "2026-07-09T01:00:00Z",
                    "headBranch": "main",
                },
            ]

        def run(self, args, *, json_output=False, allow_failure=False):
            if "cancel" in args and args[-1] == "123":
                # Simulate failure by returning None (allow_failure=True behavior)
                return None
            elif "cancel" in args:
                # Other cancels succeed
                return "Cancelled"
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithCancelError()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 2
    assert result["kept"] == 1
    assert result["cancelled"] == 0  # The only run to cancel (123) failed
    assert result["cancelled_run_ids"] == []  # No successful cancels
    assert len(result["errors"]) == 1
    assert "Failed to cancel run 123" in result["errors"][0]


def test_cancel_superseded_runs_handles_list_error(tmp_path: Path) -> None:
    """Test that cancel_superseded_runs handles list API errors gracefully."""
    from charlie_work.github import cancel_superseded_runs, GitHubError

    class FakeGitHubWithListError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()

        def run(self, args, *, json_output=False, allow_failure=False):
            if "run" in args and "list" in args:
                raise GitHubError("List failed")
            return super().run(args, json_output=json_output, allow_failure=allow_failure)

    fake_gh = FakeGitHubWithListError()
    result = cancel_superseded_runs(fake_gh, "main", "test-workflow")
    assert result["total_queued"] == 0
    assert result["kept"] == 0
    assert result["cancelled"] == 0
    assert len(result["errors"]) == 1
    assert "GitHub API error" in result["errors"][0]
