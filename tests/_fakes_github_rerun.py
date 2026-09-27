"""``FakeGitHubWithRerunCapture`` -- the ``gh run rerun``-capturing fake.

Split out of ``tests/_fakes_github.py`` (issue #1936 rework): adding the
configurable ``workflow_runs`` probe payload tipped the shared fake module
over the repo's 800-line file-size cap, and ``file_size_ratchet_baseline``
marks are lowered-only by ``scripts/refresh_file_size_ratchet.py`` -- never
raised by a feature diff. The class keeps the same name and semantics;
importers switch to ``_fakes_github_rerun``.
"""

from __future__ import annotations

from typing import Any

from _fakes_github import FakeGitHubWithChecks
from charlie_work import github as github_module


class FakeGitHubWithRerunCapture(FakeGitHubWithChecks):
    """FakeGitHub that captures gh run rerun calls and can simulate failures."""

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        *,
        rerun_ok: bool = True,
        rerun_error: str = "This workflow run cannot be retried",
        workflow_runs: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(checks)
        self.rerun_ok = rerun_ok
        self.rerun_error = rerun_error
        self.rerun_calls: list[list[str]] = []
        # Issue #1936: configurable workflow_runs_for_head payload so tests
        # can flip the containing run between in-progress and terminal.
        # None means "probe not configured" -> returns None (probe failure),
        # which the driver fails open on.
        self.workflow_runs = workflow_runs
        self.workflow_runs_calls: list[str] = []

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        self.workflow_runs_calls.append(head_sha)
        return self.workflow_runs

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):  # noqa: ANN202
        if len(args) >= 2 and args[0] == "run" and args[1] == "rerun":
            self.rerun_calls.append(list(args))
            if self.rerun_ok:
                return "DRY-RUN: gh run rerun " + " ".join(args[2:])
            return github_module.GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr=self.rerun_error,
                value=None,
                error=self.rerun_error,
            )
        return super().run(args, json_output=json_output, allow_failure=allow_failure)
