"""Host ports: worker and reviewer launch.

Leaf module (stdlib only at top). ``RealReviewLauncher`` late-binds to
``workflow._REVIEW_LAUNCHERS`` (the dict ``review_launch`` owns, re-exported by
``workflow``) and ``RealWorkerLauncher`` late-binds ``workflow.dispatch_sessions``
at call time (issue #2229), so ``patch``/``patch.dict``/``patch.object`` on
either ``workflow`` name keeps intercepting. Launch is non-blocking (the
underlying launchers use ``Popen``) and failures come back as
``LaunchOutcome`` values -- these ports never raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..adapters import AdapterSettings, SessionDispatchResult, SessionRequest


class LaunchOutcome(Protocol):
    """One launch's outcome as a value -- never a raise (issue #2229).

    The surface both lanes' records share: ``error`` carries a failure as a
    value (``None`` on success), ``pid`` the spawned worker/reviewer process
    (``None`` when nothing launched). The worker lane's
    ``SessionDispatchResult`` and the reviewer lane's ``SessionRecord`` /
    ``ClaudeWorkerRecord`` all satisfy it.
    """

    error: str | None
    pid: int | None


class ReviewLauncher(Protocol):
    def launch(self, harness: str, **kwargs: Any) -> LaunchOutcome:
        """Launch a reviewer on ``harness``; return its ``LaunchOutcome``."""
        ...


class WorkerLauncher(Protocol):
    def launch(
        self,
        repo_root: Path,
        manifest_path: Path,
        results_path: Path,
        settings: AdapterSettings,
        requests: list[SessionRequest],
    ) -> list[SessionDispatchResult]:
        """Launch one batch of worker requests -- one ``LaunchOutcome`` per request."""
        ...


def error_record(
    harness: str,
    kwargs: dict[str, Any],
    error: str,
    *,
    error_class: str | None = None,
) -> LaunchOutcome:
    """A launch-failure value shaped like the success record (``pid`` is None).

    Issue #2246: this is the reviewer launch seam's error-record constructor --
    the only place an erroring ``launch()`` result can come into existence
    without the wrapped launcher's own seam having already emitted. Each
    construction emits exactly one ``launch_failed`` event (role="reviewer";
    the record's ``issue_number`` field carries the PR number).
    """
    from .. import launch_events
    from ..claude_code import ClaudeWorkerRecord
    from ..state import utc_now

    record = ClaudeWorkerRecord(
        issue_number=int(kwargs.get("pr_number") or 0),
        branch=str(kwargs.get("branch") or ""),
        worktree_path="",
        prompt_path=str(kwargs.get("prompt_path") or ""),
        command=(),
        pid=None,
        started_at=utc_now(),
        log_path="",
        error=error,
        adapter_kind=harness,
    )
    repo_root = kwargs.get("repo_root")
    if repo_root is not None:
        launch_events.emit_launch_failed(
            launch_events.state_path_for(repo_root, kwargs.get("config")),
            role="reviewer",
            harness=harness,
            # The resolved role-chain reviewer entry; empty when the caller
            # left it unset (api reviewers pin the provider model instead).
            model=str(kwargs.get("model_override") or ""),
            pr_number=record.issue_number or None,
            error_class=error_class or launch_events.LAUNCH_ERR_INTERNAL,
            error=error,
        )
    return record


def worker_error_results(
    repo_root: Path,
    settings: AdapterSettings,
    requests: list[SessionRequest],
    error: str,
    *,
    error_class: str | None = None,
) -> list[SessionDispatchResult]:
    """One ``launch_failed`` event + failed ``SessionDispatchResult`` per request.

    Issue #2229: the worker launch seam's error-record constructor -- the
    mirror of :func:`error_record` for the reviewer lane. It is the only place
    an erroring worker ``launch()`` batch can come into existence without the
    wrapped launcher's own seam having already emitted (a raise in
    ``dispatch_sessions`` or in the port itself). Each request gets exactly
    one ``launch_failed`` event (role="worker") and one failed result.
    """
    from .. import launch_events
    from ..adapters import SessionDispatchResult

    results = []
    for request in requests:
        launch_events.emit_launch_failed(
            launch_events.state_path_for(repo_root, settings.config),
            role="worker",
            harness=settings.adapter,
            model=settings.worker_model
            or (settings.config.worker.model if settings.config else ""),
            issue_number=request.issue_number,
            error_class=error_class or launch_events.LAUNCH_ERR_INTERNAL,
            error=error,
        )
        results.append(
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter=settings.adapter,
                ok=False,
                error=error,
            )
        )
    return results


class RealReviewLauncher:
    def launch(self, harness: str, **kwargs: Any) -> LaunchOutcome:
        from .. import workflow

        launcher = workflow._REVIEW_LAUNCHERS.get(harness)
        if launcher is None:
            return error_record(
                harness,
                kwargs,
                f"unsupported reviewer harness: {harness!r}",
                error_class="config",
            )
        try:
            return launcher(**kwargs)
        except Exception as exc:  # errors from external processes come back as values
            return error_record(harness, kwargs, f"{type(exc).__name__}: {exc}")


class RealWorkerLauncher:
    def launch(
        self,
        repo_root: Path,
        manifest_path: Path,
        results_path: Path,
        settings: AdapterSettings,
        requests: list[SessionRequest],
    ) -> list[SessionDispatchResult]:
        from .. import workflow

        try:
            return workflow.dispatch_sessions(
                repo_root, manifest_path, results_path, settings, requests
            )
        except Exception as exc:  # errors from external processes come back as values
            return worker_error_results(
                repo_root, settings, requests, f"launch failed: {exc}"
            )
