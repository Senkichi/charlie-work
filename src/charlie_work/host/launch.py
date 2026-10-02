"""Host port: reviewer launch.

Leaf module (stdlib only at top). ``RealReviewLauncher`` late-binds to
``workflow._REVIEW_LAUNCHERS`` (the dict ``review_launch`` owns, re-exported by
``workflow``) so ``patch.dict``/``patch.object`` on either name and the
primitive-launcher patches on ``workflow`` keep intercepting. Launch is
non-blocking (the underlying launchers use ``Popen``) and failures come back as
a record with ``.error`` -- this port never raises.
"""

from __future__ import annotations

from typing import Any, Protocol


class ReviewLauncher(Protocol):
    def launch(self, harness: str, **kwargs: Any) -> Any:
        """Launch a reviewer on ``harness``; return a record with ``.error``/``.pid``."""
        ...


def error_record(
    harness: str,
    kwargs: dict[str, Any],
    error: str,
    *,
    error_class: str | None = None,
) -> Any:
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


class RealReviewLauncher:
    def launch(self, harness: str, **kwargs: Any) -> Any:
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
