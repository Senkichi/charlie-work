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


def error_record(harness: str, kwargs: dict[str, Any], error: str) -> Any:
    """A launch-failure value shaped like the success record (``pid`` is None)."""
    from ..claude_code import ClaudeWorkerRecord
    from ..state import utc_now

    return ClaudeWorkerRecord(
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


class RealReviewLauncher:
    def launch(self, harness: str, **kwargs: Any) -> Any:
        from .. import workflow

        launcher = workflow._REVIEW_LAUNCHERS.get(harness)
        if launcher is None:
            return error_record(harness, kwargs, f"unsupported reviewer harness: {harness!r}")
        try:
            return launcher(**kwargs)
        except Exception as exc:  # errors from external processes come back as values
            return error_record(harness, kwargs, f"{type(exc).__name__}: {exc}")
