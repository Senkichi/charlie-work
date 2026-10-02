"""The ``launch_failed`` event: one emit seam for every failed launch (issue #2246).

Before this module, a failed worker/reviewer launch surfaced only indirectly:
the review-miss reaper emitted ``review_verdict_missed`` with
``reason="launch_failed"``, and the dead-worker sweep emitted
``session_exited`` with ``failure_kind="launch_failed"`` a sweep cycle later.
An adapter-level launch error came back as a bare value (``adapters.py``'s
``except`` shim, ``api_worker``'s error record, ``devin_review_resume``'s
``_fail``) with no event naming the role, harness, or model.

Now every launch path that returns an error value emits exactly one
``launch_failed`` event at the seam where the ``SessionRecord`` /
``ClaudeWorkerRecord`` (or its caller-side shim) comes back with ``.error``
set. Existing downstream events are unchanged — ``session_exited``,
``review_verdict_missed``, ``review_exec_rejection_resume_failed``, and
``api_budget_refused`` all still fire; this event is additive.

Payload: ``role`` ("worker"/"reviewer"), ``harness`` (the role-chain harness
name, e.g. "devin-shell"/"claude-code"/"api"), ``model`` (the role-chain
entry actually tried — the provider's pinned model for ``api``, the resolved
``--model`` value otherwise; "" when the harness has no model concept or the
entry is unset), ``issue_number``/``pr_number`` (a reviewer launch keys the
PR; the linked issue is not known at the seam and stays None), a bounded
``error_class``, the (truncated) ``error`` text, and the record's
``failure_kind`` when the classifier produced one.

Emission is best-effort and never raises — ``instrumentation.log_event``
swallows I/O errors, and this wrapper catches anything else — so an
observability defect can never convert a returned error value into a raise,
which is the whole reason adapters return errors as values (CLAUDE.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .paths import runtime_paths

logger = logging.getLogger(__name__)

#: Bound on the payload's ``error`` field. Launch errors embed rendered
#: commands and OS exception text; the event needs the diagnosis, not the
#: whole output. The sidecar keeps the full string.
MAX_ERROR_CHARS = 500

# Bounded ``error_class`` vocabulary. Every emit site names the stage of the
# launch that failed — the point is a dashboard can group failures without
# parsing free-text error strings.
LAUNCH_ERR_CONFIG = "config"  # adapter/harness/provider unusable or unconfigured
LAUNCH_ERR_CREDENTIALS = "credentials"  # auth material (env var) absent
LAUNCH_ERR_BUDGET = "budget"  # spend-cap refusal
LAUNCH_ERR_WORKTREE = "worktree"  # worktree / review-checkout setup failed
LAUNCH_ERR_PROMPT = "prompt"  # prompt file read/write failed
LAUNCH_ERR_ENV = "env"  # worker environment construction failed
LAUNCH_ERR_RENDER = "render"  # command-template rendering failed
LAUNCH_ERR_SPAWN = "spawn"  # Popen/exec failed (missing binary, OS error)
LAUNCH_ERR_EXIT = "exit"  # blocking command adapter exited non-zero
LAUNCH_ERR_PRECHECK = "precheck"  # resume precondition absent (sidecar/checkout/id)
LAUNCH_ERR_INTERNAL = "internal"  # unexpected exception inside the launch path

LAUNCH_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        LAUNCH_ERR_CONFIG,
        LAUNCH_ERR_CREDENTIALS,
        LAUNCH_ERR_BUDGET,
        LAUNCH_ERR_WORKTREE,
        LAUNCH_ERR_PROMPT,
        LAUNCH_ERR_ENV,
        LAUNCH_ERR_RENDER,
        LAUNCH_ERR_SPAWN,
        LAUNCH_ERR_EXIT,
        LAUNCH_ERR_PRECHECK,
        LAUNCH_ERR_INTERNAL,
    }
)


def state_path_for(repo_root: Path, config: OrchestratorConfig | None) -> Path:
    """Resolve the ``state.json`` whose sibling ``events.db`` sinks this repo's
    launch events.

    Launch functions take ``config`` optionally (``None`` -> defaults), so the
    state-dir resolution mirrors them: ``runtime.state_dir`` under the passed
    config, else the default. The event lands in the same ``events.db`` the
    loop's append_event writes share -- no separate launch-failure store.
    """
    cfg = config or OrchestratorConfig()
    return runtime_paths(repo_root, cfg.runtime.state_dir).state_file


def emit_launch_failed(
    state_path: Path | None,
    *,
    role: str,
    harness: str,
    error_class: str,
    error: str,
    model: str = "",
    issue_number: int | None = None,
    pr_number: int | None = None,
    failure_kind: str | None = None,
) -> None:
    """Emit exactly one ``launch_failed`` event for one failed launch.

    Called at the seam where a launch comes back as an error value — never
    elsewhere — so "one event per failed launch" holds by construction. A
    ``None`` ``state_path`` (a caller that cannot locate the state dir) skips
    the write rather than raise: instrumentation is best-effort, never fatal.
    """
    try:
        if state_path is None:
            return
        if error_class not in LAUNCH_ERROR_CLASSES:
            logger.warning(
                "launch_failed emitted with unknown error_class %r; "
                "add it to launch_events.LAUNCH_ERROR_CLASSES",
                error_class,
            )
        payload: dict[str, Any] = {
            "role": role,
            "harness": harness,
            "model": model,
            "issue_number": issue_number,
            "pr_number": pr_number,
            "error_class": error_class,
            "error": str(error)[:MAX_ERROR_CHARS],
        }
        if failure_kind is not None:
            payload["failure_kind"] = failure_kind
        from .instrumentation import log_event

        # Launch-seam observability: the launch functions are standalone
        # module-level calls with no WriteGate in scope (CLAUDE.md: standalone
        # functions call log_event directly), and the emit must fire even
        # under dry_run.
        # write-gate-exempt(issue=2246): standalone launch-seam helper; no write_gate receiver in scope, emit is best-effort.
        log_event(state_path, "launch_failed", payload)
    except Exception:  # noqa: BLE001 - instrumentation must never break a launch
        logger.debug("emit_launch_failed swallowed an error", exc_info=True)
