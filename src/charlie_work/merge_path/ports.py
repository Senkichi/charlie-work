"""The merge path shell's one seam onto ``charlie_work.workflow``.

Copy of the ``dead_worker_sweep.ports`` pattern. The legacy ``merge_ready`` body
resolved these names as ``charlie_work.workflow`` module globals, and the suite
patches them there (``utc_now``, ``load_state_locked``, ``state_lock``,
``detect_cross_pr_revert``, ``linked_issue_number``, ``load_state``). Each port
resolves its attribute at *call* time, never at construction, so a patch taken
after the ports were built still takes effect.

Names nothing patches on the workflow module (``label_names``,
``summarize_checks``, ``pass_deadline_*`` ...) are imported directly by the
module that uses them -- a port is only for a name the suite reaches through
``charlie_work.workflow``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MergePathPorts:
    load_state_locked: Callable[..., dict[str, Any]]
    load_state: Callable[..., dict[str, Any]]
    state_lock: Callable[..., Any]
    linked_issue_number: Callable[..., Any]
    detect_cross_pr_revert: Callable[..., Any]
    utc_now: Callable[..., str]
    command_result: Callable[..., Any]


# ``MergePathPorts`` field -> attribute name on ``charlie_work.workflow``.
_WORKFLOW_ATTRS: dict[str, str] = {
    "load_state_locked": "load_state_locked",
    "load_state": "load_state",
    "state_lock": "state_lock",
    "linked_issue_number": "linked_issue_number",
    "detect_cross_pr_revert": "detect_cross_pr_revert",
    "utc_now": "utc_now",
    "command_result": "CommandResult",
}


def _late_bound(attr: str) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        import charlie_work.workflow as _wf

        return getattr(_wf, attr)(*args, **kwargs)

    call.__name__ = attr
    return call


def ports_from_workflow() -> MergePathPorts:
    """Ports that resolve through ``charlie_work.workflow`` on every call."""
    return MergePathPorts(**{name: _late_bound(attr) for name, attr in _WORKFLOW_ATTRS.items()})


assert {f.name for f in fields(MergePathPorts)} == set(_WORKFLOW_ATTRS)
