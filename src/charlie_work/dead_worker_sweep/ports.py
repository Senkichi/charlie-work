"""The apply shell's one seam onto ``charlie_work.workflow``.

The sweep historically reached the workflow namespace through scattered deferred
``import charlie_work.workflow as _wf`` lookups so suite patches on
``charlie_work.workflow.<name>`` stayed live. ``SweepPorts`` names every such
lookup once. ``ports_from_workflow`` resolves each attribute at *call* time (never
at construction), so a ``patch("charlie_work.workflow._worker_pid_alive")`` taken
after the ports were built still takes effect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class SweepPorts:
    worker_pid_alive: Callable[..., bool]
    remote_branch_head_sha: Callable[..., Any]
    remote_branch_ahead_count: Callable[..., Any]
    open_pr_for_orphaned_branch: Callable[..., Any]
    worktree_head_sha: Callable[..., Any]
    salvage_push_stranded_commits: Callable[..., Any]
    utc_now: Callable[..., Any]
    parse_iso_timestamp: Callable[..., Any]
    escalate_issue: Callable[..., Any]
    slugify: Callable[..., str]
    load_state: Callable[..., dict[str, Any]]
    state_lock: Callable[..., Any]
    linked_issue_number: Callable[..., Any]


# ``SweepPorts`` field -> attribute name on ``charlie_work.workflow``.
_WORKFLOW_ATTRS: dict[str, str] = {
    "worker_pid_alive": "_worker_pid_alive",
    "remote_branch_head_sha": "remote_branch_head_sha",
    "remote_branch_ahead_count": "remote_branch_ahead_count",
    "open_pr_for_orphaned_branch": "_open_pr_for_orphaned_branch",
    "worktree_head_sha": "worktree_head_sha",
    "salvage_push_stranded_commits": "salvage_push_stranded_commits",
    "utc_now": "utc_now",
    "parse_iso_timestamp": "_parse_iso_timestamp",
    "escalate_issue": "_escalate_issue",
    "slugify": "slugify",
    "load_state": "load_state",
    "state_lock": "state_lock",
    "linked_issue_number": "linked_issue_number",
}


def _late_bound(attr: str) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        import charlie_work.workflow as _wf

        return getattr(_wf, attr)(*args, **kwargs)

    call.__name__ = attr
    return call


def ports_from_workflow() -> SweepPorts:
    """Ports that resolve through ``charlie_work.workflow`` on every call."""
    return SweepPorts(**{name: _late_bound(attr) for name, attr in _WORKFLOW_ATTRS.items()})


assert {f.name for f in fields(SweepPorts)} == set(_WORKFLOW_ATTRS)
