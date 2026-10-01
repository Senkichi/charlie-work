"""Bounded relaunch for infra suite outcomes at the local merge gate (#2127).

Split out of ``local_merge_gate.py`` (shrink-only file-size ratchet). The
classification and the derived timeout are pure and live in
``charlie_work.local_gate_infra``; this is the one ``OrchestratorApp``
delegate that acts on them.
"""

from __future__ import annotations

from typing import Any

from charlie_work.local_gate_infra import LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES, SuiteOutcome


def _local_gate_infra_relaunch(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    decision: dict[str, Any],
    outcome: SuiteOutcome,
    returncode: Any = None,
    duration_seconds: Any = None,
) -> bool:
    """Timeout / summary-less death: relaunch (bounded), never route to rework.

    Never touches the merge-rework counters -- the failure is host weather,
    not a code defect a worker could fix. Past the bound the gate escalates
    under ``local_merge_gate_infra_exhausted`` so the operator sees the cause.
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    relaunches = int(record.get("local_suite_infra_relaunch_count") or 0) + 1
    if relaunches > LOCAL_SUITE_GATE_MAX_INFRA_RELAUNCHES:
        detail = (
            f"merge-gate suite hit an infra failure ({outcome.value}) "
            f"{relaunches - 1} times for pr-{pr_number} and relaunching did "
            "not help; this is not a code defect (host contention or a "
            "killed runner) -- check the host, then `charlie unescalate`"
        )
        entry["outcome"] = "error"
        entry["detail"] = detail
        self._local_merge_error_escalate(
            pr_number, issue_number, branch, detail, reason="local_merge_gate_infra_exhausted"
        )
        return False
    self._local_gate_event(
        "local_suite_infra_relaunched",
        {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "outcome": outcome.value,
            "returncode": returncode,
            "relaunch_count": relaunches,
            "duration_seconds": duration_seconds,
        },
        level="warning",
    )
    return self._local_gate_launch(
        pr_key=pr_key,
        record=record,
        entry=entry,
        branch=branch,
        base_ref=base_ref,
        decision=decision,
        reason="infra_relaunch",
        infra_relaunch_count=relaunches,
    )
