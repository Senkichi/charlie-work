"""Reuse a passing suite result at the local merge gate (#2125).

Split out of ``local_merge_gate.py`` (shrink-only file-size ratchet). The
pure keying/fail-closed decision is ``local_suite_runner.reusable_gate_result``;
this is the one ``OrchestratorApp`` delegate that acts on it.
"""

from __future__ import annotations

from typing import Any

from charlie_work import local_suite_runner


def _local_gate_try_reuse(
    self,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    decision: dict[str, Any],
    argv: list[str],
    reason: str,
    gate_head: str | None,
    gate_base: str | None,
) -> bool | None:
    """Settle the gate from a prior passing result, or return None to launch.

    Must run before ``launch_suite_gate``, which scrubs the prior result. A
    passing suite for this exact (head, base) pair settles through the normal
    ``_local_gate_resolve_result`` path: no new suite process, same drift check,
    same merge bookkeeping.
    """
    paths = local_suite_runner.suite_gate_paths(self.paths.dispatches, int(pr_key))
    reused = local_suite_runner.reusable_gate_result(
        paths, head_sha=gate_head, base_sha=gate_base, suite_argv=argv
    )
    if reused is None:
        return None
    pr_number = int(pr_key)
    self._local_gate_event(
        "local_merge_gate_result_reused",
        {
            "pr_number": pr_number,
            "issue_number": int(record.get("issue_number") or pr_number),
            "head_sha": gate_head,
            "base_sha": gate_base,
            "argv": list(argv),
            "ended_at": reused.get("ended_at"),
            "duration_seconds": reused.get("duration_seconds"),
            "reason": reason,
        },
    )
    entry["reused_result"] = True
    return self._local_gate_resolve_result(
        pr_key=pr_key,
        record={
            **record,
            "local_suite_head": gate_head,
            "local_suite_base_sha": gate_base,
            "local_suite_argv": list(argv),
        },
        entry=entry,
        branch=branch,
        base_ref=base_ref,
        live_head=gate_head or "",
        decision=decision,
        result=reused,
        paths=paths,
    )
