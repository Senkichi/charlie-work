"""Rescue tier (issue #555): one bounded strong-model rework + cross-family
review attempt inserted between "cheap-worker cap exhausted" and escalating
to a human.

This module holds pure, state-free helpers only — prompt/comment text
construction. State mutation, the durable ``rescue_attempted`` marker, and
dispatch orchestration all live on ``OrchestratorApp`` in workflow.py so the
existing ``_record_event``/``state_lock`` durable-write pattern is the only
place PR/issue state is written (CLAUDE.md invariant: atomic JSON writes,
never a parallel write path).
"""

from __future__ import annotations

from typing import Any

# Human-readable framing per eligible cause. Keys must match the ``reason``
# strings used at the three eligible interception sites in workflow.py:
# record_review's max_rework_cycles cap uses "rework_cycle_cap"; the shared
# _route_janitor_gate_failure_to_rework escalation branch uses the same
# ``reason`` values it already threads through ("merge_conflict",
# "no_op_rework").
_CAUSE_LABELS: dict[str, str] = {
    "rework_cycle_cap": "the normal rework-cycle cap (max_rework_cycles)",
    "merge_conflict": "the conflict-rework cap (max_conflict_rework_attempts)",
    "no_op_rework": "the no-op-rework cap (max_no_op_rework_attempts)",
}


def cause_label(cause: str) -> str:
    return _CAUSE_LABELS.get(cause, cause)


def build_rescue_rework_summary(cause: str, original_summary: str) -> str:
    """Prompt text handed to the rescue (Opus) rework worker.

    Frames the attempt explicitly as the bounded rescue tier so the worker
    understands this is the last automated attempt before a human looks at
    the PR — distinct from a normal request_changes rework prompt.
    """
    header = (
        "## Rescue-tier rework (issue #555)\n\n"
        f"This PR exhausted {cause_label(cause)}. This is a single, bounded "
        "rescue attempt using a stronger model before the PR escalates to a "
        "human. There will be no further automated rework cycles after this "
        "one — make the fix count.\n\n"
        "The review feedback that triggered this rescue attempt:\n\n"
    )
    return header + (original_summary or "(no summary was recorded)")


def build_rescue_review_prompt(
    *,
    pr_number: int,
    issue_number: int | None,
    branch: str,
    diff_text: str,
    cause: str,
) -> str:
    """Prompt text for the cross-family rescue reviewer.

    Asks the reviewer to emit the SAME fenced-JSON verdict contract normal
    (Claude) reviewers use, so the result can be parsed with the existing
    ``_extract_verdict_from_text``/``_validate_review_verdict`` helpers in
    workflow.py instead of a second, parallel verdict grammar.
    """
    return f"""# Cross-family rescue review — PR #{pr_number}

This PR (branch `{branch}`, linked issue #{issue_number}) reached this review
because {cause_label(cause)} exhausted the normal automated rework budget. A
stronger model was given ONE bounded rework attempt; you are reviewing the
result of that attempt.

This is the exit gate for the rescue tier: if you request changes, this PR
escalates to a human immediately — there is no further automated rework
loop. Only approve if the diff is genuinely mergeable quality; a human will
read your verdict as one of two documented positions (yours, and the rescue
worker's), not as a gate that can be re-tried.

## Diff under review

```diff
{diff_text}
```

## Required response format

End your response with a single fenced JSON block (must be the LAST fenced
block in your output) with this exact shape:

```json
{{
  "decision": "approved" | "request_changes" | "blocked",
  "summary": "one or two sentence explanation of the decision",
  "required_changes": ["list of specific required changes, only for request_changes"]
}}
```
"""


def build_rescue_escalation_comment(
    *,
    cause: str,
    rescue_branch: str,
    rescue_head_sha: str,
    cross_family_report_path: str,
    verdict_summary: str,
) -> str:
    """PR comment posted when a rescue-review verdict escalates to a human.

    Attaches BOTH artifacts per the issue spec: a reference to the rescue
    worker's diff/PR (branch + head SHA — the PR itself IS that artifact,
    since the rescue rework was pushed to the same PR branch) and the
    cross-family review report path, so the human arbitrates two documented
    positions instead of a bare stuck ticket.
    """
    return (
        "## Rescue tier exhausted — escalated to a human\n\n"
        f"This PR exhausted {cause_label(cause)}, triggering the bounded "
        "rescue tier (issue #555): one Opus rework attempt, reviewed "
        "cross-family. The rescue review did not approve the result, so "
        "this now needs a human — arbitrating two documented positions "
        "rather than a bare stuck ticket.\n\n"
        "**Artifact 1 — the rescue rework attempt:**\n"
        f"- Branch: `{rescue_branch}`\n"
        f"- Head SHA: `{rescue_head_sha}`\n"
        "- (this PR's current diff IS the rescue attempt — the rescue "
        "worker pushed directly to this branch)\n\n"
        "**Artifact 2 — the cross-family review report:**\n"
        f"- `{cross_family_report_path}`\n\n"
        f"Cross-family reviewer summary: {verdict_summary or '(no summary provided)'}\n"
    )


def build_rescue_dataclass_kwargs(cause: str) -> dict[str, Any]:
    """Marker/bookkeeping fields written onto a PR state record when a rescue
    is dispatched. Centralized here so every interception site in workflow.py
    stamps the identical shape.
    """
    return {"rescue_attempted": True, "rescue_cause": cause}


__all__ = [
    "cause_label",
    "build_rescue_rework_summary",
    "build_rescue_review_prompt",
    "build_rescue_escalation_comment",
    "build_rescue_dataclass_kwargs",
]
