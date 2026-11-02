"""Rollup handlers for the issue-flow kinds: merges, escalations and lifecycle transitions.

The orchestrator records a merge differently per path, so several kinds are merge
evidence. Each emits a ``merged`` milestone and the metrics layer counts the union once
per ``(repo, issue or PR)``:

* ``merge_succeeded``: the orchestrator's own merge (merge path, local/mdls lane); exact refs.
* ``dispatch_merged_pr_references_closed``: issues closed because their PR merged (issues only).
* ``finalize_externally_merged`` / ``reconcile merged_outside_orchestrator``: merged by the
  queue or a human; the reconcile/finalize pass noticed afterwards.

Deliberately not evidence: ``review_dispatch_lifecycle_reaped github_state=merged`` names a
PR only, and the dispatch-side kinds above name only issues, so the two cannot always be
joined; counting it measured +12 phantom merges for charlie-work over 7 days.

Note the 6h trailing re-derive window (``rollup.WINDOW``) is a fixed bound: a fact whose
source event was deleted (``instrumentation._dedupe_events``) longer ago than that is not
removed by it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from ..config import LabelConfig
from .rollup_common import Row, _int, _ints, _milestone, _refs

_TOKEN = re.compile(r"[a-z0-9_]+")
_LABEL_STATES = {
    str(getattr(LabelConfig(), f.name)).lower(): f.name
    for f in fields(LabelConfig)
    if isinstance(getattr(LabelConfig(), f.name), str)
}
_STATE_ALIASES = {"merged": "done"}


def reason_category(reason: Any) -> str | None:
    """Bounded category for an escalation reason; the raw text belongs in ``detail``.

    Enum-like tokens are kept. Free text is cut at the first ``:`` (``cross_repo_target:
    all 3 paths are absent ...`` -> ``cross_repo_target``), slugified, numbers masked and
    limited to three words, so embedded paths and counts never mint new categories.
    """
    if not isinstance(reason, str) or not reason.strip():
        return None
    if _TOKEN.fullmatch(reason):
        return reason
    head = re.sub(r"\d+", "n", reason.split(":", 1)[0].strip().lower())
    words = [w for w in re.split(r"[^a-z0-9]+", head) if w]
    return "_".join(words[:3]) or None


def escalation(ev: dict, milestone: str, reason: Any) -> list[Row]:
    issue, pr = _refs(ev)
    category = reason_category(reason)
    detail = reason if isinstance(reason, str) and reason != category else None
    row = {
        "issue": _int(issue),
        "pr": _int(pr),
        "event_kind": ev["kind"],
        "reason": category,
        "detail": detail,
    }
    return [("escalations", row), _milestone(ev, milestone, issue, pr)]


def _escalated(ev: dict) -> list[Row]:
    p = ev["payload"]
    # ``session_failed_escalated`` carries ``failure_kind`` instead of a reason.
    reason = p.get("reason") or p.get("escalation_reason") or p.get("failure_kind")
    return escalation(ev, "escalated", reason)


def _unescalate(ev: dict) -> list[Row]:
    return escalation(ev, "unescalated", ev["payload"].get("cleared_escalation_reason"))


def _reconcile(ev: dict) -> list[Row]:
    if ev["payload"].get("kind") != "merged_outside_orchestrator":
        return []
    issue, pr = _refs(ev)
    return [_milestone(ev, "merged", issue, pr, True)]


def _finalize_merged(ev: dict) -> list[Row]:
    """One merge per finalised issue. Payload lists are sorted separately, so a PR is
    paired to an issue only when the lists are the same length (the 1:1 case)."""
    p = ev["payload"]
    issues, prs = _ints(p.get("issue_numbers")), _ints(p.get("pr_numbers"))
    if not issues and not prs:
        issue, pr = _refs(ev)
        return [_milestone(ev, "merged", issue, pr, True)]
    if len(issues) == len(prs):
        return [_milestone(ev, "merged", i, pr, True) for i, pr in zip(issues, prs, strict=True)]
    if issues:
        return [_milestone(ev, "merged", i, None, True) for i in issues]
    return [_milestone(ev, "merged", None, pr, True) for pr in prs]


def _merge_succeeded(ev: dict) -> list[Row]:
    issue, pr = _refs(ev)
    return [_milestone(ev, "merged", issue, pr)]


def _merged_pr_references_closed(ev: dict) -> list[Row]:
    issues = _ints(ev["payload"].get("issue_numbers"))
    return [_milestone(ev, "merged", i, None, True) for i in issues]


def lifecycle_state(raw: str) -> str:
    """Canonical state name: a ``LabelConfig`` label (``agent:in-progress`` ->
    ``in_progress``), a state-cache status (``merged`` -> ``done``) or a human name
    (``PR open`` -> ``pr_open``)."""
    text = raw.strip().lower()
    if text in _LABEL_STATES:
        return _LABEL_STATES[text]
    name = "_".join(text.removeprefix("agent:").replace("-", " ").split())
    return _STATE_ALIASES.get(name, name)


def _lifecycle_transition(ev: dict) -> list[Row]:
    """Exact lifecycle path (issue #2226): the normalised ``to_state`` is the milestone name.

    ``approx=0`` marks it exact for the metrics layer; a row with no usable state yields
    nothing.
    """
    p = ev["payload"]
    state = p.get("to_state") or p.get("to")
    if not isinstance(state, str) or not state.strip():
        return []
    issue, pr = _refs(ev)
    return [_milestone(ev, lifecycle_state(state), issue, pr)]


FLOW_HANDLERS: dict[str, Callable[[dict], list[Row]]] = {
    "session_failed_escalated": _escalated,
    "review_dispatch_escalated": _escalated,
    "janitor_rework_escalated": _escalated,
    "dispatch_cross_repo_escalated": _escalated,
    "unescalate": _unescalate,
    "reconcile": _reconcile,
    "finalize_externally_merged": _finalize_merged,
    "merge_succeeded": _merge_succeeded,
    "dispatch_merged_pr_references_closed": _merged_pr_references_closed,
    "lifecycle_transition": _lifecycle_transition,
}
