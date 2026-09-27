"""Body-change escape for the janitor's no-op rework gate (issue #1939).

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark: these two helpers are the whole of the #1939
body-change signal, distinct from the diff/head comparisons that stay in
``janitor.py``'s ``_check_no_op_rework`` as the pre-existing checks they
already were. ``janitor.py`` imports ``_body_rework_escape_warning`` back and
``workflow.py`` re-exports ``_body_content_sha256`` through its facade block
(reached via ``_wf.`` by ``orchestration/state_record_review.py``); nothing
here imports back, so there is no cycle.

Background (swole #198 / PR #348): a request_changes finding whose required
fix lives in the PR body/description produces no code delta by construction
-- the rework's only artifact is a ``gh pr edit`` body update (applied on the
worker's behalf through the rework-outcome ``pr_body`` channel, since workers
hold no gh credential). Without this signal such a rework is
indistinguishable from a genuine no-op and the PR pins in a permanent
janitor_gate block. ``_body_rework_escape_warning`` compares the live body
against the verdict's ``reviewed_body_sha256`` baseline -- stamped by
``record_review`` from the body the reviewer actually read -- and reports the
rework as real when they differ.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _body_content_sha256(body: object) -> str:
    """Content hash of a PR body for no-op-rework comparison (issue #1939).

    ``record_review`` stamps the result into the verdict as
    ``reviewed_body_sha256``; ``_check_no_op_rework`` rehashes the live
    ``pr["body"]`` and compares. Line endings are normalized before
    hashing so a transport-level CRLF/LF flip (e.g. ``gh pr edit
    --body-file`` vs. the value the API echoes back) never counts as a
    content change -- the escape hatch must open only on a real body
    edit, never on a serialization artifact. ``None`` (a PR with no
    body, or a JSON ``null``) hashes as the empty string so verdict-time
    and live reads agree.
    """
    text = body if isinstance(body, str) else ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _body_rework_escape_warning(
    pr: Mapping[str, Any],
    pr_state: Mapping[str, Any],
    review_decision: Mapping[str, Any] | None,
) -> str | None:
    """Return the no-op-gate escape warning when the PR body changed.

    A live body whose hash differs from the verdict's
    ``reviewed_body_sha256`` baseline -- stamped from the body the reviewer
    actually read -- is real rework, so the caller skips every diff/head
    comparison below it (patch-id, head-SHA, merge-only): all are blind to
    body edits. A verdict that predates the field, or a PR payload without
    a ``body`` key, yields no comparison and fails closed -- the gate
    behaves exactly as before.
    """
    reviewed_body_sha256 = (review_decision or {}).get("reviewed_body_sha256")
    if not isinstance(reviewed_body_sha256, str) or not reviewed_body_sha256:
        reviewed_body_sha256 = pr_state.get("reviewed_body_sha256")
    if (
        isinstance(reviewed_body_sha256, str)
        and reviewed_body_sha256
        and "body" in pr
        and _body_content_sha256(pr.get("body")) != reviewed_body_sha256
    ):
        return (
            "No-op rework check satisfied: PR body changed since the "
            "request_changes verdict — a body-only rework produces no code "
            "delta but is not a no-op (issue #1939)"
        )
    return None
