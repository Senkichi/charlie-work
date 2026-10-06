"""Body-change escape for the janitor's no-op rework gate (issue #1939).

Extracted from ``janitor.py`` to keep that module under its recorded
file-size-ratchet mark: ``_body_content_sha256`` and
``_body_rework_escape_warning`` are the whole of the #1939 body-change
signal, distinct from the diff/head comparisons that stay in
``janitor.py``'s ``_check_no_op_rework`` as the pre-existing checks they
already were. ``_request_changes_body_drifted`` (issue #1983) applies the
same baseline contract at the review-queue boundary: a ``request_changes``
verdict whose recorded body hash no longer matches the live body must not
carry forward or reroute to rework -- it queues a fresh review instead.
``janitor.py`` imports ``_body_rework_escape_warning`` back,
``orchestration/github_ops_review_queue.py`` imports
``_request_changes_body_drifted``, and ``workflow.py`` re-exports
``_body_content_sha256`` through its facade block (reached via ``_wf.`` by
``orchestration/state_record_review.py``); nothing here imports back, so
there is no cycle.

Issue #2281 added the sibling exemption-claim escape:
``_trailer_exempt_escape_warning`` (the gate-side decision, mirroring
``_body_rework_escape_warning``'s warning-or-None contract) and
``no_op_escape_needs_pr_commits`` (the caller-side predicate that says
when fetching the PR's commit list is worth the REST call). The raw
claim-vs-baseline reader lives in ``test_adequacy_exempt.py`` as
``newly_claimed_exempt_reason``, next to the #2220 trailer parser it
composes. ``janitor.py`` re-exports ``no_op_escape_needs_pr_commits`` so
``workflow.py`` and ``orchestration/state_mechanical.py`` keep importing
the gate's surface from the gate's module.

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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from charlie_work.test_adequacy_exempt import newly_claimed_exempt_reason

if TYPE_CHECKING:
    from charlie_work.config import TestAdequacyConfig


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


def _request_changes_body_drifted(
    review_decision: Mapping[str, Any] | None,
    pr: Mapping[str, Any],
) -> bool:
    """True when a ``request_changes`` verdict's body baseline drifted (issue #1983).

    The carry-forward/reroute half of the #1939 body-change signal: a
    ``request_changes`` verdict whose required fixes live in the PR body is
    superseded once the live body differs from the verdict's
    ``reviewed_body_sha256`` baseline -- the body-only rework already
    happened, and neither ``_check_carry_forward`` (patch-id/line-content)
    nor ``_reroute_stranded_request_changes`` can see it. ``review_queue()``
    treats drifted verdicts as stale and queues a fresh review; the verdict
    is superseded by the new recorded verdict, never auto-approved.

    Fails closed -- a non-``request_changes`` verdict (an ``approved``
    verdict still carries forward across a body edit), a missing/empty
    baseline (a verdict recorded before ``reviewed_body_sha256`` existed),
    or a PR payload without a ``body`` key all return False and preserve
    the pre-#1983 behavior exactly.
    """
    decision = review_decision or {}
    if decision.get("decision") != "request_changes":
        return False
    reviewed_body_sha256 = decision.get("reviewed_body_sha256")
    return (
        isinstance(reviewed_body_sha256, str)
        and bool(reviewed_body_sha256)
        and "body" in pr
        and _body_content_sha256(pr.get("body")) != reviewed_body_sha256
    )


def no_op_escape_needs_pr_commits(
    test_adequacy: TestAdequacyConfig,
    review_decision: Mapping[str, Any] | None,
    live_head_sha: str | None = None,
) -> bool:
    """True when ``run_janitor`` can use the PR's commit list (issue #2281).

    Callers should only spend a ``pr_commits`` fetch when the no-op gate's
    exemption-claim escape is actually live: the test-adequacy gate (which
    sanctions the ``exempt_marker`` channel) is enabled AND a
    ``request_changes`` verdict exists for the check to evaluate against.
    Every other case -- no verdict, an approved/blocked verdict, the gate
    disabled -- makes the commit list dead weight, so this returns False
    and the caller may pass ``pr_commits=None``.

    A ``live_head_sha`` still pinned at the verdict's ``reviewed_head_sha``
    also returns False: a newly claimed exemption must arrive on a commit
    AFTER the reviewed head, so an unadvanced head means the escape cannot
    fire no matter what the commit list holds. That arm is what keeps an
    escalated (and therefore head-frozen) request_changes PR from costing
    a REST call on every loop pass. A missing/unknown live head or a
    verdict without ``reviewed_head_sha`` cannot prove the escape dead, so
    both still fetch -- undeterminable stays on the fetching side, the same
    fail-safe direction as a ``pr_state``-only ``reviewed_head_sha`` the
    predicate cannot see.
    """
    decision = review_decision or {}
    if not (test_adequacy.enabled and decision.get("decision") == "request_changes"):
        return False
    reviewed_head = decision.get("reviewed_head_sha")
    return not (
        isinstance(reviewed_head, str)
        and reviewed_head
        and live_head_sha
        and str(live_head_sha) == reviewed_head
    )


def _trailer_exempt_escape_warning(
    pr: Mapping[str, Any],
    pr_state: Mapping[str, Any],
    review_decision: Mapping[str, Any] | None,
    pr_commits: Sequence[Mapping[str, Any]] | None,
    test_adequacy: TestAdequacyConfig,
) -> str | None:
    """Return the no-op-gate escape warning on a newly claimed exemption (issue #2281).

    The #2220 trailer channel -- ``<marker> <reason>`` on a commit, usually
    an empty one -- is the test-adequacy gate's own remedy, but it produces
    no diff delta, so the patch-id comparison in ``_check_no_op_rework``
    flagged the remedy itself as a no-op (swole #487 / PR #490). A live
    exemption claim the reviewed head did not already make is substantive
    rework, resolved by ``test_adequacy_exempt.newly_claimed_exempt_reason``
    against the same decision-first ``reviewed_head_sha`` baseline the rest
    of the gate reads. Like every baseline in the gate, undeterminable
    newness fails closed: the adequacy gate disabled, no commit list, or
    the reviewed head absent from it (force-push/rebase) all return None
    and leave the checks below in charge.
    """
    if not test_adequacy.enabled:
        return None
    reviewed_head = (review_decision or {}).get("reviewed_head_sha")
    if not isinstance(reviewed_head, str) or not reviewed_head:
        reviewed_head = pr_state.get("reviewed_head_sha")
    new_exempt_reason = newly_claimed_exempt_reason(
        str(pr.get("body") or ""),
        pr_commits or (),
        str(reviewed_head or ""),
        test_adequacy.exempt_marker,
    )
    if not new_exempt_reason:
        return None
    return (
        "No-op rework check satisfied: PR newly claims a test-adequacy "
        f"exemption ({test_adequacy.exempt_marker} {new_exempt_reason}) "
        "— a trailer-only rework produces no code delta but is not a "
        "no-op (issue #2281)"
    )
