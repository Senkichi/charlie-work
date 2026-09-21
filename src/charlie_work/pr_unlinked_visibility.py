"""Pure detection/summary logic for issue-less-PR visibility.

The merge lane's per-PR loop (``_loop_body`` in
``orchestration/reap_loop.py``) skips any open PR whose linked issue cannot
be resolved -- by design, since the issue is the pipeline's unit of work
(see the ``pr-without-linked-issue-is-invisible-to-merge-lane`` project
memory). Before this module the skip was a bare ``continue`` fired before
any instrumentation: no event, no operator-facing signal. Real-world cost:
8 Dependabot PRs rotting unreviewed in job-cannon since 2026-09-02, and PRs
#91/#34 in swole since August.

This module owns the producer/detector split the issue #817 fleet-health
baseline established (``fleet_health_baseline.py``): a pure function
(``compute_unlinked_pr_transition``) decides whether a PR's observable state
has materially changed since the last emission, so the caller emits an
edge-triggered event -- once on first sight, again only on a real change --
instead of re-firing identically every pass. Kept a plain module (not under
``charlie_work.orchestration``) because every top-level ``def`` in that
package auto-installs onto ``OrchestratorApp`` as a ``self``-taking method
(``workflow_delegation.discover_delegate_modules``); these are pure helpers
with no ``self``, so they live here instead, exactly as ``fleet_dispatch``'s
pure helpers live in ``fleet_health_baseline.py`` rather than in
``charlie_work.orchestration``.

No function here performs I/O or makes a GitHub call: every input is a field
``pr_list()`` already returns (``mergeable``, ``mergeStateStatus``,
``headRefOid``, ``author``), so wiring this in adds zero GitHub calls per
pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .issue_linking import linked_issue_number

#: Event kind recorded via ``OrchestratorApp._record_event`` (dual-written to
#: ``state.json``'s event ring and ``events.db``) when an issue-less PR is
#: first seen, or when its fingerprint changes.
UNLINKED_PR_SKIPPED_EVENT_KIND = "pr_unlinked_skipped"

#: Key under ``state["prs"][str(pr_number)]`` holding this PR's persisted
#: edge-detector marker (``first_seen_at`` / ``last_fingerprint`` /
#: ``last_emitted_at``). Mirrors the ``foreign_issue_ref`` marker's placement
#: for the same reason: this PR is otherwise never touched by any other
#: writer, so there is no risk of a concurrent writer clobbering the key.
UNLINKED_PR_NOTICE_KEY = "unlinked_pr_notice"


@dataclass(frozen=True)
class UnlinkedPrFingerprint:
    """The observable, already-in-hand PR fields that define "material state".

    A change in any of these three fields means something an operator would
    care about happened: new commits were pushed (``head_sha``), or GitHub's
    own mergeability/required-check verdict moved (``mergeable``,
    ``merge_state_status``). All three come from the same ``pr_list()``
    payload the loop already fetched once this pass -- no ``pr_checks()`` or
    other per-PR call is made to build this.
    """

    mergeable: str | None
    merge_state_status: str | None
    head_sha: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "mergeable": self.mergeable,
            "merge_state_status": self.merge_state_status,
            "head_sha": self.head_sha,
        }


def unlinked_pr_fingerprint(pr: dict[str, Any]) -> UnlinkedPrFingerprint:
    """Derive the material-state fingerprint from a live ``pr_list()`` entry."""
    return UnlinkedPrFingerprint(
        mergeable=pr.get("mergeable"),
        merge_state_status=pr.get("mergeStateStatus"),
        head_sha=pr.get("headRefOid"),
    )


def unlinked_pr_author(pr: dict[str, Any]) -> tuple[str | None, bool]:
    """Return ``(login, is_bot)`` for a PR's author.

    Uses the same API-supplied ``author.type``/``author.is_bot``
    discriminator ``workflow._is_bot_comment`` uses for comment authors --
    never a hardcoded login list, so a new bot account (or a human author)
    classifies correctly with no code change.
    """
    author = pr.get("author")
    if not isinstance(author, dict):
        return None, False
    login = author.get("login")
    is_bot = author.get("type") == "Bot" or author.get("is_bot") is True
    return (str(login) if login else None), bool(is_bot)


def _observed_days_since(first_seen_at: Any, now: datetime) -> float | None:
    """Days between ``first_seen_at`` (an ISO timestamp) and ``now``, or ``None``."""
    if not isinstance(first_seen_at, str):
        return None
    try:
        first_seen_dt = datetime.fromisoformat(first_seen_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((now - first_seen_dt).total_seconds() / 86400.0, 2)


def _is_unsettled(value: str | None) -> bool:
    """True while GitHub has not finished computing mergeability yet.

    ``pr_list()`` reports ``"UNKNOWN"`` for both ``mergeable`` and
    ``mergeStateStatus`` for a window after any base- or head-branch push,
    while GitHub recomputes the real verdict -- the identical window
    ``orchestration/state_stale_checks.py`` defers on (issue #1451) and
    ``orchestration/helpers_rework.py`` re-probes rather than trusting.
    ``None`` (the field absent from the payload) is treated the same way:
    neither is a settled verdict an operator should compare against.
    """
    return value is None or str(value).upper() == "UNKNOWN"


def _settle_fingerprint(
    previous: dict[str, Any] | None, fingerprint: UnlinkedPrFingerprint
) -> dict[str, str | None]:
    """Carry forward ``mergeable``/``merge_state_status`` while they are unsettled.

    Without this, a PR that flaps ``MERGEABLE -> UNKNOWN -> MERGEABLE`` every
    pass (any push anywhere in the repo can re-trigger GitHub's async
    mergeability computation, not just a push to this PR's own branches)
    would report a fresh "transition" -- and a fresh events.db write -- on
    every single occurrence, even though nothing an operator cares about
    changed. When the current value is unsettled and a previous fingerprint
    exists, this substitutes the previous value so the comparison in
    ``compute_unlinked_pr_transition`` sees no change. ``head_sha`` is never
    carried forward: a new commit is always a real transition regardless of
    mergeability. On first sight (no previous fingerprint) there is nothing
    to carry forward, so the raw, possibly-unsettled value is used as-is --
    it becomes the first-seen baseline, and later passes settle against it.
    """
    current = fingerprint.as_dict()
    if not isinstance(previous, dict):
        return current
    for key in ("mergeable", "merge_state_status"):
        if _is_unsettled(current.get(key)):
            current[key] = previous.get(key)
    return current


def compute_unlinked_pr_transition(
    marker: dict[str, Any] | None,
    fingerprint: UnlinkedPrFingerprint,
    *,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pure edge-detector: decide whether this pass's fingerprint is new.

    Mirrors ``fleet_health_baseline._filter_fleet_health_transitions``'s
    per-entry decision, applied to one PR's persisted marker instead of a
    fleet-wide baseline map. Returns ``(new_marker, event_extra)``:

    - ``event_extra`` is ``None`` when ``fingerprint`` (after settling any
      unsettled ``mergeable``/``merge_state_status`` value via
      ``_settle_fingerprint``) equals the marker's last-emitted fingerprint
      -- nothing materially changed, so the caller must neither emit an
      event nor rewrite ``state.json`` (a standing, unresolved PR costs a
      single read per pass, not a write).
    - Otherwise ``event_extra`` carries ``observed_days`` (days since this PR
      was first observed, tracked in ``first_seen_at`` -- there is no GitHub
      "created at" field in hand without a new API call, so this is time
      since first observation, not true PR age) and ``previous_state`` (the
      prior fingerprint dict, or ``None`` on first sight) for the caller to
      fold into the emitted event's payload. ``new_marker`` is always the
      value the caller must persist.
    """
    marker = marker or {}
    previous_fingerprint = marker.get("last_fingerprint")
    current_fingerprint = _settle_fingerprint(previous_fingerprint, fingerprint)
    if previous_fingerprint == current_fingerprint:
        return marker, None
    first_seen_raw = marker.get("first_seen_at")
    first_seen_at = first_seen_raw if isinstance(first_seen_raw, str) else now.isoformat()
    new_marker = {
        "first_seen_at": first_seen_at,
        "last_fingerprint": current_fingerprint,
        "last_emitted_at": now.isoformat(),
    }
    event_extra = {
        "observed_days": _observed_days_since(first_seen_at, now),
        "previous_state": previous_fingerprint,
    }
    return new_marker, event_extra


def summarize_unlinked_prs(
    prs: list[dict[str, Any]],
    state_prs: dict[str, Any],
    *,
    branch_prefix: str,
    now: datetime,
    branch_issue_validator: Callable[[int], bool] | None = None,
) -> list[dict[str, Any]]:
    """Build the operator-facing "current set" of open, issue-less PRs.

    The complement to the edge-triggered event: an event fires once per
    material change, but an operator running ``charlie status`` must be able
    to see the whole standing set at any time, not just the last transition.
    Reads only ``prs`` (a live ``pr_list()`` payload the caller already
    fetched) and ``state_prs`` (``state["prs"]``, for each PR's persisted
    ``first_seen_at`` marker) -- no GitHub calls of its own.

    ``branch_issue_validator``, when supplied, must be the SAME validator
    instance the caller's ``linked_prs`` list resolves its own PRs with
    (issue #1229: a stale ``agent/issue-709-...`` branch left over from a
    merged PR must not bind to a closed/nonexistent issue). Both surfaces
    have to agree on one resolution rule -- passing ``None`` here while
    ``linked_prs`` validates (or vice versa) would let a stale-branch PR
    resolve differently on each side and either double-count it (appearing
    in both lists) or drop it (appearing in neither), instead of the "exactly
    one of the two lists" invariant this module promises.
    """
    summary: list[dict[str, Any]] = []
    for pr in prs:
        issue_number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=branch_prefix,
            branch_issue_validator=branch_issue_validator,
        )
        if issue_number is not None:
            continue
        pr_number = int(pr["number"])
        marker = (state_prs.get(str(pr_number)) or {}).get(UNLINKED_PR_NOTICE_KEY) or {}
        first_seen_at = marker.get("first_seen_at")
        author, is_bot = unlinked_pr_author(pr)
        summary.append(
            {
                "pr_number": pr_number,
                "title": pr.get("title"),
                "url": pr.get("url"),
                "author": author,
                "is_bot": is_bot,
                "observed_days": _observed_days_since(first_seen_at, now),
                "mergeable": pr.get("mergeable"),
                "merge_state_status": pr.get("mergeStateStatus"),
                "first_seen_at": first_seen_at,
            }
        )
    summary.sort(key=lambda entry: entry["pr_number"])
    return summary
