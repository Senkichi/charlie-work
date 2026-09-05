"""Unauthorized-merge baseline delegate for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf


def _apply_unauthorized_merge_baseline(
    self, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bound the tripwire to merges it could actually have governed.

    The tripwire asserts a policy — every merged worker PR is covered by an
    ``approved`` decision recorded at its merged head — that this
    repository's history predates. ``merged_pr_list()`` returns the last 500
    closed PRs, and many of those merged before the review gate existed,
    under the #597 hollow-verdict bug, or with no decision file at all.
    Measured against the live repo before this landed, an unbounded first
    pass yields 48 findings.

    Those 48 would append to ``loop()``'s ``errors`` bucket on *every* pass
    — there is no dedupe and the 500-PR window keeps them in scope for a
    very long time — pinning ``ok=False`` permanently and burying any real
    self-merge in constant background noise. That is the same "noise, not
    signal" failure this tripwire exists to prevent, arriving from the other
    direction: a control that can never go quiet is not a control.

    So the first pass *arms* rather than alarms. It records exactly which PRs
    were already merged-and-uncovered, emits them once as an
    ``unauthorized_merge_baseline_armed`` event so the backlog stays
    auditable, and reports nothing. Every later pass reports only merges
    absent from that baseline.

    A second, ongoing dedupe layer handles *post-arming* findings (issue
    #673). The baseline suppresses history once; it cannot suppress a
    genuine bypass that lands after arming, and without an ack mechanism
    such a finding is re-appended to ``errors`` on every single pass for as
    long as it stays inside ``merged_pr_list()``'s 500-PR window — pinning
    ``ok=False`` forever and drowning any new signal in constant noise. That
    is the same "a control that can never go quiet is not a control" failure
    the baseline exists to prevent, arriving from the other direction. So a
    finding that has been explicitly acknowledged via
    ``ack_unauthorized_merge`` (state key ``unauthorized_merge_acknowledged``,
    a ``{pr_number: {acknowledged_at, reason, by}}`` map) is filtered out of
    the reported candidates the same way the baseline filters pre-arming
    history. Acknowledgment is never automatic — it requires an explicit
    ``charlie tripwire ack`` action — or the tripwire would defeat itself.

    Why an explicit set of PR numbers and not a high-water PR number: a
    number watermark also exempts any PR that was already open when the
    control armed but merges afterwards. That is not hypothetical here — at
    arming there were three open worker PRs (#510, #585, #630), every one
    numbered below the highest merged PR (#631), so a number watermark would
    have permanently exempted all three, including the PR that adds this
    tripwire. The set is derived at runtime from live data, never hard-coded,
    and suppresses precisely the merges that already happened.

    Cost, stated plainly: a genuine bypass that lands in the same window as
    the arming pass is baselined rather than reported. It still appears in
    the armed event's PR list, which is why that event carries the full set
    rather than just a count.
    """
    import logging

    logger = logging.getLogger(__name__)
    key = _wf.UNAUTHORIZED_MERGE_BASELINE_KEY

    def _suppressed(baseline: Any) -> set[int]:
        if not isinstance(baseline, dict):
            return set()
        raw = baseline.get("pre_existing_prs") or []
        return {int(n) for n in raw if isinstance(n, int) and not isinstance(n, bool)}

    def _acked(ack_map: Any) -> set[int]:
        # The ack set is a JSON object keyed by string PR numbers (JSON keys
        # are always strings). Parse them back to int so the membership test
        # against the int ``c["pr"]`` candidates works.
        if not isinstance(ack_map, dict):
            return set()
        out: set[int] = set()
        for n in ack_map:
            try:
                out.add(int(n))
            except (TypeError, ValueError):
                continue
        return out

    state = _wf.load_state_locked(self.paths.state_file)
    if isinstance(state.get(key), dict):
        pre_existing = _suppressed(state.get(key))
        acknowledged = _acked(state.get(_wf.UNAUTHORIZED_MERGE_ACK_KEY))
        return [
            c for c in candidates if c["pr"] not in pre_existing and c["pr"] not in acknowledged
        ]

    # ---- arming pass ----
    pre_existing_now = sorted({int(c["pr"]) for c in candidates})

    if self.dry_run:
        # A preview must not write state (issues #609/#613/#621). Report what
        # an armed pass would report — nothing — without persisting, so the
        # preview is both write-free and truthful about post-arm behaviour.
        logger.info(
            "DRY-RUN: unauthorized-merge tripwire would arm with %d pre-existing "
            "uncovered merge(s); not persisting",
            len(pre_existing_now),
        )
        return []

    with _wf.state_lock(self.paths.state_file):
        locked = _wf.load_state(self.paths.state_file)
        if isinstance(locked.get(key), dict):
            # Another pass armed between the read above and this lock. Its
            # baseline wins, so arming is idempotent and never re-widens.
            pre_existing = _suppressed(locked.get(key))
            acknowledged = _acked(locked.get(_wf.UNAUTHORIZED_MERGE_ACK_KEY))
            return [
                c
                for c in candidates
                if c["pr"] not in pre_existing and c["pr"] not in acknowledged
            ]
        locked[key] = {
            "armed_at": _wf.utc_now(),
            "pre_existing_prs": pre_existing_now,
        }
        locked = self._record_event(
            locked,
            "unauthorized_merge_baseline_armed",
            {
                "pre_existing_count": len(pre_existing_now),
                "pre_existing_prs": pre_existing_now,
            },
        )
        _wf.save_state(self.paths.state_file, locked)

    logger.warning(
        "unauthorized-merge tripwire armed: %d pre-existing uncovered merge(s) "
        "recorded as baseline and will not be reported again (%s)",
        len(pre_existing_now),
        ", ".join(f"#{n}" for n in pre_existing_now) or "none",
    )
    return []
