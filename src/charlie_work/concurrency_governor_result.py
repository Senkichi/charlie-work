"""``ConcurrencyGovernorResult`` value type (wave D5b).

A leaf module: stdlib only, so both ``workflow`` (re-export) and
``orchestration.reap_dispatch`` (sole constructor) can import it with no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConcurrencyGovernorResult:
    """Result of applying concurrency governor to a dispatch limit.

    This encapsulates the concurrency limiting logic and ensures all related
    fields are bound together, eliminating Pyright's reportPossiblyUnbound
    warnings for live_count.
    """

    clamped: bool
    max_concurrent: int
    live_count: int
    available_slots: int
    dispatch_limit: int
    fleet_live_count: int = 0
    fleet_max: int = 0
    # Issue #1129: open-PR backpressure fields. Populated only when the
    # governor was called with ``apply_open_pr_backpressure=True`` (fresh-issue
    # dispatch) and ``dispatch.max_open_agent_prs`` is > 0. Left at 0 for
    # rework/recovery/loop paths, which are exempt from this clamp.
    open_pr_count: int = 0
    open_pr_max: int = 0
    # Issue #1770: CI-capacity headroom fields. ``ci_headroom_ratio`` mirrors
    # ``open_pr_max``'s exemption -- populated only for the same
    # ``apply_open_pr_backpressure=True`` (fresh-issue) call, 0.0 for
    # rework/recovery/loop paths regardless of the configured ratio.
    # ``ci_headroom`` is the ``ci_headroom_available()`` reading: ``None``
    # when the ratio is 0 (clamp off) or the data could not be trusted this
    # pass (fail-open -- see ``ci_headroom``'s docstring), an
    # int otherwise.
    ci_headroom: int | None = None
    ci_headroom_ratio: float = 0.0
    # Issue #1843: host-load backpressure fields. Unlike the two fields above
    # this term applies to EVERY governor caller (loop wave budget, rework,
    # fresh dispatch) -- a worker launch adds real host load regardless of
    # lane. ``host_load_pytest_processes``/``host_load_pytest_trees`` are the
    # ``host_load.measure_host_load`` reading for this call: ``None`` when
    # both knobs are 0 (off), when the running limit was already 0 (no launch
    # could happen, so no probe), or when the probe itself failed (fail-open
    # -- see host_load.py), ints otherwise. Issue #1903 split the term into
    # two knobs: ``host_load_max_pytest_trees`` (the governor -- clamps by
    # suite headroom ``cap - live_trees``) and ``host_load_max_pytest_processes``
    # (the fan-out brake -- strict ``>`` trip to 0 on abnormal ``-n`` width).
    # Issue #1943: both reported counts are scoped to
    # orchestrator-attributable trees (any member/ancestor command line
    # referencing a managed state/worktree path) -- CI-runner and other
    # foreign suites feed neither count.
    host_load_max_pytest_processes: int = 0
    host_load_max_pytest_trees: int = 0
    host_load_pytest_processes: int | None = None
    host_load_pytest_trees: int | None = None
    # Which term actually bound ``dispatch_limit`` this call, e.g.
    # "ci_headroom", "open_pr_max", "fleet_max", "max_concurrent",
    # "host_load", or ``None`` when nothing clamped. The terms apply in
    # sequence, each only tightening (never loosening) the running limit, so
    # whichever term last reduced it is the true binding constraint -- this
    # is what makes a "0 dispatched" pass explainable from the event alone
    # instead of requiring a reader to redo the min() by hand
    # (zero-dispatch-is-a-capacity-question-first).
    clamped_by: str | None = None

    @property
    def enabled(self) -> bool:
        """Return True if the governor is enabled (max_concurrent > 0)."""
        return self.max_concurrent > 0

    @property
    def fleet_enabled(self) -> bool:
        """Return True if the fleet governor is enabled (fleet_max > 0)."""
        return self.fleet_max > 0

    @property
    def open_pr_enabled(self) -> bool:
        """Return True if the open-PR backpressure clamp is enabled (open_pr_max > 0)."""
        return self.open_pr_max > 0

    @property
    def ci_headroom_enabled(self) -> bool:
        """Return True if the CI-headroom clamp is enabled (ci_headroom_ratio > 0)."""
        return self.ci_headroom_ratio > 0

    @property
    def host_load_enabled(self) -> bool:
        """Return True if the host-load clamp is enabled (either knob > 0)."""
        return self.host_load_max_pytest_processes > 0 or self.host_load_max_pytest_trees > 0

    @property
    def any_term_enabled(self) -> bool:
        """Return True if any governor term is enabled.

        Single point of enforcement (issue #1770 review finding 1): every
        call site that decides whether to splat ``report_fields()`` into a
        ``CommandResult.data`` dict must gate on this property, never on a
        hand-written ``or``-chain of the individual ``*_enabled`` flags. A
        hand-written chain silently stops covering new terms the moment one
        is added -- exactly what happened when ``ci_headroom_enabled`` shipped
        without being added to the ten pre-existing
        ``gov.enabled or gov.fleet_enabled or gov.open_pr_enabled`` sites, so
        a repo that opted into *only* the CI-headroom clamp (the other three
        left at 0, precisely the "opt into just the new clamp" rollout the
        config comment advertises) got its ``dispatch_limit`` clamped to 0
        with no ``ci_headroom``/``clamped_by`` field in the result to explain
        why. Deriving this from the flags themselves means the next new term
        cannot repeat that gap.
        """
        return (
            self.enabled
            or self.fleet_enabled
            or self.open_pr_enabled
            or self.ci_headroom_enabled
            or self.host_load_enabled
        )

    def report_fields(self) -> dict[str, Any]:
        """Return the fields to include in CommandResult.data when clamped."""
        fields: dict[str, Any] = {
            "concurrency_limit": self.max_concurrent,
            "live_session_count": self.live_count,
            "available_slots": self.available_slots,
        }
        if self.fleet_enabled:
            fields["fleet_concurrency_limit"] = self.fleet_max
            fields["fleet_live_session_count"] = self.fleet_live_count
        if self.open_pr_enabled:
            fields["open_pr_count"] = self.open_pr_count
            fields["open_pr_max"] = self.open_pr_max
        if self.ci_headroom_enabled:
            fields["ci_headroom"] = self.ci_headroom
            fields["ci_headroom_ratio"] = self.ci_headroom_ratio
        if self.host_load_enabled:
            fields["host_load_max_pytest_processes"] = self.host_load_max_pytest_processes
            fields["host_load_max_pytest_trees"] = self.host_load_max_pytest_trees
            fields["host_load_pytest_processes"] = self.host_load_pytest_processes
            fields["host_load_pytest_trees"] = self.host_load_pytest_trees
        if self.clamped_by is not None:
            fields["clamped_by"] = self.clamped_by
        return fields
