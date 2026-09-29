"""Local-lane kill-switch stall alarm delegate for ``OrchestratorApp`` (issue #1968).

Net-new member (not a Track 2 Phase B moved body): ``local_lanes.py`` is at
its file-size-ratchet mark, so the emit site lives in its own
``charlie_work.orchestration`` submodule and
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
onto ``OrchestratorApp`` like any moved delegate (``self`` binds via the
descriptor protocol). The shared evaluation lives in
``charlie_work.local_lane.kill_switch_stall_payloads`` -- this module is the
``self``-touching seam only (config, ``gh``, paths, state lock, event
record), matching the local_lanes split where pure mechanics live in
``charlie_work.local_lane``.

Workflow-defined names are reached through ``_wf.<name>`` (the module-object
form the package ``__init__`` docstring's rule 2 requires), so the
``charlie_work.workflow`` monkeypatch seams keep landing. Every other free
name is imported directly from its defining module.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work.github import GitHubError
from charlie_work.local_lane import kill_switch_stall_payloads


def _local_kill_switch_stall_alarm(self, *, now: Any = None) -> None:
    """Issue #1968: warn when an honored kill switch strands parked work.

    An explicit ``review_dispatch.enabled``/``auto_merge.enabled`` ``false``
    stays honored (the gated sub-phases keep skipping), but on a
    ``local_issues`` repo that dead-ends every finished ticket at
    ``agent:review-ready`` silently. Emit ``local_lane_kill_switch_stalled``
    once per pass per disabled switch while a review-ready issue is older
    than ``local_lane.kill_switch_stall_hours`` (``0`` mutes the alarm,
    never the switch).
    """
    cfg = self.config
    stall_hours = cfg.local_lane.kill_switch_stall_hours
    if not cfg.local_issues.enabled or stall_hours <= 0:
        return
    try:
        parked = self.gh.issue_list(labels=[cfg.labels.review_ready], state="open")
    except GitHubError:
        return  # intake/doctor already report the scan problem
    payloads = kill_switch_stall_payloads(
        review_dispatch_enabled=cfg.review_dispatch.enabled,
        auto_merge_enabled=cfg.auto_merge.enabled,
        parked=parked,
        issues_state=_wf.load_state_locked(self.paths.state_file).get("issues"),
        stall_hours=stall_hours,
        now=now,
    )
    if not payloads:
        return
    with _wf.state_lock(self.paths.state_file):
        locked = _wf.load_state(self.paths.state_file)
        for payload in payloads:
            locked = self._record_event(locked, "local_lane_kill_switch_stalled", payload)
        self.write_gate.save_state(locked)
