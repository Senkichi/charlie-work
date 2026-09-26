"""Local (no-remote) review/merge lane delegates for ``OrchestratorApp`` (issue #1844).

A repo whose issue backend is ``LocalFileGitHub`` has no GitHub: ``pr_view``
returns ``{}``, ``pr_list`` returns ``[]``, and ``merge_pr`` raises. The
remote lane is therefore a dead end -- every finished worker parked at
``agent:review-ready`` waiting for a human to merge by hand.

This module implements the local equivalent of the remote lane:

  worker finishes -> ``park_unpublishable_work`` labels the issue
    ``agent:review-ready`` -> ``_local_review_packets`` adopts the branch into
    a ``state["prs"]`` record marked ``"local": True`` (keyed by issue number)
    and writes the same review packet artifacts the remote lane writes
    (``prs/pr-<n>/pr.json``, ``diff.patch``, ``review-prompt.md``,
    ``review-decision.json``) -> ``_local_dispatch_reviewers`` claims and
    launches a reviewer in the same detached read-only checkout shape ->
    ``record_local_review`` ingests the verdict (same decision file, same
    labels, same rework routing) -> ``_local_merge_approved`` merges the base
    into the branch worktree, runs the full suite there, then advances the
    base branch (fast-forward or ``--no-ff``) and reaps the worktree ->
    ``_local_dispatch_rework`` re-dispatches a worker on the same branch for
    ``request_changes``/suite-failure/conflict outcomes (suspended when the
    pass carries an explicit dispatch budget of 0 -- the ``fleet stop
    --drain`` signal, issue #1716).

Every top-level ``def`` here is installed on ``OrchestratorApp`` by
``workflow_delegation._install_delegates``; pure git/suite mechanics live in
``charlie_work.local_lane`` (never in this module -- a top-level helper here
would silently become a public app method).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import charlie_work.workflow as _wf
from charlie_work.adapters import SessionRequest
from charlie_work.claude_code import resolve_review_effort
from charlie_work.github import GitHubError
from charlie_work.janitor import check_operator_containment, check_test_adequacy
from charlie_work.labels import TransitionOutcome
from charlie_work.local_lane import (
    branch_diff,
    branch_head_sha,
    ensure_branch_worktree,
    is_local_pr_record,
    local_base_branch,
    local_pr_dict,
    local_pr_records,
    merge_branch_into_base,
    run_full_suite,
    suite_command_argv,
)
from charlie_work.local_work_park import publishes_pull_requests
from charlie_work.prompts import prompt_template_digest
from charlie_work.review_decision import (
    record_decision,
    reclassify_human_call_verdict,
)
from charlie_work.safe_path import contains
from charlie_work.state import is_claim_stale, without_review_dispatch_claim
from charlie_work.worker import iter_workers
from charlie_work.worktree import (
    _merge_update_rework_branch,
    list_worktrees,
    remove_review_checkout,
    remove_worktree,
)

# Local review-record statuses. These deliberately reuse the remote lane's
# vocabulary wherever the remote state machine already understands it
# ("reviewing", "rework_requested", "approved", "blocked", "escalated",
# "merged", "closed"); "local_pending" is the one new value -- the adopted
# record between creation and its first packet commit.
LOCAL_PENDING_STATUS = "local_pending"

# Issue statuses meaning a worker process may still be writing to the branch:
# never rebuild a review packet (or merge) underneath one.
_LIVE_DISPATCH_STATUSES = frozenset({"dispatched", "dispatch_pending", "manifest_written"})

# Local records carry no GitHub-side merge check, so the claim/review loop
# applies the same stale-claim timeout the remote stalled-review sweep uses.
_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES = 5


def _local_lane(self, *, now: Any = None, limit: int | None = None) -> _wf.CommandResult:
    """One pass of the local review/merge/rework lane.

    Runs inside ``loop()`` after ``dispatch_reviews``. Read-only no-op on a
    remote backend or in dry-run; each sub-phase carries its own
    ``*_enabled`` kill switch (``review_dispatch.enabled``,
    ``auto_merge.enabled``), matching the remote lane's gating so a config
    flip cannot strand in-flight claims.

    ``limit`` is the caller's explicit dispatch budget, threaded unchanged
    (pre-governor) from ``_loop_body``: an explicit 0 -- what
    ``fleet stop --drain`` forces through ``app.loop(0)`` -- suspends the
    rework launch below (issue #1716). ``_local_dispatch_rework`` is the
    lane's only ``dispatch_sessions`` caller with no ``*_enabled`` flag of
    its own, so the forced 0 is its only drain signal; every other launcher
    (``_local_dispatch_reviewers``) is already suppressed by
    ``apply_fleet_drain_config`` flipping ``review_dispatch.enabled``.
    ``None`` and positive limits leave rework dispatch unbudgeted, matching
    pre-drain behavior -- it only claims verdict-driven candidates.
    """
    if publishes_pull_requests(self.gh):
        return _wf.CommandResult(
            True,
            "backend publishes pull requests; local lane not applicable",
            {"local": False},
        )
    if self.dry_run:
        return _wf.CommandResult(
            True,
            "dry run",
            {"local": True, "dry_run": True},
        )
    resolved_now = now
    packets = self._local_review_packets()
    dispatched: dict[str, Any] = {"launched": [], "claimed": []}
    if self.config.review_dispatch.enabled:
        dispatched = self._local_dispatch_reviewers(now=resolved_now)
    merges: list[dict[str, Any]] = []
    if self.config.auto_merge.enabled:
        merges = self._local_merge_approved()
    launches_suspended = limit is not None and limit <= 0
    rework = (
        {"dispatched": [], "failed": [], "skipped": []}
        if launches_suspended
        else self._local_dispatch_rework()
    )
    return _wf.CommandResult(
        True,
        "local lane pass complete",
        {
            "local": True,
            "adopted": packets["adopted"],
            "packets": packets["packets"],
            "reviewers_launched": dispatched.get("launched", []),
            "merges": merges,
            "rework_dispatched": rework.get("dispatched", []),
            "rework_launches_suspended": launches_suspended,
        },
    )


def _local_branch_for_issue(self, issue_number: int, issue_entry: dict[str, Any]) -> str | None:
    """Resolve the worker branch for an adopted/parked local issue.

    Order: the recorded ``branch_name`` on the issue state entry (written at
    dispatch), then a worktree scan for the conventional
    ``<branch_prefix>-<n>-*`` name (covers records pre-dating the field).
    """
    recorded = issue_entry.get("branch_name")
    if isinstance(recorded, str) and recorded:
        return recorded
    prefix = self.config.dispatch.branch_prefix
    want_prefix = f"refs/heads/{prefix}-{issue_number}"
    for entry in list_worktrees(self.repo_root):
        branch = str(entry.get("branch") or "")
        if branch == want_prefix or branch.startswith(f"{want_prefix}-"):
            return branch.removeprefix("refs/heads/")
    return None


def _local_review_packets(self) -> dict[str, Any]:
    """Adopt parked local work into review records and (re)build packets.

    Two inputs:

    * ``agent:review-ready`` issues with no lane record yet -- a worker (fresh
      or rework) just died finished and parked its branch. A local record is
      minted keyed by the issue number.
    * Existing local records whose branch head moved past the packet head
      (rework landed new commits) or whose record never got a packet
      (``local_pending``), as long as no live dispatch may still be writing
      to the branch.

    Packet contents mirror the remote ``review()`` packet: ``pr.json`` (a
    synthesized PR-shaped dict), ``checks.json`` (empty -- no CI exists),
    ``diff.patch`` (``git diff <base>...<branch>``), ``review-prompt.md``
    rendered from ``review_local.md``, and a pending ``review-decision.json``
    stub stamped with the packet head (a prior terminal verdict pinned to a
    superseded head is archived then voided first, exactly as the remote lane
    does).
    """
    results: dict[str, Any] = {"adopted": [], "packets": [], "skipped": []}
    labels = self.config.labels

    parked = self.gh.issue_list(labels=[labels.review_ready], state="open")
    state = _wf.load_state_locked(self.paths.state_file)
    base_branch = local_base_branch(self.repo_root)

    # --- adoption: mint a lane record for parked issues not yet tracked ---
    adopt: list[tuple[int, str, str]] = []  # (issue_number, branch, head)
    for issue in parked:
        issue_number = int(issue["number"])
        existing = (state.get("prs") or {}).get(str(issue_number))
        if existing is not None:
            # A record exists. Local records continue below; a non-local
            # record under the issue number is a collision on a no-remote
            # backend -- never overwrite it silently.
            if not is_local_pr_record(existing):
                results["skipped"].append(
                    {"issue": issue_number, "reason": "non_local_record_exists"}
                )
            continue
        issue_entry = (state.get("issues") or {}).get(str(issue_number), {})
        branch = self._local_branch_for_issue(issue_number, issue_entry)
        head = branch_head_sha(self.repo_root, branch) if branch else None
        if not branch or head is None:
            with _wf.state_lock(self.paths.state_file):
                locked = _wf.load_state(self.paths.state_file)
                locked = self._record_event(
                    locked,
                    "local_review_adopt_failed",
                    {
                        "issue_number": issue_number,
                        "branch": branch,
                        "reason": "no_committed_branch",
                    },
                    level="warning",
                )
                self.write_gate.save_state(locked)
            results["skipped"].append({"issue": issue_number, "reason": "no_committed_branch"})
            continue
        adopt.append((issue_number, branch, head))
        results["adopted"].append({"issue": issue_number, "branch": branch, "head": head})

    if adopt:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number, branch, head in adopt:
                pr_key = str(issue_number)
                prior = (state.get("prs") or {}).get(pr_key) or {}
                if prior and not is_local_pr_record(prior):
                    continue  # a concurrent non-local writer owns the key
                state["prs"][pr_key] = {
                    **prior,
                    "number": issue_number,
                    "local": True,
                    "issue_number": issue_number,
                    "branch": branch,
                    "headRefName": branch,
                    "headRefOid": head,
                    "baseRefName": base_branch or "HEAD",
                    "title": (state.get("issues") or {}).get(pr_key, {}).get("title"),
                    "status": LOCAL_PENDING_STATUS,
                }
                state = self._record_event(
                    state,
                    "local_review_adopted",
                    {
                        "issue_number": issue_number,
                        "pr_number": issue_number,
                        "branch": branch,
                        "head_sha": head,
                    },
                )
            self.write_gate.save_state(state)

    # --- packet build/refresh for lane records ---
    state = _wf.load_state_locked(self.paths.state_file)
    # Liveness by sidecar, not by issue status: a freshly parked issue keeps
    # its dead worker's ``status="dispatched"`` (the salvage lane never
    # rewrites it), so keying liveness off the status string alone would
    # defer packet builds forever. ``dispatch_pending`` is live by
    # definition -- the worker launch is mid-flight and has no sidecar yet.
    live_sidecar_issues = {
        w.issue_number for w in iter_workers(self._layout.sessions_dir) if w.is_alive()
    }
    for pr_key, record in sorted(local_pr_records(state).items(), key=lambda kv: int(kv[0])):
        pr_number = int(pr_key)
        status = record.get("status")
        # Terminal/judgment states are skipped; everything else ("reviewing",
        # "request_changes", "approved", "rework_requested", "local_pending")
        # gets a rebuild check -- a moved head on an approved or
        # request_changes record means rework landed and the packet (and
        # verdict pinning) must be regenerated for the new head.
        if status in ("merged", "closed", "escalated", "blocked"):
            continue
        issue_number = int(record.get("issue_number") or pr_number)
        issue_entry = (state.get("issues") or {}).get(str(issue_number), {})
        branch = str(record.get("branch") or record.get("headRefName") or "")
        head = branch_head_sha(self.repo_root, branch) if branch else None
        if head is None:
            # The branch the record points at is gone (operator deleted it, or
            # adoption recorded a bogus name). Fail closed toward a human:
            # without the branch there is nothing to review or merge.
            with _wf.state_lock(self.paths.state_file):
                locked = _wf.load_state(self.paths.state_file)
                # Local-bound (not an inline "escalated" literal): the #750
                # guard requires the constant to reach a status write
                # indirectly in functions that escalate via the helper --
                # the issue half of this escalation is ``_escalate_issue``
                # just below; this is the PR-record half.
                status = "escalated"
                locked["prs"][pr_key] = {
                    **(locked["prs"].get(pr_key) or {}),
                    "status": status,
                    "escalation_reason": "local_branch_missing",
                }
                locked = _wf._escalate_issue(
                    locked,
                    issue_number,
                    reason="local_branch_missing",
                    reason_class="mechanical",
                )
                locked = self._record_event(
                    locked,
                    "local_review_packet_failed",
                    {
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "branch": branch,
                        "reason": "local_branch_missing",
                    },
                    level="warning",
                )
                self.write_gate.save_state(locked)
            self.write_gate.transition(
                self.gh, labels, issue_number, _wf._escalation_edge("escalated", "mechanical")
            )
            results["skipped"].append({"issue": issue_number, "reason": "local_branch_missing"})
            continue

        packet_head = self._read_packet_head_oid(pr_number)
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        prompt_path = pr_dir / "review-prompt.md"
        template_sha = prompt_template_digest("review_local.md", self.prompt_dirs)
        packet_meta: dict[str, Any] = {}
        try:
            raw_meta = json.loads((pr_dir / "pr.json").read_text(encoding="utf-8"))
            if isinstance(raw_meta, dict):
                packet_meta = raw_meta
        except (OSError, json.JSONDecodeError):
            pass
        issue_status = issue_entry.get("status")
        issue_live = issue_status == "dispatch_pending" or (
            issue_status in _LIVE_DISPATCH_STATUSES and issue_number in live_sidecar_issues
        )
        needs_build = (
            status == LOCAL_PENDING_STATUS
            or packet_head != head
            or not prompt_path.exists()
            or packet_meta.get("prompt_template_sha") != template_sha
        )
        if (
            needs_build
            and not issue_live
            and status == "approved"
            and packet_head is not None
            and packet_head != head
        ):
            # The merge gate's own base-sync merge moves the branch head
            # without changing the reviewed delta (``git diff base...branch``
            # is identical before and after). A deferred merge (dirty base
            # checkout) leaves exactly that shape, and rebuilding here would
            # void the approval and burn a full re-review cycle per deferral
            # pass. Patch-id equivalence -- the same ``reviewed_patch_id``
            # invariant the remote lane records on every verdict -- keeps the
            # approval valid across the sync merge while still re-reviewing
            # any genuinely new content (different patch-id -> rebuild).
            decision = self._review_decision(pr_number)
            reviewed_patch = decision.get("reviewed_patch_id")
            live_diff = branch_diff(
                self.repo_root,
                str(record.get("baseRefName") or base_branch or "HEAD"),
                branch,
            )
            if (
                reviewed_patch
                and live_diff is not None
                and _wf._calculate_patch_id(live_diff) == reviewed_patch
            ):
                needs_build = False
        if not needs_build or issue_live:
            continue
        outcome = self._local_build_packet(
            pr_number,
            issue_number,
            record,
            issue_entry,
            branch=branch,
            head=head,
            base_branch=base_branch,
        )
        if outcome is not None:
            results["packets"].append(outcome)
    return results


def _local_build_packet(
    self,
    pr_number: int,
    issue_number: int,
    record: dict[str, Any],
    issue_entry: dict[str, Any],
    *,
    branch: str,
    head: str,
    base_branch: str | None,
) -> dict[str, Any] | None:
    """Write one local review packet; returns a per-packet result dict."""
    base_ref = str(record.get("baseRefName") or base_branch or "HEAD")
    diff = branch_diff(self.repo_root, base_ref, branch)
    if diff is None:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state = self._record_event(
                state,
                "local_review_packet_failed",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "branch": branch,
                    "base_ref": base_ref,
                    "reason": "diff_failed",
                },
                level="warning",
            )
            self.write_gate.save_state(state)
        return {"issue": issue_number, "ok": False, "reason": "diff_failed"}

    pr_dir = self.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = pr_dir / "review-prompt.md"
    diff_path = pr_dir / "diff.patch"
    pr_json_path = pr_dir / "pr.json"
    decision_path = pr_dir / "review-decision.json"
    checks_path = pr_dir / "checks.json"
    template_sha = prompt_template_digest("review_local.md", self.prompt_dirs)
    diff_path.write_text(diff, encoding="utf-8")

    # Tier-1 test-adequacy hard gate (issue #179) -- identical rule to the
    # remote lane: a diff that adds behavior without test changes is rejected
    # deterministically before any reviewer is dispatched.
    test_adequacy_section = ""
    if self.config.test_adequacy.enabled:
        test_adequacy_verdict = check_test_adequacy(
            diff,
            local_pr_dict({**record, "headRefOid": head}),
            self.config.test_adequacy,
        )
        if not test_adequacy_verdict.ok:
            self.write_gate.transition(self.gh, self.config.labels, issue_number, "review_started")
            self.record_local_review(
                pr_number,
                "request_changes",
                summary=_wf.render_test_adequacy_summary(
                    test_adequacy_verdict, self.config.test_adequacy.exempt_marker
                ),
                reviewed_head=head,
                verdict_provenance="test_adequacy_auto_reject",
            )
            return {
                "issue": issue_number,
                "ok": True,
                "auto_reject": True,
                "head": head,
                "branch": branch,
            }
        test_adequacy_section = _wf.render_test_adequacy_section(
            test_adequacy_verdict.facts, test_adequacy_verdict.warnings
        )

    merged_warnings = check_operator_containment(self.repo_root, diff, pr_number)
    janitor_section = _wf._janitor_section(merged_warnings) if merged_warnings else ""
    static_probe_section = ""
    if self.config.coverage_probe.enabled:
        static_probe_section = _wf.render_static_probe_section(
            _wf.run_static_probe(diff, self.repo_root, self.config.coverage_probe)
        )
    over_cap_section = ""
    if self.config.review_dispatch.file_size_cap_lines > 0:
        over_cap_section = _wf.render_over_cap_section(
            _wf._over_cap_file_findings(
                diff, self.repo_root, self.config.review_dispatch.file_size_cap_lines
            )
        )
    attachment_budget_section = self._build_attachment_budget_section(diff, pr_number)
    existing_decision = self._review_decision(pr_number)
    prior_review_section = self._build_prior_review_section(pr_dir, existing_decision, head)

    prompt = self._render(
        "review_local.md",
        {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "issue_title": issue_entry.get("title") or record.get("title") or "",
            "branch_name": branch,
            "head_sha": head,
            "base_ref": base_ref,
            "pr_json_path": str(pr_json_path),
            "diff_path": str(diff_path),
            "diff_size_section": _wf._diff_size_section(
                diff, self.config.review_dispatch.diff_line_threshold, diff_path
            ),
            "janitor_section": janitor_section,
            "test_adequacy_section": test_adequacy_section,
            "static_probe_section": static_probe_section,
            "over_cap_section": over_cap_section,
            "attachment_budget_section": attachment_budget_section,
            "prior_review_section": prior_review_section,
        },
    )
    prompt_path.write_text(prompt, encoding="utf-8")

    # Compare-and-swap: the branch may have advanced while the packet was
    # being written. A moved head voids this packet -- the next pass
    # rebuilds. (Issue #1036's remote-lane invariant, same shape.)
    live_head = branch_head_sha(self.repo_root, branch)
    if live_head != head:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state = self._record_event(
                state,
                "review_packet_discarded_head_moved",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "packet_head_sha": head,
                    "live_head_sha": live_head,
                    "local": True,
                },
                level="warning",
            )
            self.write_gate.save_state(state)
        return {"issue": issue_number, "ok": False, "reason": "head_moved"}

    pr_json = {
        "number": pr_number,
        "local": True,
        "title": issue_entry.get("title") or record.get("title") or "",
        "url": "",
        "state": "OPEN",
        "headRefName": branch,
        "headRefOid": head,
        "baseRefName": base_ref,
        "isCrossRepository": False,
        "issue_number": issue_number,
        "labels": [],
        "prompt_template_sha": template_sha,
        "review_turn_cap_structure_multiplier": _wf.structure_turn_cap_multiplier(
            _wf._max_touched_file_line_count(diff, self.repo_root),
            self.config.review_dispatch,
        ),
        "review_turn_cap_max_touched_lines": _wf._max_touched_file_line_count(
            diff, self.repo_root
        ),
    }
    self._write_json(pr_json_path, pr_json)
    self._write_json(checks_path, [])

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        _existing = state["prs"].get(str(pr_number), {})
        _prior_attempt_last_head = _existing.get("review_dispatch_attempt_last_head")
        _fresh_dispatch_cycle = (
            _prior_attempt_last_head is None or _prior_attempt_last_head != head
        )
        _preserved_attempt_count = int(_existing.get("review_dispatch_attempt_count", 0))

        # Stale-verdict void: identical rule to the remote packet builder --
        # a terminal decision pinned to a superseded head is archived into
        # the rounds history and the flat file resets to a head-stamped
        # "pending" stub so state and file never disagree.
        live_decision = self._review_decision(pr_number)
        live_decision_value = live_decision.get("decision")
        live_reviewed_head_sha = live_decision.get("reviewed_head_sha")
        voided_stale_verdict = live_decision_value in (
            "approved",
            "request_changes",
            "blocked",
        ) and (live_reviewed_head_sha is None or live_reviewed_head_sha != head)
        if not decision_path.exists() or voided_stale_verdict:
            if voided_stale_verdict:
                record_decision(
                    pr_dir,
                    {
                        **live_decision,
                        "verdict_provenance": live_decision.get("verdict_provenance"),
                    },
                    None,
                    archive_round=True,
                )
            record_decision(
                pr_dir,
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "decision": "pending",
                    "summary": "",
                    "required_changes": [],
                    "reviewed_at": None,
                    "verdict_provenance": None,
                },
                head,
                archive_round=False,
            )

        state["prs"][str(pr_number)] = {
            **_existing,
            "number": pr_number,
            "local": True,
            "issue_number": issue_number,
            "branch": branch,
            "headRefName": branch,
            "headRefOid": head,
            "baseRefName": base_ref,
            "title": pr_json["title"],
            "prompt_path": str(prompt_path),
            "decision_path": str(decision_path),
            **({"decision": "pending"} if voided_stale_verdict else {}),
            "status": "reviewing",
            "janitor_ok": True,
            "janitor_failures": [],
            "janitor_warnings": list(merged_warnings),
            "consecutive_failed_merge_attempts": 0,
            "review_dispatch_attempt_count": (
                0 if _fresh_dispatch_cycle else _preserved_attempt_count
            ),
            "review_dispatch_attempt_last_head": head,
            "review_log_unreadable_streak": 0,
            "review_turn_limit_miss_streak": (
                0
                if _fresh_dispatch_cycle
                else int(_existing.get("review_turn_limit_miss_streak", 0))
            ),
            "no_op_rework_attempts": 0,
            "no_op_rework_attempts_last_head": None,
            "conflict_rework_attempts": 0,
            "conflict_rework_attempts_last_head": None,
        }
        _issue_entry = state["issues"].get(str(issue_number), {})
        state["issues"][str(issue_number)] = {
            **_issue_entry,
            "number": issue_number,
            "status": "reviewing",
            "branch_name": branch,
            "merge_alert": "OK",
        }
        state = self._record_event(
            state,
            "review_packet",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "local": True,
                "review_turn_cap_structure_multiplier": pr_json[
                    "review_turn_cap_structure_multiplier"
                ],
                "review_turn_cap_max_touched_lines": pr_json["review_turn_cap_max_touched_lines"],
            },
        )
        self.write_gate.save_state(state)

    # Label side effect is best-effort and outside the lock, matching the
    # remote lane: the durable packet above is the authority.
    self.write_gate.transition(self.gh, self.config.labels, issue_number, "review_started")
    return {"issue": issue_number, "ok": True, "head": head, "branch": branch}


def _local_dispatch_reviewers(self, *, now: Any = None) -> dict[str, Any]:
    """Claim and launch reviewers for lane records whose head is packetized.

    Mirrors ``dispatch_reviews``'s claim/launch/post-claim shape at a smaller
    scale: pending claim stamped under one lock, launches outside it, then a
    second lock upgrades the claim to dispatched/failed. The verdict reap,
    stale-claim sweep, and checkout reapers are the shared
    ``stalled_review_reap``/``misc_review_verdicts`` machinery, which treats a
    local record identically once the claim fields are set.
    """
    result: dict[str, Any] = {
        "claimed": [],
        "launched": [],
        "failed": [],
        "skipped": [],
        "escalated": [],
    }
    cfg = self.config.review_dispatch
    reviews_dir = self._layout.reviews_dir

    state = _wf.load_state_locked(self.paths.state_file)
    selected: list[int] = []
    escalated: list[int] = []
    in_flight = 0
    for pr_key, record in sorted(local_pr_records(state).items(), key=lambda kv: int(kv[0])):
        if record.get("status") != "reviewing":
            continue
        rd_status = record.get("review_dispatch_status")
        if rd_status in ("review_dispatch_pending", "review_dispatch_dispatched"):
            # A live claim is owned by the dispatched reviewer; a stale one is
            # rolled to review_dispatch_failed by the stalled-review sweep
            # (which runs ahead of this pass inside dispatch_reviews). Either
            # way it is not dispatchable here.
            in_flight += 1
            continue
        if rd_status == "review_dispatch_failed":
            failed_at = record.get("review_dispatch_failed_at")
            if not is_claim_stale(
                failed_at,
                timeout_minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
                now=now,
            ):
                continue
        decision = self._review_decision(int(pr_key))
        if decision.get("decision") in ("approved", "request_changes", "blocked"):
            continue  # verdict already recorded; no reviewer needed
        prompt_path = self.paths.prs / f"pr-{pr_key}" / "review-prompt.md"
        if not prompt_path.exists():
            continue
        attempt_count = int(record.get("review_dispatch_attempt_count", 0))
        if attempt_count >= cfg.max_review_dispatch_attempts:
            escalated.append(int(pr_key))
            continue
        selected.append(int(pr_key))

    if cfg.max_concurrent_reviews > 0:
        capacity = max(0, cfg.max_concurrent_reviews - in_flight)
        result["skipped"] = selected[capacity:]
        selected = selected[:capacity]

    for pr_number in escalated:
        record = state["prs"].get(str(pr_number), {})
        issue_number = int(record.get("issue_number") or pr_number)
        with _wf.state_lock(self.paths.state_file):
            locked = _wf.load_state(self.paths.state_file)
            # Local-bound "escalated" (issue #750 guard): the issue half of
            # this escalation runs through ``_escalate_issue`` just below;
            # this is the PR-record half.
            status = "escalated"
            locked["prs"][str(pr_number)] = {
                **(locked["prs"].get(str(pr_number)) or {}),
                "status": status,
                "escalation_reason": "max_review_dispatch_attempts_exceeded",
            }
            locked = _wf._escalate_issue(
                locked,
                issue_number,
                reason="max_review_dispatch_attempts_exceeded",
                reason_class="mechanical",
            )
            locked = self._record_event(
                locked,
                "review_dispatch_escalated",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "reason": "max_review_dispatch_attempts_exceeded",
                    "local": True,
                },
            )
            self.write_gate.save_state(locked)
        self.write_gate.transition(
            self.gh,
            self.config.labels,
            issue_number,
            _wf._escalation_edge("escalated", "mechanical"),
        )
        result["escalated"].append(pr_number)

    if not selected:
        return result

    claim_stamp = _wf.utc_now()
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        resolved_efforts: dict[int, str] = {}
        resolved_turn_caps: dict[int, int] = {}
        for pr_number in selected:
            record = state["prs"].get(str(pr_number), {})
            attempt_count = int(record.get("review_dispatch_attempt_count", 0))
            review_effort_used, review_effort_arm = resolve_review_effort(
                pr_number, self.config.reviewer, self.config.claude_code
            )
            resolved_efforts[pr_number] = review_effort_used
            resolved_cap = _wf.resolve_review_turn_cap(
                cfg.review_max_turns,
                self._read_packet_turn_cap_multiplier(pr_number),
                int(record.get("review_turn_limit_miss_streak", 0)),
                cfg,
            )
            resolved_turn_caps[pr_number] = resolved_cap
            state["prs"][str(pr_number)] = {
                **record,
                "number": pr_number,
                "review_dispatch_status": "review_dispatch_pending",
                "review_dispatch_pending_at": claim_stamp,
                "review_dispatched_at": None,
                "review_dispatch_failed_at": None,
                "reviewer_pid": None,
                "reviewer_process_start_time": None,
                "review_dispatch_attempt_count": attempt_count + 1,
                "review_turn_limit_summary_posted": False,
                "review_miss_summary_posted": False,
                "review_effort_arm": review_effort_arm,
                "review_effort_used": review_effort_used,
                "review_turn_cap": resolved_cap,
            }
        state = self._record_event(
            state,
            "review_dispatch_claim",
            {
                "pr_numbers": selected,
                "count": len(selected),
                "local": True,
            },
        )
        self.write_gate.save_state(state)

    reviewer_harness = self.config.reviewer.harness
    reviewer_adapter_settings = self._adapter_settings(adapter=reviewer_harness)
    reviewer_launcher = _wf._REVIEW_LAUNCHERS.get(reviewer_harness)
    launched: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    quota_hit = False
    for pr_number in selected:
        pr_dir = self.paths.prs / f"pr-{pr_number}"
        prompt_path = pr_dir / "review-prompt.md"
        try:
            record = (
                _wf.load_state_locked(self.paths.state_file).get("prs", {}).get(str(pr_number), {})
            )
            branch = str(record.get("branch") or record.get("headRefName") or "")
            head_sha = self._read_packet_head_oid(pr_number)
            issue_number = int(record.get("issue_number") or pr_number)
            if not prompt_path.exists():
                failed.append({"pr": pr_number, "error": "review-prompt.md not found"})
                continue
            if not branch:
                failed.append({"pr": pr_number, "error": "branch missing on record"})
                continue
            if not head_sha:
                failed.append({"pr": pr_number, "error": "packet head missing"})
                continue
            if reviewer_launcher is None:
                failed.append(
                    {
                        "pr": pr_number,
                        "error": f"unsupported reviewer harness: {reviewer_harness!r}",
                    }
                )
                continue
            prompt_text = prompt_path.read_text(encoding="utf-8")
            launch_record = reviewer_launcher(
                pr_number=pr_number,
                branch=branch,
                prompt_path=prompt_path,
                prompt_text=prompt_text,
                head_sha=head_sha,
                repo_root=self.repo_root,
                reviews_dir=reviews_dir,
                config=self.config,
                worker_env=reviewer_adapter_settings.worker_env,
                materialize_dirs=self.config.dispatch.materialize_dirs,
                resolved_review_effort=resolved_efforts.get(pr_number),
                max_turns_override=resolved_turn_caps.get(pr_number),
                model_override=self.config.reviewer.model or None,
                api_worker_config=reviewer_adapter_settings.api_worker_config,
            )
            if launch_record.error or launch_record.pid is None:
                error_text = launch_record.error or "launch returned no pid"
                if launch_record.error and (
                    _wf.match_throttle_tail(
                        launch_record.error,
                        self.config.runtime.throttle_error_markers,
                    )[0]
                    or _wf.match_quota_tail(
                        launch_record.error,
                        self.config.runtime.quota_error_markers,
                    )
                ):
                    quota_hit = True
                    break
                failed.append({"pr": pr_number, "error": error_text})
            else:
                launched.append(
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "pid": launch_record.pid,
                        "process_start_time": launch_record.process_start_time,
                    }
                )
        except (OSError, GitHubError, ValueError) as exc:
            failed.append({"pr": pr_number, "error": f"{type(exc).__name__}: {exc}"})

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        launched_prs = {x["pr"] for x in launched}
        failed_prs = {x["pr"] for x in failed}
        for pr_number in selected:
            record = state["prs"].get(str(pr_number), {})
            if pr_number in launched_prs:
                info = next(x for x in launched if x["pr"] == pr_number)
                state["prs"][str(pr_number)] = {
                    **record,
                    "number": pr_number,
                    "review_dispatch_status": "review_dispatch_dispatched",
                    "review_dispatched_at": _wf.utc_now(),
                    "review_dispatch_pending_at": None,
                    "review_dispatch_failed_at": None,
                    "review_dispatch_error": None,
                    "reviewer_pid": info["pid"],
                    "reviewer_process_start_time": info["process_start_time"],
                }
            elif pr_number in failed_prs:
                info = next(x for x in failed if x["pr"] == pr_number)
                state["prs"][str(pr_number)] = {
                    **record,
                    "number": pr_number,
                    "review_dispatch_status": "review_dispatch_failed",
                    "review_dispatch_failed_at": _wf.utc_now(),
                    "review_dispatch_pending_at": None,
                    "review_dispatched_at": None,
                    "reviewer_pid": None,
                    "reviewer_process_start_time": None,
                    "review_dispatch_error": info["error"],
                }
            else:
                # Quota rollback: release the claim and refund the attempt
                # (same rule as the remote lane -- a global quota condition
                # must not consume the per-record attempt budget).
                rolled_back = without_review_dispatch_claim(record)
                attempt_count = int(record.get("review_dispatch_attempt_count", 0))
                if attempt_count > 0:
                    rolled_back["review_dispatch_attempt_count"] = attempt_count - 1
                state["prs"][str(pr_number)] = rolled_back
        if launched:
            state = self._record_event(
                state,
                "review_dispatch",
                {
                    "pr_numbers": sorted(launched_prs),
                    "local": True,
                },
            )
        self.write_gate.save_state(state)

    result["claimed"] = selected
    result["launched"] = launched
    result["failed"] = failed
    result["quota_hit"] = quota_hit
    return result


def record_local_review(
    self,
    pr_number: int,
    decision: str,
    summary: str = "",
    summary_file: Path | None = None,
    comment: bool = False,
    reviewed_head: str | None = None,
    required_changes: Sequence[str] | None = None,
    session_metrics: dict[str, Any] | None = None,
    *,
    verdict_provenance: str,
    allow_stale_head: bool = False,
    verdict_source: str | None = None,
) -> _wf.CommandResult:
    """Record a review verdict for a local-lane record (no ``pr_view``).

    Local mirror of ``record_review``: same decision file (``prs/pr-<n>/
    review-decision.json`` + rounds archive via ``record_decision``), same
    packet-vs-live head pinning rules, same rework cap and escalation routing,
    same label edges. The remote-only seams are dropped: no external-findings
    ingestion (no PR comments exist to ingest), no ``before`` timestamp
    (``_commit_timestamp`` needs ``gh``), and the verdict comment posts to the
    issue via ``issue_comment`` instead of ``pr_comment``.
    """
    if decision not in ("approved", "request_changes", "blocked"):
        return _wf.CommandResult(False, f"invalid decision: {decision}", {})
    if verdict_provenance not in _wf.VERDICT_PROVENANCE_VALUES:
        return _wf.CommandResult(False, f"invalid verdict provenance: {verdict_provenance}", {})
    summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
    # Same rule as record_review: empty summary rejects, empty
    # required_changes never does -- it derives from the summary (or marks
    # the channel "vacuous") so a malformed verdict can never wedge the lane
    # in a silent re-review loop.
    if decision in {"request_changes", "blocked"} and not summary_text.strip():
        return _wf.CommandResult(
            False,
            f"--summary or --summary-file is required for decision '{decision}'",
            {},
        )
    effective_required_changes = (
        [str(item) for item in required_changes] if required_changes else []
    )
    findings_channel: str | None = None
    if decision in {"request_changes", "blocked"} and not effective_required_changes:
        if _wf._summary_is_vacuous(summary_text):
            findings_channel = "vacuous"
        else:
            effective_required_changes = [summary_text.strip()]
            findings_channel = "derived"

    decision, reclassified_human_call = reclassify_human_call_verdict(
        decision,
        effective_required_changes,
        markers=self.config.review.human_decision_markers,
        verdict_provenance=verdict_provenance,
    )

    pr_dir = self.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)

    guard_state = _wf.load_state_locked(self.paths.state_file)
    pr_state = (guard_state.get("prs") or {}).get(str(pr_number), {})
    if not is_local_pr_record(pr_state):
        return _wf.CommandResult(
            False,
            f"prs[{pr_number}] is not a local-lane record",
            {"pr": pr_number},
        )
    issue_number = pr_state.get("issue_number")
    issue_number = int(issue_number) if issue_number is not None else pr_number
    guard_issue_state = (guard_state.get("issues") or {}).get(str(issue_number), {})
    if pr_state.get("status") in ("merged", "closed"):
        return _wf.CommandResult(
            False,
            f"local record {pr_number} is already {pr_state.get('status')}; verdict not recorded",
            {"pr": pr_number, "issue": issue_number, "terminal": True},
        )
    if pr_state.get("status") == "escalated" or guard_issue_state.get("status") == "escalated":
        return _wf.CommandResult(
            False,
            f"local record {pr_number} is escalated; verdict not recorded "
            "(unescalate the issue first)",
            {"pr": pr_number, "issue": issue_number, "escalated": True},
        )

    # Head pinning: same packet-vs-live rule as record_review. The "live"
    # head is the branch tip resolved locally, not a fetched headRefOid.
    packet_head_sha = self._read_packet_head_oid(pr_number)
    branch = str(pr_state.get("branch") or pr_state.get("headRefName") or "")
    live_head_sha = branch_head_sha(self.repo_root, branch) if branch else None

    if reviewed_head is not None:
        if packet_head_sha is not None and reviewed_head == packet_head_sha:
            if (
                not allow_stale_head
                and live_head_sha is not None
                and packet_head_sha != live_head_sha
            ):
                return _wf.CommandResult(
                    False,
                    f"reviewed head ({reviewed_head}) matches packet head "
                    f"({packet_head_sha}) but live branch head has moved to "
                    f"({live_head_sha}); verdict not recorded — head moved "
                    "during build, will re-review next pass",
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "reason": "head_moved_during_build",
                        "packet_head_sha": packet_head_sha,
                        "live_head_sha": live_head_sha,
                    },
                )
            reviewed_head_sha = reviewed_head
            reviewed_head_source = "packet"
        elif live_head_sha is not None and reviewed_head == live_head_sha:
            reviewed_head_sha = reviewed_head
            reviewed_head_source = "live"
        else:
            options: list[str] = []
            if packet_head_sha is not None:
                options.append(f"packet head {packet_head_sha}")
            if live_head_sha is not None:
                options.append(f"live head {live_head_sha}")
            return _wf.CommandResult(
                False,
                f"reviewed-head {reviewed_head} does not match "
                f"{' or '.join(options) if options else 'any available head'}",
                {},
            )
    elif (
        packet_head_sha is not None
        and live_head_sha is not None
        and packet_head_sha != live_head_sha
    ):
        return _wf.CommandResult(
            False,
            f"review packet head ({packet_head_sha}) differs from live branch "
            f"head ({live_head_sha}); pass reviewed_head to choose the head "
            "the verdict applies to",
            {},
        )
    elif packet_head_sha is not None:
        reviewed_head_sha = packet_head_sha
        reviewed_head_source = "packet"
    elif live_head_sha is not None:
        reviewed_head_sha = live_head_sha
        reviewed_head_source = "live"
    else:
        return _wf.CommandResult(False, "no packet or live branch head available", {})

    diff = self._read_packet_diff(pr_number)
    if diff is None and branch:
        diff = branch_diff(
            self.repo_root,
            str(pr_state.get("baseRefName") or local_base_branch(self.repo_root) or "HEAD"),
            branch,
        )
    reviewed_patch_id = _wf._calculate_patch_id(diff) if diff else ""
    reviewed_signature = (
        _wf._diff_content_signature(diff) if diff else _wf.DiffContentSignature((), frozenset())
    )

    decision_payload: dict[str, Any] = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "local": True,
        "decision": decision,
        "summary": summary_text,
        "required_changes": effective_required_changes,
        "verdict_provenance": verdict_provenance,
        "reviewed_head_sha": reviewed_head_sha,
        "reviewed_head_source": reviewed_head_source,
        "reviewed_patch_id": reviewed_patch_id,
        "reviewed_changed_lines": list(reviewed_signature.changed_lines),
        "reviewed_changed_files": sorted(reviewed_signature.changed_files),
        "reviewed_has_binary": reviewed_signature.has_binary,
        "carried_forward_from": [],
        "reviewed_at": _wf.utc_now(),
    }
    if verdict_source is not None:
        decision_payload["verdict_source"] = verdict_source
    if findings_channel is not None:
        decision_payload["findings_channel"] = findings_channel
    provenance_caveat: str | None = None
    if decision in {"request_changes", "blocked"}:
        provenance_caveat = _wf.provenance_caveat_for(verdict_source)
        if provenance_caveat is not None:
            decision_payload["provenance_caveat"] = provenance_caveat
    if reclassified_human_call is not None:
        matched_item, matched_marker = reclassified_human_call
        decision_payload["human_call_reclassified_from"] = "request_changes"
        decision_payload["human_call_marker"] = {
            "item": matched_item,
            "marker": matched_marker,
        }

    decision_path = pr_dir / "review-decision.json"
    rounds_dir = pr_dir / "rounds"
    round_number = 0
    escalated = False
    request_changes_count = 0
    rework_path: str | None = None

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = (state.get("prs") or {}).get(str(pr_number), {})
        request_changes_count = int(pr_state.get("request_changes_count", 0))
        if decision == "request_changes":
            # Same counting rule as record_review: only a head that actually
            # advanced since the last verdict consumes rework budget. (The
            # remote lane's rescue tier is not mirrored here -- at the cap a
            # local record escalates directly, the documented fallback when
            # ``rescue.enabled`` is off.)
            head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
            if head_advanced:
                escalated = request_changes_count >= self.config.review.max_rework_cycles
                if not escalated:
                    request_changes_count += 1
        decision_payload["escalated"] = escalated
        pr_escalation_reason = (
            "max_rework_cycles_exceeded"
            if (decision == "request_changes" and escalated)
            else ("review_blocked" if decision == "blocked" else None)
        )

        # Persist the verdict BEFORE rendering the rework brief: the brief
        # reads review-decision.json itself (issue #632).
        if decision_path.exists():
            try:
                with decision_path.open("r", encoding="utf-8") as _handle:
                    _existing_decision = json.load(_handle)
                if isinstance(_existing_decision, dict) and isinstance(
                    _existing_decision.get("authorized_override"), dict
                ):
                    decision_payload["authorized_override"] = _existing_decision[
                        "authorized_override"
                    ]
            except (OSError, json.JSONDecodeError):
                pass
        round_number = _wf._next_round_number(rounds_dir, decision_payload)
        round_dir = rounds_dir / f"round-{round_number}"
        record_decision(pr_dir, decision_payload, None)
        if decision == "request_changes" and not escalated:
            rework_path = str(
                self._write_rework_prompt(
                    local_pr_dict(
                        {**pr_state, "headRefOid": reviewed_head_sha, "issue_number": issue_number}
                    ),
                    issue_number,
                    summary_text,
                )
            )
            _wf._write_text_atomic(
                round_dir / "rework-prompt.md",
                Path(rework_path).read_text(encoding="utf-8"),
            )
            _wf._write_text_atomic(
                round_dir / "rework-dispatch-note.txt",
                (pr_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8"),
            )

        state["prs"][str(pr_number)] = {
            **pr_state,
            "number": pr_number,
            "local": True,
            "issue_number": issue_number,
            "decision": decision,
            "decision_path": str(decision_path),
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_patch_id": reviewed_patch_id,
            "carried_forward_from": [],
            "request_changes_count": request_changes_count,
            "status": "escalated" if escalated else decision,
            "consecutive_failed_merge_attempts": 0,
            "review_dispatch_status": "review_dispatch_completed",
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
            "review_dispatch_attempt_count": 0,
            "review_log_unreadable_streak": 0,
            "review_turn_limit_miss_streak": 0,
            "review_session_metrics": (
                session_metrics
                if session_metrics is not None
                else pr_state.get("review_session_metrics")
            ),
            **(
                {"escalation_reason": pr_escalation_reason}
                if pr_escalation_reason is not None
                else {}
            ),
        }
        if decision == "request_changes":
            if not escalated:
                issue_entry = {
                    **(state["issues"].get(str(issue_number)) or {}),
                    "number": issue_number,
                    "status": "rework_requested",
                    "merge_alert": "OK",
                }
                _wf.clear_escalation(issue_entry)
                _wf.clear_escalation_on_issue_prs(state, issue_number)
                prior_patch_id = pr_state.get("reviewed_patch_id")
                content_advanced = bool(reviewed_patch_id) and reviewed_patch_id != prior_patch_id
                if content_advanced and verdict_provenance in _wf.NO_OP_RESET_PROVENANCES:
                    issue_entry["no_op_checkpoint_at"] = _wf.utc_now()
                state["issues"][str(issue_number)] = issue_entry
            else:
                state = _wf._escalate_issue(
                    state,
                    issue_number,
                    reason="max_rework_cycles_exceeded",
                    reason_class="mechanical",
                )
        elif decision == "approved":
            issue_entry = {
                **(state["issues"].get(str(issue_number)) or {}),
                "number": issue_number,
                "status": "approved",
                "merge_alert": "OK",
            }
            _wf.clear_escalation(issue_entry)
            _wf.clear_escalation_on_issue_prs(state, issue_number)
            state["issues"][str(issue_number)] = issue_entry
            state["issues"][str(issue_number)].pop("worker_pid", None)
            state["issues"][str(issue_number)].pop("worker_process_start_time", None)
        elif decision == "blocked":
            state = _wf._escalate_issue(
                state,
                issue_number,
                reason="review_blocked",
                reason_class="judgment",
                status="blocked",
            )
            state["issues"][str(issue_number)].pop("worker_pid", None)
            state["issues"][str(issue_number)].pop("worker_process_start_time", None)

        event_payload: dict[str, Any] = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "escalated": escalated,
            "local": True,
            "verdict_provenance": verdict_provenance,
            "summary": _wf._truncate_for_event(summary_text),
            "required_changes": _wf._truncate_for_event("\n".join(effective_required_changes)),
        }
        if findings_channel is not None:
            event_payload["findings_channel"] = findings_channel
        if session_metrics is not None:
            event_payload["session_metrics"] = session_metrics
        state = self._record_event(state, "record_review", event_payload)
        if reclassified_human_call is not None:
            matched_item, matched_marker = reclassified_human_call
            state = self._record_event(
                state,
                "review_decision_reclassified_blocked",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "original_decision": "request_changes",
                    "matched_item": matched_item,
                    "matched_marker": matched_marker,
                },
            )
        if findings_channel == "vacuous":
            state = self._record_event(
                state,
                "required_changes_vacuous",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "decision": decision,
                },
            )
        self.write_gate.save_state(state)

    # Label side effects are best-effort and outside the lock.
    label_error: dict[str, Any] | None = None
    if decision == "request_changes":
        issue_is_closed = False
        try:
            linked_issue = self.gh.issue_view(issue_number)
            issue_is_closed = str(linked_issue.get("state") or "OPEN").upper() == "CLOSED"
        except Exception:
            pass
        if issue_is_closed:
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                state = self._record_event(
                    state,
                    "rework_label_skipped_issue_closed",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "decision": decision,
                        "escalated": escalated,
                    },
                    level="warning",
                )
                self.write_gate.save_state(state)
        else:
            target = (
                _wf._escalation_edge("escalated", "mechanical")
                if escalated
                else "rework_requested"
            )
            transition_result = self.write_gate.transition(
                self.gh, self.config.labels, issue_number, target
            )
            if transition_result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": target,
                    "outcome": transition_result.outcome.value,
                    "add_failures": transition_result.add_failures,
                    "remove_failures": transition_result.remove_failures,
                }
    elif decision == "blocked":
        transition_result = self.write_gate.transition(
            self.gh,
            self.config.labels,
            issue_number,
            _wf._escalation_edge("blocked", "judgment"),
        )
        if transition_result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": "blocked",
                "outcome": transition_result.outcome.value,
                "add_failures": transition_result.add_failures,
                "remove_failures": transition_result.remove_failures,
            }
    elif decision == "approved":
        transition_result = self.write_gate.transition(
            self.gh, self.config.labels, issue_number, "review_approved"
        )
        if transition_result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": "review_approved",
                "outcome": transition_result.outcome.value,
                "add_failures": transition_result.add_failures,
                "remove_failures": transition_result.remove_failures,
            }

    # No PR exists to comment on; the issue is the operator-facing channel.
    if not escalated and (comment or self.config.review.post_verdict_comment):
        header = f"## Fleet review - round {round_number} - {decision}"
        body_parts = [header]
        if summary_text:
            body_parts.append(summary_text)
        if provenance_caveat is not None:
            body_parts.append(provenance_caveat)
        if effective_required_changes:
            body_parts.append(
                "### Required changes\n"
                + "\n".join(f"- {rc}" for rc in effective_required_changes)
            )
        comment_body = "\n\n".join(body_parts)
        try:
            body_path = pr_dir / "verdict-comment.md"
            body_path.write_text(comment_body, encoding="utf-8")
            self.gh.issue_comment(issue_number, body_path)
        except (GitHubError, OSError) as exc:
            if label_error is None:
                label_error = {"comment_error": str(exc)}
            else:
                label_error["comment_error"] = str(exc)

    source_note = f"head from {reviewed_head_source}"
    if escalated:
        message = (
            f"review recorded — rework cap ({self.config.review.max_rework_cycles}) "
            f"reached, escalated to human ({source_note})"
        )
    else:
        message = f"review recorded ({source_note})"
    if reclassified_human_call is not None:
        message += (
            " — request_changes reclassified as blocked: finding "
            f"{reclassified_human_call[0]!r} asks for a human/operator decision"
        )
    if label_error:
        message += f" (label update failed: {label_error.get('outcome', label_error)})"
    return _wf.CommandResult(
        True,
        message,
        {
            "pr": pr_number,
            "issue": issue_number,
            "decision": decision,
            "decision_path": str(decision_path),
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_head_source": reviewed_head_source,
            "live_head_sha": live_head_sha,
            "packet_head_sha": packet_head_sha,
            "rework_path": rework_path,
            "escalated": escalated,
            "request_changes_count": request_changes_count,
            "reclassified_from": (
                "request_changes" if reclassified_human_call is not None else None
            ),
            "label_error": label_error,
            "local": True,
        },
    )


def _local_merge_approved(self) -> list[dict[str, Any]]:
    """Merge approved local branches: base-sync, full suite, advance base.

    For every lane record whose recorded verdict is ``approved``:

    1. Merge the current base branch into the branch worktree
       (``_merge_update_rework_branch`` -- the same sync-merge the remote
       rework lane performs; a real conflict routes to rework, never to an
       operator).
    2. Run the full test suite inside the updated branch worktree
       (``resolve_test_commands``'s full-suite argv; failure routes to
       rework).
    3. Advance the base branch -- fast-forward when possible, ``--no-ff``
       otherwise (``merge_branch_into_base``).
    4. Remove the worker worktree with ``git worktree remove``, mark the
       record ``merged``, and close/label the issue via the ``merged`` edge.
    """
    results: list[dict[str, Any]] = []
    worktrees_dir = self._layout.worktrees
    state = _wf.load_state_locked(self.paths.state_file)

    for pr_key, record in sorted(local_pr_records(state).items(), key=lambda kv: int(kv[0])):
        if record.get("status") != "approved":
            continue
        pr_number = int(pr_key)
        issue_number = int(record.get("issue_number") or pr_number)
        entry: dict[str, Any] = {"issue": issue_number, "pr": pr_number}
        branch = str(record.get("branch") or record.get("headRefName") or "")
        base_ref = str(record.get("baseRefName") or local_base_branch(self.repo_root) or "HEAD")
        decision = self._review_decision(pr_number)
        reviewed_head = decision.get("reviewed_head_sha")
        live_head = branch_head_sha(self.repo_root, branch) if branch else None

        # The merge gate only ever merges the exact head the reviewer
        # approved -- a verdict pinned to a superseded head is voided by the
        # packet builder's stale-verdict reset, but double-check here anyway
        # so a torn state can never authorize the wrong head.
        if live_head is None:
            entry["outcome"] = "error"
            entry["detail"] = f"branch {branch!r} does not resolve"
            results.append(entry)
            continue
        if reviewed_head and reviewed_head != live_head:
            # Head moved after approval. The common cause is this gate's own
            # base-sync merge (step 1 below) followed by a deferral or a
            # crash -- the reviewed delta is unchanged, only the head moved.
            # Honour the approval when the live diff still matches the
            # recorded patch-id; any genuinely new content (a rogue push,
            # rework landing mid-gate) falls through to the packet phase's
            # rebuild + re-review.
            reviewed_patch = decision.get("reviewed_patch_id")
            live_diff = branch_diff(self.repo_root, base_ref, branch)
            if not (
                reviewed_patch
                and live_diff is not None
                and _wf._calculate_patch_id(live_diff) == reviewed_patch
            ):
                entry["outcome"] = "skipped_head_moved"
                results.append(entry)
                continue

        worktree_path = ensure_branch_worktree(self.repo_root, branch, worktrees_dir)
        if worktree_path is None:
            entry["outcome"] = "error"
            entry["detail"] = f"could not attach a worktree for {branch!r}"
            results.append(entry)
            continue

        # (1) merge base into the branch worktree.
        try:
            conflict = _merge_update_rework_branch(
                self.repo_root,
                worktree_path,
                branch,
                base_ref,
                self.config.dispatch.injected_paths,
                self.config.dispatch.materialize_dirs,
            )
        except Exception as exc:  # ReworkBranchConflictError / RuntimeError
            entry["outcome"] = "error"
            entry["detail"] = f"base sync merge failed: {exc}"
            results.append(entry)
            self._local_merge_error_escalate(
                pr_number, issue_number, branch, f"base sync merge failed: {exc}"
            )
            continue
        if conflict is not None:
            entry["outcome"] = "conflict"
            entry["conflicted_paths"] = list(conflict.conflicted_files)
            self._local_route_merge_rework(
                pr_number,
                issue_number,
                record,
                decision,
                reason="merge_conflict",
                note=(
                    f"Merging the base branch {base_ref!r} into {branch!r} "
                    f"conflicted ({len(conflict.conflicted_files)} path(s)). "
                    "Merge the base into your branch and resolve the conflicts. "
                    "The code changes are already approved; do not re-litigate "
                    "the review."
                ),
            )
            results.append(entry)
            continue

        # (2) full suite in the updated branch worktree.
        argv = suite_command_argv(self.config.dispatch.test_command, self.repo_root)
        if argv is None:
            # No resolvable suite command: fail closed. A lane that cannot
            # verify must not merge -- escalate mechanically so a human can
            # either configure a runner or merge by hand.
            entry["outcome"] = "error"
            entry["detail"] = "no test command resolvable for full-suite gate"
            results.append(entry)
            self._local_merge_error_escalate(
                pr_number,
                issue_number,
                branch,
                "no test command resolvable for the local full-suite gate",
            )
            continue
        suite = run_full_suite(worktree_path, argv)
        suite_head = branch_head_sha(self.repo_root, branch)
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][pr_key] = {
                **(state["prs"].get(pr_key) or {}),
                "local_suite_ok": suite.ok,
                "local_suite_argv": list(suite.argv),
                "local_suite_at": _wf.utc_now(),
                "local_suite_head": suite_head,
            }
            state = self._record_event(
                state,
                "local_suite_result",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "ok": suite.ok,
                    "argv": list(suite.argv),
                    "returncode": suite.returncode,
                    "head_sha": suite_head,
                    "tail": _wf._truncate_for_event(suite.tail),
                },
            )
            self.write_gate.save_state(state)
        if not suite.ok:
            entry["outcome"] = "suite_failed"
            entry["returncode"] = suite.returncode
            self._local_route_merge_rework(
                pr_number,
                issue_number,
                record,
                decision,
                reason="suite_failed",
                note=(
                    "The full test suite failed on your branch after the base "
                    f"merge ({' '.join(suite.argv)}, exit {suite.returncode}). "
                    "The code changes are already approved; fix the failing "
                    "tests. Tail of the suite output:\n\n"
                    f"```\n{suite.tail}\n```"
                ),
            )
            results.append(entry)
            continue

        # (3) advance the base.
        outcome = merge_branch_into_base(
            self.repo_root, base_ref, branch, worktrees_dir=worktrees_dir
        )
        if outcome.status == "deferred":
            entry["outcome"] = "deferred"
            entry["detail"] = outcome.detail
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                state = self._record_event(
                    state,
                    "local_merge_deferred",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "detail": outcome.detail,
                    },
                )
                self.write_gate.save_state(state)
            results.append(entry)
            continue
        if outcome.status == "conflict":
            entry["outcome"] = "conflict"
            entry["conflicted_paths"] = list(outcome.conflicted_paths)
            self._local_route_merge_rework(
                pr_number,
                issue_number,
                record,
                decision,
                reason="merge_conflict",
                note=(
                    f"Merging {branch!r} into the base branch {base_ref!r} "
                    f"conflicted ({len(outcome.conflicted_paths)} path(s)). "
                    "Merge the base into your branch and resolve the "
                    "conflicts. The code changes are already approved; do not "
                    "re-litigate the review."
                ),
            )
            results.append(entry)
            continue
        if outcome.status == "error":
            entry["outcome"] = "error"
            entry["detail"] = outcome.detail
            self._local_merge_error_escalate(pr_number, issue_number, branch, outcome.detail)
            results.append(entry)
            continue

        # (4) merged (or already contained): terminal bookkeeping.
        merged_sha = outcome.merged_sha or suite_head or live_head
        entry["outcome"] = "merged" if outcome.status == "merged" else "already_merged"
        entry["fast_forward"] = outcome.fast_forward
        entry["merged_sha"] = merged_sha
        # force=True like every other worker-worktree teardown: the branch
        # content is committed to base by construction, so remaining
        # uncommitted material is worker scratch (.venv, build output,
        # worker-tmp) -- a plain remove would refuse on a real .venv dir and
        # leak a worktree per merged issue. ``branch`` also deletes the
        # merged branch ref when auto_merge.delete_branch is configured,
        # matching the remote lane's post-merge cleanup.
        # Issue #1476: a worktree outside the managed dir is a foreign
        # checkout (``ensure_branch_worktree`` returns the registered path
        # wherever the branch happens to live) — it belongs to whoever
        # created it and is never removed; the branch ref can't be deleted
        # while checked out anyway.
        if contains(worktrees_dir, worktree_path):
            remove_worktree(
                self.repo_root,
                worktree_path,
                force=True,
                branch=(branch if self.config.auto_merge.delete_branch else None),
            )
        remove_review_checkout(self.repo_root, pr_number, reviews_dir=self._layout.reviews_dir)
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            pr_state = state["prs"].get(pr_key, {})
            state["prs"][pr_key] = {
                **without_review_dispatch_claim(pr_state),
                "number": pr_number,
                "local": True,
                "status": "merged",
                "merged_at": _wf.utc_now(),
                "merged_sha": merged_sha,
                "merge_method": "ff" if outcome.fast_forward else "no-ff",
            }
            issue_entry = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = _wf._merged_issue_fields(
                issue_entry, issue_number
            )
            state = self._record_event(
                state,
                "merge_succeeded",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "local": True,
                    "fast_forward": outcome.fast_forward,
                    "merged_sha": merged_sha,
                    "detail": outcome.detail,
                },
            )
            self.write_gate.save_state(state)
        transition_result = self.write_gate.transition(
            self.gh, self.config.labels, issue_number, "merged"
        )
        if transition_result.outcome != TransitionOutcome.APPLIED:
            entry["label_error"] = {
                "edge": "merged",
                "outcome": transition_result.outcome.value,
            }
        # The issue itself closes on merge, same as a merged remote PR.
        try:
            self.gh.close_issue(issue_number)
        except Exception:
            entry["close_error"] = True
        results.append(entry)
    return results


def _local_route_merge_rework(
    self,
    pr_number: int,
    issue_number: int,
    record: dict[str, Any],
    decision: dict[str, Any],
    *,
    reason: str,
    note: str,
) -> None:
    """Route a merge-gate failure (conflict or suite failure) to rework."""
    pr = local_pr_dict(record)
    label_error = self._route_to_rework(
        pr,
        issue_number,
        decision,
        note,
        (
            "merge_conflict_rework_requested"
            if reason == "merge_conflict"
            else "local_suite_failed"
        ),
        extra_payload={"local": True, "reason": reason},
        extra_state={"local_merge_rework_reason": reason},
    )
    if label_error:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            issue_entry = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = {
                **issue_entry,
                "number": issue_number,
                "label_error": label_error,
            }
            self.write_gate.save_state(state)


def _local_merge_error_escalate(
    self, pr_number: int, issue_number: int, branch: str, detail: str
) -> None:
    """Escalate a merge-gate infrastructure failure (never a silent merge)."""
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        # Local-bound "escalated" (issue #750 guard): the issue half runs
        # through ``_escalate_issue`` just below; this is the PR-record half.
        status = "escalated"
        state["prs"][str(pr_number)] = {
            **(state["prs"].get(str(pr_number)) or {}),
            "status": status,
            "escalation_reason": "local_merge_error",
        }
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason="local_merge_error",
            reason_class="mechanical",
            issue_extra={"merge_alert": detail},
        )
        state = self._record_event(
            state,
            "local_merge_failed",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "branch": branch,
                "detail": _wf._truncate_for_event(detail),
            },
            level="error",
        )
        self.write_gate.save_state(state)
    self.write_gate.transition(
        self.gh,
        self.config.labels,
        issue_number,
        _wf._escalation_edge("escalated", "mechanical"),
    )


def _local_dispatch_rework(self) -> dict[str, Any]:
    """Dispatch rework workers for local records in ``rework_requested``.

    Remote ``dispatch_rework`` selects candidates from ``gh.pr_list()``,
    which is empty on a no-remote backend -- so the rework queue is read
    straight from the lane records instead. The launch itself reuses
    ``dispatch_sessions`` + ``SessionRequest(rework=True)``: worktree
    attachment, the base sync-merge, and the conflict notice are all
    origin-tolerant already.
    """
    result: dict[str, Any] = {"dispatched": [], "failed": [], "skipped": []}
    state = _wf.load_state_locked(self.paths.state_file)
    candidates: list[int] = []
    for pr_key, record in sorted(local_pr_records(state).items(), key=lambda kv: int(kv[0])):
        issue_number = int(record.get("issue_number") or pr_key)
        issue_entry = (state.get("issues") or {}).get(str(issue_number), {})
        # Remote dispatch_rework selects on the ISSUE status (the PR record
        # keeps the verdict value "request_changes"); the merge-gate rework
        # route (_route_to_rework) sets both, so keying the issue covers both
        # rework producers.
        # A rework worker already in flight carries a live-dispatch issue
        # status ("dispatched"/"dispatch_pending"), so this filter alone
        # excludes it; a stale claim is recovered by the dead-worker reap.
        if issue_entry.get("status") != "rework_requested":
            continue
        prompt_path = self.paths.prs / f"pr-{pr_key}" / "rework-prompt.md"
        if not prompt_path.exists():
            result["skipped"].append({"issue": issue_number, "reason": "missing_rework_prompt"})
            continue
        candidates.append(int(pr_key))

    if not candidates:
        return result

    requests: list[SessionRequest] = []
    request_issues: dict[int, int] = {}
    for pr_number in candidates:
        record = (state.get("prs") or {}).get(str(pr_number), {})
        issue_number = int(record.get("issue_number") or pr_number)
        issue_entry = (state.get("issues") or {}).get(str(issue_number), {})
        requests.append(
            SessionRequest(
                issue_number=issue_number,
                issue_title=str(issue_entry.get("title") or record.get("title") or ""),
                prompt_path=self.paths.prs / f"pr-{pr_number}" / "rework-prompt.md",
                branch_name=str(record.get("branch") or record.get("headRefName") or ""),
                rework=True,
            )
        )
        request_issues[pr_number] = issue_number

    # Claim before launch (dispatch_pending) so a crash between claim and
    # launch leaves a stale-claim-recoverable record, not a silent wedge.
    claim_stamp = _wf.utc_now()
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        for pr_number in candidates:
            issue_number = request_issues[pr_number]
            issue_entry = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = {
                **issue_entry,
                "number": issue_number,
                "status": "dispatch_pending",
                "dispatch_pending_at": claim_stamp,
            }
        self.write_gate.save_state(state)

    dispatch_results = _wf.dispatch_sessions(
        self.repo_root,
        self._layout.session_manifest,
        self._layout.session_results,
        self._adapter_settings(),
        requests,
    )
    successful = {r.issue_number for r in dispatch_results if r.ok}
    failed_map = {
        r.issue_number: (r.error or "dispatch failed") for r in dispatch_results if not r.ok
    }

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        for request in requests:
            issue_number = request.issue_number
            ok = issue_number in successful
            issue_entry = state["issues"].get(str(issue_number), {})
            entry = {
                **issue_entry,
                "number": issue_number,
                "branch_name": request.branch_name,
                "prompt_path": str(request.prompt_path),
                "status": "dispatched" if ok else "rework_requested",
                "dispatched_at": _wf.utc_now() if ok else None,
            }
            entry.pop("dispatch_pending_at", None)
            state["issues"][str(issue_number)] = entry
            if not ok:
                result["failed"].append(
                    {"issue": issue_number, "error": failed_map.get(issue_number)}
                )
            else:
                result["dispatched"].append(issue_number)
        state = self._record_event(
            state,
            "dispatch_rework",
            {
                "issue_numbers": sorted(successful),
                "failed_issue_numbers": sorted(failed_map),
                "local": True,
            },
        )
        self.write_gate.save_state(state)

    for issue_number in sorted(successful):
        self.write_gate.transition(self.gh, self.config.labels, issue_number, "rework_dispatched")
    return result
