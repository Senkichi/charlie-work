"""Instrumentation and loop-orchestration delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, leaf L05 (issue #1636; design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp`` (``self`` binds through the descriptor
protocol exactly as a lexical method did).

Names reached through ``_wf.`` (module-object seam, design Section 3.1 rule 2,
#1627):

* Tier D -- patched on ``charlie_work.workflow`` by the suite, so the moved body
  must keep intercepting those patches: ``load_state_locked``, ``run_preflight``,
  ``utc_now``.
* ``charlie_work.workflow`` module-level definitions: ``CommandResult`` (class),
  ``sink_census`` / ``summarize_loop_errors`` (free functions),
  ``_infra_blocked_window`` (the single cross-pass window dict, still mutated by
  ``workflow``'s escalation path -- one shared instance is load-bearing), and the
  ``UNAUTHORIZED_MERGE_{ACK,BASELINE,DETECTED}_KEY`` state-key constants.
* ``_queue_sync_merge_covered`` -- the ``queue_sync_coverage`` free function is
  re-exported by ``charlie_work.workflow`` under the *same name* as this moved
  method; referencing it bare here would bind the module-level ``def`` below
  (self-recursion), so it is reached through ``_wf._queue_sync_merge_covered``
  (the deliberate ``# noqa: F401`` re-export in ``workflow``).

All other free names are imported directly from their defining module (a
two-form patch census confirms no test patches any of them on
``charlie_work.workflow``).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import charlie_work.workflow as _wf
from charlie_work import status_snapshot
from charlie_work.attachment_budget_prompt import (
    ATTACHMENT_BUDGET_CLAUSE as _ATTACHMENT_BUDGET_CLAUSE,
)
from charlie_work.attachment_contracts import baseline as attachment_baseline
from charlie_work.instrumentation import (
    correlation_context,
    log_event,
    query_events,
    record_loop_pass,
)
from charlie_work.module_map import build_module_map
from charlie_work.preflight import PreflightPaths, emit_preflight_refusal
from charlie_work.queue_sync_coverage import _QueueSyncCoverageResult
from charlie_work.state import (
    DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS,
    ESCALATION_REASON_CLASS_BY_EVENT_KIND,
)
from charlie_work.supervise import orchestrator_root


def _backfill_missing_reason_classes(self, state: dict[str, Any]) -> dict[str, Any]:
    """One-way migration that gives legacy escalations a ``reason_class`` (issue #797).

    Escalations created before the ``reason_class`` field shipped carry no
    ``reason_class`` at all, so the de-escalation sweep cannot see them.
    ``events.db`` retains the original escalation-transition event for most
    of them; for each ``escalated``/``blocked`` issue that still lacks
    ``reason_class``, this helper looks up the most recent such event and
    derives the class from its kind.

    The mapping is deliberately conservative: only kinds that unambiguously
    indicate a process failure are mapped to ``"mechanical"``. Kinds in
    ``DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS`` (e.g.
    ``janitor_rework_escalated``, a deliberately-preserved forensic record)
    are left untouched, so they stay terminal. No event, or an unknown
    kind, also stays fail-closed.

    Backfilling only writes ``reason_class``. It does not clear the
    escalation, does not reset ``auto_deescalation_count`` or any
    per-mechanism attempt counter, and emits exactly one
    ``deescalation_reason_class_backfilled`` event per issue so the
    migration is auditable. It is idempotent: once ``reason_class`` is
    present the issue is skipped.
    """
    all_source_kinds: frozenset[str] = (
        frozenset(ESCALATION_REASON_CLASS_BY_EVENT_KIND)
        | DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS
    )
    for issue_key, issue in list(state.get("issues", {}).items()):
        if not isinstance(issue, dict):
            continue
        if issue.get("status") not in ("escalated", "blocked"):
            continue
        if "reason_class" in issue:
            continue
        if not issue_key.isdigit():
            continue
        issue_number = int(issue_key)

        events = query_events(
            self.paths.state_file,
            issue_number=issue_number,
        )
        escalation_events = [e for e in events if e["kind"] in all_source_kinds]
        if not escalation_events:
            continue
        latest = escalation_events[-1]
        reason_class = ESCALATION_REASON_CLASS_BY_EVENT_KIND.get(latest["kind"])
        if reason_class is None:
            continue

        issue = {
            **issue,
            "number": issue_number,
            "reason_class": reason_class,
        }
        state = {
            **state,
            "issues": {**state.get("issues", {}), issue_key: issue},
        }
        state = self.write_gate.record_event(
            state,
            "deescalation_reason_class_backfilled",
            {
                "issue_number": issue_number,
                "from_event_kind": latest["kind"],
                "from_event_ts": latest["ts"],
                "reason_class": reason_class,
            },
        )
    return state


def _build_module_map_value(self, issue_number: int) -> str:
    """Derive the module-map section from the live tree (issue #1444).

    Single point of enforcement for the fail-soft contract: a repo layout
    ``build_module_map`` cannot parse yields an empty string (an omitted
    section) plus a ``worker_module_map_failed`` warning event logged to
    events.db, never a dispatch failure. The worker loses placement
    steering for this one packet; the next packet rebuilds the map against
    the then-current tree.

    ``log_event`` (not ``self._record_event``) is used because
    ``_write_worker_prompt`` runs outside a state-lock context -- the
    caller (``intake`` / ``_dispatch_impl``) gathers network results and
    builds prompts before taking the state lock for label/status writes.
    """
    package_dir = self.repo_root / "src" / "charlie_work"
    src_root = self.repo_root / "src"
    try:
        return build_module_map(package_dir, src_root)
    except (OSError, SyntaxError, ValueError) as exc:
        log_event(
            self.paths.state_file,
            "worker_module_map_failed",
            {
                "issue_number": issue_number,
                "error": f"{type(exc).__name__}: {exc}",
            },
            repo=self.repo_root.name,
            level="warning",
        )
        return ""


def _build_attachment_budget_value(self, issue_number: int) -> str:
    """Derive the attachment-budget dispatch clause (issue #1460).

    Marker-file presence is the ONLY activation switch: a repo that has
    never run `attachment_contracts baseline` (no `.attachment-budgets.json`
    at the repo root) gets an empty clause, silently -- this feature is
    opt-in per repo, not a default every worker prompt must carry.

    When the marker file IS present, its structural validity is checked
    via `baseline.load` (not just existence) before the clause is
    emitted, so a hand-corrupted or tampered baseline never ships
    placement instructions that reference a broken `check-file` command.
    Fail-soft, mirroring `_build_module_map_value`'s contract exactly: a
    `TamperError` yields an empty clause (omitted section) plus a
    `worker_attachment_budget_failed` warning event, never a dispatch
    failure.

    `log_event` (not `self._record_event`) is used for the same reason
    `_build_module_map_value` uses it: `_write_worker_prompt` runs
    outside a state-lock context, so the module-level, lock-free
    primitive is the only one available here.
    """
    marker_path = self.repo_root / attachment_baseline.BASELINE_FILENAME
    if not marker_path.is_file():
        return ""
    try:
        attachment_baseline.load(marker_path)
    except (attachment_baseline.TamperError, OSError, ValueError) as exc:
        # ``TamperError`` covers baseline's own structural checks
        # (schema version, entry shape, duplicate keys); ``ValueError``
        # additionally covers a bare ``json.JSONDecodeError`` (a subclass
        # of ``ValueError``) from genuinely malformed JSON, which
        # ``baseline.loads`` does not wrap into a ``TamperError`` itself.
        # ``OSError`` guards a read race between the ``is_file()`` check
        # above and this read. Mirrors ``hook_entry._resolve_mode``'s
        # except clause for the same reason.
        log_event(
            self.paths.state_file,
            "worker_attachment_budget_failed",
            {
                "issue_number": issue_number,
                "error": str(exc),
            },
            repo=self.repo_root.name,
            level="warning",
        )
        return ""
    return _ATTACHMENT_BUDGET_CLAUSE


def _queue_sync_merge_covered(
    self,
    pr: dict[str, Any],
    reviewed_head_sha: str | None,
    live_head_sha: str | None,
) -> _QueueSyncCoverageResult:
    """Thin delegate to ``queue_sync_coverage._queue_sync_merge_covered``.

    See that function's docstring for the full four-condition predicate
    (issue #1194) and the retry/indeterminate contract (sibling-repo PRs
    #1888, #1916, #1904, #1895).

    ``queue_sync_coverage.py`` is deliberately pure (issue #1264's
    per-module WriteGate raw-primitive-call ratchet -- a new module
    starts at baseline 0, so a raw ``log_event`` call moved there would
    read as new growth even though it was already present, and already
    counted, in this file). This wrapper therefore emits the
    ``unauthorized_merge_queue_sync_covered`` audit event itself on a
    covered result, exactly where that raw call already lived (and was
    already counted) before the extraction.
    """
    queue_bot_login = self.config.auto_merge.queue_bot_login
    result = _wf._queue_sync_merge_covered(
        self.gh,
        queue_bot_login,
        pr,
        reviewed_head_sha,
        live_head_sha,
    )
    if result.covered:
        log_event(
            self.paths.state_file,
            "unauthorized_merge_queue_sync_covered",
            {
                "pr": pr.get("number"),
                "reviewed_head_sha": reviewed_head_sha,
                "live_head_sha": live_head_sha,
                "sync_parent": result.sync_parent,
                "pre_merge_base": result.pre_merge_base,
                "queue_bot_login": queue_bot_login,
            },
        )
    return result


def _reconcile_stranded_verdicts(self) -> list[dict[str, Any]]:
    """Ingest completed on-disk verdicts that were never recorded into state.

    Issue #736: a PR may have a complete ``review-decision.json`` on disk
    (written by ``record_review``) but no ``decision`` key in its state
    record -- the state update was lost (e.g. a concurrent writer clobbered
    it, or the verdict file was written by a path that didn't update
    state). Without this reconciliation, the verdict is stranded: the PR
    stays in ``status=reviewing`` forever because the only ingestion path
    that reads ``review-decision.json`` (``_reap_review_verdicts``) requires
    a live reviewer sidecar, and the stale-claim recovery branch in
    ``_detect_and_handle_stalled_reviews`` explicitly *skips* PRs whose
    on-disk verdict is already completed (issue #734's
    ``decision_already_recorded`` skip). That skip is correct for the
    stale-claim sweep -- a completed verdict is not a stale claim -- but
    it leaves the PR with no path to ingest the verdict at all.

    This method closes that gap by scanning every non-terminal PR record
    for a ``decision_path`` pointing to an existing file with a completed
    verdict (``approved``/``request_changes``/``blocked``) that is not
    reflected in ``state.decision``, and ingesting it via
    ``record_review`` so the normal verdict → rework/merge/blocked route
    can take over. It runs in ``dispatch_reviews`` ahead of the
    ``review_dispatch.enabled`` gate (issue #868), so it executes even
    when review dispatch is disabled fleet-wide -- the exact condition
    that stranded PR 1343.

    Returns a list of per-PR result dicts for diagnostics.
    """
    if self.dry_run:
        return []
    results: list[dict[str, Any]] = []
    state = _wf.load_state_locked(self.paths.state_file)
    for pr_key, pr_state in list(state.get("prs", {}).items()):
        if not isinstance(pr_state, dict) or not pr_key.isdigit():
            continue
        pr_status = pr_state.get("status")
        # Skip lifecycle-terminal PRs -- a merged/closed PR needs no
        # verdict ingestion.
        if pr_status in ("merged", "closed"):
            continue
        decision_path_str = pr_state.get("decision_path")
        if not decision_path_str:
            continue
        decision_path = Path(decision_path_str)
        if not decision_path.exists():
            continue
        try:
            with decision_path.open("r", encoding="utf-8") as handle:
                decision_data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(decision_data, dict):
            continue
        on_disk_decision = decision_data.get("decision")
        if on_disk_decision not in ("approved", "request_changes", "blocked"):
            continue
        on_disk_reviewed_head_sha = decision_data.get("reviewed_head_sha")
        # Skip only if state already ingested THIS on-disk verdict -- the
        # same decision recorded against the same reviewed head. A
        # prior-round terminal decision with a DIFFERENT
        # reviewed_head_sha than the on-disk file means a later round's
        # verdict was written to disk but never ingested into state (the
        # #736 stranded shape re-opened for round-2+); it must still be
        # reconciled. Skipping solely because ``state.decision`` holds
        # any terminal value from a prior round would strand every
        # round-2+ verdict whose state write was lost.
        #
        # ``record_review``'s own #467/#1072 head-drift guard is the
        # safety net for stale heads: it refuses to pin a verdict to a
        # head that no longer matches the packet/live head for automated
        # callers (``allow_stale_head`` defaults to ``False``). So this
        # skip's job is only to avoid re-processing a verdict already
        # ingested for the SAME on-disk decision, not to skip every PR
        # that has ever had a decision recorded.
        recorded_decision = pr_state.get("decision")
        recorded_reviewed_head_sha = pr_state.get("reviewed_head_sha")
        if (
            recorded_decision in ("approved", "request_changes", "blocked")
            and recorded_decision == on_disk_decision
            and recorded_reviewed_head_sha == on_disk_reviewed_head_sha
        ):
            continue
        # Stranded verdict found: the on-disk file carries a completed
        # decision not reflected in state -- either state has no
        # ``decision`` key at all, or it carries a prior-round decision
        # recorded against a different ``reviewed_head_sha``. Ingest it
        # via ``record_review`` so the full state-machine transition (PR
        # status, issue status, rework prompt, label transition) runs
        # exactly as if the verdict were freshly rendered.
        #
        # ``allow_stale_head`` is left at its default ``False``. In the
        # actual #736 bug scenario (state write lost, head unchanged),
        # ``record_review``'s #467/#1072 guard never fires — the packet
        # head and live head agree — so the flag would be a no-op there.
        # The flag only changes behavior when the live PR head has
        # advanced past the packet head (new commits landed while the PR
        # sat 'reviewing'), which is exactly the case where the verdict
        # is legitimately stale and must NOT be silently pinned to the
        # old diff. ``allow_stale_head=True`` is reserved for the
        # operator ``charlie verdict`` CLI (issue #467's explicit-choice
        # design); this is an automated caller and uses the default.
        # A head-drifted stranded verdict is correctly refused this pass
        # and left for the stale-claim sweep / a fresh review dispatch.
        pr_number = int(pr_key)
        record_result = self.record_review(
            pr_number,
            on_disk_decision,
            summary=decision_data.get("summary", ""),
            reviewed_head=decision_data.get("reviewed_head_sha"),
            required_changes=decision_data.get("required_changes"),
            verdict_provenance="stranded_reconciliation",
        )
        entry: dict[str, Any] = {
            "pr_number": pr_number,
            "decision": on_disk_decision,
            "ok": record_result.ok,
            "message": record_result.message,
        }
        results.append(entry)
        if record_result.ok:
            log_event(
                self.paths.state_file,
                "review_verdict_reconciled",
                {
                    "pr_number": pr_number,
                    "issue_number": decision_data.get("issue_number"),
                    "decision": on_disk_decision,
                },
            )
        else:
            log_event(
                self.paths.state_file,
                "review_verdict_reconcile_failed",
                {
                    "pr_number": pr_number,
                    "decision": on_disk_decision,
                    "reason": record_result.message,
                },
                level="warning",
            )
    return results


def ensure_labels(self) -> _wf.CommandResult:
    """Automatic startup label ensure (issue #1339).

    Runs the same idempotent LabelConfig-derived ensure as
    ``bootstrap_labels``, but records the outcome as an event so a
    missing-label drift surfaces in ``events.db`` without an operator
    having to run ``charlie bootstrap-labels`` by hand. Never raises:
    any unexpected exception is recorded as an event and swallowed, so a
    label-ensure failure can never block the supervisor/loop startup it
    runs from.
    """
    try:
        result = self._ensure_labels_core()
    except Exception as exc:  # noqa: BLE001 — ensure must never block startup
        # _ensure_labels_core itself does not raise, but a GitHub
        # implementation that raises from label_create/label_list would
        # otherwise crash the caller. Record and degrade to a failure
        # result instead of propagating.
        log_event(
            self.paths.state_file,
            "label_ensure_failed",
            {"error": f"{type(exc).__name__}: {exc}", "labels": list(self.config.labels.all)},
            repo=self.repo_root.name,
            level="error",
        )
        return _wf.CommandResult(
            False,
            f"label ensure error: {exc}",
            {"labels": list(self.config.labels.all), "missing": None},
        )
    missing = result.data.get("missing")
    if result.ok:
        log_event(
            self.paths.state_file,
            "label_ensure_ok",
            {"labels": result.data.get("labels", []), "missing": []},
            repo=self.repo_root.name,
            level="info",
        )
    elif missing is None:
        # Verification itself failed (e.g. label_list raised); the ensure
        # could not be confirmed. Surface as an error event.
        log_event(
            self.paths.state_file,
            "label_ensure_failed",
            {"message": result.message, "labels": result.data.get("labels", [])},
            repo=self.repo_root.name,
            level="error",
        )
    else:
        log_event(
            self.paths.state_file,
            "label_ensure_incomplete",
            {"missing": missing, "message": result.message},
            repo=self.repo_root.name,
            level="warning",
        )
    return result


def tripwire_status(self) -> _wf.CommandResult:
    """Report the live unauthorized-merge tripwire state without re-running detection.

    The consumer for ``unauthorized_merge_detected`` (issue #933) and for
    ``unauthorized_merge_check_skipped`` (issue #940). Reading a finding out of
    ``state.json`` is deliberately cheaper and safer than re-detecting: it needs
    no ``gh`` call, so it works while the API is down and cannot itself arm the
    baseline as a side effect. The skip counts come from ``events.db``, which is
    local SQLite and so preserves that property.

    Pending findings are the ones that are still pinning ``ok=False``:
    detected and not acknowledged.

    The two halves answer different questions and both are needed: ``pending``
    says whether the tripwire *found* anything, ``skipped_count`` says whether it
    *ran*. A pass that failed open reports no findings, so without the second
    number an unrun control is indistinguishable from a clean one.
    """
    state = _wf.load_state_locked(self.paths.state_file)

    detected = state.get(_wf.UNAUTHORIZED_MERGE_DETECTED_KEY)
    detected = detected if isinstance(detected, dict) else {}
    acked = state.get(_wf.UNAUTHORIZED_MERGE_ACK_KEY)
    acked = acked if isinstance(acked, dict) else {}
    baseline = state.get(_wf.UNAUTHORIZED_MERGE_BASELINE_KEY)
    baseline = baseline if isinstance(baseline, dict) else {}

    def _as_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    pending: list[dict[str, Any]] = []
    for raw_pr, detail in sorted(detected.items(), key=lambda kv: _as_int(kv[0]) or 0):
        if raw_pr in acked:
            continue
        entry = {"pr": _as_int(raw_pr)}
        if isinstance(detail, dict):
            entry.update(
                {
                    "detected_at": detail.get("detected_at"),
                    "issue": detail.get("issue"),
                    "head": detail.get("head"),
                    "decision": detail.get("decision"),
                    "review_dispatch_enabled": detail.get("review_dispatch_enabled"),
                }
            )
        pending.append(entry)

    # How often did the check fail open and not run at all (issue #940)? Without
    # this the command answers "are there findings?" but not "did the control
    # actually look?", and those differ precisely when it matters: a skipped pass
    # reports no findings, so a blinded tripwire and a clean one are identical in
    # every field above.
    #
    # Reading events.db here does not compromise the no-``gh`` property this method
    # is built around -- it is local SQLite, so the command still works during an
    # API outage. That is the point rather than a tolerated cost: a skip is *caused
    # by* a GitHubError, so an operator running this during a gh outage is exactly
    # the person who needs to know the tripwire has stopped looking.
    #
    # Bound the count by ``armed_at`` instead of a fixed lookback: it is real state
    # rather than a constant needing revision, and skips recorded before arming say
    # nothing about the armed control's coverage. ``skipped_window_start`` reports
    # the bound actually used, so a caller never has to guess whether a count is
    # since-arming or all-time.
    armed_at = baseline.get("armed_at")
    skipped_window_start = armed_at if isinstance(armed_at, str) and armed_at else None
    skips = query_events(
        self.paths.state_file,
        kind="unauthorized_merge_check_skipped",
        since=skipped_window_start,
    )
    last_skip = skips[-1] if skips else None
    last_skip_payload = last_skip.get("payload") if isinstance(last_skip, dict) else None
    last_skipped_reason = (
        last_skip_payload.get("reason") if isinstance(last_skip_payload, dict) else None
    )

    if not baseline:
        message = "unauthorized-merge tripwire is NOT ARMED (no baseline recorded yet)"
    elif pending:
        message = (
            f"{len(pending)} pending unauthorized-merge finding(s) pinning ok=False: "
            + ", ".join(f"#{p['pr']}" for p in pending)
        )
    else:
        message = "no pending unauthorized-merge findings"

    if last_skip is not None:
        # Appended rather than substituted: "no pending findings" stays true, and
        # the whole problem is that it reads as "all clear" on its own.
        # Conditioned on ``last_skip`` rather than ``skips`` -- equivalent, since one
        # is the other's last element, but it narrows the subscript below instead of
        # relying on a reader to re-derive that they cannot disagree.
        window = f" since {skipped_window_start}" if skipped_window_start else ""
        message += (
            f" (warning: the check did not run on {len(skips)} pass(es){window}; "
            f"most recent {last_skip['ts']})"
        )

    return _wf.CommandResult(
        True,
        message,
        {
            "armed_at": armed_at,
            "baselined_count": len(baseline.get("pre_existing_prs") or []),
            "pending": pending,
            "pending_count": len(pending),
            "acknowledged_count": len(acked),
            "detected_count": len(detected),
            "skipped_count": len(skips),
            "skipped_window_start": skipped_window_start,
            "last_skipped_at": last_skip["ts"] if last_skip else None,
            "last_skipped_reason": last_skipped_reason,
            "state_file": str(self.paths.state_file),
        },
    )


def _loop_impl(
    self, limit: int | None, *, merge: bool | None, now: datetime | None = None
) -> _wf.CommandResult:
    # Issue #1363: preflight gate. Runs BEFORE `loop_started` is recorded
    # -- a fatal host-precondition failure (disk_floor, venv_identity)
    # must refuse the pass with zero partial work, not half-run it and
    # fail midway with a generic error. Non-fatal failures (clock_sanity,
    # config_freshness) emit a warning event and the pass proceeds
    # unmodified -- on a healthy host this adds no event at all (AC3).
    preflight_result = _wf.run_preflight(
        PreflightPaths(
            repo_root=self.repo_root,
            state_dir=self.paths.root,
            # venv_identity must anchor on the orchestrator's OWN
            # checkout (the code/venv this process is actually running),
            # never on `self.repo_root` -- that is the *target* repo this
            # particular pass processes, which varies per registered
            # repo even though every one of them runs from the same
            # single orchestrator install. See PreflightPaths.
            # orchestrator_root's docstring for why conflating the two
            # would make venv_identity fatally misfire on every repo
            # other than the orchestrator's own.
            orchestrator_root=orchestrator_root(),
        ),
        self.config.runtime.preflight,
        now=now,
        config_sources=self.config.sources,
        known_config_mtimes=self._preflight_config_mtimes,
    )
    for check in preflight_result.non_fatal_failures:
        kind = (
            "preflight_config_stale" if check.name == "config_freshness" else "preflight_warning"
        )
        log_event(
            self.paths.state_file,
            kind,
            {"check": check.name, "detail": check.detail},
            repo=self.repo_root.name,
            level="warning",
        )
    if not preflight_result.ok:
        fatal_check = preflight_result.fatal_failures[0]
        emit_preflight_refusal(self.paths.state_file, fatal_check, repo=self.repo_root.name)
        return _wf.CommandResult(
            False,
            f"preflight refused pass: {fatal_check.name}: {fatal_check.detail}",
            {
                "pass_skipped": True,
                "reason": "preflight_refused",
                "check": fatal_check.name,
                "detail": fatal_check.detail,
            },
        )

    # merge=False runs the full pass (intake, dispatch, reviews, readiness
    # evaluation + labels) but skips the actual `gh pr merge` — for
    # operators sequencing same-surface PR cascades by hand, where the
    # pr_list (newest-first) merge order would land PRs in the wrong order.
    loop_start = time.monotonic()
    with correlation_context() as cid:
        start_ts = _wf.utc_now()
        log_event(
            self.paths.state_file,
            "loop_started",
            {"limit": limit, "merge": merge},
            repo=self.repo_root.name,
            correlation_id=cid,
        )
        record_loop_pass(self.paths.state_file, cid, start_ts)
        # Issue #1083: snapshot the ``agent:human-needed`` sink before the
        # pass mutates anything, so arrivals (after - before) and the
        # post-pass population are a census diff, not an event-kind
        # allow-list that would have to be maintained in lockstep with
        # every new escalation call site. Read-only; ``load_state_locked``
        # (not raw ``load_state``) so the snapshot holds the advisory lock
        # -- the lint guard in ``test_no_unlocked_load_state_in_production_code``
        # flags any bare ``load_state`` outside a ``state_lock`` context.
        sink_before = _wf.sink_census(_wf.load_state_locked(self.paths.state_file))
        result = self._loop_body(limit, merge=merge, now=now)
        elapsed = time.monotonic() - loop_start
        sink_after = _wf.sink_census(_wf.load_state_locked(self.paths.state_file))
        sink_arrivals = len(sink_after - sink_before)
        sink_population = len(sink_after)
        # Sweep clears are the one signal that cannot be derived from the
        # census diff: a departure may be an operator ``unescalate``, a
        # merge, or a close, none of which are sweep drainage.
        # ``deescalation_cleared`` has a single emitter
        # (``_deescalate_mechanical_issue``) and shares this pass's
        # correlation ID, so a count by (kind, correlation_id) is the
        # exact sweep-clear count with no allow-list to drift. Queried
        # before ``loop_completed`` is emitted so this event is not
        # self-counted.
        sink_clears = len(
            query_events(
                self.paths.state_file,
                kind="deescalation_cleared",
                correlation_id=cid,
            )
        )
        # Issue #1383: reset the cross-pass infra_blocked window when a
        # pass reviews at least one PR and finds none infra-blocked --
        # the fleet-wide infra condition has cleared, so the
        # consecutive-pass counter starts fresh next time. Queried by
        # this pass's correlation ID (same pattern as sink_clears
        # above) so the count is exact for this pass with no
        # allow-list to drift.
        #
        # Round-2 #1383 review: the reset must NOT fire merely because
        # zero ``check_infra_blocked`` events were emitted -- that is
        # also true when the pass reviewed no PRs at all (empty queue,
        # ``limit=0`` with no candidates, ...). Resetting on an
        # idle pass during a live outage silently clears
        # ``consecutive_passes``/``last_escalation`` and can prevent
        # the AC3 persistence escalation from ever firing. Gate on
        # ``reviews`` (PRs actually reviewed this pass) being
        # non-empty so the reset means "we looked and the coast was
        # clear", not "we didn't look".
        infra_blocked_this_pass = len(
            query_events(
                self.paths.state_file,
                kind="check_infra_blocked",
                correlation_id=cid,
            )
        )
        reviewed_this_pass = len(result.data.get("reviews", []))
        if reviewed_this_pass > 0 and infra_blocked_this_pass == 0:
            repo_key = str(self.repo_root)
            window = _wf._infra_blocked_window.get(repo_key)
            if window is not None and window.get("consecutive_passes", 0) > 0:
                _wf._infra_blocked_window[repo_key] = {
                    "consecutive_passes": 0,
                    "last_escalation": None,
                    "last_pass_cid": None,
                }
        # Issue #1314 item 3: operator-queue depth gauge. Emitted after
        # ``_loop_body`` so the gauge reflects post-pass state (the
        # de-escalation sweep inside ``_loop_body`` may have cleared some
        # issues), and before ``loop_completed`` so the gauge event is
        # not self-counted by any post-pass event query. Shares this
        # pass's correlation ID so the depth reading is attributable to
        # the same pass as the sink census above.
        self._maybe_emit_operator_queue_depth()
        log_event(
            self.paths.state_file,
            "loop_completed",
            {
                "ok": result.ok,
                "message": result.message,
                "elapsed_seconds": round(elapsed, 2),
                "error_count": len(result.data.get("errors", [])),
                # Issue #933: the count alone made "loop completed with 1 PR
                # error(s)" undiagnosable — two independent sessions misread a
                # 21-pass ok=False streak as healthy because no stored artifact
                # named the PR. Bounded; see summarize_loop_errors.
                **_wf.summarize_loop_errors(result.data.get("errors", [])),
                # Issue #1083: autonomy is never reported without its drop
                # rate. ``sink_arrivals`` is the first-class failure metric
                # (issues dropped to a human this pass); ``sink_clears`` is
                # the automated drainage; ``sink_population`` is the
                # point-in-time census of parked work. A rise in arrivals
                # without a matching rise in clears is the signature of a
                # growing sink, and the ratio of clears to population is
                # the drainage rate #1093's mirror-clear should move.
                "sink_population": sink_population,
                "sink_arrivals": sink_arrivals,
                "sink_clears": sink_clears,
                # Issue #1132: parked PRs were invisible — the
                # early-continue emitted zero events, so "PR untouched
                # for days" was unattributable from events.db. Surface
                # the count and PR numbers so a parked PR is diagnosable.
                "parked_prs": result.data.get("parked_prs", []),
                "parked_prs_count": len(result.data.get("parked_prs", [])),
            },
            repo=self.repo_root.name,
            correlation_id=cid,
        )
        record_loop_pass(
            self.paths.state_file,
            cid,
            start_ts,
            completed_at=_wf.utc_now(),
            ok=result.ok,
            elapsed_seconds=round(elapsed, 2),
            error_count=len(result.data.get("errors", [])),
            merge_count=len(result.data.get("merges", [])),
            review_count=len(result.data.get("reviews", [])),
            sink_population=sink_population,
            sink_arrivals=sink_arrivals,
            sink_clears=sink_clears,
        )
        if not self.dry_run:
            status_snapshot.write_status_snapshot(self)
        return result
