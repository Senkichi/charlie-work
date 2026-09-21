"""CI-checks-findings free-function family (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: two functions that render the
review packet's CI-status section from already-fetched check data
(``_ci_status_section``, ``_non_required_check_findings``); four functions
that detect a stalled dispatch cadence
(``_backlog_is_non_empty``, ``_latest_non_empty_dispatch``,
``_parse_iso_ts``, ``check_dispatch_staleness``); and two functions that
turn a failing required check's GitHub annotations into
``required_changes`` entries (``_annotation_to_required_change``,
``_required_changes_from_checks``).

Issue #1769: the dispatch-staleness baseline (``_latest_non_empty_dispatch``)
used to scan the most recent 100 ``dispatch`` events.db rows for the last
non-empty one -- bounded by event COUNT, which a sustained real stall could
exhaust, silently flipping a genuine alarm to ``no_baseline``. It now reads
the durable ``dispatch_cadence`` marker in the already-loaded state dict
(``state.record_non_empty_dispatch``/``last_non_empty_dispatch``) instead,
so this sub-cluster no longer touches events.db at all -- it reads only the
in-memory state dict its caller already holds.

These 8 names are NOT one call-graph-connected cluster -- they are three
mutually disconnected sub-clusters (no call edges between them, and the
dispatch-staleness sub-cluster imports nothing from ``.checks``). They
are combined into a single module because issue #1283's own binding text
names them together and because they share a destination theme
(workflow-side consumers of CI check data), not because of any
code-level cohesion signal -- disclosed as a judgment call, not a
natural grouping.

``workflow.py`` re-exports every symbol here via a facade import block
(mirroring ``config.py``'s ``RunnerAllocationConfig`` re-export pattern
and this repo's own ``dispatch_selection.py``/``escalation.py``/
``verdict_parsing.py``/``rework_prompts.py`` precedents), so existing
import paths and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .checks import (
    _CheckClassification,
    _classify_check_run,
    _is_failing_run,
    summarize_checks,
)
from .collect_only_gate import COLLECT_ONLY_GATE_CHECK_NAME, parse_exemption_log_marker
from .config import DispatchConfig
from .github import _job_id_from_link, label_names
from .state import is_dispatch_stale_alert_due, last_non_empty_dispatch


def _ci_status_section(
    checks: list[dict[str, Any]] | None,
    required: tuple[str, ...],
    checks_json_path: Path,
) -> str:
    """Render the $ci_status_section packet block from already-fetched CI data.

    ``run_janitor`` deterministically verifies required checks BEFORE a review
    packet is ever generated: a definitive required-check failure short-
    circuits ``review()`` long before this function is reached (see the
    ``janitor_blocked`` branch). So a reviewer re-reading ``checks.json`` to
    re-confirm what the gate already verified is pure token waste. This
    section states that verification result inline instead, while still
    surfacing everything the gate does NOT resolve: unfetchable CI data, an
    unconfigured required-check list (the gate is a no-op in that case),
    still-pending required checks, and failing non-required/informational
    checks (the gate never blocks on those).

    Pure and I/O-free like ``_janitor_section`` — safe to call every pass.
    """
    if checks is None:
        return (
            "CI status could not be fetched by the orchestrator (`gh` failure). "
            f"Do not assume checks are green — inspect `{checks_json_path}` "
            "directly if CI status matters to this review.\n"
        )

    if not required:
        return (
            "No required checks are configured for this repo, so CI status was "
            "not deterministically verified before dispatch. Inspect "
            f"`{checks_json_path}` if CI status is relevant to your review.\n"
        )

    summary = summarize_checks(checks, required)
    lines: list[str] = []
    if summary.passed:
        lines.append(
            f"Required check(s) passing — verified deterministically by the "
            f"orchestrator before dispatch: {', '.join(summary.passed)}. Do "
            "not spend turns re-inspecting these."
        )
    if summary.pending:
        lines.append(
            f"Required check(s) still pending, not yet confirmed: {', '.join(summary.pending)}."
        )
    lines.append(f"`checks.json` is available at `{checks_json_path}` if a specific doubt arises.")

    non_required_failing, non_required_cancelled = _non_required_check_findings(checks, required)
    if non_required_failing:
        lines.append(
            "Non-required/informational check(s) currently failing (the "
            "janitor gate does not block on these — weigh them yourself): "
            + ", ".join(non_required_failing)
        )
    if non_required_cancelled:
        lines.append(
            "Non-required/informational check(s) cancelled (often infra-transient, "
            "not necessarily a code failure — weigh them yourself): "
            + ", ".join(non_required_cancelled)
        )

    return "\n".join(lines) + "\n"


def _non_required_check_findings(
    checks: list[dict[str, Any]], required: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify non-required checks into (failing, cancelled) name lists.

    Reuses ``_classify_check_run`` from ``checks.py`` so the
    pass/pending/empty/SKIPPED carve-out, cancelled split, and infra/fail
    distinction are enforced in one place. The output shape is different here
    because non-required checks are advisory only:

    - PASS, PENDING, EMPTY, and SKIPPED are ignored (non-outcomes).
    - CANCELLED is reported separately (worded as "cancelled," never
      "failing") since it is frequently an infra hiccup.
    - FAIL and INFRA (INFRA_FAILURE, TIMED_OUT) are reported as failing in
      this informational context, unlike in ``summarize_checks`` where INFRA
      blocks merge as an infrastructure failure.

    Multiple runs sharing a name use worst-of semantics, mirroring
    ``summarize_checks``.
    """
    required_set = set(required)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        name = str(check.get("name") or "")
        if not name or name in required_set:
            continue
        by_name.setdefault(name, []).append(check)

    failing: list[str] = []
    cancelled: list[str] = []
    for name, runs in by_name.items():
        name_failed = False
        name_cancelled = False
        for check in runs:
            classification = _classify_check_run(check)
            if classification in {
                _CheckClassification.PASS,
                _CheckClassification.PENDING,
                _CheckClassification.EMPTY,
                _CheckClassification.SKIPPED,
            }:
                continue
            if classification == _CheckClassification.CANCELLED:
                name_cancelled = True
                continue
            # Everything else is a genuine failure in this non-required,
            # informational context: FAILURE, INFRA_FAILURE, TIMED_OUT, and
            # any other unrecognized terminal state.
            name_failed = True
        if name_failed:
            failing.append(name)
        elif name_cancelled:
            cancelled.append(name)

    return tuple(sorted(failing)), tuple(sorted(cancelled))


def _backlog_is_non_empty(reachability: dict[str, Any]) -> bool:
    """Return True only when the unfiltered backlog is observed and non-empty.

    ``observed: False`` (e.g. a failed or empty ``gh issue_list``) must never be
    treated as "backlog empty" -- that would make a silent outage look healthy.
    """
    if not reachability.get("observed"):
        return False
    return reachability.get("open_total", 0) > 0


def _latest_non_empty_dispatch(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the durable ``{"ts", "issue_numbers"}`` baseline dispatch reading.

    Issue #1769: this used to scan the most recent 100 ``dispatch`` rows in
    events.db for the newest one with non-empty ``issue_numbers`` -- bounded
    by event COUNT, not by "does a non-empty one exist". During a sustained
    stall the empty-payload passes (which happen on every healthy
    zero-dispatch pass too, so they can't be used to measure cadence
    themselves) accumulate past that bound and the lookup goes silent
    (``no_baseline``) even though the real stall is ongoing -- exactly when
    the alarm matters most, and reintroducing an unbounded scan would bring
    back the full-table-scan cost the 100-row bound existed to avoid.

    Delegates to ``state.last_non_empty_dispatch``, which reads a durable
    marker set by ``state.record_non_empty_dispatch`` whenever a dispatch
    pass actually launches issues: an O(1) dict read that cannot fall out of
    a rolling window no matter how long the stall runs. ``state`` is the
    caller's already-loaded, in-memory state dict -- this function performs
    no I/O of its own.
    """
    return last_non_empty_dispatch(state)


def _parse_iso_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def check_dispatch_staleness(
    state: dict[str, Any],
    config: DispatchConfig,
    backlog_reachability: dict[str, Any],
    *,
    recent_issue_numbers: list[int] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Issue #946: detect when a non-empty backlog has had no dispatch for too long.

    Reads the caller's already-loaded state dict for the durable baseline
    marker (see ``_latest_non_empty_dispatch``) of the most recent dispatch
    pass whose ``issue_numbers`` was non-empty. When that reading is older
    than ``config.dispatch_staleness_minutes`` and the unfiltered backlog is
    observed to be non-empty, returns a stale diagnostic. Otherwise returns
    a no-op diagnostic with ``stale: False``. This function performs no I/O
    and mutates nothing -- it is a pure read of ``state`` and
    ``backlog_reachability``; the caller is responsible for persisting the
    baseline marker (``state.record_non_empty_dispatch``) and for acting on
    ``should_emit`` (``state.arm_dispatch_stale_alert`` /
    ``clear_dispatch_stale_alert``).

    ``backlog_reachability`` must come from ``classify_backlog_reachability``.
    The ``observed: False`` case is treated as "unknown", not "empty", so a
    failed unfiltered fetch does not silently suppress the alarm.

    ``recent_issue_numbers`` lets callers short-circuit with the current pass:
    if this pass itself dispatched issues, the most recent non-empty dispatch is
    now and the check returns ``stale: False``.

    Issue #1110: ``stale`` does not fire when every ready issue is blocked by an
    open dependency (``dispatchable == 0`` and ``blocked_by_open_dependency > 0``).
    A deliberately sequenced cohort tail (e.g. #887/#888 blocked by an open
    #886) is permanently -- and correctly -- unselectable by dispatch, so a
    cadence alarm for it is a false positive that pattern-matches the #944
    four-day stall this detector exists to catch. The #944 detection stays
    intact: when ``dispatchable == 0`` and ``blocked_by_open_dependency == 0``
    (no ready issues at all, e.g. all ``missing_ready``), the alarm still fires.

    Issue #1769: ``should_emit`` answers a second, orthogonal question --
    given that ``stale`` is true, should the caller actually record a
    ``dispatch_stale`` event *this pass*? It is edge-triggered (true on
    stall onset) plus a bounded low-rate reminder (true again once
    ``config.dispatch_staleness_minutes`` have elapsed since the last
    emitted alert), reusing the staleness threshold itself as the reminder
    cadence rather than introducing a second config knob for it. It is
    always ``False`` when ``stale`` is ``False`` -- there is nothing to
    (re-)emit for a healthy pass.
    """
    result: dict[str, Any] = {
        "stale": False,
        "should_emit": False,
        "last_dispatch_at": None,
        "last_dispatch_issue_numbers": None,
        "age_seconds": None,
        "threshold_seconds": None,
        "backlog_observed": bool(backlog_reachability.get("observed", False)),
        "backlog_open_total": int(backlog_reachability.get("open_total", 0) or 0),
        # Issue #1110: surface the post-dependency-gate candidate count so a
        # reader of the staleness diagnostic can distinguish "nothing ready"
        # (the #944 case) from "ready but blocked" (the #1110 case) without
        # cross-referencing the reachability dict.
        "backlog_dispatchable": int(backlog_reachability.get("dispatchable", 0) or 0),
        "backlog_blocked_by_open_dependency": int(
            backlog_reachability.get("blocked_by_open_dependency", 0) or 0
        ),
        "reason": None,
    }

    threshold_minutes = config.dispatch_staleness_minutes
    if threshold_minutes <= 0:
        result["threshold_seconds"] = 0
        result["reason"] = "threshold_disabled"
        return result

    result["threshold_seconds"] = threshold_minutes * 60

    if now is None:
        now = datetime.now(UTC)

    if recent_issue_numbers:
        # Format the already-sampled `now` rather than taking a second,
        # uninjected clock read here -- the single-frozen-clock-per-pass
        # invariant established by #828/#838. `utc_now()` reads the real
        # wall clock, which would let this short-circuit's timestamp drift
        # from the `now` the caller sampled once for the whole pass. Uses
        # `utc_now()`'s own formula (seconds precision, trailing "Z") so the
        # string matches every other event timestamp in events.db, including
        # the `latest["ts"]` value this same field holds in the non-short-
        # circuit branch below.
        result["last_dispatch_at"] = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        result["last_dispatch_issue_numbers"] = sorted(recent_issue_numbers)
        result["age_seconds"] = 0
        result["reason"] = "current_pass_dispatched"
        return result

    if not _backlog_is_non_empty(backlog_reachability):
        if not backlog_reachability.get("observed"):
            result["reason"] = "backlog_not_observed"
        else:
            result["reason"] = "empty_backlog"
        return result

    # Issue #1110: when every ready issue is blocked by an open dependency,
    # dispatch is permanently (and correctly) idle -- there is nothing to
    # dispatch and nothing wrong with the dispatcher. Firing a cadence alarm
    # here is a false positive that pattern-matches the #944 four-day stall
    # this detector exists to catch. The ``dispatchable`` count from
    # classify_backlog_reachability already excludes dependency-blocked issues
    # (they bin as ``blocked_by_open_dependency``), so ``dispatchable == 0``
    # with ``blocked_by_open_dependency > 0`` means "ready but blocked", not
    # "nothing ready". The #944 case (``dispatchable == 0`` and
    # ``blocked_by_open_dependency == 0``) falls through to the age check below
    # and still alarms.
    if result["backlog_dispatchable"] == 0 and result["backlog_blocked_by_open_dependency"] > 0:
        result["reason"] = "all_ready_blocked_by_dependencies"
        return result

    latest = _latest_non_empty_dispatch(state)
    if latest is None:
        result["reason"] = "no_baseline"
        return result

    last_ts = _parse_iso_ts(latest["ts"])
    if last_ts is None:
        result["reason"] = "no_baseline"
        return result

    age_seconds = int((now - last_ts).total_seconds())
    result["last_dispatch_at"] = latest["ts"]
    result["last_dispatch_issue_numbers"] = list(latest["issue_numbers"])
    result["age_seconds"] = age_seconds

    if age_seconds > result["threshold_seconds"]:
        result["stale"] = True
        result["reason"] = "dispatch_stale"
        # Issue #1769 section 6 policy: edge-triggered + bounded low-rate
        # reminder, not an unconditional re-fire every pass the condition
        # holds. Reuses the staleness threshold itself as the reminder
        # interval (see docstring) rather than a second config knob.
        result["should_emit"] = is_dispatch_stale_alert_due(
            state, now=now, reminder_minutes=threshold_minutes
        )
    else:
        result["reason"] = "within_threshold"

    return result


def _annotation_to_required_change(check_name: str, annotation: dict[str, Any]) -> str | None:
    """Format a single GitHub check-run annotation as a ``required_changes`` entry.

    Returns ``None`` -- never a fabricated placeholder -- when the annotation
    carries no message or is not failure-level. A bare location with no
    explanation is not actionable, and ``warning``/``notice`` annotations are
    not required changes: they are emitted on green runs too (e.g. the
    ``actions/checkout@v4`` Node.js 20 deprecation advisory is present on
    every run of this workflow), so surfacing them as rework items sends the
    worker after unrelated noise (issue #993). Only
    ``annotation_level == "failure"`` renders. ``path``/``start_line`` are
    appended when present, but their absence does not sink the entry: the
    message alone is still real, GitHub-sourced reviewer content, so it
    renders as ``"<check>: <message>"`` rather than being dropped.
    """
    if not isinstance(annotation, dict):
        return None
    if str(annotation.get("annotation_level") or "").strip() != "failure":
        return None
    message = str(annotation.get("message") or "").strip()
    if not message:
        return None
    path = str(annotation.get("path") or "").strip()
    start_line = annotation.get("start_line")
    if path and isinstance(start_line, int):
        location = f"{path}:{start_line}"
    elif path:
        location = path
    else:
        location = None
    return f"{check_name}: {location} — {message}" if location else f"{check_name}: {message}"


def _required_changes_from_checks(
    checks: list[dict[str, Any]] | None,
    failed_required_checks: tuple[str, ...],
    fetch_annotations: Callable[[int], list[dict[str, Any]]],
) -> list[str]:
    """Build ``required_changes`` entries from the annotations on each
    genuinely-failed required check (issue #771: the CI-failure rework route
    previously recorded a verdict naming only the check, never the failure).

    ``fetch_annotations`` is injected (rather than this function calling
    ``GitHub`` directly) so it stays pure and unit-testable; the
    ``OrchestratorApp`` caller passes ``self.gh.check_run_annotations``, which
    already returns ``[]`` on any GitHub API failure -- never raises -- so
    this function inherits that fail-safe without adding its own try/except.

    Uses ``checks.py``'s own ``_is_failing_run`` classifier (the same one
    ``summarize_checks``/``compute_check_debounce`` use to decide
    ``CheckSummary.failed``) to pick which run(s) of a check name to fetch
    annotations for, rather than re-deriving a second "is this failing"
    predicate from ``state`` alone -- a name can appear in
    ``failed_required_checks`` purely on ``bucket == "fail"`` with an empty
    ``state`` (external status checks), which an enumerated-``state`` filter
    would silently skip, discarding annotations for the very run that caused
    the verdict. When a name has multiple runs (e.g. matrix legs) under
    "worst-of" semantics, only the actually-failing run's annotations are
    fetched -- a passing sibling run's annotations (if any) are not failure
    findings.

    Returns an empty list -- never a fabricated file/line -- only when
    ``checks`` is unavailable or no name in ``failed_required_checks`` is
    actually failing per ``_is_failing_run``. For each failing check, the
    failure-level annotations (warnings/notices are filtered out by
    ``_annotation_to_required_change``, issue #993) render as entries, and
    the check's own ``link`` -- real, GitHub-sourced data already present on
    every entry in ``checks`` (``PR_CHECKS_FIELDS`` always requests it) --
    is **always** appended alongside them. A process-level crash emits a
    contentless ``"Process completed with exit code 1."`` failure annotation
    that names no cause; the real cause (e.g. a TLS handshake timeout) lives
    only in the step log the link reaches. Appending the link unconditionally
    -- rather than only when *no* annotations rendered -- removes the need to
    predict which annotations are informative: a worker that can reach the
    run log can find a transient-cause failure that no annotation names, and
    one that cannot, cannot. When no failure-level annotations rendered, the
    link line carries the "no per-line annotations available" wording so the
    worker knows to look at the run log rather than search for a missing
    file/line. ``record_review``'s caller passes this straight through as
    ``required_changes``; the ``_render_required_changes_section`` tier-2
    "CI failed on X" summary fallback only fires when this list comes back
    fully empty, which now only happens when GitHub gave us neither
    failure-level annotations nor a link.
    """
    if not checks or not failed_required_checks:
        return []
    failed_names = set(failed_required_checks)
    required_changes: list[str] = []
    for check in checks:
        name = str(check.get("name") or "")
        if name not in failed_names:
            continue
        if not _is_failing_run(check):
            continue
        check_run_id = check.get("databaseId")
        entries = (
            [
                entry
                for annotation in fetch_annotations(check_run_id)
                if (entry := _annotation_to_required_change(name, annotation)) is not None
            ]
            if isinstance(check_run_id, int)
            else []
        )
        required_changes.extend(entries)
        link = str(check.get("link") or "").strip()
        if not link:
            continue
        if entries:
            required_changes.append(f"{name}: failing run — {link}")
        else:
            required_changes.append(
                f"{name}: no per-line annotations available from GitHub; "
                f"inspect the failing run at {link}"
            )
    return required_changes


def _collect_gate_exemption_section(
    checks: list[dict[str, Any]] | None,
    pr: dict[str, Any],
    exemption_label: str,
    fetch_job_log: Callable[[int], str | None],
) -> str:
    """Render the collect-gate exemption evidence for the reviewed head (#1686).

    The collect-only gate command emits a ``COLLECT-GATE-EXEMPTION v1 {...}``
    line into its Actions job log whenever ``--pr`` was passed, whether or not
    the exemption was granted. This section reads that marker back out of the
    gate job's log -- located through the PR's head-pinned check list
    (``gh pr checks`` rows describe ``pr["headRefOid"]``) -- and renders what
    the operator-applied label did on THIS head.

    ``fetch_job_log`` is injected (rather than this function calling
    ``GitHub`` directly) so the section logic stays pure and unit-testable --
    the same seam ``_required_changes_from_checks`` uses for
    ``fetch_annotations``. The caller passes a wrapper over
    ``gh.run(["api", "repos/{owner}/{repo}/actions/jobs/{id}/logs"])`` that
    returns the log text or ``None``.

    Trust model, mirroring the gate itself:

    * The label is only ever *read*: the packet states whether it is applied
      and what it waived. No label mutation happens here (and none happens
      anywhere -- the issue forbids stripping it on ``synchronize``).
    * Evidence must be head-pinned twice before it counts: the check row
      comes from the head-scoped check list AND the marker's ``head_sha``
      must equal ``pr["headRefOid"]``. A label applied for head A
      legitimately survives on head B, so evidence describing another head
      is rendered as stale, never as a waiver.
    * Missing evidence is never rendered as a waiver. When the label is
      applied but the job log is unreachable or carries no marker, the
      section says so explicitly and tells the reviewer not to treat the
      label as proof of a waiver.

    Renders ``""`` (no section) when nothing exemption-related happened:
    label absent and no active waiver evidence.
    """
    label_present = exemption_label in label_names(pr)
    head_sha = str(pr.get("headRefOid") or "")

    gate_check = next(
        (
            check
            for check in (checks or [])
            if str(check.get("name") or "") == COLLECT_ONLY_GATE_CHECK_NAME
        ),
        None,
    )

    payload: dict[str, Any] | None = None
    # "verified": marker parsed AND head_sha matches the reviewed head.
    # "stale": marker parsed but its head binding is missing or mismatched.
    # "none": log fetched, no marker (gate ran without --pr, or pre-#1686).
    # "unavailable": no log to read (job id unparseable or fetch failed).
    evidence = "none"
    if gate_check is not None:
        job_id = gate_check.get("databaseId")
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            job_id = _job_id_from_link(gate_check.get("link"))
        if job_id is None:
            evidence = "unavailable"
        else:
            log_text = fetch_job_log(job_id)
            if log_text is None:
                evidence = "unavailable"
            else:
                payload = parse_exemption_log_marker(log_text)
                if payload is not None:
                    recorded = payload.get("head_sha")
                    evidence = (
                        "verified"
                        if recorded and head_sha and str(recorded) == head_sha
                        else "stale"
                    )

    waived = payload.get("waived") if payload else None
    if not isinstance(waived, list):
        waived = []

    def _finding_line(finding: Any) -> str:
        if not isinstance(finding, dict):
            return f"- {finding}"
        line = f"- `{finding.get('kind')}`: `{finding.get('leaf_name')}`"
        source = finding.get("source_module")
        if source:
            line += f" (from `{source}`)"
        return line

    heading = "**Collect-gate exemption (issue #1686):** "

    if evidence == "verified" and payload is not None and payload.get("active"):
        if waived:
            label_state = (
                f"label `{exemption_label}` is applied to this PR"
                if label_present
                else f"label `{exemption_label}` is no longer applied (it was removed after this gate run)"
            )
            lines = [
                heading
                + f"{label_state}. The collect-only gate **waived "
                + f"{len(waived)} enforced finding(s)** on the reviewed head "
                + f"`{head_sha}`:"
            ]
            lines.extend(_finding_line(f) for f in waived)
            lines.append(
                "The gate ran the full comparison -- these findings were "
                "reported and then waived by the operator-applied label, not "
                "suppressed. An exemption does not make a deletion correct: "
                "weigh each waived finding against the diff yourself."
            )
            return "\n".join(lines) + "\n"
        label_state = (
            f"label `{exemption_label}` is applied to this PR"
            if label_present
            else f"label `{exemption_label}` is no longer applied (it was removed after this gate run)"
        )
        stale_note = "The label is stale and can be removed." if label_present else ""
        return (
            heading
            + f"{label_state}, and the gate reported no enforced findings on "
            + f"the reviewed head -- **nothing was waived**.{(' ' + stale_note) if stale_note else ''}\n"
        )

    if evidence == "verified" and payload is not None and not payload.get("active"):
        if label_present:
            return (
                heading
                + f"label `{exemption_label}` is applied to this PR NOW, but "
                + "the gate run on the reviewed head resolved it as absent "
                + f"({payload.get('detail')}) -- **no findings were waived**. "
                + "If the waiver was intended, an operator must rerun the "
                + "gate job so it re-reads the live labels.\n"
            )
        return ""  # label absent, run resolved absent: nothing happened

    if evidence == "stale":
        recorded = payload.get("head_sha") if payload else None
        label_state = (
            f"label `{exemption_label}` is applied to this PR, and " if label_present else ""
        )
        return (
            heading
            + label_state
            + "exemption evidence exists in the gate job's log but could not "
            + f"be verified against the reviewed head `{head_sha}` "
            + f"(recorded for `{recorded or 'an unknown head'}`) -- treating "
            + "it as stale evidence, NOT as a waiver for this head.\n"
        )

    if label_present:
        reason = (
            "the gate job's log could not be fetched"
            if evidence == "unavailable"
            else "the gate job's log carries no exemption record (the run may "
            "predate --pr support, or the job could not be located)"
        )
        return (
            heading
            + f"label `{exemption_label}` is applied to this PR, but no "
            + f"exemption evidence was found for the reviewed head -- {reason}. "
            + "Do NOT treat the label as proof of a waiver: verify the gate "
            + "job's output directly.\n"
        )

    return ""
