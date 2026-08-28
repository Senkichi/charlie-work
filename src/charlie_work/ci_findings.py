"""CI-checks-findings free-function family (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: two functions that render the
review packet's CI-status section from already-fetched check data
(``_ci_status_section``, ``_non_required_check_findings``); four functions
that detect a stalled dispatch cadence from events.db
(``_backlog_is_non_empty``, ``_latest_non_empty_dispatch``,
``_parse_iso_ts``, ``check_dispatch_staleness``); and two functions that
turn a failing required check's GitHub annotations into
``required_changes`` entries (``_annotation_to_required_change``,
``_required_changes_from_checks``).

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
from .config import DispatchConfig
from .instrumentation import query_events


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


def _latest_non_empty_dispatch(state_path: Path) -> dict[str, Any] | None:
    """Return the most recent ``dispatch`` event whose ``issue_numbers`` is non-empty.

    Empty-payload dispatch events happen on every healthy zero-dispatch pass,
    so they cannot be used to measure cadence. We scan newest-first.

    Bounded to the most recent 100 ``dispatch`` rows: ``query_events``'s
    ``limit=`` selects with ``ORDER BY id DESC LIMIT ?`` and then re-orders
    the result ascending, so this returns the newest 100 rows (oldest of
    that 100 first) -- exactly what "scan newest-first" below needs, not
    the oldest 100. Without a bound this ran an unindexed-by-limit full
    scan of every ``dispatch`` row on every dispatch pass; this repo's own
    events.db already holds thousands of them and the table grows without
    bound. 100 is generous headroom: even at a 5-minute dispatch cadence,
    the default 240-minute ``dispatch_staleness_minutes`` threshold only
    needs to look back ~48 dispatch events to find the most recent
    non-empty one.
    """
    events = query_events(state_path, kind="dispatch", limit=100)
    for event in reversed(events):
        issue_numbers = event.get("payload", {}).get("issue_numbers")
        if isinstance(issue_numbers, list) and issue_numbers:
            return event
    return None


def _parse_iso_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def check_dispatch_staleness(
    state_path: Path,
    config: DispatchConfig,
    backlog_reachability: dict[str, Any],
    *,
    recent_issue_numbers: list[int] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Issue #946: detect when a non-empty backlog has had no dispatch for too long.

    Reads events.db for the most recent ``dispatch`` event whose payload
    ``issue_numbers`` is non-empty. When that event is older than
    ``config.dispatch_staleness_minutes`` and the unfiltered backlog is observed
    to be non-empty, returns a stale diagnostic. Otherwise returns a no-op
    diagnostic with ``stale: False``.

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
    """
    result: dict[str, Any] = {
        "stale": False,
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

    latest = _latest_non_empty_dispatch(state_path)
    if latest is None:
        result["reason"] = "no_baseline"
        return result

    last_ts = _parse_iso_ts(latest["ts"])
    if last_ts is None:
        result["reason"] = "no_baseline"
        return result

    age_seconds = int((now - last_ts).total_seconds())
    result["last_dispatch_at"] = latest["ts"]
    result["last_dispatch_issue_numbers"] = list(latest["payload"].get("issue_numbers", []))
    result["age_seconds"] = age_seconds

    if age_seconds > result["threshold_seconds"]:
        result["stale"] = True
        result["reason"] = "dispatch_stale"
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
